import random
import pandas as pd
from scipy.spatial.distance import cosine
from helpers.model_selector import ModelSelector
from helpers.rag_handler import RAGHandler
from helpers import custom_functions as cfoos


class Evaluator:
    """Class used in RAG evaluations."""
    def __init__(self,
                 model: ModelSelector=None,
                 rag_handler: RAGHandler=None
                 ) -> None:
        # backend config
        self.model = model
        self.emb_model = model.get_emb_model()
        self.rag_handler = rag_handler
        # tuning config
        # eval_df must be generated on the standard params as baseline before running Tuner
        self.eval_df = self.generate_eval_prompts(read_local_df=True)
        self.nsamples = None
        self.correct_documents_retrieved = None
        self.correct_chunks_retrieved = None
        self.correct_top1_retrieved = None
        self.average_cosine_distance = None
        self.average_true_cosine_distance = None
        # print(f"---{datetime.now()} | The Evaluator has been awakened from its slumber")
        cfoos.LOG.info("The Evaluator has been awakened from its slumber")

    # TODO: rework for small data samples
    def get_eval_df(self, nsamples: int=100) -> pd.DataFrame:
        """
        Create a dataframe for RAG evaluation.
        Based on the format provided by RAGHandler.populate_vectdb.
        """
        nrands = []
        data = {
            "document_path": [],
            "document_folder": [],
            "document_name": [],
            "document_chunk": [],
            "document_embeddings": []
        }
        # handle smaller than nsamples collections
        nsamples = min(nsamples, self.rag_handler.collection.count())
        self.nsamples = nsamples
        # print(f"---{datetime.now()} | Creating evaluation dataframe...")
        cfoos.LOG.info("Creating evaluation dataframe...")
        # retreive vectorized data
        db_data = self.rag_handler.collection.get(include=["embeddings", "metadatas"])
        # retrieve data components
        document_vectors = db_data["embeddings"]
        entries = len(document_vectors)
        document_paths = [metadata["document_path"] for metadata in  db_data["metadatas"]]
        document_folders = [metadata["document_folder"] for metadata in  db_data["metadatas"]]
        document_names = [metadata["document_name"] for metadata in  db_data["metadatas"]]
        document_chunks = [metadata["content"] for metadata in  db_data["metadatas"]]
        # sample randomly from vector data
        for i in range(nsamples):
            irand = random.randint(0, entries - 1)
            while irand not in nrands:
                irand = random.randint(0, entries - 1)
                nrands.append(irand)
            data["document_path"].append(document_paths[irand])
            data["document_folder"].append(document_folders[irand])
            data["document_name"].append(document_names[irand])
            data["document_chunk"].append(document_chunks[irand])
            data["document_embeddings"].append(document_vectors[irand])
        # get df
        df = pd.DataFrame().from_dict(data)
        return df

    def generate_eval_prompts(self, df: pd.DataFrame=None, read_local_df: bool=False, save_df: bool=False) -> pd.DataFrame:
        """
        Generate prompts using the app's model based on randomly selected chunks.
        df has a shape returned by self.get_eval_df()
        """
        # if evaluation df already exists read it from local
        df_name = "eval_df.parquet"
        if read_local_df:
            try:
                eval_df = pd.read_parquet(df_name)
                return eval_df
            except:
                # print(f"---{datetime.now()} | Eval DF not found.")
                cfoos.LOG.info("Eval DF not found.")
                return None
        # if evaluation df doesn't exist, create it
        gen_prompts = []
        # print(f"---{datetime.now()} | Generating Prompts...")
        cfoos.LOG.info("Generating Prompts...")
        # generate prompts
        for doc_name, chunk in zip(df.document_name.values, df.document_chunk.values):
            # crop chunk for cpu models
            sysp = f"""
            Generate a question for "chunk_content" bellow.
            Use "document_name" to improve the question. 
            Try to formulate the question in the same language as the chunk.
            Question length should be less than 100 characters.
            You ONLY return the user prompt.
            """
            usrp = f"""document_name: {doc_name}
            chunk_content: {chunk}
            """
            gen_prompts.append(self.model.prompt_llm(usrp, sysprompt=sysp, singleton=True, extend_messages=False))
        # add generated prompts to eval df
        df["generated_prompts"] = gen_prompts
        self.eval_df = df.copy()
        if save_df:
            df.to_parquet(df_name, index=False)
        return df

    def prompt_db(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """Retrieve vectors from db based on prompts generated by generate_eval_prompts()."""
        outputs = {
            "output_document_name": [],
            "document_id": [],
            "output_rank": [],
            "output_distances": [],
            "output_embeddings": [],
            "output_text": [],
            "embedded_prompt": [],
            "generated_prompts": []
        }
        doc_id = 0
        # each generated prompt will retrieve top 5 vectors based on cosine distance
        # print(f"---{datetime.now()} | Querying Database...")
        cfoos.LOG.info("Querying Database...")
        for prompt in eval_df.generated_prompts.values:
            doc_id +=1
            retrieval = self.rag_handler.retrieve_vdata(prompt, include=["embeddings", "distances", "metadatas"])
            entries = len(retrieval["ids"][0])
            outputs["output_document_name"].extend([chunk["document_name"] for metadata in retrieval["metadatas"] for chunk in metadata])
            outputs["document_id"].extend([doc_id] * entries)
            outputs["output_rank"].extend([i + 1 for i in range(entries)])
            outputs["output_distances"].extend(sorted([distance for items in retrieval["distances"] for distance in items]))
            outputs["output_embeddings"].extend([vector for items in retrieval["embeddings"] for vector in items])
            outputs["output_text"].extend([chunk["content"] for metadata in retrieval["metadatas"] for chunk in metadata])
            # generated_prompts will be used for merging
            outputs["embedded_prompt"].extend([self.emb_model.encode(prompt) for i in range(entries)])
            outputs["generated_prompts"].extend([prompt for i in range(entries)])
        # extend analytical dataframe for further comparisons
        outputs_df = pd.DataFrame().from_dict(outputs)
        result_df = pd.merge(eval_df, outputs_df, on="generated_prompts", how="inner")
        return result_df

    def analyze_retrievals(self, anal_df: pd.DataFrame) -> None:
        """Analyze RAG retrieval based on the df format returned by prompt_db()"""
        df = anal_df.copy()
        # compute metrics
        true_cosine_distance = [cosine(doc_embs, out_embs) for doc_embs, out_embs in zip(df.document_embeddings, df.output_embeddings)]
        df["true_cosine_distances"] = true_cosine_distance
        df["true_cos_dist_rank"] = df.groupby("document_id")["true_cosine_distances"].rank(method="first", ascending=True).astype(int)
        df["correct_doc_retrieved"] = (df.document_name == df.output_document_name).astype(int)
        df["bullseye"] = (df.true_cosine_distances == 0).astype(int)
        df["correct_rank1"] = (df.bullseye == df.output_rank).astype(int)
        self.correct_documents_retrieved = round(df.correct_doc_retrieved.mean() * 100, 4)
        self.correct_chunks_retrieved = round((df.bullseye.sum() / df.correct_doc_retrieved.sum()) * 100, 4)
        self.correct_top1_retrieved = round(df.correct_rank1.mean() * 100, 4)
        self.average_cosine_distance = round(df.output_distances.mean(), 4)
        self.average_true_cosine_distance = round(df.true_cosine_distances.mean(), 4)
        # inform
        # print(f"---{datetime.now()} | Eval results (nsamples={self.nsamples}, chunk_size={self.rag_handler.chunk_size}, chunk_overlap={self.rag_handler.chunk_overlap})")
        # print(f"---{datetime.now()} | Correct documents retrieved: {self.correct_documents_retrieved}%")
        # print(f"---{datetime.now()} | Correct chunks retrieved: {self.correct_chunks_retrieved}%")
        # print(f"---{datetime.now()} | Correct top1 retrievals: {self.correct_top1_retrieved}%")
        # print(f"---{datetime.now()} | Average cosine distance: {self.average_cosine_distance}")
        # print(f"---{datetime.now()} | Average true cosine distance: {self.average_true_cosine_distance}")
        cfoos.LOG.info(f"Eval results (nsamples={self.nsamples}, chunk_size={self.rag_handler.chunk_size}, chunk_overlap={self.rag_handler.chunk_overlap})")
        cfoos.LOG.info(f"Correct documents retrieved: {self.correct_documents_retrieved}%")
        cfoos.LOG.info(f"Correct chunks retrieved: {self.correct_chunks_retrieved}%")
        cfoos.LOG.info(f"Correct top1 retrievals: {self.correct_top1_retrieved}%")
        cfoos.LOG.info(f"Average cosine distance: {self.average_cosine_distance}")
        cfoos.LOG.info(f"Average true cosine distance: {self.average_true_cosine_distance}")


# TODO: extend statistical analysis
# import numpy as np
# from sklearn.manifold import TSNE
# import plotly.graph_objects as go
# import plotly.express as px
# from sklearn.manifold import trustworthiness
# from sklearn.neighbors import NearestNeighbors
# from scipy.spatial.distance import pdist, squareform
# from sklearn.metrics import silhouette_score
# from sklearn.metrics.pairwise import cosine_similarity


# # prepare visual vars
# tsne = TSNE(n_components=2, random_state=91)
# reduced_vectors = tsne.fit_transform(vectors)
# color_scale = px.colors.sample_colorscale("Turbo", [i / max(1, nfolders -1) for i in range(nfolders)])
# color_map = {doc: color_scale[i] for i, doc in enumerate(set(doc_folders))}
# colors = [color_map[folder] for folder in doc_folders]
# # get visual representation of vectors
# fig = go.Figure(data=[go.Scatter(
#     x=reduced_vectors[:, 0],
#     y=reduced_vectors[:, 1],
#     mode='markers',
#     marker=dict(size=5, color=colors, opacity=0.8),
#     # get the first 100 characters based on the chunk
#     text=[f"Folder: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_folders, documents)],
#     hoverinfo='text'
# )])

# fig.update_layout(
#     title='2D Chroma Vector Store Visualization',
#     scene=dict(xaxis_title='x',yaxis_title='y'),
#     width=800,
#     height=600,
#     margin=dict(r=20, b=10, l=10, t=40)
# )

# fig.show()

# # examine Kullback-Leibler (KL) Divergence (0-1, lower is better)
# # minimizes divergence between high-dimensional and low-dimensional distributions 
# print(f"Final KL Divergence: {tsne.kl_divergence_}")

# # examine trustworthniess score (0-1, higher is better)
# # how well neighborhood relationships are preserved in 2D
# score = trustworthiness(vectors, reduced_vectors, n_neighbors=10)
# print(f"Trustworthiness Score: {score:.4f}")

# # inspect how many high-dim neighbors are still neighbors in 2D
# # +0.9 > great
# # 0.7 to 0.9 > moderate
# # < 0.5 > poor
# def knn_preservation(original, reduced, k=10):
#     knn_hd = NearestNeighbors(n_neighbors=k).fit(original)
#     knn_ld = NearestNeighbors(n_neighbors=k).fit(reduced)

#     neighbors_hd = knn_hd.kneighbors(original, return_distance=False)
#     neighbors_ld = knn_ld.kneighbors(reduced, return_distance=False)

#     preservation = np.mean([len(set(neighbors_hd[i]) & set(neighbors_ld[i])) / k for i in range(len(original))])
#     return preservation

# knn_score = knn_preservation(vectors, reduced_vectors, k=10)
# print(f"k-NN Preservation Score: {knn_score:.4f}")

# # inspect how much relative distances between points changed between high-dim VS 2D
# # 0.1 to 0.3 = acceptable
# # > 0.5 = severe
# def mean_distance_error(original, reduced):
#     dist_hd = pdist(original)  # Pairwise distances in high-D space
#     dist_ld = pdist(reduced)  # Pairwise distances in 2D

#     # handle duplicate or very close vectors in dataset 
#     valid_indices = dist_hd > 0  # Mask non-zero distances
#     dist_hd = dist_hd[valid_indices]
#     dist_ld = dist_ld[valid_indices]

#     return np.mean(np.abs(dist_hd - dist_ld) / dist_hd) if len(dist_hd) > 0 else np.nan

# mde_score = mean_distance_error(vectors, reduced_vectors)
# print(f"Mean Distance Error: {mde_score:.4f}")

# # how well defined clusters are in 2D?
# # > 0.5 = well separated clusters
# # 0.2 to 0.5 > some overlapping
# # < 0.2 poor separation
# silhouette = silhouette_score(reduced_vectors, doc_types)  # Assuming doc_types are categorical labels
# print(f"Silhouette Score: {silhouette:.4f}")

# # get cosine-similarity
# cs = cosine_similarity(vectors)
# threshold = 0.99  # Adjust if needed
# duplicates = []

# for i in range(len(vectors)):
#     for j in range(i + 1, len(vectors)):  # Avoid self-comparison
#         if x[i, j] > threshold:
#             duplicates.append((documents[i], documents[j], x[i, j]))

# # Display results
# for d1, d2, score in duplicates:
#     print(f"Duplicate Found: \nDoc 1: {d1}\nDoc 2: {d2}\nSimilarity: {score:.4f}\n")
