import socket
import pickle
import shutil
from pathlib import Path
from datetime import datetime
import gc
import time
import fitz
import chromadb
from chromadb.config import Settings
from helpers.model_selector import ModelSelector
from helpers import custom_functions as cfoos


# TODO: this bad boy needs a lifting, make time for it.
class RAGHandler():
    """Handles RAG related matters (text extraction, vectorization, retrieval...)"""
    def __init__(
            self,
            db_path: str,
            supported_extensions: list,
            collection_config: dict,
            collection_name: str,
            model: ModelSelector,
            data_path: str=None
            ) -> None:
        try:
            # backend config
            self.db_path = db_path
            self.supported_extensions = supported_extensions
            self.data_path = self.check_data_folder(data_path, self.supported_extensions)
            self.data = None
            cfoos.check_create_dir(self.db_path)
            if data_path != "None":
                self.client = chromadb.PersistentClient(
                    path=str(self.db_path),
                    settings=Settings(allow_reset=True)
                    )
                self.collection_config = collection_config
                self.collection_name = collection_name
                self.collection = self.client.get_or_create_collection(self.collection_name)
            self.device = model.device
            self.emb_model = model.get_emb_model()
            # tuning config
            self.default_chunk_sizes = {
            "cuda": 1000,
            "cpu": 450
            }
            self.chunk_size_factor = None
            self.chunk_overlap_factor = None
            self.chunk_size = None
            self.chunk_overlap = None
            # print(f"---{datetime.now()} | Initialized database with source: {self.data_path}")
            cfoos.LOG.info(f"Initialized database with source: {self.data_path}")
        except Exception:
            # print(f"---{datetime.now()} | Failed to initialize database. Verify config file")
            cfoos.LOG.info(f"Failed to initialize database. Verify config")

    @staticmethod
    def check_data_folder(path, supported_extensions: list) -> Path:
        """Verify data path is valid and contains at least a usable file."""
        data_path = Path(path)
        if data_path.exists() and data_path.is_dir():
            for file in data_path.rglob("*"):
                if file.suffix.lstrip(".") in supported_extensions:
                    return data_path
        return "None"

    def get_doc_repr(
            self,
            file_path: Path,
            content: str,
            content_summary: str="not available") -> dict:
        """Return a filled data (dict) template for chunking/vectorization."""
        doc_repr = {
            "document_name": file_path.stem,
            "content": content,
            "document_path": str(file_path),
            "document_folder": file_path.parent.name,
            "document_type": file_path.suffix,
            "document_length": len(content),
            "content_summary": content_summary,
            "processed_on": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "machine": socket.gethostname()
            }
        return doc_repr

   # TODO: eww... temporary until implementing a more elegant solution
    def reset_db_folder(self) -> None:
        """Reset db folder"""
        if self.db_path.exists():
            self.client.delete_collection(self.collection_name)
            self.client.reset()
            self.client.clear_system_cache()
            del self.client
            self.client = None
            # windows fails to delete opened files because of background process (x: antivirus) 
            # even when we closed the connection. Thus the inclusion of ignore_errors
            # this issues only happens on windows because of the zombie handles
            # since we deleted the collection and reset db, we should be fine
            gc.collect()
            time.sleep(1)
            shutil.rmtree(self.db_path, ignore_errors=True)
        # print(f"---{datetime.now()} | Reset DB")
        cfoos.LOG.info(f"Reset DB")
        cfoos.check_create_dir(self.db_path)
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(allow_reset=True)
            )

    def extract_data(self, save_metadata: bool=False) -> list[dict]:
        """Extract text data from user provided folder path."""
        # # TODO: see summarize_text
        # app_model: str,
        # summarize: bool=False,
        # app_model must come from ModelSelector. If app.device==cuda, then
        # 'summarize' the extracted text using the app's model in meta-data summarization.
        # content_summary = "not available"
        data = []
        # beware, not accounting for double documents
        # print(f"---{datetime.now()} | Extracting data...")
        cfoos.LOG.info("Extracting data...")
        for file in self.data_path.rglob("*"):
            if file.is_file():
                for extension in self.supported_extensions:
                    if extension in file.name:
                        file_path = file.absolute()
                        # TODO: this try/except block is temporary and aimed at .md
                        try:
                            # open file and extract text
                            doc = fitz.open(file_path)
                            content = "".join([page.get_text() for page in doc])
                        except:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                        # store metadata and content
                        data.append(self.get_doc_repr(file_path, content))
        if save_metadata:
            extracted_data_path = Path.cwd() / "extracted_text_data.pkl"
            # print(f"---{datetime.now()} | Saving extracted data to: {extracted_data_path}")
            cfoos.LOG.info(f"Saving extracted data to: {extracted_data_path}")
            with open(extracted_data_path, "wb") as f:
                pickle.dump(data, f)
        # store parsed path for reference
        with open("init_config.txt", "w") as f:
            f.write(str(self.data_path))
        self.data = data
        return data

    def get_chunks(
        self,
        extracted_data: list[dict],
        chunk_size_factor: float=1.0,
        chunk_overlap_factor: float=0.2) -> list[dict]:
        """
        Process data returned by extract_text into chunks.
        Get longer chunks for GPUs.
        params:
            device: "cpu" vs "cuda"
            chunk_size_factor: default value multiplier, max=2 (performance ceiling)
            chunk_overlap: chunk overlap as a percent of the chunk size (ex: 0.2 > 1000 * 0.2 = 200)
        """
        chunked_data = []
        chunk_id = 0
        # adjust factors
        if chunk_size_factor > 2.0:
            chunk_size_factor = 2.0
            # print(f"---{datetime.now()} | Adjusted chunk size factor to 2")
            cfoos.LOG.info("Adjusted chunk size factor to 2")
        if chunk_overlap_factor > 2.0:
            chunk_overlap_factor = 2.0
            # print(f"---{datetime.now()} | Adjusted chunk overlap factor to 2 (CPU restriction).")
            cfoos.LOG.info("Adjusted chunk overlap factor to 2 (CPU restriction).")
        self.chunk_size_factor = chunk_size_factor
        self.chunk_overlap_factor = chunk_overlap_factor
        self.chunk_size = int(round(self.default_chunk_sizes[self.device] * self.chunk_size_factor, 0))
        self.chunk_overlap = int(round(self.chunk_size * self.chunk_overlap_factor, 0))
        # generate chunks
        # print(f"---{datetime.now()} | Generating chunks...")
        cfoos.LOG.info(f"Generating chunks...")
        for doc in extracted_data:
        # get a copy to avoid overwrite
            doc_copy = doc.copy()
            content = doc_copy["content"]
            document_length = doc_copy["document_length"]
            document_path = doc_copy["document_path"]
            # get chunks only if document length > chunk_size
            if document_length < self.chunk_size:
                # include chunk_id
                entry = {"chunk_id": str(chunk_id)}
                # augment chunks
                doc_copy["content"] = f"folder: {doc_copy['document_folder']}, file: {doc_copy['document_name']}, text: {doc_copy['content']}"
                entry.update(doc_copy)
                chunked_data.append(entry)
                chunk_id += 1
                continue
            # idem, augment chunks
            chunks = [f"folder: {doc_copy['document_folder']}, file: {doc_copy['document_name']}, text: " + content[i : i + self.chunk_size]
                      for i in range(0, len(content), self.chunk_size - self.chunk_overlap)]
            # store chunks
            for chunk in chunks:
                entry = {"chunk_id": str(chunk_id)}
                entry.update(self.get_doc_repr(Path(document_path), chunk))
                chunked_data.append(entry)
                chunk_id += 1
        return chunked_data

    def populate_vectdb(self, chunks: list[dict], collection_config: dict=None) -> None:
        """Populate db with data processsed by get_chunks."""
        collection_config = collection_config if collection_config else self.collection_config
        # TODO: locus extra-feature
        self.reset_db_folder()
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            # use cosine distance
            metadata=collection_config
            )
        # print(f"---{datetime.now()} | Created collection: {self.collection_name} ({', '.join(f'{k}={v}' for k, v in collection_config.items())})")
        # print(f"---{datetime.now()} | Populating database...")
        cfoos.LOG.info(f"Created collection: {self.collection_name} ({', '.join(f'{k}={v}' for k, v in collection_config.items())})")
        cfoos.LOG.info("Populating database...")
        collection.add(
            ids=[chunk["chunk_id"] for chunk in chunks],
            embeddings=[self.emb_model.encode(chunk["content"]).tolist() for chunk in chunks],
            metadatas=[chunk for chunk in chunks]
        )
        self.collection = self.client.get_collection(self.collection_name)
        # print(f"---{datetime.now()} | Database ready for retrieval")
        cfoos.LOG.info("Database ready for retrieval")

    def retrieve_vdata(self,
                       prompt: str,
                       nresults: int=5,
                       include: list=["metadatas"]) -> None:
        """Retrieve vectorized data from chromadb."""
        emb_query = self.emb_model.encode(prompt).tolist()

        results = self.collection.query(
            query_embeddings=[emb_query],
            n_results=nresults,
            include=include
        )

        return results
    