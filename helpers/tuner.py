import logging
import pandas as pd
from helpers.model_selector import ModelSelector
from helpers.rag_handler import RAGHandler
from helpers.evaluator import Evaluator
from helpers import custom_functions as cfoos


class Tuner:
    """Used in tuning RAG retrieval results"""
    def __init__(
            self,
            model: ModelSelector,
            rag_handler: RAGHandler,
            evaluator: Evaluator,
            tuning_config: dict
            ) -> None:
        try:
            self.model = model
            self.rag_handler = rag_handler
            self.evaluator = evaluator
            # tuning "config" = TUNING_MODIFIERS (helpers/cfg.py)
            self.tuning_config = tuning_config
            # print(f"---{datetime.now()} | Tuner online")
            cfoos.LOG.info("Tuner online")
        except:
            # print(f"---{datetime.now()} | Failed to instantiate Tuner. Please contact dev.")
            cfoos.LOG.info("Failed to instantiate Tuner. Please contact dev.")

    def tune_retrieval(self, save_df: bool=False, short: bool=False) -> pd.DataFrame:
        """Go over tuning config and find best results. short=for testing purposes."""
        df_data = {
            "chunk_mode_id": [],
            "overlap_mode_id": [],
            "db_mode_id": [],
            "device": [],
            "chunk_size": [],
            "chunk_overlap": [],
            "chunk_size_factor": [],
            "chunk_overlap_factor": [],
            "hnsw_construction_ef": [], 
            "hnsw_search_ef": [], 
            "hnsw_m": [],
            "db_factor": [], 
            "correct_documents_retrieved": [], 
            "correct_chunks_retrieved": [], 
            "correct_top1_retrieved": [], 
            "average_cosine_distance": [], 
            "average_true_cosine_distance": [], 
            # (first 3 - true cosine distance)
            "average_cross_performance": [],
            "db_speed_factor": [],
            "prompt_retrieval_time": []
        }
        # ew, but running out of time...
        tuning_completion = 0
        # print(f"---{datetime.now()} | Tuning parameters...")
        log_level = cfoos.LOG.level
        cfoos.LOG.info("Tuning parameters...")
        for chunk_mode_id, chunk_mode in enumerate(self.tuning_config):
            cfoos.LOG.setLevel(log_level)
            cfoos.LOG.info(f"Tuning completed: {tuning_completion}%")
            for overlap_mode_id, overlap_mode in enumerate(self.tuning_config):
                for db_mode_id, db_mode in enumerate(self.tuning_config):
                    # avoid print cluter by redirecting stdout to a dummy stream
                    # with contextlib.redirect_stdout(io.StringIO()):
                    # modify collection config
                    cfoos.LOG.setLevel(logging.CRITICAL + 1)
                    collection_config = self.rag_handler.collection_config.copy()
                    for k, v in collection_config.items():
                        if isinstance(v, int):
                            collection_config[k] = int(round(v * db_mode, 0))
                    # tune chunk characteristics
                    chunks = self.rag_handler.get_chunks(
                        extracted_data=self.rag_handler.data,
                        chunk_size_factor=chunk_mode,
                        # overlap_factor is a factor of a factor, thus the oddity
                        chunk_overlap_factor=0.2 * overlap_mode)
                    self.rag_handler.populate_vectdb(chunks, collection_config)
                    # get performance metrics
                    anal_df = self.evaluator.prompt_db(self.evaluator.eval_df)
                    self.evaluator.analyze_retrievals(anal_df)
                    # store information
                    df_data["chunk_mode_id"].append(chunk_mode_id)
                    df_data["overlap_mode_id"].append(overlap_mode_id)
                    df_data["db_mode_id"].append(db_mode_id)
                    df_data["device"].append(self.model.device)
                    df_data["chunk_size"].append(self.rag_handler.chunk_size)
                    df_data["chunk_overlap"].append(self.rag_handler.chunk_overlap)
                    df_data["chunk_size_factor"].append(self.rag_handler.chunk_size_factor)
                    df_data["chunk_overlap_factor"].append(self.rag_handler.chunk_overlap_factor)
                    df_data["hnsw_construction_ef"].append(collection_config["hnsw:construction_ef"])
                    df_data["hnsw_search_ef"].append(collection_config["hnsw:search_ef"])
                    df_data["hnsw_m"].append(collection_config["hnsw:M"])
                    df_data["db_factor"].append(db_mode)
                    df_data["correct_documents_retrieved"].append(self.evaluator.correct_documents_retrieved)
                    df_data["correct_chunks_retrieved"].append(self.evaluator.correct_chunks_retrieved)
                    df_data["correct_top1_retrieved"].append(self.evaluator.correct_top1_retrieved)
                    df_data["average_cosine_distance"].append(self.evaluator.average_cosine_distance)
                    df_data["average_true_cosine_distance"].append(self.evaluator.average_true_cosine_distance)
                    # avg_perf = average of first 3 / cosine metrics 
                    # if true cosine distance is lower, penalty to score is lower
                    avg_perf = self.evaluator.correct_documents_retrieved + self.evaluator.correct_chunks_retrieved + self.evaluator.correct_top1_retrieved
                    avg_perf = (avg_perf / 3)
                    avg_perf = round(avg_perf - (self.evaluator.average_cosine_distance / self.evaluator.average_true_cosine_distance), 4)
                    df_data["average_cross_performance"].append(avg_perf)
                    db_speed_factor = collection_config["hnsw:construction_ef"] + collection_config["hnsw:search_ef"] + collection_config["hnsw:M"]
                    df_data["db_speed_factor"].append(db_speed_factor)
                    df_data["prompt_retrieval_time"].append(0)
                    if short:
                        break
            tuning_completion += 25
        df = pd.DataFrame().from_dict(df_data)
        if save_df:
            df.to_parquet("tuning_df.parquet", index=False)
        cfoos.LOG.setLevel(log_level)
        return df

    def select_user_config(self, tuning_df: pd.DataFrame, manual_opt: str=None) -> None:
        """
        Assess application profiles with regard to speed/performance.
        Prompt the user to opt either for speed or performance based on performance results.
        """
        # analysis variables
        diagnostic_cols = [
            "chunk_size", 
            "chunk_overlap", 
            "hnsw_m", 
            "hnsw_construction_ef", 
            "hnsw_search_ef", 
            "correct_documents_retrieved", 
            "correct_chunks_retrieved", 
            "correct_top1_retrieved", 
            "average_cross_performance", 
            "db_speed_factor"
            ]
        interest_cols = [
            "chunk_size_factor", 
            "chunk_overlap_factor", 
            "db_factor", 
            "average_cross_performance", 
            "db_speed_factor"
            ]
        # get dimension performances
        best_performance = tuning_df.sort_values(by=["average_cross_performance", "db_speed_factor"], ascending=[False, True])[interest_cols].iloc[0]
        best_speed = tuning_df.sort_values(by=["db_speed_factor", "average_cross_performance"], ascending=[True, False])[interest_cols].iloc[0]
        # print(f"---{datetime.now()} | Performance profile cross-accuracy: {best_performance['average_cross_performance']}")
        # print(f"---{datetime.now()} | Speed profile cross-accuracy: {best_speed['average_cross_performance']}")
        cfoos.LOG.info(f"Performance profile cross-accuracy: {best_performance['average_cross_performance']}")
        cfoos.LOG.info(f"Speed profile cross-accuracy: {best_speed['average_cross_performance']}")
        # user selection variables
        options = {
            "performance": best_performance,
            "speed": best_speed
        }
        if manual_opt:
            user_choice = manual_opt
        else:
            user_choice = None
        choices = {
            "chunk_size_factor": None,
            "chunk_overlap_factor": None,
            "db_factor": None
        }
        # opt for course of action
        while user_choice not in options.keys():
            user_choice = input("---What profile do you wish to set for the application? (performance/speed): ")
        # print(f"---{datetime.now()} | Building app for {user_choice}...")
        cfoos.LOG.info(f"Building app for {user_choice}...")
        selection = options[user_choice]
        # apply selection
        # store data
        for param in choices.keys():
            choices[param] = selection[param]
        collection_config = self.rag_handler.collection_config.copy()
        for k, v in collection_config.items():
            if isinstance(v, int):
                collection_config[k] = int(round(v * choices["db_factor"], 0))
        # re_run flow:
        chunks = self.rag_handler.get_chunks(
            extracted_data=self.rag_handler.data,
            chunk_size_factor=choices["chunk_size_factor"],
            chunk_overlap_factor=choices["chunk_overlap_factor"])
        self.rag_handler.populate_vectdb(chunks, collection_config)
        anal_df = self.evaluator.prompt_db(self.evaluator.eval_df)
        self.evaluator.analyze_retrievals(anal_df)
        cfoos.LOG.info("Application ready to use!")
