from pathlib import Path


# globals
APP_NAME = "DocsIntern"
SUPPORTED_EXTENSIONS = ["txt", "docx", "pdf", "md"]
# get project directory
# workaround as cfg.py can be imported from different locations
CFG_PATH = Path(__file__).resolve().parent
APP_DIR = CFG_PATH.parent
TESTS_PATH = APP_DIR / "tests"
# set model and component directories
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_CPU = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
MODEL_EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = MODEL.split("/")[1]
MODEL_NAME_CPU = MODEL_CPU.split("/")[1]
MODEL_NAME_EMB = MODEL_EMB.split("/")[1]
MODELS_DEV_PATH = APP_DIR / "hf"
MODELS_QUANT_PATH = APP_DIR / "quant_models"
# db config
DB_PATH = APP_DIR / "db"
DB_COLLECTION = "user_data"
COLLECTION_CONFIG = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 100,
    "hnsw:search_ef": 100,
    "hnsw:M": 16
    }
# model config
SYSTEM_PROMPT = """You are a helpful assistant.
Keep your responses short.
If you don't know something, either state your lack of knowledge.
"""
MODEL_CONFIG = {
    "do_sample": True,
    "temperature": 0.7,  
    "top_k": 50,
    "top_p": 0.9,
    "tokenizer_path": MODELS_QUANT_PATH / MODEL_NAME,
    "gpu_model_path": MODELS_QUANT_PATH / MODEL_NAME,
    "cpu_model_path": MODELS_QUANT_PATH / MODEL_NAME_CPU / "Meta-Llama-3.1-8B-Instruct-Q3_K_XL.gguf",
    "emb_model_path": MODELS_QUANT_PATH / MODEL_NAME_EMB,
    "system_prompt": SYSTEM_PROMPT,
    "database_path": DB_PATH,
    "collection_name": DB_COLLECTION
}
# tuning config
TUNING_MODIFIERS = [0.5, 1, 1.5, 2.0]
