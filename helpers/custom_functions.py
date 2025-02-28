import os
import shutil
from pathlib import Path
import logging
from torch import cuda
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_log(named_log_path: str="docs_intern.log") -> logging.Logger:
    """Setup logging config."""
    logger = logging.getLogger("custom_logger")
    # config logger
    if not logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("---%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # for some reason the file_handler breaks in gradio
    file_handler = logging.FileHandler(named_log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def _set_log_stream(log: logging.Logger, stream_handler: logging.StreamHandler) -> None:
    """Replaces the stream handler of an existing log. Requires existing format."""
    for handler in log.handlers:
        if isinstance(handler, logging.StreamHandler):
            message = handler.formatter._fmt
            datefmt = handler.formatter.datefmt
            log.removeHandler(handler)
    formatter = logging.Formatter(message, datefmt)
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

def clean(directory: Path) -> None:
    """Delete all the folders in a directory."""
    print(f"---Cleaning: {directory}")
    for folder in os.listdir(directory):
        folder_path = directory / folder
        if folder_path.is_dir():
            shutil.rmtree(directory / folder)

def save_quant_gpu_model(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_path: Path,
        clear_dev_models_dir: Path=None
        ) -> None:
    """
    Save gpu-quantized model & tokenizer to model_path. 
    If clear_dev_models_dir, empty dev_models folder.
    """

    print("---Saving gpu-quantized model...")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    if clear_dev_models_dir:
        clean(clear_dev_models_dir)
    print(f"---Saved model to: {model_path}")

def save_quant_cpu_model(
        model_name: str,
        dev_models_dir: str,
        model_path: Path,
        clear_dev_models_dir: Path=None
        ) -> None:
    """
    Save cpu-quantized model & tokenizer to model_path. 
    If clear_dev_models_dir, empty dev_models folder.
    This function is different from save_quant_gpu_model because
    working with onnx files is different.
    """
    # TODO: update with Path 
    print("---Saving cpu-quantized model...")
    for folder in os.listdir(dev_models_dir):
        if model_name in folder:
            # extract the essentials files from cpu-model folder
            cpu_model_path = dev_models_dir / folder
            # list all files
            for obj in cpu_model_path.rglob("*"):
                if obj.is_file():
                    # note to self: re-work this monstrosity
                    # if "json" in obj.name or "onnx" in obj.name:
                    if "gguf" in obj.name:
                        # copy them from dev_models to production folder
                        shutil.copy(obj, model_path)

    if clear_dev_models_dir:
        clean(clear_dev_models_dir)
    print(f"---Saved model to: {model_path}")

def save_emb_model(
        model_name: str,
        dev_models_dir: Path,
        model_path: Path,
        clear_dev_models_dir: Path=None
        ) -> None:
    retain_extensions = [".json", ".safetensors"]
    ignore_folder = ".no_exist"
    for subfolder in dev_models_dir.iterdir():
        if model_name in subfolder.name:
            for file in subfolder.rglob("*"):
                # files in .no_exist mess up with the loading process &
                # there are 2 config files into 2 different folders (pooling)
                if file.is_file() and \
                    file.suffix in retain_extensions and \
                        ignore_folder not in file.parts:
                    if "pooling" in file.parent.name.lower():
                        save_path = check_create_dir(model_path / file.parent.name) / file.name
                    else:
                        save_path = model_path / file.name
                    shutil.copy(file, save_path)
  
    if clear_dev_models_dir:
        clean(clear_dev_models_dir)
    print(f"---Saved embedding model to: {model_path}")

def check_gpu_memory() -> None:
    if cuda.is_available():
        # get memory stats for the first GPU
        total_memory = cuda.get_device_properties(0).total_memory
        reserved_memory = cuda.memory_reserved(0)
        allocated_memory = cuda.memory_allocated(0)
        free_memory = reserved_memory - allocated_memory
        print(f"Total GPU memory: {total_memory / 1e9:.2f} GB")
        print(f"Reserved GPU memory: {reserved_memory / 1e9:.2f} GB")
        print(f"Allocated GPU memory: {allocated_memory / 1e9:.2f} GB")
        print(f"Free GPU memory: {free_memory / 1e9:.2f} GB")
    else:
        print("CUDA is not available.")

def check_gpu_model_size(model: AutoModelForCausalLM) -> None:
    print(f"Model memory (GB): {round(model.get_memory_footprint() / (1024 ** 3), 2)}")

def check_cpu_model_size(model_path: Path) -> None:
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            total_size += os.path.getsize(file_path)

    model_size = total_size / (1024 ** 3)
    print(f"Model memory (GB): {round(model_size, 2)}")

def check_create_dir(path: Path) -> Path:
    """
    Check if a directory exists.
    If not, create it.
    """
    if not path.exists():
        path.mkdir()
        # print(f"---Created: {path}")
        LOG.info(f"Created: {path}")
    return path

def inspect_gpu_model(model: AutoModelForCausalLM) -> None:
    """Self-explanatory, ey?"""
    messages = []
    messages.append("---PARAMS:")
    for name, param in model.named_parameters():
        messages.append(f"{name}: {param.device}")
    messages.append("---BUFFERS:")
    for name, buffer in model.named_buffers():
        messages.append(f"{name}: {buffer.device}")
    messages.append("---CONFIG")
    messages.append(model.config)
    for message in messages:
        print(message)

LOG = get_log()
