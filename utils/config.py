import os
import csv
import time
import shutil

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig

from utils.logging_config import setup_logger
logger = setup_logger(__name__)

UTILS_DIR: str = os.path.dirname(os.path.realpath(__file__))
BASE_DIR: str = os.path.dirname(UTILS_DIR)
DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")
FFT_DIR: str = os.path.join(BASE_DIR, "model_result")
STATE_DIR: str = os.path.join(BASE_DIR, "state_history")
BATCH_SIZE: int = 8
MODEL_NAME: dict = {
    "twitter": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "finance": "ahmedrachid/FinancialBERT-Sentiment-Analysis",
    "fintwit": "StephanAkkerman/FinTwitBERT-sentiment"
}

def ensure_dataset_dir() -> None :
    """Create dataset directory if it doesn't exist"""
    os.makedirs(DATASET_DIR, exist_ok=True)

def get_dataset_path(filename: str) -> str :
    """Get dataset path from file name"""
    return os.path.join(DATASET_DIR, filename)

def ensure_fft_dir() -> None :
    """Create fully fine-tuned result directory if it doesn't exist"""
    os.makedirs(FFT_DIR, exist_ok=True)

def get_fft_path(filename: str) -> str :
    """Get fully fine-tuned result path from file name"""
    return os.path.join(FFT_DIR, filename)

def ensure_state_dir() -> None :
    """Create state history directory if it doesn't exist"""
    os.makedirs(STATE_DIR, exist_ok=True)

def get_state_dir(filename: str) -> str :
    """Get state history path from file name"""
    return os.path.join(STATE_DIR, filename)

def load_model(model_type: str, verbose: bool = False) -> tuple:
    """Load model from huggingface"""
    if verbose:
        start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME[model_type])
    config = AutoConfig.from_pretrained(MODEL_NAME[model_type])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME[model_type])
    if verbose:
        end_time = time.time()
        logger.info(f"Loading model from {MODEL_NAME[model_type]} in {end_time - start_time}s")
    return model, tokenizer, config

def load_local_model(model_path: str, verbose: bool = False) -> tuple:
    """Load model from local"""
    if verbose:
        start_time = time.time()
    model_path = os.path.join(BASE_DIR, model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    if verbose:
        end_time = time.time()
        logger.info(f"Loading model from {model_path} in {end_time - start_time}s")
    return model, tokenizer, config

def load_data(filename: str, verbose: bool = False) -> tuple:
    """Load data from csv dataset"""
    if verbose:
        start_time = time.time()
    texts = []
    labels = []
    with open(get_dataset_path(filename), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"])
    if verbose:
        end_time = time.time()
        logger.info(f"Loading data from {filename} in {end_time - start_time}s")
    return texts, labels, len(texts)

def move_files_up_and_delete(parent_dir: str):
    """Move all files from the sub directory to its parent directory"""
    for subdir in os.listdir(parent_dir):
        subdir = os.path.join(parent_dir, subdir)
        if os.path.isdir(subdir):
            for file in os.listdir(subdir):
                file_path = os.path.join(subdir, file)
                shutil.move(file_path, parent_dir)
            os.rmdir(subdir)
