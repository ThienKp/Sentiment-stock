import os

UTILS_DIR: str = os.path.dirname(os.path.realpath(__file__))
PROJ_DIR: str = os.path.dirname(UTILS_DIR)
DATASET_DIR: str = os.path.join(PROJ_DIR, "dataset")
BATCH_SIZE: int = 128
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
