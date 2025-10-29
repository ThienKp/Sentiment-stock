import os

UTILS_DIR: str = os.path.dirname(os.path.realpath(__file__))
PROJ_DIR: str = os.path.dirname(UTILS_DIR)
DATASET_DIR: str = os.path.join(PROJ_DIR, "dataset")

def ensure_dataset_dir() -> None :
    """Create dataset directory if it doesn't exist"""
    os.makedirs(DATASET_DIR, exist_ok=True)
