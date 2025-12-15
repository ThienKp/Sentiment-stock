import csv
import argparse
from tqdm import tqdm
from datasets import load_dataset
from utils.config import DATASET_DIR, ensure_dataset_dir, get_dataset_path
from utils.preprocess import text_process
from utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Editable Variable
DATA_NAME: str = "zeroshot/twitter-financial-news-sentiment"

def label_process(data: dict) -> dict :
    data["text"] = text_process(data["text"])
    new_label = ["negative", "positive", "neutral"]
    data["label"] = new_label[data["label"]]
    return data

def main(args: argparse.Namespace) -> None :
    train_data = load_dataset(DATA_NAME, split="train")
    if args.verbose:
        logger.info(f"Finish downloaded train data from {DATA_NAME}:")
        logger.info(f"There are {train_data.num_rows} entries with features {train_data.features}")
    
    test_data = load_dataset(DATA_NAME, split="validation")
    if args.verbose:
        logger.info(f"Finish downloaded train data from {DATA_NAME}:")
        logger.info(f"There are {test_data.num_rows} entries with features {test_data.features}")

    train_data = train_data.map(label_process, desc="train data processing")
    test_data = test_data.map(label_process, desc="test data processing")
    
    ensure_dataset_dir()
    with open(get_dataset_path(args.train_name + ".csv"), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id"] + train_data.column_names)

        writer.writeheader()
        for idx, data in enumerate(tqdm(train_data, desc="train data saving")):
            begin = {"id": idx}
            writer.writerow({**begin, **data})
    if args.verbose:
        logger.info(f"Successfully save the train dataset to {DATASET_DIR}")

    with open(get_dataset_path(args.test_name + ".csv"), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id"] + test_data.column_names)

        writer.writeheader()
        for idx, data in enumerate(tqdm(test_data, desc="test data saving")):
            begin = {"id": idx}
            writer.writerow({**begin, **data})
    if args.verbose:
        logger.info(f"Successfully save the test dataset to {DATASET_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download financial tweets dataset')
    parser.add_argument("--train_name", default="finance_train", help="Custom train dataset name")
    parser.add_argument("--test_name", default="finance_test", help="Custom test dataset name")
    parser.add_argument("--verbose", action="store_true", help="Verbose the output")
    args = parser.parse_args()
    main(args)
