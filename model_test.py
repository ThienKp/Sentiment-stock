import csv
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax

from utils.config import BATCH_SIZE, get_dataset_path, MODEL_NAME
from utils.preprocess import preprocess, batch_preprocess
from utils.performance import eval_metrics
from utils.logging_config import setup_logger

logger = setup_logger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Edit Variable
EXAMPLE_TEXT: str = "Covid cases are increasing fast!"
FILENAME: str = "finance_test.csv"

def load_model(model_type: str) -> tuple:
    """Load model from huggingface"""
    logger.info(MODEL_NAME[model_type])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME[model_type])
    config = AutoConfig.from_pretrained(MODEL_NAME[model_type])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME[model_type]).to(device)
    return model, tokenizer, config

def load_data(filename: str) -> tuple:
    """Load data from csv dataset"""
    texts = []
    labels = []
    with open(get_dataset_path(filename), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels

def ans_equalizer(ans: str):
    sentiments = {
        "BEARISH": "negative",
        "NEUTRAL": "neutral",
        "BULLISH": "positive"
    }
    return sentiments[ans]

def main(args: argparse.Namespace) -> None:
    start_time = time.time()
    model, tokenizer, config = load_model(args.model)
    if args.verbose:
        end_time = time.time()
        logger.info(f"Loading model in {end_time - start_time}s")

    if args.example:
        input = preprocess(EXAMPLE_TEXT)
        encoded_input = tokenizer(input, return_tensors='pt').to(device)
        output = model(**encoded_input)
        scores = output[0][0].detach().cpu().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = config.id2label[ranking[i]]
            s = scores[ranking[i]]
            print(f"{i+1}) {l} {np.round(float(s), 4)}")

    else:
        start_time = time.time()
        data, labels = load_data(FILENAME)
        if args.verbose:
            end_time = time.time()
            logger.info(f"Loading data from {FILENAME} in {end_time - start_time}s")
        total = len(data)
        pred_labels = []
        if args.verbose:
            logger.info(f"Process data in batches of {BATCH_SIZE}...")
        for i in tqdm(range(0, total, BATCH_SIZE), desc="Processing data"):
            texts = batch_preprocess(data[i: min(i + BATCH_SIZE, total)])
            encoded_texts = tokenizer(texts, return_tensors='pt', padding=True).to(device)
            outputs = model(**encoded_texts)
            scores = outputs[0].detach().cpu().numpy()
            scores = softmax(scores, axis=1)

            rankings = np.argmax(scores, axis=1)
            for ans in rankings:
                ans = config.id2label[ans]
                if args.model == "fintwit":
                    ans = ans_equalizer(ans)
                pred_labels.append(ans)
        acc, macro_f1 = eval_metrics(np.array(labels), np.array(pred_labels))
        logger.info(f"accuracy: {acc}")
        logger.info(f"f1 score: {macro_f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Testing")
    parser.add_argument("--model", type=str, choices=["twitter", "finance", "fintwit"])
    parser.add_argument("--example", action="store_true", help="Test with one-sentence example")
    parser.add_argument("--verbose", action="store_true", help="Verbose the output")
    args = parser.parse_args()
    main(args)
