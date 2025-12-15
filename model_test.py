import torch
import argparse
import numpy as np
from tqdm import tqdm
from scipy.special import softmax

from utils.config import BATCH_SIZE, load_data, load_model, load_local_model
from utils.performance import eval_metrics
from utils.preprocess import text_process
from utils.logging_config import setup_logger

logger = setup_logger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Edit Variable
FILENAME: str = "finance_test.csv"

def ans_equalizer(ans: str):
    sentiments = {
        "BEARISH": "negative",
        "NEUTRAL": "neutral",
        "BULLISH": "positive"
    }
    return sentiments[ans]

def main(args: argparse.Namespace) -> None:
    if args.local_model:
        model, tokenizer, config = load_local_model(args.local_model, args.verbose)
    elif args.model:
        model, tokenizer, config = load_model(args.model, args.verbose)
    else:
        raise ImportError("No model to test")
    model = model.to(device)
    data, labels, total = load_data(FILENAME, args.verbose)
    
    pred_labels = []
    if args.verbose:
        logger.info(f"Process data in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Processing data"):
        texts = data[i: min(i + BATCH_SIZE, total)]
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

    acc, macro_f1 = eval_metrics(labels, pred_labels)
    logger.info(f"accuracy: {acc}")
    logger.info(f"f1 score: {macro_f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Testing")
    parser.add_argument("--model", type=str, choices=["twitter", "finance", "fintwit"], help="Model type (twitter, finance, fintwit)")
    parser.add_argument("--local_model", type=str, help="Local model (local_model/...)")
    parser.add_argument("--verbose", action="store_true", help="Verbose the output")
    args = parser.parse_args()
    main(args)
