import json
import random
import argparse

from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from scipy.special import softmax

from utils.config import BATCH_SIZE, get_dataset_path, ensure_fft_dir, get_fft_path, ensure_state_dir, get_state_dir, load_model, move_files_up_and_delete
from utils.performance import compute_metrics
from utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Edit Variable
FILENAME: str = "finance_train.csv"
SEED: int = 51515 # Random Seed
random.seed(SEED)

def main(args: argparse.Namespace) -> None:
    model, tokenizer, config = load_model(args.model, args.verbose)
    raw_datasets = load_dataset("csv", data_files=get_dataset_path(FILENAME), split="train")

    # Tokenized Input in batches
    def tokenize_function(data):
        return tokenizer(data["text"], truncation=True)
    data_collator = DataCollatorWithPadding(tokenizer)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, desc="Batched tokenizing")

    # Process data
    processed_datasets = tokenized_datasets.map(lambda data: {"label": config.label2id[data["label"]]}, desc="Preprocessing")

    # ind = random.sample(range(len(processed_datasets)), 1000)
    # processed_datasets = processed_datasets.select(ind)
    processed_datasets = processed_datasets.train_test_split(test_size=0.2, shuffle=True, seed=SEED)
    train_ds = processed_datasets["train"]
    eval_ds = processed_datasets["test"]

    ensure_fft_dir()
    training_args = TrainingArguments(
        output_dir=get_fft_path(f"fft-model_{args.model}-lr_{args.lr}-decay_{args.decay}"),
        eval_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=args.lr,
        weight_decay=args.decay,
        num_train_epochs=args.epoch,
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        seed=SEED,
        data_seed=SEED,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    move_files_up_and_delete(get_fft_path(f"fft-model_{args.model}-lr_{args.lr}-decay_{args.decay}"))

    ensure_state_dir()
    with open(get_state_dir(f"fft-model_{args.model}-lr_{args.lr}-decay_{args.decay}.json"), 'w') as f:
        json.dump(trainer.state.log_history, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully Fine-tuning Model")
    parser.add_argument("--model", type=str, choices=["twitter", "finance", "fintwit"], help="Model type (twitter, finance, fintwit)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--decay", type=float, default=0.01, help="Weight decay (L2 regularization)")
    parser.add_argument("--epoch", type=int, default=5, help="Epoch number")
    parser.add_argument("--verbose", action="store_true", help="Verbose the output")
    args = parser.parse_args()
    main(args)
