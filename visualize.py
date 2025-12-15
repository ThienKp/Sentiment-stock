import os
import json
import matplotlib.pyplot as plt

from utils.config import STATE_DIR
# STATE_DIR: str = os.path.join(BASE_DIR, "state_history_1")

def main() -> None:
    finance_best_acc = 0
    finance_setup = None
    twitter_best_acc = 0
    twitter_setup = None

    # Find the highest accuracy, each from finance and twitter
    for filename in os.listdir(STATE_DIR):
        with open(os.path.join(STATE_DIR, filename)) as f:
            data = json.load(f)
            acc = max([item["eval_accuracy"] for item in data if "eval_accuracy" in item])
            if ("finance" in filename) and (acc > finance_best_acc):
                finance_best_acc = acc
                finance_setup = os.path.splitext(filename)[0]
            elif ("twitter" in filename) and (acc > twitter_best_acc):
                twitter_best_acc = acc
                twitter_setup = os.path.splitext(filename)[0]

    # Specify 2 files with default epoch 0 value
    finance_acc = [0.7427972760607648]
    finance_f1 = [0.6373314164995459]
    twitter_acc = [0.6961760083813515]
    twitter_f1 = [0.602637178339195]
    print("Best Finance Setup:", finance_setup)
    print("Best Twitter Setup:", twitter_setup)
    with open(os.path.join(STATE_DIR, finance_setup + ".json")) as f:
        data = json.load(f)
        for item in data:
            if "eval_accuracy" in item:
                finance_acc.append(item["eval_accuracy"])
                finance_f1.append(item["eval_macro_f1"])
    with open(os.path.join(STATE_DIR, twitter_setup + ".json")) as f:
        data = json.load(f)
        for item in data:
            if "eval_accuracy" in item:
                twitter_acc.append(item["eval_accuracy"])
                twitter_f1.append(item["eval_macro_f1"])

    # Plot accuracy and macro f1 score over epoch
    fig, ax = plt.subplots()

    ax.plot(range(6), finance_acc, "o-", label="Finance Accuracy")
    ax.plot(range(6), finance_f1, "o-", label="Finance Macro F1")
    ax.plot(range(6), twitter_acc, "o-", label="Twitter Accuracy")
    ax.plot(range(6), twitter_f1, "o-", label="Twitter Macro F1")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])

    fig.suptitle("Finance vs Twitter\nAccuracy & Macro F1 over epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_xticks(range(6))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True, shadow=True, ncol=2)
    ax.grid()
    plt.savefig("visual.png")

if __name__ == "__main__":
    main()