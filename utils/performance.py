import numpy as np
from utils.logging_config import setup_logger

from transformers import EvalPrediction

logger = setup_logger(__name__)

def eval_metrics(y_gt: list, y_pred: list, usage: str = "test") -> tuple:
    """
    Calculates accuracy and macro F1 score from ground truth and prediction

    Parameters
    ----------
    y_gt : list
        Ground truth labels
    y_pred : list
        Predicted labels

    Returns
    -------
    acc : float
    macro_f1 : float
    """

    if len(y_gt) != len(y_pred):
        raise ValueError(f"Dimension of y_gt ({len(y_gt)}) and y_pred ({len(y_pred)}) are not matched")
    y_gt = np.array(y_gt)
    y_pred = np.array(y_pred)
    f1_label = []
    labels = np.unique(y_gt)

    for label in labels:
        predict_label = y_pred == label
        gt_label = y_gt == label
        tp = np.sum(predict_label & gt_label)

        precision = tp / np.sum(predict_label)
        recall = tp / np.sum(gt_label)

        f1_label.append(2 * precision * recall / (precision + recall))

    acc = np.sum(y_gt == y_pred) / y_gt.size
    macro_f1 = np.mean(f1_label)

    return acc, macro_f1

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc, f1 = eval_metrics(labels, preds)

    return {"accuracy": acc, "macro_f1": f1}
