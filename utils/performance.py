import numpy as np
from utils.logging_config import setup_logger

logger = setup_logger(__name__)

def eval_metrics(y_gt: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Calculates accuracy and macro F1 score from ground truth and prediction

    Parameters
    ----------
    y_gt : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    acc : float
    macro_f1 : float
    """

    acc = np.sum(y_gt == y_pred) / y_gt.size
    precision = []
    recall = []

    for label in ["positive", "neutral", "negative"]:
        predict_label = y_pred == label
        gt_label = y_gt == label
        tp = np.sum(predict_label & gt_label)

        precision.append(tp / np.sum(predict_label))
        recall.append(tp / np.sum(gt_label))

    macro_f1 = sum([2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(3)]) / 3

    return acc, macro_f1

class PerformanceTracker:
    """Tracker for checking performance of the experiments"""

    def __init__(self, experiment_name: str):
        self.name = experiment_name
