# File: development_bankd_wales/pipeline/predictive_model/evaluation.py
"""
Evaluation of predictive model.
"""
# ----------------------------------------------------------------------------------

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# ----------------------------------------------------------------------------------


def print_metrics(labels, predictions):
    """Print evaluation metrics for predictive model.

    Args:
        labels (np.array): Ground truth labels.
        predictions (np.array): Predictions for samples.
    """

    acc = np.round(accuracy_score(labels, predictions) * 100, 2)
    f1 = np.round(f1_score(labels, predictions) * 100, 2)
    precision = np.round(precision_score(labels, predictions) * 100, 2)
    recall = np.round(recall_score(labels, predictions) * 100, 2)

    print("Accuracy:\t{}%".format(acc))
    print("F1 score:\t{}%".format(f1))
    print("Recall:\t\t{}%".format(precision))
    print("Precision:\t{}%".format(recall))


def get_baseline(n_samples, true_rate):
    """Get baseline model by creating random predictions.
    The true_rate indicates how many True/1 values should be integrated.
    For example, with a true_rate of 0.4 and 100 samples --> 40 samples are predicted as True, 60 as False.

    Args:
        n_samples (int): Number of samples (for creating baseline predictions).
        true_rate (float): How many predictions should be True/1.

    Returns:
        baseline: Baseline model predictions.
    """

    baseline = np.zeros((n_samples))
    n_true = n_samples * true_rate
    n_trues_indices = np.random.choice(
        range(0, n_samples), round(n_true), replace=False
    )
    baseline[n_trues_indices] = 1.0
    return baseline
