"""Metrics module for model evaluation and calibration."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def calculate_all_metrics(y_true, y_pred, y_prob, return_dict=True):
    """Calculate all evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'ece': expected_calibration_error(y_true, y_prob, n_bins=10)
    }

    return metrics if return_dict else pd.Series(metrics)
