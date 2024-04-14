from typing import List

import numpy as np

from elf.evaluation import mean_segmentation_accuracy


SUPPORTED_EVALS = [
    "msa", "sa50", "sa55", "sa60", "sa65", "sa70", "sa75", "sa80", "sa85", "sa90", "sa95",
]


def evaluate_predictions(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    metrics: List[str] = ["msa"]
):
    if isinstance(metrics, str):
        metrics = [metrics]

    results = {}
    for metric in metrics:
        results[metric] = evaluate_sample(prediction, ground_truth, metric.lower())

    return results


def evaluate_sample(prediction, ground_truth, metric):
    assert metric in SUPPORTED_EVALS

    if metric == "msa" or metric.startswith("sa"):
        msa, sa = mean_segmentation_accuracy(prediction, ground_truth, return_accuracies=True)

        if metric == "msa":
            score = msa
        else:
            values = list(np.arange(50, 100, 5))
            score = sa[values.index(int(metric[2:]))]

    return score
