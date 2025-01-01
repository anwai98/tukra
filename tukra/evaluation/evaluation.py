from typing import List, Dict

import numpy as np

from elf.evaluation import mean_segmentation_accuracy, dice_score


SUPPORTED_EVALS = [
    "dice_score", "msa", "sa50", "sa55", "sa60", "sa65", "sa70", "sa75", "sa80", "sa85", "sa90", "sa95",
]


def evaluate_predictions(
    prediction: np.ndarray, ground_truth: np.ndarray, metrics: List[str] = ["msa"], **metric_kwargs
) -> Dict:
    """Functionality to evaluate predictions w.r.t. the ground-truth for a specific metric.

    Args:
        prediction: The predicted labels.
        ground_truth: The ground-truth labels.
        metrics: The choice of metric for evaluation.
        metric_kwargs: Additional arguments for changing default values of evaluation measures.

    Returns:
        A dictionary of results.
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    results = {}
    for metric in metrics:
        results[metric] = _evaluate_per_sample(prediction, ground_truth, metric.lower(), **metric_kwargs)

    return results


def _evaluate_per_sample(prediction, ground_truth, metric, **metric_kwargs) -> float:
    if metric == "msa" or metric.startswith("sa"):
        msa, sa = mean_segmentation_accuracy(prediction, ground_truth, return_accuracies=True, **metric_kwargs)
        if metric == "msa":
            score = msa
        else:
            values = list(np.arange(50, 100, 5))
            score = sa[values.index(int(metric[2:]))]

    elif metric == "dice_score":
        score = dice_score(
            segmentation=prediction, groundtruth=ground_truth, threshold_seg=0.5, threshold_gt=0.5, **metric_kwargs
        )

    else:
        raise ValueError(f"'{metric}' is not a support metric by 'tukra'.")

    return score
