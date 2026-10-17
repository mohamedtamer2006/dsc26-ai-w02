"""Performance and decision metrics for binary classification and threshold selection."""

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def _validate_inputs(
    y_true: Union[Sequence, np.ndarray],
    scores: Union[Sequence, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate y_true and scores inputs according to specification."""
    try:
        y_arr = np.asarray(y_true)
        s_arr = np.asarray(scores, dtype=float)
    except Exception as e:
        raise ValueError(f"Could not convert inputs to arrays: {e}") from e

    if y_arr.ndim != 1 or s_arr.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional.")

    n = len(y_arr)
    if n == 0 or len(s_arr) != n:
        raise ValueError(f"Lengths must match and be non-empty. Got {len(y_arr)} and {len(s_arr)}.")

    if np.any(np.isnan(s_arr)):
        raise ValueError("Scores must not contain NaN.")

    # Check that labels are binary 0 or 1
    # Allow bool or 0/1 integers or floats equal to 0.0 / 1.0
    for y in y_arr:
        if isinstance(y, (bool, np.bool_)):
            val = int(y)
        elif isinstance(y, (int, float, np.integer, np.floating)):
            if np.isnan(y) or y not in (0, 1):
                raise ValueError(f"Labels must be 0 or 1, got {y}")
            val = int(y)
        else:
            raise ValueError(f"Labels must be 0 or 1, got {y}")

    y_int = y_arr.astype(int)
    return y_int, s_arr


def confusion_at(
    y_true: Union[Sequence, np.ndarray],
    scores: Union[Sequence, np.ndarray],
    threshold: float,
) -> Dict[str, int]:
    """Compute confusion matrix at a given threshold.

    A case is flagged when score >= threshold.
    """
    y_arr, s_arr = _validate_inputs(y_true, scores)
    flagged = s_arr >= threshold

    tp = int(np.sum((y_arr == 1) & flagged))
    fp = int(np.sum((y_arr == 0) & flagged))
    tn = int(np.sum((y_arr == 0) & (~flagged)))
    fn = int(np.sum((y_arr == 1) & (~flagged)))

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_at(
    y_true: Union[Sequence, np.ndarray],
    scores: Union[Sequence, np.ndarray],
    threshold: float,
) -> Tuple[float, float]:
    """Compute (precision, recall) at a given threshold.

    Precision is 0.0 when nothing is flagged.
    Recall is 0.0 when there are no positives.
    """
    conf = confusion_at(y_true, scores, threshold)
    tp = conf["tp"]
    fp = conf["fp"]
    fn = conf["fn"]

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = float(tp / (tp + fp))

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = float(tp / (tp + fn))

    return (precision, recall)


def precision_at_k(
    y_true: Union[Sequence, np.ndarray],
    scores: Union[Sequence, np.ndarray],
    k: int,
) -> float:
    """Compute precision at top k ranked scores.

    Ties preserve original order.
    Raises ValueError if k <= 0. If k > n, use n.
    """
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    y_arr, s_arr = _validate_inputs(y_true, scores)
    n = len(y_arr)
    k_eval = min(int(k), n)

    # Stable sort indices in descending order of score
    # Python's built-in sorted is Timsort, which is guaranteed stable.
    indices = sorted(range(n), key=lambda i: s_arr[i], reverse=True)[:k_eval]
    num_positives = sum(1 for idx in indices if y_arr[idx] == 1)

    return float(num_positives / k_eval)


def expected_cost(
    y_true: Union[Sequence, np.ndarray],
    scores: Union[Sequence, np.ndarray],
    threshold: float,
    *,
    cost_fp: float,
    cost_fn: float,
) -> float:
    """Calculate expected operational cost given false positive and false negative costs."""
    conf = confusion_at(y_true, scores, threshold)
    return float(conf["fp"] * float(cost_fp) + conf["fn"] * float(cost_fn))


def best_threshold(
    y_true: Union[Sequence, np.ndarray],
    scores: Union[Sequence, np.ndarray],
    *,
    cost_fp: float,
    cost_fn: float,
) -> float:
    """Find the threshold with lowest expected cost.

    Candidates are every distinct score plus math.inf ('flag nothing').
    On a tie, return the highest threshold (fewer flags for the same cost).
    """
    y_arr, s_arr = _validate_inputs(y_true, scores)
    distinct_scores = set(s_arr.tolist())
    candidates = distinct_scores | {math.inf}

    # Sorting descending by threshold ensures that ties in cost retain the highest threshold
    sorted_candidates = sorted(candidates, reverse=True)

    best_thresh = sorted_candidates[0]
    min_cost = expected_cost(y_arr, s_arr, best_thresh, cost_fp=cost_fp, cost_fn=cost_fn)

    for thresh in sorted_candidates[1:]:
        cost = expected_cost(y_arr, s_arr, thresh, cost_fp=cost_fp, cost_fn=cost_fn)
        if cost < min_cost:
            min_cost = cost
            best_thresh = thresh

    return float(best_thresh)


def threshold_for_precision(
    y_true: Union[Sequence, np.ndarray],
    scores: Union[Sequence, np.ndarray],
    min_precision: float,
) -> Optional[float]:
    """Find the lowest threshold among distinct scores whose precision is >= min_precision.

    Returns None if no threshold reaches the required precision.
    """
    y_arr, s_arr = _validate_inputs(y_true, scores)
    distinct_scores = sorted(set(s_arr.tolist()))

    valid_thresholds: List[float] = []
    for thresh in distinct_scores:
        prec, _ = precision_recall_at(y_arr, s_arr, thresh)
        if prec >= min_precision:
            valid_thresholds.append(thresh)

    if not valid_thresholds:
        return None

    # Return the lowest threshold that satisfies the constraint
    return float(min(valid_thresholds))
