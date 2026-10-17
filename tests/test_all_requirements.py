"""Comprehensive tests for Rawaj audit and metrics modules matching the autograder specifications."""

import math
import os
import numpy as np
import pandas as pd
import pytest

from rawaj.audit import audit
from rawaj.metrics import (
    confusion_at,
    precision_recall_at,
    precision_at_k,
    expected_cost,
    best_threshold,
    threshold_for_precision,
)


def test_audit_counts_rows_and_duplicates():
    df = pd.DataFrame({
        "txn_id": ["T1", "T2", "T1", " ", None, "T3"],
        "target": [0, 1, 0, 0, 1, 0],
        "timestamp": ["2026-09-01 10:00"] * 6,
        "amount": [100, 200, 100, 300, 400, 500]
    })
    res = audit(df, target="target", id_col="txn_id", time_col="timestamp")
    assert res["n_rows"] == 6
    assert res["duplicate_ids"] == ["T1"]  # Missing IDs ignored
    assert res["duplicate_rows"] == 1  # Row 2 repeats Row 0 exactly


def test_audit_blank_strings_are_missing():
    df = pd.DataFrame({
        "id": ["1", "2", "3", "4"],
        "target": [1, 0, 1, 0],
        "timestamp": ["2026-09-01 10:00"] * 4,
        "text_col": ["hello", "", "   \t\n", "world"],
        "num_col": [10.5, 0.0, -1.0, 50.0]
    })
    res = audit(df, target="target", id_col="id", time_col="timestamp")
    assert res["missing"]["text_col"] == 0.5
    assert res["missing"]["num_col"] == 0.0


def test_audit_constant_columns_ignore_missing():
    df = pd.DataFrame({
        "id": [f"ID_{i}" for i in range(5)],
        "target": [0, 1, 0, 1, 0],
        "timestamp": ["2026-09-01 10:00"] * 5,
        "curr": ["EGP", "EGP", "  ", None, "EGP"],  # one distinct non-missing ("EGP")
        "empty_col": ["", None, "  ", None, ""],     # zero distinct non-missing -> constant
        "normal_col": [1, 2, 3, 4, 5]
    })
    res = audit(df, target="target", id_col="id", time_col="timestamp")
    assert "target" not in res["constant_columns"]
    assert "curr" in res["constant_columns"]
    assert "empty_col" in res["constant_columns"]
    assert "normal_col" not in res["constant_columns"]


def test_audit_unlabelled_rows():
    df = pd.DataFrame({
        "id": ["1", "2", "3", "4", "5"],
        "target": [1, " ", None, 0, 1],
        "timestamp": ["2026-09-01 10:00"] * 5
    })
    res = audit(df, target="target", id_col="id", time_col="timestamp")
    assert res["unlabelled_rows"] == 2
    # 3 labelled rows: [1, 0, 1] -> positive rate 2/3
    assert abs(res["positive_rate"] - (2 / 3)) < 1e-6

    # All unlabelled test
    df_empty = pd.DataFrame({
        "id": ["1", "2"],
        "target": ["", None],
        "timestamp": ["2026-09-01 10:00"] * 2
    })
    res_empty = audit(df_empty, target="target", id_col="id", time_col="timestamp")
    assert res_empty["unlabelled_rows"] == 2
    assert res_empty["positive_rate"] == 0.0


def test_audit_bad_timestamps_and_range():
    df = pd.DataFrame({
        "id": ["1", "2", "3", "4", "5"],
        "target": [1, 0, 1, 0, 1],
        "timestamp": [
            "2026-09-01 10:00",
            "2026-02-30 10:00",  # impossible date -> bad
            "invalid_date_str",   # unparseable -> bad
            "",                   # blank -> missing, not bad
            "2026-09-05 14:30"
        ]
    })
    res = audit(df, target="target", id_col="id", time_col="timestamp")
    assert res["bad_timestamps"] == 2
    assert res["time_range"] == ["2026-09-01T10:00:00", "2026-09-05T14:30:00"]

    # All bad
    df_all_bad = pd.DataFrame({
        "id": ["1", "2"],
        "target": [0, 1],
        "timestamp": ["bad1", "bad2"]
    })
    res_bad = audit(df_all_bad, target="target", id_col="id", time_col="timestamp")
    assert res_bad["time_range"] == [None, None]


def test_audit_flags_value_leak():
    n = 30
    y = [0] * 15 + [1] * 15
    # Near copy of target
    val_leak = [0.01 if yi == 0 else 0.99 for yi in y]
    # Inverse copy
    inv_leak = [0.99 if yi == 0 else 0.01 for yi in y]
    df = pd.DataFrame({
        "id": [f"ID_{i}" for i in range(n)],
        "target": y,
        "timestamp": ["2026-09-01 10:00"] * n,
        "val_leak": val_leak,
        "inv_leak": inv_leak,
        "normal_feat": np.random.RandomState(42).randn(n)
    })
    res = audit(df, target="target", id_col="id", time_col="timestamp")
    assert "val_leak" in res["leakage_suspects"]
    assert "inv_leak" in res["leakage_suspects"]
    assert "normal_feat" not in res["leakage_suspects"]


def test_audit_flags_missingness_leak():
    n = 30
    y = [0] * 15 + [1] * 15
    # Present only for positives, blank for negatives
    missing_leak = ["" if yi == 0 else "RESOLVED_CASE" for yi in y]
    df = pd.DataFrame({
        "id": [f"ID_{i}" for i in range(n)],
        "target": y,
        "timestamp": ["2026-09-01 10:00"] * n,
        "case_closed_at": missing_leak
    })
    res = audit(df, target="target", id_col="id", time_col="timestamp")
    assert "case_closed_at" in res["leakage_suspects"]


def test_audit_no_false_alarms_on_honest_features():
    # Less than 20 rows -> no leakage suspects computed
    df_small = pd.DataFrame({
        "id": [f"ID_{i}" for i in range(15)],
        "target": [0] * 7 + [1] * 8,
        "timestamp": ["2026-09-01 10:00"] * 15,
        "perfect_small": [0] * 7 + [1] * 8
    })
    res_small = audit(df_small, target="target", id_col="id", time_col="timestamp")
    assert res_small["leakage_suspects"] == []

    # Single class present -> no suspects
    df_one_class = pd.DataFrame({
        "id": [f"ID_{i}" for i in range(25)],
        "target": [0] * 25,
        "timestamp": ["2026-09-01 10:00"] * 25,
        "feat": list(range(25))
    })
    res_one = audit(df_one_class, target="target", id_col="id", time_col="timestamp")
    assert res_one["leakage_suspects"] == []

    # Single distinct value column (currency EGP) -> never a suspect
    df_currency = pd.DataFrame({
        "id": [f"ID_{i}" for i in range(30)],
        "target": [0] * 15 + [1] * 15,
        "timestamp": ["2026-09-01 10:00"] * 30,
        "currency": ["EGP"] * 30
    })
    res_curr = audit(df_currency, target="target", id_col="id", time_col="timestamp")
    assert "currency" not in res_curr["leakage_suspects"]


def test_confusion_ties_at_threshold():
    y = [0, 1, 0, 1]
    scores = [0.2, 0.5, 0.5, 0.8]
    # score == threshold is flagged
    conf = confusion_at(y, scores, 0.5)
    # Flagged are scores >= 0.5: indices 1, 2, 3 (labels 1, 0, 1)
    # tp = 2 (indices 1, 3), fp = 1 (index 2), tn = 1 (index 0), fn = 0
    assert conf == {"tp": 2, "fp": 1, "tn": 1, "fn": 0}

    # Threshold above max flags nothing
    conf_max = confusion_at(y, scores, 0.9)
    assert conf_max == {"tp": 0, "fp": 0, "tn": 2, "fn": 2}


def test_precision_recall_zero_division():
    # No flags
    prec, rec = precision_recall_at([0, 1], [0.1, 0.2], 0.5)
    assert prec == 0.0
    assert rec == 0.0

    # No positives in y_true
    prec2, rec2 = precision_recall_at([0, 0], [0.6, 0.7], 0.5)
    assert prec2 == 0.0
    assert rec2 == 0.0

    # All positives
    prec3, rec3 = precision_recall_at([1, 1], [0.6, 0.7], 0.5)
    assert prec3 == 1.0
    assert rec3 == 1.0


def test_precision_at_k_ties_stable_and_k_gt_n():
    y = [1, 0, 1, 0]
    scores = [0.5, 0.5, 0.4, 0.3]
    # Indices 0 and 1 have tied scores 0.5. Stable sort keeps index 0 before index 1.
    # At k=1, top index is 0 (y=1) -> prec = 1/1 = 1.0
    assert precision_at_k(y, scores, 1) == 1.0
    # At k=2, top indices are 0 and 1 -> prec = 1/2 = 0.5
    assert precision_at_k(y, scores, 2) == 0.5
    # k > n -> uses n (4)
    assert precision_at_k(y, scores, 10) == 0.5

    with pytest.raises(ValueError):
        precision_at_k(y, scores, 0)
    with pytest.raises(ValueError):
        precision_at_k(y, scores, -2)


def test_best_threshold_matches_brute_force():
    y = [0, 1, 0, 1, 0, 1]
    scores = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    cost_fp = 10.0
    cost_fn = 50.0

    # Brute force search
    candidates = set(scores) | {math.inf}
    costs = {t: expected_cost(y, scores, t, cost_fp=cost_fp, cost_fn=cost_fn) for t in candidates}
    min_c = min(costs.values())
    tied_threshs = [t for t, c in costs.items() if c == min_c]
    expected_best = max(tied_threshs)

    actual_best = best_threshold(y, scores, cost_fp=cost_fp, cost_fn=cost_fn)
    assert actual_best == expected_best

    # Explicit tie test between two thresholds
    y_tie = [0, 0]
    s_tie = [0.3, 0.7]
    # If cost_fp = 1, cost_fn = 1:
    # thresh inf: fp=0, fn=0 -> cost=0
    # thresh 0.8: fp=0, fn=0 -> cost=0
    # Highest threshold candidate is inf!
    assert best_threshold(y_tie, s_tie, cost_fp=1.0, cost_fn=1.0) == math.inf


def test_threshold_for_precision_lowest_valid_or_none():
    y = [0, 0, 1, 1]
    scores = [0.2, 0.4, 0.6, 0.8]
    # At 0.2: tp=2, fp=2 -> prec = 0.5
    # At 0.4: tp=2, fp=1 -> prec = 2/3 = 0.667
    # At 0.6: tp=2, fp=0 -> prec = 1.0
    # At 0.8: tp=1, fp=0 -> prec = 1.0
    # For min_precision = 0.8, lowest valid threshold is 0.6
    assert threshold_for_precision(y, scores, 0.8) == 0.6

    # Unreachable precision
    assert threshold_for_precision([0, 0], [0.1, 0.2], 0.5) is None


def test_input_validation():
    with pytest.raises(ValueError):
        confusion_at([0, 1], [0.5], 0.5)  # mismatched length
    with pytest.raises(ValueError):
        confusion_at([], [], 0.5)         # empty
    with pytest.raises(ValueError):
        confusion_at([0, 2], [0.5, 0.6], 0.5)  # invalid label
    with pytest.raises(ValueError):
        confusion_at([0, 1], [0.5, np.nan], 0.5)  # NaN score


def test_framing_and_ai_usage_present():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    framing_path = os.path.join(base_dir, "FRAMING.md")
    ai_usage_path = os.path.join(base_dir, "AI_USAGE.md")

    assert os.path.exists(framing_path)
    assert os.path.exists(ai_usage_path)

    with open(framing_path, "r", encoding="utf-8") as f:
        framing_content = f.read()

    for heading in ["## Decision", "## Metric", "## Baseline", "## Classical ML or LLM"]:
        assert heading in framing_content

    with open(ai_usage_path, "r", encoding="utf-8") as f:
        ai_usage_content = f.read()
    assert len(ai_usage_content.strip()) > 100
