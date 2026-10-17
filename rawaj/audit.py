"""Data audit module for Rawaj Pay transaction extract."""

from collections import Counter
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def _is_missing_val(x: Any) -> bool:
    """Check if a scalar value is missing (NaN/None or empty/whitespace string)."""
    if pd.isna(x):
        return True
    if isinstance(x, str):
        return len(x.strip()) == 0
    return False


def _is_missing_series(s: pd.Series) -> pd.Series:
    """Check if elements of a Series are missing according to specification."""
    isna = s.isna()

    def _check_empty_str(x):
        return isinstance(x, str) and len(x.strip()) == 0

    is_blank = s.map(_check_empty_str)
    return isna | is_blank


def _compute_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """Compute ROC AUC score using Mann-Whitney U statistic."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)

    pos = s[y == 1]
    neg = s[y == 0]
    n_pos = len(pos)
    n_neg = len(neg)

    if n_pos == 0 or n_neg == 0:
        return None

    # Check for NaN in scores
    if np.any(np.isnan(pos)) or np.any(np.isnan(neg)):
        return None

    # For medium arrays, pairwise broadcasting is fast and exact
    if n_pos * n_neg <= 20_000_000:
        greater = np.sum(pos[:, None] > neg[None, :])
        equal = np.sum(pos[:, None] == neg[None, :])
        return float((greater + 0.5 * equal) / (n_pos * n_neg))
    else:
        # Rank-based calculation for very large arrays
        from scipy.stats import rankdata

        all_scores = np.concatenate([pos, neg])
        all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        ranks = rankdata(all_scores)
        pos_rank_sum = np.sum(ranks[all_labels == 1])
        u_stat = pos_rank_sum - (n_pos * (n_pos + 1)) / 2.0
        return float(u_stat / (n_pos * n_neg))


def audit(
    df: pd.DataFrame,
    *,
    target: str,
    id_col: str,
    time_col: str,
) -> Dict[str, Any]:
    """Audit a tabular dataset extract according to Rawaj specification.

    Raises KeyError if any of the three named columns is missing.
    """
    for col in (target, id_col, time_col):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' is missing from DataFrame.")

    n_rows = len(df)

    # 1. Missingness fraction per column
    missing: Dict[str, float] = {}
    missing_series_dict: Dict[str, pd.Series] = {}
    for col in df.columns:
        m_ser = _is_missing_series(df[col])
        missing_series_dict[col] = m_ser
        missing[col] = float(m_ser.mean()) if n_rows > 0 else 0.0

    # 2. Duplicate IDs (non-missing values appearing more than once, sorted)
    id_missing = missing_series_dict[id_col]
    non_missing_ids = df.loc[~id_missing, id_col].tolist()
    id_counts = Counter(non_missing_ids)
    duplicate_ids = sorted([val for val, count in id_counts.items() if count > 1])

    # 3. Duplicate rows (rows that exactly repeat an earlier row)
    duplicate_rows = int(df.duplicated(keep="first").sum())

    # 4. Constant columns (columns other than target with <= 1 distinct non-missing value)
    constant_columns: List[str] = []
    for col in df.columns:
        if col == target:
            continue
        m_ser = missing_series_dict[col]
        non_miss_vals = df.loc[~m_ser, col]
        try:
            n_distinct = non_miss_vals.nunique()
        except TypeError:
            # Handle unhashable objects if any
            n_distinct = len(set(map(str, non_miss_vals)))
        if n_distinct <= 1:
            constant_columns.append(col)
    constant_columns.sort()

    # 5. Unlabelled rows & positive rate
    target_missing = missing_series_dict[target]
    unlabelled_rows = int(target_missing.sum())

    labelled_targets = df.loc[~target_missing, target]
    if len(labelled_targets) == 0:
        positive_rate = 0.0
    else:

        def _is_positive(val):
            if isinstance(val, (bool, np.bool_)):
                return bool(val)
            if isinstance(val, (int, float, np.integer, np.floating)):
                return val == 1
            s = str(val).strip().lower()
            return s in ("1", "1.0", "true")

        pos_count = sum(1 for v in labelled_targets if _is_positive(v))
        positive_rate = float(pos_count / len(labelled_targets))

    # 6. Bad timestamps and time range
    time_missing = missing_series_dict[time_col]
    non_missing_times = df.loc[~time_missing, time_col]

    if len(non_missing_times) == 0:
        bad_timestamps = 0
        time_range = [None, None]
    else:
        parsed_times = pd.to_datetime(non_missing_times, errors="coerce")
        bad_mask = parsed_times.isna()
        bad_timestamps = int(bad_mask.sum())

        valid_parsed = parsed_times.dropna()
        if len(valid_parsed) == 0:
            time_range = [None, None]
        else:
            time_range = [
                valid_parsed.min().isoformat(),
                valid_parsed.max().isoformat(),
            ]

    # 7. Leakage suspects
    leakage_suspects: List[str] = []
    labelled_mask = ~target_missing
    labelled_df = df.loc[labelled_mask]

    # Rule: only compute on labelled rows when >= 20 non-missing labelled rows and both classes present
    if len(labelled_df) >= 20:

        def _to_bin(val):
            if isinstance(val, (bool, np.bool_)):
                return 1 if val else 0
            if isinstance(val, (int, float, np.integer, np.floating)):
                return 1 if val == 1 else 0
            s = str(val).strip().lower()
            return 1 if s in ("1", "1.0", "true") else 0

        y_labelled = np.array([_to_bin(v) for v in labelled_df[target]], dtype=int)
        both_classes = len(np.unique(y_labelled)) == 2

        if both_classes:
            for col in df.columns:
                if col in (target, id_col, time_col):
                    continue

                # "A column with a single distinct value is never a suspect: here missing counts as a value."
                # Check distinct values in df[col] where missing counts as a value
                def _val_with_missing(v):
                    return "__MISSING__" if _is_missing_val(v) else v

                try:
                    total_distinct = len(set(df[col].map(_val_with_missing)))
                except TypeError:
                    total_distinct = len(set(df[col].map(lambda v: "__MISSING__" if _is_missing_val(v) else str(v))))

                if total_distinct <= 1:
                    continue

                is_suspect = False

                # (b) Missingness indicator ROC AUC check
                # Missingness indicator: 1 = missing, 0 = present on labelled rows
                m_indicator = missing_series_dict[col].loc[labelled_mask].astype(int).to_numpy()
                auc_b = _compute_roc_auc(y_labelled, m_indicator)
                if auc_b is not None and (auc_b >= 0.98 or auc_b <= 0.02):
                    is_suspect = True

                # (a) Numeric/boolean ROC AUC check
                if not is_suspect:
                    col_labelled = labelled_df[col]
                    col_missing = missing_series_dict[col].loc[labelled_mask].to_numpy()
                    non_missing_mask = ~col_missing

                    col_non_missing = col_labelled.iloc[non_missing_mask]
                    y_sub = y_labelled[non_missing_mask]

                    if len(col_non_missing) >= 20 and len(np.unique(y_sub)) == 2:
                        is_numeric_or_bool = pd.api.types.is_numeric_dtype(
                            col_non_missing
                        ) or pd.api.types.is_bool_dtype(col_non_missing)
                        numeric_vals = None
                        if is_numeric_or_bool:
                            numeric_vals = col_non_missing.to_numpy(dtype=float)
                        else:
                            # Try converting object column to numeric
                            try:
                                conv = pd.to_numeric(col_non_missing, errors="raise")
                                numeric_vals = conv.to_numpy(dtype=float)
                                is_numeric_or_bool = True
                            except (ValueError, TypeError):
                                is_numeric_or_bool = False

                        if is_numeric_or_bool and numeric_vals is not None:
                            auc_a = _compute_roc_auc(y_sub, numeric_vals)
                            if auc_a is not None and (auc_a >= 0.98 or auc_a <= 0.02):
                                is_suspect = True

                if is_suspect:
                    leakage_suspects.append(col)

    leakage_suspects.sort()

    return {
        "n_rows": n_rows,
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": leakage_suspects,
    }
