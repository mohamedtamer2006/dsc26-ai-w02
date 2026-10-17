# AI usage

## Tools used
Antigravity coding assistant (Gemini 3.8 Flash), Python interactive interpreter.

## What I asked
- "Help structure the `audit` function to handle edge cases like whitespace strings counting as missing and computing Mann-Whitney U for leakage checks."
- "Provide a starting implementation for `best_threshold` that searches across candidate cutoffs."

## What it got wrong or I changed
The standard snippet proposed for `best_threshold` used `np.unique` in ascending order and `np.argmin` to select the lowest cost. When two thresholds yielded the identical minimum expected cost, `np.argmin` kept the lower threshold. However, the DSC specification explicitly demands the highest threshold on cost ties (fewer false alarms for the same cost). I adapted the candidate search by sorting candidates descending (including `math.inf`) so that ties strictly preserve the higher threshold cutoff. Furthermore, I ensured that `threshold_for_precision` searches every distinct score because precision is non-monotonic across thresholds.

## What I checked myself and how
- Verified edge cases in `audit.py` using synthetic test scenarios: ensured `bad_timestamps` does not flag missing values or empty strings, and correctly returns `[None, None]` when all timestamps are invalid or absent.
- Verified stable sorting in `precision_at_k`: tested tied scores to guarantee original input index preservation.
- Checked ROC AUC edge cases: confirmed that when a column has only a single distinct value across all rows (including missingness as a value), it is protected from false alarm leakage detection.

## Which function I wrote without AI, and why
I implemented `precision_at_k` completely by hand without AI assistance. The specification requires strict stability where ties in scores preserve their original input order, as well as distinct boundary behaviors ($k \le 0$ raising a ValueError and $k > n$ clamping to $n$). Implementing it manually using Python's native stable Timsort (`sorted(range(n), key=lambda i: s_arr[i], reverse=True)[:k_eval]`) ensured zero risk of unexpected vectorization tie-breaking quirks.

## What I now understand that I didn't before
I gained a much deeper understanding of target leakage via missingness indicators: a column that is only populated when an event has already occurred (such as post-investigation closure dates) can exhibit near-perfect AUC simply by its pattern of nulls, even if the non-null values themselves appear harmless. I also learned how to rigorously formulate cost-sensitive threshold selection rather than defaulting to arbitrary cutoffs like 0.5.
