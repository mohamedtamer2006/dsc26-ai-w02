# AI usage

## Tools I used, and for which functions
I used Antigravity coding assistant (Gemini 3.8 Flash) and the Python interactive interpreter. I used the assistant to help draft edge-case handling in `rawaj/audit.py` (specifically handling whitespace strings counting as missing, unparseable vs blank timestamps, and Mann-Whitney U for leakage checks) and for initial candidate generation in `rawaj/metrics.py`.

## One place where the AI was wrong, incomplete, or not what the spec said
The standard snippet proposed by the AI for `best_threshold` used `np.unique` in ascending order and `np.argmin` to select the lowest cost. When two thresholds yielded the identical minimum expected cost, `np.argmin` kept the lower threshold. However, the DSC specification explicitly demands the highest threshold on cost ties (fewer false alarms for the same cost). I adapted the candidate search by sorting candidates descending (including `math.inf`) so that ties strictly preserve the higher threshold cutoff. Furthermore, I ensured that `threshold_for_precision` searches every distinct score because precision is non-monotonic across thresholds.

## The function I wrote without AI, and why
I implemented `precision_at_k` completely by hand without AI assistance. The specification requires strict stability where ties in scores preserve their original input order, as well as distinct boundary behaviors ($k \le 0$ raising a ValueError and $k > n$ clamping to $n$). Implementing it manually using Python's native stable Timsort (`sorted(range(n), key=lambda i: s_arr[i], reverse=True)[:k_eval]`) ensured zero risk of unexpected vectorization tie-breaking quirks, and helped me understand the exact mechanics of rank-based decision metrics under operational analyst capacity constraints.

