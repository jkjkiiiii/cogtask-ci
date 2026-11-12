"""Fast class-stratified subject bootstrap for paired prediction scores."""
import numpy as np
from scipy.stats import rankdata


def _labels(labels):
    y = np.asarray(labels)
    if y.ndim != 1 or y.size == 0 or y.dtype.kind not in "iuf":
        raise ValueError("labels must be a non-empty numeric vector")
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("labels must contain only 0 and 1")
    y = y.astype(np.int8, copy=False)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("labels must contain both classes")
    return y


def _score_matrix(scores, n_subjects, field):
    try:
        values = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a numeric score vector or matrix") from exc
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != n_subjects:
        raise ValueError(f"{field} must have shape (n_repeats, {n_subjects}) or ({n_subjects},)")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError(f"{field} must contain finite scores for at least one repeat")
    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"{field} must contain probabilities in [0, 1]")
    return values


def _rank_auc(labels, scores):
    positive = labels == 1
    negative = labels == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUROC requires both label classes")
    ranks = rankdata(scores, method="average")
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0) /
                 (n_positive * n_negative))


def _weighted_auc_batch(scores, positive_indices, negative_indices,
                        positive_weights, negative_weights):
    """Compute weighted Mann-Whitney AUROC for each bootstrap row."""
    positive_scores = scores[positive_indices]
    negative_scores = scores[negative_indices]
    order = np.argsort(negative_scores, kind="mergesort")
    ordered_negative = negative_scores[order]
    lower_index = np.searchsorted(ordered_negative, positive_scores, side="left")
    upper_index = np.searchsorted(ordered_negative, positive_scores, side="right")

    cumulative = np.cumsum(negative_weights[:, order], axis=1, dtype=np.int64)
    lower = np.zeros((negative_weights.shape[0], positive_indices.size), dtype=np.float64)
    has_lower = lower_index > 0
    if has_lower.any():
        lower[:, has_lower] = cumulative[:, lower_index[has_lower] - 1]
    upper = np.zeros_like(lower)
    has_upper = upper_index > 0
    if has_upper.any():
        upper[:, has_upper] = cumulative[:, upper_index[has_upper] - 1]
    equal = upper - lower
    numerator = (positive_weights * (lower + 0.5 * equal)).sum(axis=1)
    denominator = positive_indices.size * negative_indices.size
    return numerator / denominator


def _bootstrap_counts(rng, class_size, draws):
    probabilities = np.full(class_size, 1.0 / class_size, dtype=np.float64)
    return rng.multinomial(class_size, probabilities, size=draws)


def _validate_bootstrap(n_boot, seed, confidence, batch_size):
    if type(n_boot) is not int or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


def _interval(values, confidence):
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def stratified_bootstrap_auc(labels, scores, n_boot=10_000, seed=20261112,
                             confidence=0.95, batch_size=1024):
    """Estimate mean repeated-measure AUROC and a stratified subject CI.

    ``scores`` may be one subject vector or a ``(repeats, subjects)`` matrix.
    Each bootstrap draw resamples subjects separately within each class. The
    same sampled subjects are reused across every row of a repeated-measure
    matrix.
    """
    _validate_bootstrap(n_boot, seed, confidence, batch_size)
    y = _labels(labels)
    score_matrix = _score_matrix(scores, len(y), "scores")
    positive_indices = np.flatnonzero(y == 1)
    negative_indices = np.flatnonzero(y == 0)
    point = float(np.mean([_rank_auc(y, row) for row in score_matrix]))
    bootstrap = np.empty(n_boot, dtype=np.float64)
    rng = np.random.default_rng(seed)

    start = 0
    while start < n_boot:
        size = min(batch_size, n_boot - start)
        positive_weights = _bootstrap_counts(rng, len(positive_indices), size)
        negative_weights = _bootstrap_counts(rng, len(negative_indices), size)
        batch_auc = np.zeros(size, dtype=np.float64)
        for row in score_matrix:
            batch_auc += _weighted_auc_batch(row, positive_indices, negative_indices,
                                             positive_weights, negative_weights)
        bootstrap[start:start + size] = batch_auc / score_matrix.shape[0]
        start += size

    ci_low, ci_high = _interval(bootstrap, confidence)
    return {
        "auroc": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": float(confidence),
        "n_boot": n_boot,
        "n_subjects": len(y),
        "n_repeats": int(score_matrix.shape[0]),
        "bootstrap": bootstrap,
    }


def paired_subject_bootstrap(labels, scores_a, scores_b, n_boot=10_000,
                             seed=20261112, confidence=0.95,
                             batch_size=1024):
    """Paired AUROC and Brier differences with a subject-level bootstrap.

    Both score inputs use the same subject order and must have the same number
    of repeats. The returned differences are ``A - B``. For repeated measures,
    one stratified subject resample is shared across every repeat and both
    strategies, preserving all within-subject dependence.
    """
    _validate_bootstrap(n_boot, seed, confidence, batch_size)
    y = _labels(labels)
    a = _score_matrix(scores_a, len(y), "scores_a")
    b = _score_matrix(scores_b, len(y), "scores_b")
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must have identical repeat/subject shapes")

    positive_indices = np.flatnonzero(y == 1)
    negative_indices = np.flatnonzero(y == 0)
    point_auc_a = float(np.mean([_rank_auc(y, row) for row in a]))
    point_auc_b = float(np.mean([_rank_auc(y, row) for row in b]))
    point_brier_a = float(np.mean((a - y[None, :]) ** 2))
    point_brier_b = float(np.mean((b - y[None, :]) ** 2))
    point_auc_delta = point_auc_a - point_auc_b
    point_brier_delta = point_brier_a - point_brier_b

    auc_delta = np.empty(n_boot, dtype=np.float64)
    brier_delta = np.empty(n_boot, dtype=np.float64)
    rng = np.random.default_rng(seed)
    denominator = len(y)
    brier_by_subject = ((a - y[None, :]) ** 2 - (b - y[None, :]) ** 2).mean(axis=0)

    start = 0
    while start < n_boot:
        size = min(batch_size, n_boot - start)
        positive_weights = _bootstrap_counts(rng, len(positive_indices), size)
        negative_weights = _bootstrap_counts(rng, len(negative_indices), size)
        batch_auc_delta = np.zeros(size, dtype=np.float64)
        for row_a, row_b in zip(a, b):
            batch_auc_delta += (
                _weighted_auc_batch(row_a, positive_indices, negative_indices,
                                    positive_weights, negative_weights)
                - _weighted_auc_batch(row_b, positive_indices, negative_indices,
                                      positive_weights, negative_weights)
            )
        auc_delta[start:start + size] = batch_auc_delta / a.shape[0]
        # The class-stratified draw always has the original class counts.
        sampled_brier_delta = (
            positive_weights @ brier_by_subject[positive_indices]
            + negative_weights @ brier_by_subject[negative_indices]
        ) / denominator
        brier_delta[start:start + size] = sampled_brier_delta
        start += size

    auc_ci_low, auc_ci_high = _interval(auc_delta, confidence)
    brier_ci_low, brier_ci_high = _interval(brier_delta, confidence)
    return {
        "auroc_a": point_auc_a,
        "auroc_b": point_auc_b,
        "auroc_difference": point_auc_delta,
        "auroc_difference_ci_low": auc_ci_low,
        "auroc_difference_ci_high": auc_ci_high,
        "brier_a": point_brier_a,
        "brier_b": point_brier_b,
        "brier_difference": point_brier_delta,
        "brier_difference_ci_low": brier_ci_low,
        "brier_difference_ci_high": brier_ci_high,
        "confidence": float(confidence),
        "n_boot": n_boot,
        "n_subjects": len(y),
        "n_repeats": int(a.shape[0]),
        "bootstrap_auroc_difference": auc_delta,
        "bootstrap_brier_difference": brier_delta,
    }


# Short alias for callers that already establish subject-level pairing.
paired_bootstrap = paired_subject_bootstrap
