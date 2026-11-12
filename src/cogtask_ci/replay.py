"""Validate and replay frozen, subject-level task prediction bundles.

The module is intentionally data agnostic. Bundles contain only task IDs,
subject IDs, labels, token counts, and already-fitted prediction scores.
"""
from itertools import combinations
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np
from scipy.stats import rankdata


_FOLD_FIELDS = {
    "repeat",
    "outer_fold",
    "inner_subjects",
    "inner_labels",
    "inner_scores",
    "train_token_counts",
    "test_subjects",
    "test_labels",
    "test_scores",
}


def _integer(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _subject_ids(value, field):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} contains duplicate subject IDs")
    return list(value)


def _numeric_array(value, field):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a rectangular numeric array") from exc
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{field} must contain numeric values")
    return array


def _labels(value, field, expected):
    array = _numeric_array(value, field)
    if array.ndim != 1 or array.shape[0] != expected:
        raise ValueError(f"{field} must have shape ({expected},)")
    if not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{field} must contain only binary labels 0 and 1")
    if set(array.tolist()) != {0, 1}:
        raise ValueError(f"{field} must contain both label classes")
    return array.astype(np.int8, copy=True)


def _matrix(value, field, rows, columns, *, probabilities=False, counts=False):
    array = _numeric_array(value, field)
    if array.ndim != 2 or array.shape != (rows, columns):
        raise ValueError(f"{field} must have shape ({rows}, {columns})")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must contain only finite values")
    if probabilities and ((array < 0) | (array > 1)).any():
        raise ValueError(f"{field} must contain probabilities in [0, 1]")
    if counts and ((array < 0) | (array != np.floor(array))).any():
        raise ValueError(f"{field} must contain non-negative integer token counts")
    dtype = np.int64 if counts else np.float64
    return array.astype(dtype, copy=True)


def validate_bundle(bundle):
    """Validate a schema-version-1 replay bundle and return copied arrays.

    Array rows are positional: inner rows align with ``inner_subjects`` and
    outer-test rows align with ``test_subjects``. Columns align with ``tasks``.
    """
    if not isinstance(bundle, Mapping):
        raise ValueError("bundle must be a mapping")
    if type(bundle.get("schema_version")) is not int or bundle["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    tasks = bundle.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty list")
    if any(not isinstance(task, str) or not task.strip() for task in tasks):
        raise ValueError("tasks must contain non-empty strings")
    if len(set(tasks)) != len(tasks):
        raise ValueError("tasks contains duplicate task IDs")

    budget = None
    if "budget" in bundle:
        supplied_budget = bundle["budget"]
        budget_fields = {"mode", "total_tokens", "per_task_tokens", "fixed_k"}
        if not isinstance(supplied_budget, Mapping):
            raise ValueError("budget must be a mapping")
        if set(supplied_budget) != budget_fields:
            raise ValueError("budget must contain exactly mode,total_tokens,per_task_tokens,fixed_k")
        if supplied_budget["mode"] != "total":
            raise ValueError("budget.mode must be 'total'")
        total_tokens = _integer(supplied_budget["total_tokens"], "budget.total_tokens", minimum=1)
        per_task_tokens = _integer(supplied_budget["per_task_tokens"], "budget.per_task_tokens", minimum=1)
        fixed_k = _integer(supplied_budget["fixed_k"], "budget.fixed_k", minimum=1)
        if fixed_k > len(tasks):
            raise ValueError("budget.fixed_k cannot exceed the number of tasks")
        if total_tokens % fixed_k != 0 or per_task_tokens != total_tokens // fixed_k:
            raise ValueError("budget.per_task_tokens must equal total_tokens / fixed_k exactly")
        budget = {
            "mode": "total",
            "total_tokens": total_tokens,
            "per_task_tokens": per_task_tokens,
            "fixed_k": fixed_k,
        }
    folds = bundle.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("folds must be a non-empty list")

    normalized = {"schema_version": 1, "tasks": list(tasks), "folds": []}
    if budget is not None:
        normalized["budget"] = budget
    by_repeat = {}
    fold_keys = set()
    label_by_subject = {}
    task_count = len(tasks)

    for fold_index, fold in enumerate(folds):
        if not isinstance(fold, Mapping):
            raise ValueError(f"folds[{fold_index}] must be a mapping")
        missing = _FOLD_FIELDS - set(fold)
        if missing:
            raise ValueError(f"folds[{fold_index}] is missing fields: {', '.join(sorted(missing))}")
        repeat = _integer(fold["repeat"], f"folds[{fold_index}].repeat")
        outer_fold = _integer(fold["outer_fold"], f"folds[{fold_index}].outer_fold")
        key = (repeat, outer_fold)
        if key in fold_keys:
            raise ValueError("duplicate (repeat, outer_fold) fold")
        fold_keys.add(key)

        inner_subjects = _subject_ids(fold["inner_subjects"], f"folds[{fold_index}].inner_subjects")
        test_subjects = _subject_ids(fold["test_subjects"], f"folds[{fold_index}].test_subjects")
        if set(inner_subjects) & set(test_subjects):
            raise ValueError(f"fold {key} has inner/test subject overlap")
        inner_labels = _labels(fold["inner_labels"], f"folds[{fold_index}].inner_labels", len(inner_subjects))
        test_labels = _labels(fold["test_labels"], f"folds[{fold_index}].test_labels", len(test_subjects))
        inner_scores = _matrix(fold["inner_scores"], f"folds[{fold_index}].inner_scores",
                               len(inner_subjects), task_count, probabilities=True)
        test_scores = _matrix(fold["test_scores"], f"folds[{fold_index}].test_scores",
                              len(test_subjects), task_count, probabilities=True)
        token_counts = _matrix(fold["train_token_counts"], f"folds[{fold_index}].train_token_counts",
                               len(inner_subjects), task_count, counts=True)

        for subjects, labels in ((inner_subjects, inner_labels), (test_subjects, test_labels)):
            for subject, label in zip(subjects, labels.tolist()):
                old_label = label_by_subject.setdefault(subject, label)
                if old_label != label:
                    raise ValueError(f"subject {subject!r} has inconsistent labels across folds")

        clean = {
            "repeat": repeat,
            "outer_fold": outer_fold,
            "inner_subjects": inner_subjects,
            "inner_labels": inner_labels,
            "inner_scores": inner_scores,
            "train_token_counts": token_counts,
            "test_subjects": test_subjects,
            "test_labels": test_labels,
            "test_scores": test_scores,
        }
        normalized["folds"].append(clean)
        by_repeat.setdefault(repeat, []).append(clean)

    repeats = sorted(by_repeat)
    if repeats != list(range(len(repeats))):
        raise ValueError("repeat IDs must be contiguous integers starting at 0")

    repeat_subject_sets = []
    expected_folds = None
    for repeat in repeats:
        repeat_folds = sorted(by_repeat[repeat], key=lambda item: item["outer_fold"])
        fold_ids = [fold["outer_fold"] for fold in repeat_folds]
        if fold_ids != list(range(len(fold_ids))):
            raise ValueError(f"outer_fold IDs for repeat {repeat} must be contiguous integers starting at 0")
        if expected_folds is None:
            expected_folds = len(fold_ids)
        elif len(fold_ids) != expected_folds:
            raise ValueError("all repeats must have the same number of outer folds")
        outer_subjects = [subject for fold in repeat_folds for subject in fold["test_subjects"]]
        if len(outer_subjects) != len(set(outer_subjects)):
            raise ValueError(f"outer test coverage for repeat {repeat} contains a duplicate subject")
        repeat_subject_sets.append(set(outer_subjects))
    if any(subjects != repeat_subject_sets[0] for subjects in repeat_subject_sets[1:]):
        raise ValueError("all repeats must have identical outer subject sets")

    normalized["folds"].sort(key=lambda item: (item["repeat"], item["outer_fold"]))
    return normalized


def load_bundle(source):
    """Read and validate a replay bundle from a JSON path."""
    if isinstance(source, Mapping):
        return validate_bundle(source)
    try:
        with Path(source).open("r", encoding="utf-8") as handle:
            bundle = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read replay bundle: {exc}") from exc
    return validate_bundle(bundle)


def auc_rank(labels, scores):
    """AUROC from average ranks, with identical rank patterns sharing a result."""
    y = np.asarray(labels)
    values = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or values.ndim != 1 or y.shape != values.shape or values.size == 0:
        raise ValueError("labels and scores must be non-empty, equal-length vectors")
    if not np.isfinite(values).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("AUROC requires finite scores and binary labels")
    positive = y == 1
    negative = y == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUROC requires both label classes")
    ranks = rankdata(values, method="average")
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def _task_columns(tasks):
    return {task: index for index, task in enumerate(tasks)}


def _subset_key(task_ids):
    return tuple(sorted(task_ids))


def pair_key(pair):
    """Stable public key used for each all-pairs prediction array."""
    return "pair::" + json.dumps(list(_subset_key(pair)), ensure_ascii=False, separators=(",", ":"))


def pair_reference_key(pair):
    """Stable key for a pair's train-selected constituent single reference."""
    return "pair_best_single::" + json.dumps(list(_subset_key(pair)), ensure_ascii=False, separators=(",", ":"))


def tolerance_key(tolerance):
    return f"e3_tolerance::{tolerance:.2f}"


def _median_token_total(fold, indices):
    totals = fold["train_token_counts"][:, indices].sum(axis=1)
    return float(np.median(totals))


def _fold_selection(fold, tasks, tolerances):
    columns = _task_columns(tasks)
    lexical_tasks = sorted(tasks)
    candidates = []
    single_aucs = {}
    for task in lexical_tasks:
        index = columns[task]
        single_aucs[task] = auc_rank(fold["inner_labels"], fold["inner_scores"][:, index])

    for size in range(1, len(lexical_tasks) + 1):
        for subset in combinations(lexical_tasks, size):
            indices = [columns[task] for task in subset]
            combined = fold["inner_scores"][:, indices].mean(axis=1)
            auc = auc_rank(fold["inner_labels"], combined)
            token_median = _median_token_total(fold, indices)
            candidates.append({
                "tasks": tuple(subset),
                "indices": tuple(indices),
                "k": size,
                "inner_auc": float(auc),
                "median_training_tokens": token_median,
            })

    # Global score ties prefer fewer tasks, then lower training-token burden,
    # then lexical IDs. Lower token burden is consistently the secondary rule.
    global_best = min(candidates, key=lambda item: (
        -item["inner_auc"], item["k"], item["median_training_tokens"], item["tasks"]))
    global_single_auc = max(single_aucs.values())
    global_single = min(
        (task for task, auc in single_aucs.items() if auc == global_single_auc),
        key=lambda task: (float(np.median(fold["train_token_counts"][:, columns[task]])), task),
    )

    pairs = []
    for pair in combinations(lexical_tasks, 2):
        candidate = next(item for item in candidates if item["tasks"] == pair)
        best_single_auc = max(single_aucs[task] for task in pair)
        best_single = min(
            (task for task in pair if single_aucs[task] == best_single_auc),
            key=lambda task: (float(np.median(fold["train_token_counts"][:, columns[task]])), task),
        )
        pairs.append({
            "tasks": pair,
            "inner_auc": candidate["inner_auc"],
            "best_single": best_single,
            "best_single_inner_auc": float(best_single_auc),
            "gain_over_best_constituent": float(candidate["inner_auc"] - best_single_auc),
        })
    pair_candidates = {item["tasks"]: next(candidate for candidate in candidates
                                           if candidate["tasks"] == item["tasks"])
                       for item in pairs}
    max_pair_auc = min(pairs, key=lambda item: (
        -item["inner_auc"], pair_candidates[item["tasks"]]["median_training_tokens"], item["tasks"]))
    max_pair_gain = min(pairs, key=lambda item: (
        -item["gain_over_best_constituent"], -item["inner_auc"],
        pair_candidates[item["tasks"]]["median_training_tokens"], item["tasks"]))

    tolerance_selections = {}
    for tolerance in tolerances:
        eligible = [item for item in candidates if item["inner_auc"] >= global_best["inner_auc"] - tolerance]
        selected_k = min(item["k"] for item in eligible)
        within_k = [item for item in eligible if item["k"] == selected_k]
        tolerance_selections[tolerance] = min(
            within_k,
            key=lambda item: (-item["inner_auc"], item["median_training_tokens"], item["tasks"]),
        )

    return {
        "single_aucs": single_aucs,
        "global_best_single": global_single,
        "global_best_single_inner_auc": float(global_single_auc),
        "global_best_subset": global_best,
        "pairs": pairs,
        "max_pair_auc": max_pair_auc,
        "max_pair_gain": max_pair_gain,
        "tolerance_selections": tolerance_selections,
    }


def apply_bundle(bundle, fixed_task, random_draws=1000, seed=20261112):
    """Replay task, pair, random-pair, and tolerance-based policies.

    The returned prediction mapping is ``analysis_name -> repeat ->
    subject_id -> score``. All policies reuse the exact outer-test subjects
    and labels in the input bundle. No model fitting or corpus access occurs.
    """
    clean = validate_bundle(bundle)
    if clean.get("budget", {}).get("mode") == "total":
        raise ValueError(
            "Generic apply_bundle cannot replay fixed-total-budget fit bundles: "
            "each fixed_k value is a distinct fitted input, so E2/E3 require a k-specific analysis."
        )
    tasks = clean["tasks"]
    if not isinstance(fixed_task, str) or fixed_task not in tasks:
        raise ValueError("fixed_task must match one of the bundle task IDs")
    _integer(random_draws, "random_draws")
    _integer(seed, "seed")
    tolerances = (0.00, 0.01, 0.02)
    columns = _task_columns(tasks)
    pairs = list(combinations(sorted(tasks), 2))
    if len(pairs) == 0:
        raise ValueError("at least two tasks are required for pair policies")

    predictions = {}
    selections = []
    references = {
        "fixed_task": fixed_task,
        "global_train_selected_best_single": {},
        "subset_train_selected_best_single": {},
        "global_best_subset": {},
    }
    labels = {}

    def put(name, repeat, subject_ids, scores):
        by_repeat = predictions.setdefault(name, {}).setdefault(repeat, {})
        for subject, score in zip(subject_ids, scores.tolist()):
            if subject in by_repeat:
                raise ValueError(f"policy {name!r} produced duplicate prediction for subject {subject!r}")
            by_repeat[subject] = float(score)

    for task in tasks:
        predictions[f"task::{task}"] = {}
    predictions["fixed_task"] = {}
    for pair in pairs:
        predictions[pair_key(pair)] = {}
        predictions[pair_reference_key(pair)] = {}
        references["subset_train_selected_best_single"][pair_key(pair)] = {}
    for name in (
        "global_best_single",
        "e2_max_auc_pair",
        "e2_max_gain_pair",
        "global_best_subset",
        *(tolerance_key(value) for value in tolerances),
    ):
        predictions[name] = {}
    random_names = [f"random_pair::{index:04d}" for index in range(random_draws)]
    for name in random_names:
        predictions[name] = {}

    if seed + random_draws >= 2**128:
        raise ValueError("seed plus random_draws must be less than 2**128")
    random_choices = np.empty((random_draws, len(clean["folds"])), dtype=np.int64)
    for draw in range(random_draws):
        draw_rng = np.random.default_rng(seed + draw)
        random_choices[draw] = draw_rng.integers(0, len(pairs), size=len(clean["folds"]))

    for fold_position, fold in enumerate(clean["folds"]):
        repeat = fold["repeat"]
        outer_fold = fold["outer_fold"]
        selection = _fold_selection(fold, tasks, tolerances)
        test_subjects = fold["test_subjects"]
        test_scores = fold["test_scores"]
        test_labels = fold["test_labels"]
        for subject, label in zip(test_subjects, test_labels.tolist()):
            by_repeat = labels.setdefault(repeat, {})
            if subject in by_repeat:
                raise ValueError(f"duplicate outer test subject {subject!r} in repeat {repeat}")
            by_repeat[subject] = int(label)

        for task in tasks:
            put(f"task::{task}", repeat, test_subjects, test_scores[:, columns[task]])
        put("fixed_task", repeat, test_subjects, test_scores[:, columns[fixed_task]])
        for pair in pairs:
            indices = [columns[task] for task in pair]
            put(pair_key(pair), repeat, test_subjects, test_scores[:, indices].mean(axis=1))
            best_single = min(
                pair,
                key=lambda task: (
                    -selection["single_aucs"][task],
                    float(np.median(fold["train_token_counts"][:, columns[task]])),
                    task,
                ),
            )
            put(pair_reference_key(pair), repeat, test_subjects, test_scores[:, columns[best_single]])
            references["subset_train_selected_best_single"][pair_key(pair)][f"{repeat}:{outer_fold}"] = best_single

        put("global_best_single", repeat, test_subjects,
            test_scores[:, columns[selection["global_best_single"]]])
        for policy_name, item in (("e2_max_auc_pair", selection["max_pair_auc"]),
                                  ("e2_max_gain_pair", selection["max_pair_gain"])):
            indices = [columns[task] for task in item["tasks"]]
            put(policy_name, repeat, test_subjects, test_scores[:, indices].mean(axis=1))

        global_best = selection["global_best_subset"]
        put("global_best_subset", repeat, test_subjects,
            test_scores[:, list(global_best["indices"])].mean(axis=1))
        for tolerance, item in selection["tolerance_selections"].items():
            put(tolerance_key(tolerance), repeat, test_subjects,
                test_scores[:, list(item["indices"])].mean(axis=1))

        for draw, name in enumerate(random_names):
            random_pair = pairs[int(random_choices[draw, fold_position])]
            indices = [columns[task] for task in random_pair]
            put(name, repeat, test_subjects, test_scores[:, indices].mean(axis=1))

        fold_ref = f"{repeat}:{outer_fold}"
        references["global_train_selected_best_single"][fold_ref] = selection["global_best_single"]
        references["global_best_subset"][fold_ref] = list(global_best["tasks"])
        pair_metrics = {}
        for item in selection["pairs"]:
            pkey = pair_key(item["tasks"])
            pair_metrics[pkey] = {
                "inner_auc": item["inner_auc"],
                "best_single": item["best_single"],
                "best_single_inner_auc": item["best_single_inner_auc"],
                "gain_over_best_constituent": item["gain_over_best_constituent"],
            }
        selections.append({
            "repeat": repeat,
            "outer_fold": outer_fold,
            "global_best_single": selection["global_best_single"],
            "global_best_single_inner_auc": selection["global_best_single_inner_auc"],
            "global_best_subset": list(global_best["tasks"]),
            "global_best_subset_inner_auc": global_best["inner_auc"],
            "global_best_subset_median_training_tokens": global_best["median_training_tokens"],
            "e2_max_auc_pair": list(selection["max_pair_auc"]["tasks"]),
            "e2_max_auc_pair_inner_auc": selection["max_pair_auc"]["inner_auc"],
            "e2_max_gain_pair": list(selection["max_pair_gain"]["tasks"]),
            "e2_max_gain_pair_inner_auc": selection["max_pair_gain"]["inner_auc"],
            "e2_max_gain_over_best_constituent": selection["max_pair_gain"]["gain_over_best_constituent"],
            "pairs": pair_metrics,
            "e3_tolerance_selections": {
                f"{tolerance:.2f}": {
                    "tasks": list(item["tasks"]),
                    "inner_auc": item["inner_auc"],
                    "median_training_tokens": item["median_training_tokens"],
                }
                for tolerance, item in selection["tolerance_selections"].items()
            },
            "random_pairs": {
                str(draw): list(pairs[int(random_choices[draw, fold_position])])
                for draw in range(random_draws)
            },
        })

    return {
        "tasks": list(tasks),
        "fixed_task": fixed_task,
        "predictions": predictions,
        "labels": labels,
        "selections": selections,
        "references": references,
        "random_draws": random_draws,
        "seed": seed,
    }
