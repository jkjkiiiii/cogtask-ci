"""Replay frozen task bundles and summarize a synthetic factorial across seeds.

The module does not fit models. It validates saved fit artifacts, replays
selection policies, and aggregates independent dataset seeds. Reported spread is
across generated datasets; it is not a participant confidence interval.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.stats import rankdata

from .replay import (
    apply_bundle,
    auc_rank,
    load_bundle,
    pair_key,
    pair_reference_key,
    tolerance_key,
)


DEFAULT_SEEDS = tuple(range(20261112, 20261142))
DEFAULT_EXPECTED_RUNS = 1080
DEFAULT_RANDOM_DRAWS = 1000
DEFAULT_RANDOM_SEED = 20261112
ARMS = ("text", "quantity")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    return _sha256_bytes(Path(path).read_bytes())


def _source_hash():
    source_files = {
        "factorial_summary.py": Path(__file__),
        "replay.py": Path(__file__).with_name("replay.py"),
    }
    hashes = {name: _sha256_file(path) for name, path in source_files.items()}
    return _sha256_bytes(_canonical(hashes).encode("utf-8")), hashes


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _append_failure(path, record):
    with Path(path).open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                allow_nan=False) + "\n")


def _metric_vector(labels_by_repeat, scores_by_repeat):
    repeats = sorted(labels_by_repeat)
    if not repeats or repeats != sorted(scores_by_repeat):
        raise ValueError("Replay policies have missing repeat rows")
    subject_order = None
    labels = None
    rows = []
    aucs = []
    for repeat in repeats:
        current_labels = labels_by_repeat[repeat]
        current_scores = scores_by_repeat[repeat]
        subjects = sorted(current_labels)
        if subjects != sorted(current_scores):
            raise ValueError("Replay score and label subjects do not match")
        y = np.asarray([int(current_labels[subject]) for subject in subjects], dtype=np.int8)
        values = np.asarray([float(current_scores[subject]) for subject in subjects], dtype=np.float64)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Replay scores must be finite probabilities")
        if subject_order is None:
            subject_order, labels = subjects, y
        elif subjects != subject_order or not np.array_equal(y, labels):
            raise ValueError("Subjects or labels differ across repeats")
        aucs.append(auc_rank(y, values))
        rows.append(values)
    matrix = np.stack(rows)
    return {"auroc": float(np.mean(aucs)),
            "brier": float(np.mean((matrix - labels[None, :]) ** 2)),
            "n_subjects": len(subject_order), "n_repeats": len(repeats)}, subject_order, labels


def _policy_metrics(replay, name):
    return _metric_vector(replay["labels"], replay["predictions"][name])


def _random_policy_metrics(replay, names):
    """Compute all random AUROCs together; avoid a metric-library call per draw."""
    repeats = sorted(replay["labels"])
    reference_subjects = None
    reference_labels = None
    matrices = []
    for repeat in repeats:
        label_map = replay["labels"][repeat]
        subjects = sorted(label_map)
        y = np.asarray([int(label_map[s]) for s in subjects], dtype=np.int8)
        if reference_subjects is None:
            reference_subjects, reference_labels = subjects, y
        elif subjects != reference_subjects or not np.array_equal(y, reference_labels):
            raise ValueError("Random replay policies do not share subjects and labels")
        draw_matrix = np.asarray([[float(replay["predictions"][name][repeat][s]) for s in subjects]
                                  for name in names], dtype=np.float64)
        if not np.isfinite(draw_matrix).all() or ((draw_matrix < 0) | (draw_matrix > 1)).any():
            raise ValueError("Random replay scores must be finite probabilities")
        matrices.append(draw_matrix)
    # (draw, repeat, subject); ties receive average ranks as in replay.auc_rank.
    scores = np.stack(matrices, axis=1)
    ranks = rankdata(scores, axis=2, method="average")
    positive = reference_labels == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if not n_positive or not n_negative:
        raise ValueError("AUROC requires both classes")
    rank_sums = ranks[:, :, positive].sum(axis=2)
    auc_by_repeat = (rank_sums - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)
    auc = auc_by_repeat.mean(axis=1)
    brier = ((scores - reference_labels[None, None, :]) ** 2).mean(axis=(1, 2))
    return auc, brier


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "q50": float(np.quantile(values, 0.50)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95))}


def _empirical_percentile(random_values, observed, higher_is_better=True):
    values = np.asarray(random_values, dtype=np.float64)
    if higher_is_better:
        count = np.count_nonzero(values < observed)
    else:
        count = np.count_nonzero(values > observed)
    ties = np.count_nonzero(values == observed)
    return float(100.0 * (count + 0.5 * ties) / len(values))


def _selected(counter, policy, task_ids):
    if isinstance(task_ids, str):
        task_ids = [task_ids]
    for task_id in task_ids:
        counter[(policy, task_id)] += 1


def _metrics_for_replay(replay, bundle, run, arm, bundle_sha, replay_sha,
                        fixed_task, random_draws, random_seed):
    tasks = replay["tasks"]
    if fixed_task is None:
        fixed_task = tasks[0]
    if fixed_task not in tasks:
        raise ValueError("fixed_task must name a task in every bundle")
    expected_random_names = [f"random_pair::{i:04d}" for i in range(random_draws)]
    available_random_names = [name for name in replay["predictions"] if name.startswith("random_pair::")]
    if sorted(available_random_names) != expected_random_names:
        raise ValueError("Replay random policy count or indexing is inconsistent")
    random_auc, random_brier = _random_policy_metrics(replay, expected_random_names)
    metrics = []

    def add(analysis, policy, reference=None, extra=None):
        value, _, _ = _policy_metrics(replay, policy)
        reference_value = _policy_metrics(replay, reference)[0] if reference else None
        row = {
            "analysis": analysis,
            "policy": policy,
            "reference": reference,
            "auroc": value["auroc"],
            "brier": value["brier"],
            "delta_auroc": value["auroc"] - reference_value["auroc"] if reference_value else None,
            "delta_brier": value["brier"] - reference_value["brier"] if reference_value else None,
            "reference_auroc": reference_value["auroc"] if reference_value else None,
            "reference_brier": reference_value["brier"] if reference_value else None,
            "n_subjects": value["n_subjects"],
            "n_repeats": value["n_repeats"],
        }
        if extra:
            row.update(extra)
        metrics.append(row)

    # E1: every pair and each of the three frozen references.
    for pair in combinations(sorted(tasks), 2):
        pair_policy = pair_key(pair)
        pair_text = json.dumps(list(pair), ensure_ascii=False, separators=(",", ":"))
        add("E1_pair_absolute", pair_policy, extra={"pair": pair_text})
        for ref_name, ref_policy in (
            ("global_best_single", "global_best_single"),
            ("pair_best_single", pair_reference_key(pair)),
            ("fixed_task", "fixed_task"),
        ):
            add("E1_pair_vs_reference", pair_policy, ref_policy,
                {"reference_name": ref_name, "pair": pair_text})

    # E2 policies, their gains over the global inner-selected single, and A-G.
    add("E2_strategy", "global_best_single")
    add("E2_strategy", "e2_max_auc_pair", "global_best_single")
    add("E2_strategy", "e2_max_gain_pair", "global_best_single")
    add("E2_A_minus_G", "e2_max_auc_pair", "e2_max_gain_pair")
    random_summary = {"auroc": _quantiles(random_auc), "brier": _quantiles(random_brier),
                      "draws": random_draws, "seed_start": random_seed,
                      "policy": "one uniform pair per outer fold; no score ensembling"}
    first_stats = _policy_metrics(replay, "global_best_single")[0]
    metrics.append({"analysis": "E2_random_distribution", "policy": "random_pair_distribution",
                    "reference": None, "auroc": None, "brier": None,
                    "delta_auroc": None, "delta_brier": None,
                    "random_auroc_mean": random_summary["auroc"]["mean"],
                    "random_auroc_sd": random_summary["auroc"]["sd"],
                    "random_auroc_q05": random_summary["auroc"]["q05"],
                    "random_auroc_q25": random_summary["auroc"]["q25"],
                    "random_auroc_q50": random_summary["auroc"]["q50"],
                    "random_auroc_q75": random_summary["auroc"]["q75"],
                    "random_auroc_q95": random_summary["auroc"]["q95"],
                    "random_brier_mean": random_summary["brier"]["mean"],
                    "random_brier_sd": random_summary["brier"]["sd"],
                    "random_brier_q05": random_summary["brier"]["q05"],
                    "random_brier_q25": random_summary["brier"]["q25"],
                    "random_brier_q50": random_summary["brier"]["q50"],
                    "random_brier_q75": random_summary["brier"]["q75"],
                    "random_brier_q95": random_summary["brier"]["q95"],
                    "n_subjects": first_stats["n_subjects"],
                    "n_repeats": first_stats["n_repeats"]})
    for policy in ("e2_max_auc_pair", "e2_max_gain_pair"):
        value = _policy_metrics(replay, policy)[0]
        metrics.append({"analysis": "E2_random_position", "policy": policy,
                        "reference": "random_pair_distribution", "auroc": value["auroc"],
                        "brier": value["brier"], "delta_auroc": None, "delta_brier": None,
                        "auroc_random_percentile": _empirical_percentile(random_auc, value["auroc"]),
                        "brier_random_percentile": _empirical_percentile(random_brier, value["brier"], False),
                        "n_subjects": value["n_subjects"], "n_repeats": value["n_repeats"]})

    # E3 reports score and resource burden against the global best single task.
    # Burden is the mean across outer folds of the within-training-fold median
    # pre-crop token count for the selected task set.
    task_columns = {task: index for index, task in enumerate(tasks)}
    fold_lookup = {(fold["repeat"], fold["outer_fold"]): fold for fold in bundle["folds"]}

    def e3_burden(selection, policy):
        if policy == "global_best_single":
            selected = [selection["global_best_single"]]
        elif policy == "global_best_subset":
            selected = selection["global_best_subset"]
        else:
            tolerance = policy.split("::", 1)[1]
            selected = selection["e3_tolerance_selections"][tolerance]["tasks"]
        fold = fold_lookup[(selection["repeat"], selection["outer_fold"])]
        indices = [task_columns[task] for task in selected]
        totals = fold["train_token_counts"][:, indices].sum(axis=1)
        return len(selected), float(np.median(totals))

    global_burdens = [e3_burden(row, "global_best_subset") for row in replay["selections"]]
    global_mean_tasks = float(np.mean([item[0] for item in global_burdens]))
    global_mean_tokens = float(np.mean([item[1] for item in global_burdens]))
    for policy in ("global_best_subset", *(tolerance_key(t) for t in (0.00, 0.01, 0.02))):
        add("E3_budget_policy", policy, "global_best_subset")
        burdens = [e3_burden(row, policy) for row in replay["selections"]]
        metrics[-1].update({
            "mean_selected_tasks": float(np.mean([item[0] for item in burdens])),
            "delta_mean_selected_tasks": float(np.mean([item[0] for item in burdens])) - global_mean_tasks,
            "mean_training_token_burden": float(np.mean([item[1] for item in burdens])),
            "delta_mean_training_token_burden": float(np.mean([item[1] for item in burdens])) - global_mean_tokens,
        })

    counts = Counter()
    opportunities = Counter()
    for selection in replay["selections"]:
        for field, policy in (("global_best_single", "global_best_single"),
                              ("global_best_subset", "global_best_subset"),
                              ("e2_max_auc_pair", "e2_max_auc_pair"),
                              ("e2_max_gain_pair", "e2_max_gain_pair")):
            _selected(counts, policy, selection[field])
            opportunities[policy] += 1
        for tol, selected in selection["e3_tolerance_selections"].items():
            policy = "e3_tolerance::" + tol
            _selected(counts, policy, selected["tasks"])
            opportunities[policy] += 1
        for pair in selection["random_pairs"].values():
            _selected(counts, "random_pair", pair)
            opportunities["random_pair"] += 1
    for pair in combinations(sorted(tasks), 2):
        policy = "e1_reference::" + pair_key(pair)
        fold_map = replay["references"]["subset_train_selected_best_single"][pair_key(pair)]
        for selected in fold_map.values():
            _selected(counts, policy, selected)
            opportunities[policy] += 1
    # An unselected task has zero frequency; it is not a missing dataset seed.
    for policy in opportunities:
        for task in tasks:
            counts.setdefault((policy, task), 0)
    selection_counts = [
        {"policy": policy, "task": task, "count": count,
         "selection_opportunities": opportunities[policy]}
        for (policy, task), count in sorted(counts.items())
    ]
    return {
        "schema_version": 1,
        "run_id": run["run_id"],
        "seed": run["seed"],
        "condition": run["condition"],
        "arm": arm,
        "fixed_task": fixed_task,
        "input_bundle_sha256": bundle_sha,
        "replay_code_sha256": replay_sha,
        "random_draws": random_draws,
        "random_seed": random_seed,
        "metrics": metrics,
        "random_distribution": random_summary,
        "task_selection_counts": selection_counts,
        "uncertainty": "Across-seed spread only; no per-dataset participant bootstrap or clinical CI.",
    }


def _read_cached_or_replay(run, arm, fit_root, output, replay_sha,
                           fixed_task, random_draws, random_seed):
    from .replay import load_bundle

    run_dir = fit_root / run["run_id"]
    fit = run["fit_result"]
    bundle_path = run_dir / (arm + "_bundle.json")
    bundle_sha = _sha256_file(bundle_path)
    if fit.get("bundle_sha256", {}).get(arm) != bundle_sha:
        raise ValueError("Bundle hash does not match its completed fit result")
    cache_path = output / "arm_replays" / (run["run_id"] + "__" + arm + ".json")
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        for key, value in {
            "run_id": run["run_id"], "arm": arm, "input_bundle_sha256": bundle_sha,
            "replay_code_sha256": replay_sha, "fixed_task": fixed_task or cached.get("fixed_task"),
            "random_draws": random_draws, "random_seed": random_seed,
        }.items():
            if cached.get(key) != value:
                raise ValueError("Cached arm replay provenance differs at " + key)
        return cached
    bundle = load_bundle(bundle_path)
    task = fixed_task or bundle["tasks"][0]
    replay = apply_bundle(bundle, fixed_task=task, random_draws=random_draws,
                          seed=random_seed)
    result = _metrics_for_replay(replay, bundle, run, arm, bundle_sha, replay_sha,
                                 task, random_draws, random_seed)
    _atomic_json(cache_path, result)
    return result


def _condition_key(condition):
    return _canonical(condition)


def _flatten_metrics(record):
    base = {"run_id": record["run_id"], "seed": record["seed"],
            "arm": record["arm"], "fixed_task": record["fixed_task"]}
    base.update(record["condition"])
    return [{**base, **metric} for metric in record["metrics"]]


def _aggregate(rows, expected_seeds):
    groups = {}
    condition_fields = sorted({key for row in rows
                              for key in row
                              if key not in {"run_id", "seed", "arm", "fixed_task", "analysis", "policy",
                                             "reference", "reference_name", "pair", "auroc", "brier",
                                             "delta_auroc", "delta_brier", "reference_auroc", "reference_brier",
                                             "n_subjects", "n_repeats", "random_auroc_mean", "random_auroc_sd",
                                             "random_auroc_q05", "random_auroc_q25", "random_auroc_q50",
                                             "random_auroc_q75", "random_auroc_q95", "random_brier_mean",
                                             "random_brier_sd", "random_brier_q05", "random_brier_q25",
                                             "random_brier_q50", "random_brier_q75", "random_brier_q95",
                                             "auroc_random_percentile", "brier_random_percentile",
                                             "mean_selected_tasks", "delta_mean_selected_tasks",
                                             "mean_training_token_burden",
                                             "delta_mean_training_token_burden"}})
    fixed_fields = ["arm", "fixed_task", "analysis", "policy", "reference", "reference_name", "pair"]
    excluded = {"run_id", "seed", *condition_fields, *fixed_fields}
    numeric = sorted({key for row in rows for key, value in row.items()
                      if key not in excluded and isinstance(value, (int, float)) and value is not None})
    identity_fields = condition_fields + fixed_fields
    for row in rows:
        key = tuple(row.get(field) for field in identity_fields)
        groups.setdefault(key, []).append(row)
    summary = []
    expected_seed_set = set(expected_seeds)
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        seen_seeds = [row["seed"] for row in group]
        unique_seeds = set(seen_seeds)
        duplicate_seed = len(seen_seeds) != len(unique_seeds)
        complete = (unique_seeds == expected_seed_set and not duplicate_seed and
                    all(bool(row.get("run_id")) for row in group))
        row = dict(zip(identity_fields, key))
        row["n_seeds"] = len(unique_seeds)
        row["status"] = "complete" if complete else "incomplete"
        for metric in numeric:
            values = [float(item[metric]) for item in group
                      if item.get(metric) is not None and math.isfinite(float(item[metric]))]
            if complete and len(values) == len(expected_seeds):
                quantile = _quantiles(values)
                row[metric + "_mean_across_seeds"] = quantile["mean"]
                row[metric + "_sd_across_seeds"] = quantile["sd"]
                row[metric + "_q05_across_seeds"] = quantile["q05"]
                row[metric + "_q50_across_seeds"] = quantile["q50"]
                row[metric + "_q95_across_seeds"] = quantile["q95"]
            else:
                for suffix in ("mean", "sd", "q05", "q50", "q95"):
                    row[metric + "_" + suffix + "_across_seeds"] = None
        summary.append(row)
    return summary, condition_fields


def summarize_factorial(fit_root, output, *, expected_seeds=DEFAULT_SEEDS,
                        expected_runs=DEFAULT_EXPECTED_RUNS,
                        random_draws=DEFAULT_RANDOM_DRAWS,
                        random_seed=DEFAULT_RANDOM_SEED,
                        fixed_task=None, arms=ARMS, expected_conditions=None):
    """Replay completed factorial fit bundles and write per-run and across-seed tables.

    Existing per-arm replay caches are reused only when both the fit bundle hash
    and the current replay source hash match. Mismatches fail instead of silently
    regenerating a different summary.
    """
    fit_root, output = Path(fit_root), Path(output)
    if not fit_root.is_dir():
        raise ValueError("fit_root must be an existing directory")
    if type(expected_runs) is not int or expected_runs < 1:
        raise ValueError("expected_runs must be a positive integer")
    if type(random_draws) is not int or random_draws < 1:
        raise ValueError("random_draws must be a positive integer")
    if type(random_seed) is not int or random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    expected_seeds = tuple(expected_seeds)
    if not expected_seeds or any(type(seed) is not int for seed in expected_seeds):
        raise ValueError("expected_seeds must contain integer seeds")
    if len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("expected_seeds cannot contain duplicates")
    if expected_conditions is None:
        from .factorial import factorial_conditions
        expected_conditions = factorial_conditions()
    expected_conditions = tuple(dict(condition) for condition in expected_conditions)
    if not expected_conditions or any(not condition for condition in expected_conditions):
        raise ValueError("expected_conditions must contain non-empty condition dictionaries")
    expected_condition_keys = {_condition_key(condition) for condition in expected_conditions}
    if len(expected_condition_keys) != len(expected_conditions):
        raise ValueError("expected_conditions cannot contain duplicates")
    if not arms or any(not isinstance(arm, str) or not arm for arm in arms):
        raise ValueError("arms must contain one or more nonempty names")
    replay_sha, replay_sources = _source_hash()
    output.mkdir(parents=True, exist_ok=True)
    failures_path = output / "failures.jsonl"
    runs = []
    failures = []
    common_provenance = None

    for run_dir in sorted(path for path in fit_root.iterdir() if path.is_dir()):
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            failure = {"phase": "fit_validation", "run_id": run_dir.name,
                       "error_type": "MissingFitResult", "error": "Run directory has no result.json",
                       "recorded_at_utc": _now()}
            failures.append(failure)
            _append_failure(failures_path, failure)
            continue
        try:
            fit_result = json.loads(result_path.read_text(encoding="utf-8"))
            if fit_result.get("status") != "complete":
                raise ValueError("Fit result status is not complete")
            run_id = fit_result.get("run_id", run_dir.name)
            if run_id != run_dir.name:
                raise ValueError("run_id does not match fit directory name")
            seed = fit_result.get("seed")
            if type(seed) is not int or seed not in set(expected_seeds):
                raise ValueError("Fit seed is not part of expected_seeds")
            condition = fit_result.get("condition")
            if not isinstance(condition, dict) or not condition:
                raise ValueError("Fit result is missing a condition dictionary")
            if _condition_key(condition) not in expected_condition_keys:
                raise ValueError("Fit condition is not part of expected_conditions")
            input_hash = fit_result.get("input_sha256")
            if input_hash is not None:
                input_path = run_dir / "input.jsonl"
                if not input_path.is_file() or _sha256_file(input_path) != input_hash:
                    raise ValueError("Synthetic input.jsonl is missing or differs from input_sha256")
            provenance = {key: fit_result.get(key) for key in (
                "code_file_sha256", "config_sha256", "protocol_sha256",
                "protocol_git_commit", "code_git_commit", "environment")
                          if key in fit_result}
            if common_provenance is None:
                common_provenance = provenance
            elif provenance != common_provenance:
                raise ValueError("Fit run provenance differs across datasets")
            run = {"run_id": run_id, "seed": seed, "condition": condition,
                   "fit_result": fit_result}
            runs.append(run)
        except Exception as error:
            failure = {"phase": "fit_validation", "run_id": run_dir.name,
                       "error_type": type(error).__name__, "error": str(error),
                       "recorded_at_utc": _now()}
            failures.append(failure)
            _append_failure(failures_path, failure)

    if len(runs) != expected_runs:
        failure = {"phase": "fit_validation", "error_type": "IncompleteFitSweep",
                   "error": f"Discovered {len(runs)} valid fit runs; expected {expected_runs}",
                   "recorded_at_utc": _now()}
        failures.append(failure)
        _append_failure(failures_path, failure)

    cache_records = []
    for run in runs:
        for arm in arms:
            try:
                record = _read_cached_or_replay(run, arm, fit_root, output,
                                                replay_sha, fixed_task,
                                                random_draws, random_seed)
                cache_records.append(record)
            except Exception as error:
                failure = {"phase": "replay", "run_id": run["run_id"], "arm": arm,
                           "error_type": type(error).__name__, "error": str(error),
                           "recorded_at_utc": _now()}
                failures.append(failure)
                _append_failure(failures_path, failure)

    flattened = []
    selection_rows = []
    for record in cache_records:
        flattened.extend(_flatten_metrics(record))
        for count in record["task_selection_counts"]:
            row = {"run_id": record["run_id"], "seed": record["seed"],
                   "arm": record["arm"], "fixed_task": record["fixed_task"],
                   **record["condition"], **count}
            selection_rows.append(row)

    # Add explicit placeholders for missing condition × seed × arm cells so
    # a condition absent for every seed cannot disappear from aggregation.
    metric_id_fields = ("fixed_task", "analysis", "policy", "reference", "reference_name", "pair")
    templates = {arm: {} for arm in arms}
    numeric_fields = {arm: set() for arm in arms}
    metric_names = {
        "auroc", "brier", "delta_auroc", "delta_brier", "reference_auroc",
        "reference_brier", "n_subjects", "n_repeats", "random_auroc_mean",
        "random_auroc_sd", "random_auroc_q05", "random_auroc_q25", "random_auroc_q50",
        "random_auroc_q75", "random_auroc_q95", "random_brier_mean", "random_brier_sd",
        "random_brier_q05", "random_brier_q25", "random_brier_q50", "random_brier_q75",
        "random_brier_q95", "auroc_random_percentile", "brier_random_percentile",
        "mean_selected_tasks", "delta_mean_selected_tasks", "mean_training_token_burden",
        "delta_mean_training_token_burden",
    }
    nonmetric_fields = {"run_id", "seed", "arm", *metric_id_fields, *expected_conditions[0].keys()}
    for row in flattened:
        arm = row["arm"]
        identity = {field: row.get(field) for field in metric_id_fields}
        templates.setdefault(arm, {})[_canonical(identity)] = identity
        for field, value in row.items():
            if field not in nonmetric_fields and (isinstance(value, (int, float)) or field in metric_names):
                numeric_fields.setdefault(arm, set()).add(field)
    present_cells = {(_condition_key(record["condition"]), record["seed"], record["arm"])
                     for record in cache_records}
    for condition in expected_conditions:
        condition_key = _condition_key(condition)
        for arm in arms:
            for seed in expected_seeds:
                if (condition_key, seed, arm) in present_cells:
                    continue
                for identity in templates.get(arm, {}).values():
                    placeholder = {"run_id": "", "seed": seed, "arm": arm,
                                   **condition, **identity}
                    placeholder.update({field: None for field in numeric_fields.get(arm, ())})
                    flattened.append(placeholder)

    condition_summary, condition_fields = _aggregate(flattened, expected_seeds)

    metric_fields = sorted({key for row in flattened for key in row})
    if not metric_fields:
        metric_fields = ["run_id", "seed", "arm", "analysis", "policy"]
    selection_fields = sorted({key for row in selection_rows for key in row})
    if not selection_fields:
        selection_fields = ["run_id", "seed", "arm", "policy", "task", "count"]
    summary_fields = sorted({key for row in condition_summary for key in row})
    if not summary_fields:
        summary_fields = ["status", "n_seeds"]
    _write_csv(output / "run_metrics.csv", flattened, metric_fields)
    _write_csv(output / "task_selection_counts.csv", selection_rows, selection_fields)
    _write_csv(output / "condition_summary.csv", condition_summary, summary_fields)

    selection_groups = {}
    condition_field_names = tuple(expected_conditions[0])
    for row in selection_rows:
        condition = {field: row[field] for field in condition_field_names}
        key = (_condition_key(condition), row["arm"], row["policy"], row["task"])
        selection_groups.setdefault(key, []).append(row)
    selection_summary = []
    expected_seed_set = set(expected_seeds)
    for (condition_key, arm, policy, task), group in sorted(
            selection_groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        condition = json.loads(condition_key)
        seeds = {row["seed"] for row in group}
        frequencies = [row["count"] / row["selection_opportunities"]
                       for row in group if row["selection_opportunities"]]
        complete = seeds == expected_seed_set and len(seeds) == len(group)
        values = np.asarray(frequencies, dtype=np.float64)
        selection_summary.append({
            **condition, "arm": arm, "policy": policy, "task": task,
            "n_seeds": len(seeds), "status": "complete" if complete else "incomplete",
            "selection_count_sum": sum(int(row["count"]) for row in group),
            "selection_opportunities_sum": sum(int(row["selection_opportunities"]) for row in group),
            "mean_frequency_across_seeds": float(values.mean()) if complete and len(values) else None,
            "sd_frequency_across_seeds": (float(values.std(ddof=1)) if complete and len(values) > 1
                                           else (0.0 if complete and len(values) == 1 else None)),
        })
    selection_summary_fields = sorted({key for row in selection_summary for key in row})
    _write_csv(output / "task_selection_summary.csv", selection_summary,
               selection_summary_fields or ["status", "n_seeds"])

    run_ids = {run["run_id"] for run in runs}
    complete_fit_runs = len(run_ids)
    arm_replays = len(cache_records)
    overall_complete = (complete_fit_runs == expected_runs and
                        arm_replays == expected_runs * len(arms) and not failures and
                        bool(condition_summary) and all(row["status"] == "complete"
                                                        for row in condition_summary))
    metadata = {
        "schema_version": 1,
        "status": "complete" if overall_complete else "incomplete",
        "fit_root_label": fit_root.name,
        "fit_runs_discovered": complete_fit_runs,
        "expected_fit_runs": expected_runs,
        "arm_replays_complete": arm_replays,
        "expected_arm_replays": expected_runs * len(arms),
        "arms": list(arms),
        "expected_seeds": list(expected_seeds),
        "expected_conditions": list(expected_conditions),
        "random_draws_per_arm": random_draws,
        "random_seed_start": random_seed,
        "fixed_task": fixed_task,
        "replay_source_sha256": replay_sha,
        "replay_source_files_sha256": replay_sources,
        "failed_items": len(failures),
        "uncertainty": "Across independent generated datasets only; no per-run bootstrap or clinical confidence interval.",
        "updated_at_utc": _now(),
    }
    _atomic_json(output / "summary_metadata.json", metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=DEFAULT_EXPECTED_RUNS)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--random-draws", type=int, default=DEFAULT_RANDOM_DRAWS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--fixed-task")
    args = parser.parse_args(argv)
    try:
        result = summarize_factorial(
            args.fit_root, args.output,
            expected_seeds=args.expected_seeds,
            expected_runs=args.expected_runs,
            random_draws=args.random_draws,
            random_seed=args.random_seed,
            fixed_task=args.fixed_task,
        )
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 2
    print(f"Summary status: {result['status']}; fits {result['fit_runs_discovered']}/{result['expected_fit_runs']}; "
          f"arm replays {result['arm_replays_complete']}/{result['expected_arm_replays']}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
