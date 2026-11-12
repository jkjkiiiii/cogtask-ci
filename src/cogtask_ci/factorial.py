"""Deterministic synthetic task-signal factorial and resumable fit-only runner."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
from importlib.metadata import version

import numpy as np

from .data import Record, records_to_jsonl
from .evaluation import Config, outer_assignments, preflight, run_arm


BASE_STRENGTHS = (0.60, 0.70, 0.80)
ADDED_TYPES = ("no_information", "duplicated", "complementary")
QUANTITY_LEVELS = (False, True)
POSITIVE_FRACTIONS = (0.50, 0.30)
FACTORIAL_SEEDS = tuple(range(20261112, 20261142))
TASKS = ("task_a", "task_b", "task_c")
SHORT_TOKENS = 12
LONG_TOKENS = 48


def factorial_conditions():
    """Return the fixed 36-condition order used by the synthetic runner."""
    return [
        {"base_strength": base, "added_type": added, "quantity": quantity,
         "positive_fraction": prevalence}
        for base in BASE_STRENGTHS
        for added in ADDED_TYPES
        for quantity in QUANTITY_LEVELS
        for prevalence in POSITIVE_FRACTIONS
    ]


def condition_id(seed, condition):
    return (f"seed{seed}_base{condition['base_strength']:.2f}_"
            f"{condition['added_type']}_quantity{'on' if condition['quantity'] else 'off'}_"
            f"positive{condition['positive_fraction']:.2f}")


def _signal(rng, label, strength):
    probability = strength if label == 1 else 1.0 - strength
    return int(rng.random() < probability)


def _length(rng, label, quantity):
    if not quantity:
        return SHORT_TOKENS
    p_long = 0.65 if label == 1 else 0.35
    return LONG_TOKENS if rng.random() < p_long else SHORT_TOKENS


def _segments(task, signal, length):
    marker = f"{task}_signal_{'x' if signal else 'y'}"
    return ((marker,) + ("shared",) * (length - 1),)


def generate_factorial_records(subjects=200, base_strength=0.60,
                               added_type="no_information", quantity=False,
                               positive_fraction=0.50, seed=20261112):
    """Generate 3-task records with controlled conditional lexical signals.

    Task A has sensitivity and specificity equal to ``base_strength``. Tasks B/C
    are fair noise, exact copies of A, or independent signals with sensitivity
    and specificity 0.75. When quantity is enabled, long responses are four
    times the short length and occur with probabilities 0.35 (label 0) and
    0.65 (label 1). In the duplicated condition, lengths are copied with text.
    """
    if type(subjects) is not int or subjects < 12:
        raise ValueError("subjects must be an integer >=12")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(base_strength) not in (int, float) or not 0.5 < base_strength < 1:
        raise ValueError("base_strength must be between 0.5 and 1")
    if added_type not in ADDED_TYPES:
        raise ValueError("added_type must be no_information, duplicated or complementary")
    if type(quantity) is not bool:
        raise ValueError("quantity must be boolean")
    if positive_fraction not in POSITIVE_FRACTIONS:
        raise ValueError("positive_fraction must be 0.50 or 0.30")
    n_positive = subjects * positive_fraction
    if not float(n_positive).is_integer():
        raise ValueError("subjects must allow an exact positive count at this prevalence")

    rng = np.random.default_rng(seed)
    labels = np.array([1] * int(n_positive) + [0] * (subjects - int(n_positive)), dtype=int)
    rng.shuffle(labels)
    records = []
    for i, label_value in enumerate(labels):
        label = int(label_value)
        base_signal = _signal(rng, label, float(base_strength))
        task_signals = [base_signal]
        for _ in (1, 2):
            if added_type == "no_information":
                task_signals.append(int(rng.random() < 0.5))
            elif added_type == "duplicated":
                task_signals.append(base_signal)
            else:
                task_signals.append(_signal(rng, label, 0.75))

        task_lengths = [_length(rng, label, quantity)]
        for _ in (1, 2):
            task_lengths.append(task_lengths[0] if added_type == "duplicated"
                                else _length(rng, label, quantity))

        for j, task in enumerate(TASKS):
            if added_type == "duplicated" and j > 0:
                segments = _segments(TASKS[0], task_signals[0], task_lengths[0])
            else:
                segments = _segments(task, task_signals[j], task_lengths[j])
            records.append(Record(f"S{i:04d}", task, label, segments))
    return records


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _source_hashes():
    source_dir = Path(__file__).parent
    return {path.name: _sha256(path.read_bytes()) for path in sorted(source_dir.glob("*.py"))}


def _environment():
    return {"python": sys.version, "platform": platform.platform(),
            "numpy": version("numpy"), "scikit_learn": version("scikit-learn")}


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class _QuietLogger:
    def info(self, *args, **kwargs):
        pass


def run_factorial(output, *, start_index=0, limit=None, seed_values=FACTORIAL_SEEDS,
                  conditions=None, subjects=200, config=None, provenance_context=None):
    """Fit each selected synthetic condition once and cache fold replay bundles.

    A complete sweep is 36 conditions by 30 predetermined seeds (1,080 runs).
    ``start_index`` and ``limit`` select a stable slice of that fixed ordered
    list. Runs store fitting outputs without per-run bootstrap; uncertainty is
    intended to be summarized across the independent synthetic seeds.
    """
    output = Path(output)
    if output.exists() and not output.is_dir():
        raise ValueError("output must be a directory")
    output.mkdir(parents=True, exist_ok=True)
    if type(start_index) is not int or start_index < 0:
        raise ValueError("start_index must be a nonnegative integer")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    seed_values = tuple(seed_values)
    if not seed_values or any(type(seed) is not int or seed < 0 for seed in seed_values):
        raise ValueError("seed_values must contain nonnegative integer seeds")
    conditions = list(conditions) if conditions is not None else factorial_conditions()
    if not conditions:
        raise ValueError("At least one condition is required")
    config = config or Config(outer_folds=3, inner_folds=3, repeats=1, seed=42, bootstrap=1)
    config.validate()
    if config.per_task_budget is not None or config.total_token_budget is not None:
        raise ValueError("Factorial runs require full-text task models; token budgets are not supported")
    if config.repeats != 1:
        raise ValueError("The factorial seed is the replication unit; set repeats=1")
    provenance_context = dict(provenance_context or {})
    allowed_context = {"protocol_sha256", "protocol_git_commit", "code_git_commit"}
    if set(provenance_context) - allowed_context:
        raise ValueError("provenance_context may contain only protocol and code commit identifiers")

    jobs = [(seed, condition) for seed in seed_values for condition in conditions]
    if start_index >= len(jobs):
        raise ValueError("start_index is outside the selected factorial job list")
    selected_jobs = jobs[start_index:start_index + limit if limit is not None else None]
    source_hashes = _source_hashes()
    code_hash = _sha256(_canonical_bytes(source_hashes))
    config_value = asdict(config)
    config_hash = _sha256(_canonical_bytes(config_value))
    summaries = []

    for seed, condition in selected_jobs:
        condition_value = dict(condition)
        run_id = condition_id(seed, condition_value)
        run_dir = output / run_id
        records = generate_factorial_records(subjects=subjects, seed=seed, **condition_value)
        input_bytes = records_to_jsonl(records)
        input_hash = _sha256(input_bytes)
        condition_hash = _sha256(_canonical_bytes(condition_value))
        expected = {
            "schema_version": 1,
            "run_id": run_id,
            "condition": condition_value,
            "condition_sha256": condition_hash,
            "seed": seed,
            "subjects": subjects,
            "positive_subjects": sum(record.label for record in records[::len(TASKS)]),
            "input_sha256": input_hash,
            "config": config_value,
            "config_sha256": config_hash,
            "code_sha256": code_hash,
            "code_file_sha256": source_hashes,
            "protocol_sha256": provenance_context.get("protocol_sha256"),
            "protocol_git_commit": provenance_context.get("protocol_git_commit"),
            "code_git_commit": provenance_context.get("code_git_commit"),
            "code_commit": provenance_context.get("code_git_commit"),
            "environment": _environment(),
        }
        allowed_files = {"input.jsonl", "text_bundle.json", "quantity_bundle.json", "result.json"}
        if run_dir.exists():
            if not run_dir.is_dir():
                raise ValueError("Condition path exists and is not a directory: " + run_id)
            unexpected = {p.name for p in run_dir.iterdir()} - allowed_files
            if unexpected:
                raise ValueError("Unexpected files in resumable run " + run_id + ": " + ",".join(sorted(unexpected)))
            result_path = run_dir / "result.json"
            if result_path.exists():
                result = _read_json(result_path)
                for key, value in expected.items():
                    if result.get(key) != value:
                        raise ValueError("Existing run provenance differs for " + run_id + ": " + key)
                if result.get("status") == "complete":
                    for arm in ("text", "quantity"):
                        bundle_path = run_dir / (arm + "_bundle.json")
                        if not bundle_path.is_file() or _sha256(bundle_path.read_bytes()) != result.get("bundle_sha256", {}).get(arm):
                            raise ValueError("Completed run has a missing or changed replay bundle: " + run_id)
                    summaries.append({"run_id": run_id, "status": "complete", "resumed": True})
                    continue
                if result.get("status") not in {"running", "failed"}:
                    raise ValueError("Cannot resume this run state: " + run_id)
            else:
                raise ValueError("Existing run has no result.json and cannot be resumed safely: " + run_id)
            input_path = run_dir / "input.jsonl"
            if input_path.exists() and _sha256(input_path.read_bytes()) != input_hash:
                raise ValueError("Existing synthetic input changed: " + run_id)
        else:
            run_dir.mkdir()
            (run_dir / "input.jsonl").write_bytes(input_bytes)
            started = datetime.now(timezone.utc).isoformat()
            _atomic_json(run_dir / "result.json", {**expected, "status": "running",
                                                   "started_at_utc": started,
                                                   "timestamp_utc": started})

        input_path = run_dir / "input.jsonl"
        if not input_path.exists():
            input_path.write_bytes(input_bytes)
        result_path = run_dir / "result.json"
        result = _read_json(result_path)
        bundle_hashes = dict(result.get("bundle_sha256", {}))
        if result.get("status") == "failed":
            result["status"] = "running"
            for key in ("error_type", "error", "failed_at_utc"):
                result.pop(key, None)
            result["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(result_path, result)
        try:
            labels = {record.subject_id: record.label for record in records}
            subject_ids = sorted(labels)
            complete = subject_ids
            assignments = outer_assignments(subject_ids, complete, labels, config)
            preflight(records, assignments, config)
            for arm in ("text", "quantity"):
                bundle_path = run_dir / (arm + "_bundle.json")
                if bundle_path.exists():
                    if arm not in bundle_hashes or _sha256(bundle_path.read_bytes()) != bundle_hashes[arm]:
                        raise ValueError("Untracked or changed partial bundle prevents safe resume: " + run_id)
                    continue
                collected = {}
                run_arm(records, arm, config, assignments, _QuietLogger(),
                        replay_collector=collected)
                bundle_bytes = _canonical_bytes(collected[arm]) + b"\n"
                bundle_path.write_bytes(bundle_bytes)
                bundle_hashes[arm] = _sha256(bundle_bytes)
                result["bundle_sha256"] = dict(bundle_hashes)
                _atomic_json(result_path, result)

            result.update(expected)
            result["status"] = "complete"
            result["bundle_sha256"] = bundle_hashes
            completed = datetime.now(timezone.utc).isoformat()
            result["completed_at_utc"] = completed
            result["timestamp_utc"] = completed
            _atomic_json(result_path, result)
        except Exception as error:
            result["status"] = "failed"
            result["error_type"] = type(error).__name__
            result["error"] = str(error)
            result["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
            result["bundle_sha256"] = bundle_hashes
            _atomic_json(result_path, result)
            raise
        summaries.append({"run_id": run_id, "status": "complete", "resumed": False})
    return summaries

