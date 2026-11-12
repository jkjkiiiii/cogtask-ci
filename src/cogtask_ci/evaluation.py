"""Nested participant-level task selection and conditional paired uncertainty."""
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
import csv
import hashlib
import json
import logging
import platform
import sys
from importlib.metadata import version
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from .data import load_dataset, describe, crop
from .model import fit_predict


@dataclass(frozen=True)
class Config:
    outer_folds: int = 5
    inner_folds: int = 3
    repeats: int = 1
    seed: int = 42
    bootstrap: int = 2000
    # ``token_budget`` is the 0.1.x spelling retained for source compatibility.
    token_budget: int | None = None
    lowercase: bool = False
    token_budget_per_task: int | None = None
    total_token_budget: int | None = None

    def validate(self):
        for name in ("outer_folds", "inner_folds", "repeats", "bootstrap", "seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(name + " must be an integer")
        if min(self.outer_folds, self.inner_folds) < 2 or self.repeats < 1 or self.bootstrap < 1 or self.seed < 0 or self.seed >= 2**31:
            raise ValueError("Need folds>=2, repeats>=1, bootstrap>=1 and 0<=seed<2**31")
        if type(self.lowercase) is not bool:
            raise ValueError("lowercase must be a boolean")
        budgets = [self.token_budget, self.token_budget_per_task, self.total_token_budget]
        if sum(value is not None for value in budgets) > 1:
            raise ValueError("Choose only one of token_budget, token_budget_per_task or total_token_budget")
        for name, value in (("token_budget", self.token_budget),
                            ("token_budget_per_task", self.token_budget_per_task),
                            ("total_token_budget", self.total_token_budget)):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(name + " must be a positive integer")

    @property
    def per_task_budget(self):
        """Return the per-task budget, including the legacy spelling."""
        return self.token_budget_per_task if self.token_budget_per_task is not None else self.token_budget


def validate_total_token_budget(total_budget, n_tasks):
    """Require exact equal integer allocation for every task count in an analysis."""
    if type(total_budget) is not int or total_budget < 1:
        raise ValueError("total_token_budget must be a positive integer")
    if n_tasks < 1:
        raise ValueError("At least one task is required")
    invalid = [k for k in range(1, n_tasks + 1) if total_budget % k]
    if invalid:
        raise ValueError("total_token_budget must be divisible by every tested task count; "
                         "not divisible by k=" + ",".join(map(str, invalid)))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def population(rows):
    tasks = sorted({r.task for r in rows})
    labels = {r.subject_id: r.label for r in rows}
    lookup = {(r.subject_id, r.task): r for r in rows}
    complete = sorted(s for s in labels if all((s, t) in lookup for t in tasks))
    return tasks, labels, lookup, complete


def split_seed(config, repeat, outer=None):
    value = config.seed + repeat if outer is None else config.seed + repeat * 1000 + outer * 10 + 7
    return value % (2**32)


def preflight(rows, assignments, config):
    if not rows:
        raise ValueError("No eligible records")
    tasks, labels, lookup, complete = population(rows)
    for rep, assignment in enumerate(assignments):
        for outer in range(config.outer_folds):
            train = [s for s in labels if assignment[s] != outer]
            if set(labels[s] for s in complete if assignment[s] == outer) != {0, 1}:
                raise ValueError("Every outer test fold needs both classes in the complete-task set")
            inner = allocate(train, complete, labels, config.inner_folds, split_seed(config, rep, outer))
            for task in tasks:
                for fold in range(config.inner_folds):
                    if {labels[s] for s in train if inner[s] != fold and (s, task) in lookup} != {0, 1}:
                        raise ValueError("Every task's inner training fold needs both classes")


def allocate(subjects, complete, labels, folds, seed):
    """Stratify the common evaluable set; distribute extra training subjects."""
    complete = sorted(set(subjects) & set(complete))
    if any(sum(labels[s] == y for s in complete) < folds for y in (0, 1)):
        raise ValueError(f"Complete-task evaluation needs at least {folds} subjects per class in every splitting population")
    assignment = {}
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splitter.split(complete, [labels[s] for s in complete])):
        assignment.update({complete[i]: fold for i in test})
    extras = sorted(set(subjects) - set(complete))
    rng = np.random.default_rng(seed)
    for y in (0, 1):
        group = [s for s in extras if labels[s] == y]
        rng.shuffle(group)
        assignment.update({s: i % folds for i, s in enumerate(group)})
    return assignment


def outer_assignments(subjects, complete, labels, config, supplied=None):
    if supplied is None:
        return [allocate(subjects, complete, labels, config.outer_folds, split_seed(config, i)) for i in range(config.repeats)]
    with Path(supplied).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"repeat", "subject_id", "outer_fold"}.issubset(reader.fieldnames or []):
            raise ValueError("Split CSV requires repeat,subject_id,outer_fold (zero-based)")
        assignments = [{} for _ in range(config.repeats)]
        for r in reader:
            rep, fold, subject = int(r["repeat"]), int(r["outer_fold"]), r["subject_id"]
            if not 0 <= rep < config.repeats or not 0 <= fold < config.outer_folds or subject not in subjects or subject in assignments[rep]:
                raise ValueError("Invalid, unknown or duplicate split assignment")
            assignments[rep][subject] = fold
    if any(set(a) != set(subjects) for a in assignments):
        raise ValueError("Split CSV must assign every subject once per repeat")
    return assignments


def select_subsets(y, task_scores, tasks):
    """Exact inner-OOF AUROC selection; lexical enumeration breaks ties."""
    selected = {}
    for k in range(1, len(tasks) + 1):
        best, best_auc = None, -1
        for subset in combinations(range(len(tasks)), k):
            auc = roc_auc_score(y, task_scores[:, subset].mean(axis=1))
            if auc > best_auc:
                best, best_auc = subset, float(auc)
        selected[k] = (best, best_auc)
    return selected


def run_arm(rows, arm, config, assignments, logger, replay_collector=None, original_token_counts=None):
    tasks, labels, lookup, complete = population(rows)
    subjects = sorted(labels)
    predictions, selections, split_audit, replay_folds = [], [], [], []
    original_token_counts = original_token_counts or {
        (r.subject_id, r.task): r.n_tokens for r in rows
    }
    for rep, assignment in enumerate(assignments):
        for outer in range(config.outer_folds):
            train = [s for s in subjects if assignment[s] != outer]
            test = [s for s in complete if assignment[s] == outer]
            if set(labels[s] for s in test) != {0, 1}:
                raise ValueError("Every outer test fold needs both classes in the complete-task set")
            inner_eval = sorted(set(train) & set(complete))
            inner = allocate(train, inner_eval, labels, config.inner_folds, split_seed(config, rep, outer))
            inner_scores = np.full((len(inner_eval), len(tasks)), np.nan)
            positions = {s: i for i, s in enumerate(inner_eval)}
            for fold in range(config.inner_folds):
                tr = [s for s in train if inner[s] != fold]
                va = [s for s in inner_eval if inner[s] == fold]
                for s in train:
                    split_audit.append(dict(arm=arm, repeat=rep, outer_fold=outer, inner_fold=fold, subject_id=s,
                                            role=("validation" if s in positions else "held_out_incomplete") if inner[s] == fold else "train"))
                for j, task in enumerate(tasks):
                    training = [lookup[s, task] for s in tr if (s, task) in lookup]
                    validation = [lookup[s, task] for s in va]
                    scores = fit_predict(training, validation, arm, config.seed)
                    inner_scores[[positions[s] for s in va], j] = scores
            if not np.isfinite(inner_scores).all():
                raise ValueError("Inner OOF predictions are incomplete or nonfinite")
            selected = select_subsets([labels[s] for s in inner_eval], inner_scores, tasks)
            outer_scores = np.column_stack([
                fit_predict([lookup[s, t] for s in train if (s, t) in lookup], [lookup[s, t] for s in test], arm, config.seed)
                for t in tasks])
            replay_folds.append(dict(
                repeat=rep,
                outer_fold=outer,
                inner_subjects=list(inner_eval),
                inner_labels=[labels[s] for s in inner_eval],
                inner_scores=inner_scores.tolist(),
                train_token_counts=[[original_token_counts.get((s, task), 0) for task in tasks]
                                    for s in inner_eval],
                test_subjects=list(test),
                test_labels=[labels[s] for s in test],
                test_scores=outer_scores.tolist(),
            ))
            logger.info("arm=%s repeat=%d outer_fold=%d train=%d test=%d", arm, rep, outer, len(train), len(test))
            for k, (subset, auc) in selected.items():
                names = [tasks[i] for i in subset]
                selections.append(dict(arm=arm, repeat=rep, outer_fold=outer, k=k,
                                       selected_tasks=json.dumps(names), inner_auc=auc))
                fused = outer_scores[:, subset].mean(axis=1)
                for i, s in enumerate(test):
                    predictions.append(dict(arm=arm, repeat=rep, outer_fold=outer, subject_id=s,
                                            label=labels[s], k=k, score=float(fused[i])))
    if replay_collector is not None:
        replay_collector[arm] = {"schema_version": 1, "tasks": tasks, "folds": replay_folds}
    return predictions, selections, split_audit


def run_total_budget_arm(rows, arm, config, assignments, total_token_budget, logger,
                         replay_collector=None, original_token_counts=None):
    """Fit task models separately at total_budget / k for each tested k."""
    tasks, labels, lookup, complete = population(rows)
    validate_total_token_budget(total_token_budget, len(tasks))
    subjects = sorted(labels)
    predictions, selections, split_audit = [], [], []
    replay_folds_by_k = {k: [] for k in range(1, len(tasks) + 1)}
    original_token_counts = original_token_counts or {
        (r.subject_id, r.task): r.n_tokens for r in rows
    }
    for rep, assignment in enumerate(assignments):
        for outer in range(config.outer_folds):
            train = [s for s in subjects if assignment[s] != outer]
            test = [s for s in complete if assignment[s] == outer]
            if set(labels[s] for s in test) != {0, 1}:
                raise ValueError("Every outer test fold needs both classes in the complete-task set")
            inner_eval = sorted(set(train) & set(complete))
            inner = allocate(train, inner_eval, labels, config.inner_folds, split_seed(config, rep, outer))
            positions = {s: i for i, s in enumerate(inner_eval)}
            # Folds are shared for every k, so the audit is written once per fold.
            for fold in range(config.inner_folds):
                for s in train:
                    split_audit.append(dict(arm=arm, repeat=rep, outer_fold=outer, inner_fold=fold,
                                            subject_id=s,
                                            role=("validation" if s in positions else "held_out_incomplete")
                                            if inner[s] == fold else "train"))
            y_inner = [labels[s] for s in inner_eval]
            for k in range(1, len(tasks) + 1):
                per_task_budget = total_token_budget // k
                inner_scores = np.full((len(inner_eval), len(tasks)), np.nan)
                for fold in range(config.inner_folds):
                    tr = [s for s in train if inner[s] != fold]
                    va = [s for s in inner_eval if inner[s] == fold]
                    for j, task in enumerate(tasks):
                        training = [replace(lookup[s, task], segments=crop(lookup[s, task].segments, per_task_budget))
                                    for s in tr if (s, task) in lookup]
                        validation = [replace(lookup[s, task], segments=crop(lookup[s, task].segments, per_task_budget))
                                      for s in va]
                        scores = fit_predict(training, validation, arm, config.seed)
                        inner_scores[[positions[s] for s in va], j] = scores
                if not np.isfinite(inner_scores).all():
                    raise ValueError("Total-budget inner OOF predictions are incomplete or nonfinite")
                selected_subset, inner_auc = select_subsets(y_inner, inner_scores, tasks)[k]
                outer_scores = np.column_stack([
                    fit_predict(
                        [replace(lookup[s, task], segments=crop(lookup[s, task].segments, per_task_budget))
                         for s in train if (s, task) in lookup],
                        [replace(lookup[s, task], segments=crop(lookup[s, task].segments, per_task_budget))
                         for s in test],
                        arm, config.seed)
                    for task in tasks
                ])
                replay_folds_by_k[k].append(dict(
                    repeat=rep,
                    outer_fold=outer,
                    inner_subjects=list(inner_eval),
                    inner_labels=list(y_inner),
                    inner_scores=inner_scores.tolist(),
                    train_token_counts=[[original_token_counts.get((s, task), 0) for task in tasks]
                                        for s in inner_eval],
                    test_subjects=list(test),
                    test_labels=[labels[s] for s in test],
                    test_scores=outer_scores.tolist(),
                ))
                logger.info("arm=%s repeat=%d outer_fold=%d k=%d per_task_budget=%d train=%d test=%d",
                            arm, rep, outer, k, per_task_budget, len(train), len(test))
                selections.append(dict(arm=arm, repeat=rep, outer_fold=outer, k=k,
                                       selected_tasks=json.dumps([tasks[i] for i in selected_subset]),
                                       inner_auc=inner_auc))
                fused = outer_scores[:, selected_subset].mean(axis=1)
                for i, subject in enumerate(test):
                    predictions.append(dict(arm=arm, repeat=rep, outer_fold=outer, subject_id=subject,
                                            label=labels[subject], k=k, score=float(fused[i])))
    if replay_collector is not None:
        for k, folds in replay_folds_by_k.items():
            key = f"{arm}_total{total_token_budget}_k{k}"
            replay_collector[key] = {
                "schema_version": 1,
                "tasks": tasks,
                "folds": folds,
                "budget": {"mode": "total", "total_tokens": total_token_budget,
                           "per_task_tokens": total_token_budget // k, "fixed_k": k},
            }
    return predictions, selections, split_audit


def summarize(predictions, bootstrap, seed):
    summary = []
    for arm in sorted({r["arm"] for r in predictions}):
        rows = [r for r in predictions if r["arm"] == arm]
        subjects = sorted({r["subject_id"] for r in rows})
        repeats = sorted({r["repeat"] for r in rows})
        ks = sorted({r["k"] for r in rows})
        labels = {r["subject_id"]: r["label"] for r in rows}
        y = np.array([labels[s] for s in subjects])
        index = {(r["repeat"], r["k"], r["subject_id"]): r["score"] for r in rows}
        arrays = {k: np.array([[index[rep, k, s] for s in subjects] for rep in repeats]) for k in ks}
        def estimate(values, ids):
            return float(np.mean([roc_auc_score(y[ids], scores[ids]) for scores in values]))
        all_ids = np.arange(len(y))
        points = {k: estimate(arrays[k], all_ids) for k in ks}
        samples = {k: [] for k in ks}
        rng = np.random.default_rng(seed)
        classes = [np.flatnonzero(y == label) for label in (0, 1)]
        for _ in range(bootstrap):
            ids = np.concatenate([rng.choice(g, len(g), replace=True) for g in classes])
            for k in ks:
                samples[k].append(estimate(arrays[k], ids))
        baseline = np.array(samples[1])
        for k in ks:
            vals = np.array(samples[k])
            low, high = np.quantile(vals, [.025, .975])
            dl, dh = np.quantile(vals - baseline, [.025, .975])
            summary.append(dict(arm=arm, k=k, n_subjects=len(y), n_repeats=len(repeats),
                                auroc=points[k], ci_low=float(low), ci_high=float(high),
                                delta_vs_single=points[k] - points[1], delta_ci_low=float(dl), delta_ci_high=float(dh)))
    return summary


def contrasts(predictions, bootstrap, seed):
    """Within-population paired arm differences, sharing subject resamples."""
    result = []
    arms = {r["arm"] for r in predictions}
    for left, right in (("text", "quantity"), ("text_matched", "quantity_matched"),
                        ("fixed_text", "text_matched"), ("fixed_total_text", "text_matched")):
        if not {left, right}.issubset(arms):
            continue
        a = [r for r in predictions if r["arm"] == left]
        b = [r for r in predictions if r["arm"] == right]
        ka = {(r["repeat"], r["k"], r["subject_id"]): r for r in a}
        kb = {(r["repeat"], r["k"], r["subject_id"]): r for r in b}
        if set(ka) != set(kb) or any(ka[k]["label"] != kb[k]["label"] for k in ka):
            raise ValueError("Paired arm comparison requires identical subjects, labels and repeats")
        subjects = sorted({r["subject_id"] for r in a})
        repeats = sorted({r["repeat"] for r in a})
        y = np.array([next(r["label"] for r in a if r["subject_id"] == s) for s in subjects])
        for k in sorted({r["k"] for r in a}):
            x = np.array([[ka[rep, k, s]["score"] for s in subjects] for rep in repeats])
            z = np.array([[kb[rep, k, s]["score"] for s in subjects] for rep in repeats])
            x1 = np.array([[ka[rep, 1, s]["score"] for s in subjects] for rep in repeats])
            z1 = np.array([[kb[rep, 1, s]["score"] for s in subjects] for rep in repeats])
            def delta(ids):
                return float(np.mean([roc_auc_score(y[ids], p[ids]) - roc_auc_score(y[ids], q[ids]) for p, q in zip(x, z)]))
            def gain_delta(ids):
                return delta(ids) - float(np.mean([roc_auc_score(y[ids], p[ids]) - roc_auc_score(y[ids], q[ids]) for p, q in zip(x1, z1)]))
            rng = np.random.default_rng(seed)
            groups = [np.flatnonzero(y == v) for v in (0, 1)]
            resamples = [np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]) for _ in range(bootstrap)]
            values = [delta(ids) for ids in resamples]
            low, high = np.quantile(values, [.025, .975])
            gl, gh = np.quantile([gain_delta(ids) for ids in resamples], [.025, .975])
            result.append(dict(left_arm=left, right_arm=right, k=k, n_subjects=len(subjects),
                               auroc_difference=delta(np.arange(len(y))), ci_low=float(low), ci_high=float(high),
                               gain_difference=gain_delta(np.arange(len(y))), gain_ci_low=float(gl), gain_ci_high=float(gh)))
    return result


def evaluate(input_path, output, config=None, splits=None):
    """Write a new local run, retaining logs/status if evaluation fails."""
    config = config or Config()
    config.validate()
    source_hash = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    split_hash = hashlib.sha256(Path(splits).read_bytes()).hexdigest() if splits else None
    rows = load_dataset(input_path, config.lowercase)
    output = Path(output)
    if output.exists():
        raise ValueError("Output directory must not exist")
    tasks, labels, lookup, complete = population(rows)
    if config.total_token_budget is not None:
        validate_total_token_budget(config.total_token_budget, len(tasks))
    assignments = outer_assignments(sorted(labels), complete, labels, config, splits)
    # Validate nested splits before creating a run or fitting anything.
    preflight(rows, assignments, config)
    output.mkdir(parents=True)
    logger = logging.getLogger("cogtask_ci.run." + str(output.resolve()))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    status = {"status": "running"}
    save_json(output / "status.json", status)
    try:
        save_json(output / "config.json", asdict(config))
        save_json(output / "environment.json", {"python": sys.version, "platform": platform.platform(),
                  "packages": {p: version(p) for p in ("cogtask-ci", "numpy", "scipy", "scikit-learn", "matplotlib")}})
        save_json(output / "provenance.json", {"input_sha256": source_hash,
                  "package_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
                  "splits_sha256": split_hash,
                  "method": "Nested subject-level selection; fold-local TF-IDF; equal probability fusion",
                  "uncertainty": "Class-stratified paired subject bootstrap conditional on realized fits/folds; no refitting",
                  "privacy": "Predictions, splits, audit and report contain participant-derived information. Keep local."})
        write_csv(output / "outer_splits.csv", [dict(repeat=i, subject_id=s, outer_fold=a[s]) for i, a in enumerate(assignments) for s in sorted(a)])
        coverage = describe(rows)
        coverage["excluded_from_common_evaluation"] = sorted(set(labels) - set(complete))
        jobs = [("text", rows, assignments), ("quantity", rows, assignments)]
        total_budget_job = None
        token_budget_per_task = config.per_task_budget
        original_token_counts = {(r.subject_id, r.task): r.n_tokens for r in rows}
        if token_budget_per_task is not None:
            eligible = sorted(s for s in complete
                              if all(lookup[s, t].n_tokens >= token_budget_per_task for t in tasks))
            eligible_set = set(eligible)
            coverage["fixed_budget_subjects"] = len(eligible)
            coverage["excluded_from_fixed_budget"] = sorted(set(labels) - eligible_set)
            coverage["per_task_token_budget"] = token_budget_per_task
            matched = [r for r in rows if r.subject_id in eligible_set]
            # Reuse original outer fold identities, never optimize folds on outcomes.
            matched_assignments = [{s: assignment[s] for s in eligible} for assignment in assignments]
            jobs.extend([
                ("text_matched", matched, matched_assignments),
                ("quantity_matched", matched, matched_assignments),
                ("fixed_text", [replace(r, segments=crop(r.segments, token_budget_per_task))
                                for r in matched], matched_assignments),
            ])
        elif config.total_token_budget is not None:
            total = config.total_token_budget
            # Use one common population for all k. This rule also covers the
            # largest per-task prefix, which occurs at k=1.
            eligible = sorted(s for s in complete
                              if all(lookup[s, t].n_tokens >= total for t in tasks))
            eligible_set = set(eligible)
            coverage["total_budget_subjects"] = len(eligible)
            coverage["excluded_from_total_budget"] = sorted(set(labels) - eligible_set)
            coverage["total_token_budget"] = total
            coverage["total_budget_population_rule"] = (
                "complete-task subjects with at least total_token_budget tokens in every task")
            matched = [r for r in rows if r.subject_id in eligible_set]
            matched_assignments = [{s: assignment[s] for s in eligible} for assignment in assignments]
            jobs.extend([
                ("text_matched", matched, matched_assignments),
                ("quantity_matched", matched, matched_assignments),
            ])
            total_budget_job = ("fixed_total_text", matched, matched_assignments)
        save_json(output / "coverage.json", coverage)
        all_predictions, all_selections, all_audit = [], [], []
        for arm, records, arm_assignments in jobs:
            try:
                preflight(records, arm_assignments, config)
            except ValueError as e:
                raise ValueError(f"Arm {arm}: {e}") from e
        if total_budget_job is not None:
            try:
                preflight(total_budget_job[1], total_budget_job[2], config)
            except ValueError as e:
                raise ValueError(f"Arm {total_budget_job[0]}: {e}") from e
        replay_collector = {}
        for arm, records, arm_assignments in jobs:
            p, s, audit = run_arm(records, arm, config, arm_assignments, logger,
                                  replay_collector=replay_collector,
                                  original_token_counts=original_token_counts)
            all_predictions.extend(p)
            all_selections.extend(s)
            all_audit.extend(audit)
        if total_budget_job is not None:
            arm, records, arm_assignments = total_budget_job
            p, s, audit = run_total_budget_arm(records, arm, config, arm_assignments,
                                               config.total_token_budget, logger,
                                               replay_collector=replay_collector,
                                               original_token_counts=original_token_counts)
            all_predictions.extend(p)
            all_selections.extend(s)
            all_audit.extend(audit)
        replay_dir = output / "replay_bundles"
        replay_dir.mkdir()
        for arm, bundle in replay_collector.items():
            save_json(replay_dir / (arm + ".json"), bundle)
        write_csv(output / "predictions.csv", all_predictions)
        write_csv(output / "selections.csv", all_selections)
        write_csv(output / "inner_split_audit.csv", all_audit)
        metrics = summarize(all_predictions, config.bootstrap, config.seed)
        write_csv(output / "summary.csv", metrics)
        write_csv(output / "contrasts.csv", contrasts(all_predictions, config.bootstrap, config.seed))
        if hashlib.sha256(Path(input_path).read_bytes()).hexdigest() != source_hash:
            raise ValueError("Input changed during evaluation")
        if splits and hashlib.sha256(Path(splits).read_bytes()).hexdigest() != split_hash:
            raise ValueError("Split file changed during evaluation")
        from .report import render_report
        render_report(output)
        status = {"status": "complete", "summary_rows": len(metrics)}
        save_json(output / "status.json", status)
        logger.info("Run completed")
        return metrics
    except Exception as e:
        save_json(output / "status.json", {"status": "failed", "error_type": type(e).__name__, "error": str(e)})
        logger.exception("Run failed")
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
