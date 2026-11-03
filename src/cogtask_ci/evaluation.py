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
    token_budget: int | None = None
    lowercase: bool = False

    def validate(self):
        for name in ("outer_folds", "inner_folds", "repeats", "bootstrap", "seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(name + " must be an integer")
        if min(self.outer_folds, self.inner_folds) < 2 or self.repeats < 1 or self.bootstrap < 1 or self.seed < 0 or self.seed >= 2**31:
            raise ValueError("Need folds>=2, repeats>=1, bootstrap>=1 and 0<=seed<2**31")
        if type(self.lowercase) is not bool:
            raise ValueError("lowercase must be a boolean")
        if self.token_budget is not None and (type(self.token_budget) is not int or self.token_budget < 1):
            raise ValueError("token_budget must be a positive integer")


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


def run_arm(rows, arm, config, assignments, logger):
    tasks, labels, lookup, complete = population(rows)
    subjects = sorted(labels)
    predictions, selections, split_audit = [], [], []
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
            logger.info("arm=%s repeat=%d outer_fold=%d train=%d test=%d", arm, rep, outer, len(train), len(test))
            for k, (subset, auc) in selected.items():
                names = [tasks[i] for i in subset]
                selections.append(dict(arm=arm, repeat=rep, outer_fold=outer, k=k,
                                       selected_tasks=json.dumps(names), inner_auc=auc))
                fused = outer_scores[:, subset].mean(axis=1)
                for i, s in enumerate(test):
                    predictions.append(dict(arm=arm, repeat=rep, outer_fold=outer, subject_id=s,
                                            label=labels[s], k=k, score=float(fused[i])))
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
    for left, right in (("text", "quantity"), ("text_matched", "quantity_matched"), ("fixed_text", "text_matched")):
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
        if config.token_budget:
            eligible = sorted(s for s in complete if all(lookup[s, t].n_tokens >= config.token_budget for t in tasks))
            coverage["fixed_budget_subjects"] = len(eligible)
            coverage["excluded_from_fixed_budget"] = sorted(set(labels) - set(eligible))
            matched = [r for r in rows if r.subject_id in set(eligible)]
            # Reuse original outer fold identities, never optimize folds on outcomes.
            ma = [{s: a[s] for s in eligible} for a in assignments]
            jobs.extend([("text_matched", matched, ma), ("quantity_matched", matched, ma),
                         ("fixed_text", [replace(r, segments=crop(r.segments, config.token_budget)) for r in matched], ma)])
        save_json(output / "coverage.json", coverage)
        all_predictions, all_selections, all_audit = [], [], []
        for arm, records, arm_assignments in jobs:
            try:
                preflight(records, arm_assignments, config)
            except ValueError as e:
                raise ValueError(f"Arm {arm}: {e}") from e
        for arm, records, arm_assignments in jobs:
            p, s, audit = run_arm(records, arm, config, arm_assignments, logger)
            all_predictions.extend(p)
            all_selections.extend(s)
            all_audit.extend(audit)
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
