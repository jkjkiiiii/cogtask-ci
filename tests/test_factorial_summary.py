import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from cogtask_ci.factorial_summary import summarize_factorial


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score(label, task, subject):
    if task == 0:
        return 0.2 + 0.6 * label
    if task == 1:
        return 0.8 - 0.6 * label
    return 0.35 if int(subject[-1]) % 2 == 0 else 0.65


def _bundle():
    tasks = ["task_a", "task_b", "task_c"]
    labels = {f"s{i}": int(i % 2 == 1) for i in range(8)}
    folds = []
    for outer_fold in range(2):
        test_subjects = [f"s{i}" for i in range(8) if i // 4 == outer_fold]
        inner_subjects = [f"s{i}" for i in range(8) if i // 4 != outer_fold]
        folds.append({
            "repeat": 0,
            "outer_fold": outer_fold,
            "inner_subjects": inner_subjects,
            "inner_labels": [labels[s] for s in inner_subjects],
            "inner_scores": [[_score(labels[s], task, s) for task in range(3)]
                             for s in inner_subjects],
            "train_token_counts": [[10 + task for task in range(3)] for _ in inner_subjects],
            "test_subjects": test_subjects,
            "test_labels": [labels[s] for s in test_subjects],
            "test_scores": [[_score(labels[s], task, s) for task in range(3)]
                            for s in test_subjects],
        })
    return {"schema_version": 1, "tasks": tasks, "folds": folds}


class FactorialSummaryTests(unittest.TestCase):
    def _make_fit_root(self, root):
        condition = {"base_strength": 0.6, "added_type": "no_information",
                     "quantity": False, "positive_fraction": 0.5}
        for seed in (11, 12):
            run_id = f"seed{seed}_cell0"
            run_dir = root / run_id
            run_dir.mkdir(parents=True)
            bundle_hashes = {}
            for arm in ("text", "quantity"):
                bundle_path = run_dir / f"{arm}_bundle.json"
                bundle_path.write_text(json.dumps(_bundle()), encoding="utf-8")
                bundle_hashes[arm] = _sha(bundle_path)
            result = {"status": "complete", "run_id": run_id, "seed": seed,
                      "condition": condition, "bundle_sha256": bundle_hashes}
            (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return condition

    def test_replay_and_complete_or_incomplete_cross_seed_aggregation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit_root = root / "fits"
            fit_root.mkdir()
            condition = self._make_fit_root(fit_root)

            complete_out = root / "complete"
            complete = summarize_factorial(
                fit_root, complete_out, expected_seeds=(11, 12), expected_runs=2,
                random_draws=16, random_seed=7, fixed_task="task_a",
                expected_conditions=(condition,),
            )
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(complete["arm_replays_complete"], 4)
            self.assertEqual(len(list((complete_out / "arm_replays").glob("*.json"))), 4)
            self.assertTrue((complete_out / "task_selection_summary.csv").is_file())
            with (complete_out / "run_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
                run_rows = list(csv.DictReader(handle))
            e1 = [row for row in run_rows if row["analysis"] == "E1_pair_vs_reference"]
            self.assertEqual(len(e1), 2 * 2 * 3 * 3)
            e3 = [row for row in run_rows if row["analysis"] == "E3_budget_policy"]
            self.assertTrue(e3)
            self.assertTrue(all(row["mean_selected_tasks"] != "" for row in e3))
            self.assertTrue(all(row["delta_mean_training_token_burden"] != "" for row in e3))
            self.assertTrue(all(row["reference"] == "global_best_subset" for row in e3))
            self.assertTrue(all(float(row["delta_auroc"]) == 0 for row in e3
                                if row["policy"] == "global_best_subset"))
            with (complete_out / "task_selection_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                frequencies = list(csv.DictReader(handle))
            self.assertTrue(all(row["status"] == "complete" for row in frequencies))
            self.assertTrue(any(float(row["mean_frequency_across_seeds"]) == 0 for row in frequencies))
            with (complete_out / "condition_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                summaries = list(csv.DictReader(handle))
            self.assertTrue(summaries)
            self.assertTrue(all(row["status"] == "complete" for row in summaries))
            self.assertTrue(all(row["auroc_mean_across_seeds"] != "" for row in summaries
                                if row["analysis"] == "E2_strategy"))

            incomplete_out = root / "incomplete"
            incomplete = summarize_factorial(
                fit_root, incomplete_out, expected_seeds=(11, 12, 13), expected_runs=3,
                random_draws=16, random_seed=7, fixed_task="task_a",
                expected_conditions=(condition,),
            )
            self.assertEqual(incomplete["status"], "incomplete")
            with (incomplete_out / "condition_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                summaries = list(csv.DictReader(handle))
            self.assertTrue(summaries)
            self.assertTrue(all(row["status"] == "incomplete" for row in summaries))
            self.assertTrue(all(row["auroc_mean_across_seeds"] == "" for row in summaries))

            missing_condition = {"base_strength": 0.7, "added_type": "duplicated",
                                 "quantity": True, "positive_fraction": 0.3}
            missing_out = root / "missing_condition"
            missing = summarize_factorial(
                fit_root, missing_out, expected_seeds=(11, 12), expected_runs=4,
                random_draws=16, random_seed=7, fixed_task="task_a",
                expected_conditions=(condition, missing_condition),
            )
            self.assertEqual(missing["status"], "incomplete")
            with (missing_out / "condition_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                summaries = list(csv.DictReader(handle))
            missing_rows = [row for row in summaries if row["base_strength"] == "0.7"]
            self.assertTrue(missing_rows)
            self.assertTrue(all(row["status"] == "incomplete" for row in missing_rows))


if __name__ == "__main__":
    unittest.main()
