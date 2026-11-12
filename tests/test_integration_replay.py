import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cogtask_ci import Config, evaluate
from cogtask_ci.cli import _paired_policy_difference, main
from cogtask_ci.demo import generate_demo
from cogtask_ci.data import Record, records_to_jsonl
from cogtask_ci.evaluation import validate_total_token_budget
from cogtask_ci.replay import apply_bundle, load_bundle


class ReplayIntegration(unittest.TestCase):
    def test_evaluate_writes_per_arm_bundles_and_replay_cli(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = generate_demo(root / "input.jsonl", subjects=24, seed=17)
            config = Config(outer_folds=2, inner_folds=2, repeats=1, bootstrap=3, seed=42)

            def constant_scores(train, test, arm, seed):
                return np.full(len(test), 0.5, dtype=float)

            with patch("cogtask_ci.evaluation.fit_predict", constant_scores):
                evaluate(source, root / "run", config)
            bundle_path = root / "run" / "replay_bundles" / "text.json"
            bundle = load_bundle(bundle_path)
            self.assertEqual(bundle["schema_version"], 1)
            self.assertEqual(bundle["tasks"], ["task_a", "task_b", "task_redundant"])
            self.assertEqual(len(bundle["folds"]), config.outer_folds)
            first = bundle["folds"][0]
            self.assertEqual(len(first["inner_subjects"]), len(first["inner_scores"]))
            self.assertEqual(len(first["inner_subjects"]), len(first["train_token_counts"]))
            self.assertEqual(len(first["test_subjects"]), len(first["test_scores"]))
            replayed = apply_bundle(bundle, fixed_task=bundle["tasks"][0], random_draws=0, seed=42)
            self.assertIn("global_best_subset", replayed["predictions"])

            replay_path = root / "replay.json"
            self.assertEqual(main(["replay", str(bundle_path), "--fixed-task", bundle["tasks"][0],
                                   "--random-draws", "0", "--output", str(replay_path)]), 0)
            saved = json.loads(replay_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["bundle_sha256"], hashlib.sha256(bundle_path.read_bytes()).hexdigest())
            comparison_path = root / "comparison.json"
            self.assertEqual(main(["compare", str(bundle_path), str(bundle_path),
                                   "--left-fixed-task", bundle["tasks"][0],
                                   "--right-fixed-task", bundle["tasks"][0],
                                   "--policy", "global_best_subset", "--random-draws", "0",
                                   "--bootstrap", "20", "--seed", "13",
                                   "--output", str(comparison_path)]), 0)
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            self.assertEqual(comparison["auroc_difference"], 0.0)
            self.assertEqual(comparison["ci_low"], 0.0)
            self.assertEqual(comparison["ci_high"], 0.0)
            self.assertEqual(comparison["brier_difference"], 0.0)

    def test_paired_difference_requires_same_subjects(self):
        base = {
            "predictions": {"global_best_subset": {0: {"a": 0.1, "b": 0.2, "c": 0.8, "d": 0.9}}},
            "labels": {0: {"a": 0, "b": 0, "c": 1, "d": 1}},
        }
        self.assertEqual(_paired_policy_difference(base, base, "global_best_subset", 20, 42)["auroc_difference"], 0)
        mismatched = {"predictions": {"global_best_subset": {0: {"a": 0.1}}},
                      "labels": {0: {"a": 0}}}
        with self.assertRaisesRegex(ValueError, "identical subjects"):
            _paired_policy_difference(base, mismatched, "global_best_subset", 20, 42)

    def test_budget_contract(self):
        positional = Config(5, 3, 1, 42, 2000, None, True)
        self.assertTrue(positional.lowercase)
        self.assertIsNone(positional.token_budget_per_task)
        for budget, n_tasks in ((60, 3), (61, 4), (2520, 10)):
            if budget == 61 and n_tasks == 4:
                with self.assertRaisesRegex(ValueError, "divisible"):
                    validate_total_token_budget(budget, n_tasks)
            else:
                validate_total_token_budget(budget, n_tasks)
        with self.assertRaises(ValueError):
            Config(token_budget_per_task=20, total_token_budget=60).validate()

    def test_total_budget_refits_each_k_and_keeps_k_tagged_bundles(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = []
            for i in range(30):
                label = i % 2
                for task in ("task_a", "task_b", "task_c"):
                    rows.append(Record(f"S{i:03d}", task, label,
                                        (("signal_x" if label else "signal_y",) + ("shared",) * 99,)))
            source = root / "input.jsonl"
            source.write_bytes(records_to_jsonl(rows))
            seen_task_lengths = set()

            def constant_scores(train, test, arm, seed):
                if arm == "fixed_total_text":
                    seen_task_lengths.update(record.n_tokens for record in train + test)
                return np.full(len(test), 0.5, dtype=float)

            cfg = Config(outer_folds=3, inner_folds=2, repeats=1, bootstrap=3,
                         total_token_budget=60, seed=42)
            with patch("cogtask_ci.evaluation.fit_predict", constant_scores):
                evaluate(source, root / "run", cfg)
            self.assertEqual(seen_task_lengths, {20, 30, 60})
            bundles = sorted((root / "run" / "replay_bundles").glob("fixed_total_text_total60_k*.json"))
            self.assertEqual(len(bundles), 3)
            for k, path in enumerate(bundles, 1):
                bundle = load_bundle(path)
                self.assertEqual(bundle["budget"], {"mode": "total", "total_tokens": 60,
                                                     "per_task_tokens": 60 // k, "fixed_k": k})
                with self.assertRaisesRegex(ValueError, "k-specific analysis"):
                    apply_bundle(bundle, fixed_task="task_a", random_draws=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
