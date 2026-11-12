import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cogtask_ci.data import load_dataset, records_to_jsonl
from cogtask_ci.evaluation import Config
from cogtask_ci.factorial import (
    ADDED_TYPES,
    FACTORIAL_SEEDS,
    TASKS,
    factorial_conditions,
    generate_factorial_records,
    run_factorial,
)


class SyntheticFactorial(unittest.TestCase):
    def test_fixed_grid_and_seed_list(self):
        conditions = factorial_conditions()
        self.assertEqual(len(conditions), 36)
        self.assertEqual({row["added_type"] for row in conditions}, set(ADDED_TYPES))
        self.assertEqual({row["quantity"] for row in conditions}, {False, True})
        self.assertEqual({row["positive_fraction"] for row in conditions}, {0.3, 0.5})
        self.assertEqual(len(FACTORIAL_SEEDS), 30)
        self.assertEqual((FACTORIAL_SEEDS[0], FACTORIAL_SEEDS[-1]), (20261112, 20261141))

    def test_generated_rows_are_reproducible_and_have_exact_prevalence(self):
        kwargs = dict(subjects=200, base_strength=0.70, added_type="complementary",
                      quantity=True, positive_fraction=0.30, seed=20261112)
        first = generate_factorial_records(**kwargs)
        second = generate_factorial_records(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 600)
        self.assertEqual(sum(r.label for r in first[::3]), 60)
        self.assertEqual({r.task for r in first}, set(TASKS))
        self.assertEqual(records_to_jsonl(first), records_to_jsonl(second))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "synthetic.jsonl"
            path.write_bytes(records_to_jsonl(first))
            loaded = load_dataset(path)
            self.assertEqual(len(loaded), len(first))
            self.assertEqual(loaded[0], first[0])

    def test_duplicate_condition_copies_text_and_length(self):
        rows = generate_factorial_records(subjects=40, base_strength=0.60,
                                          added_type="duplicated", quantity=True,
                                          positive_fraction=0.5, seed=13)
        by_key = {(r.subject_id, r.task): r for r in rows}
        for subject in sorted({r.subject_id for r in rows}):
            a, b, c = (by_key[subject, task] for task in TASKS)
            self.assertEqual(a.segments, b.segments)
            self.assertEqual(a.segments, c.segments)
            self.assertIn(a.n_tokens, {12, 48})

    def test_quantity_off_is_equal_length_and_total_ratio_is_exact(self):
        rows = generate_factorial_records(subjects=40, base_strength=0.80,
                                          added_type="no_information", quantity=False,
                                          positive_fraction=0.30, seed=17)
        self.assertEqual(sum(r.label for r in rows[::3]), 12)
        self.assertEqual({r.n_tokens for r in rows}, {12})

    def test_runner_caches_bundles_and_resumes_by_hash(self):
        conditions = [factorial_conditions()[0]]
        config = Config(outer_folds=2, inner_folds=2, repeats=1, seed=42, bootstrap=1)
        calls = []

        def fake_run_arm(rows, arm, config, assignments, logger, replay_collector=None,
                         original_token_counts=None):
            calls.append(arm)
            tasks = sorted({record.task for record in rows})
            replay_collector[arm] = {"schema_version": 1, "tasks": tasks, "folds": []}
            return [], [], []

        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "runs"
            with patch("cogtask_ci.factorial.run_arm", fake_run_arm):
                first = run_factorial(output, seed_values=(42,), conditions=conditions,
                                      subjects=40, config=config)
                second = run_factorial(output, seed_values=(42,), conditions=conditions,
                                       subjects=40, config=config)
            self.assertEqual(calls, ["text", "quantity"])
            self.assertFalse(first[0]["resumed"])
            self.assertTrue(second[0]["resumed"])
            result = json.loads(next(output.glob("*/result.json")).read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "complete")
            for arm in ("text", "quantity"):
                self.assertTrue((next(output.iterdir()) / (arm + "_bundle.json")).is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
