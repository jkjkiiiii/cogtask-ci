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
    def test_driver_commit_is_recorded_and_required_for_resume(self):
        with tempfile.TemporaryDirectory() as td:
            context = {"driver_git_commit": "a" * 40}
            run_factorial(td, subjects=40, seed_values=(42,),
                          conditions=factorial_conditions()[:1], provenance_context=context)
            result = json.loads(next(Path(td).glob('*/result.json')).read_text(encoding='utf-8'))
            self.assertEqual(result['driver_git_commit'], context['driver_git_commit'])
            with self.assertRaisesRegex(ValueError, 'provenance differs'):
                run_factorial(td, subjects=40, seed_values=(42,),
                              conditions=factorial_conditions()[:1],
                              provenance_context={"driver_git_commit": "b" * 40})

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
            labels = {record.subject_id: record.label for record in rows}
            folds = []
            for outer_fold in range(config.outer_folds):
                test_subjects = sorted(s for s in labels if assignments[0][s] == outer_fold)
                inner_subjects = sorted(s for s in labels if assignments[0][s] != outer_fold)
                folds.append({
                    "repeat": 0,
                    "outer_fold": outer_fold,
                    "inner_subjects": inner_subjects,
                    "inner_labels": [labels[s] for s in inner_subjects],
                    "inner_scores": [[0.4 + 0.2 * labels[s]] * len(tasks) for s in inner_subjects],
                    "train_token_counts": [[12] * len(tasks) for _ in inner_subjects],
                    "test_subjects": test_subjects,
                    "test_labels": [labels[s] for s in test_subjects],
                    "test_scores": [[0.4 + 0.2 * labels[s]] * len(tasks) for s in test_subjects],
                })
            replay_collector[arm] = {"schema_version": 1, "tasks": tasks, "folds": folds}
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
            run_dir = next(output.iterdir())
            result_path = run_dir / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "complete")
            for arm in ("text", "quantity"):
                self.assertTrue((run_dir / (arm + "_bundle.json")).is_file())

            # Simulate an interrupted legacy checkpoint between bundle write
            # and hash-manifest update. The orphan is never trusted: regenerate
            # it from the deterministic inputs and retain the valid sibling.
            result["status"] = "running"
            result["bundle_sha256"].pop("text")
            result_path.write_text(json.dumps(result), encoding="utf-8")
            (run_dir / "text_bundle.json").write_bytes(b"corrupt untracked orphan")
            (run_dir / "quantity_bundle.json.tmp").write_bytes(b"partial temporary")
            calls.clear()
            with patch("cogtask_ci.factorial.run_arm", fake_run_arm):
                resumed = run_factorial(output, seed_values=(42,), conditions=conditions,
                                        subjects=40, config=config)
            self.assertEqual(calls, ["text"])
            self.assertTrue(resumed[0]["resumed"])
            self.assertEqual(json.loads(result_path.read_text(encoding="utf-8"))["status"], "complete")
            self.assertFalse((run_dir / "quantity_bundle.json.tmp").exists())
            from cogtask_ci.replay import load_bundle
            self.assertEqual(load_bundle(run_dir / "text_bundle.json")["tasks"],
                             ["task_a", "task_b", "task_c"])

            # Simulate the new journal order: hash committed, then process
            # stopped before the atomic bundle replacement. Refit and require
            # the deterministic output to match the recorded expected hash.
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["status"] = "running"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            (run_dir / "text_bundle.json").unlink()
            calls.clear()
            with patch("cogtask_ci.factorial.run_arm", fake_run_arm):
                run_factorial(output, seed_values=(42,), conditions=conditions,
                              subjects=40, config=config)
            self.assertEqual(calls, ["text"])
            self.assertTrue((run_dir / "text_bundle.json").is_file())

            # A bundle whose recorded hash no longer matches is rejected,
            # even though the runner could overwrite it with a fresh fit.
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["status"] = "running"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            (run_dir / "text_bundle.json").write_bytes(b"altered tracked artifact")
            calls.clear()
            with patch("cogtask_ci.factorial.run_arm", fake_run_arm):
                with self.assertRaisesRegex(ValueError, "Changed partial bundle"):
                    run_factorial(output, seed_values=(42,), conditions=conditions,
                                  subjects=40, config=config)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
