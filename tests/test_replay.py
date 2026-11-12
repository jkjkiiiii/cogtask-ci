import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cogtask_ci.replay import (
    apply_bundle,
    auc_rank,
    load_bundle,
    pair_key,
    validate_bundle,
)


def _fold(repeat, outer_fold, test_subjects, test_labels, test_scores,
          inner_subjects=None, inner_labels=None, inner_scores=None,
          token_counts=None):
    inner_subjects = inner_subjects or [f"train-{repeat}-{outer_fold}-{i}" for i in range(4)]
    inner_labels = inner_labels or [0, 0, 1, 1]
    inner_scores = inner_scores or [
        [0.1, 0.1, 0.9],
        [0.2, 0.2, 0.8],
        [0.8, 0.8, 0.2],
        [0.9, 0.9, 0.1],
    ]
    token_counts = token_counts or [[10, 10, 10] for _ in inner_subjects]
    return {
        "repeat": repeat,
        "outer_fold": outer_fold,
        "inner_subjects": list(inner_subjects),
        "inner_labels": list(inner_labels),
        "inner_scores": inner_scores,
        "train_token_counts": token_counts,
        "test_subjects": list(test_subjects),
        "test_labels": list(test_labels),
        "test_scores": test_scores,
    }


def _bundle():
    # Two tasks are perfect and one is reversed. The task order is deliberately
    # non-lexical to exercise ID-based tie breaking.
    folds = []
    for repeat in range(2):
        folds.append(_fold(
            repeat, 0, ["s0", "s1"], [0, 1],
            [[0.1, 0.1, 0.9], [0.9, 0.9, 0.1]],
            inner_scores=[
                [0.1, 0.1, 0.9], [0.2, 0.2, 0.8],
                [0.8, 0.8, 0.2], [0.9, 0.9, 0.1],
            ],
        ))
        folds.append(_fold(
            repeat, 1, ["s2", "s3"], [0, 1],
            [[0.2, 0.3, 0.8], [0.8, 0.7, 0.2]],
            inner_scores=[
                [0.1, 0.2, 0.9], [0.2, 0.3, 0.8],
                [0.8, 0.4, 0.2], [0.9, 0.5, 0.1],
            ],
        ))
    return {"schema_version": 1, "tasks": ["zeta", "alpha", "middle"], "folds": folds}


class ReplayValidation(unittest.TestCase):
    def test_one_fold_train_test_bundle_and_file_loader(self):
        # The one outer-test fold can be a holdout set; inner training subjects
        # need not belong to the outer-test population.
        bundle = {
            "schema_version": 1,
            "tasks": ["one", "two"],
            "folds": [_fold(0, 0, ["test-a", "test-b", "test-c", "test-d"],
                            [0, 0, 1, 1],
                            [[0.1, 0.2], [0.2, 0.3], [0.8, 0.7], [0.9, 0.8]],
                            inner_subjects=[f"train-{i}" for i in range(8)],
                            inner_labels=[0, 0, 0, 0, 1, 1, 1, 1],
                            inner_scores=[[i / 10, i / 10] for i in range(8)],
                            token_counts=[[i, i + 1] for i in range(8)])],
        }
        clean = validate_bundle(bundle)
        self.assertEqual(clean["folds"][0]["inner_subjects"][0], "train-0")
        self.assertEqual(clean["folds"][0]["test_subjects"][0], "test-a")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.json"
            path.write_text(json.dumps(bundle), encoding="utf-8")
            loaded = load_bundle(path)
        self.assertEqual(loaded["tasks"], bundle["tasks"])
        self.assertEqual(len(loaded["folds"]), 1)

    def test_rejects_duplicates_misalignment_and_leakage(self):
        source = _bundle()
        bad = json.loads(json.dumps(source))
        bad["tasks"].append("alpha")
        with self.assertRaisesRegex(ValueError, "duplicate task"):
            validate_bundle(bad)

        bad = _bundle()
        bad["folds"][0]["inner_subjects"][1] = bad["folds"][0]["inner_subjects"][0]
        with self.assertRaisesRegex(ValueError, "duplicate subject"):
            validate_bundle(bad)

        bad = _bundle()
        bad["folds"][0]["inner_subjects"][0] = "s0"
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_bundle(bad)

        bad = _bundle()
        bad["folds"][0]["test_scores"][0][0] = 1.01
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            validate_bundle(bad)

        bad = _bundle()
        bad["folds"][0]["inner_scores"] = [[0.1, 0.2]]
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_bundle(bad)

    def test_rejects_outer_coverage_and_label_changes(self):
        bad = _bundle()
        bad["folds"][1]["test_subjects"] = ["s0", "s3"]
        bad["folds"][1]["test_labels"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "duplicate subject"):
            validate_bundle(bad)

        bad = _bundle()
        # Same subject appears with a different label in another repeat.
        bad["folds"][2]["inner_subjects"][0] = "s2"
        bad["folds"][2]["inner_labels"][0] = 1
        with self.assertRaisesRegex(ValueError, "inconsistent labels"):
            validate_bundle(bad)

        bad = _bundle()
        bad["folds"][2]["test_subjects"] = ["other-0", "other-1"]
        with self.assertRaisesRegex(ValueError, "identical outer subject sets"):
            validate_bundle(bad)


class ReplaySelection(unittest.TestCase):
    def test_rankdata_ties_and_new_tolerance_burden_tiebreak(self):
        y = np.array([0, 0, 1, 1])
        first = np.array([0.1, 0.2, 0.8, 0.9])
        monotone = np.array([0.01, 0.04, 0.64, 0.81])
        self.assertEqual(auc_rank(y, first), auc_rank(y, monotone))

        # `alpha` and `zeta` have the exact same rank pattern. All singles are
        # perfect; `middle` is uninformative.
        fold = _fold(
            0, 0, ["test-0", "test-1", "test-2", "test-3"], [0, 0, 1, 1],
            [[0.1, 0.01, 0.5], [0.2, 0.04, 0.5], [0.8, 0.64, 0.5], [0.9, 0.81, 0.5]],
            inner_scores=[
                [0.1, 0.01, 0.5], [0.2, 0.04, 0.5],
                [0.8, 0.64, 0.5], [0.9, 0.81, 0.5],
            ],
            token_counts=[[1, 1, 2] for _ in range(4)],
        )
        bundle = {"schema_version": 1, "tasks": ["zeta", "alpha", "middle"], "folds": [fold]}
        result = apply_bundle(bundle, "middle", random_draws=4, seed=9)
        selected = result["selections"][0]
        self.assertEqual(selected["global_best_single"], "alpha")
        self.assertEqual(selected["global_best_subset"], ["alpha"])
        self.assertEqual(selected["e3_tolerance_selections"]["0.00"]["tasks"], ["alpha"])
        self.assertEqual(selected["e2_max_auc_pair"], ["alpha", "zeta"])
        self.assertEqual(selected["e2_max_gain_pair"], ["alpha", "zeta"])

        # Lower burden is the tie-break for global singles, global subsets,
        # pair references, and E3. Lexical IDs break ties only after burden.
        fold["train_token_counts"] = [[1, 100, 2] for _ in range(4)]
        result = apply_bundle(bundle, "middle", random_draws=0, seed=9)
        selected = result["selections"][0]
        self.assertEqual(selected["global_best_single"], "zeta")
        self.assertEqual(selected["global_best_subset"], ["zeta"])
        self.assertEqual(selected["e3_tolerance_selections"]["0.00"]["tasks"], ["zeta"])
        self.assertEqual(selected["e2_max_auc_pair"], ["middle", "zeta"])
        self.assertEqual(selected["e2_max_gain_pair"], ["middle", "zeta"])
        self.assertEqual(result["references"]["subset_train_selected_best_single"][
            pair_key(("alpha", "zeta"))]["0:0"], "zeta")

    def test_predictions_are_fold_local_and_random_policies_reproducible(self):
        bundle = _bundle()
        first = apply_bundle(bundle, "alpha", random_draws=12, seed=123)
        second = apply_bundle(bundle, "alpha", random_draws=12, seed=123)
        self.assertEqual(first["predictions"], second["predictions"])
        self.assertEqual(first["selections"], second["selections"])
        self.assertEqual(len(first["selections"]), 4)
        self.assertEqual(len(first["predictions"][pair_key(("alpha", "zeta"))][0]), 4)
        for repeat in (0, 1):
            self.assertEqual(set(first["predictions"]["global_best_single"][repeat]),
                             {"s0", "s1", "s2", "s3"})
            self.assertEqual(set(first["labels"][repeat]), {"s0", "s1", "s2", "s3"})
        self.assertEqual(first["references"]["fixed_task"], "alpha")
        self.assertEqual(first["predictions"]["fixed_task"], first["predictions"]["task::alpha"])
        self.assertIn("random_pair::0011", first["predictions"])
        pairs = [("alpha", "middle"), ("alpha", "zeta"), ("middle", "zeta")]
        expected = {}
        for draw in range(12):
            expected[draw] = np.random.default_rng(123 + draw).integers(0, len(pairs), size=4)
        for fold_position, row in enumerate(first["selections"]):
            for draw in range(12):
                self.assertEqual(row["random_pairs"][str(draw)], list(pairs[expected[draw][fold_position]]))
        no_random = apply_bundle(bundle, "alpha", random_draws=0, seed=123)
        self.assertFalse(any(name.startswith("random_pair::") for name in no_random["predictions"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
