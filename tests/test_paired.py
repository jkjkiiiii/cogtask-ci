import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from cogtask_ci.paired import paired_subject_bootstrap, stratified_bootstrap_auc


class PairedBootstrap(unittest.TestCase):
    def test_auc_matches_independent_reference_with_ties(self):
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        scores = np.array([0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8])
        result = stratified_bootstrap_auc(labels, scores, n_boot=300, seed=51, batch_size=37)
        self.assertEqual(result["auroc"], roc_auc_score(labels, scores))
        self.assertEqual(result["n_repeats"], 1)
        self.assertEqual(result["n_subjects"], len(labels))
        self.assertEqual(result["bootstrap"].shape, (300,))
        self.assertLessEqual(result["ci_low"], result["auroc"])
        self.assertGreaterEqual(result["ci_high"], result["auroc"])

    def test_repeated_measures_reuse_subject_draws(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        ascending = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        descending = 1.0 - ascending
        result = stratified_bootstrap_auc(
            labels,
            np.stack([ascending, descending]),
            n_boot=500,
            seed=7,
            batch_size=64,
        )
        # For every shared subject resample, reversed rankings complement one
        # another, including ties created by duplicated bootstrap subjects.
        np.testing.assert_allclose(result["bootstrap"], 0.5, rtol=0, atol=1e-15)
        self.assertEqual(result["auroc"], 0.5)
        self.assertEqual(result["n_repeats"], 2)

    def test_paired_auc_and_brier_differences(self):
        labels = np.array([0, 0, 1, 1, 0, 1])
        a = np.array([
            [0.1, 0.2, 0.8, 0.9, 0.4, 0.7],
            [0.2, 0.3, 0.7, 0.8, 0.4, 0.6],
        ])
        b = a * 0.8 + 0.1
        result = paired_subject_bootstrap(labels, a, b, n_boot=400, seed=19, batch_size=55)
        expected_auc_a = np.mean([roc_auc_score(labels, row) for row in a])
        expected_auc_b = np.mean([roc_auc_score(labels, row) for row in b])
        expected_brier_a = np.mean((a - labels[None, :]) ** 2)
        expected_brier_b = np.mean((b - labels[None, :]) ** 2)
        self.assertEqual(result["auroc_a"], expected_auc_a)
        self.assertEqual(result["auroc_b"], expected_auc_b)
        self.assertAlmostEqual(result["brier_difference"], expected_brier_a - expected_brier_b)
        self.assertEqual(result["bootstrap_auroc_difference"].shape, (400,))
        self.assertEqual(result["bootstrap_brier_difference"].shape, (400,))

    def test_identical_methods_have_zero_paired_differences(self):
        labels = [0, 0, 1, 1]
        scores = np.array([[0.1, 0.3, 0.7, 0.9], [0.2, 0.4, 0.6, 0.8]])
        result = paired_subject_bootstrap(labels, scores, scores.copy(), n_boot=120, seed=4)
        self.assertEqual(result["auroc_difference"], 0.0)
        self.assertEqual(result["brier_difference"], 0.0)
        np.testing.assert_array_equal(result["bootstrap_auroc_difference"], 0.0)
        np.testing.assert_array_equal(result["bootstrap_brier_difference"], 0.0)

    def test_validation_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "both classes"):
            stratified_bootstrap_auc([0, 0], [0.1, 0.2], n_boot=5)
        with self.assertRaisesRegex(ValueError, "\[0, 1\]"):
            stratified_bootstrap_auc([0, 1], [0.1, 1.2], n_boot=5)
        with self.assertRaisesRegex(ValueError, "identical repeat/subject shapes"):
            paired_subject_bootstrap([0, 0, 1, 1], np.ones(4) * 0.5,
                                     np.ones((2, 4)) * 0.5, n_boot=5)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            stratified_bootstrap_auc([0, 1], [0.1, 0.9], n_boot=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
