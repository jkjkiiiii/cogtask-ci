import unittest
from cogtask_ci.showcase import SCENARIOS, scenario_rows


class SyntheticScenarios(unittest.TestCase):
    def test_prespecified_design_and_reproducibility(self):
        for name in SCENARIOS:
            rows = scenario_rows(name, 24, 20261103)
            self.assertEqual(rows, scenario_rows(name, 24, 20261103))
            self.assertEqual(len(rows), 72)
            self.assertEqual({r['label'] for r in rows}, {0, 1})
            lengths = {sum(map(len, r['segments'])) for r in rows}
            self.assertTrue(lengths <= {20, 80} if name == 'content_quantity' else lengths == {20})
        redundant = scenario_rows('redundant', 24)
        for i in range(0, len(redundant), 3):
            self.assertEqual(redundant[i]['segments'], redundant[i+1]['segments'])
            self.assertEqual(redundant[i]['segments'], redundant[i+2]['segments'])

    def test_null_has_same_sentences_with_either_label(self):
        rows = scenario_rows('null', 160)
        for label in (0, 1):
            observed = {tuple(r['segments'][1]) for r in rows if r['label'] == label}
            self.assertEqual(len(observed), 2)
        with self.assertRaises(ValueError):
            scenario_rows('unknown')
        with self.assertRaises(ValueError):
            scenario_rows('null', 25)

    def test_content_and_amount_are_not_the_same_cue(self):
        rows = scenario_rows('content_quantity', 160)
        for label in (0, 1):
            combinations = {(tuple(r['segments'][1]), sum(map(len, r['segments']))) for r in rows if r['label'] == label}
            self.assertEqual(len(combinations), 4)


if __name__ == '__main__':
    unittest.main()
