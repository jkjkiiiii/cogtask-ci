"""Regression cases identified during the pre-publication review."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from cogtask_ci.data import Record, import_csv
from cogtask_ci.evaluation import summarize
from cogtask_ci.model import fit_predict


class ReviewRegressions(unittest.TestCase):
    def test_duplicate_csv_headers(self):
        with tempfile.TemporaryDirectory() as td:
            source, target = Path(td)/'bad.csv', Path(td)/'out.jsonl'
            source.write_text('subject_id,task,label,text,text\na,t,0,one,two\n')
            with self.assertRaisesRegex(ValueError, 'unique'):
                import_csv(source, target)
            self.assertFalse(target.exists())

    def test_failed_fit_and_nonfinite_predictions(self):
        train = [Record(str(i), 'task', i % 2, (('shared', 'word'),)) for i in range(4)]
        def nonconverged(*args, **kwargs):
            warnings.warn('test nonconvergence', ConvergenceWarning)
        with patch('cogtask_ci.model.LogisticRegression.fit', side_effect=nonconverged):
            with self.assertRaisesRegex(ValueError, 'did not converge'):
                fit_predict(train, train[:1], 'text', 42)
        with patch('cogtask_ci.model.LogisticRegression.predict_proba', return_value=np.array([[np.nan, np.nan]])):
            with self.assertRaisesRegex(ValueError, 'Invalid predicted'):
                fit_predict(train, train[:1], 'text', 42)

    def test_repeat_auroc_is_not_auroc_of_mean_scores(self):
        rows = []
        for rep, scores in enumerate(([.4, .6], [1., 0.])):
            for k in (1, 2):
                for i in (0, 1):
                    rows.append(dict(arm='text', repeat=rep, k=k, subject_id=str(i), label=i, score=scores[i] if k == 1 else float(i)))
        values = summarize(rows, 5, 42)
        self.assertEqual(values[0]['auroc'], .5)
        self.assertEqual(values[1]['auroc'], 1)
        self.assertEqual(values[1]['delta_vs_single'], .5)
        self.assertEqual(values[1]['delta_ci_low'], .5)
        self.assertEqual(values[1]['delta_ci_high'], .5)

    def test_render_preserves_backend(self):
        import subprocess, sys
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def write(name, rows):
                with (root/name).open('w', newline='', encoding='utf-8') as f:
                    writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            write('summary.csv', [dict(arm='text', k=1, auroc=.5, ci_low=.4, ci_high=.6, delta_vs_single=0, delta_ci_low=0, delta_ci_high=0)])
            write('selections.csv', [dict(arm='text', k=1, selected_tasks='["task"]')])
            write('contrasts.csv', [dict(left_arm='text', right_arm='quantity', k=1, auroc_difference=0)])
            (root/'coverage.json').write_text('{}')
            code = "import sys,matplotlib; matplotlib.use('svg'); from cogtask_ci import render_report; render_report(sys.argv[1]); assert matplotlib.get_backend().lower() == 'svg'"
            subprocess.run([sys.executable, '-c', code, str(root)], check=True)
            self.assertTrue((root/'report.html').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
