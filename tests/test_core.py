import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from cogtask_ci import Config, evaluate, load_dataset, validate_dataset, import_csv
from cogtask_ci.data import crop
from cogtask_ci.demo import generate_demo
from cogtask_ci.evaluation import select_subsets, summarize, contrasts
from cogtask_ci.model import word_ngrams, char_ngrams, vectorize


class Contracts(unittest.TestCase):
    def test_gaps_and_train_vocabulary(self):
        train = [[['a', 'cat'], ['dog']], [['a', 'cat'], ['dog']]]
        x, z, w, c = vectorize(train, [[['heldoutonly']]])
        self.assertNotIn('b:cat dog', w.vocabulary_)
        self.assertIn('u:a', w.vocabulary_)
        self.assertNotIn('u:heldoutonly', w.vocabulary_)
        self.assertEqual(crop((('a', 'b'), ('c', 'd')), 3), (('a', 'b'), ('c',)))
        reference = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), lowercase=False).build_analyzer()
        for token in ['a', 'cat', "I've", 'über', '认知']:
            self.assertEqual(char_ngrams([[token]]), reference(token))

    def test_invalid_data(self):
        with tempfile.TemporaryDirectory() as td:
            p = generate_demo(Path(td) / 'data.jsonl', 24)
            lines = p.read_text().splitlines()
            for bad in [lines + [lines[0]], [lines[0].replace('"label": 0', '"label": true')] + lines[1:], [lines[0].replace('"shared"', '"two words"', 1)] + lines[1:]]:
                p.write_text('\n'.join(bad))
                with self.assertRaises(ValueError):
                    load_dataset(p)
            p.write_text('\n'.join(lines))
            r = json.loads(lines[1]); r['label'] = 1
            p.write_text('\n'.join([lines[0], json.dumps(r)] + lines[2:]))
            with self.assertRaises(ValueError):
                load_dataset(p)

    def test_selection_and_manual_statistics(self):
        y = [0, 0, 1, 1]
        scores = np.array([[.1, .9], [.2, .8], [.8, .2], [.9, .1]])
        selected = select_subsets(y, scores, ['a', 'b'])
        self.assertEqual(selected[1][0], (0,))
        self.assertEqual(selected[2][0], (0, 1))
        tied = select_subsets(y, np.ones((4, 2)) * .5, ['a', 'b'])
        self.assertEqual(tied[1][0], (0,))
        rows = [dict(arm='text', repeat=0, subject_id=str(i), label=y[i], k=k, score=float(scores[i, 0]) if k == 1 else .5) for k in (1, 2) for i in range(4)]
        summary = summarize(rows, 20, 42)
        self.assertEqual(summary[0]['auroc'], 1)
        self.assertEqual(summary[1]['auroc'], .5)
        self.assertEqual(summary[1]['delta_vs_single'], -.5)
        self.assertEqual(summary[0]['delta_ci_low'], 0)
        self.assertEqual(summary[0]['delta_ci_high'], 0)
        quantity = [dict(r, arm='quantity', score=.5) for r in rows]
        self.assertEqual(contrasts(rows + quantity, 20, 42)[0]['auroc_difference'], .5)
        paired = contrasts(rows + quantity, 20, 42)
        self.assertEqual(paired[0]['gain_difference'], 0)
        self.assertEqual(paired[1]['gain_difference'], -.5)

    def test_csv_and_config(self):
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / 'text.csv', Path(td) / 'out.jsonl'
            src.write_text('subject_id,task,label,text\na,one,0,One token\nb,one,1,Other token\n', encoding='utf-8')
            import_csv(src, dst)
            self.assertEqual(load_dataset(dst)[0].segments, (('One', 'token'),))
            with self.assertRaises(ValueError):
                import_csv(src, dst)
            src.write_text('subject_id,task,label,text\na,one,0\n')
            with self.assertRaisesRegex(ValueError, 'missing or extra cells'):
                import_csv(src, Path(td)/'malformed.jsonl')
        for cfg in [Config(inner_folds=1), Config(repeats=0), Config(bootstrap=0), Config(token_budget=True), Config(lowercase=1)]:
            with self.assertRaises(ValueError):
                cfg.validate()


class EndToEnd(unittest.TestCase):
    def test_backend_and_seed_edge(self):
        import subprocess, sys
        code = "import matplotlib; matplotlib.use('svg'); import cogtask_ci; import cogtask_ci.report; assert matplotlib.get_backend().lower() == 'svg'"
        subprocess.run([sys.executable, '-c', code], check=True)
        from cogtask_ci.evaluation import split_seed
        cfg = Config(seed=2**31-1)
        self.assertTrue(0 <= split_seed(cfg, 999999, 999999) < 2**32)

    def test_repeated_cv_and_invalid_splits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = generate_demo(root/'input.jsonl', 24)
            cfg = Config(outer_folds=2, inner_folds=2, repeats=2, bootstrap=3)
            result = evaluate(source, root/'run', cfg)
            self.assertTrue(all(r['n_repeats'] == 2 and r['n_subjects'] == 24 for r in result))
            with (root/'run'/'predictions.csv').open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            keys = [(r['arm'], r['repeat'], r['subject_id'], r['k']) for r in rows]
            self.assertEqual(len(keys), len(set(keys)))
            splits = root/'bad.csv'
            splits.write_text('repeat,subject_id,outer_fold\n0,unknown,0\n')
            with self.assertRaises(ValueError):
                evaluate(source, root/'invalid', cfg, splits)
            self.assertFalse((root/'invalid').exists())

    def test_split_isolation_reproducibility_and_fixed_population(self):
        from cogtask_ci.model import fit_predict as real_fit
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = generate_demo(root / 'input.jsonl', 30)
            # One short subject excluded only from the fixed-budget analysis.
            lines = [json.loads(s) for s in source.read_text().splitlines()]
            lines[0]['segments'] = [['shared', 'word']]
            source.write_text('\n'.join(json.dumps(r) for r in lines))
            original = source.read_bytes()
            cfg = Config(outer_folds=3, inner_folds=2, bootstrap=12, token_budget=20)
            calls = []
            def audited(train, test, arm, seed):
                tr, te = {r.subject_id for r in train}, {r.subject_id for r in test}
                self.assertFalse(tr & te)
                calls.append((tr, te))
                return real_fit(train, test, arm, seed)
            with patch('cogtask_ci.evaluation.fit_predict', audited):
                evaluate(source, root / 'first', cfg)
            self.assertTrue(calls)
            evaluate(source, root / 'second', cfg, root / 'first' / 'outer_splits.csv')
            self.assertEqual((root/'first'/'predictions.csv').read_bytes(), (root/'second'/'predictions.csv').read_bytes())
            self.assertEqual((root/'first'/'summary.csv').read_bytes(), (root/'second'/'summary.csv').read_bytes())
            self.assertEqual(source.read_bytes(), original)
            coverage = json.loads((root/'first'/'coverage.json').read_text())
            self.assertEqual(coverage['fixed_budget_subjects'], 29)
            with (root/'first'/'outer_splits.csv').open(encoding='utf-8-sig') as f:
                outer = {(int(r['repeat']), r['subject_id']): int(r['outer_fold']) for r in csv.DictReader(f)}
            with (root/'first'/'inner_split_audit.csv').open(encoding='utf-8-sig') as f:
                for r in csv.DictReader(f):
                    self.assertNotEqual(outer[int(r['repeat']), r['subject_id']], int(r['outer_fold']))
            with (root/'first'/'predictions.csv').open(encoding='utf-8-sig') as f:
                predictions = list(csv.DictReader(f))
            populations = {}
            for r in predictions:
                populations.setdefault((r['arm'], r['k']), set()).add(r['subject_id'])
            self.assertEqual(populations['fixed_text', '1'], populations['text_matched', '3'])
            self.assertEqual(populations['text', '1'], populations['text', '3'])
            for name in ['report.html', 'task_budget.png', 'task_budget.pdf', 'task_budget.svg', 'selection_frequency.csv', 'contrasts.csv', 'run.log', 'environment.json', 'provenance.json']:
                self.assertTrue((root/'first'/name).stat().st_size > 0)
            self.assertEqual(json.loads((root/'first'/'status.json').read_text())['status'], 'complete')
            with self.assertRaises(ValueError):
                evaluate(source, root/'first', cfg)

    def test_missing_and_failed_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = generate_demo(root/'input.jsonl', 30)
            lines = p.read_text().splitlines()
            p.write_text('\n'.join(lines[1:]))
            evaluate(p, root/'missing', Config(outer_folds=3, inner_folds=2, bootstrap=3))
            cov = json.loads((root/'missing'/'coverage.json').read_text())
            self.assertEqual(cov['complete_subjects'], 29)
            self.assertEqual(cov['incomplete_subjects'], 1)
            with self.assertRaises(ValueError):
                evaluate(p, root/'too_many_folds', Config(outer_folds=20))
            self.assertFalse((root/'too_many_folds').exists())
            with patch('cogtask_ci.evaluation.fit_predict') as fit:
                with self.assertRaises(ValueError):
                    evaluate(p, root/'bad_budget', Config(outer_folds=3, inner_folds=2, bootstrap=3, token_budget=1000))
                fit.assert_not_called()
            self.assertEqual(json.loads((root/'bad_budget'/'status.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
