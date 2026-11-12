# Data, configuration and API reference

For frozen prediction bundles, replay, paired bundle comparisons, total-token-budget refits and the synthetic factorial, see [evaluation reference](reference_evaluation.md).

## JSONL input

```json
{"subject_id":"S001","task":"description","label":1,"segments":[["a","person","walking"],["near","a","house"]]}
```

Required fields: trimmed nonempty string `subject_id`, trimmed nonempty string `task`, integer 0/1 `label`, and a nonempty list `segments` of nonempty token lists. Tokens must be nonempty strings without whitespace. Boolean labels, duplicate participant/task keys and conflicting labels for a participant are rejected. Extra JSON properties are ignored. Omit missing tasks rather than representing them with empty text. One analysis accepts one dataset and one binary endpoint, with at most ten tasks. Namespace IDs yourself if your source combines cohorts.

Normalization is explicit: no implicit lowercasing, apostrophe mapping, Unicode normalization or speaker removal is performed. `lowercase=true` applies Python string lowercasing to supplied tokens. Segments are lexical discontinuities, not automatically inferred sentences.

## Configuration

Pass a UTF-8 JSON object with `--config`, or create `Config(...)` in Python. Unknown keys fail. `examples/default_config.json` specifies:

| Field | Default | Meaning |
|---|---|---|
| `outer_folds` | 5 | Participant outer folds, at least 2 |
| `inner_folds` | 3 | Inner folds in each outer-training partition, at least 2 |
| `repeats` | 1 | Complete repeated outer-CV runs |
| `seed` | 42 | Nonnegative integer below 2^31; split-derived seeds wrap modulo 2^32 |
| `bootstrap` | 2000 | Class-stratified participant draws, at least 1; small demo counts are not recommended for reporting intervals |
| `token_budget` | null | Optional positive token count **per task** for deterministic prefix cropping |
| `lowercase` | false | Explicit token lowercasing |

The matched crop population is the complete-task set with enough tokens in every task. Its three arms use the same participants for training/evaluation and retain original outer-fold identities. All arms are preflight-checked before any model fit. If that subset makes retained folds infeasible, the run fails rather than silently changing folds.

## Fixed outer splits

CSV fields: `repeat,subject_id,outer_fold`, with zero-based numeric indices. Every input subject appears once per repeat, including incomplete-task subjects. Unknown, duplicate or out-of-range entries fail. Both classes must occur in each outer-test complete-task subset and in required inner partitions. Exported `outer_splits.csv` is directly reusable. Inner folds remain algorithmically generated; arbitrary historical inner split import is not supported.

## Python API

```python
from cogtask_ci import Config, evaluate, validate_dataset, import_csv, render_report

coverage = validate_dataset("local_data/text.jsonl")
metrics = evaluate("local_data/text.jsonl", "results/new_run", Config())
report = render_report("results/new_run")
```

`load_dataset(path, lowercase=False)` returns validated `Record` values; `validate_dataset` returns coverage counts. `import_csv(source, output, lowercase=False)` validates before writing a new JSONL. `evaluate(input_path, output, config=None, splits=None)` returns summary dictionaries and writes a new run. `render_report(run_dir)` returns the HTML path and regenerates figures in that run. Validation/evaluation errors normally raise `ValueError`; CLI errors return exit code 2 and success returns 0. `--version` prints the software version.

## Statistical outputs

For an arm A, gain(k) = AUROC(A,k) − AUROC(A,1). The single-task reference is selected in each outer-training partition, not chosen using outer-test outcomes.

`contrasts.csv` reports left-minus-right `auroc_difference`, its `ci_low/ci_high`, and `gain_difference` = gain(left,k) − gain(right,k), with `gain_ci_low/gain_ci_high`. All comparisons are paired on identical participants, repeats and task budgets. At k=1 each gain is zero by definition, while the absolute-AUROC difference can be nonzero. These intervals are pointwise, with no multiplicity adjustment. They are conditional on realized fits, folds and selections, not a refit bootstrap or a measure of model-development uncertainty.

Scores are probabilities output by class-weighted logistic regression; they are not calibrated clinical risks. Aggregating task probabilities is part of the prespecified policy. Do not select a task budget using its best outer-test score and describe that budget as independently confirmed.
