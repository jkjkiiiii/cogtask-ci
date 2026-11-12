# Nested evaluation, frozen replay and synthetic factorial

This reference describes evaluation controls and reusable fold outputs. All examples use artificial or user supplied data; the package does not map labels to a diagnosis.

## Token budgets

`Config(token_budget=N)` is retained as a compatibility alias for `token_budget_per_task=N`. The CLI spelling is explicit:

```console
cogtask-ci evaluate local_data/text.jsonl --token-budget-per-task 30 --output results/per_task
cogtask-ci evaluate local_data/text.jsonl --total-token-budget 60 --output results/fixed_total
```

The two modes are mutually exclusive, including when one value comes from `--config`. Per-task mode refits the text model on the first N tokens of every task. A k-task set therefore uses kN tokens; it is a prefix sensitivity analysis.

Total-token mode holds the selected task set at exactly B tokens. For every tested k, each per-task model is refit on the first B/k tokens. B must be divisible by every k from 1 through the number of tasks, so every set gets an equal integer allocation. For example, 60 works for three tasks; 61 does not. The common population is chosen before fitting and includes complete-task participants with at least B tokens in every task. This eligibility rule is conservative and keeps the evaluation population fixed for all k. `coverage.json` records the restriction.

Total-token predictions are written to the normal local run outputs. The evaluator also writes one `fixed_total_text_totalB_kK.json` bundle per task count. Each carries its total budget, per-task prefix and fixed k. The generic `replay` command rejects these tagged bundles: their task scores come from separate per-task refits at B/k, so an all-k policy replay would mix models trained at different token budgets. Keep these as fit artifacts for fixed-k analyses and compare only bundles with the same total budget and k.

## Per-arm replay bundles

`evaluate` writes `replay_bundles/<arm>.json` alongside the ordinary outputs. Schema version 1 stores:

- sorted task identifiers;
- inner OOF participants, labels, task score matrices and pre-crop token counts for each outer fold;
- outer test participants, labels and per-task scores from models fitted on that fold's training participants.

Rows in each matrix align positionally with the associated subject list. The bundle includes participant identifiers, labels, token counts and predictions. Treat it as participant-derived data and keep it local.

Replay applies selection rules to saved fold outputs without refitting models:

```console
cogtask-ci replay results/my_run/replay_bundles/text.json \
  --fixed-task task_a --random-draws 1000 --seed 20261112 \
  --output results/my_run/replay.json
```

The `replay` command is for standard fixed-fit bundles. Total-token bundles are intentionally tagged and rejected by generic replay; inspect them only in a fixed-k analysis that respects their per-task budget metadata.

The result contains predictions for each frozen policy and fold-level selections. Replay does not alter the source run. The fixed task is explicit; policy scores share the bundle's outer test subjects.

Compare one policy across two bundles with a paired participant bootstrap:

```console
cogtask-ci compare left_bundle.json right_bundle.json \
  --left-fixed-task task_a --right-fixed-task task_a \
  --policy global_best_subset --bootstrap 2000 --seed 20261112 \
  --output results/paired_comparison.json
```

The comparison requires the same repeats, participants and labels. It reports left-minus-right AUROC and a class-stratified paired participant-bootstrap interval conditional on the frozen fits and folds. It does not include model refitting or selection uncertainty.

The Python entry points are `cogtask_ci.replay.load_bundle`, `validate_bundle` and `apply_bundle`. See [data and configuration reference](reference.md) for the ordinary evaluation API.

## Synthetic factorial

The factorial generator creates only synthetic `task_a`, `task_b` and `task_c` text. It does not read local study data or use pretrained embeddings. The fixed design has 36 cells: base task sensitivity/specificity of 0.60, 0.70 or 0.80; added-task signal of no information, exact duplication or independent sensitivity/specificity 0.75; quantity signal off/on; and exact positive-label prevalence 0.50/0.30. Each dataset has 200 participants. When quantity is on, long responses contain four times as many tokens as short responses; their probabilities are 0.35 for label 0 and 0.65 for label 1. Duplicated tasks copy task A's text and length exactly.

Each cell is generated for the predetermined seeds 20261112 through 20261141, for 1,080 runs total. The runner fits each text and quantity arm once and saves its fold predictions for later policy replay. It does not bootstrap each run; uncertainty can be estimated over the independent seeds after replay. `--start-index` and `--limit` choose a stable slice of this fixed ordered list, and rerunning the same slice resumes from matching input, configuration, protocol and source hashes.

```console
cogtask-ci factorial --output results/factorial --start-index 0 --limit 9 \
  --protocol-sha256 <frozen-protocol-hash> \
  --protocol-git-commit <protocol-commit> \
  --code-git-commit <code-commit>
```

Omitting `--start-index` and `--limit` runs all 1,080 synthetic conditions. Each completed run stores its synthetic JSONL input, `text_bundle.json`, `quantity_bundle.json`, and `result.json` with data/configuration/code hashes, environment, timestamp and supplied protocol/code identifiers. These outputs are local artifacts, not clinical results.

For the task-signal generator and runner from Python:

```python
from cogtask_ci.factorial import factorial_conditions, generate_factorial_records, run_factorial

conditions = factorial_conditions()
records = generate_factorial_records(
    subjects=200, base_strength=0.70, added_type="complementary",
    quantity=True, positive_fraction=0.30, seed=20261112,
)
runs = run_factorial("results/factorial", start_index=0, limit=9)
```

## Optional text embeddings

The frozen text-embedding adapter is an optional dependency group. Install it only when the embedding adapter is needed:

```console
python -m pip install ".[embeddings]"
```

The default TF-IDF evaluation and all synthetic factorial fits work without PyTorch or Transformers. See the adapter API documentation for model revision and cache requirements.
