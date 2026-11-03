# Quickstart

## Install from the repository

Use Python 3.10 or later. Open a terminal **in the repository root**:

```console
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux/macOS, then:

```console
python -m pip install .
python -m cogtask_ci --version
python -m cogtask_ci demo --output results/quickstart
```

Open `results/quickstart/run/report.html`. The demo uses artificial text and balanced synthetic labels. It downloads no corpus. Choose a new output directory on subsequent runs. This repository is installable locally; `pip install cogtask-ci` from PyPI is not advertised until a real PyPI release exists.

## Use pre-cleaned text

Create `local_data/text.csv` with `subject_id,task,label,text`. Each row represents one participant's task; labels are integer 0/1. Use the same participant ID across tasks. The CSV importer only splits on Unicode whitespace: it does not remove examiner speech, normalize punctuation, parse CHAT or infer a clinical label.

```console
python -m cogtask_ci import-csv local_data/text.csv --output local_data/text.jsonl
python -m cogtask_ci validate local_data/text.jsonl
python -m cogtask_ci evaluate local_data/text.jsonl --config examples/default_config.json --output results/my_run
```

For segmented or language-specific tokens, provide JSONL directly, as in [data and configuration reference](reference.md). Unknown/uncertain transcription spans should separate token segments so word bigrams cannot cross them. For languages without whitespace word boundaries, tokenize before import.

## Understand a run

- `summary.csv`: AUROC and paired gain over the inner-selected single-task policy at every task count.
- `contrasts.csv`: between-arm differences in absolute AUROC **and** in multi-task gain. These answer different questions.
- `selection_frequency.csv`: inner-selected task combinations across outer fits, not causal task importance.
- `coverage.json`: source coverage, common evaluation population, and exclusions from the optional matched analysis.
- `report.html` and `task_budget.*`: local report and publication-exportable plots. The values are research evaluations, not calibrated diagnostic probabilities.

Every task count shares the complete-task evaluation population within an arm. Incomplete participants may contribute available training records. Optional cropping uses `token_budget` tokens **per task**, not a fixed total across a task set. This is a general software workflow; no study dataset or study results are distributed.

## Notebooks

```console
python -m pip install ".[notebook]"
python -m jupyterlab
```

Open `examples/01_quickstart.ipynb`, then `examples/02_task_complementarity.ipynb`. Both generate synthetic data into new temporary directories and can be executed repeatedly. They intentionally ship without execution output. Their printout tells you where local results were saved.

## Common failures

| Error | Action |
|---|---|
| Output already exists | Choose a new run directory; outputs are not silently overwritten. |
| Insufficient complete-task subjects per class | Check coverage and choose a scientifically justified validation design before looking at results. A smaller fold count changes the protocol. |
| Empty fold-local vocabulary | Inspect tokenization and sample size. Rare training-only tokens may all fail the fixed `min_df=2` rule. Do not fit a vocabulary on the entire dataset. |
| Matched crop arm fails preflight | Increase data coverage or choose a justified per-task crop length. Original outer folds are retained; no favorable replacement folds are searched. |
| Logistic regression did not converge | Inspect the indicated failed run and training data; the run is not marked complete. |
| Console command not found | Use `python -m cogtask_ci` with the Python interpreter in which you installed the package. |
| Old Jupyter extension fails | Try a fresh virtual environment rather than modifying unrelated research environments. |

Keep participant-level output local. The source release builder excludes `results/` and `local_data/`; publish the generated source ZIP, not a ZIP of the whole working directory.
