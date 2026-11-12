# Synthetic examples

All examples use artificial Group 0/1 labels and programmatically assembled English sentences. They are software demonstrations, not transcripts from people or clinical evidence. No study data, study results or manuscript comparisons are used.

## Four scenarios

| Scenario | Construction | Question |
|---|---|---|
| Complementary | Each task independently samples a noisy lexical marker, conditional on the artificial label; every response has the same length. | Can combining tasks recover independent artificial signals? |
| Redundant | A participant's same response is copied to all three tasks. | Does repeating identical information create an improvement? |
| Beyond quantity | Length and lexical markers are sampled independently conditional on the artificial label; both carry information. Long responses repeat grammatical sentences. | Does lexical information add discrimination beyond an informative token-count baseline? |
| Null | Task markers are independent of labels and response lengths are constant. | What happens when the generator contains no designed label signal? |

The two equal-length marker sentences are:

> The window is closed and the room looks quiet today.
>
> The window is open and the room looks bright today.

Each response also contains: “A person follows the path and passes a wooden gate.” These are deliberately templated, readable sentences rather than simulations of clinical speech. In the first two scenarios, marker probabilities are 0.2/0.8 for Group 0/1. In the null scenario they are 0.5 in both groups. In the beyond-quantity scenario, marker probabilities are 0.2/0.8 by group and the probability of a four-copy response is 0.35/0.65; the two draws are independent conditional on the artificial label. A one-copy response has 20 tokens, a four-copy response 80. The text model is not guaranteed to be count-blind: repetition can change normalized TF-IDF features.

## Reproduce the README overview

From the repository root, after installing this source version:

```console
python -m cogtask_ci showcase --output results/showcase --assets docs/assets --replace-assets
```

Defaults are 160 participants per scenario, three tasks, balanced labels, generator seed 20261103, three outer/two inner folds, one repeat, evaluation seed 42 and 500 class-stratified participant bootstrap draws. For design version 2, the revised C-scenario probabilities were fixed before its first run; the seed and A/B/D constructions were retained. No seed search or outcome-based scenario selection is performed. New output directories are required. `--replace-assets` explicitly replaces only the five named showcase asset files.

The full nested evaluation, per-participant synthetic predictions, logs and environment records remain in the local output directory. Only the overview PNG/SVG, aggregate synthetic summary/paired-comparison CSVs and generation/evaluation configuration are copied into `docs/assets/`. The optional `--subjects`, `--seed` and `--bootstrap` arguments are for exploration; altered runs should not be described as the same frozen illustration.

## Read the figure

All panels plot mean outer-OOF AUROC by task count. Blue is the text model; orange is the token-count model. Bands are conditional pointwise 95% intervals. The dotted reference is AUROC 0.5. These are realized estimates from one prespecified simulation, not population truths or acceptance thresholds.

Under exact duplicated tasks, the deterministic fitting and fusion produce identical scores across task budgets. Independent artificial markers may improve combination performance, but improvement need not be monotone. A null realization need not score exactly 0.5; the conditional intervals do not account for rerunning the entire generator and development process. In C, both amount and lexical content are label-dependent by construction. Its annotation reports the paired text-minus-count AUROC difference at three tasks, not a clinical effect. D intentionally remains close to chance for both models.

## Interactive walkthroughs

- `examples/03_synthetic_showcase.ipynb` displays the sentence format and runs a smaller version of all four scenarios. Its smaller settings are for interactive speed and differ from the README illustration.
- `examples/04_csv_workflow.ipynb` creates a tiny synthetic CSV, imports it, validates coverage including a missing task, and runs evaluation. Replace only the input with your own authorized pre-cleaned data; maintain participant IDs across tasks.

The existing quickstart and complementarity notebooks remain available for the original minimal token-based demo. No synthetic result here should be reported as cognitive-impairment detection performance.
