# Changes in the 0.2.0 candidate

This candidate adds reference-aware evaluation using frozen fold predictions. It does not include participant data or study results.

- `evaluate` retains its existing default outputs and now saves per-task outer predictions and inner OOF bundles for replay.
- `replay` evaluates fixed pairs, three single-task references, absolute-versus-incremental pair selection, random pair policies, and tolerance-based task counts without fitting again.
- `compare` reports paired AUROC and Brier differences while retaining participants and repeats as the sampling structure. Intervals condition on saved fits; they do not include the full model-development process.
- Per-task token budgets and fixed total token budgets are separate, mutually exclusive modes. Total-budget models are refitted for each task count and cannot be mixed in generic all-count replay.
- Frozen multilingual embeddings are optional. Revisions and artifact hashes, deterministic FP32 execution, complete window coverage, and text-only cache identities are recorded. The base installation and ordinary CI do not download weights.
- A synthetic factorial covers task strength, redundancy/complementarity, response quantity and label imbalance. Cached predictions support later policy analysis.
- CI covers Windows and Linux on Python 3.10 and 3.12, with synthetic tests, notebook execution, package building and an allowlisted source archive.

## Compatibility and reproducibility

Existing `evaluate` and `report` command forms remain available. `Config.token_budget` remains an alias for the per-task budget, and the original positional configuration fields retain their order. Added output files do not change the existing CSV column contracts.

Subset AUROC selection now uses the average-rank formula consistently. This removes last-bit floating-point discrepancies that could otherwise break mathematical ties differently between fitting and replay. A tied selection can therefore differ from an earlier version; retain the exact software version and saved predictions when reproducing an earlier run.

The random-policy distribution describes random selection, not a participant confidence interval. A tolerance is an operational task-reduction setting, not an equivalence or noninferiority margin. A confidence interval crossing zero does not establish equality.

See [evaluation and replay](reference_evaluation.md), [frozen embeddings](embeddings.md), and [examples](examples.md) for use and limitations. This document describes a candidate version; it does not imply a release tag or package publication.
