# Contributing

Install in a fresh environment with `python -m pip install -e ".[dev,notebook]"`. Run `python -m unittest discover -s tests -v`, execute both example notebooks, and build with `python -m build` before requesting review. Run commands from the repository root. Contributions are provided under the project's MIT License.

Use synthetic examples and focused regression tests. Never post participant transcripts, labels, predictions, credentials or identifying screenshots in issues or pull requests. Report the software/dependency versions, command, configuration and a minimal artificial input reproducing the issue.

Method changes must describe effects on eligibility, token segments, fitting partitions, task ordering/ties, selection, fusion and uncertainty. A code cleanup must not silently change the statistical protocol. Preserve subject isolation and training-only feature fitting. Do not silently change folds or choose preprocessing based on held-out results.

Public documentation, code comments, CLI messages and examples are maintained in English. Translations are optional additions, not a requirement to use or contribute to the project. Keep participant-level research data and study-result comparisons outside the public repository.

Pull requests should state the problem, behavioral change, tests run and any remaining limitation. Do not add generated results, virtual environments or executed notebook outputs.
