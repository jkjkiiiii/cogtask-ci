"""Command-line entry points, all output paths explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from .data import validate_dataset, import_csv
from .evaluation import Config, evaluate
from .demo import generate_demo
from .report import render_report


def _write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def _paired_policy_difference(left, right, policy, bootstrap, seed):
    """Compare one policy on identical replay subjects with paired resampling."""
    import numpy as np
    from .paired import paired_subject_bootstrap

    if type(bootstrap) is not int or bootstrap < 1 or type(seed) is not int or seed < 0:
        raise ValueError("bootstrap must be positive and seed must be nonnegative")
    if policy not in left.get("predictions", {}) or policy not in right.get("predictions", {}):
        raise ValueError("Policy is not available in both replay results")
    left_scores = {int(rep): rows for rep, rows in left["predictions"][policy].items()}
    right_scores = {int(rep): rows for rep, rows in right["predictions"][policy].items()}
    left_labels = {int(rep): rows for rep, rows in left["labels"].items()}
    right_labels = {int(rep): rows for rep, rows in right["labels"].items()}
    repeats = sorted(left_scores)
    if repeats != sorted(right_scores) or repeats != sorted(left_labels) or repeats != sorted(right_labels):
        raise ValueError("Replay comparison requires identical repeat IDs")
    subjects = None
    y = None
    x_by_repeat, z_by_repeat = [], []
    for rep in repeats:
        x, z = left_scores[rep], right_scores[rep]
        xl, zl = left_labels[rep], right_labels[rep]
        current = sorted(x)
        if current != sorted(z) or current != sorted(xl) or current != sorted(zl):
            raise ValueError("Replay comparison requires identical subjects and labels")
        if any(int(xl[s]) != int(zl[s]) for s in current):
            raise ValueError("Replay comparison labels differ")
        values = np.asarray([int(xl[s]) for s in current], dtype=int)
        if set(values.tolist()) != {0, 1}:
            raise ValueError("Replay comparison requires both classes")
        if subjects is None:
            subjects, y = current, values
        elif current != subjects or not np.array_equal(values, y):
            raise ValueError("Subject sets and labels must be constant across repeats")
        x_by_repeat.append(np.asarray([float(x[s]) for s in current]))
        z_by_repeat.append(np.asarray([float(z[s]) for s in current]))

    statistics = paired_subject_bootstrap(y, np.stack(x_by_repeat), np.stack(z_by_repeat),
                                          n_boot=bootstrap, seed=seed)
    return {
        "policy": policy,
        "contrast": "left minus right",
        "n_subjects": len(subjects),
        "n_repeats": len(repeats),
        "auroc_left": float(statistics["auroc_a"]),
        "auroc_right": float(statistics["auroc_b"]),
        "auroc_difference": float(statistics["auroc_difference"]),
        "auroc_difference_ci_low": float(statistics["auroc_difference_ci_low"]),
        "auroc_difference_ci_high": float(statistics["auroc_difference_ci_high"]),
        "brier_left": float(statistics["brier_a"]),
        "brier_right": float(statistics["brier_b"]),
        "brier_difference": float(statistics["brier_difference"]),
        "brier_difference_ci_low": float(statistics["brier_difference_ci_low"]),
        "brier_difference_ci_high": float(statistics["brier_difference_ci_high"]),
        "ci_low": float(statistics["auroc_difference_ci_low"]),
        "ci_high": float(statistics["auroc_difference_ci_high"]),
        "bootstrap_replicates": bootstrap,
        "bootstrap_seed": seed,
        "uncertainty": "paired participant bootstrap conditional on frozen fits and folds",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cogtask-ci")
    from . import __version__
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate", help="Validate token-segment JSONL")
    p.add_argument("input", type=Path)
    p.add_argument("--lowercase", action="store_true")
    p = sub.add_parser("import-csv", help="Convert pre-cleaned text CSV to JSONL")
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lowercase", action="store_true")
    p = sub.add_parser("evaluate", help="Run nested task-budget evaluation")
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--splits", type=Path)
    budget = p.add_mutually_exclusive_group()
    budget.add_argument("--token-budget-per-task", type=int,
                        help="Fit first-N-token models with N tokens per task (legacy config key: token_budget)")
    budget.add_argument("--total-token-budget", type=int,
                        help="Hold the total prefix-token count fixed across each selected task set")
    p = sub.add_parser("replay", help="Apply frozen fold predictions without refitting")
    p.add_argument("bundle", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixed-task", required=True)
    p.add_argument("--random-draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261112)
    p = sub.add_parser("compare", help="Compare one paired policy across two frozen bundles")
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--left-fixed-task", required=True)
    p.add_argument("--right-fixed-task", required=True)
    p.add_argument("--policy", default="global_best_subset")
    p.add_argument("--random-draws", type=int, default=1000)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261112)
    p = sub.add_parser("report", help="Regenerate report in an existing run")
    p.add_argument("run_dir", type=Path)
    p = sub.add_parser("demo", help="Generate artificial data and run a small example")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--subjects", type=int, default=60)
    p = sub.add_parser("showcase", help="Run four synthetic sentence examples and make a 2x2 overview")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--subjects", type=int, default=160)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--assets", type=Path)
    p.add_argument("--replace-assets", action="store_true")
    p = sub.add_parser("factorial", help="Fit and save synthetic factorial evaluation bundles")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--protocol-sha256")
    p.add_argument("--protocol-git-commit")
    p.add_argument("--code-git-commit")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            print(json.dumps(validate_dataset(args.input, args.lowercase), indent=2))
        elif args.command == "import-csv":
            print(json.dumps(import_csv(args.input, args.output, args.lowercase), indent=2))
        elif args.command == "evaluate":
            values = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
            if args.token_budget_per_task is not None:
                if any(values.get(key) is not None for key in ("token_budget", "token_budget_per_task", "total_token_budget")):
                    raise ValueError("Do not combine a CLI budget flag with a budget in --config")
                values["token_budget_per_task"] = args.token_budget_per_task
            if args.total_token_budget is not None:
                if any(values.get(key) is not None for key in ("token_budget", "token_budget_per_task", "total_token_budget")):
                    raise ValueError("Do not combine a CLI budget flag with a budget in --config")
                values["total_token_budget"] = args.total_token_budget
            cfg = Config(**values)
            evaluate(args.input, args.output, cfg, args.splits)
            print("Completed: " + str(args.output / "report.html"))
        elif args.command == "replay":
            from .replay import apply_bundle, load_bundle
            bundle = load_bundle(args.bundle)
            result = apply_bundle(bundle, fixed_task=args.fixed_task,
                                  random_draws=args.random_draws, seed=args.seed)
            result["bundle_sha256"] = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
            _write_json_new(args.output, result)
            print("Replay saved: " + str(args.output))
        elif args.command == "compare":
            from .replay import apply_bundle, load_bundle
            left_bundle = load_bundle(args.left)
            right_bundle = load_bundle(args.right)
            left_result = apply_bundle(left_bundle, fixed_task=args.left_fixed_task,
                                       random_draws=args.random_draws, seed=args.seed)
            right_result = apply_bundle(right_bundle, fixed_task=args.right_fixed_task,
                                        random_draws=args.random_draws, seed=args.seed)
            comparison = _paired_policy_difference(left_result, right_result, args.policy,
                                                   args.bootstrap, args.seed)
            comparison["left_bundle_sha256"] = hashlib.sha256(args.left.read_bytes()).hexdigest()
            comparison["right_bundle_sha256"] = hashlib.sha256(args.right.read_bytes()).hexdigest()
            _write_json_new(args.output, comparison)
            print("Comparison saved: " + str(args.output))
        elif args.command == "report":
            print(render_report(args.run_dir))
        elif args.command == "showcase":
            from .showcase import build_showcase
            print(build_showcase(args.output, args.subjects, args.seed, args.bootstrap, args.assets, args.replace_assets))
        elif args.command == "factorial":
            from .factorial import run_factorial
            provenance = {key: value for key, value in {
                "protocol_sha256": args.protocol_sha256,
                "protocol_git_commit": args.protocol_git_commit,
                "code_git_commit": args.code_git_commit,
            }.items() if value is not None}
            completed = run_factorial(args.output, start_index=args.start_index,
                                      limit=args.limit, provenance_context=provenance)
            print(f"Synthetic factorial complete: {len(completed)} selected runs")
        else:
            if args.output.exists():
                raise ValueError("Demo output must not exist")
            source = generate_demo(args.output / "synthetic.jsonl", args.subjects)
            evaluate(source, args.output / "run", Config(outer_folds=3, inner_folds=2, bootstrap=100, token_budget=20))
            print("Synthetic demo (not clinical evidence): " + str(args.output / "run" / "report.html"))
        return 0
    except (ValueError, TypeError, OSError) as e:
        print("ERROR: " + str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
