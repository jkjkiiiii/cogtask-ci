"""Command-line entry points, all output paths explicit."""
import argparse
import json
from pathlib import Path
import sys
from .data import validate_dataset, import_csv
from .evaluation import Config, evaluate
from .demo import generate_demo
from .report import render_report


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
    p = sub.add_parser("report", help="Regenerate report in an existing run")
    p.add_argument("run_dir", type=Path)
    p = sub.add_parser("demo", help="Generate artificial data and run a small example")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--subjects", type=int, default=60)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            print(json.dumps(validate_dataset(args.input, args.lowercase), indent=2))
        elif args.command == "import-csv":
            print(json.dumps(import_csv(args.input, args.output, args.lowercase), indent=2))
        elif args.command == "evaluate":
            cfg = Config(**json.loads(args.config.read_text(encoding="utf-8-sig"))) if args.config else Config()
            evaluate(args.input, args.output, cfg, args.splits)
            print("Completed: " + str(args.output / "report.html"))
        elif args.command == "report":
            print(render_report(args.run_dir))
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
