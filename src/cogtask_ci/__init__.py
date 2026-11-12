"""Research evaluation, not a diagnostic model or clinical label mapping."""
__version__ = "0.2.0"

from .data import load_dataset, validate_dataset, import_csv
from .evaluation import Config, evaluate
def render_report(run_dir):
    """Render a local report without modifying the caller's plotting backend."""
    from .report import render_report as render
    return render(run_dir)

__all__ = ["Config", "evaluate", "validate_dataset", "load_dataset", "import_csv", "render_report"]
