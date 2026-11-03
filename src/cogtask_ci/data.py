"""Explicit token segments and participant-level input contracts."""
from dataclasses import dataclass
from pathlib import Path
import csv
import json


@dataclass(frozen=True)
class Record:
    subject_id: str
    task: str
    label: int
    segments: tuple[tuple[str, ...], ...]

    @property
    def n_tokens(self):
        return sum(map(len, self.segments))


def load_dataset(path, lowercase=False):
    rows = []
    seen, labels = set(), {}
    with Path(path).open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                subject, task, label, segments = (r[k] for k in ("subject_id", "task", "label", "segments"))
                if not all(isinstance(x, str) and x.strip() == x and x for x in (subject, task)):
                    raise ValueError("subject_id/task must be nonempty trimmed strings")
                if type(label) is not int or label not in (0, 1):
                    raise ValueError("label must be integer 0 or 1")
                if not isinstance(segments, list) or not segments or not all(isinstance(s, list) and s for s in segments):
                    raise ValueError("segments must contain nonempty token lists")
                tokens = [t for s in segments for t in s]
                if not all(isinstance(t, str) and t and not any(c.isspace() for c in t) for t in tokens):
                    raise ValueError("tokens must be nonempty strings without whitespace")
                key = (subject, task)
                if key in seen:
                    raise ValueError("duplicate subject/task")
                if subject in labels and labels[subject] != label:
                    raise ValueError("conflicting subject label")
                seen.add(key)
                labels[subject] = label
                rows.append(Record(subject, task, label, tuple(tuple(t.lower() if lowercase else t for t in s) for s in segments)))
            except (ValueError, KeyError, TypeError) as e:
                raise ValueError(f"Invalid JSONL line {line_no}: {e}") from e
    if not rows:
        raise ValueError("Dataset is empty")
    if set(labels.values()) != {0, 1}:
        raise ValueError("Dataset must contain both classes")
    if len({r.task for r in rows}) > 10:
        raise ValueError("At most 10 tasks are supported (exhaustive selection)")
    return rows


def describe(rows):
    tasks = sorted({r.task for r in rows})
    labels = {r.subject_id: r.label for r in rows}
    coverage = {s: set() for s in labels}
    for r in rows:
        coverage[r.subject_id].add(r.task)
    complete = sorted(s for s in labels if len(coverage[s]) == len(tasks))
    return {"records": len(rows), "subjects": len(labels), "tasks": tasks,
            "classes": {str(y): sum(v == y for v in labels.values()) for y in (0, 1)},
            "complete_subjects": len(complete), "incomplete_subjects": len(labels) - len(complete),
            "complete_classes": {str(y): sum(labels[s] == y for s in complete) for y in (0, 1)},
            "task_coverage": {t: sum(r.task == t for r in rows) for t in tasks}}


def validate_dataset(path, lowercase=False):
    return describe(load_dataset(path, lowercase))


def import_csv(source, output, lowercase=False):
    """Whitespace tokenization only; no speaker extraction or corpus-specific rules."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Output already exists")
    records = []
    with source.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV column names must be unique")
        if not {"subject_id", "task", "label", "text"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires subject_id,task,label,text")
        for line_no, r in enumerate(reader, 2):
            if any(r.get(k) is None for k in ("subject_id", "task", "label", "text")) or None in r:
                raise ValueError(f"CSV line {line_no}: missing or extra cells")
            if r["label"] not in {"0", "1"}:
                raise ValueError("CSV labels must be 0 or 1")
            text = r["text"].lower() if lowercase else r["text"]
            records.append({"subject_id": r["subject_id"], "task": r["task"], "label": int(r["label"]), "segments": [text.split()]})
    # Validate before touching the requested destination.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / "validate.jsonl"
        probe.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
        summary = validate_dataset(probe)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as f:
            f.write(probe.read_text(encoding="utf-8"))
    return summary


def crop(segments, budget):
    result, left = [], budget
    for segment in segments:
        if left <= 0:
            break
        result.append(segment[:left])
        left -= len(result[-1])
    return tuple(result)
