"""Artificial lexical signals, explicitly unrelated to clinical performance."""
from pathlib import Path
import json
import numpy as np


def generate_demo(output, subjects=60, seed=42, complementary=True):
    output = Path(output)
    if output.exists():
        raise ValueError("Demo data already exists")
    if subjects < 24 or subjects % 2:
        raise ValueError("Demo subjects must be even and >=24")
    rng = np.random.default_rng(seed)
    records = []
    for i in range(subjects):
        label = i % 2
        signals = [int(rng.random() < (.75 if label else .25)), int(rng.random() < (.75 if label else .25))]
        for j, task in enumerate(("task_a", "task_b", "task_redundant")):
            signal = signals[1] if j == 1 and complementary else signals[0]
            # Equal lengths prevent quantity alone encoding the synthetic label.
            tokens = ["shared", "context", "word", "story"] * 5 + ["bright" if signal else "quiet"] * 6
            tokens += [str(x) for x in rng.integers(0, 5, 4)]
            records.append(dict(subject_id=f"S{i:04d}", task=task, label=label, segments=[tokens[:15], tokens[15:]]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    return output
