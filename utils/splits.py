from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Sequence


def make_subject_split(
    subjects: Iterable[str],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[str]]:
    """Create a deterministic subject-independent train/val/test split."""
    ordered = sorted(set(subjects))
    if not ordered:
        raise ValueError("subjects must not be empty")
    if len(ratios) != 3 or any(value < 0 for value in ratios):
        raise ValueError("ratios must contain three non-negative values")

    rng = random.Random(seed)
    rng.shuffle(ordered)

    total = len(ordered)
    n_train = int(total * ratios[0])
    n_val = int(total * ratios[1])

    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


def save_split_manifest(
    split: dict[str, Sequence[str]],
    output_path: str | Path,
    seed: int = 42,
) -> Path:
    """Write split membership and counts to a JSON manifest."""
    payload = {
        "seed": seed,
        "counts": {name: len(list(items)) for name, items in split.items()},
        "splits": {name: list(items) for name, items in split.items()},
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
