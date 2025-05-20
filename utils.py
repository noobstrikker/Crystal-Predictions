class EarlyStopper:

    def __init__(self, patience: int = 10, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = float("inf")
        self.counter = 0
        self.stop = False

    def __call__(self, score: float) -> bool:
        """
        Update with the new metric value; return True if training should stop.
        """
        if score < self.best_score - self.delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop
    
import os, json, random, datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int | None = None, deterministic: bool = True) -> int:
    """
    Seed Python-random, NumPy and PyTorch (+ CUDA) in one call.
    If *seed* is None we draw one at random and **return it** so callers
    can record the value they actually used.
    """
    if seed is None:
        seed = random.randrange(1, 2**32 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def record_run_meta(out_dir: Path,*, filename: str = "run_meta.json", **extra) -> None:
    """Write seed + any other key/value pairs to *out_dir/run_meta.json*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / filename
    meta = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), **extra}
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)