"""tune_optuna.py ▸ Automatic hyper‑parameter search with Optuna

This script performs Bayesian optimisation over three knobs:
  • batch_size         categorical [16, 32, 64, 128]
  • learning_rate      log-uniform 1e-5 … 1e-2
  • hidden_channels    categorical [32, 64, 128, 256]

It reuses the same data pipeline and model/training utilities that
`main.py` uses, so no other code has to change.

Usage (single‑GPU laptop example):
  $ python tune_optuna.py --dataset my_dataset --trials 50 --jobs 4

A SQLite study database is written next to the dataset so you can resume
or inspect with `optuna-dashboard`. When finished, the best parameters
are dumped to JSON so you can feed them back into `main.py`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import optuna
import torch
from torch_geometric.data import DataLoader

from data_preprocessing import split_data, extract_label
from data_retrival import load_data_local
from graph_builder import build_graph_batch
from GNN.my_model import CrystalGNN, CrystalGNNTransformer
from GNN.train import train_model, evaluate_loss_model as evaluate_model
from utils import EarlyStopper

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "DownloadedCrystalProperties"
STUDIES_DIR = ROOT / "optuna_studies"
STUDIES_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(trial: optuna.Trial, dataset: str, max_epochs: int = 100) -> float:
    """A single optimisation trial - builds loaders, model, trains, returns val loss."""

    # Hyper‑parameters to sample
    batch_size: int = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])
    lr: float = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    hidden: int = trial.suggest_categorical("hidden_channels", [32, 64, 128, 256, 512])

    # Load & split data only once per trial (OK for ≤ few hundred trials)
    raw = load_data_local(dataset)
    train_set, val_set, _ = split_data(raw)
    
    # Create (crystal_obj, structure) tuples
    train_data = [(crystal_obj, crystal_obj.structure) for crystal_obj in train_set]
    val_data = [(crystal_obj, crystal_obj.structure) for crystal_obj in val_set]
    
    # Extract labels and build graphs
    train_labeled = extract_label(train_data)
    val_labeled = extract_label(val_data)
    
    train_graphs = build_graph_batch(train_labeled)
    val_graphs = build_graph_batch(val_labeled)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)

    model = CrystalGNN(num_features=train_graphs[0].num_features,
                       hidden_channels=hidden).to(DEVICE)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    stopper = EarlyStopper(patience=15, delta=1e-4)
    best_val = float("inf")

    for epoch in range(max_epochs):
        train_model(model, train_loader, optimiser, criterion, DEVICE)
        val_loss = evaluate_model(model, val_loader, criterion, DEVICE)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        best_val = min(best_val, val_loss)
        if stopper(val_loss):
            break

    return best_val


def run_search(dataset: str, n_trials: int, n_jobs: int) -> Tuple[str, Dict]:
    study_path = STUDIES_DIR / f"{dataset}.db"
    storage = f"sqlite:///{study_path}"
    study = optuna.create_study(direction="minimize", study_name=dataset,
                                storage=storage, load_if_exists=True)

    study.optimize(lambda t: objective(t, dataset), n_trials=n_trials, n_jobs=n_jobs)

    print("\n=== Best trial ===")
    print(" Value:", study.best_value)
    print(" Params:")
    for k, v in study.best_params.items():
        print(f"  • {k}: {v}")

    out_json = STUDIES_DIR / f"best_params_{dataset}.json"
    with out_json.open("w") as fh:
        json.dump(study.best_params, fh, indent=2)
    print(f"Saved best params → {out_json.relative_to(ROOT)}")

    return str(out_json), study.best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyper-parameter search")
    parser.add_argument("--dataset", required=True, help="Dataset name (without .txt)")
    parser.add_argument("--trials", type=int, default=40, help="Number of Optuna trials")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs (processes)")
    args = parser.parse_args()

    dataset_path = DATA_DIR / f"{args.dataset}.txt"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    run_search(args.dataset, n_trials=args.trials, n_jobs=args.jobs)