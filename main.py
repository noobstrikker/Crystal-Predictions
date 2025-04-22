import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch_geometric.data import DataLoader

from data_retrival import load_data_local
from data_preprocessing import split_data, extract_label
from graph_builder import build_graph_batch
from GNN.my_model import CrystalGNN
from GNN.train import (
    train_model,
    evaluate_model,
    evaluate_model_performance,
)


ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "DownloadedCrystalProperties"
CFG_PATH = ROOT_DIR / "config_defaults.json"
MODELS_DIR = ROOT_DIR / "trained_models"


FALLBACK_DEFAULTS: dict[str, Any] = {
    "output": "best_model.pth",
    "batch_size": 32,
    "epochs": 100,
    "lr": 0.001,
    "hidden_channels": 64,
}

def load_defaults() -> dict[str, Any]:
    """Load defaults from JSON or return built‑in fallbacks."""
    if CFG_PATH.exists():
        try:
            with CFG_PATH.open("r") as fh:
                disk_cfg = json.load(fh)
            return {**FALLBACK_DEFAULTS, **disk_cfg}
        except Exception:
            print("Warning: could not parse config_defaults.json – using built‑in defaults.")
    return FALLBACK_DEFAULTS.copy()

def save_defaults(cfg: dict[str, Any]) -> None:
    """Persist new defaults to disk."""
    with CFG_PATH.open("w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"Saved new defaults to {CFG_PATH}")


def ask_choice(prompt: str, options: list[str]) -> str:
    print(f"\n{prompt}")
    for idx, opt in enumerate(options, 1):
        print(f" [{idx}] {opt}")
    while True:
        try:
            sel = input("Choice: ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(options):
                return options[int(sel) - 1]
        except ValueError:
            pass
        print(f"Invalid choice. Please select a number between 1 and {len(options)}.")

def ask_value(prompt: str, default: Any, cast) -> Any:
    val = input(f"{prompt} (default: {default}): ").strip()
    return cast(val) if val else default


def parse_args(datasets: list[str], defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train CrystalGNN on a selected dataset with optional overrides",
    )
    p.add_argument("-d", "--dataset", choices=datasets, help="Dataset to train on")
    p.add_argument("--interactive", action="store_true", help="Force interactive editing")

    p.add_argument("-o", "--output", default=defaults["output"], help="Model filename")
    p.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    p.add_argument("--epochs", type=int, default=defaults["epochs"])
    p.add_argument("--lr", type=float, default=defaults["lr"])
    p.add_argument("--hidden_channels", type=int, default=defaults["hidden_channels"])
    return p.parse_args()


def main() -> None:
    print("CrystalGNN Training Script starting...")
    print("Loading defaults...")
    if not DATASET_DIR.exists():
        sys.exit(f"ERROR: dataset directory not found: {DATASET_DIR}")
    available = sorted(p.stem if p.is_file() else p.name for p in DATASET_DIR.iterdir())

    defaults = load_defaults()
    args = parse_args(available, defaults)

    # Prompt for dataset if not given on cli
    if args.dataset is None:
        args.dataset = ask_choice("Select dataset:", available)

    # asks for model filename (shows current value)
    args.output = input(f"Model filename (default {args.output}): ").strip() or args.output

    # Ask if user wants to tweak other hyperparams
    if args.interactive or input("Alter other defaults? (y/N): ").strip().lower() == "y":
        args.batch_size = ask_value("Batch size", args.batch_size, int)
        args.epochs = ask_value("Epochs", args.epochs, int)
        args.lr = ask_value("Learning rate", args.lr, float)
        args.hidden_channels = ask_value("Hidden channels", args.hidden_channels, int)

    # Offer to save new values as defaults
    if input("Save these values as new defaults? (y/N): ").strip().lower() == "y":
        new_defaults = {
            "output": args.output,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_channels": args.hidden_channels,
        }
        save_defaults(new_defaults)

    # training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading dataset: {args.dataset}...")
    dataset = load_data_local(args.dataset)
    train_set, val_set, test_set = split_data(dataset)
    print(f"Train set: {len(train_set)} | Validation set: {len(val_set)} | Test set: {len(test_set)}")
    train_graphs = build_graph_batch(extract_label(train_set))
    val_graphs = build_graph_batch(extract_label(val_set))
    test_graphs = build_graph_batch(extract_label(test_set))

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)


    Models_path = (MODELS_DIR / args.output).resolve()
    Models_path.parent.mkdir(parents=True, exist_ok=True)

    model = CrystalGNN(
        num_features=train_graphs[0].num_features,
        hidden_channels=args.hidden_channels,
    ).to(device)
    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_model(model, train_loader, optimiser, criterion, device)
        v_loss = evaluate_model(model, val_loader, criterion, device)
        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), Models_path)
            print(f"★ Saved new best model → {Models_path}")
        print(f"E{epoch:03d} | train {tr_loss:.4f} | val {v_loss:.4f}")

    # Final test
    model.load_state_dict(torch.load(Models_path))
    t_loss = evaluate_model(model, test_loader, criterion, device)
    print(f"\nBest model {args.output} | test loss: {t_loss:.4f}")

    evaluate_model_performance(model, test_loader, device, property_name="Target Property")


if __name__ == "__main__":
    main()