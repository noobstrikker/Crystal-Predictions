import argparse
import json
import traceback
import os
from pathlib import Path
from typing import Any, List

import torch
import torch.optim as optim
from torch_geometric.data import DataLoader

from data_retrival import get_materials_data, get_single_materials, save_data_local, load_data_local
from data_preprocessing import split_data, extract_label
from graph_builder import build_graph_batch
from GNN.my_model import CrystalGNN, CrystalGNNTransformer
from GNN.train import train_model
from GNN.train import evaluate_loss_model as evaluate_model
from GNN.evaluation import evaluate_model_performance
from GNN.evaluation import evaluate_predictions
from utils import EarlyStopper, set_seed, record_run_meta
from optuna_tune import run_search

GLOBAL_SEED: int | None = None
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
    "patience": 15,
}

def load_defaults() -> dict[str, Any]:
    if CFG_PATH.exists():
        try:
            with CFG_PATH.open("r") as fh:
                return {**FALLBACK_DEFAULTS, **json.load(fh)}
        except Exception:
            pass
    return FALLBACK_DEFAULTS.copy()

def save_defaults(cfg: dict[str, Any]) -> None:
    with CFG_PATH.open("w") as fh:
        json.dump(cfg, fh, indent=2)

def ask_choice(prompt: str, options: List[str]) -> str:
    while True:
        print(f"\n{prompt}")
        for idx, opt in enumerate(options, 1):
            print(f" [{idx}] {opt}")
        sel = input("Choice: ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(options):
            return options[int(sel) - 1]
        print("Invalid choice.")

def ask_value(prompt: str, default: Any, cast):
    while True:
        val = input(f"{prompt} (default {default}): ").strip()
        if not val:
            return default
        try:
            return cast(val)
        except ValueError:
            print("Invalid value.")

def ensure_pth(filename: str) -> str:
    return filename if filename.lower().endswith(".pth") else filename + ".pth"

def action_retrieve() -> None:
    """
    Either fetch a whole batch (as before) **or** a single crystal
    using get_single_materials().
    """
    mode = ask_choice(
        "Retrieve bulk dataset or a single crystal?",
        ["Bulk dataset", "Single crystal"]
    )

    if mode == "Bulk dataset":
        
        name = input("Filename for the new dataset: ").strip()
        if not name:
            print("Filename cannot be empty.")
            return
        size = ask_value("How many records to fetch", 500, int)
        data = get_materials_data(size)

    else:  
        crystal_id = input("Materials Project ID (e.g. 12345 or mp-12345): ").strip()
        if not crystal_id:
            print("ID cannot be empty.")
            return

        # Accepts both “12345” and “mp-12345”
        crystal_id_clean = crystal_id.removeprefix("mp-")
        data = get_single_materials(crystal_id_clean)

        name = input("Filename to save the crystal (leave blank for default): ").strip()
        if not name:
            name = f"mp_{crystal_id_clean}_single"

    
    if not data:
        print("No data returned.")
        return

    save_data_local(name, data)
    print(f"Saved {len(data)} record(s) to 'DownloadedCrystalProperties/{name}.txt'")

def parse_train_args(datasets: List[str], defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-d", "--dataset", choices=datasets)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("-o", "--output", default=defaults["output"])
    p.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    p.add_argument("--epochs", type=int, default=defaults["epochs"])
    p.add_argument("--lr", type=float, default=defaults["lr"])
    p.add_argument("--hidden_channels", type=int, default=defaults["hidden_channels"])
    p.add_argument("--patience",type=int,default=defaults["patience"])
    return p.parse_args([])

def action_train() -> None:
    if not DATASET_DIR.exists():
        print("Dataset directory not found.")
        return
    available = sorted(p.stem for p in DATASET_DIR.glob("*.txt"))
    if not available:
        print("No datasets available.")
        return
    defaults = load_defaults()
    args = parse_train_args(available, defaults)
    if args.dataset is None:
        args.dataset = ask_choice("Select dataset", available)
    args.output = ensure_pth(input(f"Model filename (default {args.output}): ").strip() or args.output)
    if args.interactive or input("Alter other defaults? (y/N): ").strip().lower() == "y":
        args.batch_size = ask_value("Batch size", args.batch_size, int)
        args.epochs = ask_value("Epochs", args.epochs, int)
        args.lr = ask_value("Learning rate", args.lr, float)
        args.hidden_channels = ask_value("Hidden channels", args.hidden_channels, int)
        args.patience = ask_value("patience (early-stop epochs)", args.patience, int)
    if input("Save these values as new defaults? (y/N): ").strip().lower() == "y":
        save_defaults(vars(args))
    seed = args.seed if getattr(args, "seed", None) is not None else GLOBAL_SEED
    seed = set_seed(seed)
    print(f"Using seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data = load_data_local(args.dataset)
    train_set, val_set, test_set = split_data(raw_data)
    

    train_data = extract_label([(crystal_obj, crystal_obj.structure) for crystal_obj in train_set])
    val_data = extract_label([(crystal_obj, crystal_obj.structure) for crystal_obj in val_set])
    test_data = extract_label([(crystal_obj, crystal_obj.structure) for crystal_obj in test_set])
    
    
    train_graphs = build_graph_batch(train_data)
    val_graphs = build_graph_batch(val_data)
    test_graphs = build_graph_batch(test_data)
    
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)
    model_path = MODELS_DIR / args.output
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_dir  = model_path.parent / "meta"
    meta_name = f"{model_path.stem}_meta.json"
    params    = vars(args).copy(); params["seed"] = seed
    record_run_meta(meta_dir, filename=meta_name,
                seed=seed, params=params)
    model = CrystalGNNTransformer(num_features=train_graphs[0].num_features, hidden_channels=args.hidden_channels).to(device)
    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    patience = 15
    early_stopper =EarlyStopper(patience=args.patience, delta=1e-4)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_model(model, train_loader, optimiser, criterion, device)
        v_loss = evaluate_model(model, val_loader, criterion, device)

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model → {model_path}")

        print(f"E{epoch:03d}  train {tr_loss:.4f}  val {v_loss:.4f}")

        if early_stopper(v_loss):
            print(f"⏹  Early-stopping triggered (no val-loss improve for {args.patience} epochs)")
            break

    model.load_state_dict(torch.load(model_path))
    t_loss = evaluate_model(model, test_loader, criterion, device)
    print(f"Test loss: {t_loss:.4f}")
    evaluate_model_performance(model, test_loader, device, property_name="Target Property")

def action_infer() -> None:
    """Pick a trained model and a dataset, then print  material_id → metal / non-metal."""
    import torch
    from pathlib import Path
    from torch_geometric.data import DataLoader

    print("=== Inference ===")

    
    MODELS_DIR = Path("trained_models")
    models = sorted(p.name for p in MODELS_DIR.glob("*.pth"))
    if not models:
        print("No models found in 'trained_models/'. Train first.")
        return
    model_name = ask_choice("Select model", models)

    
    DATA_DIR = Path("DownloadedCrystalProperties")
    datasets = sorted(p.stem for p in DATA_DIR.glob("*.txt"))
    if not datasets:
        print("No datasets found in 'DownloadedCrystalProperties/'.")
        return
    dataset_name = ask_choice("Select dataset", datasets)

    
    raw_data = load_data_local(dataset_name)                  
    graphs   = build_graph_batch([(c, c.structure) for c in raw_data])
    loader   = DataLoader(graphs, batch_size=64)

    sample  = graphs[0]                                       
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    state_path = MODELS_DIR / model_name
    state      = torch.load(state_path, map_location=device)
    hidden     = state["node_encoder.weight"].shape[0]
    

    model = CrystalGNNTransformer(num_features=sample.num_features,
                       hidden_channels=hidden).to(device)
    model.load_state_dict(state)
    model.eval()

    
    preds = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            probs  = torch.sigmoid(logits).view(-1)
            preds.extend((probs > 0.5).int().cpu().tolist())

    
    print(f"\nPredicted metallicity for '{dataset_name}':")
    for crystal_obj, pred in zip(raw_data, preds):
        label = "metal" if pred else "non-metal"
        print(f"{crystal_obj.material_id:<15} → {label}")

def action_tune() -> None:
    """
    Launch Optuna hyper-parameter search for a chosen dataset.
    Search space:
        • batch_size       {16, 32, 64, 128, 256, 512}
        • learning_rate    log-uniform 1e-5 … 1e-2
        • hidden_channels  {32, 64, 128, 256, 512}
    """
    print("=== Hyper-parameter Tuning ===")

    
    if not DATASET_DIR.exists():
        print("Dataset directory not found.")
        return
    datasets = sorted(p.stem for p in DATASET_DIR.glob("*.txt"))
    if not datasets:
        print("No datasets available.")
        return

    name   = ask_choice("Select dataset", datasets)
    trials = ask_value("Number of Optuna trials", 40, int)
    jobs   = ask_value("Parallel jobs (processes)", 1, int)

    try:
        path, best = run_search(name, n_trials=trials, n_jobs=jobs)
        print("\nBest params:", best)
        print(f"Saved to {path}")
    except Exception as err:
        print(f"[ERROR] Auto-tune failed: {err}")

def action_set_seed() -> None:
    """Prompt for a seed and make it the session-wide default."""
    global GLOBAL_SEED
    txt = input("Random seed (ENTER = random): ").strip()
    GLOBAL_SEED = set_seed(int(txt) if txt else None)
    print(f"Global seed set to {GLOBAL_SEED}")

ACTIONS = {
    "Retrieve new dataset or single crystal": action_retrieve,
    "Train a model": action_train,
    "Use a model on a dataset": action_infer,
    "Hyper-parameter tuning": action_tune,
    "Set global random seed": action_set_seed,
    "Exit": None,
}

def main() -> None:
    while True:
        try:
            choice = ask_choice("What would you like to do?", list(ACTIONS.keys()))
            if choice == "Exit":
                break
            ACTIONS[choice]()
        except Exception:
            traceback.print_exc()
    print("Goodbye.")

if __name__ == "__main__":
    main()
