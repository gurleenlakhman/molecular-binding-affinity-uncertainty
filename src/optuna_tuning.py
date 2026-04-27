"""Optuna hyperparameter search for the GAT v2 model."""

import pathlib

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"

NODE_IN_DIM = 7
TRAIN_EPOCHS = 50
N_TRIALS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ────────────────────────────────────────────────────────────────────

class GATRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.mlp(x).squeeze(-1)


# ── Data ─────────────────────────────────────────────────────────────────────

def load_splits() -> tuple[list, list]:
    train = torch.load(DATA_DIR / "graphs_v2_train.pt", weights_only=False)
    val = torch.load(DATA_DIR / "graphs_v2_val.pt", weights_only=False)
    return train, val


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch(model: GATRegressor, loader: DataLoader, optimiser: torch.optim.Optimizer) -> None:
    model.train()
    criterion = nn.MSELoss()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimiser.zero_grad()
        loss = criterion(model(batch.x, batch.edge_index, batch.batch), batch.y.squeeze(-1))
        loss.backward()
        optimiser.step()


@torch.no_grad()
def evaluate(model: GATRegressor, loader: DataLoader) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        total += criterion(model(batch.x, batch.edge_index, batch.batch), batch.y.squeeze(-1)).item() * batch.num_graphs
        n += batch.num_graphs
    return total / n


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(train_graphs: list, val_graphs: list):
    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        heads = trial.suggest_categorical("heads", [2, 4, 8])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=batch_size)

        model = GATRegressor(NODE_IN_DIM, hidden_dim, heads, dropout).to(DEVICE)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(TRAIN_EPOCHS):
            train_epoch(model, train_loader, optimiser)
            val_mse = evaluate(model, val_loader)
            trial.report(val_mse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return evaluate(model, val_loader)

    return objective


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Running {N_TRIALS} Optuna trials ({TRAIN_EPOCHS} epochs each)...\n")

    train_graphs, val_graphs = load_splits()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(make_objective(train_graphs, val_graphs), n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nBest val MSE:  {study.best_value:.4f}")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Optimization history
    axes = plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "optuna_history.png", dpi=150)
    plt.close()
    print(f"\nHistory plot saved    -> {RESULTS_DIR / 'optuna_history.png'}")

    # Hyperparameter importance
    # Hyperparameter importance
    try:
        axes = plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "optuna_importance.png", dpi=150)
        plt.close()
        print(f"Importance plot saved -> {RESULTS_DIR / 'optuna_importance.png'}")
    except ValueError as e:
        print(f"Skipping importance plot: {e}")

if __name__ == "__main__":
    main()
