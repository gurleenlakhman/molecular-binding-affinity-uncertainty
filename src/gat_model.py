"""Graph Attention Network: train on PyG molecular graphs, predict pChEMBL values."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"

NODE_IN_DIM = 3       # [atomic_num, degree, is_aromatic]
HIDDEN_DIM = 128
NUM_HEADS = 4
DROPOUT = 0.2
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ────────────────────────────────────────────────────────────────────

class GATRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.dropout = dropout

        # Three GATConv layers; first two concat heads, last averages
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, dropout=dropout)

        # 2-layer MLP head
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


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> dict[str, list]:
    print("Loading molecular graphs...")
    datasets = {}
    for split in ("train", "val", "test"):
        graphs = torch.load(DATA_DIR / f"graphs_{split}.pt", weights_only=False)
        datasets[split] = graphs
        print(f"  {split}: {len(graphs)} graphs")
    return datasets


# ── Training ─────────────────────────────────────────────────────────────────

def train(model: GATRegressor, loader: DataLoader, optimiser: torch.optim.Optimizer) -> float:
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimiser.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(pred, batch.y.squeeze(-1))
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: GATRegressor, loader: DataLoader) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch.x, batch.edge_index, batch.batch)
        total_loss += criterion(pred, batch.y.squeeze(-1)).item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model: GATRegressor, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        preds.append(model(batch.x, batch.edge_index, batch.batch).cpu().numpy())
        targets.append(batch.y.squeeze(-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


# ── Plotting ─────────────────────────────────────────────────────────────────

def save_scatter(preds: np.ndarray, targets: np.ndarray, rmse: float, r2: float):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, alpha=0.3, s=10, color="steelblue")
    lims = [min(targets.min(), preds.min()) - 0.5, max(targets.max(), preds.max()) + 0.5]
    ax.plot(lims, lims, "r--", linewidth=1, label="ideal")
    ax.set_xlabel("Actual pChEMBL")
    ax.set_ylabel("Predicted pChEMBL")
    ax.set_title(f"GAT — test set\nRMSE={rmse:.3f}  R²={r2:.3f}")
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "gat_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Scatter plot saved -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}\n")

    datasets = load_data()

    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=BATCH_SIZE)
    test_loader = DataLoader(datasets["test"], batch_size=BATCH_SIZE)

    model = GATRegressor(NODE_IN_DIM, HIDDEN_DIM, NUM_HEADS, DROPOUT).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters\n")
    print(f"{'Epoch':>5}  {'Train MSE':>10}  {'Val MSE':>10}")
    print("-" * 30)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimiser)
        val_loss = evaluate(model, val_loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}")

    print()

    preds, targets = predict(model, test_loader)
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    r2 = float(r2_score(targets, preds))
    rho, _ = spearmanr(targets, preds)

    print("── Test set metrics ──────────────────")
    print(f"  RMSE:              {rmse:.4f}")
    print(f"  R²:                {r2:.4f}")
    print(f"  Spearman ρ:        {rho:.4f}")

    model_path = RESULTS_DIR / "gat_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved -> {model_path}")

    save_scatter(preds, targets, rmse, r2)


if __name__ == "__main__":
    main()
