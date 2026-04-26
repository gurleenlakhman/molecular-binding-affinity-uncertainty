"""MLP baseline: train on Morgan fingerprints, predict pChEMBL values."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"

HIDDEN_DIMS = [512, 256]
DROPOUT = 0.2
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    print("Loading fingerprints and labels...")
    fps = np.load(DATA_DIR / "morgan_fps.npz")

    datasets = {}
    for split in ("train", "val", "test"):
        x = torch.tensor(fps[split], dtype=torch.float32)
        y = torch.tensor(
            pd.read_csv(DATA_DIR / f"{split}.csv")["pchembl"].values,
            dtype=torch.float32,
        )
        datasets[split] = TensorDataset(x, y)
        print(f"  {split}: {len(x)} samples, fingerprint dim {x.shape[1]}")

    return datasets


# ── Training ─────────────────────────────────────────────────────────────────

def train(model: MLP, loader: DataLoader, optimiser: torch.optim.Optimizer) -> float:
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimiser.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(x_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: MLP, loader: DataLoader) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        total_loss += criterion(model(x_batch), y_batch).item() * len(x_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model: MLP, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for x_batch, y_batch in loader:
        preds.append(model(x_batch.to(DEVICE)).cpu().numpy())
        targets.append(y_batch.numpy())
    return np.concatenate(preds), np.concatenate(targets)


# ── Plotting ─────────────────────────────────────────────────────────────────

def save_scatter(preds: np.ndarray, targets: np.ndarray, rmse: float, r2: float):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, alpha=0.3, s=10, color="steelblue")
    lims = [min(targets.min(), preds.min()) - 0.5, max(targets.max(), preds.max()) + 0.5]
    ax.plot(lims, lims, "r--", linewidth=1, label="ideal")
    ax.set_xlabel("Actual pChEMBL")
    ax.set_ylabel("Predicted pChEMBL")
    ax.set_title(f"MLP baseline — test set\nRMSE={rmse:.3f}  R²={r2:.3f}")
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "mlp_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Scatter plot saved -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}\n")

    datasets = load_data()
    in_dim = datasets["train"].tensors[0].shape[1]

    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=BATCH_SIZE)
    test_loader = DataLoader(datasets["test"], batch_size=BATCH_SIZE)

    model = MLP(in_dim, HIDDEN_DIMS, DROPOUT).to(DEVICE)
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

    # Test evaluation
    preds, targets = predict(model, test_loader)
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    r2 = float(r2_score(targets, preds))
    rho, _ = spearmanr(targets, preds)

    print("── Test set metrics ──────────────────")
    print(f"  RMSE:              {rmse:.4f}")
    print(f"  R²:                {r2:.4f}")
    print(f"  Spearman ρ:        {rho:.4f}")

    model_path = RESULTS_DIR / "mlp_baseline.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved -> {model_path}")

    save_scatter(preds, targets, rmse, r2)


if __name__ == "__main__":
    main()
