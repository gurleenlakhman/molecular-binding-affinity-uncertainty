"""UQ analysis for the MLP baseline: uncertainty bands and sanity-check plots."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"

HIDDEN_DIMS = [512, 256]
DROPOUT = 0.2
BATCH_SIZE = 128
N_PASSES = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model (must match mlp_baseline.py exactly) ────────────────────────────────

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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_data():
    fps = np.load(DATA_DIR / "morgan_fps.npz")
    x = torch.tensor(fps["test"], dtype=torch.float32)
    df = pd.read_csv(DATA_DIR / "test.csv")
    y = torch.tensor(df["pchembl"].values, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE), y.numpy()


# ── MC Dropout inference ──────────────────────────────────────────────────────

@torch.no_grad()
def mc_predict(model: MLP, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.train()  # keep dropout active
    all_passes = []
    for _ in range(N_PASSES):
        preds = []
        for x_batch, _ in loader:
            preds.append(model(x_batch.to(DEVICE)).cpu().numpy())
        all_passes.append(np.concatenate(preds))
    samples = np.stack(all_passes)          # (N_PASSES, n_samples)
    return samples.mean(axis=0), samples.std(axis=0)


# ── Plot 1: uncertainty bands sorted by actual pChEMBL ───────────────────────

def save_bands_plot(means: np.ndarray, targets: np.ndarray, stds: np.ndarray):
    order = np.argsort(targets)
    x = np.arange(len(order))
    t = targets[order]
    m = means[order]
    s = stds[order]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, t, color="black", linewidth=1, label="Actual pChEMBL")
    ax.plot(x, m, color="steelblue", linewidth=1, label="Mean prediction")
    ax.fill_between(x, m - s, m + s, alpha=0.3, color="steelblue", label="±1 std")
    ax.set_xlabel("Molecules (sorted by actual pChEMBL)")
    ax.set_ylabel("pChEMBL")
    ax.set_title(f"MLP MC Dropout (N={N_PASSES}) — uncertainty bands")
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "uq_mlp_bands.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Bands plot saved -> {out}")


# ── Plot 2: error distribution by uncertainty quartile ───────────────────────

def save_sanity_plot(means: np.ndarray, targets: np.ndarray, stds: np.ndarray):
    errors = means - targets
    threshold_high = np.percentile(stds, 75)
    threshold_low = np.percentile(stds, 25)

    high_unc_errors = errors[stds >= threshold_high]
    low_unc_errors = errors[stds <= threshold_low]

    mae_high = np.abs(high_unc_errors).mean()
    mae_low = np.abs(low_unc_errors).mean()

    print(f"MAE — top 25% most uncertain:    {mae_high:.4f}")
    print(f"MAE — bottom 25% least uncertain: {mae_low:.4f}")

    bins = np.linspace(errors.min() - 0.1, errors.max() + 0.1, 40)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    axes[0].hist(high_unc_errors, bins=bins, color="tomato", edgecolor="white", linewidth=0.4)
    axes[0].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_title(f"Top 25% most uncertain\nMAE = {mae_high:.3f}")
    axes[0].set_xlabel("Prediction error (mean pred − actual)")
    axes[0].set_ylabel("Count")

    axes[1].hist(low_unc_errors, bins=bins, color="steelblue", edgecolor="white", linewidth=0.4)
    axes[1].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_title(f"Bottom 25% least uncertain\nMAE = {mae_low:.3f}")
    axes[1].set_xlabel("Prediction error (mean pred − actual)")

    fig.suptitle(f"MLP MC Dropout (N={N_PASSES}) — error by uncertainty group", y=1.01)
    plt.tight_layout()
    out = RESULTS_DIR / "uq_mlp_sanity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sanity plot saved  -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}\n")

    loader, targets = load_test_data()
    in_dim = next(iter(loader))[0].shape[1]

    model = MLP(in_dim, HIDDEN_DIMS, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(RESULTS_DIR / "mlp_baseline.pt", map_location=DEVICE, weights_only=True))
    print(f"Loaded weights from results/mlp_baseline.pt")
    print(f"Running {N_PASSES} MC Dropout forward passes on {len(targets)} test molecules...\n")

    means, stds = mc_predict(model, loader)

    save_bands_plot(means, targets, stds)
    print()
    save_sanity_plot(means, targets, stds)


if __name__ == "__main__":
    main()
