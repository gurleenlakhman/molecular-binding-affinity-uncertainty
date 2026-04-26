"""MC Dropout uncertainty quantification on the trained MLP baseline."""

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
    smiles = df["canonical_smiles"].tolist()
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE), smiles, y.numpy()


# ── MC Dropout inference ──────────────────────────────────────────────────────

@torch.no_grad()
def mc_predict(model: MLP, loader: DataLoader, n_passes: int) -> tuple[np.ndarray, np.ndarray]:
    model.train()  # keep dropout active
    all_passes = []
    for _ in range(n_passes):
        preds = []
        for x_batch, _ in loader:
            preds.append(model(x_batch.to(DEVICE)).cpu().numpy())
        all_passes.append(np.concatenate(preds))
    samples = np.stack(all_passes)          # (N_PASSES, n_samples)
    return samples.mean(axis=0), samples.std(axis=0)


# ── Plotting ──────────────────────────────────────────────────────────────────

def save_uncertainty_plot(means: np.ndarray, targets: np.ndarray, stds: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(targets, means, c=stds, cmap="plasma", alpha=0.5, s=12)
    lims = [min(targets.min(), means.min()) - 0.5, max(targets.max(), means.max()) + 0.5]
    ax.plot(lims, lims, "k--", linewidth=1, label="ideal")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Predictive std (MC Dropout)")
    ax.set_xlabel("Actual pChEMBL")
    ax.set_ylabel("Mean predicted pChEMBL")
    ax.set_title(f"MLP MC Dropout (N={N_PASSES}) — test set")
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "uq_mlp_uncertainty.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Uncertainty plot saved -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}\n")

    loader, smiles, targets = load_test_data()
    in_dim = next(iter(loader))[0].shape[1]

    model = MLP(in_dim, HIDDEN_DIMS, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(RESULTS_DIR / "mlp_baseline.pt", map_location=DEVICE, weights_only=True))
    print(f"Loaded weights from results/mlp_baseline.pt")
    print(f"Running {N_PASSES} MC Dropout forward passes on {len(targets)} test molecules...\n")

    means, stds = mc_predict(model, loader, N_PASSES)

    print(f"Mean predictive std (uncertainty): {stds.mean():.4f}")
    print(f"Std of predictive std:             {stds.std():.4f}\n")

    save_uncertainty_plot(means, targets, stds)

    df_results = pd.DataFrame({
        "smiles": smiles,
        "actual_pchembl": targets,
        "mean_pred": means,
        "uncertainty_std": stds,
    })

    most_uncertain = df_results.nlargest(5, "uncertainty_std")
    least_uncertain = df_results.nsmallest(5, "uncertainty_std")

    print("── 5 most uncertain molecules ────────────────────────────")
    for _, row in most_uncertain.iterrows():
        print(f"  std={row.uncertainty_std:.4f}  actual={row.actual_pchembl:.2f}  {row.smiles}")

    print("\n── 5 least uncertain molecules ───────────────────────────")
    for _, row in least_uncertain.iterrows():
        print(f"  std={row.uncertainty_std:.4f}  actual={row.actual_pchembl:.2f}  {row.smiles}")


if __name__ == "__main__":
    main()
