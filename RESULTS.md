# Results

## Model Comparison

All models are evaluated on the scaffold test set (1,329 molecules). The MLP and GAT v1 were trained for 100 epochs; GAT v2 was trained for 200 epochs. All training was done on CPU.

| Model | Features | RMSE | R² | Spearman ρ |
|---|---|---|---|---|
| MLP baseline | Morgan FP (2048-bit) | 0.93 | 0.65 | 0.78 |
| GAT v1 | 3 atom features | 1.10 | 0.52 | 0.67 |
| GAT v2 | 7 atom features | 1.05 | 0.56 | 0.72 |

The MLP baseline outperforms both GAT variants under CPU-constrained training. Expanding atom features from 3 to 7 (adding formal charge, hydrogen counts, and ring membership) consistently improves all three metrics for the GAT.

## Uncertainty Quantification

MC Dropout UQ uses N=50 stochastic forward passes with dropout active at inference time. Predictive standard deviation is used as the uncertainty estimate.

**MLP MC Dropout**

| Group | Mean std | MAE |
|---|---|---|
| Top 25% most uncertain | 0.47 | 0.72 |
| Bottom 25% least uncertain | 0.21 | 0.68 |
| All test molecules | 0.34 | — |

**GAT v2 MC Dropout**

| Metric | Value |
|---|---|
| Mean predictive std | 0.45 |
| Std of predictive std | 0.14 |

High-uncertainty molecules have higher MAE than low-uncertainty molecules for the MLP (0.72 vs 0.68), confirming that the uncertainty estimates carry signal. The gap is modest, reflecting limitations of MC Dropout calibration discussed in DISCUSSION.md.
