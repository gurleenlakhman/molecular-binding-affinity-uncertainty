# Discussion

## 1. GAT vs MLP

The GAT underperformed the MLP baseline (RMSE 1.10 vs 0.93) despite longer training. Atom features were minimal — only atomic number, degree, and aromaticity — which limits the structural information available to the attention layers. Richer features like formal charge, hydrogen count, and ring membership would likely close the gap. Val MSE was still decreasing at epoch 200, suggesting the model had not fully converged. With GPU training and early stopping, results would improve significantly given that the GAT has a stronger structural inductive bias than a fingerprint-based MLP for capturing local chemical environments.

## 2. GAT v2 vs v1

Expanding atom features from 3 to 7 improved RMSE from 1.10 to 1.05, R² from 0.52 to 0.56, and Spearman ρ from 0.67 to 0.72. This confirms that richer atom featurization is a meaningful lever even without changing the model architecture. The MLP still outperforms both GAT variants, likely due to CPU-constrained training preventing full convergence. With GPU training and early stopping, the GAT would likely surpass the MLP given its ability to reason over molecular topology rather than a fixed-length hashed fingerprint.

## 3. Uncertainty Calibration

MC Dropout uncertainty correlates weakly with error (MAE 0.72 for high-uncertainty vs 0.68 for low-uncertainty molecules for the MLP) — the gap is small, suggesting the uncertainty estimates are not well calibrated in an absolute sense. MC Dropout is a computationally cheap approximation to Bayesian inference and is known to underestimate uncertainty and produce poorly calibrated intervals. Better calibration could be achieved with deep ensembles, which average over independently trained models with different random seeds, or with conformal prediction, which provides distribution-free coverage guarantees without any distributional assumptions.

## 4. Scaffold Split vs Random Split

Using a scaffold split ensures the test set contains structurally novel molecules, making evaluation more realistic for drug discovery applications where models must generalize to new chemical series. A random split would inflate all metrics significantly because structurally similar molecules would leak between train and test sets, making the evaluation measure memorization rather than generalization. The scaffold split gives a harder and more honest benchmark.

## 5. Dataset Bias

ChEMBL skews toward potent, publishable compounds — weak binders and inactive molecules are systematically underrepresented because negative results are rarely reported. This publication bias means the model is trained on a non-representative slice of chemical space and may perform poorly on truly novel scaffolds with unknown or low activity. Incorporating decoy molecules or inactive compounds from sources like PubChem BioAssay could mitigate this.

## 6. Future Work

- GPU training with early stopping for full GAT convergence
- Deeper atom featurization (chirality, hybridization, partial charge) and bond featurization (conjugation, stereo) as used in established benchmarks like MoleculeNet
- Graph Transformer architectures (e.g. GPS, Graphormer) that combine local message passing with global attention
- Conformal prediction wrappers for rigorous, coverage-guaranteed uncertainty intervals
- Multi-target generalization: training across multiple kinases to test whether shared representations improve both accuracy and uncertainty calibration
