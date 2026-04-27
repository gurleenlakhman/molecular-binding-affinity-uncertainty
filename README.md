# Molecular Binding Affinity Uncertainty

This project builds and evaluates machine learning models for predicting the binding affinity (pChEMBL) of small molecules against EGFR (Epidermal Growth Factor Receptor, ChEMBL target CHEMBL203), a clinically important kinase in cancer biology. Compounds and activity data are sourced from the ChEMBL database. Two model families are compared: a Morgan fingerprint MLP baseline and a Graph Attention Network (GAT) that operates directly on molecular graphs, using both 3-feature and 7-feature atom representations. Uncertainty quantification (UQ) is performed on both models using MC Dropout, which approximates a Bayesian posterior by running multiple stochastic forward passes at inference time and measuring the variance across predictions.

## Dataset

- **Target:** EGFR (CHEMBL203)
- **Source:** ChEMBL 33 via the ChEMBL Python client
- **Size:** 13,286 molecules with pChEMBL values
- **Split:** Scaffold split (80 / 10 / 10 train / val / test) using RDKit Bemis–Murcko scaffolds, ensuring the test set contains structurally novel molecules

## Setup

```bash
conda env create -f environment.yml
conda activate molprop
```

## Reproducing Results

Run scripts in this order from the project root:

1. `python -m src.data_pull` — fetch EGFR activity data from ChEMBL
2. `python -m src.scaffold_split` — split into train / val / test by scaffold
3. `python -m src.featurize` — compute Morgan fingerprints and 3-feature PyG graphs
4. `python -m src.featurize_v2` — compute 7-feature PyG graphs
5. `python -m src.mlp_baseline` — train MLP on Morgan fingerprints
6. `python -m src.gat_model` — train GAT v1 (3 atom features)
7. `python -m src.gat_model_v2` — train GAT v2 (7 atom features)
8. `python -m src.uq_mlp` — MC Dropout UQ on MLP
9. `python -m src.uq_gat` — MC Dropout UQ on GAT v2
10. `python -m src.uq_analysis` — uncertainty band and sanity-check plots for MLP
11. `python -m src.optuna_tuning` — Optuna hyperparameter search for GAT v2

## Project Structure

```
molprop-uq/
├── data/
│   ├── egfr_raw.csv
│   ├── train.csv / val.csv / test.csv
│   ├── morgan_fps.npz
│   ├── graphs_train.pt / graphs_val.pt / graphs_test.pt
│   └── graphs_v2_train.pt / graphs_v2_val.pt / graphs_v2_test.pt
├── results/
│   ├── mlp_baseline.pt
│   ├── mlp_scatter.png
│   ├── gat_model.pt
│   ├── gat_scatter.png
│   ├── gat_model_v2.pt
│   ├── gat_scatter_v2.png
│   ├── uq_mlp_uncertainty.png
│   ├── uq_mlp_bands.png
│   ├── uq_mlp_sanity.png
│   ├── uq_gat_uncertainty.png
│   ├── optuna_history.png
│   └── optuna_importance.png
├── src/
│   ├── data_pull.py
│   ├── scaffold_split.py
│   ├── featurize.py
│   ├── featurize_v2.py
│   ├── mlp_baseline.py
│   ├── gat_model.py
│   ├── gat_model_v2.py
│   ├── uq_mlp.py
│   ├── uq_gat.py
│   ├── uq_analysis.py
│   └── optuna_tuning.py
├── notebooks/
│   └── 01_eda.ipynb
├── environment.yml
└── requirements.txt
```
