"""Featurize SMILES into Morgan fingerprints and PyG molecular graphs (v2: 7 atom features)."""

import pathlib

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import Data

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS
)

BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


# ── Morgan fingerprints ──────────────────────────────────────────────────────

def smiles_to_morgan(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = _MORGAN_GEN.GetFingerprint(mol)
    return np.array(fp, dtype=np.uint8)


def build_morgan_arrays(
    dfs: dict[str, pd.DataFrame],
) -> dict[str, np.ndarray]:
    arrays = {}
    for split, df in dfs.items():
        print(f"  [{split}] computing Morgan fingerprints for {len(df)} molecules...")
        fps, skipped = [], 0
        for smiles in df["canonical_smiles"]:
            fp = smiles_to_morgan(smiles)
            if fp is None:
                skipped += 1
            else:
                fps.append(fp)
        if skipped:
            print(f"    Warning: {skipped} invalid SMILES skipped.")
        arrays[split] = np.stack(fps)
        print(f"    -> shape {arrays[split].shape}")
    return arrays


# ── PyG graphs (v2: 7 atom features) ────────────────────────────────────────

def smiles_to_graph(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features: [atomic_num, degree, is_aromatic,
    #                 formal_charge, num_explicit_hs, num_implicit_hs, is_in_ring]
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            int(atom.GetIsAromatic()),
            atom.GetFormalCharge(),
            atom.GetNumExplicitHs(),
            atom.GetNumImplicitHs(),
            int(atom.IsInRing()),
        ])
    x = torch.tensor(node_feats, dtype=torch.float)

    # Edge index and edge attributes (undirected: add both directions)
    edge_index_list, edge_attr_list = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
        edge_index_list += [[i, j], [j, i]]
        edge_attr_list += [[bond_type], [bond_type]]

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_graphs(df: pd.DataFrame, split: str) -> list[Data]:
    print(f"  [{split}] building graphs for {len(df)} molecules...")
    graphs, skipped = [], 0
    for smiles, pchembl in zip(df["canonical_smiles"], df["pchembl"]):
        g = smiles_to_graph(smiles)
        if g is None:
            skipped += 1
            continue
        g.y = torch.tensor([pchembl], dtype=torch.float)
        graphs.append(g)
    if skipped:
        print(f"    Warning: {skipped} invalid SMILES skipped.")
    print(f"    -> {len(graphs)} graphs built")
    return graphs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    splits = ["train", "val", "test"]

    print("Loading CSVs...")
    dfs = {s: pd.read_csv(DATA_DIR / f"{s}.csv") for s in splits}
    for s, df in dfs.items():
        print(f"  {s}: {len(df)} rows")

    # PyG graphs with expanded atom features
    print("\nBuilding PyG molecular graphs (v2: 7 atom features)...")
    for split in splits:
        graphs = build_graphs(dfs[split], split)
        out_path = DATA_DIR / f"graphs_v2_{split}.pt"
        torch.save(graphs, out_path)
        print(f"Saved {len(graphs)} graphs -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
