"""Scaffold split of egfr_raw.csv into train/val/test sets."""

import pathlib
import random
from collections import defaultdict

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
RANDOM_SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1  # test = remaining 0.1


def get_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(df: pd.DataFrame, train_frac: float, val_frac: float, seed: int):
    # Group indices by scaffold
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    invalid = []
    for idx, smiles in enumerate(df["canonical_smiles"]):
        scaffold = get_scaffold(smiles)
        if scaffold is None:
            invalid.append(idx)
        else:
            scaffold_to_indices[scaffold].append(idx)

    if invalid:
        print(f"  Warning: {len(invalid)} molecules with invalid SMILES excluded.")

    # Shuffle scaffold groups deterministically
    rng = random.Random(seed)
    groups = list(scaffold_to_indices.values())
    rng.shuffle(groups)

    n_total = sum(len(g) for g in groups)
    train_cutoff = int(n_total * train_frac)
    val_cutoff = int(n_total * (train_frac + val_frac))

    train_idx, val_idx, test_idx = [], [], []
    running = 0
    for group in groups:
        if running < train_cutoff:
            train_idx.extend(group)
        elif running < val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
        running += len(group)

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def main():
    raw_path = DATA_DIR / "egfr_raw.csv"
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} molecules from {raw_path}")

    train, val, test = scaffold_split(df, TRAIN_FRAC, VAL_FRAC, RANDOM_SEED)

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        out_path = DATA_DIR / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"  {split_name}: {len(split_df)} molecules -> {out_path}")


if __name__ == "__main__":
    main()
