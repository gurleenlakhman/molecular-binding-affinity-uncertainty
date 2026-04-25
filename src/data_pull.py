import os
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client

TARGET_CHEMBL_ID = "CHEMBL203"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "egfr_raw.csv")


def main():
    # --- Fetch ---
    print(f"Querying ChEMBL for IC50 activities on {TARGET_CHEMBL_ID} (EGFR)...")
    records = list(
        new_client.activity.filter(
            target_chembl_id=TARGET_CHEMBL_ID,
            standard_type="IC50",
        ).only(
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
            "standard_units",
            "target_organism",
            "target_type",
            "assay_chembl_id",
        )
    )
    df = pd.DataFrame(records)
    print(f"  Retrieved {len(df)} raw records.")

    # --- Filter: human + single protein ---
    print("\nFiltering for human, single-protein targets...")
    before = len(df)
    df = df[df["target_organism"].str.lower() == "homo sapiens"]
    print(f"  Human only:        {len(df)} rows (dropped {before - len(df)})")

    before = len(df)
    
    # --- Filter: nM units only (required for the pChEMBL formula) ---
    before = len(df)
    df = df[df["standard_units"] == "nM"]
    print(f"  nM units only:     {len(df)} rows (dropped {before - len(df)})")

    # --- Convert IC50 -> pChEMBL ---
    print("\nConverting IC50 (nM) to pChEMBL = -log10(IC50 / 1e9)...")
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df["pchembl"] = -np.log10(df["standard_value"] / 1e9)
    # log10 of zero or negative is undefined
    df.loc[df["standard_value"] <= 0, "pchembl"] = np.nan
    print(f"  pChEMBL computed for {df['pchembl'].notna().sum()} rows.")

    # --- Drop nulls ---
    print("\nRemoving nulls and duplicates...")
    before = len(df)
    df = df.dropna(subset=["canonical_smiles", "pchembl"])
    print(f"  After null drop:   {len(df)} rows (dropped {before - len(df)})")

    # --- Deduplicate on SMILES (keep highest pChEMBL) ---
    before = len(df)
    df = df.sort_values("pchembl", ascending=False).drop_duplicates(subset=["canonical_smiles"])
    print(f"  After dedup:       {len(df)} rows (dropped {before - len(df)})")

    # --- Save ---
    print(f"\nSaving to {os.path.abspath(OUTPUT_PATH)}...")
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    df = df[["molecule_chembl_id", "canonical_smiles", "standard_value", "standard_units", "pchembl", "assay_chembl_id"]].reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDone. {len(df)} compounds saved.")
    print(f"pChEMBL range: {df['pchembl'].min():.2f} – {df['pchembl'].max():.2f}")
    print(f"pChEMBL mean:  {df['pchembl'].mean():.2f}")


if __name__ == "__main__":
    main()
