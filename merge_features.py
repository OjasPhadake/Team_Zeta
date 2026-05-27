"""
merge_features.py
=================
Merge ntl_features.csv and worldpop_features.csv on pincode.

Input:
  ntl_features.csv          — 19,281 pincodes, 15 columns (14 features + industrial_flag)
  worldpop_features.csv     — 19,313 pincodes, 20 columns (16 features + 2 flags + area)

Output:
  merged_features.csv       — 19,313 pincodes, 34 columns (left join on worldpop)

Logic:
  - Left join on worldpop (keep all pincodes with population data)
  - Missing NTL features → filled with 0 (indicates no nighttime activity)
  - Missing NTL flags → set to 0 (no contamination if no NTL data)
"""

import pandas as pd
from pathlib import Path

SEP = "=" * 70

print(f"{SEP}")
print("  MERGE: NTL Features + WorldPop Features")
print(f"{SEP}\n")

# Load both CSVs
print("  Loading ntl_features.csv ...")
ntl_df = pd.read_csv("ntl_features.csv")
print(f"    → {len(ntl_df):,} pincodes, {len(ntl_df.columns)} columns")

print("  Loading worldpop_features.csv ...")
pop_df = pd.read_csv("worldpop_features.csv")
print(f"    → {len(pop_df):,} pincodes, {len(pop_df.columns)} columns")

# Left join: keep all worldpop pincodes (all have population, not all have NTL)
print(f"\n  Left join on pincode (worldpop as base) ...")
merged = pop_df.merge(ntl_df, on="pincode", how="left")
print(f"    → {len(merged):,} pincodes after merge")

# Check how many pincodes are missing NTL data
ntl_missing = merged["log_ntl_sol"].isna().sum()
print(f"    → {ntl_missing:,} pincodes missing NTL data ({100*ntl_missing/len(merged):.1f}%)")

# Fill NTL features with 0 (no nighttime lights = 0, not missing)
ntl_cols = [c for c in merged.columns if c.startswith("log_ntl") or
            c in ["lit_pixel_ratio", "ntl_sol_per_pixel", "spike_ratio",
                  "p95_to_median_ratio", "cv_ntl", "lit_pixel_count", "total_pixel_count"]]

print(f"\n  Filling missing NTL features with 0 (no light activity) ...")
merged[ntl_cols] = merged[ntl_cols].fillna(0)

# Fill NTL flags with 0 (no flag if no NTL data)
merged["industrial_flag"] = merged["industrial_flag"].fillna(0).astype(int)

print(f"    → {len(ntl_cols)} NTL feature columns filled")

# Verify no missing values remain
n_missing = merged.isna().sum().sum()
print(f"\n  Missing value check: {n_missing} missing values in merged dataframe")

# Column order for clarity
log_cols = [c for c in merged.columns if c.startswith("log_")]
ntl_cols = [c for c in merged.columns if c.startswith("ntl_")]
pop_cols = [c for c in merged.columns if c.startswith("pop_")]
other_cols = [c for c in merged.columns if c not in log_cols + ntl_cols + pop_cols
              and c != "pincode" and c != "area_km2"
              and c not in ["industrial_flag", "sparse_sample_flag", "uninhabited_flag"]]

col_order = (
    ["pincode", "area_km2"] +
    log_cols +
    pop_cols +
    ntl_cols +
    other_cols +
    ["industrial_flag", "sparse_sample_flag", "uninhabited_flag"]
)
merged = merged[col_order]

print(f"\n  Column ordering: {len(merged.columns)} columns")
print(f"    First 5 cols: {merged.columns[:5].tolist()}")
print(f"    Last 5 cols:  {merged.columns[-5:].tolist()}")

# Save
out_path = Path("merged_features.csv")
merged.to_csv(out_path, index=False)
size_mb = out_path.stat().st_size / 1024 / 1024

print(f"\n{SEP}")
print(f"  Saved → {out_path.name}")
print(f"  Size: {len(merged):,} rows × {len(merged.columns)} cols ({size_mb:.1f} MB)")
print(f"{SEP}")
