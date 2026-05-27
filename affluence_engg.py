"""
affluence_engg.py
=================
Compute affluence ranking from merged NTL + WorldPop features.

CLI Flags:
  --mode {default,econ,urban,relative}
    default    : Balanced weighting across all signals
    econ       : Emphasize economic activity (NTL-heavy)
    urban      : Emphasize urban development (density + organized lights)
    relative   : Emphasize relative peer standing (regional percentiles)

Input: merged_features.csv
Output: affluence_ranking.csv

Structure:
  Step 1: Create 7 cross-signal features (NTL-WorldPop interactions)
  Step 2: Compute affluence scores with mode-specific weights
  Step 3: Rank pincodes by affluence percentile
  Step 4: Assign confidence scores (downweight contaminated pincodes)
  Step 5: Save output with all signals, scores, and flags
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import argparse

SEP = "=" * 70

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Compute affluence ranking from NTL + WorldPop features"
)
parser.add_argument(
    "--mode",
    choices=["default", "econ", "urban", "relative"],
    default="default",
    help="Weighting mode for affluence scoring",
)
args = parser.parse_args()
mode = args.mode

print(f"{SEP}")
print(f"  AFFLUENCE RANKING: {mode.upper()} MODE")
print(f"{SEP}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Load merged features
# ─────────────────────────────────────────────────────────────────────────────
print("  Loading merged_features.csv ...")
merged = pd.read_csv("merged_features.csv")
print(f"    → {len(merged):,} pincodes, {len(merged.columns)} columns\n")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Cross-signal features
# ─────────────────────────────────────────────────────────────────────────────
print(f"{SEP}")
print("  Step 1 — Cross-signal features")
print(f"{SEP}\n")

agg = merged[["pincode", "area_km2", "industrial_flag", "sparse_sample_flag",
              "uninhabited_flag", "pop_density_state_pct", "pop_sum_state_pct"]].copy()

# 1a. NTL per capita: NTL intensity per person
print("  1a. ntl_per_capita (NTL per person) ...")
agg["ntl_per_capita"] = (
    np.exp(merged["log_ntl_sol"]) / (np.exp(merged["log_pop_sum"]) + 1)
).clip(0, 100)
print(f"      mean={agg['ntl_per_capita'].mean():.6f}  max={agg['ntl_per_capita'].max():.6f}")

# 1b. Urban intensity: joint urbanization (lights + density)
print("  1b. urban_intensity (lights + density) ...")
agg["urban_intensity"] = (
    merged["log_ntl_p90"] + merged["log_pop_density"]
) / 2
print(f"      mean={agg['urban_intensity'].mean():.4f}  max={agg['urban_intensity'].max():.4f}")

# 1c. Slum risk score: high density + low lights
print("  1c. slum_risk_score (density × anti-lights) ...")
agg["slum_risk_score"] = (
    merged["log_pop_density"] * (1.0 - merged["log_ntl_p90"])
).clip(0, 100)
print(f"      mean={agg['slum_risk_score'].mean():.4f}  max={agg['slum_risk_score'].max():.4f}")

# 1d. Settlement quality: light intensity per unit population variation
print("  1d. settlement_quality (lights per pop variation) ...")
agg["settlement_quality"] = (
    merged["log_ntl_p90"] / (merged["pop_cv"] + 0.5)
).clip(0, 100)
print(f"      mean={agg['settlement_quality'].mean():.4f}  max={agg['settlement_quality'].max():.4f}")

# 1e. Development gap: relative regional prosperity (NTL vs local peers)
print("  1e. development_gap (NTL vs regional average) ...")
agg["development_gap"] = (
    merged["log_ntl_p90"] - merged["pop_density_state_pct"]
).clip(-1, 2)
print(f"      mean={agg['development_gap'].mean():.4f}  max={agg['development_gap'].max():.4f}")

# 1f. Economic density: economic activity per km²
print("  1f. economic_density (NTL per area) ...")
agg["economic_density"] = (
    np.exp(merged["log_ntl_sol"]) / (agg["area_km2"] + 1)
).clip(0, 1000)
print(f"      mean={agg['economic_density'].mean():.4f}  max={agg['economic_density'].max():.4f}")

# 1g. Density-light ratio: population per unit NTL (high=slum, low=sparse/industrial)
print("  1g. density_light_ratio (population per NTL) ...")
agg["density_light_ratio"] = (
    merged["log_pop_density"] / (merged["log_ntl_median"] + 0.5)
).clip(0, 100)
print(f"      mean={agg['density_light_ratio'].mean():.4f}  max={agg['density_light_ratio'].max():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Affluence scoring with mode-specific weights
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  Step 2 — Affluence scoring ({mode.upper()} mode)")
print(f"{SEP}\n")

# Define weights for each mode
weights = {
    "default": {
        "ntl_per_capita": 0.25,
        "urban_intensity": 0.20,
        "settlement_quality": 0.15,
        "development_gap": 0.15,
        "economic_density": 0.10,
        "slum_risk_score": -0.15,
    },
    "econ": {
        "ntl_per_capita": 0.35,
        "urban_intensity": 0.15,
        "settlement_quality": 0.10,
        "development_gap": 0.10,
        "economic_density": 0.15,
        "slum_risk_score": -0.10,
    },
    "urban": {
        "ntl_per_capita": 0.15,
        "urban_intensity": 0.30,
        "settlement_quality": 0.20,
        "development_gap": 0.10,
        "economic_density": 0.10,
        "slum_risk_score": -0.20,
    },
    "relative": {
        "ntl_per_capita": 0.20,
        "urban_intensity": 0.15,
        "settlement_quality": 0.15,
        "development_gap": 0.30,
        "economic_density": 0.10,
        "slum_risk_score": -0.10,
    },
}

w = weights[mode]
print(f"  Weights for {mode.upper()} mode:")
for feat, weight in w.items():
    sign = "+" if weight > 0 else ""
    print(f"    {sign}{weight:5.2f}  × {feat}")

# Compute affluence score
print(f"\n  Computing affluence_raw ...")
agg["affluence_raw"] = (
    w["ntl_per_capita"] * agg["ntl_per_capita"]
    + w["urban_intensity"] * agg["urban_intensity"]
    + w["settlement_quality"] * agg["settlement_quality"]
    + w["development_gap"] * agg["development_gap"]
    + w["economic_density"] * agg["economic_density"]
    + w["slum_risk_score"] * agg["slum_risk_score"]
)

print(f"    mean={agg['affluence_raw'].mean():.4f}")
print(f"    std={agg['affluence_raw'].std():.4f}")
print(f"    min={agg['affluence_raw'].min():.4f}")
print(f"    max={agg['affluence_raw'].max():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Affluence percentile ranking
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  Step 3 — Percentile ranking (vs non-contaminated pincodes)")
print(f"{SEP}\n")

# Mark contaminated pincodes
agg["is_contaminated"] = (
    (merged["industrial_flag"] == 1) |
    (merged["sparse_sample_flag"] == 1) |
    (merged["uninhabited_flag"] == 1)
).astype(int)

n_contaminated = agg["is_contaminated"].sum()
n_clean = len(agg) - n_contaminated
print(f"  Clean pincodes: {n_clean:,}")
print(f"  Contaminated: {n_contaminated:,} ({100*n_contaminated/len(agg):.1f}%)")
print(f"    - industrial_flag: {(merged['industrial_flag']==1).sum():,}")
print(f"    - sparse_sample_flag: {(merged['sparse_sample_flag']==1).sum():,}")
print(f"    - uninhabited_flag: {(merged['uninhabited_flag']==1).sum():,}")

# Rank against clean pincodes only
clean_scores = agg[agg["is_contaminated"] == 0]["affluence_raw"]
agg["affluence_percentile"] = agg["affluence_raw"].apply(
    lambda x: 100 * stats.percentileofscore(clean_scores, x, kind="rank") / 100
).clip(0, 100)

print(f"\n  Affluence percentile [0-100] computed against {n_clean:,} clean pincodes")
print(f"    Top 1% threshold: {agg['affluence_percentile'].quantile(0.99):.2f}")
print(f"    Top 10% threshold: {agg['affluence_percentile'].quantile(0.90):.2f}")
print(f"    Top 25% threshold: {agg['affluence_percentile'].quantile(0.75):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Confidence scoring
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  Step 4 — Confidence weighting")
print(f"{SEP}\n")

agg["confidence"] = 1.0

# Downweight contaminated pincodes
agg.loc[merged["industrial_flag"] == 1, "confidence"] *= 0.3
agg.loc[merged["sparse_sample_flag"] == 1, "confidence"] *= 0.6
agg.loc[merged["uninhabited_flag"] == 1, "confidence"] *= 0.2

print(f"  Confidence scores assigned:")
print(f"    1.0 (full):          {(agg['confidence']==1.0).sum():,} pincodes")
print(f"    0.6 (sparse sample): {(agg['confidence']==0.6).sum():,} pincodes")
print(f"    0.3 (industrial):    {(agg['confidence']==0.3).sum():,} pincodes")
print(f"    0.2 (uninhabited):   {(agg['confidence']==0.2).sum():,} pincodes")
print(f"    (Some pincodes have multiple flags, confidence is product of all)")

# Recompute percentile with confidence weighting
agg["affluence_percentile_weighted"] = agg["affluence_percentile"] * agg["confidence"]

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Save output
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  Step 5 — Prepare output columns")
print(f"{SEP}\n")

# Merge back core NTL and WorldPop features for context
output = agg.copy()
output["log_ntl_p90"] = merged["log_ntl_p90"]
output["log_pop_sum"] = merged["log_pop_sum"]
output["log_pop_density"] = merged["log_pop_density"]
output["pop_concentration"] = merged["pop_concentration"]
output["pop_cv"] = merged["pop_cv"]
output["pop_gini"] = merged["pop_gini"]
output["high_density_ratio"] = merged["high_density_ratio"]
output["pop_sum_state_pct"] = merged["pop_sum_state_pct"]

# Reorder columns
col_order = (
    ["pincode", "area_km2"] +
    # Core NTL features
    ["log_ntl_p90"] +
    # Core WorldPop features
    ["log_pop_sum", "log_pop_density", "pop_concentration", "pop_cv", "pop_gini",
     "high_density_ratio", "pop_density_state_pct", "pop_sum_state_pct"] +
    # Cross-signals
    ["ntl_per_capita", "urban_intensity", "slum_risk_score", "settlement_quality",
     "development_gap", "economic_density", "density_light_ratio"] +
    # Affluence outputs
    ["affluence_raw", "affluence_percentile", "affluence_percentile_weighted",
     "confidence"] +
    # Flags
    ["industrial_flag", "sparse_sample_flag", "uninhabited_flag", "is_contaminated"]
)

output = output[col_order]

# Save
out_path = Path("affluence_ranking.csv")
output.to_csv(out_path, index=False)
size_mb = out_path.stat().st_size / 1024 / 1024

print(f"{SEP}")
print(f"  Saved → {out_path.name}")
print(f"  Size: {len(output):,} rows × {len(output.columns)} cols ({size_mb:.1f} MB)")
print(f"  Mode: {mode.upper()}")
print(f"{SEP}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Sample output
# ─────────────────────────────────────────────────────────────────────────────
print("  Top 20 pincodes by affluence_percentile:\n")
top20 = output.nlargest(20, "affluence_percentile")[
    ["pincode", "affluence_percentile", "log_ntl_p90", "log_pop_density",
     "confidence", "is_contaminated"]
]
for idx, row in top20.iterrows():
    contaminated = "⚠️ CONTAMINATED" if row["is_contaminated"] else ""
    print(
        f"  {row['pincode']:6.0f}  affluence={row['affluence_percentile']:6.2f}  "
        f"ntl={row['log_ntl_p90']:5.2f}  density={row['log_pop_density']:5.2f}  "
        f"conf={row['confidence']:.1f}  {contaminated}"
    )

print(f"\n  Bottom 20 pincodes by affluence_percentile:\n")
bottom20 = output.nsmallest(20, "affluence_percentile")[
    ["pincode", "affluence_percentile", "log_ntl_p90", "log_pop_density",
     "confidence", "is_contaminated"]
]
for idx, row in bottom20.iterrows():
    contaminated = "⚠️ CONTAMINATED" if row["is_contaminated"] else ""
    print(
        f"  {row['pincode']:6.0f}  affluence={row['affluence_percentile']:6.2f}  "
        f"ntl={row['log_ntl_p90']:5.2f}  density={row['log_pop_density']:5.2f}  "
        f"conf={row['confidence']:.1f}  {contaminated}"
    )

print("\n")
