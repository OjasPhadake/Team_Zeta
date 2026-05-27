"""
worldpop_feature_engg.py
========================
Builds 16 model-ready WorldPop features from:
  worldpop_pincode.csv        — pincode-level rasterstats aggregation
  worldpop_pixels_sample.csv  — 500K stratified 1km pixel sample

Output: worldpop_features.csv — one row per pincode, 16 normalised feature
columns + 2 binary flags + area_km2.

Column note (worldpop_pincode.csv)
───────────────────────────────────
Due to a column alignment issue in eda_worldpop.py (now fixed via explicit
rename), the column 'pop_mean' in the current CSV stores the bounding-box
pixel count (from add_stats), NOT the rasterstats mean pixel value.
This script handles the current file correctly:
  • pop_pixel_count  → center-within pixel count → area_km2 = × 0.01
  • pop_mean         → bounding-box pixel count  → denominator for ratios
  • pop_true_mean    → recomputed as pop_sum / pop_pixel_count

If you re-run eda_worldpop.py after the fix (delete worldpop_pincode.csv
first), pop_mean will become the correct float mean and pop_pixel_count will
remain the valid pixel count — the derived formulas below still work.

DO NOT include NTL cross-signal columns (ntl_per_capita, light_density_ratio,
pop_wtd_ntl) in worldpop_features.csv — they belong in the merge step.

Runtime: < 2 min (all in-memory on 500K-row pixel CSV).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats
from sklearn.preprocessing import MinMaxScaler

SEP = "=" * 65

# ─────────────────────────────────────────────────────────────────────────────
# 0. Load inputs
# ─────────────────────────────────────────────────────────────────────────────
print(f"{SEP}\n  Step 0 — Load inputs\n{SEP}")

pin = pd.read_csv("worldpop_pincode.csv")
px  = pd.read_csv("worldpop_pixels_sample.csv")

# Drop NTL cross-signal columns — reserved for the merge step
NTL_COLS = ["ntl_sol_raw", "ntl_per_capita", "light_density_ratio", "pop_wtd_ntl"]
pin = pin.drop(columns=[c for c in NTL_COLS if c in pin.columns])

print(f"  Pincode CSV  : {len(pin):,} rows")
print(f"  Pixel sample : {len(px):,} rows  |  {px['pincode'].nunique():,} pincodes")

# Detect which file version we have: pre-fix (pop_mean ≈ large integer count)
# vs post-fix (pop_mean ≈ small float mean value).
_median_pop_mean = pin["pop_mean"].median()
_pre_fix = _median_pop_mean > 50
if _pre_fix:
    print(f"  pop_mean median = {_median_pop_mean:.1f} → pre-fix CSV detected")
    print(f"  Using pop_mean as bounding-box pixel count denominator.")
    BBOX_COUNT_COL = "pop_mean"
else:
    print(f"  pop_mean median = {_median_pop_mean:.4f} → post-fix CSV detected")
    print(f"  Computing bounding-box count as pop_px_nonzero.")
    BBOX_COUNT_COL = "pop_px_nonzero"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Base features from worldpop_pincode.csv
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Step 1 — Base features (pincode CSV)\n{SEP}")

agg = pin[["pincode", "pop_sum", "pop_pixel_count", "pop_std",
           "pop_p50", "pop_p90", "pop_p95",
           "pop_px_above_100",
           BBOX_COUNT_COL]].copy()

agg = agg.reset_index(drop=True)
agg = agg.rename(columns={BBOX_COUNT_COL: "pop_bbox_count"})
agg["pop_bbox_count"] = agg["pop_bbox_count"].fillna(0)

# 1a. Area and density
agg["area_km2"]     = agg["pop_pixel_count"] * 0.01
agg["pop_density"]  = (agg["pop_sum"]
                       / agg["area_km2"].replace(0, np.nan)).fillna(0)
agg["log_pop_sum"]      = np.log1p(agg["pop_sum"])
agg["log_pop_density"]  = np.log1p(agg["pop_density"])
print(f"  1a. area_km2       : mean={agg['area_km2'].mean():.3f}  max={agg['area_km2'].max():.2f} km²")
print(f"  1a. pop_density    : mean={agg['pop_density'].mean():.1f}  max={agg['pop_density'].max():.0f} p/km²")

# 1b. True mean per pixel (people / 100m pixel)
agg["pop_true_mean"] = (agg["pop_sum"]
                        / agg["pop_pixel_count"].replace(0, np.nan)).fillna(0)

# 1c. Coefficient of variation (pop_std / pop_true_mean)
agg["pop_cv"] = (agg["pop_std"]
                 / agg["pop_true_mean"].replace(0, np.nan)).fillna(0).clip(upper=10)
print(f"  1c. pop_cv         : mean={agg['pop_cv'].mean():.2f}  max={agg['pop_cv'].max():.2f}")

# 1d. Percentile-based distribution shape
agg["log_pop_p90"]        = np.log1p(agg["pop_p90"])
agg["log_pop_p95"]        = np.log1p(agg["pop_p95"])
agg["pop_concentration"]  = (agg["pop_p90"]
                              / agg["pop_p50"].replace(0, np.nan)).fillna(1).clip(upper=50)
agg["pop_p95_p90_ratio"]  = (agg["pop_p95"]
                              / agg["pop_p90"].replace(0, np.nan)).fillna(1).clip(upper=20)
print(f"  1d. pop_concentration : mean={agg['pop_concentration'].mean():.2f}")
print(f"  1d. pop_p95_p90_ratio : mean={agg['pop_p95_p90_ratio'].mean():.2f}")

# 1e. High-density ratio (internally consistent: both numerator and denominator
#     from bounding-box pixel set — ratio is always in [0, 1])
agg["high_density_ratio"] = (agg["pop_px_above_100"]
                              / agg["pop_bbox_count"].replace(0, np.nan)).fillna(0).clip(upper=1)
print(f"  1e. high_density_ratio : mean={agg['high_density_ratio'].mean():.4f}  "
      f"max={agg['high_density_ratio'].max():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Pixel-level features from worldpop_pixels_sample.csv
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Step 2 — Pixel-level features (pixel sample)\n{SEP}")

# 2a. Sample pixel count per pincode (used for sparse_sample_flag)
sample_counts = px.groupby("pincode").size().rename("sample_pixel_count")
agg = agg.join(sample_counts, on="pincode")
agg["sample_pixel_count"] = agg["sample_pixel_count"].fillna(0).astype(int)
print(f"  2a. sample_pixel_count : min={agg['sample_pixel_count'].min()}  "
      f"median={agg['sample_pixel_count'].median():.0f}  "
      f"max={agg['sample_pixel_count'].max():,}")

# Helper functions for groupby
def _gini(arr):
    arr = np.sort(np.abs(arr))
    n = len(arr)
    if n < 2 or arr.sum() == 0:
        return np.nan
    idx = np.arange(1, n + 1)
    return (2 * np.dot(idx, arr) - (n + 1) * arr.sum()) / (n * arr.sum())

def _top20_share(arr):
    if arr.sum() == 0 or len(arr) == 0:
        return np.nan
    threshold = np.percentile(arr, 80)
    return float(arr[arr >= threshold].sum() / arr.sum())

def _pop_centroid_shift(grp):
    w = grp["pop_per_100m_pixel"].values
    if w.sum() == 0 or len(w) < 2:
        return 0.0
    lat_wtd = np.average(grp["latitude"].values,  weights=w)
    lon_wtd = np.average(grp["longitude"].values, weights=w)
    return float(np.sqrt((lat_wtd - grp["latitude"].mean())**2
                         + (lon_wtd - grp["longitude"].mean())**2))

def _spatial_extent(grp):
    lat_iqr = grp["latitude"].quantile(0.9)  - grp["latitude"].quantile(0.1)
    lon_iqr = grp["longitude"].quantile(0.9) - grp["longitude"].quantile(0.1)
    return float(lat_iqr * lon_iqr)

vals = px["pop_per_100m_pixel"].values
grp  = px.groupby("pincode")

# 2b. Gini coefficient of pixel density distribution
print("  2b. Computing pop_gini ...")
pop_gini = grp["pop_per_100m_pixel"].apply(_gini).rename("pop_gini")
agg = agg.join(pop_gini, on="pincode")

# 2c. Skewness (heavy right tail = slum pockets / extreme spikes)
print("  2c. Computing pop_skew ...")
pop_skew = grp["pop_per_100m_pixel"].skew().rename("pop_skew")
agg = agg.join(pop_skew, on="pincode")
agg["pop_skew"] = agg["pop_skew"].clip(lower=-5, upper=15)

# 2d. Top-20% share (Pareto concentration)
print("  2d. Computing pop_top20_share ...")
pop_top20 = grp["pop_per_100m_pixel"].apply(_top20_share).rename("pop_top20_share")
agg = agg.join(pop_top20, on="pincode")

# 2e. p10 density floor (high = formal planned settlement with no empty corners)
print("  2e. Computing pop_p10_density ...")
pop_p10 = grp["pop_per_100m_pixel"].quantile(0.1).rename("pop_p10_density")
agg = agg.join(pop_p10, on="pincode")
agg["log_pop_p10"] = np.log1p(agg["pop_p10_density"].fillna(0))

# 2f. Population-weighted centroid shift
print("  2f. Computing pop_centroid_shift ...")
pop_shift = grp.apply(_pop_centroid_shift, include_groups=False).rename("pop_centroid_shift")
agg = agg.join(pop_shift, on="pincode")
agg["pop_centroid_shift"] = agg["pop_centroid_shift"].fillna(0)

# 2g. Spatial extent (IQR lat × IQR lon)
print("  2g. Computing pop_spatial_extent ...")
pop_extent = grp.apply(_spatial_extent, include_groups=False).rename("pop_spatial_extent")
agg = agg.join(pop_extent, on="pincode")
agg["pop_spatial_extent"] = agg["pop_spatial_extent"].fillna(0)

print(f"\n  Pixel features summary:")
print(f"    pop_gini          : mean={agg['pop_gini'].mean():.3f}  ({agg['pop_gini'].isna().sum()} NaN)")
print(f"    pop_skew          : mean={agg['pop_skew'].mean():.2f}  ({agg['pop_skew'].isna().sum()} NaN)")
print(f"    pop_top20_share   : mean={agg['pop_top20_share'].mean():.3f}")
print(f"    pop_p10_density   : mean={agg['pop_p10_density'].mean():.4f}")
print(f"    pop_centroid_shift: mean={agg['pop_centroid_shift'].mean():.5f}°")
print(f"    pop_spatial_extent: mean={agg['pop_spatial_extent'].mean():.5f}°²")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Peer-relative features (3-digit pincode prefix = rough state/region proxy)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Step 3 — Peer-relative features\n{SEP}")

agg["_prefix"] = agg["pincode"].astype(str).str[:3]

agg["pop_density_state_pct"] = (agg.groupby("_prefix")["pop_density"]
                                   .rank(pct=True))
agg["pop_sum_state_pct"]     = (agg.groupby("_prefix")["pop_sum"]
                                   .rank(pct=True))
agg = agg.drop(columns=["_prefix"])

print(f"  pop_density_state_pct : mean={agg['pop_density_state_pct'].mean():.3f}  "
      f"(3-digit prefix groups)")
print(f"  pop_sum_state_pct     : mean={agg['pop_sum_state_pct'].mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Contamination / quality flags
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Step 4 — Contamination flags\n{SEP}")

# uninhabited_flag: essentially zero-population pincode
# (uninhabited land, protected areas, coastal/forest pincodes)
agg["uninhabited_flag"] = (agg["pop_sum"] < 1.0).astype(int)
n_uninhabited = agg["uninhabited_flag"].sum()
print(f"  uninhabited_flag  : {n_uninhabited:,} pincodes ({100*n_uninhabited/len(agg):.1f}%)")

# sparse_sample_flag: fewer than 10 pixels in the sample → pixel-derived features
# (pop_gini, pop_skew, pop_top20_share, pop_centroid_shift) are unreliable
agg["sparse_sample_flag"] = (agg["sample_pixel_count"] < 10).astype(int)
n_sparse = agg["sparse_sample_flag"].sum()
print(f"  sparse_sample_flag: {n_sparse:,} pincodes ({100*n_sparse/len(agg):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Clip outliers before scaling
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Step 5 — Outlier clipping\n{SEP}")

# pop_density: extreme outliers from tiny pincodes (large pop, tiny area)
cap_density = float(agg["pop_density"].quantile(0.995))
agg["pop_density"]     = agg["pop_density"].clip(upper=cap_density)
agg["log_pop_density"] = np.log1p(agg["pop_density"])
print(f"  pop_density cap (99.5th): {cap_density:.1f} persons/km²")

# pop_centroid_shift: cap at 99.5th to remove tiny pincodes with wild centroids
cap_shift = float(agg["pop_centroid_shift"].quantile(0.995))
agg["pop_centroid_shift"] = agg["pop_centroid_shift"].clip(upper=cap_shift)
print(f"  pop_centroid_shift cap  : {cap_shift:.5f}°")

# pop_spatial_extent: cap at 99.5th
cap_extent = float(agg["pop_spatial_extent"].quantile(0.995))
agg["pop_spatial_extent"] = agg["pop_spatial_extent"].clip(upper=cap_extent)
print(f"  pop_spatial_extent cap  : {cap_extent:.5f}°²")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MinMax normalisation
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Step 6 — MinMax normalisation\n{SEP}")

# Scale features — primary sort column: log_pop_density
SCALE_COLS = [
    # Scale (from pincode CSV)
    "log_pop_sum",
    "log_pop_density",
    "log_pop_p90",
    "log_pop_p95",
    # Distribution shape (from pincode CSV)
    "pop_concentration",
    "pop_p95_p90_ratio",
    "high_density_ratio",
    "pop_cv",
    # Distribution shape (from pixel sample)
    "pop_gini",
    "pop_skew",
    "pop_top20_share",
    "log_pop_p10",
    # Spatial topology (from pixel sample)
    "pop_centroid_shift",
    "pop_spatial_extent",
    # Context-relative
    "pop_density_state_pct",
    "pop_sum_state_pct",
]

# Fill NaN (sparse pincodes) with median before scaling
fill_medians = agg[SCALE_COLS].median()
agg[SCALE_COLS] = agg[SCALE_COLS].fillna(fill_medians)

agg[SCALE_COLS] = MinMaxScaler().fit_transform(agg[SCALE_COLS])
print(f"  Normalised {len(SCALE_COLS)} feature columns to [0, 1].")
print(f"  uninhabited_flag and sparse_sample_flag left as binary.")

# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Validation — top 20 by log_pop_density\n{SEP}")

top20_df = agg.nlargest(20, "log_pop_density")[
    ["pincode", "log_pop_density", "log_pop_p90",
     "high_density_ratio", "uninhabited_flag"]
]
print(top20_df.to_string(index=False))

# Reference pincodes
print(f"\n  Reference pincodes (raw density and log_pop_density rank):")
refs = {
    110001: "New Delhi GPO",
    400051: "BKC Mumbai",
    560034: "Koramangala",
    122002: "Gurugram DLF",
    500034: "Banjara Hills",
    393002: "Ankleshwar (industrial)",
    110073: "Dwarka (dense residential)",
}
agg_idx = agg.set_index("pincode")
for pc, name in refs.items():
    if pc not in agg_idx.index:
        continue
    row  = agg_idx.loc[pc]
    rank = int((agg_idx["log_pop_density"] >= row["log_pop_density"]).sum())
    print(f"  {pc}  {name:<30}  rank={rank}/{len(agg):<6}  "
          f"log_pop_density={row['log_pop_density']:.4f}  "
          f"high_dens_ratio={row['high_density_ratio']:.4f}  "
          f"uninhabited={int(row['uninhabited_flag'])}")

# Uninhabited flag distribution
print(f"\n  Uninhabited pincodes rank distribution (log_pop_density):")
uninh = agg[agg["uninhabited_flag"] == 1]
in_top25 = (uninh["log_pop_density"] > agg["log_pop_density"].quantile(0.75)).sum()
in_bot25 = (uninh["log_pop_density"] < agg["log_pop_density"].quantile(0.25)).sum()
print(f"    In top 25% by density : {in_top25}  (should be ~0 for forest/desert pincodes)")
print(f"    In bottom 25%         : {in_bot25}  (should be most of them)")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out_cols = ["pincode", "area_km2"] + SCALE_COLS + ["uninhabited_flag", "sparse_sample_flag"]
out_df   = agg[out_cols]
out_df.to_csv("worldpop_features.csv", index=False)

size_mb = Path("worldpop_features.csv").stat().st_size / 1024 / 1024
print(f"\n{SEP}")
print(f"  Saved → worldpop_features.csv  ({len(out_df):,} rows, {size_mb:.1f} MB)")
print(f"  Primary sort column : log_pop_density  (urban intensity, NOT affluence)")
print(f"  Feature columns ({len(SCALE_COLS)}):")
for c in SCALE_COLS:
    print(f"    {c}")
print(f"  Non-scaled columns  : area_km2, uninhabited_flag, sparse_sample_flag")
print(SEP)
