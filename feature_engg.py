"""
feature_engg.py
===============
Builds 14 model-ready NTL features from:
  ntl_pixels_500m.csv          — full pixel-level data (lat, lon, ntl_radiance, year)
  ntl_pincode_aggregated.csv   — pincode-level GEE stats

Output: ntl_features.csv — one row per pincode, 14 normalised feature columns.

Runtime estimate (45M-row pixel CSV):
  Step 0  rasterize + assign pincodes : ~3–8 min  (cached after first run)
  Step 2  groupby quantiles            : ~5–10 min
  Steps 3–6                            : < 1 min
  Total first run  : ~15–20 min
  Total subsequent : ~10–12 min  (cache hit skips rasterization)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

SEP = "=" * 60

# ─────────────────────────────────────────────────────────────────────────────
# 0. Load files + assign pixels to pincodes
# ─────────────────────────────────────────────────────────────────────────────
print(f"{SEP}\n  Step 0 — Load + pincode assignment\n{SEP}")

# Pincode-level CSV (for ntl_max used later in spike_ratio)
pincode_df = pd.read_csv("ntl_pincode_aggregated.csv")
junk = [c for c in ["tessellate", "extrude", "visibility"] if c in pincode_df.columns]
pincode_df = pincode_df.drop(columns=junk)
print(f"  Pincode CSV  : {len(pincode_df):,} rows  |  columns: {pincode_df.columns.tolist()}")

# Pixel CSV — load with float32 to halve memory (1.4 GB → ~700 MB)
print("  Loading pixel CSV (large file, may take a minute) ...")
px = pd.read_csv(
    "ntl_pixels_500m.csv",
    dtype={"latitude": "float32", "longitude": "float32",
           "ntl_radiance": "float32", "year": "int16"},
)
print(f"  Pixel CSV    : {len(px):,} rows")

if len(px) < 50_000:
    print("  WARNING: pixel CSV has < 50,000 rows — this looks like the 5,000-sample file,")
    print("  not the full export.  Per-pincode percentiles will be very noisy.")
    print("  Run pixel_extract.py first to get the complete dataset.\n")

# ── Assign pincodes via rasterization (fast) ──────────────────────────────────
CACHE = Path("pixel_pincode_cache.parquet")

if "pincode" not in px.columns:
    if CACHE.exists():
        print(f"  Loading cached pincode assignments from {CACHE} ...")
        px["pincode"] = pd.read_parquet(CACHE)["pincode"].values
    else:
        print("  No pincode column found — rasterizing boundaries ...")
        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds

        gdf = gpd.read_file("india_pincodes.geojson")
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        # Map sequential GDF row index → actual pincode
        pins = pd.read_csv("extracted_pincodes.csv")
        assert len(gdf) == len(pins), "GDF / extracted_pincodes.csv length mismatch"
        gdf["pincode"] = pins["Pincode"].values

        # Detect pixel resolution from the CSV itself so the raster grid is aligned
        sample_lons = np.sort(px["longitude"].sample(min(200_000, len(px)),
                                                      random_state=0).unique())
        sample_lats = np.sort(px["latitude"].sample(min(200_000, len(px)),
                                                     random_state=0).unique())
        lon_res = float(np.median(np.diff(sample_lons)))
        lat_res = float(np.median(np.diff(sample_lats)))
        print(f"  Detected pixel resolution: lon={lon_res:.6f}°  lat={lat_res:.6f}°")

        # Expand bbox by half a pixel so all pixel centres fall inside
        lon_min = float(px["longitude"].min()) - lon_res / 2
        lat_min = float(px["latitude"].min())  - lat_res / 2
        lon_max = float(px["longitude"].max()) + lon_res / 2
        lat_max = float(px["latitude"].max())  + lat_res / 2

        width  = int(round((lon_max - lon_min) / lon_res))
        height = int(round((lat_max - lat_min) / lat_res))
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
        print(f"  Raster grid  : {height} rows × {width} cols  (~{height*width/1e6:.1f}M cells)")

        # Rasterize: polygon index → 1-based int32 (0 = outside all polygons)
        shapes = [
            (geom, idx + 1)
            for idx, geom in enumerate(gdf.geometry)
            if geom is not None and not geom.is_empty
        ]
        print(f"  Rasterizing {len(shapes):,} polygons ...")
        raster = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="int32",
        )

        # Lookup array: 1-based sequential index → actual pincode int
        # Index 0 = outside India → stays 0
        pincode_lookup = np.zeros(len(gdf) + 1, dtype=np.int64)
        for idx, pc in enumerate(gdf["pincode"].values):
            pincode_lookup[idx + 1] = int(pc)

        # Map each pixel to its raster cell then to its pincode
        col_idx = np.clip(
            ((px["longitude"].values - lon_min) / lon_res).astype(int), 0, width - 1
        )
        row_idx = np.clip(
            ((lat_max - px["latitude"].values) / lat_res).astype(int), 0, height - 1
        )
        seq = raster[row_idx, col_idx]
        px["pincode"] = pincode_lookup[seq]

        print(f"  Saving cache → {CACHE}")
        px[["pincode"]].astype("int64").to_parquet(CACHE)

# Drop pixels that didn't fall inside any pincode polygon
n_before = len(px)
px = px[px["pincode"] > 0].copy()
print(f"  Pixels inside pincodes : {len(px):,}  (dropped {n_before - len(px):,} outside)")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Clean the pixel CSV
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Step 1 — Pixel cleaning\n{SEP}")

# 1a. Drop atmospheric scatter / sensor noise
n_before = len(px)
px = px[px["ntl_radiance"] >= 0.4]
print(f"  1a. Dropped {n_before - len(px):,} pixels with radiance < 0.4  →  {len(px):,} remain")

# 1b. Percentile-clip industrial spikes nationally
if len(px) >= 50_000:
    cap = float(px["ntl_radiance"].quantile(0.995))
    print(f"  1b. 99.5th percentile cap : {cap:.4f} DN")
else:
    cap = 200.0
    print(f"  1b. Sample too small — using fixed cap : {cap:.1f} DN")

px["ntl_radiance"] = px["ntl_radiance"].clip(upper=cap)
print(f"      Max radiance after clip : {px['ntl_radiance'].max():.4f} DN")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Re-aggregate intensity features from cleaned pixels
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Step 2 — Re-aggregate from cleaned pixels\n{SEP}")

# 2a. Median
print("  2a. Computing median ...")
agg = px.groupby("pincode")["ntl_radiance"].median().rename("ntl_median").reset_index()

# 2b. Percentiles — one groupby call, three quantiles
print("  2b. Computing p75 / p90 / p95 (slowest step) ...")
pcts = (
    px.groupby("pincode")["ntl_radiance"]
    .quantile([0.75, 0.90, 0.95])
    .unstack(level=-1)
)
pcts.columns = ["ntl_p75", "ntl_p90", "ntl_p95"]
agg = agg.join(pcts, on="pincode")

# 2c. Sum of lights
print("  2c. Computing SOL (sum) ...")
agg = agg.join(
    px.groupby("pincode")["ntl_radiance"].sum().rename("ntl_sol"),
    on="pincode",
)

# 2d. Standard deviation
print("  2d. Computing std ...")
agg = agg.join(
    px.groupby("pincode")["ntl_radiance"].std().fillna(0).rename("ntl_std"),
    on="pincode",
)

print(f"  Aggregated {len(agg):,} pincodes from cleaned pixels.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Spatial distribution features
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Step 3 — Spatial distribution features\n{SEP}")

# 3a. Lit pixel count: pixels with radiance > 1.0 after cleaning
print("  3a. Lit pixel count (radiance > 1.0) ...")
lit_counts = (
    px[px["ntl_radiance"] > 1.0]
    .groupby("pincode")
    .size()
    .rename("lit_pixel_count")
)
agg = agg.join(lit_counts, on="pincode")
agg["lit_pixel_count"] = agg["lit_pixel_count"].fillna(0).astype(int)

# 3b. Total pixel count: all cleaned pixels assigned to each pincode
print("  3b. Total pixel count ...")
total_counts = px.groupby("pincode").size().rename("total_pixel_count")
agg = agg.join(total_counts, on="pincode")
agg["total_pixel_count"] = agg["total_pixel_count"].fillna(0).astype(int)

# 3c. Lit pixel ratio — key industrial contamination separator
agg["lit_pixel_ratio"] = (
    agg["lit_pixel_count"] / agg["total_pixel_count"].replace(0, np.nan)
).fillna(0)
print(f"  3c. lit_pixel_ratio  |  mean={agg['lit_pixel_ratio'].mean():.3f}  max={agg['lit_pixel_ratio'].max():.3f}")

# 3d. Area-normalised SOL
agg["ntl_sol_per_pixel"] = (
    agg["ntl_sol"] / agg["total_pixel_count"].replace(0, np.nan)
).fillna(0)
print(f"  3d. ntl_sol_per_pixel  |  mean={agg['ntl_sol_per_pixel'].mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Industrial contamination flags
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Step 4 — Industrial contamination flags\n{SEP}")

# ntl_max from the original GEE aggregation (pre-cleaning, intentional —
# spike_ratio measures raw peak vs the cleaned distribution)
ntl_max_map = pincode_df.set_index("pincode")["ntl_max"]
agg["ntl_max"] = agg["pincode"].map(ntl_max_map).fillna(0)

# 4a. Spike ratio
agg["spike_ratio"] = (
    agg["ntl_max"] / agg["ntl_p90"].replace(0, 0.001)
).clip(upper=20)
print(f"  4a. spike_ratio          |  mean={agg['spike_ratio'].mean():.2f}  max={agg['spike_ratio'].max():.2f}")

# 4b. Skewness proxy
agg["p95_to_median_ratio"] = (
    agg["ntl_p95"] / agg["ntl_median"].replace(0, 0.001)
).clip(upper=50)
print(f"  4b. p95_to_median_ratio  |  mean={agg['p95_to_median_ratio'].mean():.2f}")

# 4c. Coefficient of variation
agg["cv_ntl"] = (
    agg["ntl_std"] / agg["ntl_median"].replace(0, 0.001)
).clip(upper=10)
print(f"  4c. cv_ntl               |  mean={agg['cv_ntl'].mean():.3f}")

# 4d. Binary industrial flag
agg["industrial_flag"] = (
    (agg["spike_ratio"] > 5) & (agg["lit_pixel_ratio"] < 0.15)
).astype(int)
n_flagged = agg["industrial_flag"].sum()
print(f"  4d. industrial_flag      |  {n_flagged:,} pincodes flagged ({100*n_flagged/len(agg):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Log-transform skewed intensity features
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Step 5 — Log-transform\n{SEP}")

intensity_cols = ["ntl_sol", "ntl_median", "ntl_p75", "ntl_p90", "ntl_p95", "ntl_std"]
for col in intensity_cols:
    agg[f"log_{col}"] = np.log1p(agg[col])
    print(f"  log_{col:<20}  mean={agg[f'log_{col}'].mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Min-max normalise all feature columns (industrial_flag stays binary)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Step 6 — Min-max normalisation\n{SEP}")

log_cols   = [f"log_{c}" for c in intensity_cols]
ratio_cols = ["lit_pixel_ratio", "ntl_sol_per_pixel",
              "spike_ratio", "p95_to_median_ratio", "cv_ntl"]
count_cols = ["lit_pixel_count", "total_pixel_count"]
feat_cols  = log_cols + ratio_cols + count_cols   # 14 columns, industrial_flag excluded

agg[feat_cols] = MinMaxScaler().fit_transform(agg[feat_cols].fillna(0))
print(f"  Normalised {len(feat_cols)} columns to [0, 1].")
print(f"  industrial_flag left as binary (not normalised).")

# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Validation — top 20 by log_ntl_p90\n{SEP}")
top20_df = agg.nlargest(20, "log_ntl_p90")[
    ["pincode", "log_ntl_p90", "lit_pixel_ratio", "spike_ratio", "industrial_flag"]
]
print(top20_df.to_string(index=False))

# Sanity-check reference pincodes
expected_affluent   = [110016, 110021, 400051, 560034, 122002, 500034]
expected_industrial = [393002, 361008, 768201]

top20_set = set(top20_df["pincode"].tolist())
hits = [p for p in expected_affluent if p in top20_set]
print(f"\n  Affluent reference pincodes in top 20 : {hits} / {expected_affluent}")

agg_indexed = agg.set_index("pincode")
for p in expected_industrial:
    if p in agg_indexed.index:
        rank = int((agg_indexed["log_ntl_p90"] >= agg_indexed.loc[p, "log_ntl_p90"]).sum())
        flag = int(agg_indexed.loc[p, "industrial_flag"])
        print(f"  Industrial {p}  rank={rank}/{len(agg)}  industrial_flag={flag}")

if len(hits) == 0:
    print("\n  NOTE: No reference affluent pincodes in top 20 — consider")
    print("  tightening the Step 1b clip to 99.0th percentile and re-running.")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out_cols = ["pincode"] + feat_cols + ["industrial_flag"]
agg[out_cols].to_csv("ntl_features.csv", index=False)
size_mb = Path("ntl_features.csv").stat().st_size / 1024 / 1024

print(f"\n{SEP}")
print(f"  Saved → ntl_features.csv  ({len(agg):,} rows, {size_mb:.1f} MB)")
print(f"  Features ({len(feat_cols) + 1} total):")
for c in out_cols[1:]:
    print(f"    {c}")
print(SEP)
