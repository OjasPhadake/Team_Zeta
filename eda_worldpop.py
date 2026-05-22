"""
eda_worldpop.py
===============
Exploratory analysis of WorldPop India 2020 (ind_ppp_2020.tif).

Place this script in the same folder as ind_ppp_2020.tif and run:
    python eda_worldpop.py

All outputs are written to ./eda_worldpop_outputs/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import rasterio
from rasterio.windows import Window
from scipy import stats

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ── config ────────────────────────────────────────────────────────────────────
TIF   = "ind_ppp_2020.tif"
OUT   = "eda_worldpop_outputs"
os.makedirs(OUT, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight"})

C = {
    "blue":   "#185FA5",
    "teal":   "#0F6E56",
    "amber":  "#854F0B",
    "coral":  "#993C1D",
    "purple": "#534AB7",
}

def save(fig, name):
    path = f"{OUT}/{name}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved → {name}")

def hline(title=""):
    print(f"\n{'─'*62}")
    if title:
        print(f"  {title}")
        print(f"{'─'*62}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Open raster, read metadata, sample pixel values
# ═════════════════════════════════════════════════════════════════════════════
hline("Step 1 — raster metadata")

with rasterio.open(TIF) as src:
    meta      = src.meta
    bounds    = src.bounds
    res_deg   = src.res                          # degrees per pixel
    res_m     = (res_deg[0] * 111320,            # approx metres
                 res_deg[1] * 111320)
    nodata    = src.nodata
    crs       = src.crs
    n_rows    = src.height
    n_cols    = src.width
    n_bands   = src.count

    print(f"  File        : {TIF}")
    print(f"  CRS         : {crs}")
    print(f"  Shape       : {n_rows} rows × {n_cols} cols × {n_bands} band(s)")
    print(f"  Resolution  : {res_m[0]:.0f} m × {res_m[1]:.0f} m")
    print(f"  Bounds      : lon {bounds.left:.3f}–{bounds.right:.3f}")
    print(f"              : lat {bounds.bottom:.3f}–{bounds.top:.3f}")
    print(f"  NoData      : {nodata}")
    print(f"  Dtype       : {meta['dtype']}")

    # ── Efficient population sample: every 100th row ────────────────────────
    # Avoids loading the full ~4 GB raster into RAM.
    print("\n  Sampling raster (every 100th row)…")
    sampled_rows = []
    for i in range(0, n_rows, 100):
        window = Window(0, i, n_cols, 1)
        row = src.read(1, window=window).astype("float32").ravel()
        sampled_rows.append(row)
    raw_sample = np.concatenate(sampled_rows)

    # ── Full-raster population total (row by row, memory-safe) ──────────────
    print("  Computing total population (row-by-row)…")
    total_pop   = 0.0
    valid_px    = 0
    nonzero_px  = 0
    row_sums    = []  # latitude → total pop (for latitude profile)
    lat_centers = []

    for i in range(n_rows):
        window = Window(0, i, n_cols, 1)
        row = src.read(1, window=window).astype("float32").ravel()
        row[row == nodata] = np.nan
        row[row < 0]       = np.nan
        row_sum = np.nansum(row)
        total_pop   += row_sum
        valid_px    += np.sum(~np.isnan(row))
        nonzero_px  += np.sum(row > 0)
        row_sums.append(row_sum)
        # latitude of this pixel row (top-left corner convention)
        lat_centers.append(bounds.top - (i + 0.5) * res_deg[1])

# ── Clean sample ─────────────────────────────────────────────────────────────
raw_sample[raw_sample == nodata] = np.nan
raw_sample[raw_sample < 0]       = np.nan
sample = raw_sample[~np.isnan(raw_sample)]
pop_sample = sample[sample > 0]   # non-zero pixels only

print(f"\n  Total population (full raster) : {total_pop:>15,.0f}")
print(f"  Valid pixels                   : {valid_px:>15,}")
print(f"  Non-zero pixels                : {nonzero_px:>15,}")
print(f"  Zero / nodata pixels           : {valid_px - nonzero_px:>15,}")

# ── Write summary text ────────────────────────────────────────────────────────
pct_vals = [1, 5, 25, 50, 75, 90, 95, 99, 99.9]
pcts     = {p: float(np.percentile(pop_sample, p)) for p in pct_vals}

summary_lines = [
    "WorldPop India 2020 — EDA Summary",
    "=" * 50,
    f"File              : {TIF}",
    f"CRS               : {crs}",
    f"Shape             : {n_rows} rows × {n_cols} cols",
    f"Resolution        : {res_m[0]:.0f} m × {res_m[1]:.0f} m",
    f"NoData            : {nodata}",
    "",
    "Population statistics (non-zero pixels, sampled):",
    f"  Total pop (full): {total_pop:,.0f}",
    f"  Non-zero pixels : {nonzero_px:,}  ({100*nonzero_px/valid_px:.1f}% of valid pixels)",
    f"  Min             : {pop_sample.min():.6f}",
    f"  Mean            : {pop_sample.mean():.4f}",
    f"  Std             : {pop_sample.std():.4f}",
    f"  Max             : {pop_sample.max():.2f}",
    "",
    "Percentiles (non-zero pixels):",
]
for p, v in pcts.items():
    summary_lines.append(f"  p{str(p):<5} : {v:.4f}")

summary_lines += [
    "",
    "Interpretation:",
    "  Values = PPP-adjusted population per 100m pixel.",
    "  Most pixels < 1 person (rural/forest/water).",
    "  Dense urban cores (BKC, Connaught Place) exceed 500.",
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(f"{OUT}/01_raster_summary.txt", "w") as f:
    f.write(summary_text)
print(f"\n  saved → 01_raster_summary.txt")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Pixel-value distribution plots
# ═════════════════════════════════════════════════════════════════════════════
hline("Step 2 — pixel value distributions")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 2a. Log-histogram of non-zero pixels
axes[0, 0].hist(np.log1p(pop_sample), bins=100,
                color=C["blue"], alpha=0.85, edgecolor="none")
axes[0, 0].set_title("Distribution of pixel values (non-zero)")
axes[0, 0].set_xlabel("log₁₊(population per 100m pixel)")
axes[0, 0].set_ylabel("Pixel count")
for p, c in [(50, "gray"), (90, C["amber"]), (99, C["coral"])]:
    v = np.log1p(pcts[p])
    axes[0, 0].axvline(v, color=c, lw=1.2, ls="--",
                       label=f"p{p} = {pcts[p]:.2f}")
axes[0, 0].legend(fontsize=9)

# 2b. CDF
sv  = np.sort(pop_sample)
cdf = np.linspace(0, 1, len(sv))
axes[0, 1].plot(np.log1p(sv), cdf, color=C["blue"], lw=1.5)
for p, c in [(50, "gray"), (90, C["amber"]), (99, C["coral"])]:
    v = np.log1p(pcts[p])
    axes[0, 1].axvline(v, color=c, lw=1, ls="--", label=f"p{p}")
axes[0, 1].set_title("Cumulative distribution (CDF)")
axes[0, 1].set_xlabel("log₁₊(population per pixel)")
axes[0, 1].set_ylabel("Fraction of pixels")
axes[0, 1].legend(fontsize=9)

# 2c. Population bucket breakdown
bins   = [0, 0.5, 1, 5, 10, 25, 50, 100, 250, 500, pop_sample.max() + 1]
labels = ["<0.5","0.5–1","1–5","5–10","10–25","25–50",
          "50–100","100–250","250–500","500+"]
bucket = pd.cut(pop_sample, bins=bins, labels=labels, right=False)
bc = bucket.value_counts().reindex(labels)
bars = axes[1, 0].bar(bc.index, bc.values, color=C["teal"],
                       alpha=0.85, edgecolor="none")
axes[1, 0].set_title("Pixel count by population band")
axes[1, 0].set_xlabel("Population per pixel")
axes[1, 0].set_ylabel("Pixel count")
axes[1, 0].tick_params(axis="x", rotation=35)
for bar, val in zip(bars, bc.values):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:,.0f}", ha="center", va="bottom", fontsize=7)

# 2d. Density of high-value pixels (top 5%)
p95_val = pcts[95]
high    = pop_sample[pop_sample >= p95_val]
axes[1, 1].hist(high, bins=80, color=C["coral"], alpha=0.85, edgecolor="none")
axes[1, 1].set_title(f"Top 5% pixels (pop ≥ {p95_val:.2f})")
axes[1, 1].set_xlabel("Population per pixel")
axes[1, 1].set_ylabel("Pixel count")
axes[1, 1].xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{x:.0f}"))

fig.suptitle("WorldPop India 2020 — pixel value distributions", fontsize=14)
plt.tight_layout()
save(fig, "02_pixel_distributions.png")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Spatial (latitude) population profile
# ═════════════════════════════════════════════════════════════════════════════
hline("Step 3 — latitude population profile")

lat_df = pd.DataFrame({"lat": lat_centers, "pop": row_sums})
# Smooth with 20-row rolling mean to reduce noise
lat_df["pop_smooth"] = lat_df["pop"].rolling(20, center=True).mean()

fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(lat_df["lat"], lat_df["pop_smooth"],
                color=C["blue"], alpha=0.45)
ax.plot(lat_df["lat"], lat_df["pop_smooth"],
        color=C["blue"], lw=1.2)

# Annotate major population latitude bands
for lat, lbl in [(28.6,"Delhi"), (19.1,"Mumbai"),
                 (12.9,"Bengaluru"), (22.6,"Kolkata"), (13.1,"Chennai")]:
    ax.axvline(lat, color=C["coral"], lw=0.8, ls="--", alpha=0.7)
    ax.text(lat, ax.get_ylim()[1]*0.85, lbl,
            rotation=90, va="top", ha="right",
            fontsize=8, color=C["coral"])

ax.set_xlabel("Latitude (°N)")
ax.set_ylabel("Total population per latitude row")
ax.set_title("Population distribution by latitude — India 2020")
ax.invert_xaxis()   # north at left
fig.tight_layout()
save(fig, "03_latitude_population_profile.png")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Zero vs non-zero pixel breakdown
# ═════════════════════════════════════════════════════════════════════════════
hline("Step 4 — zero vs non-zero coverage")

zero_px    = valid_px - nonzero_px
labels_pie = ["Non-zero (populated)", "Zero / uninhabited"]
sizes      = [nonzero_px, zero_px]
colors_pie = [C["blue"], "#D3D1C7"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

wedges, texts, autotexts = axes[0].pie(
    sizes, labels=labels_pie, colors=colors_pie,
    autopct="%1.1f%%", startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=1.5)
)
axes[0].set_title("Pixel coverage breakdown")

# Decile breakdown of populated pixels
decile_pops = []
for d in range(10):
    lo = np.percentile(pop_sample,  d    * 10)
    hi = np.percentile(pop_sample, (d+1) * 10)
    mask = (pop_sample >= lo) & (pop_sample < hi)
    decile_pops.append(pop_sample[mask].sum())

decile_labels = [f"D{i+1}" for i in range(10)]
axes[1].bar(decile_labels, decile_pops,
            color=[C["teal"] if i < 8 else C["coral"] for i in range(10)],
            alpha=0.85, edgecolor="none")
axes[1].set_title("Population share by pixel decile")
axes[1].set_xlabel("Pixel decile (D1 = least dense)")
axes[1].set_ylabel("Total population in decile")
axes[1].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

fig.suptitle("WorldPop India 2020 — coverage and concentration", fontsize=14)
plt.tight_layout()
save(fig, "04_coverage_and_concentration.png")

# Print concentration insight
top_decile_pct = 100 * decile_pops[-1] / sum(decile_pops)
print(f"\n  Top decile (densest 10% of non-zero pixels) holds "
      f"{top_decile_pct:.1f}% of sampled population")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Log-normality test and skewness stats
# ═════════════════════════════════════════════════════════════════════════════
hline("Step 5 — distributional statistics")

log_sample = np.log1p(pop_sample)

# Shapiro-Wilk on a 5000-point subsample (full sample too large)
sub = np.random.default_rng(42).choice(log_sample, size=min(5000, len(log_sample)),
                                        replace=False)
shapiro_stat, shapiro_p = stats.shapiro(sub)

skewness    = float(stats.skew(pop_sample))
log_skewness= float(stats.skew(log_sample))
kurtosis    = float(stats.kurtosis(pop_sample))

dist_stats = "\n".join([
    "Distributional Statistics (non-zero pixels)",
    "-" * 45,
    f"  Raw skewness        : {skewness:.2f}  (> 0 = right-skewed)",
    f"  log1p skewness      : {log_skewness:.2f}  (closer to 0 = more normal)",
    f"  Raw kurtosis        : {kurtosis:.2f}  (0 = normal; > 0 = heavy tails)",
    "",
    f"  Shapiro-Wilk (log)  : W={shapiro_stat:.4f}, p={shapiro_p:.4f}",
    f"  → {'log-normal is a reasonable approximation' if shapiro_p > 0.01 else 'distribution departs from log-normal'}",
    "",
    "Implication for modelling:",
    "  Always use log1p(population) as input to XGBoost, not raw values.",
    "  Raw population has extreme right skew that will dominate tree splits.",
])
print("\n" + dist_stats)
with open(f"{OUT}/05_distributional_stats.txt", "w") as f:
    f.write(dist_stats)
print(f"\n  saved → 05_distributional_stats.txt")

# Q-Q plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
stats.probplot(sub, dist="norm", plot=axes[0])
axes[0].set_title("Q-Q plot: log1p(population) vs normal")
axes[0].get_lines()[0].set(color=C["blue"], markersize=2, alpha=0.5)
axes[0].get_lines()[1].set(color=C["coral"], lw=1.5)

axes[1].hist(log_sample, bins=80, color=C["purple"],
             alpha=0.85, edgecolor="none", density=True)
mu, sigma = log_sample.mean(), log_sample.std()
x = np.linspace(log_sample.min(), log_sample.max(), 300)
axes[1].plot(x, stats.norm.pdf(x, mu, sigma),
             color=C["coral"], lw=2, label="fitted normal")
axes[1].set_title("log1p(population) vs fitted normal")
axes[1].set_xlabel("log1p(population)")
axes[1].set_ylabel("Density")
axes[1].legend(fontsize=9)

fig.suptitle("WorldPop India 2020 — log-normality check", fontsize=14)
plt.tight_layout()
save(fig, "05_log_normality_check.png")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Grid-cell population summary table (top 50 dense grid cells)
# ═════════════════════════════════════════════════════════════════════════════
hline("Step 6 — densest grid cells in sample")

# Re-sample but keep coordinates this time (every 500th row, all cols)
print("  Re-sampling with coordinates…")
dense_rows = []
with rasterio.open(TIF) as src:
    for i in range(0, n_rows, 500):
        window = Window(0, i, n_cols, 1)
        row_data = src.read(1, window=window).astype("float32").ravel()
        lat_row  = bounds.top - (i + 0.5) * res_deg[1]
        for j, val in enumerate(row_data):
            if val > 0 and val != nodata:
                lon = bounds.left + (j + 0.5) * res_deg[0]
                dense_rows.append({"latitude": lat_row,
                                   "longitude": lon,
                                   "pop_per_pixel": float(val)})

pixel_df = pd.DataFrame(dense_rows)
top50 = (pixel_df.nlargest(50, "pop_per_pixel")
                 .reset_index(drop=True))
top50.index += 1
top50.to_csv(f"{OUT}/06_top50_dense_pixels.csv")
print(f"\n  Top 10 densest 100m pixels in sample:")
print(top50.head(10).to_string(index=True,
      float_format=lambda x: f"{x:.4f}"))
print(f"\n  saved → 06_top50_dense_pixels.csv")



# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — Pincode-level aggregation (rasterstats)
#
# Uses rasterstats.zonal_stats to aggregate the full 100m WorldPop raster
# to pincode polygons.  This is the authoritative population feature CSV.
#
# Runtime: 20–60 min on first run (19,312 polygons × 100m raster).
# Re-running skips this step if worldpop_pincode.csv already exists.
# ═════════════════════════════════════════════════════════════════════════════
hline("Step 7 — pincode-level population aggregation")

import geopandas as gpd
from rasterstats import zonal_stats

PINCODE_CSV = "worldpop_pincode.csv"
GEO_PATH    = "india_pincodes.geojson"
PINS_PATH   = "extracted_pincodes.csv"

def _load_pincodes():
    gdf = gpd.read_file(GEO_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    pins = pd.read_csv(PINS_PATH)
    assert len(gdf) == len(pins), "GDF / extracted_pincodes.csv length mismatch"
    gdf["pincode"] = pins["Pincode"].values
    return gdf

# Custom zonal stats: pixel counts above thresholds
def _px_above_100(x):
    arr = x.compressed() if hasattr(x, "compressed") else x[~np.isnan(x)]
    return int((arr > 100).sum())

def _px_nonzero(x):
    arr = x.compressed() if hasattr(x, "compressed") else x[~np.isnan(x)]
    return int((arr > 0).sum())

if os.path.exists(PINCODE_CSV):
    print(f"  {PINCODE_CSV} already exists — loading from disk (delete to re-run zonal stats).")
    pop_df = pd.read_csv(PINCODE_CSV)
else:
    gdf = _load_pincodes()
    print(f"  Loaded {len(gdf):,} pincode polygons.")
    print(f"  Running zonal stats — estimated runtime 20–60 min …")

    import time
    t0 = time.time()
    stats_list = zonal_stats(
        gdf,
        TIF,
        stats=["sum", "mean", "std", "count",
               "percentile_50", "percentile_90", "percentile_95"],
        add_stats={"pop_px_above_100": _px_above_100,
                   "pop_px_nonzero":   _px_nonzero},
        nodata=nodata,
        all_touched=True,
        geojson_out=False,
    )
    elapsed = time.time() - t0
    print(f"  Zonal stats complete in {elapsed/60:.1f} min.")

    pop_df = pd.DataFrame(stats_list)
    pop_df.columns = [
        "pop_sum", "pop_mean", "pop_std", "pop_pixel_count",
        "pop_p50", "pop_p90", "pop_p95",
        "pop_px_above_100", "pop_px_nonzero",
    ]
    pop_df.insert(0, "pincode", gdf["pincode"].values)
    pop_df = pop_df.fillna(0)

    # ── Density & scale features ──────────────────────────────────────────────
    # Each 100m pixel = 0.01 km²
    pop_df["area_km2"]        = pop_df["pop_pixel_count"] * 0.01
    pop_df["pop_density"]     = (pop_df["pop_sum"]
                                 / pop_df["area_km2"].replace(0, np.nan)).fillna(0)
    pop_df["log_pop_sum"]     = np.log1p(pop_df["pop_sum"])
    pop_df["log_pop_density"] = np.log1p(pop_df["pop_density"])

    # ── Settlement structure features ─────────────────────────────────────────
    # pop_cv: coefficient of variation — low = uniform settlement, high = patchy
    pop_df["pop_cv"] = (pop_df["pop_std"]
                        / pop_df["pop_mean"].replace(0, np.nan)).fillna(0).clip(upper=10)

    # pop_concentration: p90/p50 — high = a few very dense pockets
    pop_df["pop_concentration"] = (pop_df["pop_p90"]
                                   / pop_df["pop_p50"].replace(0, np.nan)).fillna(1).clip(upper=50)

    # high_density_ratio: fraction of pixels with > 100 persons per 100m pixel
    pop_df["high_density_ratio"] = (pop_df["pop_px_above_100"]
                                    / pop_df["pop_pixel_count"].replace(0, np.nan)).fillna(0)

    # settled_area_ratio: fraction of pincode with at least one resident
    pop_df["settled_area_ratio"] = (pop_df["pop_px_nonzero"]
                                    / pop_df["pop_pixel_count"].replace(0, np.nan)).fillna(0)

    # ── Cross-signal features (NTL × population) ─────────────────────────────
    # Requires ntl_pincode_aggregated.csv for raw (unscaled) ntl_sum column.
    NTL_RAW = "ntl_pincode_aggregated.csv"
    if os.path.exists(NTL_RAW):
        ntl_raw = pd.read_csv(NTL_RAW)[["pincode", "ntl_sum"]].rename(
            columns={"ntl_sum": "ntl_sol_raw"}
        )
        pop_df = pop_df.merge(ntl_raw, on="pincode", how="left")
        pop_df["ntl_sol_raw"] = pop_df["ntl_sol_raw"].fillna(0)

        # ntl_per_capita: highest single feature for industrial separation
        pop_df["ntl_per_capita"] = (
            pop_df["ntl_sol_raw"] / pop_df["pop_sum"].replace(0, np.nan)
        ).fillna(0).clip(upper=pop_df["ntl_sol_raw"].quantile(0.999))

        # light_density_ratio: SOL normalised by persons/km²
        pop_df["light_density_ratio"] = (
            pop_df["ntl_sol_raw"] / pop_df["pop_density"].replace(0, np.nan)
        ).fillna(0).clip(upper=pop_df["ntl_sol_raw"].quantile(0.999))

        # pop_wtd_ntl: population-weighted light
        total_pop = pop_df["pop_sum"].sum()
        pop_df["pop_wtd_ntl"] = (
            pop_df["ntl_sol_raw"] * pop_df["pop_sum"]
        ) / max(total_pop, 1)

        print(f"  NTL cross-signal features added (from {NTL_RAW}).")
    else:
        print(f"  {NTL_RAW} not found — NTL cross-signal features skipped.")

    pop_df.to_csv(PINCODE_CSV, index=False)
    size_kb = os.path.getsize(PINCODE_CSV) // 1024
    print(f"  saved → {PINCODE_CSV}  ({len(pop_df):,} rows, {size_kb:,} KB)")

# ── Quick pincode-level summary ───────────────────────────────────────────────
print(f"\n  Pincode CSV shape  : {pop_df.shape}")
print(f"  Columns            : {pop_df.columns.tolist()}")
print(f"  Total India pop    : {pop_df['pop_sum'].sum():,.0f}")
print(f"  Pincodes pop > 0   : {(pop_df['pop_sum'] > 0).sum():,}")

print(f"\n  Top 10 pincodes by pop_density:")
disp_cols = ["pincode", "pop_sum", "pop_density", "settled_area_ratio", "high_density_ratio"]
disp_cols = [c for c in disp_cols if c in pop_df.columns]
print(pop_df.nlargest(10, "pop_density")[disp_cols].to_string(index=False))

print(f"\n  Reference pincodes:")
refs = {110001: "New Delhi GPO", 400051: "BKC Mumbai",
        560034: "Koramangala",   122002: "Gurugram DLF",
        393002: "Ankleshwar (industrial)"}
for pc, name in refs.items():
    row = pop_df[pop_df["pincode"] == pc]
    if not row.empty:
        r = row.iloc[0]
        parts = [f"pop={r['pop_sum']:,.0f}", f"density={r['pop_density']:,.1f}/km²",
                 f"settled={r['settled_area_ratio']:.2f}"]
        if "ntl_per_capita" in pop_df.columns:
            parts.append(f"ntl_per_capita={r['ntl_per_capita']:.3f}")
        print(f"    {pc} {name}: {' | '.join(parts)}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — 500K stratified pixel sample (1km downsampled grid)
#
# Reads the TIF downsampled 12× to ~1km resolution (32 MB in memory).
# Builds a 1km pincode raster for pixel→pincode assignment.
# Stratified sample: per-pincode proportional to valid pixel count,
# ensuring every pincode with population has at least 1 sample.
# ═════════════════════════════════════════════════════════════════════════════
hline("Step 8 — 500K stratified pixel sample")

PIXEL_CSV  = "worldpop_pixels_sample.csv"
SAMPLE_N   = 500_000

if os.path.exists(PIXEL_CSV):
    print(f"  {PIXEL_CSV} already exists — loading from disk.")
    pixel_df = pd.read_csv(PIXEL_CSV)
else:
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.features import rasterize as _rasterize

    # ── 8a. Downsample WorldPop to 1km ────────────────────────────────────────
    FACTOR = 12          # 100m × 12 = 1200m ≈ 1km
    print(f"  Reading WorldPop at 1/{FACTOR} resolution (~1km) …")
    with rasterio.open(TIF) as src:
        out_h = src.height // FACTOR
        out_w  = src.width  // FACTOR
        data_1km = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.average,   # mean pop per 100m pixel in each 1km cell
        ).astype(np.float32)
        # Correct transform for the downsampled grid
        transform_1km = src.transform * Affine.scale(
            src.width  / out_w,
            src.height / out_h,
        )
    print(f"  Grid: {out_h} × {out_w}  ({out_h * out_w / 1e6:.2f}M cells, "
          f"{data_1km.nbytes / 1024**2:.0f} MB)")

    # ── 8b. Build 1km pincode raster ─────────────────────────────────────────
    if "gdf" not in dir():
        gdf = _load_pincodes()

    print(f"  Rasterizing {len(gdf):,} pincode polygons onto {out_h}×{out_w} grid …")
    shapes_1km = [
        (geom, idx + 1)
        for idx, geom in enumerate(gdf.geometry)
        if geom is not None and not geom.is_empty
    ]
    pincode_raster = _rasterize(
        shapes_1km,
        out_shape=(out_h, out_w),
        transform=transform_1km,
        fill=0,
        dtype="int32",
    )

    # Lookup array: 1-based sequential index → actual pincode integer
    pincode_lookup = np.zeros(len(gdf) + 1, dtype=np.int64)
    for idx, pc in enumerate(gdf["pincode"].values):
        pincode_lookup[idx + 1] = int(pc)

    # ── 8c. Collect all valid pixels with pincodes ────────────────────────────
    valid_mask = (data_1km > 0) & (data_1km != nodata) & ~np.isnan(data_1km)
    row_idx, col_idx = np.where(valid_mask)

    seq_vals     = pincode_raster[row_idx, col_idx]
    pincode_vals = pincode_lookup[seq_vals]
    pop_vals     = data_1km[row_idx, col_idx]

    # Pixel centres via rasterio (correct for any affine transform)
    lons_arr, lats_arr = rasterio.transform.xy(transform_1km, row_idx, col_idx)
    lons_arr = np.asarray(lons_arr, dtype=np.float32)
    lats_arr = np.asarray(lats_arr, dtype=np.float32)

    # Keep only pixels inside a pincode polygon
    inside       = pincode_vals > 0
    lons_arr     = lons_arr[inside]
    lats_arr     = lats_arr[inside]
    pop_vals     = pop_vals[inside]
    pincode_vals = pincode_vals[inside]
    print(f"  Valid pixels inside pincodes: {inside.sum():,}")

    # ── 8d. Stratified sample proportional to pincode pixel count ─────────────
    print(f"  Stratifying to {SAMPLE_N:,} samples (proportional to area per pincode) …")
    unique_pcs, pc_counts = np.unique(pincode_vals, return_counts=True)
    total_valid   = len(pincode_vals)

    # Proportional targets; every represented pincode gets at least 1 sample
    pc_targets = np.maximum(1, np.round(
        pc_counts / total_valid * SAMPLE_N
    ).astype(int))

    # Trim rounding overshoot from largest pincodes first
    overshoot = int(pc_targets.sum()) - SAMPLE_N
    if overshoot > 0:
        for i in np.argsort(-pc_targets):
            trim = min(pc_targets[i] - 1, overshoot)
            pc_targets[i] -= trim
            overshoot      -= trim
            if overshoot <= 0:
                break

    rng = np.random.default_rng(42)
    sampled_idx = []
    for pc, target in zip(unique_pcs, pc_targets):
        pool = np.where(pincode_vals == pc)[0]
        n    = min(int(target), len(pool))
        sampled_idx.append(rng.choice(pool, size=n, replace=False))

    idx_all  = np.concatenate(sampled_idx)
    pixel_df = pd.DataFrame({
        "latitude":              np.round(lats_arr[idx_all].astype(float), 6),
        "longitude":             np.round(lons_arr[idx_all].astype(float), 6),
        "pop_per_100m_pixel":    np.round(pop_vals[idx_all].astype(float), 4),
        "pincode":               pincode_vals[idx_all],
    })

    pixel_df.to_csv(PIXEL_CSV, index=False)
    size_kb = os.path.getsize(PIXEL_CSV) // 1024
    print(f"  saved → {PIXEL_CSV}  ({len(pixel_df):,} rows, {size_kb:,} KB)")

# ── Sample coverage report ────────────────────────────────────────────────────
grp = pixel_df.groupby("pincode").size()
print(f"\n  Pixel sample coverage:")
print(f"    Total samples          : {len(pixel_df):,}")
print(f"    Pincodes represented   : {pixel_df['pincode'].nunique():,}")
print(f"    Samples/pincode — min  : {grp.min()}")
print(f"    Samples/pincode — median:{grp.median():.0f}")
print(f"    Samples/pincode — max  : {grp.max():,}")
print(f"    Population range       : {pixel_df['pop_per_100m_pixel'].min():.4f}"
      f" – {pixel_df['pop_per_100m_pixel'].max():.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — Pincode-level EDA plots (population features)
# ═════════════════════════════════════════════════════════════════════════════
hline("Step 9 — pincode-level EDA charts")

# 9a. Population distribution across pincodes
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].hist(np.log1p(pop_df["pop_sum"]), bins=60,
             color=C["blue"], alpha=0.85, edgecolor="none")
axes[0].set_title("log₁₊(pop_sum) — total population")
axes[0].set_xlabel("log₁₊(persons)")
axes[0].set_ylabel("Pincodes")

axes[1].hist(np.log1p(pop_df["pop_density"].clip(upper=pop_df["pop_density"].quantile(0.99))),
             bins=60, color=C["teal"], alpha=0.85, edgecolor="none")
axes[1].set_title("log₁₊(pop_density) — persons/km²")
axes[1].set_xlabel("log₁₊(persons/km²)")
axes[1].set_ylabel("Pincodes")

axes[2].hist(pop_df["settled_area_ratio"], bins=40,
             color=C["purple"], alpha=0.85, edgecolor="none")
axes[2].set_title("settled_area_ratio")
axes[2].set_xlabel("Fraction of pincode with ≥ 1 resident")
axes[2].set_ylabel("Pincodes")

fig.suptitle("WorldPop India 2020 — pincode-level population distributions", fontsize=13)
plt.tight_layout()
save(fig, "07_pincode_population_distributions.png")

# 9b. Settlement structure: pop_cv vs pop_density scatter
fig, ax = plt.subplots(figsize=(9, 6))
x = np.log1p(pop_df["pop_density"].clip(upper=pop_df["pop_density"].quantile(0.995)))
y = pop_df["pop_cv"].clip(upper=5)
ax.hexbin(x, y, gridsize=60, cmap="YlOrRd", mincnt=1)
ax.set_xlabel("log₁₊(pop_density)  — persons/km²", fontsize=10)
ax.set_ylabel("pop_cv (coefficient of variation, clipped at 5)", fontsize=10)
ax.set_title("Settlement structure: density vs variability", fontsize=12)
cb = fig.colorbar(ax.collections[0], ax=ax, label="Pincode count")
fig.tight_layout()
save(fig, "08_density_vs_cv.png")

# 9c. NTL cross-signal charts (only if NTL features were computed)
if "ntl_per_capita" in pop_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x_ntl  = np.log1p(pop_df["pop_density"].clip(upper=pop_df["pop_density"].quantile(0.995)))
    y_ntl  = np.log1p(pop_df["ntl_per_capita"].clip(
        upper=pop_df["ntl_per_capita"].quantile(0.995)))
    axes[0].hexbin(x_ntl, y_ntl, gridsize=55, cmap="Blues", mincnt=1)
    axes[0].set_xlabel("log₁₊(pop_density)", fontsize=9)
    axes[0].set_ylabel("log₁₊(ntl_per_capita)", fontsize=9)
    axes[0].set_title("NTL per capita vs population density\n"
                      "(industrial = high NTL, low pop → top-left)", fontsize=10)
    fig.colorbar(axes[0].collections[0], ax=axes[0], label="Pincodes")

    y2 = np.log1p(pop_df["ntl_per_capita"].clip(
        upper=pop_df["ntl_per_capita"].quantile(0.995)))
    axes[1].hist(y2, bins=60, color=C["amber"], alpha=0.85, edgecolor="none")
    axes[1].set_xlabel("log₁₊(ntl_per_capita)", fontsize=9)
    axes[1].set_ylabel("Pincodes", fontsize=9)
    axes[1].set_title("Distribution of NTL per capita", fontsize=10)

    fig.suptitle("NTL × Population cross-signal features", fontsize=13)
    plt.tight_layout()
    save(fig, "09_ntl_per_capita.png")

# 9d. Pixel sample: spatial spread (lat/lon scatter coloured by population)
fig, ax = plt.subplots(figsize=(9, 10))
sample_plot = pixel_df.sample(min(100_000, len(pixel_df)), random_state=1)
sc = ax.scatter(
    sample_plot["longitude"], sample_plot["latitude"],
    c=np.log1p(sample_plot["pop_per_100m_pixel"]),
    s=0.4, alpha=0.5, cmap="YlOrRd", rasterized=True,
)
fig.colorbar(sc, ax=ax, label="log₁₊(pop per 100m pixel)", shrink=0.6)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"WorldPop pixel sample — {len(sample_plot):,} points\n"
             f"(colour = log population density)", fontsize=11)
ax.set_aspect("equal")
fig.tight_layout()
save(fig, "10_pixel_sample_spatial.png")


# ═════════════════════════════════════════════════════════════════════════════
# DONE
# ═════════════════════════════════════════════════════════════════════════════
hline("All done")
print(f"\n  Output folder : {OUT}/")
print("  Files written :")
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(f"{OUT}/{f}") // 1024
    print(f"    {f:<45} {size:>5} KB")

print(f"\n  Root CSVs:")
for csv in [PINCODE_CSV, PIXEL_CSV]:
    if os.path.exists(csv):
        size_kb = os.path.getsize(csv) // 1024
        print(f"    {csv:<40} {size_kb:>6} KB")