import pandas as pd
import numpy as np

SEP = "=" * 60

# ── 0. Replace synthetic pincode_id with real pincodes ────────────────────────
agg = pd.read_csv("ntl_pincode_aggregated.csv")

if "pincode_id" in agg.columns:
    pins = pd.read_csv("extracted_pincodes.csv")
    assert len(agg) == len(pins), (
        f"Row count mismatch: aggregated={len(agg)}, pincodes={len(pins)}"
    )
    agg = agg.drop(columns=["pincode_id"])
    agg.insert(0, "pincode", pins["Pincode"].values)
    agg.to_csv("ntl_pincode_aggregated.csv", index=False)
    print("Pincode replacement done.\n")

# Drop leftover KML metadata columns if present
kml_junk = [c for c in ["tessellate", "extrude", "visibility"] if c in agg.columns]
if kml_junk:
    agg = agg.drop(columns=kml_junk)

pix = pd.read_csv("ntl_pixels_500m.csv")

# ═══════════════════════════════════════════════════════════════
# 1. PINCODE-LEVEL AGGREGATED FILE
# ═══════════════════════════════════════════════════════════════
print(SEP)
print("  FILE: ntl_pincode_aggregated.csv")
print(SEP)

print(f"\nShape   : {agg.shape[0]:,} rows × {agg.shape[1]} columns")
print(f"Columns : {list(agg.columns)}")
print(f"Memory  : {agg.memory_usage(deep=True).sum() / 1024:.1f} KB")

print("\n── Dtypes ──")
print(agg.dtypes.to_string())

print("\n── Missing values ──")
missing = agg.isnull().sum()
print(missing[missing > 0].to_string() if missing.any() else "  None")

print("\n── Descriptive statistics (NTL columns) ──")
ntl_cols = ["ntl_mean", "ntl_max", "ntl_sum", "ntl_std", "ntl_lit_pixel_count"]
print(agg[ntl_cols].describe().round(4).to_string())

print("\n── Median (NTL columns) ──")
print(agg[ntl_cols].median().round(4).to_string())

print("\n── Dark pincodes (zero lit pixels) ──")
dark = (agg["ntl_lit_pixel_count"] == 0).sum()
print(f"  {dark:,} / {len(agg):,} pincodes ({100*dark/len(agg):.1f}%) have no lit pixels")

print("\n── Brightest 10 pincodes (ntl_mean) ──")
print(agg.nlargest(10, "ntl_max")[["pincode", "ntl_mean", "ntl_max", "ntl_lit_pixel_count"]].to_string(index=False))

print("\n── Dimmest 10 lit pincodes (ntl_mean, excluding zero-pixel) ──")
lit = agg[agg["ntl_lit_pixel_count"] > 0]
print(lit.nsmallest(10, "ntl_mean")[["pincode", "ntl_mean", "ntl_max", "ntl_lit_pixel_count"]].to_string(index=False))

print("\n── Total economic activity proxy (ntl_sum) — top 10 ──")
print(agg.nlargest(10, "ntl_sum")[["pincode", "ntl_sum", "ntl_lit_pixel_count"]].to_string(index=False))

print("\n── Pincode coverage by radiance bucket ──")
bins   = [0, 1, 5, 20, 50, 100, np.inf]
labels = ["0–1", "1–5", "5–20", "20–50", "50–100", "100+"]
agg["radiance_bucket"] = pd.cut(agg["ntl_mean"], bins=bins, labels=labels, right=False)
print(agg["radiance_bucket"].value_counts().sort_index().to_string())
agg = agg.drop(columns=["radiance_bucket"])

# ═══════════════════════════════════════════════════════════════
# 2. PIXEL-LEVEL FILE
# ═══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  FILE: ntl_pixels_500m.csv")
print(SEP)

print(f"\nShape   : {pix.shape[0]:,} rows × {pix.shape[1]} columns")
print(f"Columns : {list(pix.columns)}")
print(f"Memory  : {pix.memory_usage(deep=True).sum() / 1024:.1f} KB")

print("\n── Dtypes ──")
print(pix.dtypes.to_string())

print("\n── Missing values ──")
missing_pix = pix.isnull().sum()
print(missing_pix[missing_pix > 0].to_string() if missing_pix.any() else "  None")

print("\n── Descriptive statistics ──")
print(pix.describe().round(4).to_string())

print("\n── Median ──")
print(pix[["latitude", "longitude", "ntl_radiance"]].median().round(4).to_string())

print("\n── Geographic bounding box ──")
print(f"  Latitude  : {pix['latitude'].min():.4f} → {pix['latitude'].max():.4f}")
print(f"  Longitude : {pix['longitude'].min():.4f} → {pix['longitude'].max():.4f}")

print("\n── Radiance distribution ──")
bins_pix   = [0, 1, 5, 20, 50, 100, np.inf]
labels_pix = ["0–1", "1–5", "5–20", "20–50", "50–100", "100+"]
pix["radiance_bucket"] = pd.cut(pix["ntl_radiance"], bins=bins_pix, labels=labels_pix, right=False)
counts = pix["radiance_bucket"].value_counts().sort_index()
pcts   = (counts / len(pix) * 100).round(1)
summary = pd.DataFrame({"count": counts, "pct": pcts})
print(summary.to_string())

print("\n── Top 10 brightest pixels ──")
print(pix.nlargest(10, "ntl_radiance")[["latitude", "longitude", "ntl_radiance"]].to_string(index=False))

print(f"\n{SEP}")
print("  EDA COMPLETE")
print(SEP)
