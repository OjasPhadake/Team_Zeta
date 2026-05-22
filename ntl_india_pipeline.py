"""
ntl_india_pipeline.py
=====================
Fetches NASA Black Marble nighttime light data at 500m resolution for India
and aggregates it to pincode level, producing a ready-to-use CSV.

TWO BACKENDS — pick one:
  --backend blackmarble   Uses blackmarblepy (World Bank package).
                          Downloads HDF5 tiles → xarray → zonal stats.
                          Best for offline / batch use.

  --backend gee           Uses Google Earth Engine Python API.
                          Runs entirely in the cloud, no large downloads.
                          Best for full-India in one shot.

OUTPUTS (both backends):
  ntl_pixels_500m.csv          Every 500m pixel: lat, lon, ntl_radiance
  ntl_pincode_aggregated.csv   One row per pincode: ntl_mean, ntl_max,
                               ntl_sum, ntl_std, ntl_lit_pixel_count

SETUP:
  pip install blackmarblepy geopandas rasterstats pandas numpy rasterio
  pip install earthengine-api          # for GEE backend

  NASA token  → https://urs.earthdata.nasa.gov/  (free account → Profile → Token)
  GEE auth    → run:  earthengine authenticate
  Pincode GeoJSON → https://data.opencity.in/dataset/india-pincode-maps-2025

QUICKSTART:
  # BlackMarblePy (single state, e.g. Maharashtra):
  python ntl_india_pipeline.py --backend blackmarble \\
      --token YOUR_NASA_TOKEN \\
      --state Maharashtra \\
      --year 2023 \\
      --pincodes india_pincodes_2025.geojson

  # Google Earth Engine (full India, no download):
  python ntl_india_pipeline.py --backend gee \\
      --year 2023 \\
      --pincodes india_pincodes_2025.geojson
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# INDIA BOUNDING BOXES  (min_lon, min_lat, max_lon, max_lat)
# Add your state here if missing.
# ─────────────────────────────────────────────────────────────────────────────
STATE_BBOX = {
    "india":             (68.1, 6.5,  97.4, 37.1),
    "maharashtra":       (72.6, 15.6, 80.9, 22.0),
    "karnataka":         (74.0, 11.5, 78.6, 18.5),
    "tamilnadu":         (76.2, 8.0,  80.4, 13.6),
    "gujarat":           (68.2, 20.1, 74.5, 24.7),
    "rajasthan":         (69.5, 23.1, 78.3, 30.2),
    "uttarpradesh":      (77.1, 23.9, 84.6, 30.4),
    "delhi":             (76.8, 28.4, 77.4, 28.9),
    "westbengal":        (85.8, 21.5, 89.9, 27.2),
    "andhrapradesh":     (76.8, 12.6, 84.8, 19.9),
    "telangana":         (77.2, 15.8, 81.3, 19.9),
    "madhyapradesh":     (74.0, 21.1, 82.8, 26.9),
    "kerala":            (74.8, 8.3,  77.4, 12.8),
    "odisha":            (81.4, 17.8, 87.5, 22.6),
    "chhattisgarh":      (80.2, 17.8, 84.4, 24.1),
    "punjab":            (73.9, 29.5, 76.9, 32.5),
    "haryana":           (74.5, 27.7, 77.6, 30.9),
    "bihar":             (83.3, 24.3, 88.3, 27.5),
    "jharkhand":         (83.3, 21.9, 87.9, 25.3),
    "assam":             (89.7, 24.1, 96.0, 28.2),
    "himachalpradesh":   (75.6, 30.4, 79.0, 33.2),
    "uttarakhand":       (77.6, 28.7, 81.1, 31.5),
    "goa":               (73.7, 14.9, 74.3, 15.8),
}


# ═════════════════════════════════════════════════════════════════════════════
# BACKEND 1 — BlackMarblePy
# ═════════════════════════════════════════════════════════════════════════════

def run_blackmarble(args, pincodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Downloads NASA Black Marble tiles via blackmarblepy,
    reprojects to WGS84 GeoTIFF, then runs zonal stats per pincode.
    """
    try:
        from blackmarble import BlackMarble, Product
        import xarray as xr
    except ImportError:
        sys.exit("Install blackmarblepy:  pip install blackmarblepy xarray")

    token = args.token or os.environ.get("BLACKMARBLE_TOKEN")
    if not token:
        sys.exit(
            "NASA Earthdata token required.\n"
            "  Get one free at: https://urs.earthdata.nasa.gov/\n"
            "  Pass via --token or set env var BLACKMARBLE_TOKEN"
        )

    # ── Region of interest ───────────────────────────────────────────────────
    state_key = args.state.lower().replace(" ", "")
    if state_key not in STATE_BBOX:
        sys.exit(
            f"State '{args.state}' not in STATE_BBOX dict.\n"
            f"Available: {', '.join(STATE_BBOX.keys())}\n"
            f"Or add it manually to STATE_BBOX at the top of this file."
        )
    min_lon, min_lat, max_lon, max_lat = STATE_BBOX[state_key]

    from shapely.geometry import box
    roi_gdf = gpd.GeoDataFrame(
        geometry=[box(min_lon, min_lat, max_lon, max_lat)],
        crs="EPSG:4326"
    )

    # ── Product selection ────────────────────────────────────────────────────
    # VNP46A3 = monthly composite (recommended: average over 12 months for annual)
    # VNP46A4 = yearly composite (single download, less granularity)
    product = Product.VNP46A4 if args.annual else Product.VNP46A3

    # ── Date range ───────────────────────────────────────────────────────────
    if args.annual:
        # Yearly product: one entry per year
        date_range = pd.date_range(
            start=f"{args.year}-01-01",
            end=f"{args.year}-12-31",
            freq="YS"   # year-start
        )
    else:
        # Monthly product: all 12 months → we'll average them
        date_range = pd.date_range(
            start=f"{args.year}-01-01",
            end=f"{args.year}-12-01",
            freq="MS"   # month-start
        )

    print(f"\n{'='*60}")
    print(f"  Backend      : BlackMarblePy")
    print(f"  State/Region : {args.state}")
    print(f"  Product      : {product.name}  ({'annual' if args.annual else 'monthly avg'})")
    print(f"  Year         : {args.year}")
    print(f"  Pincodes     : {len(pincodes_gdf)}")
    print(f"{'='*60}\n")

    # ── Download + extract raster ────────────────────────────────────────────
    print("Downloading Black Marble tiles (may take a few minutes)...")
    bm = BlackMarble(
        token=token,
        output_directory=Path("ntl_tiles_cache"),
        output_skip_if_exists=True,     # resume-safe: skips already-downloaded tiles
        drop_values_by_quality_flag=[255],  # drop fill/no-data pixels
    )

    # bm.raster() returns an xarray.Dataset clipped to roi_gdf
    ds = bm.raster(
        roi_gdf,
        product_id=product,
        date_range=date_range,
    )

    # ── Average monthly composites → single annual raster ───────────────────
    # Variable name: 'Gap_Filled_DNB_BRDF-Corrected_NTL' for VNP46A3/A4
    var_name = [v for v in ds.data_vars][0]
    print(f"  Raster variable: {var_name}")

    if "time" in ds.dims and len(ds.time) > 1:
        print(f"  Averaging {len(ds.time)} monthly composites → annual mean...")
        ntl_arr = ds[var_name].mean(dim="time").values
    else:
        ntl_arr = ds[var_name].squeeze().values

    # ── Write to GeoTIFF for zonal stats ────────────────────────────────────
    tif_path = f"ntl_{state_key}_{args.year}.tif"
    height, width = ntl_arr.shape

    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
    with rasterio.open(
        tif_path, "w",
        driver="GTiff",
        height=height, width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(ntl_arr.astype(np.float32), 1)
    print(f"  GeoTIFF written → {tif_path}")

    # ── Pixel-level CSV (500m grid) ──────────────────────────────────────────
    pixels_df = _geotiff_to_pixel_csv(tif_path, args.year)
    pixels_df.to_csv("ntl_pixels_500m.csv", index=False)
    print(f"  Pixel CSV  → ntl_pixels_500m.csv  ({len(pixels_df):,} lit pixels)")

    # ── Zonal stats per pincode ──────────────────────────────────────────────
    pincodes_gdf = _zonal_stats_per_pincode(pincodes_gdf, tif_path, args.pincode_col)
    return pincodes_gdf


# ═════════════════════════════════════════════════════════════════════════════
# BACKEND 2 — Google Earth Engine (cloud, no big download)
# ═════════════════════════════════════════════════════════════════════════════

def run_gee(args, pincodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Uses Google Earth Engine Python API to compute per-pincode NTL stats
    entirely in the cloud, then downloads a small results CSV.

    No large raster downloads. Scales to full India with no memory issues.
    """
    try:
        import ee
    except ImportError:
        sys.exit("Install GEE client:  pip install earthengine-api")

    # ── Authenticate ─────────────────────────────────────────────────────────
    project = args.gee_project or os.environ.get("EE_PROJECT")
    if not project:
        sys.exit(
            "GEE backend requires a Google Cloud project ID.\n"
            "  Pass via --gee-project YOUR_PROJECT_ID\n"
            "  or set env var:  export EE_PROJECT=your-project-id\n"
            "  (Enable Earth Engine API at: https://console.cloud.google.com/)"
        )
    try:
        ee.Initialize(project=project)
        print(f"  GEE: Authenticated (project={project}).")
    except Exception:
        print("  GEE: Running authentication (browser will open)...")
        ee.Authenticate()
        ee.Initialize(project=project)

    print(f"\n{'='*60}")
    print(f"  Backend  : Google Earth Engine")
    print(f"  Year     : {args.year}")
    print(f"  Pincodes : {len(pincodes_gdf)}")
    print(f"  Scale    : 500m")
    print(f"{'='*60}\n")

    # ── Annual mean NTL composite ─────────────────────────────────────────────
    # VIIRS/DNB Monthly V1 cloud-free composite — avg_rad band (nW·cm⁻²·sr⁻¹)
    ntl = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate(f"{args.year}-01-01", f"{args.year}-12-31")
        .select("avg_rad")
        .mean()
        .rename("ntl_radiance")
    )

    # ── Process pincodes in chunks (GEE has a 5000-feature limit per call) ───
    CHUNK = 500
    n_pincodes = len(pincodes_gdf)
    print(f"Processing {n_pincodes} pincodes in chunks of {CHUNK}...")

    all_results = []
    pincode_col = args.pincode_col

    for start in range(0, n_pincodes, CHUNK):
        end = min(start + CHUNK, n_pincodes)
        chunk = pincodes_gdf.iloc[start:end].copy()
        print(f"  Chunk {start}–{end} ...", end=" ", flush=True)

        # Convert chunk to EE FeatureCollection
        features = []
        for _, row in chunk.iterrows():
            geom_json = row.geometry.__geo_interface__
            feat = ee.Feature(
                ee.Geometry(geom_json),
                {pincode_col: str(row[pincode_col])}
            )
            features.append(feat)
        fc = ee.FeatureCollection(features)

        # Zonal stats — mean, max, sum, stdDev per pincode
        reducer = (
            ee.Reducer.mean()
            .combine(ee.Reducer.max(),    sharedInputs=True)
            .combine(ee.Reducer.sum(),    sharedInputs=True)
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.count(),  sharedInputs=True)
        )

        stats = ntl.reduceRegions(
            collection=fc,
            reducer=reducer,
            scale=500,           # 500m — native Black Marble resolution
            tileScale=4,         # parallelism; increase to 8 if quota errors
        )

        # Pull results synchronously (small per-chunk payload)
        try:
            info = stats.getInfo()
            for feat in info["features"]:
                p = feat["properties"]
                all_results.append({
                    pincode_col:         p.get(pincode_col, ""),
                    "ntl_mean":          round(p.get("mean",   0) or 0, 4),
                    "ntl_max":           round(p.get("max",    0) or 0, 4),
                    "ntl_sum":           round(p.get("sum",    0) or 0, 4),
                    "ntl_std":           round(p.get("stdDev", 0) or 0, 4),
                    "ntl_lit_pixel_count": int(p.get("count",  0) or 0),
                })
            print(f"OK ({len(info['features'])} features)")
        except Exception as e:
            print(f"WARN: chunk failed — {e}")
            # Mark chunk as missing; fill later
            for _, row in chunk.iterrows():
                all_results.append({
                    pincode_col: str(row[pincode_col]),
                    "ntl_mean": np.nan, "ntl_max": np.nan,
                    "ntl_sum": np.nan, "ntl_std": np.nan,
                    "ntl_lit_pixel_count": 0,
                })

    results_df = pd.DataFrame(all_results)

    # ── Generate pixel-level CSV via sampling (representative grid) ──────────
    print("\nGenerating 500m pixel CSV for India bounding box...")
    pixels_df = _gee_pixel_sample(ntl, pincodes_gdf, args.year)
    pixels_df.to_csv("ntl_pixels_500m.csv", index=False)
    print(f"  Pixel CSV → ntl_pixels_500m.csv  ({len(pixels_df):,} pixels)")

    # ── Merge stats back into GeoDataFrame ──────────────────────────────────
    pincodes_gdf = pincodes_gdf.merge(
        results_df, on=pincode_col, how="left"
    )
    return pincodes_gdf


def _gee_pixel_sample(ntl_image, pincodes_gdf, year, max_pixels=5_000):
    """
    Sample the NTL image at 500m resolution within India's extent,
    returning a DataFrame of (latitude, longitude, ntl_radiance, year).
    Capped at max_pixels ≤ 5000 (GEE collection query limit). This is a
    representative sample; full pincode aggregates are in the main output.
    """
    import ee

    bounds = pincodes_gdf.total_bounds  # [min_lon, min_lat, max_lon, max_lat]
    region = ee.Geometry.Rectangle(
        [bounds[0], bounds[1], bounds[2], bounds[3]]
    )

    samples = ntl_image.addBands(ee.Image.pixelLonLat()).sample(
        region=region,
        scale=500,
        numPixels=max_pixels,
        seed=42,
        geometries=False,
    )

    print(f"  Sampling up to {max_pixels:,} pixels at 500m ...", end=" ", flush=True)
    info = samples.getInfo()
    feats = info.get("features", [])
    print(f"{len(feats):,} returned")

    rows = []
    for f in feats:
        p = f["properties"]
        rad = p.get("ntl_radiance", 0) or 0
        if rad > 0:
            rows.append({
                "latitude":      round(p.get("latitude",  0), 6),
                "longitude":     round(p.get("longitude", 0), 6),
                "ntl_radiance":  round(rad, 4),
                "year":          year,
            })

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _geotiff_to_pixel_csv(tif_path: str, year: int) -> pd.DataFrame:
    """Convert a GeoTIFF to a flat CSV of lit pixels with lat/lon."""
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform

    rows_idx, cols_idx = np.where(
        (~np.isnan(data)) & (data > 0)
    )
    if len(rows_idx) == 0:
        return pd.DataFrame(columns=["latitude", "longitude", "ntl_radiance", "year"])

    lons, lats = rasterio.transform.xy(transform, rows_idx, cols_idx)
    return pd.DataFrame({
        "latitude":     np.round(np.array(lats), 6),
        "longitude":    np.round(np.array(lons), 6),
        "ntl_radiance": np.round(data[rows_idx, cols_idx], 4),
        "year":         year,
    })


def _zonal_stats_per_pincode(
    pincodes_gdf: gpd.GeoDataFrame,
    tif_path: str,
    pincode_col: str,
) -> gpd.GeoDataFrame:
    """
    Run rasterstats.zonal_stats against a GeoTIFF and append
    ntl_mean, ntl_max, ntl_sum, ntl_std, ntl_lit_pixel_count columns.
    Uses all_touched=True so even the smallest pincode gets at least one pixel.
    """
    try:
        from rasterstats import zonal_stats
    except ImportError:
        sys.exit("Install rasterstats:  pip install rasterstats")

    if pincodes_gdf.crs.to_epsg() != 4326:
        pincodes_gdf = pincodes_gdf.to_crs("EPSG:4326")

    print(f"  Running zonal stats for {len(pincodes_gdf)} pincodes...", end=" ", flush=True)
    stats = zonal_stats(
        pincodes_gdf,
        tif_path,
        stats=["mean", "max", "sum", "std", "count"],
        nodata=np.nan,
        all_touched=True,   # critical for small pincodes < 1 pixel
        geojson_out=False,
    )

    pincodes_gdf = pincodes_gdf.copy()
    pincodes_gdf["ntl_mean"]            = [round(s["mean"]   or 0, 4) for s in stats]
    pincodes_gdf["ntl_max"]             = [round(s["max"]    or 0, 4) for s in stats]
    pincodes_gdf["ntl_sum"]             = [round(s["sum"]    or 0, 4) for s in stats]
    pincodes_gdf["ntl_std"]             = [round(s["std"]    or 0, 4) for s in stats]
    pincodes_gdf["ntl_lit_pixel_count"] = [int(s["count"]   or 0)    for s in stats]
    print("done")

    missing = (pincodes_gdf["ntl_lit_pixel_count"] == 0).sum()
    total   = len(pincodes_gdf)
    print(f"  Coverage: {total - missing}/{total} pincodes ({100*(total-missing)/total:.1f}%)")

    if missing > 0:
        print(f"  Filling {missing} gap pincodes with nearest-neighbour imputation...")
        pincodes_gdf = _fill_gaps(pincodes_gdf)

    return pincodes_gdf


def _fill_gaps(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    KNN imputation for pincodes with zero pixel coverage.
    Uses 5 nearest neighbours weighted by inverse distance.
    """
    try:
        from sklearn.neighbors import KNeighborsRegressor
    except ImportError:
        print("  sklearn not found — skipping gap fill (pip install scikit-learn)")
        return gdf

    has  = gdf["ntl_lit_pixel_count"] > 0
    miss = ~has
    if miss.sum() == 0:
        return gdf

    centroids = gdf.geometry.centroid
    X = np.column_stack([centroids.x, centroids.y])

    for col in ["ntl_mean", "ntl_max", "ntl_sum", "ntl_std"]:
        knn = KNeighborsRegressor(n_neighbors=min(5, has.sum()), weights="distance")
        knn.fit(X[has], gdf.loc[has, col])
        gdf.loc[miss, col] = knn.predict(X[miss]).round(4)

    gdf.loc[miss, "ntl_imputed"] = True
    gdf.loc[has,  "ntl_imputed"] = False
    print(f"  Gap fill complete — {miss.sum()} pincodes imputed.")
    return gdf


def _save_pincode_csv(pincodes_gdf: gpd.GeoDataFrame, pincode_col: str, year: int):
    """Save the final pincode-level aggregated CSV (no geometry column)."""
    keep_cols = [pincode_col, "ntl_mean", "ntl_max", "ntl_sum",
                 "ntl_std", "ntl_lit_pixel_count"]
    if "ntl_imputed" in pincodes_gdf.columns:
        keep_cols.append("ntl_imputed")

    # Add any other non-geometry columns already in the GDF
    extra = [c for c in pincodes_gdf.columns
             if c not in keep_cols + ["geometry"]]
    cols = extra + keep_cols

    out = pincodes_gdf[[c for c in cols if c in pincodes_gdf.columns]].copy()
    out.to_csv("ntl_pincode_aggregated.csv", index=False)
    print(f"\n  Pincode CSV → ntl_pincode_aggregated.csv  ({len(out):,} rows)")
    print(f"\n  Column descriptions:")
    print(f"    ntl_mean            — avg radiance (nW·cm⁻²·sr⁻¹) across all 500m pixels in pincode")
    print(f"    ntl_max             — peak radiance pixel in pincode (urban core signal)")
    print(f"    ntl_sum             — total light emission (proxy for economic activity)")
    print(f"    ntl_std             — spatial variability (high = urban–rural mix)")
    print(f"    ntl_lit_pixel_count — number of pixels with radiance > 0")
    if "ntl_imputed" in pincodes_gdf.columns:
        print(f"    ntl_imputed         — True if value was imputed (no direct pixel overlap)")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="NASA Black Marble NTL → India pincode aggregation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--backend", choices=["blackmarble", "gee"], default="gee",
        help="Data backend. 'gee' = Google Earth Engine (recommended for full India). "
             "'blackmarble' = blackmarblepy (state-by-state, offline-capable)."
    )
    ap.add_argument(
        "--pincodes", default="india_pincodes.geojson",
        help="Path to pincode GeoJSON/shapefile. "
             "Download from: https://data.opencity.in/dataset/india-pincode-maps-2025"
    )
    ap.add_argument(
        "--pincode-col", dest="pincode_col", default="pincode",
        help="Column name in pincodes file that holds the pincode value. "
             "If not found, the pipeline auto-detects common alternatives or "
             "generates a synthetic 'pincode_id' column."
    )
    ap.add_argument(
        "--year", type=int, default=2023,
        help="Year for NTL data (2012–2023 available)."
    )
    ap.add_argument(
        "--state", default="india",
        help="[BlackMarble backend only] State name (e.g. Maharashtra). "
             "Use 'india' for full country (slow). See STATE_BBOX dict."
    )
    ap.add_argument(
        "--token", default=None,
        help="[BlackMarble backend only] NASA Earthdata bearer token. "
             "Or set env var BLACKMARBLE_TOKEN."
    )
    ap.add_argument(
        "--annual", action="store_true",
        help="[BlackMarble backend only] Use VNP46A4 annual product instead of "
             "averaging 12 monthly VNP46A3 composites."
    )
    ap.add_argument(
        "--gee-project", dest="gee_project", default=None,
        help="[GEE backend only] Google Cloud project ID with Earth Engine API enabled. "
             "Or set env var EE_PROJECT."
    )
    args = ap.parse_args()

    # ── Load pincode boundaries ───────────────────────────────────────────────
    if not Path(args.pincodes).exists():
        print(f"ERROR: Pincode file not found: {args.pincodes}")
        print("Download it from: https://data.opencity.in/dataset/india-pincode-maps-2025")
        sys.exit(1)

    print(f"\nLoading pincode boundaries from {args.pincodes}...")
    pincodes_gdf = gpd.read_file(args.pincodes)
    if pincodes_gdf.crs is None:
        pincodes_gdf = pincodes_gdf.set_crs("EPSG:4326")
    elif pincodes_gdf.crs.to_epsg() != 4326:
        pincodes_gdf = pincodes_gdf.to_crs("EPSG:4326")

    if args.pincode_col not in pincodes_gdf.columns:
        # Try common alternative column names
        _PINCODE_ALTS = ["Pincode", "PINCODE", "PIN", "pin", "postcode",
                         "zipcode", "zip", "id", "ID", "FID", "name", "Name"]
        _found = next((c for c in _PINCODE_ALTS if c in pincodes_gdf.columns), None)
        if _found:
            args.pincode_col = _found
            print(f"  Pincode column '{args.pincode_col}' not found — "
                  f"using '{_found}' instead.")
        else:
            # Generate sequential synthetic IDs (zero-padded 6-digit strings)
            pincodes_gdf = pincodes_gdf.reset_index(drop=True)
            pincodes_gdf["pincode_id"] = (
                pincodes_gdf.index.astype(str).str.zfill(6)
            )
            args.pincode_col = "pincode_id"
            print(f"  No pincode column found in file "
                  f"(columns: {list(pincodes_gdf.columns[:-1])}).")
            print(f"  Generated synthetic sequential IDs → column 'pincode_id'.")

    print(f"  Loaded {len(pincodes_gdf):,} pincode polygons.")

    # ── Run selected backend ──────────────────────────────────────────────────
    if args.backend == "blackmarble":
        pincodes_gdf = run_blackmarble(args, pincodes_gdf)
    else:
        pincodes_gdf = run_gee(args, pincodes_gdf)

    # ── Save pincode CSV ──────────────────────────────────────────────────────
    _save_pincode_csv(pincodes_gdf, args.pincode_col, args.year)

    print("\n" + "="*60)
    print("  ALL DONE")
    print("  Output files:")
    print("    ntl_pixels_500m.csv         — 500m pixel grid (lat, lon, radiance)")
    print("    ntl_pincode_aggregated.csv  — one row per pincode")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()