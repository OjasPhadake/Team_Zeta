"""
pixel_extract.py
================
Extracts ALL 500m NTL pixels for India via a GEE batch export.
Writes ntl_pixels_500m.csv — never touches ntl_pincode_aggregated.csv.

TWO-PHASE USAGE
───────────────
Phase 1 — submit the export task and wait:
  python pixel_extract.py --gee-project save-hrs-ids --year 2023

  When the task completes the script tries to auto-download the GeoTIFF
  from Google Drive.  If auto-download is unavailable (see note below),
  it prints manual download instructions and exits.

Phase 2 — convert an already-downloaded GeoTIFF to CSV:
  python pixel_extract.py --convert ntl_india_pixels_2023.tif --year 2023

AUTO-DOWNLOAD NOTE
──────────────────
Auto-download uses the Google Drive API.  It works when application-default
credentials include Drive scope.  If you hit an auth error run once:

  gcloud auth application-default login \
      --scopes https://www.googleapis.com/auth/drive.readonly,\
https://www.googleapis.com/auth/cloud-platform

then re-run Phase 1 (the task is already done — it will skip straight to
download + convert).

OUTPUT SIZE WARNING
───────────────────
The GeoTIFF is ~50–150 MB.  The pixel CSV (all lit pixels in India) can
be 300–900 MB.  Make sure you have enough free disk space.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


INDIA_BBOX   = [68.1, 6.5, 97.4, 37.1]   # [min_lon, min_lat, max_lon, max_lat]
DRIVE_FOLDER = "EarthEngineExports"        # GEE always writes here by default
OUT_CSV      = "ntl_pixels_500m.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1a — submit GEE batch export task
# ─────────────────────────────────────────────────────────────────────────────

def submit_export(year: int, project: str) -> "ee.batch.Task":
    import ee

    ntl = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .select("avg_rad")
        .mean()
        .rename("ntl_radiance")
    )

    region = ee.Geometry.Rectangle(INDIA_BBOX)
    filename = f"ntl_india_pixels_{year}"

    task = ee.batch.Export.image.toDrive(
        image          = ntl,
        description    = filename,
        fileNamePrefix = filename,
        folder         = DRIVE_FOLDER,
        region         = region,
        scale          = 500,
        crs            = "EPSG:4326",
        maxPixels      = int(1e10),
        fileFormat     = "GeoTIFF",
        formatOptions  = {"cloudOptimized": True},
    )
    task.start()
    print(f"  Task ID  : {task.id}")
    print(f"  Monitor  : https://code.earthengine.google.com/tasks")
    return task, filename


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1b — poll until the task finishes
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_task(task, poll_interval: int = 30):
    print(f"  Polling every {poll_interval}s (Ctrl-C to abort polling; task keeps running)...")
    try:
        while True:
            status = task.status()
            state  = status["state"]
            ts     = time.strftime("%H:%M:%S")
            print(f"  [{ts}]  {state}", flush=True)
            if state == "COMPLETED":
                print("  Export complete.\n")
                return
            if state in ("FAILED", "CANCELLED"):
                msg = status.get("error_message", "no details")
                sys.exit(f"  Task {state}: {msg}")
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print(
            "\n  Polling stopped (task is still running in GEE).\n"
            "  Re-run with --convert once the task finishes and you've\n"
            f"  downloaded '{DRIVE_FOLDER}/<filename>.tif' from Google Drive."
        )
        sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1c — auto-download from Google Drive
# ─────────────────────────────────────────────────────────────────────────────

def download_from_drive(drive_filename: str, dest_path: str) -> bool:
    """
    Returns True on success, False if credentials lack Drive scope
    (caller will print manual instructions).
    """
    try:
        import google.auth
        from googleapiclient import discovery
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError:
        print("  google-api-python-client not found — skipping auto-download.")
        print("  pip install google-api-python-client")
        return False

    DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
    try:
        creds, _ = google.auth.default(scopes=[DRIVE_SCOPE])
    except Exception as exc:
        print(f"  Drive auth failed ({exc})")
        return False

    try:
        drive = discovery.build("drive", "v3", credentials=creds, cache_discovery=False)

        # Locate the folder first (GEE creates it if it doesn't exist)
        folder_res = drive.files().list(
            q=(f"name='{DRIVE_FOLDER}' "
               f"and mimeType='application/vnd.google-apps.folder' "
               f"and trashed=false"),
            fields="files(id)",
        ).execute()
        folders = folder_res.get("files", [])
        parent_clause = ""
        if folders:
            parent_clause = f" and '{folders[0]['id']}' in parents"

        # Find the exported file
        file_res = drive.files().list(
            q=f"name='{drive_filename}' and trashed=false{parent_clause}",
            fields="files(id, name, size)",
        ).execute()
        files = file_res.get("files", [])
        if not files:
            print(f"  File '{drive_filename}' not found in Drive yet.")
            return False

        file_id = files[0]["id"]
        size_mb = int(files[0].get("size", 0)) / 1024 / 1024
        print(f"  Found '{drive_filename}' ({size_mb:.1f} MB) — downloading ...")

        request = drive.files().get_media(fileId=file_id)
        with open(dest_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                progress, done = downloader.next_chunk()
                pct = int(progress.progress() * 100)
                print(f"    {pct:3d}%", end="\r", flush=True)
        print(f"  Downloaded → {dest_path}    ")
        return True

    except Exception as exc:
        print(f"  Drive download error: {exc}")
        return False


def print_manual_instructions(drive_filename: str, tif_path: str, year: int):
    print()
    print("  ── Manual download instructions ──────────────────────────")
    print(f"  1. Open https://drive.google.com")
    print(f"  2. Navigate to folder '{DRIVE_FOLDER}'")
    print(f"  3. Download '{drive_filename}'  →  save as  '{tif_path}'")
    print(f"  4. Then run:")
    print(f"       python pixel_extract.py --convert {tif_path} --year {year}")
    print("  ──────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — convert GeoTIFF → pixel CSV
# ─────────────────────────────────────────────────────────────────────────────

def geotiff_to_pixel_csv(tif_path: str, year: int, out_csv: str = OUT_CSV):
    print(f"\nConverting {tif_path} → {out_csv} ...")

    with rasterio.open(tif_path) as src:
        data      = src.read(1).astype(np.float64)
        transform = src.transform
        nodata    = src.nodata

    # Keep only valid lit pixels
    valid = (data > 0) & ~np.isnan(data)
    if nodata is not None:
        valid &= data != nodata

    rows_idx, cols_idx = np.where(valid)
    n_lit = len(rows_idx)

    if n_lit == 0:
        print("  WARNING: no lit pixels found in raster.")
        pd.DataFrame(columns=["latitude", "longitude", "ntl_radiance", "year"]).to_csv(
            out_csv, index=False
        )
        return

    print(f"  Lit pixels found : {n_lit:,}")
    lons, lats = rasterio.transform.xy(transform, rows_idx, cols_idx)

    df = pd.DataFrame({
        "latitude":     np.round(np.asarray(lats, dtype=np.float64), 6),
        "longitude":    np.round(np.asarray(lons, dtype=np.float64), 6),
        "ntl_radiance": np.round(data[rows_idx, cols_idx], 4),
        "year":         year,
    })
    df.to_csv(out_csv, index=False)

    size_mb = Path(out_csv).stat().st_size / 1024 / 1024
    print(f"  Written → {out_csv}  ({n_lit:,} rows, {size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Full India NTL pixel extraction via GEE batch export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--gee-project", dest="gee_project", default=None,
                    help="Google Cloud project ID (or set env EE_PROJECT)")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--convert", metavar="TIF", default=None,
                    help="Skip export: convert a local GeoTIFF directly to CSV")
    ap.add_argument("--keep-tif", action="store_true",
                    help="Keep the GeoTIFF after conversion (default: delete it)")
    args = ap.parse_args()

    tif_path = args.convert or f"ntl_india_pixels_{args.year}.tif"

    # ── Phase 2 only: user already has the GeoTIFF ──────────────────────────
    if args.convert:
        if not Path(args.convert).exists():
            sys.exit(f"ERROR: file not found: {args.convert}")
        geotiff_to_pixel_csv(args.convert, args.year)
        return

    # ── Phase 1: GEE auth ────────────────────────────────────────────────────
    try:
        import ee
    except ImportError:
        sys.exit("pip install earthengine-api")

    project = args.gee_project or os.environ.get("EE_PROJECT")
    if not project:
        sys.exit(
            "Google Cloud project required.\n"
            "  Pass --gee-project YOUR_PROJECT_ID  or  set EE_PROJECT env var."
        )

    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    print(f"  GEE authenticated (project={project})\n")

    # ── Phase 1a: submit ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  Submitting GEE batch export ...")
    print("=" * 60)
    task, drive_filename = submit_export(args.year, project)
    drive_filename_with_ext = drive_filename + ".tif"

    # ── Phase 1b: wait ───────────────────────────────────────────────────────
    wait_for_task(task)

    # ── Phase 1c: download ───────────────────────────────────────────────────
    print("Attempting auto-download from Google Drive ...")
    ok = download_from_drive(drive_filename_with_ext, tif_path)

    if not ok:
        print_manual_instructions(drive_filename_with_ext, tif_path, args.year)
        sys.exit(0)

    # ── Phase 2: convert ─────────────────────────────────────────────────────
    geotiff_to_pixel_csv(tif_path, args.year)

    if not args.keep_tif:
        Path(tif_path).unlink()
        print(f"  Removed {tif_path}  (pass --keep-tif to retain it)")

    print("\n" + "=" * 60)
    print("  DONE  →  ntl_pixels_500m.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
