"""
create_affluence_maps.py
========================
Generate affluence choropleth maps (PNG + interactive HTML) from affluence_ranking.csv.

Maps produced in affluence_plots/:
  india_affluence.png / .html           — All-India heatmap
  maharashtra_affluence.png / .html     — Maharashtra
  tamil_nadu_affluence.png / .html      — Tamil Nadu
  delhi_up_affluence.png / .html        — Delhi + Uttar Pradesh

Color scheme:
  - Affluence percentile: RdYlGn (red=low, green=high)
  - Contaminated (industrial/sparse/uninhabited): light grey, hatched border
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import plotly.graph_objects as go
import json
from pathlib import Path

SEP = "=" * 70
OUT = Path("affluence_plots")
OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load & merge data
# ─────────────────────────────────────────────────────────────────────────────
print(f"{SEP}\n  Loading data\n{SEP}")

gdf = gpd.read_file("india_pincodes.geojson")
pins = pd.read_csv("extracted_pincodes.csv")
gdf["pincode"] = pins["Pincode"].values

ranking = pd.read_csv("affluence_ranking.csv")

# Merge geometry with affluence
geo = gdf.merge(ranking, on="pincode", how="inner")
geo = geo.set_crs("EPSG:4326")

print(f"  Loaded {len(geo):,} pincode polygons")
print(f"  Contaminated: {geo['is_contaminated'].sum():,} ({100*geo['is_contaminated'].mean():.1f}%)")

# Split into clean vs contaminated
clean = geo[geo["is_contaminated"] == 0].copy()
dirty = geo[geo["is_contaminated"] == 1].copy()
print(f"  Clean: {len(clean):,} | Dirty: {len(dirty):,}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Simplify geometry for HTML (reduce file size while keeping shape)
# ─────────────────────────────────────────────────────────────────────────────
print("  Simplifying geometry for HTML maps (tolerance=0.005°) ...")
geo_simple = geo.copy()
geo_simple["geometry"] = geo_simple["geometry"].simplify(0.005, preserve_topology=True)
print("  Done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Color map & style settings
# ─────────────────────────────────────────────────────────────────────────────
CMAP = "RdYlGn"
CONT_COLOR = "#D3D3D3"   # light grey for contaminated
CONT_ALPHA = 0.50
CLEAN_ALPHA = 0.92
FIGSIZE_INDIA = (18, 20)
FIGSIZE_STATE = (14, 12)
DPI = 180

NORM = Normalize(vmin=0, vmax=100)

# State pincode ranges
REGIONS = {
    "india": {
        "title": "India — Pincode Affluence Heatmap",
        "subtitle": "(NTL × WorldPop composite | greyed pincodes = industrial/sparse data)",
        "mask": None,
        "figsize": FIGSIZE_INDIA,
        "bounds": None,
    },
    "maharashtra": {
        "title": "Maharashtra — Pincode Affluence Heatmap",
        "subtitle": "Pincodes 400000–449999",
        "mask": (400000, 449999),
        "figsize": FIGSIZE_STATE,
        "bounds": None,
    },
    "tamil_nadu": {
        "title": "Tamil Nadu — Pincode Affluence Heatmap",
        "subtitle": "Pincodes 600000–643999",
        "mask": (600000, 643999),
        "figsize": FIGSIZE_STATE,
        "bounds": None,
    },
    "delhi_up": {
        "title": "Delhi + Uttar Pradesh — Pincode Affluence Heatmap",
        "subtitle": "Delhi (110xxx) + Uttar Pradesh (200xxx–285xxx)",
        "mask": [(110000, 110999), (200000, 285999)],
        "figsize": FIGSIZE_STATE,
        "bounds": None,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: filter GDF by pincode mask
# ─────────────────────────────────────────────────────────────────────────────
def filter_region(gdf_full, mask):
    if mask is None:
        return gdf_full
    if isinstance(mask[0], tuple):
        # Multiple ranges (Delhi + UP)
        mask_bool = pd.Series([False] * len(gdf_full), index=gdf_full.index)
        for lo, hi in mask:
            mask_bool = mask_bool | gdf_full["pincode"].between(lo, hi)
        return gdf_full[mask_bool]
    else:
        lo, hi = mask
        return gdf_full[gdf_full["pincode"].between(lo, hi)]

# ─────────────────────────────────────────────────────────────────────────────
# PNG MAPS (Matplotlib + GeoPandas)
# ─────────────────────────────────────────────────────────────────────────────
print(f"{SEP}\n  Generating PNG maps\n{SEP}\n")

for region_key, cfg in REGIONS.items():
    print(f"  {region_key} PNG ...")

    # Filter geometry to region
    region_geo = filter_region(geo, cfg["mask"])
    region_clean = region_geo[region_geo["is_contaminated"] == 0]
    region_dirty = region_geo[region_geo["is_contaminated"] == 1]

    fig, ax = plt.subplots(1, 1, figsize=cfg["figsize"])
    ax.set_facecolor("#F0F4F8")
    fig.patch.set_facecolor("#F0F4F8")

    # Plot contaminated (grey) first so clean pincodes render on top
    if len(region_dirty) > 0:
        region_dirty.plot(
            ax=ax,
            color=CONT_COLOR,
            edgecolor="#AAAAAA",
            linewidth=0.15,
            alpha=CONT_ALPHA,
        )

    # Plot clean pincodes colored by affluence_percentile
    if len(region_clean) > 0:
        region_clean.plot(
            ax=ax,
            column="affluence_percentile",
            cmap=CMAP,
            norm=NORM,
            edgecolor="#555555",
            linewidth=0.15,
            alpha=CLEAN_ALPHA,
            legend=False,
        )

    # Outline — plot all pincodes with no fill for borders only
    region_geo.plot(
        ax=ax,
        facecolor="none",
        edgecolor="#666666",
        linewidth=0.1,
    )

    # Colorbar
    sm = ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, orientation="vertical")
    cbar.set_label("Affluence Percentile", fontsize=13, labelpad=10)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(["0\n(Low)", "25", "50\n(Median)", "75", "100\n(High)"])
    cbar.ax.tick_params(labelsize=10)

    # Legend patch for contaminated
    grey_patch = mpatches.Patch(
        facecolor=CONT_COLOR, edgecolor="#AAAAAA", alpha=CONT_ALPHA,
        label=f"Contaminated (industrial/sparse) — {len(region_dirty):,} pincodes"
    )
    ax.legend(
        handles=[grey_patch],
        loc="lower left",
        fontsize=9,
        framealpha=0.85,
        edgecolor="#cccccc",
    )

    # Title
    ax.set_title(cfg["title"], fontsize=18, fontweight="bold", pad=12)
    ax.text(
        0.5, 1.005, cfg["subtitle"],
        transform=ax.transAxes, ha="center", fontsize=10,
        color="#555555"
    )

    # Annotation: top 1% threshold
    ax.text(
        0.02, 0.02,
        f"Clean pincodes: {len(region_clean):,} | Contaminated: {len(region_dirty):,}",
        transform=ax.transAxes, fontsize=8, color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="#cccccc")
    )

    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    out_path = OUT / f"{region_key}_affluence.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    size_kb = out_path.stat().st_size / 1024
    print(f"    → {out_path.name}  ({size_kb:.0f} KB)")

# ─────────────────────────────────────────────────────────────────────────────
# HTML INTERACTIVE MAPS (Plotly choropleth_mapbox)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Generating interactive HTML maps\n{SEP}\n")

# Build a GeoJSON dict with pincode as feature id (using simplified geometry)
def make_geojson_with_id(gdf_subset):
    """Build a Plotly-compatible GeoJSON with pincode as feature id."""
    features = []
    for _, row in gdf_subset.iterrows():
        geom = row.geometry.__geo_interface__
        features.append({
            "type": "Feature",
            "id": str(int(row["pincode"])),
            "geometry": geom,
            "properties": {"pincode": int(row["pincode"])},
        })
    return {"type": "FeatureCollection", "features": features}

# Center coordinates for mapbox
CENTERS = {
    "india": {"lat": 22.0, "lon": 82.0, "zoom": 4.0},
    "maharashtra": {"lat": 19.2, "lon": 76.5, "zoom": 6.5},
    "tamil_nadu": {"lat": 10.8, "lon": 78.3, "zoom": 6.5},
    "delhi_up": {"lat": 27.0, "lon": 80.5, "zoom": 5.8},
}

for region_key, cfg in REGIONS.items():
    print(f"  {region_key} HTML ...")

    region_simple = filter_region(geo_simple, cfg["mask"])
    region_clean_s = region_simple[region_simple["is_contaminated"] == 0]
    region_dirty_s = region_simple[region_simple["is_contaminated"] == 1]

    center = CENTERS[region_key]

    # Build GeoJSON for clean pincodes
    geojson_clean = make_geojson_with_id(region_clean_s)

    # Clean choropleth trace
    trace_clean = go.Choroplethmapbox(
        geojson=geojson_clean,
        locations=region_clean_s["pincode"].astype(str).tolist(),
        z=region_clean_s["affluence_percentile"].tolist(),
        colorscale="RdYlGn",
        zmin=0,
        zmax=100,
        marker_opacity=0.85,
        marker_line_width=0.3,
        marker_line_color="#555555",
        colorbar=dict(
            title=dict(text="Affluence<br>Percentile", font=dict(size=13)),
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0<br>(Low)", "25", "50<br>(Median)", "75", "100<br>(High)"],
            thickness=18,
            len=0.75,
        ),
        hovertemplate=(
            "<b>Pincode: %{location}</b><br>"
            "Affluence: %{z:.1f}<br>"
            "<extra></extra>"
        ),
        name="Clean pincodes",
    )

    traces = [trace_clean]

    # Contaminated grey trace
    if len(region_dirty_s) > 0:
        geojson_dirty = make_geojson_with_id(region_dirty_s)
        trace_dirty = go.Choroplethmapbox(
            geojson=geojson_dirty,
            locations=region_dirty_s["pincode"].astype(str).tolist(),
            z=[0.5] * len(region_dirty_s),
            colorscale=[[0, "#CCCCCC"], [1, "#CCCCCC"]],
            zmin=0,
            zmax=1,
            showscale=False,
            marker_opacity=0.45,
            marker_line_width=0.2,
            marker_line_color="#AAAAAA",
            hovertemplate=(
                "<b>Pincode: %{location}</b><br>"
                "Contaminated (industrial/sparse)<br>"
                "<extra></extra>"
            ),
            name="Contaminated",
        )
        traces.insert(0, trace_dirty)  # render below clean

    fig = go.Figure(data=traces)
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=center["zoom"],
        mapbox_center={"lat": center["lat"], "lon": center["lon"]},
        title=dict(
            text=(
                f"<b>{cfg['title']}</b>"
                f"<br><sup>{cfg['subtitle']}</sup>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin={"r": 10, "t": 80, "l": 10, "b": 10},
        height=700,
    )

    # Annotation
    fig.add_annotation(
        text=(
            f"Clean: {len(region_clean_s):,} pincodes | "
            f"Contaminated (grey): {len(region_dirty_s):,} pincodes"
        ),
        xref="paper", yref="paper",
        x=0.5, y=-0.02,
        showarrow=False,
        font=dict(size=11, color="#555555"),
        align="center",
    )

    out_path = OUT / f"{region_key}_affluence.html"
    fig.write_html(out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"    → {out_path.name}  ({size_mb:.1f} MB)")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  Done! All maps saved to {OUT}/")
print(f"{SEP}\n")
files = sorted(OUT.iterdir())
for f in files:
    size = f.stat().st_size / 1024
    unit = "KB"
    if size > 1024:
        size /= 1024
        unit = "MB"
    print(f"  {f.name:<45}  {size:6.1f} {unit}")
print(f"\n{SEP}")
