"""
affluence_analysis.py
=====================
Validate affluence ranking against reference pincodes and generate analysis plots.

Input: affluence_ranking.csv
Output:
  - Console report (reference pincode checks, correlation analysis)
  - Plots in affluence_analysis_outputs/:
    * distribution_affluence.html — histogram + percentiles
    * scatter_ntl_vs_density.html — interactive scatter plot
    * scatter_affluence_vs_slum.html — affluence vs slum risk
    * reference_pincodes.html — validation of known affluent/poor areas
"""

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

SEP = "=" * 70

print(f"{SEP}")
print("  AFFLUENCE ANALYSIS & VALIDATION")
print(f"{SEP}\n")

# Load affluence ranking
print("  Loading affluence_ranking.csv ...")
ranking = pd.read_csv("affluence_ranking.csv")
print(f"    → {len(ranking):,} pincodes\n")

# ─────────────────────────────────────────────────────────────────────────────
# Reference pincodes (known affluent/poor areas)
# ─────────────────────────────────────────────────────────────────────────────
print(f"{SEP}")
print("  Reference Pincodes Validation")
print(f"{SEP}\n")

references = {
    "Delhi South (affluent)": [110016, 110021, 110019, 110025],
    "Mumbai South (affluent)": [400051, 400050, 400026],
    "Bangalore South (affluent)": [560034, 560080, 560047],
    "Hyderabad (affluent)": [500034, 500072],
    "Delhi Fringe (lower)": [110009, 110006, 110095],
    "Mumbai Suburbs (mixed)": [400708, 400614],
}

print("  Expected affluent areas:\n")
for region, pincodes in references.items():
    if "affluent" in region or "Affluent" in region:
        print(f"  {region}:")
        hits = []
        for pc in pincodes:
            if pc in ranking["pincode"].values:
                row = ranking[ranking["pincode"] == pc].iloc[0]
                hits.append(pc)
                rank_pct = row["affluence_percentile"]
                confidence = row["confidence"]
                contaminated = "⚠️" if row["is_contaminated"] else "✓"
                print(
                    f"    {pc:6.0f}  percentile={rank_pct:6.2f}  "
                    f"conf={confidence:.1f}  {contaminated}"
                )
            else:
                print(f"    {pc:6.0f}  NOT FOUND in ranking")
        if hits:
            avg_percentile = ranking[ranking["pincode"].isin(hits)][
                "affluence_percentile"
            ].mean()
            print(f"    Average percentile: {avg_percentile:.2f}\n")

print("  Expected lower-affluence areas:\n")
for region, pincodes in references.items():
    if "lower" in region or "mixed" in region.lower():
        print(f"  {region}:")
        for pc in pincodes:
            if pc in ranking["pincode"].values:
                row = ranking[ranking["pincode"] == pc].iloc[0]
                rank_pct = row["affluence_percentile"]
                confidence = row["confidence"]
                contaminated = "⚠️" if row["is_contaminated"] else "✓"
                print(
                    f"    {pc:6.0f}  percentile={rank_pct:6.2f}  "
                    f"conf={confidence:.1f}  {contaminated}"
                )
            else:
                print(f"    {pc:6.0f}  NOT FOUND in ranking")

# ─────────────────────────────────────────────────────────────────────────────
# Correlation analysis
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  Correlation Analysis")
print(f"{SEP}\n")

cross_signals = [
    "ntl_per_capita",
    "urban_intensity",
    "slum_risk_score",
    "settlement_quality",
    "development_gap",
    "economic_density",
    "density_light_ratio",
]

print("  Cross-signal correlations with affluence_percentile:\n")
correlations = {}
for sig in cross_signals:
    corr = ranking["affluence_percentile"].corr(ranking[sig])
    correlations[sig] = corr
    direction = "↑" if corr > 0 else "↓"
    print(f"    {direction}  {corr:+7.3f}  {sig}")

print(f"\n  NTL-WorldPop correlation:\n")
corr_ntl_pop = ranking["log_ntl_p90"].corr(ranking["log_pop_density"])
print(f"    {corr_ntl_pop:+.3f}  (lights vs population density)")

print(f"\n  Slum risk vs affluence:\n")
corr_slum = ranking["slum_risk_score"].corr(ranking["affluence_percentile"])
print(f"    {corr_slum:+.3f}  (should be negative)")

# ─────────────────────────────────────────────────────────────────────────────
# Distribution statistics
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  Affluence Distribution Statistics")
print(f"{SEP}\n")

print(f"  affluence_percentile:\n")
print(f"    mean        : {ranking['affluence_percentile'].mean():.2f}")
print(f"    median      : {ranking['affluence_percentile'].median():.2f}")
print(f"    std         : {ranking['affluence_percentile'].std():.2f}")
print(f"    min         : {ranking['affluence_percentile'].min():.2f}")
print(f"    max         : {ranking['affluence_percentile'].max():.2f}")
print(f"    p10         : {ranking['affluence_percentile'].quantile(0.10):.2f}")
print(f"    p25         : {ranking['affluence_percentile'].quantile(0.25):.2f}")
print(f"    p75         : {ranking['affluence_percentile'].quantile(0.75):.2f}")
print(f"    p90         : {ranking['affluence_percentile'].quantile(0.90):.2f}")

print(f"\n  Contamination in top 10%:\n")
top10 = ranking[ranking["affluence_percentile"] >= ranking["affluence_percentile"].quantile(0.90)]
n_contaminated_top10 = top10["is_contaminated"].sum()
print(f"    {n_contaminated_top10} contaminated out of {len(top10)} ({100*n_contaminated_top10/len(top10):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Create output directory for plots
# ─────────────────────────────────────────────────────────────────────────────
out_dir = Path("affluence_analysis_outputs")
out_dir.mkdir(exist_ok=True)

print(f"\n{SEP}")
print(f"  Generating plots in {out_dir}/")
print(f"{SEP}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Affluence distribution
# ─────────────────────────────────────────────────────────────────────────────
print("  1. distribution_affluence.html ...")
fig = go.Figure()

fig.add_trace(
    go.Histogram(
        x=ranking["affluence_percentile"],
        nbinsx=100,
        name="All pincodes",
        marker_color="rgba(0, 100, 200, 0.7)",
    )
)

# Mark percentile thresholds
for pct in [25, 50, 75, 90, 99]:
    val = ranking["affluence_percentile"].quantile(pct / 100)
    fig.add_vline(
        x=val,
        line_dash="dash",
        line_color="red" if pct >= 90 else "orange" if pct >= 75 else "gray",
        annotation_text=f"p{pct} = {val:.1f}",
        annotation_position="top",
    )

fig.update_layout(
    title="Affluence Percentile Distribution Across All Pincodes",
    xaxis_title="Affluence Percentile [0-100]",
    yaxis_title="Count of Pincodes",
    height=500,
    hovermode="x",
)
fig.write_html(out_dir / "distribution_affluence.html")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: NTL vs Population Density (scatter)
# ─────────────────────────────────────────────────────────────────────────────
print("  2. scatter_ntl_vs_density.html ...")
fig = px.scatter(
    ranking,
    x="log_pop_density",
    y="log_ntl_p90",
    color="affluence_percentile",
    size="area_km2",
    hover_name="pincode",
    hover_data={
        "affluence_percentile": ":.1f",
        "confidence": ":.2f",
        "industrial_flag": True,
        "sparse_sample_flag": True,
    },
    color_continuous_scale="Viridis",
    title="NTL Intensity vs Population Density (colored by affluence)",
    labels={
        "log_pop_density": "Log Population Density",
        "log_ntl_p90": "Log NTL (90th percentile)",
    },
    height=600,
)
fig.update_layout(hovermode="closest")
fig.write_html(out_dir / "scatter_ntl_vs_density.html")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Affluence vs Slum Risk
# ─────────────────────────────────────────────────────────────────────────────
print("  3. scatter_affluence_vs_slum.html ...")
fig = px.scatter(
    ranking,
    x="slum_risk_score",
    y="affluence_percentile",
    color="log_pop_density",
    size="area_km2",
    hover_name="pincode",
    hover_data={
        "affluence_percentile": ":.1f",
        "slum_risk_score": ":.3f",
        "confidence": ":.2f",
    },
    color_continuous_scale="RdYlGn_r",
    title="Affluence vs Slum Risk (colored by population density)",
    labels={
        "slum_risk_score": "Slum Risk Score (density × anti-lights)",
        "affluence_percentile": "Affluence Percentile",
    },
    height=600,
)
fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Median affluence")
fig.add_vline(x=ranking["slum_risk_score"].median(), line_dash="dash", line_color="gray")
fig.update_layout(hovermode="closest")
fig.write_html(out_dir / "scatter_affluence_vs_slum.html")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Reference pincodes
# ─────────────────────────────────────────────────────────────────────────────
print("  4. reference_pincodes.html ...")

ref_data = []
for region, pincodes in references.items():
    for pc in pincodes:
        if pc in ranking["pincode"].values:
            row = ranking[ranking["pincode"] == pc].iloc[0]
            ref_data.append({
                "pincode": pc,
                "region": region,
                "affluence_percentile": row["affluence_percentile"],
                "confidence": row["confidence"],
                "is_contaminated": row["is_contaminated"],
            })

ref_df = pd.DataFrame(ref_data)

fig = px.bar(
    ref_df,
    x="pincode",
    y="affluence_percentile",
    color="region",
    hover_data={"confidence": ":.2f", "is_contaminated": True},
    title="Reference Pincodes: Affluence Percentiles",
    labels={"affluence_percentile": "Affluence Percentile"},
    height=500,
)
fig.update_xaxes(tickangle=-45)
fig.update_layout(hovermode="x")
fig.write_html(out_dir / "reference_pincodes.html")

print(f"\n{SEP}")
print(f"  Analysis complete!")
print(f"  Plots saved to {out_dir}/")
print(f"{SEP}\n")
