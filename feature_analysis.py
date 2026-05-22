"""
feature_analysis.py
===================
Analyses ntl_features.csv and supports pincode-level queries.

USAGE
  Full analysis + charts:
    python feature_analysis.py

  Pincode query only (no charts):
    python feature_analysis.py --pincode 110001 --query-only

  Full analysis + pincode query + charts:
    python feature_analysis.py --pincode 110001

  Skip chart generation:
    python feature_analysis.py --no-charts

CHARTS SAVED  (to ntl_charts/)
  1. ntl_spearman_correlation.png   — Spearman corr heatmap (all features)
  2. ntl_covariance_matrix.png      — Covariance matrix heatmap
  3. ntl_feature_distributions.png  — Histograms for every feature, clean vs industrial
  4. ntl_industrial_comparison.png  — Box plots: flagged vs clean on key features
  5. ntl_p90_vs_lit_ratio.png       — Scatter: log_ntl_p90 vs lit_pixel_ratio
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for WSL / headless)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)
CHART_DIR = Path("ntl_charts")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SEP   = "=" * 65
TSEP  = "-" * 65
RANK_COL = "log_ntl_p90"   # primary ranking column

LOG_COLS   = ["log_ntl_sol", "log_ntl_median", "log_ntl_p75",
              "log_ntl_p90", "log_ntl_p95",    "log_ntl_std"]
RATIO_COLS = ["lit_pixel_ratio", "ntl_sol_per_pixel",
              "spike_ratio",     "p95_to_median_ratio", "cv_ntl"]
COUNT_COLS = ["lit_pixel_count", "total_pixel_count"]
FEAT_COLS  = LOG_COLS + RATIO_COLS + COUNT_COLS

FEATURE_DESC = {
    "log_ntl_sol":          "Total light emission (log-scaled)",
    "log_ntl_median":       "Typical brightness, outlier-resistant (log)",
    "log_ntl_p75":          "Upper-mid brightness distribution (log)",
    "log_ntl_p90":          "Primary affluence predictor (log) ← RANK KEY",
    "log_ntl_p95":          "Near-peak brightness (log)",
    "log_ntl_std":          "Within-pincode variability (log)",
    "lit_pixel_ratio":      "Fraction of pixels with radiance > 1.0  [0–1]",
    "ntl_sol_per_pixel":    "Area-normalised SOL",
    "spike_ratio":          "Peak-to-p90 ratio — flare/industrial signal",
    "p95_to_median_ratio":  "Skewness proxy — high = industrial",
    "cv_ntl":               "Coefficient of variation",
    "lit_pixel_count":      "Pixels with radiance > 1.0",
    "total_pixel_count":    "All pixels assigned to pincode (>= 0.4 DN)",
    "industrial_flag":      "Binary: 1 = likely industrial contamination",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 30, char: str = "█") -> str:
    """ASCII bar for a normalised [0,1] value."""
    filled = int(round(value * width))
    return char * filled + "·" * (width - filled)


def _pct_rank(series: pd.Series, value: float) -> float:
    """Percentile rank of value in series (0 = bottom, 100 = top)."""
    return float((series < value).sum() / len(series) * 100)


def _quintile_label(pct: float) -> str:
    if pct >= 80: return "Top 20 %   ★★★★★"
    if pct >= 60: return "Top 40 %   ★★★★"
    if pct >= 40: return "Middle     ★★★"
    if pct >= 20: return "Bottom 40 % ★★"
    return             "Bottom 20 % ★"


def load(path: str = "ntl_features.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"ERROR: {path} not found. Run feature_engg.py first.")
    df = df.sort_values(RANK_COL, ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1          # rank 1 = highest log_ntl_p90
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Section printers
# ─────────────────────────────────────────────────────────────────────────────

def print_overview(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  OVERVIEW")
    print(SEP)
    print(f"  Pincodes        : {len(df):,}")
    print(f"  Features        : {len(FEAT_COLS)}  (+  industrial_flag binary)")
    print(f"  Rank column     : {RANK_COL}")
    missing = df[FEAT_COLS + ["industrial_flag"]].isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print(f"  Missing values  : none")
    else:
        print(f"  Missing values  :")
        for col, n in missing.items():
            print(f"    {col:<28} {n:,}")
    flagged = df["industrial_flag"].sum()
    print(f"  Industrial flag : {flagged:,} pincodes  ({100*flagged/len(df):.1f}%)")


def print_descriptive_stats(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  DESCRIPTIVE STATISTICS  (all values normalised [0,1])")
    print(SEP)
    stats = df[FEAT_COLS].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]).T
    stats = stats[["mean", "std", "min", "25%", "50%", "75%", "90%", "99%", "max"]]
    stats.columns = ["mean", "std", "min", "p25", "p50", "p75", "p90", "p99", "max"]
    print(stats.round(4).to_string())


def print_correlation_with_rank(df: pd.DataFrame):
    print(f"\n{SEP}")
    print(f"  CORRELATION WITH {RANK_COL.upper()}  (Spearman)")
    print(SEP)
    corrs = (
        df[FEAT_COLS]
        .corrwith(df[RANK_COL], method="spearman")
        .drop(RANK_COL)
        .sort_values(key=abs, ascending=False)
    )
    for feat, r in corrs.items():
        direction = "+" if r >= 0 else "-"
        bar_val   = abs(r)
        print(f"  {feat:<28}  {r:+.3f}  {direction} {_bar(bar_val, 20)}")


def print_feature_correlations(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  TOP 10 INTER-FEATURE CORRELATIONS  (Pearson, absolute)")
    print(SEP)
    corr_mat = df[FEAT_COLS].corr().abs()
    # Get upper triangle pairs only
    pairs = (
        corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
        .head(10)
    )
    for (a, b), r in pairs.items():
        print(f"  {a:<28}  ↔  {b:<28}  r={r:.3f}")


def print_industrial_breakdown(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("  INDUSTRIAL FLAG BREAKDOWN")
    print(SEP)
    clean = df[df["industrial_flag"] == 0]
    flagd = df[df["industrial_flag"] == 1]
    cols  = ["log_ntl_p90", "lit_pixel_ratio", "spike_ratio",
             "p95_to_median_ratio", "log_ntl_median"]
    col_w = 18
    print(f"  {'Feature':<28}  {'Clean mean':>{col_w}}  {'Flagged mean':>{col_w}}  Δ")
    print(f"  {TSEP}")
    for c in cols:
        cm = clean[c].mean()
        fm = flagd[c].mean()
        diff = fm - cm
        sign = "+" if diff >= 0 else ""
        print(f"  {c:<28}  {cm:>{col_w}.4f}  {fm:>{col_w}.4f}  {sign}{diff:.4f}")

    print(f"\n  Rank distribution of flagged pincodes:")
    pct_in_top10 = (flagd["rank"] <= int(len(df) * 0.1)).sum()
    pct_in_bot50 = (flagd["rank"] > int(len(df) * 0.5)).sum()
    print(f"    In top 10% by {RANK_COL}  : {pct_in_top10:,}")
    print(f"    In bottom 50% by {RANK_COL}: {pct_in_bot50:,}")


def print_top_bottom(df: pd.DataFrame, n: int = 20):
    display_cols = ["rank", "pincode", "log_ntl_p90",
                    "lit_pixel_ratio", "spike_ratio", "industrial_flag"]

    print(f"\n{SEP}")
    print(f"  TOP {n} PINCODES  by {RANK_COL}")
    print(SEP)
    print(df.head(n)[display_cols].to_string(index=False))

    print(f"\n{SEP}")
    print(f"  BOTTOM {n} PINCODES  by {RANK_COL}")
    print(SEP)
    print(df.tail(n).sort_values("rank", ascending=False)[display_cols].to_string(index=False))


def print_distribution_buckets(df: pd.DataFrame):
    print(f"\n{SEP}")
    print(f"  DISTRIBUTION BUCKETS  —  {RANK_COL}")
    print(SEP)
    n   = len(df)
    buckets = [
        ("Top 1 %   (rank 1–{})".format(max(1, n//100)),       df["rank"] <= n // 100),
        ("Top 5 %   (rank 1–{})".format(n // 20),              df["rank"] <= n // 20),
        ("Top 10 %  (rank 1–{})".format(n // 10),              df["rank"] <= n // 10),
        ("Top 25 %  (rank 1–{})".format(n // 4),               df["rank"] <= n // 4),
        ("Bottom 25 % (rank {}+)".format(3 * n // 4),          df["rank"] > 3 * n // 4),
        ("Bottom 10 % (rank {}+)".format(9 * n // 10),         df["rank"] > 9 * n // 10),
    ]
    for label, mask in buckets:
        subset  = df[mask]
        med_p90 = subset["log_ntl_p90"].median()
        med_lpr = subset["lit_pixel_ratio"].median()
        n_ind   = subset["industrial_flag"].sum()
        print(f"  {label:<40}  n={len(subset):>5,}  "
              f"median_p90={med_p90:.4f}  "
              f"lit_ratio={med_lpr:.3f}  "
              f"industrial={n_ind}")


# ─────────────────────────────────────────────────────────────────────────────
# Pincode query
# ─────────────────────────────────────────────────────────────────────────────

def query_pincode(df: pd.DataFrame, pincode: int):
    print(f"\n{SEP}")
    print(f"  PINCODE QUERY  →  {pincode}")
    print(SEP)

    row = df[df["pincode"] == pincode]
    if row.empty:
        print(f"  Pincode {pincode} not found in ntl_features.csv.")
        print(f"  This could mean it had no lit pixels after cleaning, or the")
        print(f"  pincode doesn't exist in the boundary file.")
        return

    row = row.iloc[0]
    n   = len(df)

    # ── Rank + percentile ────────────────────────────────────────────────────
    rank      = int(row["rank"])
    pct_rank  = _pct_rank(df[RANK_COL], row[RANK_COL])
    quintile  = _quintile_label(pct_rank)
    ind_flag  = int(row["industrial_flag"])

    print(f"  Rank (by {RANK_COL})  : {rank:,} / {n:,}")
    print(f"  Percentile rank         : {pct_rank:.1f}th  ({quintile})")
    print(f"  Industrial flag         : {'YES — treat NTL signals with caution' if ind_flag else 'NO'}")

    # ── Feature breakdown ────────────────────────────────────────────────────
    print(f"\n  {'Feature':<28}  {'Value':>8}  {'Natl avg':>8}  {'Natl pct':>9}  Bar (national)")
    print(f"  {TSEP}")

    national_means = df[FEAT_COLS].mean()
    for col in FEAT_COLS:
        val      = row[col]
        avg      = national_means[col]
        pct      = _pct_rank(df[col], val)
        bar      = _bar(val, 25)
        indicator = "▲" if val > avg else "▼"
        print(f"  {col:<28}  {val:>8.4f}  {avg:>8.4f}  {pct:>8.1f}%  {indicator} {bar}")

    # ── Similar pincodes by log_ntl_p90 ─────────────────────────────────────
    print(f"\n  Nearest pincodes by rank  (±5 positions around rank {rank})")
    print(f"  {TSEP}")
    nearby = df[
        (df["rank"] >= max(1, rank - 5)) &
        (df["rank"] <= rank + 5) &
        (df["pincode"] != pincode)
    ][["rank", "pincode", "log_ntl_p90", "lit_pixel_ratio", "industrial_flag"]]
    print(nearby.to_string(index=False))

    # ── State-level context (prefix of pincode) ──────────────────────────────
    prefix = str(pincode)[:3]
    state_peers = df[df["pincode"].astype(str).str.startswith(prefix)]
    if len(state_peers) > 1:
        state_rank = int((state_peers[RANK_COL] >= row[RANK_COL]).sum())
        print(f"\n  Within '{prefix}xx' prefix group  ({len(state_peers):,} pincodes):")
        print(f"    State-level rank  : {state_rank} / {len(state_peers)}")
        print(f"    State median p90  : {state_peers[RANK_COL].median():.4f}  "
              f"(national: {df[RANK_COL].median():.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str):
    CHART_DIR.mkdir(exist_ok=True)
    path = CHART_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def chart_spearman(df: pd.DataFrame):
    """Spearman correlation matrix for all 14 features."""
    corr = df[FEAT_COLS].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True     # show lower triangle only

    sns.heatmap(
        corr,
        mask=mask,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        linewidths=0.4, linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Spearman ρ"},
        ax=ax,
    )
    ax.set_title("Spearman Correlation Matrix — NTL Features", fontsize=13, pad=14)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    _save(fig, "ntl_spearman_correlation.png")


def chart_covariance(df: pd.DataFrame):
    """Covariance matrix for all 14 normalised features."""
    cov = df[FEAT_COLS].cov()

    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.zeros_like(cov, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(
        cov,
        mask=mask,
        annot=True, fmt=".4f", annot_kws={"size": 6.5},
        cmap="RdBu_r", center=0,
        linewidths=0.4, linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Covariance"},
        ax=ax,
    )
    ax.set_title("Covariance Matrix — NTL Features (normalised [0,1])", fontsize=13, pad=14)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    _save(fig, "ntl_covariance_matrix.png")


def chart_distributions(df: pd.DataFrame):
    """
    Histograms for every feature (14 panels), overlaid by industrial flag.
    Clean pincodes = blue, industrial flagged = red.
    """
    clean = df[df["industrial_flag"] == 0]
    flagd = df[df["industrial_flag"] == 1]

    n_cols = 4
    n_rows = -(-len(FEAT_COLS) // n_cols)   # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 4.2, n_rows * 3.2))
    axes = axes.flatten()

    for i, col in enumerate(FEAT_COLS):
        ax = axes[i]
        ax.hist(clean[col].dropna(), bins=40, color="#4C72B0",
                alpha=0.65, label="Clean", density=True, edgecolor="none")
        ax.hist(flagd[col].dropna(), bins=40, color="#C44E52",
                alpha=0.65, label="Industrial", density=True, edgecolor="none")
        ax.set_title(col, fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Normalised value", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        if i == 0:
            ax.legend(fontsize=7)

    # Hide any unused panels
    for j in range(len(FEAT_COLS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Feature Distributions — Clean vs Industrial-Flagged Pincodes",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "ntl_feature_distributions.png")


def chart_industrial_boxplots(df: pd.DataFrame):
    """
    Side-by-side box plots for 6 key features, grouped by industrial_flag.
    Shows exactly how much the flag separates the two populations.
    """
    key_feats = ["log_ntl_p90", "lit_pixel_ratio", "spike_ratio",
                 "p95_to_median_ratio", "cv_ntl", "ntl_sol_per_pixel"]

    plot_df = df[key_feats + ["industrial_flag"]].melt(
        id_vars="industrial_flag",
        var_name="feature",
        value_name="value",
    )
    plot_df["Group"] = plot_df["industrial_flag"].map({0: "Clean", 1: "Industrial"})

    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.boxplot(
        data=plot_df,
        x="feature", y="value", hue="Group",
        palette={"Clean": "#4C72B0", "Industrial": "#C44E52"},
        width=0.55, fliersize=1.8, linewidth=0.9,
        ax=ax,
    )
    ax.set_title("Key Feature Distributions — Clean vs Industrial-Flagged Pincodes",
                 fontsize=12, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Normalised value [0–1]", fontsize=9)
    ax.tick_params(axis="x", labelsize=8.5, rotation=18)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(title="Group", fontsize=9, title_fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    _save(fig, "ntl_industrial_comparison.png")


def chart_p90_scatter(df: pd.DataFrame):
    """
    Scatter plot: log_ntl_p90 (y) vs lit_pixel_ratio (x).
    Coloured by industrial_flag. Diagonal = the decision boundary intuition.
    """
    clean = df[df["industrial_flag"] == 0].sample(min(5_000, len(df)), random_state=42)
    flagd = df[df["industrial_flag"] == 1]

    fig, ax = plt.subplots(figsize=(9, 6.5))

    ax.scatter(
        clean["lit_pixel_ratio"], clean["log_ntl_p90"],
        s=6, alpha=0.35, color="#4C72B0", label=f"Clean ({len(df[df['industrial_flag']==0]):,})",
        rasterized=True,
    )
    ax.scatter(
        flagd["lit_pixel_ratio"], flagd["log_ntl_p90"],
        s=9, alpha=0.55, color="#C44E52", label=f"Industrial ({len(flagd):,})",
        rasterized=True,
    )

    # Reference lines for industrial_flag thresholds (spike_ratio > 5 captured in p90)
    ax.axvline(0.15, color="#C44E52", lw=0.8, ls="--", alpha=0.6,
               label="lit_ratio < 0.15 threshold")

    ax.set_xlabel("lit_pixel_ratio  (spatial coverage)", fontsize=10)
    ax.set_ylabel("log_ntl_p90  (rank key, normalised)", fontsize=10)
    ax.set_title("NTL Brightness vs Spatial Coverage — by Industrial Flag", fontsize=12, pad=12)
    ax.legend(fontsize=9, markerscale=2.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    _save(fig, "ntl_p90_vs_lit_ratio.png")


def save_charts(df: pd.DataFrame):
    print(f"\n{SEP}")
    print(f"  GENERATING CHARTS  →  {CHART_DIR}/")
    print(SEP)
    chart_spearman(df)
    chart_covariance(df)
    chart_distributions(df)
    chart_industrial_boxplots(df)
    chart_p90_scatter(df)
    print(f"  All 5 charts saved to {CHART_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="NTL feature analysis + pincode query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python feature_analysis.py\n"
            "  python feature_analysis.py --pincode 110001\n"
            "  python feature_analysis.py --pincode 110001 --query-only\n"
            "  python feature_analysis.py --top 50\n"
        ),
    )
    ap.add_argument("--pincode",    type=int, default=None,
                    help="Query a specific pincode (shows rank + all features)")
    ap.add_argument("--query-only", action="store_true",
                    help="Skip the full analysis, only run the pincode query")
    ap.add_argument("--top",        type=int, default=20,
                    help="Number of top/bottom pincodes to display (default 20)")
    ap.add_argument("--file",       default="ntl_features.csv",
                    help="Path to the features CSV (default: ntl_features.csv)")
    ap.add_argument("--no-charts",  action="store_true",
                    help="Skip chart generation")
    args = ap.parse_args()

    df = load(args.file)

    if not args.query_only:
        print_overview(df)
        print_descriptive_stats(df)
        print_correlation_with_rank(df)
        print_feature_correlations(df)
        print_industrial_breakdown(df)
        print_distribution_buckets(df)
        print_top_bottom(df, n=args.top)

    if args.pincode is not None:
        query_pincode(df, args.pincode)
    elif args.query_only:
        ap.error("--query-only requires --pincode")

    if not args.no_charts and not args.query_only:
        save_charts(df)

    print()


if __name__ == "__main__":
    main()
