# Pincode-Level Affluence Ranking System for India

A complete geospatial affluence ranking system combining **nighttime lights (NTL)** and **population density (WorldPop)** features to create multi-dimensional affluence scores for 19,312 Indian pincodes.

## Quick Start

```bash
# 1. Merge NTL + WorldPop features
python merge_features.py

# 2. Generate affluence ranking (default mode: balanced)
python affluence_engg.py --mode default

# 3. Validate and generate plots
python affluence_analysis.py
```

**Output:** `affluence_ranking.csv` with 19,312 pincodes × 26 columns (affluence percentile, 7 cross-signals, confidence scores, contamination flags)

## Key Features

### 7 Cross-Signal Features (NTL × WorldPop Interactions)

| Feature | Measures | Correlation |
|---------|----------|---|
| `ntl_per_capita` | Economic activity per person | +0.144 |
| `urban_intensity` | Lights + density combined | **+0.788** |
| `settlement_quality` | Organized vs chaotic layout | **+0.816** |
| `development_gap` | Regional peer standing | +0.651 |
| `economic_density` | NTL per km² | +0.370 |
| `slum_risk_score` | High density + low lights | **-0.740** |
| `density_light_ratio` | Population per unit light | -0.620 |

### 4 Weighting Modes

- **`--mode default`** (Balanced) — Recommended for general use
- **`--mode econ`** (Economic-heavy) — Emphasize NTL intensity, metro growth
- **`--mode urban`** (Urban-development) — Emphasize organized cities, anti-slum
- **`--mode relative`** (Regional-peer) — Within-state comparison

## Output: affluence_ranking.csv

**Main Columns:**
- `affluence_percentile` [0-100] — Rank vs clean pincodes
- `confidence` [0.2-1.0] — Data reliability (downweighted for industrial/sparse/uninhabited)
- 7 cross-signals (NTL-WorldPop interactions)
- 9 core features (NTL + WorldPop base metrics)
- 4 contamination flags (industrial, sparse_sample, uninhabited, is_contaminated)

## Validation Results ✓

**Reference Affluent Areas (Known High-Affluence Pincodes):**

| City | Pincodes | Avg Percentile | Status |
|------|----------|---|---|
| Delhi South | 110016, 110021, 110019, 110025 | 99.89 | ✓ Top 0.1% |
| Mumbai South | 400051, 400050, 400026 | 99.82 | ✓ Top 0.2% |
| Bangalore South | 560034, 560080, 560047 | 99.80 | ✓ Top 0.2% |
| Hyderabad | 500034, 500072 | 99.82 | ✓ Top 0.2% |

**Data Quality:**
- Clean pincodes: 9,786 (50.7%) — Full confidence
- Sparse sample: 7,995 (41.4%) — Confidence ×0.6
- Industrial: 1,553 (8.0%) — Confidence ×0.3
- Uninhabited: 4 (0.02%) — Confidence ×0.2

## Files

### Scripts
```
merge_features.py              Step 1: Merge NTL + WorldPop
affluence_engg.py              Step 2: Compute affluence scores (4 modes)
affluence_analysis.py          Step 3: Validate & generate plots
compare_modes.py               Optional: Compare mode rankings
```

### Data Outputs
```
merged_features.csv            19,312 pincodes × 34 cols (9.9 MB)
affluence_ranking.csv          19,312 pincodes × 26 cols (6.9 MB) [MAIN OUTPUT]
```

### Visualizations (Interactive HTML)
```
affluence_analysis_outputs/
├─ distribution_affluence.html       Histogram of percentiles (4.9 MB)
├─ scatter_ntl_vs_density.html       NTL vs population density (6.6 MB)
├─ scatter_affluence_vs_slum.html    Affluence vs slum risk (6.0 MB)
└─ reference_pincodes.html           Validation of known areas (4.7 MB)
```

### Documentation
```
AFFLUENCE_SYSTEM.md            Complete technical documentation (14 KB)
EXECUTION_GUIDE.txt            Quick start & usage examples (11 KB)
SYSTEM_SUMMARY.txt             Architecture overview (this file)
README.md                       This file
```

## Usage Examples

### Get top 100 affluent pincodes
```python
import pandas as pd
ranking = pd.read_csv("affluence_ranking.csv")
top100 = ranking.nlargest(100, "affluence_percentile")[
    ["pincode", "affluence_percentile", "confidence", "area_km2"]
]
print(top100)
```

### Filter by confidence
```python
high_conf = ranking[ranking["confidence"] >= 0.9]
print(f"High confidence pincodes: {len(high_conf)}")
```

### Compare modes
```bash
python affluence_engg.py --mode econ
python affluence_engg.py --mode urban
python affluence_engg.py --mode relative
python compare_modes.py
```

## Interpretation

### Affluence Percentile Tiers

- **99-100** — Ultra-affluent [Top 1%] → Prime residential, CBD
- **90-98** — Very affluent [Top 10%] → Affluent neighborhoods
- **75-89** — Affluent [Top 25%] → Upper-middle-class areas
- **50-74** — Mid-affluence [Middle] → Mixed development, small towns
- **25-49** — Lower affluence → Underdeveloped, early-stage
- **0-24** — Low affluence [Bottom 25%] → Slums, remote villages

## Architecture

```
ntl_features.csv + worldpop_features.csv
        ↓
    merge_features.py
        ↓
    merged_features.csv (19,312 × 34)
        ↓
    affluence_engg.py (4 modes)
        ├─ Compute 7 cross-signals
        ├─ Weight by mode
        ├─ Rank against clean pincodes
        └─ Assign confidence
        ↓
    affluence_ranking.csv (19,312 × 26) [MAIN OUTPUT]
        ↓
    affluence_analysis.py
        ├─ Validate reference pincodes
        ├─ Correlation analysis
        └─ Generate 4 interactive plots
```

## Key Insights

### Cross-Signal Correlations with Affluence

1. **settlement_quality** (+0.816) — Organized development is strongest driver
2. **urban_intensity** (+0.788) — Joint urbanization effect
3. **development_gap** (+0.651) — Regional prosperity matters
4. **slum_risk_score** (-0.740) — High density + low lights = anti-affluence

### Data Quality Distribution

- **50.7%** — High-confidence (complete data)
- **41.4%** — Sparse sample (small pincodes, <10 pixels)
- **8.0%** — Industrial contamination (NTL flares/refineries)
- **0.02%** — Uninhabited (no population)

**Recommendation:** Use confidence-weighting for strict affluence filtering; downweight sparse samples when using distribution features (pop_cv, pop_gini).

## Contamination Handling

| Flag | Count | Issue | Confidence | Action |
|------|-------|-------|---|---|
| `industrial_flag` | 1,553 (8%) | NTL from flares/refineries | ×0.3 | Exclude for residential affluence |
| `sparse_sample_flag` | 7,995 (41.4%) | <10 pixels in sample | ×0.6 | Use raw features only |
| `uninhabited_flag` | 4 (0.02%) | No population | ×0.2 | Exclude from analysis |

## Next Steps

1. **Immediate:** Review top 100 pincodes, validate against known areas
2. **Short-term:** Cross-validate with property prices, bank deposits, school quality
3. **Medium-term:** Incorporate temporal trends, fine-tune weights
4. **Long-term:** Build predictive models, integrate with external datasets

## Documentation

- **Technical Details:** See [AFFLUENCE_SYSTEM.md](AFFLUENCE_SYSTEM.md)
- **Quick Start:** See [EXECUTION_GUIDE.txt](EXECUTION_GUIDE.txt)
- **Architecture:** See [SYSTEM_SUMMARY.txt](SYSTEM_SUMMARY.txt)

## Questions?

Common issues and solutions are documented in [EXECUTION_GUIDE.txt](EXECUTION_GUIDE.txt) FAQ section.

---

**Version:** 1.0  
**Status:** Production Ready  
**Last Updated:** May 27, 2026
