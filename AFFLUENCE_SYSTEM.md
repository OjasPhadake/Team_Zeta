# Pincode-Level Affluence Ranking System

## Overview

This system combines **NTL (nighttime lights)** and **WorldPop (population density)** features to create a multi-dimensional affluence ranking for Indian pincodes. The key insight is that affluence is not just about lights or density separately, but about their **cross-signals**:

- **High NTL + High density** → Affluent urban core
- **High NTL + Low density** → Industrial or wealthy low-density areas
- **Low NTL + High density** → Slum/informal settlements
- **Low NTL + Low density** → Rural/underdeveloped areas

---

## Architecture

### Pipeline Overview

```
merged_features.csv (NTL + WorldPop combined)
        ↓
    affluence_engg.py (--mode flag)
        ├─ Compute 7 cross-signal features
        ├─ Compute affluence score (weighted)
        ├─ Rank against clean pincodes
        └─ Assign confidence weights
        ↓
    affluence_ranking.csv (final output)
        ↓
    affluence_analysis.py
        ├─ Validate reference pincodes
        ├─ Correlation analysis
        └─ Generate 4 interactive plots
```

### Files

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `merge_features.py` | Combine NTL + WorldPop | ntl_features.csv, worldpop_features.csv | merged_features.csv |
| `affluence_engg.py` | Compute affluence score | merged_features.csv | affluence_ranking.csv |
| `affluence_analysis.py` | Validate & visualize | affluence_ranking.csv | affluence_analysis_outputs/ |
| `compare_modes.py` | Compare weighting modes | affluence_ranking.csv | Console report |

---

## Key Features

### Cross-Signal Features (7 total)

Generated from NTL × WorldPop interactions:

1. **`ntl_per_capita`** = NTL intensity / population
   - Measures economic activity per person
   - High = efficient, developed economy

2. **`urban_intensity`** = (NTL + Density) / 2
   - Joint urbanization metric
   - High = metro city

3. **`slum_risk_score`** = Density × (1 - NTL)
   - Flags high-density, low-light areas
   - High = potential slum/informal settlement

4. **`settlement_quality`** = NTL / (population variation + 0.5)
   - Organized vs chaotic settlement
   - High = organized, planned city

5. **`development_gap`** = NTL - regional peer average
   - Relative regional prosperity
   - High = ahead of local peers

6. **`economic_density`** = NTL intensity / area
   - Economic activity per km²
   - High = compact, developed

7. **`density_light_ratio`** = Density / NTL
   - Population per unit light
   - High = slum; Low = sparse/industrial

### Affluence Scoring Modes

Each mode reweights the 7 cross-signals with different priorities:

#### **`--mode default`** (Balanced)
```
affluence = 
  0.25 × ntl_per_capita              (economic efficiency)
+ 0.20 × urban_intensity             (urbanization)
+ 0.15 × settlement_quality          (organization)
+ 0.15 × development_gap             (regional standing)
+ 0.10 × economic_density            (footprint)
- 0.15 × slum_risk_score             (contamination penalty)
```
**Use for:** General affluence ranking

#### **`--mode econ`** (Economic-Heavy)
```
Emphasizes: ntl_per_capita (0.35), economic_density (0.15)
Deemphasizes: slum_risk_score penalty (-0.10)
```
**Use for:** Economic activity, industrial hubs, metros

#### **`--mode urban`** (Urban-Development-Heavy)
```
Emphasizes: urban_intensity (0.30), settlement_quality (0.20), 
            slum_risk_score penalty (-0.20)
Deemphasizes: ntl_per_capita (0.15)
```
**Use for:** Organized city development, anti-slum metrics

#### **`--mode relative`** (Regional-Peer-Heavy)
```
Emphasizes: development_gap (0.30) — uses state-level percentiles
```
**Use for:** Within-state inequality, regional competition

---

## Output: affluence_ranking.csv

**Columns (26 total):**

| Group | Columns |
|-------|---------|
| **ID & Metadata** | pincode, area_km2 |
| **NTL Core** | log_ntl_p90 |
| **WorldPop Core** | log_pop_sum, log_pop_density, pop_concentration, pop_cv, pop_gini, high_density_ratio, pop_density_state_pct, pop_sum_state_pct |
| **Cross-Signals** | ntl_per_capita, urban_intensity, slum_risk_score, settlement_quality, development_gap, economic_density, density_light_ratio |
| **Affluence Scores** | affluence_raw (raw composite), affluence_percentile (0-100 ranked), affluence_percentile_weighted (confidence-adjusted) |
| **Confidence** | confidence (1.0 = full, 0.6 = sparse sample, 0.3 = industrial, 0.2 = uninhabited) |
| **Flags** | industrial_flag, sparse_sample_flag, uninhabited_flag, is_contaminated |

**Key Columns:**

- **`affluence_percentile`** [0-100]: Rank against clean (non-contaminated) pincodes
  - 99+ = Top 1% affluent
  - 90-99 = Top 10% affluent
  - 50 = Median
  - <10 = Lower affluence

- **`confidence`** [0.2-1.0]: Data reliability weight
  - 1.0 = Full confidence (complete NTL + WorldPop data)
  - 0.6 = Sparse sample (WorldPop sample < 10 pixels)
  - 0.3 = Industrial contamination (NTL spikes from flares/refineries)
  - 0.2 = Uninhabited (forests, deserts, no population)

- **`is_contaminated`** [0/1]: Binary flag if any contamination flag is set

---

## Usage

### Step 1: Generate Ranking (Default Mode)
```bash
python affluence_engg.py --mode default
```
**Output:** `affluence_ranking.csv`

### Step 2: Run Validation & Analysis
```bash
python affluence_analysis.py
```
**Outputs:**
- Console report: Reference pincode validation, correlation analysis
- `affluence_analysis_outputs/`:
  - `distribution_affluence.html` — Histogram of affluence percentiles
  - `scatter_ntl_vs_density.html` — NTL vs population density (colored by affluence)
  - `scatter_affluence_vs_slum.html` — Affluence vs slum risk
  - `reference_pincodes.html` — Validation of known affluent areas

### Step 3: Compare Modes (Optional)
```bash
python affluence_engg.py --mode econ
python affluence_engg.py --mode urban
python affluence_engg.py --mode relative
python compare_modes.py
```

---

## Validation Results

### Reference Pincodes (Known Affluent Areas)

**Delhi South** (Pincodes 110016, 110021, 110019, 110025):
- Average percentile: **99.89**
- Validation: ✓ Correctly ranked in top 0.1%

**Mumbai South** (Pincodes 400051, 400050, 400026):
- Average percentile: **99.82**
- Validation: ✓ Correctly ranked in top 0.2%

**Bangalore South** (Pincodes 560034, 560080, 560047):
- Average percentile: **99.80**
- Validation: ✓ Correctly ranked in top 0.2%

**Hyderabad** (Pincodes 500034, 500072):
- Average percentile: **99.82**
- Validation: ✓ Correctly ranked in top 0.2%

### Key Metrics

- **Cross-signal correlations with affluence:**
  - settlement_quality: +0.816 (strongest positive)
  - urban_intensity: +0.788
  - development_gap: +0.651
  - slum_risk_score: -0.740 (anti-correlated as expected)

- **Data quality:**
  - Clean pincodes: 9,786 (50.7%)
  - Contaminated: 9,526 (49.3%)
    - Sparse sample: 7,995 (41.4%) — small pincodes with unreliable features
    - Industrial flag: 1,553 (8.0%) — NTL refineries/flares
    - Uninhabited: 4 (0.02%) — no population

- **Distribution:**
  - Top 1% threshold: 99.96th percentile
  - Top 10% threshold: 99.34th percentile
  - Median: 58.94th percentile

---

## Contamination Handling

### Industrial Flag (`industrial_flag == 1`)
- **What:** NTL spikes from gas flares, oil refineries
- **Count:** 1,553 pincodes (8%)
- **Confidence penalty:** ×0.3
- **Action:** Downweight in top affluence tier; may indicate industrial wealth, not residential affluence

### Sparse Sample Flag (`sparse_sample_flag == 1`)
- **What:** < 10 pixels in WorldPop sample → pixel-level features unreliable
- **Count:** 7,995 pincodes (41.4%)
- **Confidence penalty:** ×0.6
- **Action:** Use with caution; prefer log_pop_density and log_pop_sum over distribution features

### Uninhabited Flag (`uninhabited_flag == 1`)
- **What:** Pop_sum < 1.0 person (forests, deserts, coastal areas)
- **Count:** 4 pincodes (0.02%)
- **Confidence penalty:** ×0.2
- **Action:** Exclude from affluence ranking; these aren't human settlements

### Confidence-Weighted Percentile
- `affluence_percentile_weighted` = `affluence_percentile` × `confidence`
- Use this for strict filtering (only keep confidence ≥ 0.9 for highest confidence)

---

## Interpretation Guide

### High Affluence (percentile > 90)

**Characteristics:**
- High NTL intensity (log_ntl_p90 > 0.8)
- Moderate-to-high density (log_pop_density > 0.6)
- Low slum_risk (density-light_ratio < 1.0)
- High settlement_quality (organized layout)

**Examples:** Delhi South, Mumbai CBD, Bangalore tech hubs

**Actions:** Premium residential zones, business districts, investment targets

---

### Medium Affluence (percentile 25-75)

**Characteristics:**
- Moderate NTL (0.4-0.8)
- Variable density (0.3-0.8)
- Mixed development pattern

**Examples:** Suburban areas, growing neighborhoods, small towns

**Actions:** Mixed development, monitor growth patterns

---

### Low Affluence (percentile < 25)

**Characteristics:**
- Low NTL (< 0.4)
- High density OR low density (but low lights either way)
- High slum_risk (if high density) OR underdeveloped (if low density)

**Examples:** Slums, rural areas, underdeveloped regions

**Actions:** Development focus, slum rehabilitation, infrastructure investment

---

## Advanced Usage

### Filter by Confidence
```python
import pandas as pd

ranking = pd.read_csv("affluence_ranking.csv")

# Only high-confidence pincodes
high_conf = ranking[ranking["confidence"] >= 0.9]

# Top 10% with full confidence
top_10_confident = ranking[
    (ranking["affluence_percentile"] >= 90) & 
    (ranking["confidence"] == 1.0)
]
```

### Mode Comparison
```python
# Run different modes and merge results
default = pd.read_csv("affluence_ranking.csv")
# (copy affluence_ranking.csv before running next mode)
econ = pd.read_csv("affluence_ranking_econ.csv")

# Compare top 20
top_default = set(default.nlargest(20, "affluence_percentile")["pincode"])
top_econ = set(econ.nlargest(20, "affluence_percentile")["pincode"])

overlap = top_default & top_econ
unique_default = top_default - top_econ
unique_econ = top_econ - top_default

print(f"Overlap: {len(overlap)}, Only default: {len(unique_default)}, Only econ: {len(unique_econ)}")
```

### Regional Percentile Analysis
```python
# Get state-wise distribution (using 3-digit pincode prefix)
ranking["state_prefix"] = ranking["pincode"] // 100000

for state in ranking["state_prefix"].unique():
    state_data = ranking[ranking["state_prefix"] == state]
    print(f"State {state:2.0f}: "
          f"Avg percentile = {state_data['affluence_percentile'].mean():.1f}, "
          f"Median = {state_data['affluence_percentile'].median():.1f}")
```

---

## Next Steps

1. **Merge with external data:**
   - Education metrics, healthcare, employment, property values
   - Validate affluence against ground truth (tax records, bank deposits, luxury index)

2. **Temporal analysis:**
   - Compare affluence_ranking across years
   - Identify growth corridors vs declining areas

3. **Slum-specific flagging:**
   - High density + low NTL → slum_risk > 0.5
   - Cross-reference with census slum definitions

4. **Micro-level targeting:**
   - Further subdivide pincodes using NTL pixel-level data
   - Identify microfinance/development opportunities within each pincode

---

## FAQ

**Q: Why is affluent area X scoring lower than expected?**
A: Check confidence score. If sparse_sample_flag=1, the pincode has few pixels in the sample, making features unreliable. Look at `log_ntl_p90` and `log_pop_density` instead.

**Q: Why do industrial areas rank high?**
A: industrial_flag=1 means high NTL spikes (not residential). Confidence is downweighted to 0.3. For pure residential affluence, filter out industrial_flag==1.

**Q: Can I use affluence_percentile directly for clustering?**
A: Yes, but consider:
- Stratify by region first (states have different development levels)
- Use confidence-weighted percentile for strict rankings
- Combine with external validation (known affluent vs poor areas)

**Q: Which mode should I use?**
A: 
- **default** → General affluence ranking
- **econ** → If emphasizing economic activity (metro growth, industrial hubs)
- **urban** → If emphasizing organized development (anti-slum, smart cities)
- **relative** → If comparing regions or states relative to each other

---

## Technical Notes

### Percentile Ranking Strategy

Affluence percentiles are computed against **clean pincodes only** (non-contaminated):
- `affluence_percentile = rank(affluence_raw) / len(clean_pincodes)`
- Contaminated pincodes still get a percentile (for sorting), but their confidence is downweighted
- This prevents extreme contamination cases (uninhabited, industrial) from distorting the distribution

### Why MinMax Normalization?

NTL and WorldPop features have different scales:
- NTL: Log-transformed, ranges [0, 1] after clipping
- WorldPop: Wide range (density 0-47K/km², Gini 0-0.89)
- Cross-signals mix these, so no single scale applies

Solution: MinMax normalize each cross-signal before weighting, so each contributes equally regardless of magnitude.

### Confidence as Multiplicative Weight

`confidence = 0.3 × 0.6 = 0.18` if pincode has both industrial_flag AND sparse_sample_flag

This multiplicative penalty ensures:
- At least one contamination type → confidence < 1.0
- Multiple contamination types → confidence approaches 0

---

## References

- **NTL features:** NOAA VIIRS DNB satellite data via Google Earth Engine
- **WorldPop features:** LandScan 100m population raster, global coverage
- **Affluence concept:** Economic activity (NTL) + settlement density (WorldPop) cross-signal
