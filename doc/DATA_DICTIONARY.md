# Data Dictionary

**Related**: [PIPELINE.md](PIPELINE.md) | [METHODOLOGY.md](METHODOLOGY.md) | [ARCHITECTURE.md](ARCHITECTURE.md)
**Status**: Active
**Last Updated**: 2025-12-31

---

## Overview

This document defines all variables used in the NFIP Claims Disaggregation analysis, including NFIP claims data, building footprints, flood zones, and output probability scores.

---

## NFIP Claims Variables

Source: FEMA OpenFEMA API / `data_work/claims_prepared.csv`

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `ClaimID` | string | - | Unique claim identifier (internal) |
| `ZIP` | string | - | 5-digit ZIP code of property |
| `FloodZone` | string | - | FEMA flood zone (AE, AO, X, A, B) |
| `BFE` | float | feet | Base Flood Elevation (NAVD88) |
| `DateOfLoss` | date | - | Date claim was filed |
| `YearOfLoss` | int | - | Year of flood event |
| `AmountPaidBldg` | float | USD | Building claim payment amount |
| `AmountPaidCont` | float | USD | Contents claim payment amount |
| `OccupancyType` | int | - | Property occupancy type (1=residential) |

### Flood Zone Codes

| Code | Description | SFHA Status |
|------|-------------|-------------|
| AE | Base floodplain with BFE | Yes |
| AO | Shallow flooding (1-3 ft) | Yes |
| A | Base floodplain, no BFE | Yes |
| A04, A05 | Numbered A zones | Yes |
| X | Minimal flood hazard | No |
| B | Moderate flood hazard (legacy) | No |

---

## Building Footprint Variables

Source: Microsoft Building Footprints / `data_work/buildings_prepared.gpkg`

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `BldgID` | int | - | Unique building identifier |
| `geometry` | polygon | - | Building footprint geometry |
| `centroid` | point | - | Building centroid (derived) |
| `ZIP` | string | - | ZIP code from parcel join |
| `FloodZone` | string | - | NFHL flood zone from spatial join |
| `SFHA` | bool | - | Special Flood Hazard Area flag |
| `ELEVATION` | float | feet | Ground elevation (NAVD88, from DEM) |
| `Parcel_ID` | string | - | Associated parcel identifier |
| `Total_Asse` | float | USD | Total assessed value |
| `ClassCode` | string | - | Property classification code |
| `BuildYear` | int | year | Parcel-reported building year |

### Elevation Source

Elevation extracted from USGS 3DEP 10m DEM tiles using zonal statistics (mean value within building footprint).

---

## Inundation Variables

Source: USGS/modeled extent / `data_work/inundation.gpkg`

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `geometry` | polygon | - | Inundation extent boundary |
| `event_date` | date | - | Date of flood event |
| `source` | string | - | Data source (USGS, modeled, etc.) |

---

## Bootstrap Output Variables

Source: `data_work/bootstrap_results.csv`

| Variable | Type | Unit | Range | Description |
|----------|------|------|-------|-------------|
| `BldgID` | int | - | - | Building identifier |
| `probability` | float | - | [0, 1] | Probability of claim association |
| `n_draws` | int | - | [0, N×Claims] | Times selected in bootstrap |
| `n_claims` | int | - | - | Number of claims matched to this building |
| `inundated` | bool | - | - | Ground truth: within inundation extent |

### Probability Calculation

$$P(b) = \frac{\text{n\_draws}_b}{N \times |\text{matched\_claims}|}$$

Where N = number of bootstrap iterations (default: 1000).

---

## Validation Metrics

Source: `data_work/validation_metrics.csv`

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `roc_auc` | float | [0, 1] | Area Under ROC Curve |
| `pr_auc` | float | [0, 1] | Area Under Precision-Recall Curve |
| `brier_score` | float | [0, 1] | Mean squared probability error |
| `prevalence` | float | [0, 1] | Proportion of inundated buildings |
| `n_buildings` | int | - | Total buildings scored |
| `n_inundated` | int | - | Buildings within inundation extent |

---

## Diagnostic Variables

Source: `data_work/diagnostics/filter_chain.csv`

| Variable | Type | Description |
|----------|------|-------------|
| `ClaimID` | string | Claim identifier |
| `n_initial` | int | Candidates after ZIP+FZ match |
| `n_after_elev` | int | Candidates after elevation filter |
| `n_final` | int | Final candidate pool size |
| `n_unique_sel` | int | Unique buildings selected in bootstrap |
| `var_sel` | float | Variance of selection counts |
| `top_share` | float | Share of draws to top candidate |
| `matched` | bool | Whether claim was matched |

---

## Data Files Summary

### Input Files (`data_raw/`)

| File | Format | Description |
|------|--------|-------------|
| `nfhl/S_FLD_HAZ_AR.shp` | Shapefile | FEMA NFHL flood zones |

### Intermediate Files (`data_work/`)

| File | Format | Records | Description |
|------|--------|---------|-------------|
| `claims_prepared.csv` | CSV | 261 | Filtered NFIP claims |
| `buildings_prepared.gpkg` | GeoPackage | 34,350 | Buildings (largest per parcel) |
| `buildings_all.gpkg` | GeoPackage | 43,271 | All buildings (before filter) |
| `inundation.gpkg` | GeoPackage | 1 | March 2019 flood extent |

### Output Files (`data_work/`)

| File | Format | Description |
|------|--------|-------------|
| `bootstrap_results.csv` | CSV | Building probability scores |
| `validation_metrics.csv` | CSV | ROC-AUC, PR-AUC, Brier |
| `parameter_sweep_results.csv` | CSV | Table 1 metrics (4 configs) |

### Diagnostic Files (`data_work/diagnostics/`)

| File | Format | Description |
|------|--------|-------------|
| `filter_chain.csv` | CSV | Per-claim matching details |
| `summary_by_fz.csv` | CSV | Stats by flood zone |
| `summary_by_zip.csv` | CSV | Stats by ZIP code |

### Sensitivity Files (`data_work/sensitivity/`)

| File | Format | Description |
|------|--------|-------------|
| `iteration_convergence.csv` | CSV | N iterations test |
| `elevation_sensitivity.csv` | CSV | Elevation tolerance test |
| `buffer_sensitivity.csv` | CSV | Inundation buffer test |

### Robustness Files (`data_work/robustness/`)

| File | Format | Description |
|------|--------|-------------|
| `seed_stability.csv` | CSV | 20-seed stability test |
| `spatial_cv.csv` | CSV | Leave-ZIP-out CV |
| `jackknife_claims.csv` | CSV | Leave-one-claim-out |
| `bootstrap_ci.csv` | CSV | 100 bootstrap samples |
