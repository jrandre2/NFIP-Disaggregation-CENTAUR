# NFIP Claims Disaggregation Project

**Paper Title**: An Approach to Probabilistic Disaggregation of National Flood Insurance Program (NFIP) Claims onto Building Footprints

**Manuscript ID**: RA-00555-2025
**Target Journal**: Risk Analysis
**Status**: Major Revision (R&R1)

## Project Overview

This project develops and validates a probabilistic spatial disaggregation method to assign anonymized NFIP claims to individual building footprints, enabling structure-level flood risk mapping while preserving policyholder privacy.

### Key Methodological Components

- **Hierarchical filtering** based on ZIP code, flood zone, and building attributes
- **Bootstrap resampling** to quantify uncertainty in claim assignment
- **Building-specific likelihood scores** for flood claim association
- **Topographic covariates** (slope and distance-to-SFHA) for outside-floodplain checks
- **External validation** using FEMA housing assistance totals at ZIP level
- **Loss-size stratification** to assess upper-tail sensitivity

### Study Area

- Dodge County, Nebraska
- Douglas County, Nebraska
- March 2019 flood event (Dodge + Douglas)
- 2011 late spring storms event (Cass + Dakota)

## Analysis Recreation Status

All analyses from the original paper have been successfully recreated using open-source Python tools:

| Phase | Status | Key Results |
|-------|--------|-------------|
| Data Preparation | ✅ Complete | 261 claims, 34,350 buildings |
| Core Bootstrap | ✅ Complete | ROC-AUC 0.969, PR-AUC 0.874 |
| Parameter Sweep | ✅ Complete | Table 1 (4 configurations) |
| Sensitivity Analysis | ✅ Complete | Iteration, elevation, buffer tests |
| Per-Claim Diagnostics | ✅ Complete | 99.6% match rate |
| Publication Figures | ✅ Complete | ROC, PR, calibration plots |
| Robustness Checks | ✅ Complete | Seed stability, jackknife, spatial CV |
| External Validation | ✅ Complete | ZIP-level FEMA IA correlation |
| Loss-Size Stratification | ✅ Complete | Low/mid/high payment strata |

**Full results**: See [doc/ANALYSIS_RESULTS.md](doc/ANALYSIS_RESULTS.md)

## Repository Structure

```
├── manuscript_quarto/      # Manuscript in Quarto format
│   ├── index.qmd          # Main manuscript
│   ├── appendix-*.qmd     # Appendices
│   ├── _quarto-risa.yml   # Risk Analysis journal profile
│   └── REVISION_TRACKER.md # Review response tracking
├── scripts/               # Analysis scripts
│   └── NEFloodMitigation/ # Cloned GitHub repository with geospatial scripts
├── doc/                   # Documentation
│   └── reviews/           # Review cycle files
├── src/                   # CENTAUR pipeline stages
└── data_work/             # Processed data (gitignored)
```

## Current Status

**Revision in Progress** - Addressing major revision from Risk Analysis journal.

### Key Reviewer Concerns to Address

1. Method novelty positioning (Reviewer 2 - Critical)
2. Comparison to Wagner (2022) approach (Critical)
3. Case selection justification (Dodge County vs Texas/Louisiana)
4. Data source documentation (OpenFEMA fields)
5. Generalizability beyond single event
6. Intended audience and use cases

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run full analysis recreation
python src/stages/s00_prepare_nfip.py     # Prepare data
python src/stages/s01_bootstrap_disagg.py  # Core bootstrap
python src/stages/s02_parameter_sweep.py   # Table 1 recreation
python src/stages/s02b_sensitivity_analysis.py  # Sensitivity tests
python src/stages/s02c_claim_diagnostics.py     # Per-claim diagnostics
python src/stages/s03_figures.py           # Generate figures
python src/stages/s04_robustness.py        # Robustness checks
python src/stages/s02d_ia_validation.py    # FEMA housing assistance validation
python src/stages/s02e_acs_uptake.py       # ACS policy uptake context

# Review management
python src/pipeline.py review_status       # Check review status
python src/pipeline.py review_verify       # Verify revision completeness
python src/pipeline.py review_response     # Generate response letter

# Render manuscript
cd manuscript_quarto && ./render_all.sh --profile risa
```

## Pipeline Stages

The analysis pipeline in `src/stages/`:

| Stage | Script | Purpose |
|-------|--------|---------|
| s00 | `s00_prepare_nfip.py` | Data preparation (claims, buildings, elevation, flood zones) |
| s00b | `s00b_download_nfhl.py` | Download NFHL flood zones from FEMA API |
| s01 | `s01_bootstrap_disagg.py` | Core bootstrap disaggregation algorithm |
| s02 | `s02_parameter_sweep.py` | Table 1 parameter sweep (4 configurations) |
| s02b | `s02b_sensitivity_analysis.py` | Sensitivity analysis (iterations, elevation, buffer) |
| s02c | `s02c_claim_diagnostics.py` | Per-claim filter chain diagnostics |
| s02d | `s02d_ia_validation.py` | FEMA housing assistance ZIP validation |
| s02e | `s02e_acs_uptake.py` | ACS-based policy uptake context |
| s03 | `s03_figures.py` | Publication figures (ROC, PR, calibration) |
| s04 | `s04_robustness.py` | Robustness checks (seed stability, jackknife, spatial CV) |

## Data and Code

- Repository: https://github.com/jrandre2/NFIP-Disaggregation-CENTAUR
- OpenFEMA: NFIP Claims, NFIP Policies, Housing Assistance Owners
  - https://www.fema.gov/openfema-data-page/fima-nfip-claims
  - https://www.fema.gov/openfema-data-page/fima-nfip-policies
  - https://www.fema.gov/openfema-data-page/housing-assistance-owners
- FEMA NFHL: https://www.fema.gov/flood-maps/national-flood-hazard-layer
- USGS 3DEP: https://www.usgs.gov/3d-elevation-program
- U.S. Census TIGER/ACS: https://api.census.gov/data/2019/acs/acs5.html

### Original GitHub Scripts

The original ArcPy-based scripts are in `scripts/NEFloodMitigation/Data/Geospatial-Scripts/`:

| Script | Purpose |
|--------|---------|
| `Bootstrap_Parameter_Testing_Pipeline.py` | Original bootstrap pipeline (ArcPy) |
| `NFIPPolicyDescriptivesBootstrap.py` | NFIP policy diagnostics |
| `CenPy_ACS_to_Shapefile.py` | Census data preparation |
| `DEM_to_Points.py` | Elevation extraction |

## Team

- **PI**: Dr. Zhenghong Tang (University of Nebraska-Lincoln)
- **Co-Investigators**: Dr. Yunwoo Nam, Dr. Jesse Andrews, Dr. Jiyoung Lee

---

*Built on the CENTAUR research workflow platform*
