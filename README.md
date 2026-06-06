# NFIP Claims Disaggregation (CENTAUR)

**An Approach to Probabilistic Disaggregation of National Flood Insurance Program (NFIP) Claims onto Building Footprints**

**Manuscript ID**: RA-00555-2025 | **Journal**: Risk Analysis | **Status**: Major Revision (R&R1)

**Authors**: Jesse Andrews, Zhenghong Tang, Yunwoo Nam, Jiyoung Lee (University of Nebraska-Lincoln)

---

## What This Project Does

This project develops and validates a probabilistic spatial method to assign anonymized NFIP flood insurance claims to individual building footprints. The approach enables structure-level flood risk mapping while preserving policyholder privacy — useful for local planners, floodplain managers, and researchers who need building-scale flood exposure data that the public NFIP dataset does not provide directly.

**Study area**: Dodge and Douglas Counties, Nebraska (March 2019 flood); Cass and Dakota Counties (2011 late spring storms)

**Key methods**: Hierarchical filtering by ZIP, flood zone, and building attributes; bootstrap resampling for uncertainty; topographic covariates (slope, distance to SFHA); external validation against FEMA housing assistance totals.

**Core results**: ROC-AUC 0.969, PR-AUC 0.874 on held-out test set; 99.6% claim match rate.

---

## Manuscript

The latest compiled manuscript is at the repository root:

- [`manuscript.docx`](manuscript.docx) — Word format (for journal submission and editing)
- [`manuscript.pdf`](manuscript.pdf) — PDF format (for reading)

Source: `manuscript_quarto/index.qmd` (Quarto, requires Quarto + LaTeX to re-render)

The revision response letter is in `manuscript_quarto/RESPONSE_LETTER_COMBINED.*`.

---

## Repository Structure

```
├── manuscript.docx / manuscript.pdf   # Latest compiled manuscript (root)
├── manuscript_quarto/                 # Quarto source + revision materials
│   ├── index.qmd                      # Main manuscript source
│   ├── references.bib                 # Bibliography
│   ├── REVISION_TRACKER.md            # Point-by-point reviewer response tracker
│   ├── RESPONSE_LETTER_COMBINED.*     # Combined response letter (docx/md/pdf)
│   └── figures/                       # Publication figures (PDF + PNG)
├── src/stages/                        # Analysis pipeline stages (s00–s04)
├── scripts/                           # Utility and revision scripts
│   └── NEFloodMitigation/             # Related sub-project (separate git repo)
├── data_raw/                          # Raw input data (gitignored; see DATA_ACQUISITION.md)
├── data_work/                         # Intermediate outputs (gitignored)
├── doc/                               # Extended documentation
└── demo/                              # Demo data and example run
```

## Running the Analysis

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run pipeline stages in order
python src/stages/s00_prepare_nfip.py          # Prepare NFIP claims + buildings
python src/stages/s01_bootstrap_disagg.py      # Core bootstrap disaggregation
python src/stages/s02_parameter_sweep.py       # Table 1 (4 configurations)
python src/stages/s02b_sensitivity_analysis.py # Sensitivity tests
python src/stages/s02c_claim_diagnostics.py    # Per-claim diagnostics
python src/stages/s02d_ia_validation.py        # External FEMA IA validation
python src/stages/s02e_acs_uptake.py           # ACS policy uptake context
python src/stages/s03_figures.py               # Publication figures
python src/stages/s04_robustness.py            # Robustness checks

# 3. Re-render manuscript (requires Quarto + LaTeX)
cd manuscript_quarto && ./render_all.sh --profile risa
```

See [`doc/GETTING_STARTED.md`](doc/GETTING_STARTED.md) for data acquisition and full setup.

---

## Data Sources

All input data are publicly available and gitignored locally. See [`doc/DATA_ACQUISITION.md`](doc/DATA_ACQUISITION.md) for download instructions.

| Dataset | Source |
|---------|--------|
| NFIP Claims (OpenFEMA) | https://www.fema.gov/openfema-data-page/fima-nfip-claims |
| NFIP Policies (OpenFEMA) | https://www.fema.gov/openfema-data-page/fima-nfip-policies |
| Housing Assistance Owners | https://www.fema.gov/openfema-data-page/housing-assistance-owners |
| FEMA NFHL Flood Zones | https://www.fema.gov/flood-maps/national-flood-hazard-layer |
| USGS 3DEP Elevation | https://www.usgs.gov/3d-elevation-program |
| U.S. Census TIGER/ACS | https://api.census.gov/data/2019/acs/acs5.html |

---

## Related Sub-Project

`scripts/NEFloodMitigation/` is a **separate git repository** ([NEFloodMitigation-Risk-Assessment-and-Community-Adaptation](https://github.com/jrandre2/NEFloodMitigation-Risk-Assessment-and-Community-Adaptation)) containing the original ArcPy-based geospatial scripts this project's Python pipeline was built from. It is registered as a git submodule. The CENTAUR pipeline in `src/stages/` is the recommended open-source replacement that does not require ArcGIS.

---

## Documentation Index

| Doc | Purpose |
|-----|---------|
| [`doc/METHODOLOGY.md`](doc/METHODOLOGY.md) | Full methodological details |
| [`doc/ANALYSIS_RESULTS.md`](doc/ANALYSIS_RESULTS.md) | Numerical results summary |
| [`doc/DATA_DICTIONARY.md`](doc/DATA_DICTIONARY.md) | Variable definitions |
| [`doc/PIPELINE.md`](doc/PIPELINE.md) | Pipeline architecture |
| [`doc/REPRODUCTION.md`](doc/REPRODUCTION.md) | Exact reproduction steps |
| [`doc/CHANGELOG.md`](doc/CHANGELOG.md) | Change history |
| [`manuscript_quarto/REVISION_TRACKER.md`](manuscript_quarto/REVISION_TRACKER.md) | Reviewer response tracker |

---

*Built on the CENTAUR research workflow platform.*
