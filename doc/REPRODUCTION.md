# Reproduction Guide

**Related**: [PIPELINE.md](PIPELINE.md) | [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | [DATA_ACQUISITION.md](DATA_ACQUISITION.md)
**Status**: Project-specific
**Last Updated**: 2026-06-06

---

## Overview

This guide documents how to reproduce the full analysis for "An Approach to Probabilistic Disaggregation of National Flood Insurance Program (NFIP) Claims onto Building Footprints" (Risk Analysis, RA-00555-2025).

Study areas: Dodge and Douglas Counties, Nebraska (March 2019 flood); Cass and Dakota Counties (2011 late spring storms).

---

## Prerequisites

### Software Requirements

- Python 3.10+
- Quarto 1.3+ with LaTeX (for manuscript re-render only)
- Git

### Python Packages

```bash
pip install -r requirements.txt
# For spatial dependencies:
pip install -r requirements-spatial.txt
```

---

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/jrandre2/NFIP-Disaggregation-CENTAUR.git
cd NFIP-Disaggregation-CENTAUR
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-spatial.txt
```

### 3. Obtain Raw Data

All input data are publicly available. See [DATA_ACQUISITION.md](DATA_ACQUISITION.md) for download instructions. Place files in `data_raw/`.

Key sources:
- NFIP Claims (OpenFEMA): https://www.fema.gov/openfema-data-page/fima-nfip-claims
- NFIP Policies (OpenFEMA): https://www.fema.gov/openfema-data-page/fima-nfip-policies
- FEMA NFHL Flood Zones: https://www.fema.gov/flood-maps/national-flood-hazard-layer
- USGS 3DEP Elevation: https://www.usgs.gov/3d-elevation-program
- U.S. Census TIGER/ACS: https://api.census.gov/data/2019/acs/acs5.html

---

## Run Pipeline

### Full Reproduction (in order)

```bash
source .venv/bin/activate

# Stage 0: Prepare NFIP claims + buildings
python src/stages/s00_prepare_nfip.py

# Stage 1: Core bootstrap disaggregation
python src/stages/s01_bootstrap_disagg.py

# Stage 2: Parameter sweep (Table 1 — 4 configurations)
python src/stages/s02_parameter_sweep.py

# Stage 2b: Sensitivity tests
python src/stages/s02b_sensitivity_analysis.py

# Stage 2c: Per-claim diagnostics
python src/stages/s02c_claim_diagnostics.py

# Stage 2d: External FEMA IA validation
python src/stages/s02d_ia_validation.py

# Stage 2e: ACS policy uptake context
python src/stages/s02e_acs_uptake.py

# Stage 3: Publication figures
python src/stages/s03_figures.py

# Stage 4: Robustness checks
python src/stages/s04_robustness.py

# Re-render manuscript (requires Quarto + LaTeX)
cd manuscript_quarto && ./render_all.sh --profile risa
```

### Expected Key Outputs

| File | Description |
|------|-------------|
| `data_work/bootstrap_results.csv` | Bootstrap disaggregation output |
| `data_work/parameter_sweep_results.csv` | Table 1 configurations |
| `data_work/validation_metrics.csv` | ROC-AUC, PR-AUC, match rate |
| `data_work/sensitivity/` | Sensitivity analysis results |
| `data_work/robustness/` | Robustness check results |
| `manuscript_quarto/figures/` | Publication figures (PDF + PNG) |

---

## Output Files

### Data Files

| File | Description |
|------|-------------|
| `data_work/panel.parquet` | Main analysis panel |
| `data_work/diagnostics/*.csv` | Estimation results |

### Figures

| File | Description |
|------|-------------|
| `manuscript_quarto/figures/[fig1].png` | [Description] |
| `manuscript_quarto/figures/[fig2].png` | [Description] |

### Manuscript

| File | Description |
|------|-------------|
| `manuscript_quarto/_output/index.html` | HTML manuscript |
| `manuscript_quarto/_output/[Name].pdf` | PDF manuscript |
| `manuscript_quarto/_output/[Name].docx` | Word manuscript |

---

## Verification

### Check Output Counts

[Document expected counts at each stage]

```bash
# Example verification commands
python -c "import pandas as pd; print(pd.read_parquet('data_work/panel.parquet').shape)"
```

### Expected Results

[Document key expected results for verification]

---

## Troubleshooting

### Common Issues

**Issue: Package not found**
```bash
pip install -r requirements.txt
```

**Issue: Quarto not found**
```bash
# Use project-local quarto
../tools/bin/quarto render
```

**Issue: Memory error**
[Suggest workarounds for memory-intensive steps]
