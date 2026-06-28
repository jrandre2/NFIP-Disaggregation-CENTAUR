# Manuscript Table & Statistic Provenance

Source of every number in `index.qmd`. The authoritative analysis outputs live in
`data_work/revision/{dodge,douglas}/` (note: `data_work/` is git-ignored, so these
are regenerated locally, not committed). The bootstrap is **deterministic**
(seed 42, 1000 draws per claim; see `src/config.py` `BOOTSTRAP_SEED`,
`BOOTSTRAP_ITERATIONS`). The naive model is deterministic with no seed.

## Regeneration

```bash
source .venv/bin/activate
# 1. Prepare per-county inputs (claims, policies, all/largest buildings, inundation)
python scripts/prepare_revision_data.py --counties dodge,douglas
# 2. Bootstrap variants, payment strata, policy-vs-claim comparison
python scripts/run_revision_experiments.py --counties dodge,douglas
# 3. Naive spatial-aggregation baselines (Dodge only; Table @tbl-naive)
python scripts/run_naive_baselines.py --county dodge
# 4. External validation and ACS uptake context
python src/stages/s02d_ia_validation.py
python src/stages/s02e_acs_uptake.py
# 5. Figures (reads data_work/revision/)
python src/stages/s03_figures.py
```

## Table → source mapping

| Manuscript element | Source file (`data_work/...`) | Producing command |
|---|---|---|
| Baseline ROC/PR/Brier (prose, @sec-results-performance) | `revision/{county}/metrics_claims_variants.csv` row `zcta_fz` | `run_revision_experiments.py` |
| @tbl-spatial-units (ZIP/ZCTA/CBG) | `revision/{county}/metrics_claims_variants.csv` rows `baseline_zip_fz`, `zcta_fz`, `cbg_fz` | `run_revision_experiments.py` |
| @tbl-covariates (slope, dist-to-SFHA) | `revision/{county}/metrics_claims_variants.csv` rows `baseline_zip_fz`, `slope_le_2deg`, `dist_sfha_500m` | `run_revision_experiments.py` |
| @tbl-pr-adjusted | derived: PR-AUC ÷ prevalence from `metrics_claims_variants.csv` `zcta_fz` | (computed in prose) |
| @tbl-payment-strata | `revision/{county}/metrics_payment_strata.csv` (`n_claims` = per-stratum total; `matched` = scored) | `run_revision_experiments.py` |
| @tbl-policy-claims (distribution) | `revision/{county}/policy_claim_distribution.csv` | `run_revision_experiments.py` |
| Inline KS / Pearson r / n (policy vs claim) | `revision/{county}/policy_claim_comparison.csv` | `run_revision_experiments.py` |
| @tbl-naive (Dodge only) | `revision/dodge/metrics_naive.csv` | `run_naive_baselines.py` |
| @tbl-ia-validation | `ia_validation/ia_zip_summary.csv` | `s02d_ia_validation.py` |
| ACS uptake medians (inline) | `acs_policy_summary.csv` | `s02e_acs_uptake.py` |
| Figures (ROC/PR/calibration/metric/hex/zoom) | `revision/{county}/*` | `s03_figures.py` |

## Verification status (2026-06-27)

ROC/PR/Brier in every table reconcile to the CSVs above to three decimals.
`run_naive_baselines.py` reproduces @tbl-naive ROC/PR/Brier exactly; Log-Loss is
within ~0.01-0.03 (a near-zero clipping-convention difference).

## Superseded outputs

Top-level `data_work/{parameter_sweep_results,validation_metrics,claim_matching_stats}.csv`
are from an earlier **ZIP-based, single-county** exploration (written by
`src/stages/s01_bootstrap_disagg.py` and `s02_parameter_sweep.py`, 261 claims,
ROC ~0.969) and do **not** correspond to the manuscript's ZCTA baseline. They have
been moved to `data_work/_archive/zip_baseline_run/` to avoid confusion; nothing in
the manuscript or figure pipeline reads them.

## Known reproducibility caveats

- The Douglas County NFHL is obtained portably with
  `python src/stages/s00b_download_nfhl.py --county douglas`
  (writes `data_raw/nfhl/douglas_county_flood_zones.gpkg`), and the Douglas
  boundary is derived from `data_raw/cbg`, so both counties are runnable. A fresh
  download reflects the current effective NFHL and may differ marginally from the
  original product vintage used for the committed numbers. A verified full
  from-raw rerun (fresh NFHL + CBG boundary) reproduced the Douglas baseline
  ROC-AUC exactly (0.899; 178 matched claims), with negligible drift in PR-AUC
  (0.691 -> 0.685) and Brier (0.296 -> 0.294).
- `prepare_revision_data.py` still reads buildings/parcels/elevation/inundation
  from `Freeze and Flight` and ZCTA from `ML Vision Broadband` via absolute paths;
  parameterizing those into config is future work.
- "Largest building per parcel" is implemented as the highest assessed value per
  parcel (`prepare_revision_data.py`), used consistently across all tables.
