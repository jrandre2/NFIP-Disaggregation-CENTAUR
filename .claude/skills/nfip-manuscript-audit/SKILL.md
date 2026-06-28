---
name: nfip-manuscript-audit
description: Project adapter for auditing the NFIP claim-disaggregation manuscript (Tang, Andrews, Nam, Lee). Use when fact-checking numbers, tables, figures, cross-references, methods, or claims in this repository's flood-claim disaggregation paper, or when verifying manuscript prose against the pipeline outputs in data_work/.
---

# NFIP Disaggregation Manuscript Audit

Base: `/Users/jesseandrews/.codex/skills/academic-methods-manuscript-audit/SKILL.md`
In-session equivalent: the `manuscript-audit` skill.

Use when: auditing `manuscript_quarto/index.qmd` for internal consistency, measurement, model reporting, sample construction, robustness honesty, and claim support; or before revising results, tables, figures, the abstract, or limitations.

Keep reusable cross-method audit discipline in the base skill. This wrapper keeps only NFIP-project facts, source-of-truth files, and domain-specific high-risk failure modes.

## Local Context

Manuscript and deliverables
- Source of truth: `manuscript_quarto/index.qmd` (single-file Quarto; abstract + body + appendix).
- Rendered outputs (derived): `manuscript.docx`, `manuscript.pdf` (repo root), `manuscript_quarto/_output/`, `submission/jfrm/`.
- Bibliography: `manuscript_quarto/references.bib`. Figures: `manuscript_quarto/figures/fig_*.png`.
- Tracker / response: `manuscript_quarto/REVISION_TRACKER.md`, `RESPONSE_LETTER_COMBINED.md`.
- Target journal: JFRM (`_quarto-jfrm.yml`); alternates in `journal_configs/` and `_quarto-{jeem,aer,nhaz,risa}.yml`.

Evidence sources for numbers (verify, never report from memory)
- `data_work/validation_metrics.csv` - headline ROC-AUC / PR-AUC / Brier / prevalence for one run.
- `data_work/parameter_sweep_results.csv` - per-config ROC/PR/Brier/log-loss, match rates, candidate counts.
- `data_work/sensitivity/` - elevation, buffer, iteration-convergence tests.
- `data_work/robustness/` - bootstrap CI, jackknife, seed stability, spatial CV.
- `data_work/ia_validation/ia_zip_summary.csv` and `ia_zip_detail.csv` - FEMA Housing Assistance ZIP correlations.
- `data_work/acs_policy_summary.csv`, `acs_policy_uptake.csv` - ACS uptake context.
- `data_work/claim_matching_stats.csv` - per-claim candidate counts and match flags.
- Regenerate with `.venv` active: `python src/pipeline.py run_estimation`, `make_figures` (see CLAUDE.md). Diagnostics live in `data_work/diagnostics/`.

Provenance caveat (check dates before trusting a match)
- Several `data_work/` outputs are dated Dec 31 / Jan 1 and reflect an earlier ZIP-based run; the manuscript baseline is ZCTA-based. Treat a numeric mismatch as a possible stale-output / un-committed-rerun issue, not an automatic manuscript error. State explicitly when a table cannot be reproduced from current outputs.

## Domain High-Risk Failure Modes

- Spatial-unit confusion: the baseline is ZCTA + FEMA flood-zone, largest-building-per-parcel. ZIP, ZCTA, and census block group (CBG) are distinct; claim ZIPs are treated as ZCTA identifiers. Do not call the baseline "ZIP" in prose.
- Matched vs total claims: Dodge 239 matched of 240; Douglas 178 matched of 183. Keep denominators straight across the spatial-unit, covariate, payment-strata, and lat/long tables; flag any table whose claim counts imply a different denominator.
- Likelihood score $L_b$ is an evidence index, not a formal probability. Reject prose that calls it a probability or a forecast of future loss.
- Validation scope: ROC/PR evaluation exists only for the March 2019 event (the only event with an inundation footprint). Reject multi-event performance claims.
- PR-AUC depends on class prevalence; verify prevalence-adjusted lift (`PR-AUC / prevalence`) recomputes from displayed values or note the rounding source.
- Small-sample fragility: report FEMA Housing Assistance correlations only when ZIP overlap >= 4 (Douglas sits at 4); upper-tail loss strata are tiny (Douglas > P90, n = 19, ROC < 0.5).
- Calibration vs discrimination: naive-baseline Brier can beat the model's Brier even when the model wins on ROC/PR; do not let discrimination claims imply better calibration.
- Privacy/provenance: never claim recovery of true claim coordinates. Nebraska parcels are proprietary; the NE DNR inundation polygon is non-public. Keep these out of any "fully public data" overclaim.
- Bootstrap setting: $n = 1000$ draws per claim (stated in the Appendix).

## Workflow

1. Load the base skill, then read the local files named above (manuscript first, then the relevant `data_work/` evidence).
2. Apply the base audit procedure under these constraints. Compare abstract, methods, results prose, every table, every figure caption, and the appendix for agreement on counts, units, ROC/PR/Brier values, prevalence, and cross-references (`@tbl-*`, `@fig-*`, `@eq-*`, `@sec-*`).
3. For each numeric claim, name the `data_work/` file that supports it; if none does, say so rather than asserting correctness.
4. Report findings ordered by severity with manuscript line references and the evidence source checked.

## Verification

- Support every critical finding with a manuscript location and a `data_work/` source (or an explicit "no current source").
- Confirm all `@`-cross-references resolve and all referenced `figures/fig_*.png` exist.
- Re-run citation key reconciliation (`index.qmd` vs `references.bib`) when references change.
- Do not propose number changes from memory; propose a rerun or an author check instead.

## Portability Notes

Reusable cross-method audit rules belong in the base skill. NFIP file paths, the ZCTA baseline, county claim counts, the likelihood-index framing, and the provenance caveat belong here.
