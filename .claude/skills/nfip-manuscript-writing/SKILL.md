---
name: nfip-manuscript-writing
description: Project adapter for writing, editing, and copyediting the NFIP claim-disaggregation manuscript in this repository. Use when drafting or revising prose, captions, the abstract, or response-letter text for the flood-claim disaggregation paper, and for copyedit passes that must preserve standalone academic voice and project terminology.
---

# NFIP Disaggregation Manuscript Writing And Editing

Base: `/Users/jesseandrews/.codex/skills/academic-manuscript-writing-editing/SKILL.md`

Use when: writing or copyediting `manuscript_quarto/index.qmd`, tightening a section, revising captions, or producing review-response prose. For pure number/claim auditing, use the companion adapter `nfip-manuscript-audit` first.

Keep the shared writing standard, AI-trope avoidance, and render discipline in the base skill. This wrapper keeps only NFIP-project files, terminology, and conventions.

## Local Context

- Source: `manuscript_quarto/index.qmd`. Bibliography: `references.bib`. Figures: `figures/fig_*.png`.
- Render: `cd manuscript_quarto && ./render_all.sh` (Quarto -> HTML/PDF/DOCX); outputs land in `_output/` and the repo-root `manuscript.{docx,pdf}`. Re-render after editing `.qmd`.
- Repo writing standard (CLAUDE.md): no script names, file paths, or repository jargon in manuscript prose; no metacommentary about the writing or pipeline process; no TODO/FIXME; self-contained academic prose.
- Authors: Zhenghong Tang (corresponding, UNL), Jesse Andrews (Texas Tech), Yunwoo Nam (UNL), Jiyoung Lee (UNL). Target journal: JFRM.

## Project Terminology And Conventions

Preserve these exactly; do not let synonyms drift:
- "building footprints", "candidate buildings / candidate pool", "bootstrap ensemble", "draw counts".
- "likelihood score" or "evidence index" for $L_b$ - never "probability" or "forecast".
- "ZCTA + flood-zone baseline", "largest building per parcel"; keep ZIP, ZCTA, and census block group distinct.
- "Special Flood-Hazard Area", "inundation footprint", "Zone X", county names "Dodge" and "Douglas", the "March 2019" event.
- "disaggregation" (not "downscaling"); "anonymized" NFIP claims.

Define on first use in each reading context (abstract, body, standalone captions), exempting only journal-standard terms:
- NFIP, FEMA (already defined), SFHA, NFHL (National Flood Hazard Layer), ZCTA (ZIP Code Tabulation Area), CBG (census block group), ACS (American Community Survey), IHP (Individuals and Households Program), ROC-AUC, PR-AUC, KS (Kolmogorov-Smirnov), FOIA, 3DEP.

House-style reminders that recur in this manuscript (the base skill explains why):
- Avoid em dashes (`---`) as a prose device; use commas, colons, or separate sentences.
- Reserve "significant" for statistical significance with a named test/threshold; otherwise "substantial", "notable", "meaningful".
- Report p-values as `p < 0.001`, never `p \approx 0`.
- Pair "rank correlation" with Spearman, "correlation" with Pearson.
- Keep hyphenation consistent (e.g. "micro-hotspot"); "flood damage" (uncountable), not "flood damages".

## Workflow

1. Load the base skill, then read this adapter and the current `index.qmd`.
2. Classify the task (writing / copyedit / caption / response prose) and work credibility-risk-first for mixed tasks.
3. Apply the base writing standard with the terminology above; do not change numbers or claims (that is the audit adapter's job - route number questions there).
4. Report the exact edits made and whether a re-render was run.

## Verification

- Confirm `@tbl-*`, `@fig-*`, `@eq-*`, and `@sec-*` references still resolve after edits.
- Re-render via `render_all.sh` (or state why not) before treating a DOCX/PDF deliverable as final.
- Run a closure pass for AI-writing tropes, computationalized prose, and pipeline jargon leaking into manuscript voice.

## Portability Notes

Reusable writing rules and trope lists belong in the base skill. NFIP terminology, the acronym list, render command, and author/journal facts belong here.
