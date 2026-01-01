Response to Reviewers
Manuscript: An Approach to Probabilistic Disaggregation of National Flood Insurance Program (NFIP) Claims onto Building Footprints
Journal: Risk Analysis

We thank the editor and reviewers for their thoughtful and constructive comments. We revised the manuscript substantially. Key updates include:
- Reframed the contribution as a transparent disaggregation workflow with building-level uncertainty surfaces rather than a novel algorithm.
- Switched the baseline spatial unit to ZCTA with flood-zone matching and clarified filter logic.
- Added a Douglas County replication (March 2019) to assess robustness.
- Added a data sources and versions table with explicit years and dataset provenance.
- Added sensitivity tests for ZIP vs ZCTA vs census block group, coarse latitude/longitude, and construction-year tolerances.
- Added policy vs claim distribution comparisons to contextualize insurance penetration.
- Expanded limitations and use-case discussion.

Reviewer 1

R1.1 Comment: The paper’s contribution to broader scholarship is weak and should be articulated better. Clarify practical utility and address fat-tailed loss issues.
Response: We rewrote the Abstract, Introduction, and Discussion to clarify that the contribution is a reproducible disaggregation workflow that yields building-level likelihood surfaces and uncertainty diagnostics for post-event analysis, not a prediction model. We explicitly caution against interpreting the outputs as forecasts in fat-tailed loss settings and emphasize their use for spatializing historical claims when precise locations are unavailable.

R1.2 Comment: The case selection (Dodge County) needs justification; consider other regions or events.
Response: We added a robustness replication in Douglas County using the same March 2019 event window (204 residential claims) and report performance alongside Dodge County (261 residential claims). The Methods and Results now justify the Nebraska case on data availability (parcel data, inundation footprint) and include cross-county validation to address generalizability.

R1.3 Comment: Claims occur outside floodplains; what additional parameters could improve accuracy?
Response: We now report SFHA and Zone X claim counts in both counties and discuss the limits of relying solely on flood zones and BFE. We added a discussion of candidate covariates (distance to water bodies, topographic indices, drainage infrastructure, land cover) and report sensitivity tests for spatial units, latitude/longitude, and construction-year constraints.

R1.4 Comment: Clarify intended audience and use cases; NFIP already has precise locations via FOIA.
Response: We clarify that the method is intended for planners and researchers without FOIA access to precise locations. The Discussion now explains that the workflow complements, but does not replace, internal NFIP datasets and supports post-event mapping, micro-hotspot identification, and risk communication.

R1.5 Comment: The paper overlooks limitations such as outdated flood maps and uninsured properties.
Response: We added a limitations paragraph noting NFHL map currency issues and reporting SFHA policy penetration of approximately 0.29 to 0.30 in both counties. We explicitly state that the method addresses insured claims and discuss uninsured exposure as a limitation and future research direction.

Reviewer 2

R2.1 Comment: The novelty is oversold; the method is a simple GIS matching algorithm.
Response: We revised the Abstract and Discussion to avoid claims of algorithmic novelty. The contribution is now framed as a transparent and reproducible disaggregation workflow that produces building-level likelihood surfaces and uncertainty diagnostics from public data.

R2.2 Comment: Single-event focus raises generalizability concerns.
Response: We added a Douglas County replication for the same event and report performance metrics in the Results. We also emphasize that broader generalizability requires multi-event validation and identify this as future work.

R2.3 Comment: The manuscript omits relevant related work such as Wagner (2022).
Response: We added and discussed Wagner (2022) in the literature review, noting how our building-level disaggregation with uncertainty complements prior sub-county analyses using OpenFEMA data.

R2.4 Comment: Data sources and years are unclear (NFHL year, ZIP map year, etc.).
Response: We added a Data Sources and Versions table with explicit years/versions, including NFHL DFIRM IDs, ZCTA 2022, and CBG 2020. We also added citations for each dataset.

R2.5 Comment (L246-248): Provide a table of matching characteristics and sources.
Response: The Data Sources and Versions table and the Methods section now list the matching attributes (ZCTA/ZIP, flood zone, BFE, construction year, assessed value) and their sources.

R2.6 Comment (L273): Clarify whether policies were placed and provide denominators and penetration.
Response: We clarify that the algorithm disaggregates claims only; policies are used for coverage context. We added policy vs claim distribution results to the main text and report candidate pool denominators and match rates: Dodge mean 892 candidates (median 872; max 8,199), 260 of 261 claims matched, 6,552 buildings with nonzero probability; Douglas mean 790 (median 839; max 8,214), 198 of 204 matched, 5,575 buildings with nonzero probability. SFHA policy penetration is approximately 0.29 to 0.30.

R2.7 Comment (L277): Elevation filter introduced late.
Response: We moved the elevation tolerance description into the Methods and clarified its role in the filtering sequence.

R2.8 Comment (L282-283): Building stock characteristics source unclear.
Response: We now specify that assessed value, classification code, parcel ZIP, and construction year come from Nebraska statewide parcel-assessor data (2023) linked to Microsoft building footprints.

R2.9 Comment (L290-291): OpenFEMA spatial attributes were not used.
Response: We document the availability of latitude/longitude and census block group fields, explain that lat/long are rounded to 0.1 degrees and often fall outside county boundaries, and report sensitivity tests. Weighted lat/long performs comparably to the baseline but does not improve calibration enough to justify inclusion.

R2.10 Comment (L296-297): Non-residential claims filtering.
Response: We explicitly state that we restrict to occupancyType=1 (single-family residential). This yields 261 claims in Dodge County and 204 in Douglas County for March 2019; non-residential claims are excluded.

R2.11 Comment (L298): ZIP code instability and census block alternatives.
Response: We switch the baseline to ZCTA (stable Census polygons) and include ZIP and CBG sensitivity results. CBGs can improve precision-recall but reduce ROC-AUC and calibration, indicating over-concentration in narrow candidate pools.

R2.12 Comment (L299-300): Flood-zone restrictions and parameter definitions are unclear.
Response: We clarified that A-subzones are collapsed to A and B is mapped to X, with BFE tolerance set to +/-0.5 ft where available and assessed values derived from parcel improvement values.

R2.13 Comment (L301): Elevation filter meaning.
Response: We clarify that the elevation filter is a conditional constraint, not a preference ordering. It is applied only when it does not eliminate all candidates.

R2.14 Comment (L308): Parameter sweep should be in Methods.
Response: We added a dedicated Sensitivity and Robustness Tests section in Methods that describes the parameter variations and spatial-unit comparisons.

R2.15 Comment (L633): Event choice, repeat claims, and denominators.
Response: We justify the event choice based on data availability and add Douglas County replication. We also report candidate denominators (above) and note that repeat-claim probability is beyond the scope of this event-specific study.

R2.16 Comment (L730): Aggregation sentence unclear.
Response: We rewrote the sentence in the Discussion to state that aggregation to coarse units obscures within-unit heterogeneity and limits building-level hotspot detection.

We hope these revisions address the reviewers' concerns and improve the clarity and contribution of the manuscript.
