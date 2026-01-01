---
manuscript: main
cycle_number: 1
source_type: actual
journal: Risk Analysis
submission_round: R&R1
decision: major_revision
reviewer_ids:
  - R1
  - R2
start_commit: null
---

# Revision Tracker: Response to Reviewer Comments

**Document**: An Approach to Probabilistic Disaggregation of NFIP Claims
**Review**: #1
**Type**: Actual (Journal Review)
**Journal**: Risk Analysis
**Round**: R&R1
**Decision**: Major Revision
**Reviewers**: R1, R2
**Last Updated**: 2026-01-01

---

## Summary Statistics

| Category | Total | Addressed | Partially Addressed | Beyond Scope | Pending |
|----------|-------|-----------|---------------------|--------------|---------|
| R1 Major Comments | 5 | 2 | 3 | 0 | 0 |
| R1 Minor Comments | 3 | 2 | 1 | 0 | 0 |
| R2 Major Comments | 4 | 3 | 1 | 0 | 0 |
| R2 Minor Comments | 12 | 11 | 1 | 0 | 0 |

---

## R1 Comments

### R1 Major 1: Contribution to Flood Risk Science

**Status**: ADDRESSED

**Reviewer's Comment**:
> While the technical approach seems innovative, the paper's contribution to the broader scholarship is weak and should be articulated better. It lacks a clear articulation of how these likelihood scores advance flood risk science or support predictive modeling. For example, given the fat-tailed nature of disaster losses, past claims are often poor predictors of future events—how does this method address that challenge? The authors should clarify the practical utility of these scores: what can future researchers or practitioners learn from them, and how might they improve flood risk management strategies?

**Validity Assessment**: VALID

The reviewer raises a legitimate concern about the practical contribution beyond the technical method.

**Response**:

Reframed the contribution as a workflow that produces building-level likelihood and uncertainty surfaces (not a new algorithm), clarified that outputs are historical spatialization rather than forecasts, and added loss-size stratification plus a fat-tail caveat. See `manuscript_quarto/index.qmd` (Abstract; Discussion `#sec-discussion`; Loss-Size Stratification `#sec-results-loss`).

**Files Modified**:
- manuscript_quarto/index.qmd (Introduction, Discussion sections)

---

### R1 Major 2: Case Selection Justification

**Status**: PARTIALLY ADDRESSED

**Reviewer's Comment**:
> The case selection from Dodge County, Nebraska also raises questions. Why was this location chosen, given that other regions experience more frequent or diverse flood events? Areas with repetitive flooding, for example parts of Texas or Louisiana, could offer richer datasets for validating the model across multiple events. This would strengthen the generalizability and robustness of the scoring method.

**Validity Assessment**: VALID

Need to provide clear justification for case selection and acknowledge generalizability limitations.

**Response**:

Expanded the analysis beyond Dodge County to include Douglas County (same 2019 event) and added a 2011 replication in Cass and Dakota Counties. The Discussion now acknowledges the single-region limitation and calls for broader multi-region validation. This is only partial because we do not yet have inundation layers or comparable validation data for other regions/events, and the explicit rationale for choosing Nebraska remains brief. See `manuscript_quarto/index.qmd` (Abstract; Results `#sec-results-ia`; Discussion `#sec-discussion`).

**Files Modified**:
- manuscript_quarto/index.qmd (Methods, Limitations sections)

---

### R1 Major 3: Claims Outside Floodplains

**Status**: PARTIALLY ADDRESSED

**Reviewer's Comment**:
> The validation relies on claims from a floodplain-heavy region, whereas many recent events (e.g., 2017 Hurricane Harvey) show that significant damage (claims) also happen outside designated floodplains. The fact that we see claims from outside of floodplains show that prior flood incidents have motivated homeowners to purchase insurance even if their homes were not required to carry one. What additional parameters beyond floodplain designation and Base Flood Elevation (BFE) could be incorporated to improve predictive accuracy in such contexts?

**Validity Assessment**: VALID

This is an important methodological point about Zone X properties and outside-SFHA claims.

**Response**:

Added slope and distance-to-SFHA filters for Zone X claims and reported their effects in sensitivity results. The Discussion also notes additional covariates (distance to water bodies, drainage infrastructure, land cover) as future work. This is only partial because outside-SFHA claims are sparse in the Nebraska sample, and we lack additional covariate data and validation layers that would allow stronger empirical testing beyond slope/distance. See `manuscript_quarto/index.qmd` (Methods `#sec-methods-sensitivity`; Results `#sec-results-topo`; Discussion `#sec-discussion`).

**Files Modified**:
- manuscript_quarto/index.qmd (Methods, Discussion sections)

---

### R1 Major 4: Intended Audience and Use Cases

**Status**: ADDRESSED

**Reviewer's Comment**:
> More broadly, the paper should clarify the intended audience and use cases for disaggregated historic probabilities. Is the goal to inform insurance pricing, vulnerability assessments, or loss forecasting? NFIP already possesses precise claim location data—how does this method add value beyond existing datasets? For instance, NFIP claims that include geographic coordinates close to actual claim sites can be requested from FEMA through FOIA and will likely provide better means for validation compared to how it is done in the paper.

**Validity Assessment**: VALID

Need to clarify target users and acknowledge FOIA data availability.

**Response**:

Clarified the intended audience (practitioners and researchers using public data) and added explicit use cases (post-event assessment, micro-hotspot identification, risk communication). The Discussion also addresses FOIA access, noting that precise coordinates can be obtained but are often impractical for time-sensitive or local analyses. See `manuscript_quarto/index.qmd` (Discussion `#sec-discussion`).

**Files Modified**:
- manuscript_quarto/index.qmd (Introduction, Discussion sections)

---

### R1 Major 5: Key Limitations

**Status**: PARTIALLY ADDRESSED

**Reviewer's Comment**:
> Finally, the paper overlooks key limitations such as outdated floodplain maps and the fact that not all homes in floodplains carry flood insurance. Since insurance is only mandated for federally backed mortgages, many vulnerable homes may be uninsured yet still sustain damage. How can we also estimate their likelihood of loss?

**Validity Assessment**: VALID

Important limitations that should be acknowledged.

**Response**:

Added limitations on NFHL map currency and quantified SFHA penetration (~0.29--0.30). Added policy vs. claim distributions and ACS uptake context, and flagged policy-uptake modeling as a future step for uninsured exposure. This is only partial because we do not have a calibrated uninsured-exposure model or parcel-level uninsured loss observations to estimate uninsured loss likelihoods directly. See `manuscript_quarto/index.qmd` (Data distribution `#sec-data-distribution`; Results `#sec-results-policies`; Discussion `#sec-discussion`).

**Files Modified**:
- manuscript_quarto/index.qmd (Limitations section)

---

### R1 Minor 1: Fat-Tailed Losses

**Status**: ADDRESSED

**Reviewer's Comment**:
> Given the fat-tailed nature of disaster losses, past claims are often poor predictors of future events.

**Response**:

Explicitly notes fat-tailed loss behavior in the Discussion and uses loss-size stratification to show upper-tail variability. See `manuscript_quarto/index.qmd` (Results `#sec-results-loss`; Discussion `#sec-discussion`).

---

### R1 Minor 2: FOIA Validation

**Status**: ADDRESSED

**Reviewer's Comment**:
> NFIP claims that include geographic coordinates close to actual claim sites can be requested from FEMA through FOIA and will likely provide better means for validation.

**Response**:

Added an explicit FOIA acknowledgment and explains why public-data workflows are needed for timely/local analyses. See `manuscript_quarto/index.qmd` (Discussion `#sec-discussion`).

---

### R1 Minor 3: Uninsured Properties

**Status**: PARTIALLY ADDRESSED

**Reviewer's Comment**:
> How can we also estimate their likelihood of loss?

**Response**:

Added penetration context and policy-uptake discussion, and flagged uninsured exposure modeling as future work. This remains partial because uninsured-loss estimation requires additional exposure data or a calibrated uptake model that we do not yet have. See `manuscript_quarto/index.qmd` (Results `#sec-results-policies`; Discussion `#sec-discussion`).

---

## R2 Comments

### R2 Major 1: Method Novelty Oversold

**Status**: ADDRESSED

**Reviewer's Comment**:
> While I thoroughly agree with the motivation for disaggregating the NFIP data onto structures, I strongly believe that the authors are overselling the novelty of the method, since it is little more than a simple matching algorithm based on property level characteristics aggregated through GIS.

**Validity Assessment**: VALID

This is a critical concern. Need to reframe the contribution more honestly.

**Response**:

Downplayed novelty and reframed the contribution as a transparent workflow with uncertainty diagnostics; the title was changed to “An Approach…” and the Discussion states the contribution is not a new algorithm. See `manuscript_quarto/index.qmd` (title block; Discussion `#sec-discussion`).

**Files Modified**:
- manuscript_quarto/index.qmd (Abstract, Introduction, Contribution statement)

---

### R2 Major 2: Single Event Generalizability

**Status**: PARTIALLY ADDRESSED

**Reviewer's Comment**:
> The study's focus on disaggregating these data for a single (small) flood event raise the question of how the results may be biased or would be different in a different location or for a different event, even though the OpenFEMA data are publicly available nationwide (and similar datasets are available elsewhere).

**Validity Assessment**: VALID

Need to acknowledge limitation and discuss generalizability.

**Response**:

Added a 2011 replication (Cass and Dakota Counties) and expanded to two counties in 2019, while explicitly noting that broader generalizability still requires more regions and flood types. This is only partial because other regions/events lack compatible inundation layers for ROC/PR evaluation, and the 2011 replication relies on ZIP-level validation only. See `manuscript_quarto/index.qmd` (Abstract; Results `#sec-results-ia`; Discussion `#sec-discussion`).

**Files Modified**:
- manuscript_quarto/index.qmd (Limitations, Future Work)

---

### R2 Major 3: Missing Wagner (2022) Comparison

**Status**: ADDRESSED

**Reviewer's Comment**:
> The study does not leverage all of the geospatial data that is made available through OpenFEMA or reference studies that have proposed similar approaches to disaggregate the OpenFEMA data and for which replication code is publicly available (e.g., Wagner 2022 https://www.aeaweb.org/articles?id=10.1257/pol.20200378).

**Validity Assessment**: VALID

Critical omission - must cite and compare to Wagner (2022).

**Response**:

Added Wagner (2022) and positioned this study as extending tract-level linking to building-level matching with uncertainty quantification. Also evaluated OpenFEMA lat/long and block-group fields in sensitivity tests and added public code availability. See `manuscript_quarto/index.qmd` (Literature review; Methods `#sec-methods-sensitivity`; Results `#sec-results-latlon`; Data/Code Availability `#sec-availability`) and `manuscript_quarto/references.bib`.

**Files Modified**:
- manuscript_quarto/index.qmd (Literature Review, Methods)
- manuscript_quarto/references.bib

---

### R2 Major 4: Data Source Documentation

**Status**: ADDRESSED

**Reviewer's Comment**:
> In terms of the clarity of presentation, I found the methods and the description of the data used difficult to follow. While the data section discusses several datasets used in the study, the source of the data is not always provided. (For example, what year of FEMA flood hazard maps were used to derive flood zones? Do the year of the zip code maps correspond to the year of the flood?)

**Validity Assessment**: VALID

Need to add data source table with years and versions.

**Response**:

Added a Data Sources and Versions table with years/versions, key fields, and notes. See `manuscript_quarto/index.qmd` (`#sec-data-sources`).

**Files Modified**:
- manuscript_quarto/index.qmd (Data section)

---

### R2 Minor 1: Table of Matching Characteristics (L246-248)

**Status**: ADDRESSED

**Reviewer's Comment**:
> Perhaps provide a table of the relevant characteristics that were used to match claims to buildings at their sources.

**Response**:

Provided a Data Sources and Versions table listing matching characteristics and their sources, and enumerated matching attributes in Methods. See `manuscript_quarto/index.qmd` (`#sec-data-sources`; `#sec-methods-framework`).

---

### R2 Minor 2: Policies vs Claims Confusion (L273)

**Status**: ADDRESSED

**Reviewer's Comment**:
> There is a reference to flood policies but earlier no discussion of whether OpenFEMA policies were also being placed, or only claims. Moreover, the authors provide no denominators for the information in section 3.3.2. How many possible buildings were eligible to receive a claim placement? Is the policy penetration within the SFHA 100%?

**Response**:

Clarified that policies are used for coverage context (not disaggregated) and added denominators: candidate-pool sizes and SFHA policy penetration. See `manuscript_quarto/index.qmd` (Data `#sec-data-distribution`; Results `#sec-results-performance`; Policies section `#sec-results-policies`).

---

### R2 Minor 3: Elevation Filter Order (L277)

**Status**: ADDRESSED

**Reviewer's Comment**:
> Line 277 refers to an elevation filter that isn't introduced until much later in the paper which is confusing.

**Response**:

Moved filter definitions (including elevation tolerance) into the main Methods narrative before Results. See `manuscript_quarto/index.qmd` (`#sec-methods-framework`).

---

### R2 Minor 4: Building Stock Characteristics Source (L282-283)

**Status**: ADDRESSED

**Reviewer's Comment**:
> The authors refer to characteristics of the building stock, but provide no information on where these characteristics are derived from or what they are.

**Response**:

Identified parcel assessor records as the source of building characteristics (construction year, assessed values). See `manuscript_quarto/index.qmd` (`#sec-data-sources`; `#sec-data-buildings`).

---

### R2 Minor 5: OpenFEMA Spatial Attributes (L290-291)

**Status**: ADDRESSED

**Reviewer's Comment**:
> The OpenFEMA claims contain additional spatial attributes, including latitude and longitude (rounded to 0.1 degrees), census block, zip code, etc. Why weren't any of these considered to reduce the potential degrees of freedom when assigning claims to structures?

**Response**:

Explained that OpenFEMA lat/long are coarse and sometimes outside county boundaries, and evaluated lat/long and block groups in sensitivity tests. See `manuscript_quarto/index.qmd` (Data `#sec-methods`; Results `#sec-results-latlon`; `#sec-results-spatial`).

---

### R2 Minor 6: Non-Residential Claims Filter (L296-297)

**Status**: ADDRESSED

**Reviewer's Comment**:
> Claims can also be on non-residential buildings (I believe there is an agricultural flag in the dataset). Did you pre-filter these out of the dataset? If so, how many claims remained.

**Response**:

Specified occupancyType=1 filtering and reported remaining claim counts for both counties. See `manuscript_quarto/index.qmd` (Data `#sec-methods`).

---

### R2 Minor 7: ZIP Code Temporal Issues (L298)

**Status**: ADDRESSED

**Reviewer's Comment**:
> Zip codes can be problematic as they change frequently. When considering locations and events beyond the example herein, how might this pose a challenge for the algorithm? How would this be different if using census block groups as the spatial unit to disaggregate from?

**Response**:

Discussed ZCTA vs ZIP stability and noted 2022 ZCTA geometry mismatch; compared ZIP/ZCTA/CBG in sensitivity results. See `manuscript_quarto/index.qmd` (Results `#sec-results-spatial`; Discussion `#sec-discussion`).

---

### R2 Minor 8: Flood Zone Restrictions Clarity (L299-300)

**Status**: ADDRESSED

**Reviewer's Comment**:
> What is meant by flood zone restrictions? Which building characteristics were considered, what are elevation tolerances and how were property values derived? Not enough information is provided about these choices and why they were made.

**Response**:

Defined flood-zone handling (A-subzones collapsed; legacy B to X) and listed optional filters with tolerances and value-source notes. See `manuscript_quarto/index.qmd` (`#sec-methods-framework`).

---

### R2 Minor 9: Elevation Filter Meaning (L301)

**Status**: ADDRESSED

**Reviewer's Comment**:
> Is the idea here the NFIP claims are preferentially assigned to lower elevation building matches before higher elevation building matches?

**Response**:

Clarified that filtering constrains candidate pools and that bootstrap draws are uniform (no preferential assignment by elevation). See `manuscript_quarto/index.qmd` (`#sec-methods-bootstrap`).

---

### R2 Minor 10: Parameter Sweep Location (L308)

**Status**: ADDRESSED

**Reviewer's Comment**:
> "the parameter sweep described later in the Results section" Shouldn't this be described in the methods section?

**Response**:

Moved the parameter-sweep/sensitivity description into Methods and referenced it before Results. See `manuscript_quarto/index.qmd` (`#sec-methods-sensitivity`).

---

### R2 Minor 11: Event Selection Justification (L633)

**Status**: PARTIALLY ADDRESSED

**Reviewer's Comment**:
> Why was this event chosen? How do the performance metrics look during a larger or a smaller event in the same area? Could one assume that there is a higher probability of a repeat claim if a similar event hits the same area? Also, how does this number relate to the total number of buildings where claims could be assigned?

**Response**:

Added candidate-pool counts and nonzero-building totals, and included a 2011 replication to give multi-event context. This is only partial because we lack repeated-event claim identifiers and comparable validation layers to test repeat-claim dynamics or event-size effects, and the event-selection rationale is not yet explicit. See `manuscript_quarto/index.qmd` (Results `#sec-results-performance`; `#sec-results-ia`; Discussion `#sec-discussion`).

---

### R2 Minor 12: Aggregation Clarity (L730)

**Status**: ADDRESSED

**Reviewer's Comment**:
> "The aggregation of those data has limited granular flood risk analysis" The meaning of this sentence is not clear and is important to the content of the paper. It should be elaborated.

**Response**:

Expanded the abstract/introduction language on how aggregation limits granular risk analysis and motivated building-level disaggregation. See `manuscript_quarto/index.qmd` (Abstract; Introduction `#sec-introduction`).

---

## Verification Checklist

- [ ] All VALID - ACTION NEEDED items addressed
- [x] Wagner (2022) cited and compared
- [x] Data source table added
- [x] Contribution reframed appropriately
- [ ] All code runs without errors
- [x] Manuscript text updated
- [ ] Tables/figures reflect changes
- [ ] Quarto renders without errors
- [ ] Changes committed to git
- [ ] Response letter generated

---

*Review initialized: 2025-12-31*
*Last updated: 2026-01-01*
