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
**Last Updated**: 2025-12-31

---

## Summary Statistics

| Category | Total | Addressed | Beyond Scope | Pending |
|----------|-------|-----------|--------------|---------|
| R1 Major Comments | 5 | 0 | 0 | 5 |
| R1 Minor Comments | 3 | 0 | 0 | 3 |
| R2 Major Comments | 4 | 0 | 0 | 4 |
| R2 Minor Comments | 12 | 0 | 0 | 12 |

---

## R1 Comments

### R1 Major 1: Contribution to Flood Risk Science

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> While the technical approach seems innovative, the paper's contribution to the broader scholarship is weak and should be articulated better. It lacks a clear articulation of how these likelihood scores advance flood risk science or support predictive modeling. For example, given the fat-tailed nature of disaster losses, past claims are often poor predictors of future events—how does this method address that challenge? The authors should clarify the practical utility of these scores: what can future researchers or practitioners learn from them, and how might they improve flood risk management strategies?

**Validity Assessment**: VALID

The reviewer raises a legitimate concern about the practical contribution beyond the technical method.

**Response**:

[To be completed - need to strengthen the discussion of practical applications and how scores can inform risk management despite fat-tailed loss distributions]

**Files Modified**:
- manuscript_quarto/index.qmd (Introduction, Discussion sections)

---

### R1 Major 2: Case Selection Justification

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> The case selection from Dodge County, Nebraska also raises questions. Why was this location chosen, given that other regions experience more frequent or diverse flood events? Areas with repetitive flooding, for example parts of Texas or Louisiana, could offer richer datasets for validating the model across multiple events. This would strengthen the generalizability and robustness of the scoring method.

**Validity Assessment**: VALID

Need to provide clear justification for case selection and acknowledge generalizability limitations.

**Response**:

[To be completed - explain data availability, project funding context, and plans for multi-event validation]

**Files Modified**:
- manuscript_quarto/index.qmd (Methods, Limitations sections)

---

### R1 Major 3: Claims Outside Floodplains

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> The validation relies on claims from a floodplain-heavy region, whereas many recent events (e.g., 2017 Hurricane Harvey) show that significant damage (claims) also happen outside designated floodplains. The fact that we see claims from outside of floodplains show that prior flood incidents have motivated homeowners to purchase insurance even if their homes were not required to carry one. What additional parameters beyond floodplain designation and Base Flood Elevation (BFE) could be incorporated to improve predictive accuracy in such contexts?

**Validity Assessment**: VALID

This is an important methodological point about Zone X properties and outside-SFHA claims.

**Response**:

[To be completed - discuss potential additional parameters like distance to water bodies, historical inundation, topographic indices]

**Files Modified**:
- manuscript_quarto/index.qmd (Methods, Discussion sections)

---

### R1 Major 4: Intended Audience and Use Cases

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> More broadly, the paper should clarify the intended audience and use cases for disaggregated historic probabilities. Is the goal to inform insurance pricing, vulnerability assessments, or loss forecasting? NFIP already possesses precise claim location data—how does this method add value beyond existing datasets? For instance, NFIP claims that include geographic coordinates close to actual claim sites can be requested from FEMA through FOIA and will likely provide better means for validation compared to how it is done in the paper.

**Validity Assessment**: VALID

Need to clarify target users and acknowledge FOIA data availability.

**Response**:

[To be completed - clarify that method is for researchers/planners who cannot access FOIA data; discuss use cases]

**Files Modified**:
- manuscript_quarto/index.qmd (Introduction, Discussion sections)

---

### R1 Major 5: Key Limitations

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> Finally, the paper overlooks key limitations such as outdated floodplain maps and the fact that not all homes in floodplains carry flood insurance. Since insurance is only mandated for federally backed mortgages, many vulnerable homes may be uninsured yet still sustain damage. How can we also estimate their likelihood of loss?

**Validity Assessment**: VALID

Important limitations that should be acknowledged.

**Response**:

[To be completed - add limitations section addressing map currency and insurance penetration]

**Files Modified**:
- manuscript_quarto/index.qmd (Limitations section)

---

### R1 Minor 1: Fat-Tailed Losses

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> Given the fat-tailed nature of disaster losses, past claims are often poor predictors of future events.

**Response**:

[Address in response to R1 Major 1]

---

### R1 Minor 2: FOIA Validation

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> NFIP claims that include geographic coordinates close to actual claim sites can be requested from FEMA through FOIA and will likely provide better means for validation.

**Response**:

[Acknowledge FOIA option; explain why remote sensing validation was chosen]

---

### R1 Minor 3: Uninsured Properties

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> How can we also estimate their likelihood of loss?

**Response**:

[Discuss as future work - method focuses on insured properties only]

---

## R2 Comments

### R2 Major 1: Method Novelty Oversold

**Status**: VALID - ACTION NEEDED (CRITICAL)

**Reviewer's Comment**:
> While I thoroughly agree with the motivation for disaggregating the NFIP data onto structures, I strongly believe that the authors are overselling the novelty of the method, since it is little more than a simple matching algorithm based on property level characteristics aggregated through GIS.

**Validity Assessment**: VALID

This is a critical concern. Need to reframe the contribution more honestly.

**Response**:

[To be completed - reframe contribution around uncertainty quantification and building-level scores rather than algorithmic novelty]

**Files Modified**:
- manuscript_quarto/index.qmd (Abstract, Introduction, Contribution statement)

---

### R2 Major 2: Single Event Generalizability

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> The study's focus on disaggregating these data for a single (small) flood event raise the question of how the results may be biased or would be different in a different location or for a different event, even though the OpenFEMA data are publicly available nationwide (and similar datasets are available elsewhere).

**Validity Assessment**: VALID

Need to acknowledge limitation and discuss generalizability.

**Response**:

[To be completed - acknowledge limitation, discuss plans for multi-event validation]

**Files Modified**:
- manuscript_quarto/index.qmd (Limitations, Future Work)

---

### R2 Major 3: Missing Wagner (2022) Comparison

**Status**: VALID - ACTION NEEDED (CRITICAL)

**Reviewer's Comment**:
> The study does not leverage all of the geospatial data that is made available through OpenFEMA or reference studies that have proposed similar approaches to disaggregate the OpenFEMA data and for which replication code is publicly available (e.g., Wagner 2022 https://www.aeaweb.org/articles?id=10.1257/pol.20200378).

**Validity Assessment**: VALID

Critical omission - must cite and compare to Wagner (2022).

**Response**:

[To be completed - add Wagner citation, compare approaches, explain differences]

**Files Modified**:
- manuscript_quarto/index.qmd (Literature Review, Methods)
- manuscript_quarto/references.bib

---

### R2 Major 4: Data Source Documentation

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> In terms of the clarity of presentation, I found the methods and the description of the data used difficult to follow. While the data section discusses several datasets used in the study, the source of the data is not always provided. (For example, what year of FEMA flood hazard maps were used to derive flood zones? Do the year of the zip code maps correspond to the year of the flood?)

**Validity Assessment**: VALID

Need to add data source table with years and versions.

**Response**:

[To be completed - create comprehensive data source table]

**Files Modified**:
- manuscript_quarto/index.qmd (Data section)

---

### R2 Minor 1: Table of Matching Characteristics (L246-248)

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> Perhaps provide a table of the relevant characteristics that were used to match claims to buildings at their sources.

**Response**:

[Add table listing all matching attributes and their sources]

---

### R2 Minor 2: Policies vs Claims Confusion (L273)

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> There is a reference to flood policies but earlier no discussion of whether OpenFEMA policies were also being placed, or only claims. Moreover, the authors provide no denominators for the information in section 3.3.2. How many possible buildings were eligible to receive a claim placement? Is the policy penetration within the SFHA 100%?

**Response**:

[Clarify terminology; add denominator information]

---

### R2 Minor 3: Elevation Filter Order (L277)

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> Line 277 refers to an elevation filter that isn't introduced until much later in the paper which is confusing.

**Response**:

[Reorder methods section for logical flow]

---

### R2 Minor 4: Building Stock Characteristics Source (L282-283)

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> The authors refer to characteristics of the building stock, but provide no information on where these characteristics are derived from or what they are.

**Response**:

[Add source information for parcel/assessor data]

---

### R2 Minor 5: OpenFEMA Spatial Attributes (L290-291)

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> The OpenFEMA claims contain additional spatial attributes, including latitude and longitude (rounded to 0.1 degrees), census block, zip code, etc. Why weren't any of these considered to reduce the potential degrees of freedom when assigning claims to structures?

**Response**:

[Explain why census block wasn't used; discuss lat/lon rounding limitations]

---

### R2 Minor 6: Non-Residential Claims Filter (L296-297)

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> Claims can also be on non-residential buildings (I believe there is an agricultural flag in the dataset). Did you pre-filter these out of the dataset? If so, how many claims remained.

**Response**:

[Clarify filtering; add counts before/after filtering]

---

### R2 Minor 7: ZIP Code Temporal Issues (L298)

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> Zip codes can be problematic as they change frequently. When considering locations and events beyond the example herein, how might this pose a challenge for the algorithm? How would this be different if using census block groups as the spatial unit to disaggregate from?

**Response**:

[Discuss ZIP code stability; address census block alternative]

---

### R2 Minor 8: Flood Zone Restrictions Clarity (L299-300)

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> What is meant by flood zone restrictions? Which building characteristics were considered, what are elevation tolerances and how were property values derived? Not enough information is provided about these choices and why they were made.

**Response**:

[Expand methods with explicit parameter definitions]

---

### R2 Minor 9: Elevation Filter Meaning (L301)

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> Is the idea here the NFIP claims are preferentially assigned to lower elevation building matches before higher elevation building matches?

**Response**:

[Clarify elevation filter logic]

---

### R2 Minor 10: Parameter Sweep Location (L308)

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> "the parameter sweep described later in the Results section" Shouldn't this be described in the methods section?

**Response**:

[Move parameter sweep description to Methods]

---

### R2 Minor 11: Event Selection Justification (L633)

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> Why was this event chosen? How do the performance metrics look during a larger or a smaller event in the same area? Could one assume that there is a higher probability of a repeat claim if a similar event hits the same area? Also, how does this number relate to the total number of buildings where claims could be assigned?

**Response**:

[Justify event selection; discuss repeatability]

---

### R2 Minor 12: Aggregation Clarity (L730)

**Status**: VALID - ACTION NEEDED

**Reviewer's Comment**:
> "The aggregation of those data has limited granular flood risk analysis" The meaning of this sentence is not clear and is important to the content of the paper. It should be elaborated.

**Response**:

[Clarify sentence about aggregation limitations]

---

## Verification Checklist

- [ ] All VALID - ACTION NEEDED items addressed
- [ ] Wagner (2022) cited and compared
- [ ] Data source table added
- [ ] Contribution reframed appropriately
- [ ] All code runs without errors
- [ ] Manuscript text updated
- [ ] Tables/figures reflect changes
- [ ] Quarto renders without errors
- [ ] Changes committed to git
- [ ] Response letter generated

---

*Review initialized: 2025-12-31*
*Last updated: 2025-12-31*
