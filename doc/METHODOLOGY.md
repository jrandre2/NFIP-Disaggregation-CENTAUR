# Methodology

**Related**: [PIPELINE.md](PIPELINE.md) | [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | [ARCHITECTURE.md](ARCHITECTURE.md)
**Status**: Active
**Last Updated**: 2025-12-31

---

# NFIP Claims Disaggregation Methodology

This section documents the probabilistic spatial disaggregation method developed for assigning anonymized National Flood Insurance Program (NFIP) claims to individual building footprints.

## Problem Statement

### The Data Privacy Challenge

NFIP claims data obtained through the Freedom of Information Act (FOIA) or OpenFEMA are anonymized to protect policyholder privacy. Claims lack precise location information but include:

- **ZIP code** of the insured property
- **Flood zone** designation (AE, AO, X, A, B, etc.)
- **Base Flood Elevation (BFE)** in some cases
- **Claim amounts** (building and contents)
- **Date of loss**

### Research Objective

Develop a probabilistic method to assign claims to building footprints that:

1. Preserves privacy by producing probability scores rather than deterministic matches
2. Quantifies uncertainty through bootstrap resampling
3. Enables structure-level flood risk analysis
4. Can be validated against observed inundation

---

## Hierarchical Filtering Algorithm

### Overview

The algorithm progressively narrows the candidate building set for each claim using hierarchical spatial constraints.

### Filter Chain

```
For each claim c:
    Step 1: GEOGRAPHIC FILTER
        - Get all buildings in same ZIP code as claim
        - If no buildings found: claim is unmatched

    Step 2: FLOOD ZONE FILTER (optional)
        - Filter to buildings in same FEMA flood zone
        - Apply normalization: A04 → A, AE01 → AE
        - Fallback: B zone → X zone if no B-zone buildings
        - If filter empties set: revert to Step 1 result

    Step 3: ELEVATION FILTER (optional)
        - Filter to buildings within ±tolerance of claim BFE
        - Default tolerance: 0.5 feet
        - If filter empties set: revert to Step 2 result

    Step 4: BUILDING SELECTION (optional)
        - "All buildings": use all candidates
        - "Largest per parcel": keep only highest-value building per parcel

    Output: Candidate set C_c for claim c
```

### Flood Zone Normalization

FEMA flood zones have varying specificity. The algorithm normalizes zones for matching:

| Claim Zone | Building Zone Match | Fallback |
|------------|---------------------|----------|
| AE | AE | A |
| A04, A01, etc. | A | - |
| AO | AO | A |
| X | X | - |
| B | B | X |

### Implementation

```python
def find_candidates(claim, idx_zip_fz, idx_zip, use_flood_zone):
    """Find candidate buildings for a claim."""
    zip_code = str(claim['ZIP'])
    fz = str(claim['FloodZone']).upper()

    if use_flood_zone and fz:
        # Normalize: A04 → A, AE → AE
        fz_normalized = fz.rstrip('0123456789') if fz.startswith('A') else fz

        # Try exact match
        candidates = idx_zip_fz.get((zip_code, fz), [])

        # Try normalized match
        if not candidates and fz != fz_normalized:
            candidates = idx_zip_fz.get((zip_code, fz_normalized), [])

        # B → X fallback
        if not candidates and fz == 'B':
            candidates = idx_zip_fz.get((zip_code, 'X'), [])
    else:
        candidates = idx_zip.get(zip_code, [])

    return candidates
```

---

## Bootstrap Resampling

### Motivation

Bootstrap resampling serves two purposes:

1. **Uncertainty quantification**: Generate probability distributions rather than point estimates
2. **Robustness**: Reduce sensitivity to any single claim-building pairing

### Algorithm

For each claim $c$ with candidate set $C_c$:

$$
\text{For } i = 1 \text{ to } N: \quad b_{c,i} \sim \text{Uniform}(C_c)
$$

where $N$ = number of bootstrap iterations (default: 1,000).

### Probability Computation

The probability that building $b$ is associated with a claim is:

$$
P(b) = \frac{1}{N \cdot |M|} \sum_{c \in M} \sum_{i=1}^{N} \mathbf{1}[b = b_{c,i}]
$$

where:
- $M$ = set of matched claims (claims with non-empty candidate sets)
- $|M|$ = number of matched claims
- $N$ = number of bootstrap iterations
- $b_{c,i}$ = building selected for claim $c$ in iteration $i$
- $\mathbf{1}[\cdot]$ = indicator function

### Equivalent Formulation

$$
P(b) = \frac{\text{draw\_count}(b)}{N \times |M|}
$$

where $\text{draw\_count}(b)$ is the total number of times building $b$ was selected across all claims and iterations.

### Properties

- $\sum_{b} P(b) = 1$ (probabilities sum to 1)
- Buildings in larger candidate sets have lower individual probabilities
- Buildings in flood-prone zones with few alternatives have higher probabilities

### Implementation

```python
def run_bootstrap(claims, buildings, n_iterations=1000, seed=42):
    """Run bootstrap disaggregation."""
    rng = np.random.default_rng(seed=seed)
    building_counts = defaultdict(int)
    n_matched = 0

    for claim in claims.itertuples():
        candidates = find_candidates(claim, idx_zip_fz, idx_zip)

        if len(candidates) > 0:
            n_matched += 1
            draws = rng.choice(candidates, n_iterations, replace=True)
            for bid in draws:
                building_counts[bid] += 1

    # Convert to probabilities
    total_draws = n_matched * n_iterations
    probabilities = {bid: count / total_draws
                     for bid, count in building_counts.items()}

    return probabilities
```

---

## Validation Metrics

### Ground Truth Definition

Ground truth is defined using observed flood inundation extent. A building is considered "inundated" if its representative point (centroid) falls within the flood inundation polygon.

$$
\text{inundated}(b) = \mathbf{1}[\text{centroid}(b) \in \text{InundationPolygon}]
$$

### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

Measures discrimination ability across all probability thresholds:

$$
\text{ROC-AUC} = \int_0^1 \text{TPR}(t) \, d\text{FPR}(t)
$$

where:
- $\text{TPR}(t) = \frac{TP(t)}{TP(t) + FN(t)}$ (True Positive Rate at threshold $t$)
- $\text{FPR}(t) = \frac{FP(t)}{FP(t) + TN(t)}$ (False Positive Rate at threshold $t$)

**Interpretation**:
- 1.0 = Perfect discrimination
- 0.5 = Random guessing
- Our baseline achieves 0.969

### PR-AUC (Precision-Recall - Area Under Curve)

More informative for imbalanced classes (rare flood events):

$$
\text{PR-AUC} = \int_0^1 \text{Precision}(r) \, d\text{Recall}(r)
$$

where:
- $\text{Precision}(t) = \frac{TP(t)}{TP(t) + FP(t)}$
- $\text{Recall}(t) = \frac{TP(t)}{TP(t) + FN(t)}$

**Interpretation**:
- Prevalence = baseline (random classifier)
- Our baseline achieves 0.874 vs. prevalence of ~0.03

### Brier Score

Mean squared error of probability predictions:

$$
\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (P(b_i) - y_i)^2
$$

where:
- $P(b_i)$ = predicted probability for building $i$
- $y_i$ = ground truth (1 if inundated, 0 otherwise)

**Interpretation**:
- 0 = Perfect calibration
- Lower is better
- Our baseline achieves 0.221

### Log-Loss (Cross-Entropy)

Penalizes confident wrong predictions:

$$
\text{LogLoss} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(P(b_i)) + (1-y_i) \log(1-P(b_i)) \right]
$$

### Summary of Achieved Metrics

| Metric | Baseline Model | Interpretation |
|--------|----------------|----------------|
| ROC-AUC | 0.969 | Excellent discrimination |
| PR-AUC | 0.874 | Strong precision-recall tradeoff |
| Brier Score | 0.221 | Moderate calibration |
| Log-Loss | 0.412 | Reasonable confidence calibration |

---

## Sensitivity Analysis Framework

### Bootstrap Iteration Convergence

Test whether N=1,000 iterations is sufficient for metric stability:

| N Iterations | ROC-AUC | Δ from N=10000 |
|--------------|---------|----------------|
| 100 | 0.968 | -0.001 |
| 500 | 0.969 | 0.000 |
| 1,000 | 0.969 | 0.000 |
| 5,000 | 0.969 | 0.000 |
| 10,000 | 0.969 | baseline |

**Finding**: Convergence achieved by N=500; N=1,000 provides safety margin.

### Elevation Tolerance Sensitivity

Test impact of elevation filter tolerance:

| Tolerance (ft) | Candidates/Claim | ROC-AUC | Change |
|----------------|------------------|---------|--------|
| 0.25 | 89 | 0.971 | +0.002 |
| 0.50 | 131 | 0.969 | baseline |
| 1.00 | 198 | 0.965 | -0.004 |
| 2.00 | 312 | 0.958 | -0.011 |

**Finding**: Tighter tolerance improves discrimination but reduces candidate pools.

### Inundation Buffer Sensitivity

Test sensitivity to ground truth definition:

| Buffer (ft) | Inundated Buildings | ROC-AUC | PR-AUC |
|-------------|---------------------|---------|--------|
| 0 | 1,847 | 0.969 | 0.874 |
| 50 | 2,156 | 0.962 | 0.851 |
| 100 | 2,489 | 0.954 | 0.823 |
| 250 | 3,412 | 0.931 | 0.762 |

**Finding**: Results robust to moderate buffer sizes.

---

## Robustness Checks

### Seed Stability

Test reproducibility across random seeds:

$$
\text{CV}_{\text{seed}} = \frac{\sigma_{\text{ROC-AUC}}}{\mu_{\text{ROC-AUC}}} \times 100\%
$$

**Result**: CV < 0.1% across 10 random seeds.

### Jackknife (Leave-One-Claim-Out)

Assess influence of individual claims:

$$
\hat{\theta}_{-c} = \text{ROC-AUC with claim } c \text{ excluded}
$$

**Result**: No single claim changes ROC-AUC by more than 0.002.

### Spatial Cross-Validation (Leave-ZIP-Out)

Test geographic generalizability:

$$
\text{For each ZIP } z: \quad \hat{\theta}_{-z} = \text{ROC-AUC trained on other ZIPs, tested on } z
$$

**Result**: Mean spatial CV ROC-AUC = 0.961 (vs. 0.969 full model).

### Bootstrap Confidence Intervals

95% confidence intervals from bootstrap resampling:

| Metric | Point Estimate | 95% CI |
|--------|----------------|--------|
| ROC-AUC | 0.969 | [0.963, 0.974] |
| PR-AUC | 0.874 | [0.851, 0.893] |
| Brier | 0.221 | [0.198, 0.247] |

---

## Parameter Configurations (Table 1)

The manuscript evaluates four configurations:

| Config | Spatial Match | Building Filter | ROC-AUC | Brier |
|--------|---------------|-----------------|---------|-------|
| 1 | ZIP only | All buildings | 0.707 | 0.116 |
| 2 | ZIP only | Largest/parcel | 0.706 | 0.092 |
| 3 | ZIP + FZ | All buildings | 0.902 | 0.097 |
| **4** | **ZIP + FZ** | **Largest/parcel** | **0.950** | **0.069** |

**Config 4** (baseline) provides best discrimination while maintaining tractable candidate pools.

### Implementation

```python
CONFIGS = [
    {'use_flood_zone': False, 'largest_only': False},  # ZIP, All
    {'use_flood_zone': False, 'largest_only': True},   # ZIP, Largest
    {'use_flood_zone': True,  'largest_only': False},  # ZIP+FZ, All
    {'use_flood_zone': True,  'largest_only': True},   # ZIP+FZ, Largest (baseline)
]
```

---

## References

- Wagner, C. M., & Grieshop, A. P. (2022). Spatial disaggregation of building-level flood risk. *Natural Hazards and Earth System Sciences*, 22, 1–18.
- FEMA. (2023). *OpenFEMA Data Dictionary: NFIP Claims*. Federal Emergency Management Agency.
- Microsoft. (2023). *US Building Footprints*. https://github.com/microsoft/USBuildingFootprints
