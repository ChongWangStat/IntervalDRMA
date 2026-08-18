# IntervalDRMA

Reproducibility materials for **Contrast-Level Aggregate-Data Dose--Response
Meta-Analysis for Heterogeneous Exposure Intervals**.

This repository is intentionally limited to the code, data, machine-readable
results, and generated figures needed to reproduce and validate the analyses.
Manuscript and journal-submission files are maintained separately.

The method represents every exposure support as an interval with nonnegative
width. Positive-width intervals are represented by conditional means under a
working exposure distribution. Exact exposure values are zero-width intervals;
their representative means are evaluated directly at the reported values.
The conditional mean is the unique constant minimizing mean squared error for
the exposure within a support, and it is coherent when supports are split or
combined using their exposure probabilities. The model is a support-based
extension of conventional point-dose DRMA that contains conventional
contrast-design DRMA as an exact special case: when every support width is zero,
the support mapping is the identity under the same basis, covariance, and
random-effects specification. Midpoint analysis
is the second nested special case: it is recovered exactly when exposure is
Uniform within each bounded interval.

## Repository contents

- `data/analytic_contrasts.xlsx`: the public, analysis-ready example dataset
  containing the 33 contrast-level observations used in the methodological
  illustration, with a data dictionary.
- `data/radon_lung_cancer_contrasts.xlsx`: a source-linked dataset of 56
  adjusted residential-radon--lung-cancer OR contrasts from 16 studies,
  together with historical and current country-specific distribution inputs.
- `code/run_analysis.py`: primary eight-study AFO working analysis, OR-only and
  explicit-support restrictions, midpoint and open-tail comparisons, 30
  distribution/correlation sensitivity analyses (including a registry-source
  sensitivity),
  decorrelated-residual diagnostics, and a support-mapping fragility analysis,
  with a studentized parametric-bootstrap interval for the primary fit.
- `code/radon_example_analysis.py`: independent analysis using
  outcome-independent country-specific Lognormal residential-radon
  distributions, including CR2, one-contrast-per-study, distribution-source,
  correlation, and open-tail point-score sensitivities.
- `code/interval_drma_core.py`: unit-equivariant core GLS/profile-REML
  estimator with an exact heterogeneity boundary and adaptive optimization.
- `code/interval_drma.py`: one-call second-stage fitting interface for
  analysis-ready support means and adjusted log ORs.
- `code/mixed_support_simulation.py`: mixed-support simulation study, including
  study-count experiments at K = 10, 20, and 30.
- `code/make_score_assumption_figure.py`: direct visual benchmark of the
  distribution-based mappings and conventional assigned-score analyses.
- `code/make_distance_density_figure.py`: vector and high-resolution versions
  of the regional Gamma working-distribution figure.
- `code/make_support_to_design_figure.py`: vector support-to-design framework
  diagram.
- `code/make_support_fragility_figure.py`: local support-mapping fragility
  diagnostic.
- `code/support_uncertainty.py`: numerical delta-method and parametric-bootstrap
  utilities for propagating first-stage support-distribution uncertainty.
- `code/test_intervaldrma.py`: 20 deterministic checks of unit rescaling,
  exact and extreme-tail supports, boundary heterogeneity, ordering, the
  zero-width rule, and primary point estimates, including agreement with a
  separately constructed conventional DRMA fit when all supports are points,
  and exact agreement with midpoint DRMA under Uniform interval exposure.
- `code/reproduce_all.py`: one-command reproduction of tests, analyses,
  simulations, figures, and a machine-readable environment manifest.
- `results/`: machine-readable analysis, simulation, and environment results,
  plus reproducibly generated figures in `results/figures/`.

## Reproduce

Python 3.13.9 was used for the archived results. From the repository root:

```text
python -m pip install -r code/requirements.txt
python code/reproduce_all.py
```

The equivalent individual commands are:

```text
python code/test_intervaldrma.py
python code/run_analysis.py --output results/analysis_results.json
python code/radon_example_analysis.py --output results/radon_example_results.json
python code/mixed_support_simulation.py --replicates 5000 --output results/mixed_support_simulation_results.json
python code/make_score_assumption_figure.py
python code/make_distance_density_figure.py
python code/make_support_to_design_figure.py
python code/make_support_fragility_figure.py --results results/analysis_results.json --output results/figures/Support_mapping_fragility.pdf
```

The full simulation uses fixed random-number seeds and may take several minutes.
It reports bias, root mean squared error (RMSE), confidence-interval coverage,
and type-I error under dedicated null-effect settings and resampled empirical
study-block configurations. The analysis outputs
also provide a study-count Student-t sensitivity interval for random-effects
slopes and a studentized parametric-bootstrap interval when few independent
study blocks are available. These are labeled sensitivity intervals: the
bootstrap's repeated-sampling coverage has not been validated for the present
small-study design.
The core estimator standardizes design values internally, evaluates the exact
`tau = 0` boundary, expands its dimensionless upper bracket adaptively, and
reports convergence and boundary diagnostics before transforming results back
to the requested exposure units.
The application standard errors are conditional on the fitted support
distribution. The fitting scripts requested numerical Hessians, but the
observation-level distance files and Hessian matrices were not retained in the
reproducibility bundle, so a defensible first-stage covariance cannot be
reconstructed from the stored point estimates. The code includes the general
unconditional covariance and bootstrap procedures for use when a first-stage
covariance is retained. The profiled REML criterion in the machine-readable
outputs is an optimization diagnostic within a fixed design mapping and must
not be compared across mappings that change the fixed-effect design vector.
Both empirical-analysis result files also report leave-one-study-out refits for
influence assessment.

## Data provenance

The workbook contains the subset of reported association estimates selected
from a co-author-curated living systematic-review dataset for the illustrative
analysis. The workbook retains the extracted values and recorded support
bounds. The code applies the source-based exact-distance interpretation to the
five Schultz comparisons. The `Type` field is retained because it identifies
the support-mapping rule used by the analysis: 1 denotes an explicit
comparison/reference pair and 2 denotes a binary-buffer proxy.

The primary AFO working analysis contains all 33 mapped contrasts from eight
studies: 29 reported ORs and four Simões animal-count hazard ratios treated as
approximate ORs on the log-OR analysis scale. An OR-only restriction removes
the four hazard-ratio proxies. A strict explicit-support OR restriction retains
15 reported ORs from four studies, including five exact-distance Schultz spline
contrasts, and removes the binary-buffer and animal-count proxy mappings. The
independent radon dataset contains 56 reported adjusted OR contrasts.

The first-stage regional site layers are not interchangeable. North Carolina
and Wisconsin use permitted-facility sources. The Pennsylvania fit used a
public agricultural-producer directory, and the Netherlands fit used
livestock-keyword-filtered agricultural businesses in Eijsden-Margraten; these
two are geospatial proxies rather than verified AFO permit registries. The
analysis therefore includes a source-quality sensitivity that replaces their
Gamma parameters with the pooled North Carolina/Wisconsin permitted-facility
working distribution.

The independent radon workbook reproduces the time-weighted exposure supports,
adjusted ORs, and confidence limits in Table 2 of Pavia et al. (2003):
https://doi.org/10.1590/S0042-96862003001000008. Country-specific Lognormal
parameters are taken from UNSCEAR 2000, the WHO 2009 handbook, and UNSCEAR 2024.
The estimator uses no exposure-category frequencies, event counts, or
denominators. The cumulative-exposure-only Letourneau study is excluded because
its scale is incompatible with the common time-weighted Bq/m^3 metric.

## Citation and archiving

Repository URL: https://github.com/ChongWangStat/IntervalDRMA

Versioned Zenodo archive: https://doi.org/10.5281/zenodo.21984414

## License

The MIT License applies to the software. The data files contain factual
numerical information extracted from the cited publications and independently
organized and processed by the authors. No source-article text, figures, or
table images are reproduced.
