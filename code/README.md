# Mixed-support simulation

This directory contains the primary-analysis and simulation programs used for
the IntervalDRMA study. The code uses the observed mixture of 28 positive-width
and five zero-width exposure-support comparisons.

## Inputs

Place the analytic workbook at:

`../data/analytic_contrasts.xlsx`

Alternatively, supply another location with `--data-file`. The scripts read
the `Analytic contrasts` worksheet and expect the documented column names.

The independent nonuniform residential-radon example uses:

`../data/radon_lung_cancer_contrasts.xlsx`

## Environment

Create a clean Python 3.13.9 environment and install the recorded dependencies:

```text
python -m pip install -r requirements.txt
```

From the repository root, `python code/reproduce_all.py` runs the complete test,
analysis, simulation, figure, and environment-manifest workflow.

## One-call fitting interface

After calculating outcome-independent means for the comparison and reference
supports, fit the second-stage model with aligned row-level arrays:

```python
from interval_drma import fit_interval_drma

fit = fit_interval_drma(
    log_effect=log_ors,
    standard_error=standard_errors,
    study_id=study_ids,
    comparison_mean=comparison_support_means,
    reference_mean=reference_support_means,
    rho=0.5,
    bootstrap_replicates=2000,
)
```

Exact exposures enter as their reported values. For intervals, the means must
come from an outcome-independent empirical, grouped-data, or external exposure
distribution; midpoint inputs should be identified as a Uniform working-model
analysis.

## Reproduce the primary analysis

From the repository root, run:

```text
python code/run_analysis.py --output results/analysis_results.json
```

This reproduces the primary eight-study working analysis of all 33 mapped
contrasts, the OR-only and explicit-support restrictions, the midpoint and
infinite-tail comparisons, and the 30 distribution/correlation sensitivity settings
(including the NC/Wisconsin permitted-facility source proxy), descriptive
decorrelated-residual diagnostics, and the local support-mapping fragility
diagnostic.

## Reproduce the nonuniform radon illustration

From the repository root, run:

```text
python code/radon_example_analysis.py --output results/radon_example_results.json
```

This analyzes 56 adjusted OR contrasts from 16 studies in the published Pavia
et al. residential-radon meta-analysis. Outcome-independent country-specific
Lognormal parameters come from UNSCEAR/WHO residential surveys. The script
performs contrast-level GLS/profile REML, working-correlation and
distribution-source sensitivity analyses, a studentized parametric-bootstrap
interval, a CR2 cluster-sandwich sensitivity, a one-contrast-per-study
sensitivity, and conventional midpoint/open-tail point-score comparisons. No
exposure-category frequencies or outcome counts are used.
Both application scripts also write leave-one-study-out random-effects refits
to their machine-readable result files.

To regenerate the direct score-assumption comparison figure, run:

```text
python code/make_score_assumption_figure.py
```

To regenerate the vector regional-density figure, run:

```text
python code/make_distance_density_figure.py
```

To regenerate the vector framework and support-fragility figures, run:

```text
python code/make_support_to_design_figure.py
python code/make_support_fragility_figure.py --results results/analysis_results.json --output results/figures/Support_mapping_fragility.pdf
```

## Propagate support-distribution uncertainty

`support_uncertainty.py` provides a central-difference Jacobian, the
first-order delta-method covariance calculation, and a parametric support
bootstrap. These generic utilities apply when the analyst has an estimated
covariance matrix for external support-distribution parameters. For strictly
positive parameters, use them on the log-parameter scale.

## Reproduce the reported simulation

From the repository root, run:

```text
python code/mixed_support_simulation.py --replicates 5000 --output results/mixed_support_simulation_results.json
```

Or specify the workbook explicitly:

```text
python code/mixed_support_simulation.py --data-file "path/to/workbook.xlsx" --replicates 5000 --output results/mixed_support_simulation_results.json
```

The random-number seeds are fixed within the script. Simulation results report
bias, RMSE, Normal-interval and study-count Student-t interval coverage, and
type-I error in dedicated null-effect settings. A separate generalizability
experiment resamples empirical study-block configurations at K = 10, 20, and 30. The archived
outputs report point-estimation error, interval width and coverage, type-I error,
boundary rates, convergence, and Monte Carlo standard errors. Profiled REML criterion values in
the JSON files are optimization diagnostics within a fixed design mapping and
must not be compared across mappings that change the fixed-effect design.

## Validation tests

After installing the dependencies, run:

```text
python code/test_intervaldrma.py
```

The 20-test validation suite checks unit multipliers from $10^{-6}$ to $10^6$,
the exact $\tau=0$ boundary, row/study ordering, extreme open tails and narrow
supports, and both nested special cases in Proposition 1. Point
support means supplied through the generalized interface must match a
separately assembled conventional GLS/profile-REML calculation to numerical
precision, and conditional means under Uniform interval exposure must match
midpoint-scored DRMA exactly.
