"""Small public interface for second-stage contrast-level IntervalDRMA fitting.

The analyst first supplies one representative mean for each comparison and
reference support.  Those means may be reported group means, conditional means
from an outcome-independent exposure model, or point values for zero-width
supports.  This module then fits the fixed- and random-slope models used in the
manuscript without requiring callers to construct internal Study objects.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from interval_drma_core import (
    FastReml,
    Row,
    Study,
    Support,
    parametric_bootstrap_interval,
)


def _as_vector(values: Sequence, name: str, dtype=float) -> np.ndarray:
    vector = np.asarray(values, dtype=dtype)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence")
    return vector


def fit_interval_drma(
    log_effect: Sequence[float],
    standard_error: Sequence[float],
    study_id: Sequence,
    comparison_mean: Sequence[float],
    reference_mean: Sequence[float],
    *,
    rho: float = 0.5,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 314159,
) -> dict:
    """Fit IntervalDRMA from analysis-ready contrast-level arrays.

    Parameters are aligned row by row.  ``log_effect`` contains adjusted
    log odds ratios and ``standard_error`` their standard errors.  ``comparison_mean``
    and ``reference_mean`` are the outcome-independent support representatives.
    Rows sharing ``study_id`` are assigned an exchangeable working correlation
    ``rho``; use one row per identifier when the estimates are independent.
    """
    y = _as_vector(log_effect, "log_effect")
    se = _as_vector(standard_error, "standard_error")
    comparison = _as_vector(comparison_mean, "comparison_mean")
    reference = _as_vector(reference_mean, "reference_mean")
    ids = _as_vector(study_id, "study_id", dtype=object)
    lengths = {len(y), len(se), len(comparison), len(reference), len(ids)}
    if len(lengths) != 1:
        raise ValueError("All input sequences must have the same length")
    if not np.all(np.isfinite(y)):
        raise ValueError("log_effect contains a nonfinite value")
    if not np.all(np.isfinite(se)) or np.any(se <= 0):
        raise ValueError("standard_error values must be finite and positive")
    if not np.all(np.isfinite(comparison)) or not np.all(np.isfinite(reference)):
        raise ValueError("Support means must be finite")
    if not (-1.0 / max(len(y) - 1, 1) < rho < 1.0):
        raise ValueError("rho must yield positive-definite study covariance blocks")

    unique_ids = list(dict.fromkeys(ids.tolist()))
    studies = []
    design = []
    outcomes = []
    for block_number, identifier in enumerate(unique_ids, start=1):
        indices = np.flatnonzero(ids == identifier)
        block_rows = [
            Row(
                refid=block_number,
                region=str(identifier),
                comparison=Support(comparison[index], comparison[index], point=True),
                reference=Support(reference[index], reference[index], point=True),
                se=se[index],
                log_effect=y[index],
                effect_measure="LOG_RATIO",
            )
            for index in indices
        ]
        block_se = se[indices]
        correlation = np.full((len(indices), len(indices)), rho, dtype=float)
        np.fill_diagonal(correlation, 1.0)
        covariance = np.diag(block_se) @ correlation @ np.diag(block_se)
        studies.append(Study(block_number, block_rows, covariance))
        design.append(comparison[indices] - reference[indices])
        outcomes.append(y[indices])

    if len(studies) < 2:
        raise ValueError("At least two independent study blocks are required")
    if not any(np.any(np.abs(x) > 0) for x in design):
        raise ValueError("At least one comparison-reference mean contrast is required")

    fitter = FastReml(studies, design)
    random_effects = fitter.fit_detailed(outcomes)
    if bootstrap_replicates:
        random_effects["parametric_bootstrap"] = parametric_bootstrap_interval(
            studies,
            design,
            outcomes,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            fitted=random_effects,
        )

    inverses = [np.linalg.inv(study.sigma) for study in studies]
    denominator = sum(
        float(x @ inverse @ x)
        for x, inverse in zip(design, inverses)
    )
    numerator = sum(
        float(x @ inverse @ outcome)
        for x, inverse, outcome in zip(design, inverses, outcomes)
    )
    fixed_beta = numerator / denominator
    fixed_se = math.sqrt(1.0 / denominator)

    return {
        "n_studies": len(studies),
        "n_contrasts": len(y),
        "rho": rho,
        "design_contrast": np.concatenate(design).tolist(),
        "random_effects": random_effects,
        "fixed_effects": {
            "beta": fixed_beta,
            "se_beta": fixed_se,
            "beta_ci": [
                fixed_beta - 1.96 * fixed_se,
                fixed_beta + 1.96 * fixed_se,
            ],
        },
    }


__all__ = ["fit_interval_drma"]
