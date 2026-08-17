"""Utilities for propagating uncertainty in support-distribution parameters.

The functions implement the delta-method and parametric-bootstrap procedures
described in the manuscript. They are generic because the external covariance
matrix for the motivating regional Gamma fits is not available.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


Array = np.ndarray


def numerical_jacobian(
    fit_function: Callable[[Array], Array],
    parameter_estimate: Array,
    relative_step: float = 1e-5,
) -> Array:
    """Estimate d fit(parameter) / d parameter by central differences."""
    parameter_estimate = np.asarray(parameter_estimate, dtype=float)
    if parameter_estimate.ndim != 1:
        raise ValueError("parameter_estimate must be one-dimensional")
    if relative_step <= 0:
        raise ValueError("relative_step must be positive")

    baseline = np.atleast_1d(
        np.asarray(fit_function(parameter_estimate.copy()), dtype=float)
    )
    jacobian = np.empty((baseline.size, parameter_estimate.size), dtype=float)
    for column, value in enumerate(parameter_estimate):
        step = relative_step * max(1.0, abs(float(value)))
        plus = parameter_estimate.copy()
        minus = parameter_estimate.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (
            np.atleast_1d(np.asarray(fit_function(plus), dtype=float))
            - np.atleast_1d(np.asarray(fit_function(minus), dtype=float))
        ) / (2.0 * step)
    return jacobian


def delta_method_covariance(
    conditional_covariance: Array,
    jacobian: Array,
    support_parameter_covariance: Array,
) -> Array:
    """Add first-stage support uncertainty to conditional model covariance."""
    conditional_covariance = np.asarray(conditional_covariance, dtype=float)
    jacobian = np.asarray(jacobian, dtype=float)
    support_parameter_covariance = np.asarray(
        support_parameter_covariance, dtype=float
    )
    if conditional_covariance.shape != (
        jacobian.shape[0],
        jacobian.shape[0],
    ):
        raise ValueError("conditional covariance is incompatible with the Jacobian")
    if support_parameter_covariance.shape != (
        jacobian.shape[1],
        jacobian.shape[1],
    ):
        raise ValueError("support covariance is incompatible with the Jacobian")
    return (
        conditional_covariance
        + jacobian @ support_parameter_covariance @ jacobian.T
    )


def parametric_support_bootstrap(
    fit_function: Callable[[Array], Array],
    parameter_estimate: Array,
    support_parameter_covariance: Array,
    conditional_covariance_function: Callable[[Array], Array] | None = None,
    draws: int = 2000,
    seed: int = 20260811,
) -> dict[str, Array]:
    """Propagate first-stage uncertainty by multivariate-normal refitting.

    For strictly positive parameters, callers may pass log-transformed
    parameters and write ``fit_function`` to exponentiate before refitting.
    """
    parameter_estimate = np.asarray(parameter_estimate, dtype=float)
    support_parameter_covariance = np.asarray(
        support_parameter_covariance, dtype=float
    )
    if draws < 2:
        raise ValueError("draws must be at least two")
    if support_parameter_covariance.shape != (
        parameter_estimate.size,
        parameter_estimate.size,
    ):
        raise ValueError("support covariance has incompatible dimensions")

    rng = np.random.default_rng(seed)
    parameter_draws = rng.multivariate_normal(
        parameter_estimate, support_parameter_covariance, size=draws
    )
    fitted_draws = np.asarray(
        [np.atleast_1d(fit_function(draw)) for draw in parameter_draws],
        dtype=float,
    )
    between_support_covariance = np.atleast_2d(
        np.cov(fitted_draws, rowvar=False, ddof=1)
    )
    if conditional_covariance_function is None:
        average_conditional_covariance = np.zeros_like(
            between_support_covariance
        )
    else:
        average_conditional_covariance = np.mean(
            np.asarray(
                [conditional_covariance_function(draw) for draw in parameter_draws],
                dtype=float,
            ),
            axis=0,
        )
    return {
        "parameter_draws": parameter_draws,
        "fitted_draws": fitted_draws,
        "between_support_covariance": between_support_covariance,
        "average_conditional_covariance": average_conditional_covariance,
        "total_covariance": (
            average_conditional_covariance + between_support_covariance
        ),
    }
