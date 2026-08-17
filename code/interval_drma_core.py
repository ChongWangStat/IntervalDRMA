"""Core data structures and profile-REML estimator for IntervalDRMA.

The estimator standardizes the design contrasts internally.  This makes the
optimization invariant to the exposure unit while estimates are returned on
the user's original scale.  The between-study standard deviation may equal
zero exactly; no positive numerical lower bound is imposed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import t as student_t


@dataclass(frozen=True)
class Support:
    lower: float
    upper: float
    point: bool = False
    upper_from_dmax: bool = False

    def with_dmax_multiplier(self, multiplier: float) -> "Support":
        if self.point or not self.upper_from_dmax:
            return self
        return Support(
            lower=self.lower,
            upper=self.upper * multiplier,
            point=False,
            upper_from_dmax=True,
        )


@dataclass
class Row:
    refid: int
    region: str
    comparison: Support
    reference: Support
    se: float
    log_effect: float
    effect_measure: str
    author: str = ""
    comparison_label: str = ""
    reference_label: str = ""
    dmax: float | None = None
    dmax_reason: str = ""
    mapping_type: int = 1


@dataclass
class Study:
    refid: int
    rows: list[Row]
    sigma: np.ndarray


class FastReml:
    """One-slope profile REML with unit-invariant internal standardization."""

    def __init__(self, studies: list[Study], x_est: list[np.ndarray]):
        if len(studies) != len(x_est) or len(studies) < 2:
            raise ValueError("Studies and designs must align and contain at least two blocks")

        raw_design = [np.asarray(x, dtype=float) for x in x_est]
        if any(x.ndim != 1 for x in raw_design):
            raise ValueError("Each study design must be one-dimensional")
        if any(len(x) != len(study.sigma) for study, x in zip(studies, raw_design)):
            raise ValueError("Each design must align with its sampling covariance")
        if any(not np.all(np.isfinite(x)) for x in raw_design):
            raise ValueError("Design contrasts must be finite")

        concatenated = np.concatenate(raw_design)
        self.design_scale = float(np.sqrt(np.mean(concatenated**2)))
        if not math.isfinite(self.design_scale) or self.design_scale <= 0:
            raise ValueError("At least one nonzero finite design contrast is required")

        self.x_est = [x / self.design_scale for x in raw_design]
        self.a = []
        self.sigma_inv = []
        self.logdet_sigma = []
        self.n_studies = len(studies)
        for study, x in zip(studies, self.x_est):
            covariance = np.asarray(study.sigma, dtype=float)
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0 or not math.isfinite(float(logdet)):
                raise ValueError("Sampling covariance must be positive definite")
            inverse = np.linalg.inv(covariance)
            self.sigma_inv.append(inverse)
            self.logdet_sigma.append(float(logdet))
            self.a.append(float(x @ inverse @ x))
        self.a = np.asarray(self.a)
        self.logdet_sigma = np.asarray(self.logdet_sigma)

    def _profile_objective(self, y_list: list[np.ndarray]):
        if len(y_list) != self.n_studies:
            raise ValueError("Outcome blocks must align with study blocks")
        outcomes = [np.asarray(y, dtype=float) for y in y_list]
        if any(len(y) != len(x) for y, x in zip(outcomes, self.x_est)):
            raise ValueError("Each outcome block must align with its design")
        if any(not np.all(np.isfinite(y)) for y in outcomes):
            raise ValueError("Outcome blocks must be finite")

        c = np.asarray(
            [x @ inverse @ y for x, inverse, y in zip(self.x_est, self.sigma_inv, outcomes)]
        )
        d = np.asarray([y @ inverse @ y for inverse, y in zip(self.sigma_inv, outcomes)])

        def objective(tau_scaled: float) -> float:
            if tau_scaled < 0 or not math.isfinite(tau_scaled):
                return math.inf
            tau2 = tau_scaled * tau_scaled
            factor = 1.0 + tau2 * self.a
            xv_x = self.a / factor
            xv_y = c / factor
            yv_y = d - tau2 * c * c / factor
            information = float(xv_x.sum())
            if information <= 0 or not math.isfinite(information):
                return math.inf
            beta_scaled = float(xv_y.sum() / information)
            quadratic = float(
                (yv_y - 2.0 * beta_scaled * xv_y + beta_scaled**2 * xv_x).sum()
            )
            logdet = float((self.logdet_sigma + np.log(factor)).sum())
            value = 0.5 * (logdet + quadratic + math.log(information))
            return value if math.isfinite(value) else math.inf

        return objective, c

    @staticmethod
    def _adaptive_upper(objective) -> tuple[float, int]:
        """Find an upper bracket after the profile objective stops decreasing."""
        previous = objective(0.0)
        if not math.isfinite(previous):
            raise RuntimeError("Profile REML objective is not finite at tau=0")
        upper = 1.0
        expansions = 0
        current = objective(upper)
        while current < previous and expansions < 40:
            previous = current
            upper *= 2.0
            current = objective(upper)
            expansions += 1
        if not math.isfinite(current):
            raise RuntimeError("Profile REML objective became nonfinite during bracketing")
        if current < previous:
            raise RuntimeError("Could not bracket the profile REML optimum adaptively")
        return upper, expansions

    def _optimize(self, objective) -> dict:
        upper, expansions = self._adaptive_upper(objective)
        upper_u = math.log1p(upper)
        result = minimize_scalar(
            lambda u: objective(math.expm1(float(u))),
            bounds=(0.0, upper_u),
            method="bounded",
            options={"xatol": 2e-10, "maxiter": 1000},
        )
        if not result.success or not math.isfinite(float(result.fun)):
            raise RuntimeError(f"Profile REML optimization failed: {result.message}")

        interior_tau = math.expm1(float(result.x))
        candidates = [
            (0.0, objective(0.0), "zero"),
            (interior_tau, float(result.fun), "interior"),
            (upper, objective(upper), "upper"),
        ]
        tau_scaled, criterion, location = min(candidates, key=lambda item: item[1])
        tolerance = 1e-10 * (1.0 + abs(float(criterion)))
        zero_value = candidates[0][1]
        if zero_value <= criterion + tolerance:
            tau_scaled, criterion, location = 0.0, zero_value, "zero"
        if location == "upper":
            raise RuntimeError("Profile REML optimum lies on the adaptive upper bracket")
        return {
            "tau_scaled": float(tau_scaled),
            "criterion": float(criterion),
            "optimizer_success": True,
            "optimizer_message": str(result.message),
            "adaptive_upper_scaled": float(upper),
            "upper_expansions": int(expansions),
            "tau_at_zero_boundary": bool(tau_scaled == 0.0),
        }

    def _coefficient(self, c: np.ndarray, tau_scaled: float) -> tuple[float, float]:
        factor = 1.0 + tau_scaled**2 * self.a
        information_scaled = float((self.a / factor).sum())
        beta_scaled = float((c / factor).sum() / information_scaled)
        return beta_scaled / self.design_scale, information_scaled

    def standard_error(self, tau: float) -> float:
        tau_scaled = float(tau) * self.design_scale
        factor = 1.0 + tau_scaled**2 * self.a
        information_scaled = float((self.a / factor).sum())
        return math.sqrt(1.0 / information_scaled) / self.design_scale

    def fit(self, y_list: list[np.ndarray]) -> tuple[float, float]:
        objective, c = self._profile_objective(y_list)
        optimized = self._optimize(objective)
        tau_scaled = optimized["tau_scaled"]
        beta, _ = self._coefficient(c, tau_scaled)
        return float(beta), float(tau_scaled / self.design_scale)

    def fit_detailed(self, y_list: list[np.ndarray]) -> dict:
        """Return estimates, intervals, and optimization diagnostics."""
        objective, c = self._profile_objective(y_list)
        optimized = self._optimize(objective)
        tau_scaled = optimized["tau_scaled"]
        beta, information_scaled = self._coefficient(c, tau_scaled)
        tau = tau_scaled / self.design_scale
        se_beta = math.sqrt(1.0 / information_scaled) / self.design_scale
        study_df = self.n_studies - 1
        study_t_critical = float(student_t.ppf(0.975, study_df))

        # This standard chi-square profile cutoff is retained as an approximate,
        # descriptive interval.  The lower endpoint is the exact zero boundary.
        target = float(optimized["criterion"]) + 1.920729410347062
        root_function = lambda value: objective(value) - target
        if tau_scaled == 0.0 or root_function(0.0) <= 0:
            tau_lower_scaled = 0.0
        else:
            tau_lower_scaled = float(brentq(root_function, 0.0, tau_scaled))

        upper = max(float(optimized["adaptive_upper_scaled"]), 1.0)
        upper_value = root_function(upper)
        expansions = 0
        while upper_value < 0 and expansions < 40:
            upper *= 2.0
            upper_value = root_function(upper)
            expansions += 1
        tau_upper_scaled = (
            float(brentq(root_function, tau_scaled, upper))
            if upper_value >= 0
            else math.inf
        )

        return {
            "beta": float(beta),
            "se_beta": float(se_beta),
            "beta_ci": [beta - 1.96 * se_beta, beta + 1.96 * se_beta],
            "beta_ci_study_t": [
                beta - study_t_critical * se_beta,
                beta + study_t_critical * se_beta,
            ],
            "study_t_df": study_df,
            "tau": float(tau),
            "tau_ci": [
                float(tau_lower_scaled / self.design_scale),
                float(tau_upper_scaled / self.design_scale),
            ],
            "tau_ci_is_approximate_descriptive": True,
            "reml_criterion_at_optimum": float(optimized["criterion"]),
            "optimization": {
                **optimized,
                "design_scale": self.design_scale,
                "internal_standardization": "root mean square of design contrasts",
            },
        }


def parametric_bootstrap_interval(
    studies: list[Study],
    x: list[np.ndarray],
    y: list[np.ndarray],
    replicates: int = 2000,
    seed: int = 314159,
    fitted: dict | None = None,
) -> dict:
    """Return conditional basic and studentized parametric-bootstrap intervals."""
    if replicates < 200:
        raise ValueError("At least 200 bootstrap replicates are required")
    fitter = FastReml(studies, x)
    if fitted is None:
        fitted = fitter.fit_detailed(y)
    beta = float(fitted["beta"])
    tau = float(fitted["tau"])
    se_beta = float(fitted["se_beta"])
    rng = np.random.default_rng(seed)
    chol = [np.linalg.cholesky(study.sigma) for study in studies]
    beta_boot = np.empty(replicates)
    studentized = np.empty(replicates)
    zero_boundary = 0

    for r in range(replicates):
        random_slopes = rng.normal(0.0, tau, size=len(studies))
        y_boot = [
            (beta + random_slopes[k]) * x[k]
            + chol[k] @ rng.normal(size=len(x[k]))
            for k in range(len(studies))
        ]
        beta_r, tau_r = fitter.fit(y_boot)
        se_r = fitter.standard_error(tau_r)
        beta_boot[r] = beta_r
        studentized[r] = (beta_r - beta) / se_r
        zero_boundary += int(tau_r == 0.0)

    beta_quantiles = np.quantile(beta_boot, [0.025, 0.975])
    pivotal_quantiles = np.quantile(studentized, [0.025, 0.975])
    return {
        "replicates": replicates,
        "seed": seed,
        "conditional_on_support_means": True,
        "tau_zero_boundary_rate": zero_boundary / replicates,
        "basic_ci": [
            float(2.0 * beta - beta_quantiles[1]),
            float(2.0 * beta - beta_quantiles[0]),
        ],
        "studentized_ci": [
            float(beta - pivotal_quantiles[1] * se_beta),
            float(beta - pivotal_quantiles[0] * se_beta),
        ],
        "studentized_quantiles": [
            float(pivotal_quantiles[0]),
            float(pivotal_quantiles[1]),
        ],
    }


__all__ = ["FastReml", "Row", "Study", "Support", "parametric_bootstrap_interval"]
