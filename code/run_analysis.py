"""Reproduce the primary and sensitivity analyses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from interval_drma_core import FastReml, parametric_bootstrap_interval
from mixed_support_simulation import (
    DEFAULT_BOOK,
    REGION_PARAMS,
    build_studies,
    design_vector,
    read_rows,
)


OUTPUT = Path(__file__).with_name("analysis_results.json")


def observed_outcomes(studies):
    return [np.asarray([row.log_effect for row in study.rows]) for study in studies]


def fit_fixed(studies, x, y):
    inverses = [np.linalg.inv(study.sigma) for study in studies]
    denominator = sum(float(xk @ inv @ xk) for xk, inv in zip(x, inverses))
    numerator = sum(
        float(xk @ inv @ yk) for xk, inv, yk in zip(x, inverses, y)
    )
    beta = numerator / denominator
    se_beta = math.sqrt(1.0 / denominator)
    return {
        "beta": beta,
        "se_beta": se_beta,
        "beta_ci": [beta - 1.96 * se_beta, beta + 1.96 * se_beta],
    }


def common_parameters(shape: float, scale: float):
    return {region: (shape, scale) for region in REGION_PARAMS}


def residual_diagnostics(studies, x, y, beta: float, tau: float) -> dict:
    """Return descriptive decorrelated residual diagnostics for a fitted model."""
    residuals = []
    q_residual = 0.0
    for study, xk, yk in zip(studies, x, y):
        covariance = study.sigma + tau**2 * np.outer(xk, xk)
        decorrelated = np.linalg.solve(
            np.linalg.cholesky(covariance), yk - beta * xk
        )
        q_residual += float(decorrelated @ decorrelated)
        residuals.extend(
            {
                "study_refid": int(study.refid),
                "within_study_index": index + 1,
                "decorrelated_residual": float(value),
            }
            for index, value in enumerate(decorrelated)
        )

    reference_df = sum(len(yk) for yk in y) - 1
    largest = max(residuals, key=lambda item: abs(item["decorrelated_residual"]))
    return {
        "q_residual": q_residual,
        "reference_df": reference_df,
        "descriptive_p_value": float(chi2.sf(q_residual, reference_df)),
        "max_abs_decorrelated_residual": abs(largest["decorrelated_residual"]),
        "largest_residual": largest,
        "interpretation": (
            "Descriptive diagnostic only: tau, the working covariance, and "
            "support-distribution quantities may be estimated."
        ),
    }


def fit_setting(
    rows,
    rho,
    distribution="gamma",
    region_params=None,
    bootstrap_replicates=0,
    bootstrap_seed=314159,
    infinite_open_tails=False,
):
    studies = build_studies(rows, rho)
    x = [
        design_vector(
            study,
            distribution,
            region_params=region_params,
            infinite_open_tails=infinite_open_tails,
        )
        for study in studies
    ]
    y = observed_outcomes(studies)
    random_effects = FastReml(studies, x).fit_detailed(y)
    if bootstrap_replicates:
        random_effects["parametric_bootstrap"] = parametric_bootstrap_interval(
            studies,
            x,
            y,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            fitted=random_effects,
        )
    random_effects["residual_diagnostics"] = residual_diagnostics(
        studies,
        x,
        y,
        random_effects["beta"],
        random_effects["tau"],
    )
    information_by_study = []
    covariance_condition_numbers = []
    for study, xk in zip(studies, x):
        marginal = study.sigma + random_effects["tau"] ** 2 * np.outer(xk, xk)
        information_by_study.append(float(xk @ np.linalg.solve(marginal, xk)))
        covariance_condition_numbers.append(
            {
                "study_refid": int(study.refid),
                "sampling_covariance_condition_number": float(
                    np.linalg.cond(study.sigma)
                ),
                "marginal_covariance_condition_number": float(np.linalg.cond(marginal)),
            }
        )
    total_information = sum(information_by_study)
    random_effects["study_information_weights"] = [
        {
            "study_refid": int(study.refid),
            "weight": float(value / total_information),
        }
        for study, value in zip(studies, information_by_study)
    ]
    random_effects["covariance_condition_numbers"] = covariance_condition_numbers
    return {
        "random_effects": random_effects,
        "fixed_effects": fit_fixed(studies, x, y),
    }


def leave_one_study_out(rows, rho=0.5, distribution="gamma"):
    """Refit the primary random-effects model after omitting each study block."""
    results = []
    study_ids = list(dict.fromkeys(row.refid for row in rows))
    for study_id in study_ids:
        retained = [row for row in rows if row.refid != study_id]
        fitted = fit_setting(retained, rho=rho, distribution=distribution)
        random_effects = fitted["random_effects"]
        results.append(
            {
                "omitted_study_refid": int(study_id),
                "n_studies": len(study_ids) - 1,
                "n_comparisons": len(retained),
                "beta": random_effects["beta"],
                "se_beta": random_effects["se_beta"],
                "beta_ci": random_effects["beta_ci"],
                "tau": random_effects["tau"],
            }
        )
    return results


def support_mapping_fragility(rows, rho=0.5, distribution="gamma"):
    """Numerically assess local sensitivity to each interval-based design contrast.

    The reported change is the first-order change in the pooled slope induced by
    a 10% perturbation of one mapped contrast, holding the other contrasts and
    reported outcomes fixed. Exact point-to-point contrasts are omitted because
    their design values do not depend on an interval distribution.
    """
    studies = build_studies(rows, rho)
    x = [design_vector(study, distribution) for study in studies]
    y = observed_outcomes(studies)
    baseline = FastReml(studies, x).fit_detailed(y)
    diagnostics = []

    for study_index, study in enumerate(studies):
        for within_index, row in enumerate(study.rows):
            if row.comparison.point and row.reference.point:
                continue
            value = float(x[study_index][within_index])
            step = max(1e-5, abs(value) * 1e-4)
            x_plus = [item.copy() for item in x]
            x_minus = [item.copy() for item in x]
            x_plus[study_index][within_index] += step
            x_minus[study_index][within_index] -= step
            fit_plus = FastReml(studies, x_plus).fit_detailed(y)
            fit_minus = FastReml(studies, x_minus).fit_detailed(y)
            derivative = (fit_plus["beta"] - fit_minus["beta"]) / (2.0 * step)
            change = derivative * 0.1 * value
            diagnostics.append(
                {
                    "study_refid": int(study.refid),
                    "author": row.author,
                    "within_study_index": within_index + 1,
                    "comparison": row.comparison_label,
                    "reference": row.reference_label,
                    "mapped_contrast": value,
                    "d_beta_d_x": derivative,
                    "beta_change_for_10pct_design_shift": change,
                }
            )

    diagnostics.sort(
        key=lambda item: abs(item["beta_change_for_10pct_design_shift"]),
        reverse=True,
    )
    return {
        "baseline_beta": baseline["beta"],
        "perturbation": "10% local perturbation of one mapped interval contrast",
        "point_to_point_contrasts_excluded": True,
        "rows": diagnostics,
    }


def main(data_file: Path, bootstrap_replicates: int = 2000):
    rows = read_rows(data_file)
    full_mixed_proxy = fit_setting(
        rows,
        rho=0.5,
        bootstrap_replicates=bootstrap_replicates,
    )
    midpoint = fit_setting(rows, rho=0.5, distribution="midpoint")
    or_only_rows = [row for row in rows if row.effect_measure == "OR"]
    or_only = fit_setting(or_only_rows, rho=0.5)
    explicit_or_rows = [
        row
        for row in rows
        if row.effect_measure == "OR" and row.mapping_type == 1
    ]
    explicit_or = fit_setting(explicit_or_rows, rho=0.5)
    explicit_or_midpoint = fit_setting(
        explicit_or_rows, rho=0.5, distribution="midpoint"
    )
    explicit_or_infinite_tail = fit_setting(
        explicit_or_rows,
        rho=0.5,
        infinite_open_tails=True,
    )
    representative_rows = []
    for study_id in dict.fromkeys(row.refid for row in rows):
        candidates = [row for row in rows if row.refid == study_id]
        representative_rows.append(min(candidates, key=lambda row: row.se))
    representative_contrast_sensitivity = fit_setting(
        representative_rows, rho=0.0
    )
    leave_one_out = leave_one_study_out(rows)
    fragility = support_mapping_fragility(rows)

    actual_regions = tuple(dict.fromkeys(row.region for row in rows))
    actual_parameters = [REGION_PARAMS[region] for region in actual_regions]
    shapes = np.asarray([value[0] for value in actual_parameters])
    means = np.asarray([value[0] * value[1] for value in actual_parameters])
    pooled_shape = float(shapes.mean())
    pooled_scale = float(means.mean() / pooled_shape)

    all_shapes = np.asarray([value[0] for value in REGION_PARAMS.values()])
    all_means = np.asarray([value[0] * value[1] for value in REGION_PARAMS.values()])
    five_region_shape = float(all_shapes.mean())
    five_region_scale = float(all_means.mean() / five_region_shape)

    registry_regions = ("NC", "Wisconsin")
    registry_parameters = [REGION_PARAMS[region] for region in registry_regions]
    registry_shape = float(np.mean([value[0] for value in registry_parameters]))
    registry_mean = float(np.mean([value[0] * value[1] for value in registry_parameters]))
    registry_scale = registry_mean / registry_shape
    registry_proxy_parameters = dict(REGION_PARAMS)
    registry_proxy_parameters["PA"] = (registry_shape, registry_scale)
    registry_proxy_parameters["Netherlands"] = (registry_shape, registry_scale)

    specifications = {
        "regional": None,
        "pooled_actual_regions": common_parameters(pooled_shape, pooled_scale),
        "pooled_five_regions_with_iowa_reference": common_parameters(
            five_region_shape, five_region_scale
        ),
        "pooled_scale_minus_20_percent": common_parameters(
            pooled_shape, pooled_scale * 0.8
        ),
        "pooled_scale_plus_20_percent": common_parameters(
            pooled_shape, pooled_scale * 1.2
        ),
        "nc_wi_registry_proxy_for_pa_netherlands": registry_proxy_parameters,
    }
    sensitivity = []
    for name, parameters in specifications.items():
        for rho in (0.0, 0.25, 0.5, 0.75, 0.9):
            result = fit_setting(rows, rho=rho, region_params=parameters)
            result.update(
                {
                    "specification": name,
                    "rho": rho,
                    "shape": (
                        None
                        if parameters is None
                        else (
                            registry_shape
                            if name == "nc_wi_registry_proxy_for_pa_netherlands"
                            else pooled_shape
                        )
                    ),
                    "scale": (
                        None
                        if parameters is None
                        else (
                            registry_scale
                            if name == "nc_wi_registry_proxy_for_pa_netherlands"
                            else next(iter(parameters.values()))[1]
                        )
                    ),
                }
            )
            sensitivity.append(result)

    sensitivity_envelope = {
        "n_prespecified_settings": len(sensitivity),
        "beta_estimate_range": [
            min(item["random_effects"]["beta"] for item in sensitivity),
            max(item["random_effects"]["beta"] for item in sensitivity),
        ],
        "normal_interval_envelope": [
            min(item["random_effects"]["beta_ci"][0] for item in sensitivity),
            max(item["random_effects"]["beta_ci"][1] for item in sensitivity),
        ],
        "study_t_interval_envelope": [
            min(item["random_effects"]["beta_ci_study_t"][0] for item in sensitivity),
            max(item["random_effects"]["beta_ci_study_t"][1] for item in sensitivity),
        ],
        "interpretation": (
            "Descriptive distributional-sensitivity envelope, not a formal "
            "identified set or a simultaneous confidence interval."
        ),
    }

    return {
        "data_file": data_file.name,
        "n_comparisons": len(rows),
        "n_studies": len(set(row.refid for row in rows)),
        "n_zero_width_comparisons": sum(
            row.comparison.point and row.reference.point for row in rows
        ),
        "afo_primary_working_analysis": {
            "analysis": (
                "Primary eight-study AFO working analysis: all mapped contrasts; "
                "binary-buffer and animal-count supports are explicit working proxies, "
                "and four reported HRs are treated as approximate ORs on the log scale"
            ),
            "n_comparisons": len(rows),
            "n_studies": len(set(row.refid for row in rows)),
            **full_mixed_proxy,
        },
        "afo_primary_midpoint_comparison": midpoint,
        "afo_or_only_restriction": {
            "analysis": "OR-only restriction; four hazard-ratio proxy contrasts excluded",
            "n_comparisons": len(or_only_rows),
            "n_studies": len(set(row.refid for row in or_only_rows)),
            "excluded_comparisons": len(rows) - len(or_only_rows),
            **or_only,
        },
        "afo_explicit_support_restriction": {
            "analysis": (
                "Strict support-mapping restriction: reported ORs with explicit distance-support "
                "pairs, including exact-distance spline contrasts; binary-buffer and "
                "animal-count proxies and HRs excluded"
            ),
            "n_comparisons": len(explicit_or_rows),
            "n_studies": len(set(row.refid for row in explicit_or_rows)),
            "excluded_binary_or_proxies": len(or_only_rows) - len(explicit_or_rows),
            "excluded_hr_proxies": len(rows) - len(or_only_rows),
            **explicit_or,
        },
        "afo_explicit_support_midpoint_restriction": explicit_or_midpoint,
        "afo_explicit_support_infinite_open_tail_sensitivity": {
            "analysis": (
                "Strict explicit-support restriction with genuinely right-open source "
                "categories evaluated as (d, infinity) under the Gamma working distribution"
            ),
            **explicit_or_infinite_tail,
        },
        "afo_analysis_hierarchy": [
            {
                "analysis": "all_mapped_contrasts_primary",
                "n_comparisons": len(rows),
                "n_studies": len(set(row.refid for row in rows)),
                **full_mixed_proxy,
            },
            {
                "analysis": "or_only_restriction",
                "n_comparisons": len(or_only_rows),
                "n_studies": len(set(row.refid for row in or_only_rows)),
                **or_only,
            },
            {
                "analysis": "explicit_support_or_restriction",
                "n_comparisons": len(explicit_or_rows),
                "n_studies": len(set(row.refid for row in explicit_or_rows)),
                **explicit_or,
            },
        ],
        # Backward-compatible aliases for archived scripts that used the earlier
        # four-study-primary presentation. The current analysis treats these as
        # restriction analyses.
        "afo_primary_explicit_or": {
            "analysis": "Legacy key: strict explicit-support OR restriction",
            "n_comparisons": len(explicit_or_rows),
            "n_studies": len(set(row.refid for row in explicit_or_rows)),
            **explicit_or,
        },
        "afo_primary_infinite_open_tail_sensitivity": {
            "analysis": "Legacy key: strict explicit-support open-tail restriction",
            **explicit_or_infinite_tail,
        },
        "afo_expansion_hierarchy": [
            {
                "stage": "explicit_distance_or",
                "n_comparisons": len(explicit_or_rows),
                "n_studies": len(set(row.refid for row in explicit_or_rows)),
                **explicit_or,
            },
            {
                "stage": "plus_binary_buffer_or_proxies",
                "n_comparisons": len(or_only_rows),
                "n_studies": len(set(row.refid for row in or_only_rows)),
                **or_only,
            },
            {
                "stage": "plus_animal_count_hr_proxies",
                "n_comparisons": len(rows),
                "n_studies": len(set(row.refid for row in rows)),
                **full_mixed_proxy,
            },
        ],
        # Retained for backward reproducibility of earlier analysis versions.
        "primary_conditional_mean": full_mixed_proxy,
        "midpoint_comparison": midpoint,
        "effect_measure_sensitivity": {
            "analysis": "OR-only; four hazard-ratio proxy contrasts excluded",
            "n_comparisons": len(or_only_rows),
            "n_studies": len(set(row.refid for row in or_only_rows)),
            "excluded_comparisons": len(rows) - len(or_only_rows),
            **or_only,
        },
        "leave_one_study_out": leave_one_out,
        "one_most_precise_contrast_per_study_sensitivity": {
            "selection_rule": (
                "One source-reported contrast with the smallest log-scale standard "
                "error was retained per study; rho=0 because one contrast remains per block"
            ),
            "n_comparisons": len(representative_rows),
            "n_studies": len(representative_rows),
            **representative_contrast_sensitivity,
        },
        "support_mapping_fragility": fragility,
        "pooled_gamma": {
            "actual_study_regions": list(actual_regions),
            "shape": pooled_shape,
            "scale": pooled_scale,
        },
        "five_region_gamma_with_iowa_reference": {
            "shape": five_region_shape,
            "scale": five_region_scale,
        },
        "registry_source_sensitivity": {
            "verified_registry_regions": list(registry_regions),
            "proxy_regions_replaced": ["PA", "Netherlands"],
            "shape": registry_shape,
            "scale": registry_scale,
            "interpretation": (
                "Source-quality sensitivity: the pooled NC/Wisconsin permitted-"
                "facility Gamma parameters replace the PA producer-directory and "
                "Netherlands agricultural-business proxy parameters."
            ),
        },
        "reml_criterion_note": (
            "The profiled REML criterion is an optimization diagnostic within a "
            "fixed design mapping and is not comparable across mappings that "
            "change the fixed-effects design vector."
        ),
        "sensitivity_envelope": sensitivity_envelope,
        "sensitivity": sensitivity,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce the interval dose-response meta-analysis."
    )
    parser.add_argument("--data-file", type=Path, default=DEFAULT_BOOK)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    results = main(
        args.data_file.resolve(),
        bootstrap_replicates=args.bootstrap_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
