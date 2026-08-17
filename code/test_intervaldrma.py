"""Small deterministic checks for the unified support implementation."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interval_drma_core import (
    FastReml,
    Study,
    Support,
    parametric_bootstrap_interval,
)
from mixed_support_simulation import (
    BASE_BETA,
    BASE_TAU,
    build_studies,
    design_vector,
    gamma_mean,
    midpoint,
    read_rows,
    simulate_resampled_study_count,
)
from radon_example_analysis import (
    build_contrasts as build_radon_contrasts,
    read_workbook as read_radon_workbook,
    run_analysis as run_radon_analysis,
)
from interval_drma import fit_interval_drma
from run_analysis import main as run_afo_analysis
from support_uncertainty import delta_method_covariance, numerical_jacobian


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    ROOT / "data"
    if (ROOT / "data").exists()
    else ROOT / "repository" / "IntervalDRMA" / "data"
)
DATA = DATA_ROOT / "analytic_contrasts.xlsx"
RADON_DATA = DATA_ROOT / "radon_lung_cancer_contrasts.xlsx"


class UnifiedSupportTests(unittest.TestCase):
    def test_zero_width_mean_is_exact_value(self):
        point = Support(1.5, 1.5, point=True)
        self.assertEqual(gamma_mean(point, shape=2.0, scale=3.0), 1.5)
        self.assertEqual(midpoint(point), 1.5)

    def test_all_zero_width_model_equals_conventional_drma(self):
        """Verify Proposition 1 at the implementation level."""
        log_effect = np.asarray([0.10, 0.21, -0.08, -0.14, 0.06])
        standard_error = np.asarray([0.12, 0.16, 0.11, 0.15, 0.13])
        study_id = np.asarray(["A", "A", "B", "B", "C"], dtype=object)
        comparison_dose = np.asarray([2.0, 4.0, 1.0, 3.0, 5.0])
        reference_dose = np.asarray([0.0, 1.0, 0.0, 1.0, 2.0])
        rho = 0.35

        generalized = fit_interval_drma(
            log_effect,
            standard_error,
            study_id,
            comparison_dose,
            reference_dose,
            rho=rho,
        )

        conventional_studies = []
        conventional_design = []
        conventional_outcomes = []
        for block_number, identifier in enumerate(dict.fromkeys(study_id), start=1):
            indices = np.flatnonzero(study_id == identifier)
            block_se = standard_error[indices]
            correlation = np.full((len(indices), len(indices)), rho)
            np.fill_diagonal(correlation, 1.0)
            covariance = np.diag(block_se) @ correlation @ np.diag(block_se)
            conventional_studies.append(Study(block_number, [], covariance))
            conventional_design.append(
                comparison_dose[indices] - reference_dose[indices]
            )
            conventional_outcomes.append(log_effect[indices])

        conventional_random = FastReml(
            conventional_studies, conventional_design
        ).fit_detailed(conventional_outcomes)
        inverse = [np.linalg.inv(study.sigma) for study in conventional_studies]
        denominator = sum(
            float(x @ weight @ x)
            for x, weight in zip(conventional_design, inverse)
        )
        numerator = sum(
            float(x @ weight @ y)
            for x, weight, y in zip(
                conventional_design, inverse, conventional_outcomes
            )
        )
        conventional_fixed_beta = numerator / denominator
        conventional_fixed_se = math.sqrt(1.0 / denominator)

        np.testing.assert_allclose(
            generalized["design_contrast"],
            np.concatenate(conventional_design),
            rtol=0,
            atol=0,
        )
        self.assertTrue(
            math.isclose(
                generalized["random_effects"]["beta"],
                conventional_random["beta"],
                rel_tol=0,
                abs_tol=1e-14,
            )
        )
        self.assertTrue(
            math.isclose(
                generalized["random_effects"]["tau"],
                conventional_random["tau"],
                rel_tol=0,
                abs_tol=1e-14,
            )
        )
        self.assertTrue(
            math.isclose(
                generalized["fixed_effects"]["beta"],
                conventional_fixed_beta,
                rel_tol=0,
                abs_tol=1e-14,
            )
        )
        self.assertTrue(
            math.isclose(
                generalized["fixed_effects"]["se_beta"],
                conventional_fixed_se,
                rel_tol=0,
                abs_tol=1e-14,
            )
        )

    def test_uniform_interval_model_equals_midpoint_drma(self):
        """Verify the second special case in Proposition 1."""
        rows = read_rows(DATA)
        studies = build_studies(rows, rho=0.5)
        uniform_design = [design_vector(study, "uniform") for study in studies]
        midpoint_design = [design_vector(study, "midpoint") for study in studies]
        for uniform, midpoint_values in zip(uniform_design, midpoint_design):
            np.testing.assert_allclose(uniform, midpoint_values, rtol=0, atol=0)

        outcomes = [
            np.asarray([row.log_effect for row in study.rows]) for study in studies
        ]
        uniform_fit = FastReml(studies, uniform_design).fit_detailed(outcomes)
        midpoint_fit = FastReml(studies, midpoint_design).fit_detailed(outcomes)
        self.assertTrue(
            math.isclose(
                uniform_fit["beta"], midpoint_fit["beta"], rel_tol=0, abs_tol=0
            )
        )
        self.assertTrue(
            math.isclose(
                uniform_fit["tau"], midpoint_fit["tau"], rel_tol=0, abs_tol=0
            )
        )

    def test_schultz_design_uses_point_contrasts(self):
        rows = read_rows(DATA)
        studies = build_studies(rows, rho=0.5)
        schultz = next(study for study in studies if study.refid == 32508669)
        observed = design_vector(schultz, "gamma")
        np.testing.assert_allclose(observed, [-4.0, -3.5, -3.0, -2.5, -2.0])

    def test_primary_point_estimates(self):
        rows = read_rows(DATA)
        studies = build_studies(rows, rho=0.5)
        design = [design_vector(study, "gamma") for study in studies]
        outcomes = [
            np.asarray([row.log_effect for row in study.rows]) for study in studies
        ]
        beta, tau = FastReml(studies, design).fit(outcomes)
        self.assertTrue(math.isclose(beta, -0.0694688774, abs_tol=1e-8))
        self.assertTrue(math.isclose(tau, 0.1292531501, abs_tol=1e-8))

    def test_primary_small_sample_interval(self):
        results = run_afo_analysis(DATA, bootstrap_replicates=200)
        random_effects = results["primary_conditional_mean"]["random_effects"]
        self.assertEqual(random_effects["study_t_df"], 7)
        self.assertTrue(
            random_effects["beta_ci_study_t"][0]
            < random_effects["beta_ci"][0]
        )
        self.assertTrue(
            random_effects["beta_ci_study_t"][1]
            > random_effects["beta_ci"][1]
        )

    def test_primary_residual_diagnostics(self):
        results = run_afo_analysis(DATA, bootstrap_replicates=200)
        diagnostics = results["primary_conditional_mean"]["random_effects"][
            "residual_diagnostics"
        ]
        self.assertTrue(
            math.isclose(
                diagnostics["q_residual"],
                27.0820446519,
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
        )
        self.assertEqual(diagnostics["reference_df"], 32)
        self.assertTrue(
            math.isclose(
                diagnostics["max_abs_decorrelated_residual"],
                2.4029759368,
                abs_tol=1e-8,
            )
        )

    def test_source_quality_sensitivity_and_envelope(self):
        results = run_afo_analysis(DATA, bootstrap_replicates=0)
        self.assertEqual(len(results["sensitivity"]), 30)
        source_rows = [
            item
            for item in results["sensitivity"]
            if item["specification"]
            == "nc_wi_registry_proxy_for_pa_netherlands"
        ]
        self.assertEqual(len(source_rows), 5)
        envelope = results["sensitivity_envelope"]
        self.assertEqual(envelope["n_prespecified_settings"], 30)
        self.assertLess(envelope["beta_estimate_range"][0], envelope["beta_estimate_range"][1])

    def test_effect_measure_mix(self):
        rows = read_rows(DATA)
        self.assertEqual(sum(row.effect_measure == "OR" for row in rows), 29)
        self.assertEqual(sum(row.effect_measure != "OR" for row in rows), 4)

    def test_radon_example_primary_estimate(self):
        results = run_radon_analysis(RADON_DATA, bootstrap_replicates=200)
        primary = results["primary_historical_distribution"]["random_effects"]
        self.assertEqual(results["n_studies"], 16)
        self.assertEqual(results["n_contrasts"], 56)
        self.assertTrue(math.isclose(primary["beta"], 0.0007813805, abs_tol=1e-10))
        self.assertTrue(math.isclose(primary["tau"], 0.0004690817, abs_tol=1e-10))
        self.assertEqual(primary["study_t_df"], 15)
        self.assertTrue(math.isclose(primary["or_per_100_bq_m3"], 1.0812719164, abs_tol=1e-9))

    def test_radon_distribution_and_open_tail_sensitivities(self):
        results = run_radon_analysis(RADON_DATA, bootstrap_replicates=0)
        current = results["unscear_2024_distribution_sensitivity"]["random_effects"]
        scores = results["point_score_sensitivity"]
        self.assertTrue(math.isclose(current["or_per_100_bq_m3"], 1.0833354987, abs_tol=1e-9))
        self.assertGreater(
            scores[0]["random_effects"]["or_per_100_bq_m3"],
            scores[-1]["random_effects"]["or_per_100_bq_m3"],
        )

    def test_leave_one_study_out_outputs(self):
        afo = run_afo_analysis(DATA, bootstrap_replicates=0)
        radon = run_radon_analysis(RADON_DATA, bootstrap_replicates=0)
        self.assertEqual(len(afo["leave_one_study_out"]), 8)
        self.assertEqual(len(radon["leave_one_study_out"]), 16)
        self.assertTrue(
            all(row["n_studies"] == 7 for row in afo["leave_one_study_out"])
        )
        self.assertTrue(
            all(row["n_studies"] == 15 for row in radon["leave_one_study_out"])
        )

    def test_public_interface_reproduces_radon_fit(self):
        categories, distributions = read_radon_workbook(RADON_DATA)
        records = build_radon_contrasts(categories, distributions, "historical")
        result = fit_interval_drma(
            [row["log_or"] for row in records],
            [row["se_log_or"] for row in records],
            [row["study_id"] for row in records],
            [row["comparison_mean"] for row in records],
            [row["reference_mean"] for row in records],
            rho=0.5,
        )
        self.assertEqual(result["n_studies"], 16)
        self.assertTrue(
            math.isclose(
                result["random_effects"]["beta"], 0.0007813805, abs_tol=1e-10
            )
        )

    def test_reml_is_unit_equivariant_across_twelve_orders(self):
        categories, distributions = read_radon_workbook(RADON_DATA)
        records = build_radon_contrasts(categories, distributions, "historical")
        log_effect = np.asarray([row["log_or"] for row in records])
        standard_error = np.asarray([row["se_log_or"] for row in records])
        study_id = np.asarray([row["study_id"] for row in records], dtype=object)
        comparison = np.asarray([row["comparison_mean"] for row in records])
        reference = np.asarray([row["reference_mean"] for row in records])
        baseline = fit_interval_drma(
            log_effect,
            standard_error,
            study_id,
            comparison,
            reference,
            rho=0.5,
        )["random_effects"]

        for multiplier in (1e-6, 1e-3, 1.0, 1e3, 1e6):
            fitted = fit_interval_drma(
                log_effect,
                standard_error,
                study_id,
                comparison * multiplier,
                reference * multiplier,
                rho=0.5,
            )["random_effects"]
            self.assertTrue(fitted["optimization"]["optimizer_success"])
            self.assertTrue(
                math.isclose(
                    fitted["beta"] * multiplier,
                    baseline["beta"],
                    rel_tol=2e-7,
                    abs_tol=1e-13,
                )
            )
            self.assertTrue(
                math.isclose(
                    fitted["tau"] * multiplier,
                    baseline["tau"],
                    rel_tol=2e-7,
                    abs_tol=1e-13,
                )
            )
            self.assertTrue(
                math.isclose(
                    fitted["se_beta"] * multiplier,
                    baseline["se_beta"],
                    rel_tol=2e-7,
                    abs_tol=1e-13,
                )
            )

    def test_reml_allows_exact_zero_heterogeneity_boundary(self):
        result = fit_interval_drma(
            log_effect=[0.1, 0.2, 0.15, 0.3],
            standard_error=[0.1, 0.1, 0.1, 0.1],
            study_id=["A", "A", "B", "B"],
            comparison_mean=[1.0, 2.0, 1.5, 3.0],
            reference_mean=[0.0, 0.0, 0.0, 0.0],
            rho=0.0,
        )["random_effects"]
        self.assertEqual(result["tau"], 0.0)
        self.assertTrue(result["optimization"]["tau_at_zero_boundary"])
        self.assertEqual(result["tau_ci"][0], 0.0)

    def test_reml_is_invariant_to_row_and_study_order(self):
        log_effect = np.asarray([0.10, 0.21, -0.08, -0.14, 0.06])
        standard_error = np.asarray([0.12, 0.16, 0.11, 0.15, 0.13])
        study_id = np.asarray(["A", "A", "B", "B", "C"], dtype=object)
        comparison = np.asarray([2.0, 4.0, 1.0, 3.0, 5.0])
        reference = np.asarray([0.0, 1.0, 0.0, 1.0, 2.0])
        baseline = fit_interval_drma(
            log_effect,
            standard_error,
            study_id,
            comparison,
            reference,
            rho=0.35,
        )["random_effects"]
        order = np.asarray([4, 2, 3, 1, 0])
        permuted = fit_interval_drma(
            log_effect[order],
            standard_error[order],
            study_id[order],
            comparison[order],
            reference[order],
            rho=0.35,
        )["random_effects"]
        self.assertTrue(math.isclose(permuted["beta"], baseline["beta"], abs_tol=1e-13))
        self.assertTrue(math.isclose(permuted["tau"], baseline["tau"], abs_tol=1e-13))

    def test_extreme_gamma_supports_are_stable_or_informative(self):
        tail = Support(100.0, math.inf)
        tail_mean = gamma_mean(tail, shape=2.0, scale=3.0)
        self.assertGreater(tail_mean, 100.0)
        narrow = Support(50.0, 50.0001)
        narrow_mean = gamma_mean(narrow, shape=2.0, scale=3.0)
        self.assertGreaterEqual(narrow_mean, narrow.lower)
        self.assertLessEqual(narrow_mean, narrow.upper)

    def test_resampled_study_count_design(self):
        rows = read_rows(DATA)
        studies = build_studies(rows, rho=0.5)
        design = [design_vector(study, "gamma") for study in studies]
        result = simulate_resampled_study_count(
            np.random.default_rng(123),
            studies,
            design,
            BASE_BETA,
            BASE_TAU,
            replicates=200,
            n_studies=10,
            templates=5,
        )
        self.assertEqual(result["n_studies"], 10)
        self.assertEqual(result["inference"]["study_t_df"], 9)
        self.assertTrue(0.0 <= result["inference"]["study_t_coverage"] <= 1.0)

    def test_parametric_bootstrap_interval_is_ordered(self):
        rows = read_rows(DATA)
        studies = build_studies(rows, rho=0.5)
        design = [design_vector(study, "gamma") for study in studies]
        outcomes = [
            np.asarray([row.log_effect for row in study.rows]) for study in studies
        ]
        result = parametric_bootstrap_interval(
            studies, design, outcomes, replicates=200, seed=1234
        )
        self.assertLess(result["studentized_ci"][0], result["studentized_ci"][1])

    def test_delta_method_support_uncertainty(self):
        fit_function = lambda psi: np.asarray(
            [2.0 * psi[0], psi[0] - psi[1]], dtype=float
        )
        psi = np.asarray([1.0, 3.0])
        jacobian = numerical_jacobian(fit_function, psi)
        expected_jacobian = np.asarray([[2.0, 0.0], [1.0, -1.0]])
        np.testing.assert_allclose(jacobian, expected_jacobian, atol=1e-7)

        conditional = np.diag([0.04, 0.09])
        support_covariance = np.asarray([[0.01, 0.002], [0.002, 0.04]])
        observed = delta_method_covariance(
            conditional, jacobian, support_covariance
        )
        expected = (
            conditional
            + expected_jacobian @ support_covariance @ expected_jacobian.T
        )
        np.testing.assert_allclose(observed, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
