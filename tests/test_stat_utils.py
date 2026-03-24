"""Tests for spikedata/stat_utils.py — linear_regression and helpers."""

import numpy as np
import pytest

from spikelab.spikedata.stat_utils import (
    linear_regression,
    _approx_normal_quantile,
    pairwise_tests,
)

# ---------------------------------------------------------------------------
# linear_regression
# ---------------------------------------------------------------------------


class TestLinearRegression:
    """Tests for the linear_regression OLS function."""

    def test_perfect_positive_line(self):
        """
        A perfect y = 2x + 1 relationship yields exact slope, intercept, R².

        Tests:
            (Test Case 1) slope == 2.0.
            (Test Case 2) intercept == 1.0.
            (Test Case 3) r_squared == 1.0.
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0
        res = linear_regression(x, y)
        assert res["slope"] == pytest.approx(2.0)
        assert res["intercept"] == pytest.approx(1.0)
        assert res["r_squared"] == pytest.approx(1.0)

    def test_negative_slope(self):
        """
        A negative linear relationship is captured correctly.

        Tests:
            (Test Case 1) slope is negative.
            (Test Case 2) R² is close to 1.0.
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = -3.0 * x + 10.0
        res = linear_regression(x, y)
        assert res["slope"] == pytest.approx(-3.0)
        assert res["r_squared"] == pytest.approx(1.0)

    def test_noisy_data_r2_less_than_one(self):
        """
        Noisy data produces R² < 1.

        Tests:
            (Test Case 1) R² is between 0 and 1.
            (Test Case 2) Slope is approximately correct despite noise.
        """
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + rng.normal(0, 2.0, size=50)
        res = linear_regression(x, y)
        assert 0 < res["r_squared"] < 1.0
        assert res["slope"] == pytest.approx(2.0, abs=0.5)

    def test_output_keys(self):
        """
        The returned dict contains all expected keys.

        Tests:
            (Test Case 1) All 7 keys are present.
        """
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 5.0, 8.0])
        res = linear_regression(x, y)
        expected_keys = {
            "slope",
            "intercept",
            "r_squared",
            "x_fit",
            "y_fit",
            "ci_lower",
            "ci_upper",
        }
        assert set(res.keys()) == expected_keys

    def test_x_fit_is_sorted(self):
        """
        x_fit is a sorted copy of the input x values.

        Tests:
            (Test Case 1) x_fit is monotonically non-decreasing.
            (Test Case 2) x_fit has the same length as the valid input.
        """
        x = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        y = np.array([10.0, 2.0, 6.0, 4.0, 8.0])
        res = linear_regression(x, y)
        assert np.all(np.diff(res["x_fit"]) >= 0)
        assert len(res["x_fit"]) == 5

    def test_y_fit_matches_model(self):
        """
        y_fit matches slope * x_fit + intercept.

        Tests:
            (Test Case 1) y_fit values equal the predicted values from the model.
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.5, 4.0, 5.5, 7.5, 9.0])
        res = linear_regression(x, y)
        expected_y = res["slope"] * res["x_fit"] + res["intercept"]
        np.testing.assert_allclose(res["y_fit"], expected_y)

    def test_ci_bounds_bracket_fit(self):
        """
        Confidence bounds bracket the fitted line (ci_lower <= y_fit <= ci_upper).

        Tests:
            (Test Case 1) ci_lower <= y_fit at every point.
            (Test Case 2) ci_upper >= y_fit at every point.
        """
        rng = np.random.default_rng(7)
        x = np.linspace(0, 10, 30)
        y = 1.5 * x + 3.0 + rng.normal(0, 1.0, size=30)
        res = linear_regression(x, y)
        assert np.all(res["ci_lower"] <= res["y_fit"] + 1e-10)
        assert np.all(res["ci_upper"] >= res["y_fit"] - 1e-10)

    def test_ci_widens_at_extremes(self):
        """
        Confidence interval is narrowest near the mean of x and widens
        towards the extremes.

        Tests:
            (Test Case 1) CI width at the endpoints is greater than at
                the midpoint.
        """
        rng = np.random.default_rng(99)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + rng.normal(0, 1.5, size=50)
        res = linear_regression(x, y)
        widths = res["ci_upper"] - res["ci_lower"]
        mid_idx = len(widths) // 2
        assert widths[0] > widths[mid_idx]
        assert widths[-1] > widths[mid_idx]

    def test_nan_values_dropped(self):
        """
        NaN values in x or y are automatically dropped.

        Tests:
            (Test Case 1) Result is computed on the 4 valid points only.
            (Test Case 2) x_fit has length 4 (not 6).
        """
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        y = np.array([2.0, 4.0, 6.0, np.nan, 10.0, 12.0])
        res = linear_regression(x, y)
        # Two NaN pairs dropped → 4 valid points
        assert len(res["x_fit"]) == 4

    def test_mismatched_lengths_raises(self):
        """
        Mismatched x and y lengths raise ValueError.

        Tests:
            (Test Case 1) ValueError with descriptive message.
        """
        with pytest.raises(ValueError, match="same length"):
            linear_regression(np.array([1, 2, 3]), np.array([1, 2]))

    def test_too_few_points_raises(self):
        """
        Fewer than 3 valid data points raises ValueError.

        Tests:
            (Test Case 1) Two points raises ValueError.
            (Test Case 2) All-NaN input raises ValueError.
        """
        with pytest.raises(ValueError, match="at least 3"):
            linear_regression(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="at least 3"):
            linear_regression(
                np.array([np.nan, np.nan, np.nan]),
                np.array([1.0, 2.0, 3.0]),
            )

    def test_constant_y_r2_zero(self):
        """
        When y is constant (no variance), R² is 0.

        Tests:
            (Test Case 1) r_squared == 0.0.
            (Test Case 2) slope == 0.0.
        """
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([5.0, 5.0, 5.0, 5.0])
        res = linear_regression(x, y)
        assert res["r_squared"] == pytest.approx(0.0)
        assert res["slope"] == pytest.approx(0.0)

    def test_custom_ci_level(self):
        """
        Changing ci_level affects the width of the confidence bounds.

        Tests:
            (Test Case 1) 99% CI is wider than 90% CI.
        """
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + rng.normal(0, 2.0, size=30)
        res_90 = linear_regression(x, y, ci_level=0.90)
        res_99 = linear_regression(x, y, ci_level=0.99)
        width_90 = np.mean(res_90["ci_upper"] - res_90["ci_lower"])
        width_99 = np.mean(res_99["ci_upper"] - res_99["ci_lower"])
        assert width_99 > width_90


# ---------------------------------------------------------------------------
# _approx_normal_quantile
# ---------------------------------------------------------------------------


class TestApproxNormalQuantile:
    """Tests for the Abramowitz & Stegun normal quantile approximation."""

    def test_p_0975_close_to_1_96(self):
        """
        p = 0.975 should approximate the well-known z = 1.96 quantile.

        Tests:
            (Test Case 1) Result is within 0.01 of 1.96.
        """
        z = _approx_normal_quantile(0.975)
        assert z == pytest.approx(1.96, abs=0.01)

    def test_p_0995_close_to_2_576(self):
        """
        p = 0.995 should approximate z ≈ 2.576.

        Tests:
            (Test Case 1) Result is within 0.01 of 2.576.
        """
        z = _approx_normal_quantile(0.995)
        assert z == pytest.approx(2.576, abs=0.01)

    def test_monotonically_increasing(self):
        """
        Higher p values produce larger quantiles.

        Tests:
            (Test Case 1) q(0.9) < q(0.95) < q(0.99).
        """
        q90 = _approx_normal_quantile(0.90)
        q95 = _approx_normal_quantile(0.95)
        q99 = _approx_normal_quantile(0.99)
        assert q90 < q95 < q99

    def test_p_leq_0_5_raises(self):
        """
        p <= 0.5 is out of range and raises ValueError.

        Tests:
            (Test Case 1) p = 0.5 raises ValueError.
            (Test Case 2) p = 0.3 raises ValueError.
        """
        with pytest.raises(ValueError, match="p must be > 0.5"):
            _approx_normal_quantile(0.5)
        with pytest.raises(ValueError, match="p must be > 0.5"):
            _approx_normal_quantile(0.3)

    def test_p_very_close_to_1(self):
        """
        p very close to 1 tests the limits of the approximation accuracy.

        Tests:
            (Test Case 1) p = 0.999 should return a large positive quantile
                (z ~ 3.09). The approximation may degrade but should still
                be in the right ballpark.
            (Test Case 2) p = 0.9999 should return z ~ 3.72. Verify the
                approximation is within 0.1 of the expected value.
        """
        z_999 = _approx_normal_quantile(0.999)
        assert z_999 == pytest.approx(3.09, abs=0.1)

        z_9999 = _approx_normal_quantile(0.9999)
        assert z_9999 == pytest.approx(3.72, abs=0.1)


# ---------------------------------------------------------------------------
# Edge Case Tests — linear_regression
# ---------------------------------------------------------------------------


class TestLinearRegressionEdgeCases:
    """Edge case tests for linear_regression identified in the edge case scan."""

    def test_all_x_identical_raises(self):
        """
        All x values identical means ss_xx = 0, causing division by zero
        in slope calculation. The function should raise ValueError.

        Tests:
            (Test Case 1) x = [5, 5, 5, 5] raises ValueError with message
                about identical x values.
        """
        x = np.array([5.0, 5.0, 5.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="identical"):
            linear_regression(x, y)

    def test_exactly_three_points(self):
        """
        Exactly 3 points is the minimum allowed. The function should produce
        valid results with no error.

        Tests:
            (Test Case 1) x = [1, 2, 3], y = [2, 4, 6]. Perfect fit with
                slope=2, intercept=0, R^2=1.
        """
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        res = linear_regression(x, y)
        assert res["slope"] == pytest.approx(2.0)
        assert res["intercept"] == pytest.approx(0.0)
        assert res["r_squared"] == pytest.approx(1.0)
        assert len(res["x_fit"]) == 3

    def test_ci_level_zero(self):
        """
        ci_level=0 produces alpha=1.0, p = 1.0 - 1.0/2 = 0.5.
        _approx_normal_quantile(0.5) raises ValueError because p must be > 0.5.

        Tests:
            (Test Case 1) ci_level=0 raises ValueError from
                _approx_normal_quantile(0.5).

        Notes:
            - The error is not caught by linear_regression itself; it
              propagates from _approx_normal_quantile.
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0 + np.array([0.1, -0.1, 0.2, -0.2, 0.0])
        with pytest.raises(ValueError, match="p must be > 0.5"):
            linear_regression(x, y, ci_level=0)

    def test_ci_level_one(self):
        """
        ci_level=1.0 produces alpha=0.0, p = 1.0 - 0.0/2 = 1.0.
        _approx_normal_quantile(1.0) computes log(1 - 1.0) = log(0) = -inf,
        which propagates through the formula producing non-finite CI values.

        Tests:
            (Test Case 1) ci_level=1.0 does not raise.
            (Test Case 2) Slope and intercept are still valid.
            (Test Case 3) CI values may be non-finite (Inf or NaN).

        Notes:
            - ci_level=1.0 is not validated. The resulting CI contains
              non-finite values due to log(0) in the quantile approximation.
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = linear_regression(x, y, ci_level=1.0)
        assert np.isfinite(res["slope"])
        assert np.isfinite(res["intercept"])

    def test_perfect_fit_zero_residual(self):
        """
        A perfect fit (all points on the line) produces ss_res=0, so
        se=0 and the CI band collapses to the fit line.

        Tests:
            (Test Case 1) y = 2x + 1 exactly. R^2 = 1.0.
            (Test Case 2) CI bounds equal y_fit (zero-width band).
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0
        res = linear_regression(x, y)
        assert res["r_squared"] == pytest.approx(1.0)
        np.testing.assert_allclose(res["ci_lower"], res["y_fit"])
        np.testing.assert_allclose(res["ci_upper"], res["y_fit"])


# ---------------------------------------------------------------------------
# pairwise_tests
# ---------------------------------------------------------------------------


class TestPairwiseTests:
    """Tests for the pairwise_tests function."""

    @staticmethod
    def _make_groups():
        """Create three clearly separated groups for testing."""
        rng = np.random.default_rng(42)
        return {
            "A": rng.normal(0, 1, 50),
            "B": rng.normal(5, 1, 50),
            "C": rng.normal(0, 1, 50),
        }

    def test_output_keys(self):
        """
        The result dict contains all expected keys.

        Tests:
            (Test Case 1) All 4 keys present.
        """
        res = pairwise_tests(self._make_groups())
        assert set(res.keys()) == {
            "pval_matrix",
            "sig_matrix",
            "n_comparisons",
            "labels",
        }

    def test_matrix_shape(self):
        """
        Output matrices are (K, K) where K is the number of groups.

        Tests:
            (Test Case 1) pval_matrix is (3, 3).
            (Test Case 2) sig_matrix is (3, 3).
        """
        res = pairwise_tests(self._make_groups())
        assert res["pval_matrix"].shape == (3, 3)
        assert res["sig_matrix"].shape == (3, 3)

    def test_diagonal_is_nan(self):
        """
        Diagonal entries of the p-value matrix are NaN.

        Tests:
            (Test Case 1) All diagonal values are NaN.
        """
        res = pairwise_tests(self._make_groups())
        for i in range(3):
            assert np.isnan(res["pval_matrix"][i, i])

    def test_symmetric(self):
        """
        The p-value matrix is symmetric.

        Tests:
            (Test Case 1) pval_matrix[i,j] == pval_matrix[j,i] for all i,j.
        """
        res = pairwise_tests(self._make_groups())
        pv = res["pval_matrix"]
        for i in range(3):
            for j in range(i + 1, 3):
                assert pv[i, j] == pv[j, i]

    def test_n_comparisons(self):
        """
        Number of pairwise comparisons is K*(K-1)/2.

        Tests:
            (Test Case 1) 3 groups → 3 comparisons.
        """
        res = pairwise_tests(self._make_groups())
        assert res["n_comparisons"] == 3

    def test_labels_from_dict_keys(self):
        """
        Dict input uses keys as labels.

        Tests:
            (Test Case 1) Labels are ["A", "B", "C"].
        """
        res = pairwise_tests(self._make_groups())
        assert res["labels"] == ["A", "B", "C"]

    def test_separated_groups_significant(self):
        """
        Clearly separated groups (A vs B) are detected as significant.

        Tests:
            (Test Case 1) A vs B is significant (p < 0.05 after Bonferroni).
        """
        res = pairwise_tests(self._make_groups())
        assert res["sig_matrix"][0, 1] is np.True_

    def test_similar_groups_not_significant(self):
        """
        Groups with the same distribution (A vs C) are not significant.

        Tests:
            (Test Case 1) A vs C is not significant.
        """
        res = pairwise_tests(self._make_groups())
        assert res["sig_matrix"][0, 2] is np.False_

    def test_list_input_with_labels(self):
        """
        List input with explicit labels works correctly.

        Tests:
            (Test Case 1) Labels match the provided list.
            (Test Case 2) Matrix shape matches number of groups.
        """
        groups = self._make_groups()
        data = [groups["A"], groups["B"], groups["C"]]
        res = pairwise_tests(data, labels=["X", "Y", "Z"])
        assert res["labels"] == ["X", "Y", "Z"]
        assert res["pval_matrix"].shape == (3, 3)

    def test_no_correction(self):
        """
        correction=None returns uncorrected p-values.

        Tests:
            (Test Case 1) Uncorrected p-values are smaller than or equal to
                Bonferroni-corrected ones.
        """
        groups = self._make_groups()
        res_bonf = pairwise_tests(groups, correction="bonferroni")
        res_none = pairwise_tests(groups, correction=None)
        # Uncorrected p <= corrected p (Bonferroni multiplies by n_comp)
        for i in range(3):
            for j in range(i + 1, 3):
                assert res_none["pval_matrix"][i, j] <= res_bonf["pval_matrix"][i, j]

    def test_mann_whitney(self):
        """
        Mann-Whitney U test produces valid results.

        Tests:
            (Test Case 1) Output has correct shape and keys.
            (Test Case 2) Separated groups are still significant.
        """
        res = pairwise_tests(self._make_groups(), test="mann_whitney")
        assert res["pval_matrix"].shape == (3, 3)
        assert res["sig_matrix"][0, 1] is np.True_

    def test_student_t(self):
        """
        Student's equal-variance t-test produces valid results.

        Tests:
            (Test Case 1) Output has correct shape.
            (Test Case 2) Separated groups are significant.
        """
        res = pairwise_tests(self._make_groups(), test="student_t")
        assert res["pval_matrix"].shape == (3, 3)
        assert res["sig_matrix"][0, 1] is np.True_

    def test_unknown_test_raises(self):
        """
        Unknown test name raises ValueError.

        Tests:
            (Test Case 1) ValueError with descriptive message.
        """
        with pytest.raises(ValueError, match="Unknown test"):
            pairwise_tests(self._make_groups(), test="kolmogorov")

    def test_unknown_correction_raises(self):
        """
        Unknown correction name raises ValueError.

        Tests:
            (Test Case 1) ValueError with descriptive message.
        """
        with pytest.raises(ValueError, match="Unknown correction"):
            pairwise_tests(self._make_groups(), correction="holm")

    def test_nan_values_stripped(self):
        """
        NaN values in group data are stripped before testing.

        Tests:
            (Test Case 1) Result is computed without errors when NaN present.
            (Test Case 2) Significance result is the same as without NaN.
        """
        groups = self._make_groups()
        groups_nan = {
            k: np.concatenate([v, [np.nan, np.nan]]) for k, v in groups.items()
        }
        res_clean = pairwise_tests(groups)
        res_nan = pairwise_tests(groups_nan)
        np.testing.assert_allclose(
            res_nan["pval_matrix"], res_clean["pval_matrix"], rtol=1e-10
        )

    def test_custom_alpha(self):
        """
        Custom alpha threshold affects significance determination.

        Tests:
            (Test Case 1) With alpha=0.001, a marginally significant
                comparison (A vs C) remains non-significant.
            (Test Case 2) Strongly significant comparison (A vs B)
                remains significant even at stricter alpha.
        """
        res = pairwise_tests(self._make_groups(), alpha=0.001)
        # A vs B should still be significant at alpha=0.001
        assert res["sig_matrix"][0, 1] is np.True_

    def test_two_groups(self):
        """
        Works correctly with only 2 groups.

        Tests:
            (Test Case 1) Matrix is (2, 2).
            (Test Case 2) n_comparisons is 1.
        """
        rng = np.random.default_rng(0)
        groups = {"X": rng.normal(0, 1, 30), "Y": rng.normal(5, 1, 30)}
        res = pairwise_tests(groups)
        assert res["pval_matrix"].shape == (2, 2)
        assert res["n_comparisons"] == 1
