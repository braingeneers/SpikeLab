"""Tests for spikedata/stat_utils.py — linear_regression and helpers."""

import pathlib
import sys

import numpy as np
import pytest

# Ensure project root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpikeLab.spikedata.stat_utils import linear_regression, _approx_normal_quantile

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
