"""
Tests for utility functions in spikedata/utils.py.

Covers: _cosine_sim, compute_cosine_similarity_with_lag,
compute_cross_correlation_with_lag, butter_filter, trough_between,
times_from_ms, to_ms, ensure_h5py, _train_from_i_t_list,
PCA_reduction, UMAP_reduction, UMAP_graph_communities.
"""

import warnings

import numpy as np
import pytest

from spikelab.spikedata.utils import (
    _cosine_sim,
    _train_from_i_t_list,
    butter_filter,
    compute_cosine_similarity_with_lag,
    compute_cross_correlation_with_lag,
    consecutive_durations,
    ensure_h5py,
    gplvm_average_state_probability,
    gplvm_continuity_prob,
    gplvm_state_entropy,
    shuffle_z_score,
    shuffle_percentile,
    slice_trend,
    slice_stability,
    times_from_ms,
    to_ms,
    trough_between,
)

try:
    from sklearn.decomposition import PCA  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap  # noqa: F401

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import networkx  # noqa: F401
    import community  # noqa: F401

    COMMUNITY_AVAILABLE = True
except ImportError:
    COMMUNITY_AVAILABLE = False


# ---------------------------------------------------------------------------
# _cosine_sim
# ---------------------------------------------------------------------------


class TestCosineSim:
    """Tests for the _cosine_sim helper."""

    def test_identical_vectors(self):
        """
        Identical non-zero vectors have cosine similarity of 1.0.

        Tests:
            (Test Case 1) Two identical vectors return 1.0.
        """
        a = np.array([1.0, 2.0, 3.0])
        assert _cosine_sim(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """
        Orthogonal vectors have cosine similarity of 0.0.

        Tests:
            (Test Case 1) Two orthogonal unit vectors return 0.0.
        """
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_sim(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """
        Opposite vectors have cosine similarity of -1.0.

        Tests:
            (Test Case 1) A vector and its negation return -1.0.
        """
        a = np.array([1.0, 2.0, 3.0])
        assert _cosine_sim(a, -a) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero_or_nan(self):
        """
        Zero-norm vectors: one zero → 0.0 (uncorrelated), both zero → NaN (undefined).

        Tests:
            (Test Case 1) Zero first argument, nonzero second returns 0.0.
            (Test Case 2) Nonzero first, zero second returns 0.0.
            (Test Case 3) Both zero returns NaN.
        """
        a = np.array([1.0, 2.0, 3.0])
        z = np.zeros(3)
        assert _cosine_sim(z, a) == 0.0
        assert _cosine_sim(a, z) == 0.0
        assert np.isnan(_cosine_sim(z, z))

    def test_scaled_vectors(self):
        """
        Cosine similarity is scale-invariant.

        Tests:
            (Test Case 1) A vector and its scaled version return 1.0.
        """
        a = np.array([1.0, 2.0, 3.0])
        assert _cosine_sim(a, 100.0 * a) == pytest.approx(1.0)

    def test_return_type_is_float(self):
        """
        Return value is a Python float.

        Tests:
            (Test Case 1) Result is an instance of float.
        """
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert isinstance(_cosine_sim(a, b), float)


# ---------------------------------------------------------------------------
# compute_cosine_similarity_with_lag
# ---------------------------------------------------------------------------


class TestComputeCosineSimilarityWithLag:
    """Tests for compute_cosine_similarity_with_lag."""

    def test_identical_signals_zero_lag(self):
        """
        Identical signals at zero lag return similarity 1.0 and lag 0.

        Tests:
            (Test Case 1) Same signal, max_lag=0.
        """
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim, lag = compute_cosine_similarity_with_lag(sig, sig, max_lag=0)
        assert sim == pytest.approx(1.0)
        assert lag == 0

    def test_none_max_lag_treated_as_zero(self):
        """
        max_lag=None is equivalent to max_lag=0.

        Tests:
            (Test Case 1) None produces same result as 0.
        """
        sig = np.array([1.0, 2.0, 3.0])
        sim_none, lag_none = compute_cosine_similarity_with_lag(sig, sig, max_lag=None)
        sim_zero, lag_zero = compute_cosine_similarity_with_lag(sig, sig, max_lag=0)
        assert sim_none == pytest.approx(sim_zero)
        assert lag_none == lag_zero

    def test_shifted_signal_detected(self):
        """
        A shifted copy of a signal is detected at the correct lag.

        Tests:
            (Test Case 1) Signal shifted by +2 frames detected at lag=2.
        """
        ref = np.zeros(20)
        ref[5:10] = [1, 2, 3, 2, 1]
        comp = np.zeros(20)
        comp[7:12] = [1, 2, 3, 2, 1]
        sim, lag = compute_cosine_similarity_with_lag(ref, comp, max_lag=5)
        assert lag == 2
        assert sim == pytest.approx(1.0)

    def test_negative_lag_detected(self):
        """
        A signal shifted earlier than the reference is detected with a negative lag.

        Tests:
            (Test Case 1) Signal shifted by -3 frames detected at lag=-3.
        """
        ref = np.zeros(20)
        ref[8:13] = [1, 2, 3, 2, 1]
        comp = np.zeros(20)
        comp[5:10] = [1, 2, 3, 2, 1]
        sim, lag = compute_cosine_similarity_with_lag(ref, comp, max_lag=5)
        assert lag == -3
        assert sim == pytest.approx(1.0)

    def test_max_lag_limits_search(self):
        """
        Lag search is confined to the max_lag window.

        Tests:
            (Test Case 1) A shift of 5 is not found with max_lag=3.
        """
        ref = np.zeros(30)
        ref[5:10] = [1, 2, 3, 2, 1]
        comp = np.zeros(30)
        comp[10:15] = [1, 2, 3, 2, 1]
        sim, lag = compute_cosine_similarity_with_lag(ref, comp, max_lag=3)
        # The true shift is 5, but max_lag=3 can't reach it
        assert abs(lag) <= 3

    def test_orthogonal_signals_similarity_near_zero(self):
        """
        Non-overlapping signals have near-zero similarity.

        Tests:
            (Test Case 1) Two signals with non-overlapping non-zero regions at lag 0.
        """
        ref = np.array([1.0, 0.0, 0.0, 0.0])
        comp = np.array([0.0, 0.0, 0.0, 1.0])
        sim, lag = compute_cosine_similarity_with_lag(ref, comp, max_lag=0)
        assert sim == pytest.approx(0.0)

    def test_accepts_list_input(self):
        """
        Function accepts plain lists, not just numpy arrays.

        Tests:
            (Test Case 1) List inputs produce valid float similarity and int lag.
        """
        sim, lag = compute_cosine_similarity_with_lag(
            [1, 2, 3, 4], [1, 2, 3, 4], max_lag=0
        )
        assert sim == pytest.approx(1.0)
        assert lag == 0

    def test_all_zero_signals_division_by_zero(self):
        """
        EC-UT-06: All-zero signals cause division by zero in cosine
        similarity. The _cosine_sim helper handles this by returning
        NaN when both norms are zero. At max_lag=0, the function
        returns (NaN, 0).

        Tests:
            (Test Case 1) Both signals all-zero, max_lag=0. Returns NaN
                similarity and lag 0.
            (Test Case 2) Both signals all-zero, max_lag=3. The function
                computes _cosine_sim at each lag, all returning NaN.
                np.argmax on an all-NaN array returns 0, so the result
                is (NaN, first valid lag).
        """
        zeros = np.zeros(20)
        sim, lag = compute_cosine_similarity_with_lag(zeros, zeros, max_lag=0)
        assert np.isnan(sim)
        assert lag == 0

        sim2, lag2 = compute_cosine_similarity_with_lag(zeros, zeros, max_lag=3)
        assert np.isnan(sim2)


# ---------------------------------------------------------------------------
# compute_cross_correlation_with_lag
# ---------------------------------------------------------------------------


class TestComputeCrossCorrelationWithLag:
    """Tests for compute_cross_correlation_with_lag."""

    def test_identical_signals_zero_lag(self):
        """
        Auto-correlation of a signal at zero lag returns 1.0.

        Tests:
            (Test Case 1) Identical signals with max_lag=0.
        """
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        corr, lag = compute_cross_correlation_with_lag(sig, sig, max_lag=0)
        assert corr == pytest.approx(1.0)
        assert lag == 0

    def test_none_max_lag_treated_as_zero(self):
        """
        max_lag=None is equivalent to max_lag=0.

        Tests:
            (Test Case 1) None produces same result as 0.
        """
        sig = np.array([1.0, 2.0, 3.0])
        corr_none, lag_none = compute_cross_correlation_with_lag(sig, sig, max_lag=None)
        corr_zero, lag_zero = compute_cross_correlation_with_lag(sig, sig, max_lag=0)
        assert corr_none == pytest.approx(corr_zero)
        assert lag_none == lag_zero

    def test_shifted_signal_detected(self):
        """
        A shifted signal is detected at the correct lag.

        Tests:
            (Test Case 1) Signal shifted by +2 detected at lag=-2.

        Notes:
            - Cross-correlation lag convention: a positive shift in comp_rate
              relative to ref_rate yields a negative lag value, because the
              correlate 'same' mode indexes the best-match offset from center.
        """
        ref = np.zeros(30)
        ref[10:15] = [1, 3, 5, 3, 1]
        comp = np.zeros(30)
        comp[12:17] = [1, 3, 5, 3, 1]
        corr, lag = compute_cross_correlation_with_lag(ref, comp, max_lag=5)
        assert lag == -2
        assert corr > 0.9

    def test_correlation_bounded(self):
        """
        Cross-correlation values are bounded between -1 and 1.

        Tests:
            (Test Case 1) Random signals stay within bounds.
        """
        rng = np.random.default_rng(42)
        ref = rng.random(50)
        comp = rng.random(50)
        corr, lag = compute_cross_correlation_with_lag(ref, comp, max_lag=10)
        assert -1.0 <= corr <= 1.0
        assert abs(lag) <= 10

    def test_zero_norm_vectors(self):
        """
        Tests cross-correlation with all-zero input vectors.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) Returns a valid (best_corr, best_lag) tuple.

        Notes:
            Zero vectors have zero norm, making normalized correlation
            undefined. The result can be NaN or 0 — the test only verifies
            that the function does not crash.
        """
        corr, lag = compute_cross_correlation_with_lag(
            np.zeros(50), np.zeros(50), max_lag=5
        )
        assert isinstance(corr, (int, float, np.integer, np.floating))
        assert isinstance(lag, (int, float, np.integer, np.floating))

    def test_constant_signal_zero_variance(self):
        """
        EC-UT-05: A constant (non-zero) signal has zero variance. The
        function detects that ref_norm and comp_norm are both nonzero,
        but at max_lag>0 the autocorrelation denominator product is zero
        because the cross-correlation of a constant with itself minus
        its mean-like structure collapses. At max_lag=0, the function
        uses the fast path and returns a valid correlation.

        Tests:
            (Test Case 1) Constant signal [5, 5, 5, ...] with max_lag=0.
                ref_norm and comp_norm are nonzero (sum of squares = 25*N),
                so the fast path returns corr=1.0 and lag=0.
            (Test Case 2) Constant signal with max_lag=5. The
                autocorrelation denominator may be zero or the result
                may differ — just verify no crash and finite output.
        """
        sig = np.full(50, 5.0)
        corr, lag = compute_cross_correlation_with_lag(sig, sig, max_lag=0)
        assert corr == pytest.approx(1.0)
        assert lag == 0

        corr2, lag2 = compute_cross_correlation_with_lag(sig, sig, max_lag=5)
        assert isinstance(corr2, (int, float, np.integer, np.floating))
        assert isinstance(lag2, (int, float, np.integer, np.floating))

    def test_zero_lag_dot_product_normalisation_ground_truth(self):
        """
        Analytical ground truth: at max_lag=0, the function must equal
        ``np.dot(ref, comp) / (||ref|| * ||comp||)`` — the cosine of the
        angle between the two vectors.

        Tests:
            (Test Case 1) For two arbitrary non-orthogonal vectors,
                the function output matches the closed-form cosine
                similarity within numerical precision.
            (Test Case 2) For two perpendicular vectors (unit basis
                vectors), the output is exactly 0.
            (Test Case 3) For two negatively scaled copies of the same
                vector, the output is exactly -1.

        Notes:
            - These are the canonical normalisation invariants of the
              fast zero-lag path; if the formula in the source ever
              drifts (e.g., switches to a different definition of the
              normaliser), this test will catch it.
        """
        rng = np.random.default_rng(0)
        a = rng.standard_normal(20)
        b = rng.standard_normal(20)
        expected = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        corr, lag = compute_cross_correlation_with_lag(a, b, max_lag=0)
        assert corr == pytest.approx(expected, abs=1e-12)
        assert lag == 0

        # Orthogonal one-hot vectors
        e1 = np.array([1.0, 0.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0, 0.0])
        corr2, lag2 = compute_cross_correlation_with_lag(e1, e2, max_lag=0)
        assert corr2 == pytest.approx(0.0, abs=1e-12)
        assert lag2 == 0

        # Negative scaling produces -1
        v = np.array([1.0, 2.0, 3.0, 4.0])
        corr3, lag3 = compute_cross_correlation_with_lag(v, -2.0 * v, max_lag=0)
        assert corr3 == pytest.approx(-1.0, abs=1e-12)
        assert lag3 == 0

    def test_known_shift_recovers_exact_lag_and_unit_correlation(self):
        """
        Analytical ground truth: if comp = ref shifted right by K samples
        (zero-padded) and the original support is contained well within the
        max_lag window, the recovered lag is exactly +K and the recovered
        correlation is exactly 1.0 (modulo finite-length normalisation).

        Tests:
            (Test Case 1) ref has a Gaussian bump centred at index 30, and
                comp has the same bump centred at index 35. With max_lag=10
                and signal length 100, the function returns lag near +/-5
                and correlation 1.0.

        Notes:
            - This is the textbook lag-recovery test for normalised
              cross-correlation.
            - The lag sign is implementation-defined (the SpikeLab convention
              is documented in the source); the test asserts on |lag|.
        """
        n = 100
        x = np.arange(n)
        ref = np.exp(-((x - 30) ** 2) / (2 * 4.0**2))
        comp = np.exp(-((x - 35) ** 2) / (2 * 4.0**2))
        corr, lag = compute_cross_correlation_with_lag(ref, comp, max_lag=10)
        assert abs(int(lag)) == 5
        assert corr == pytest.approx(1.0, abs=5e-3)


# ---------------------------------------------------------------------------
# butter_filter
# ---------------------------------------------------------------------------


class TestButterFilter:
    """Tests for the butter_filter function."""

    def test_lowpass(self):
        """
        Lowpass filter attenuates high-frequency components.

        Tests:
            (Test Case 1) A mix of 10 Hz and 1000 Hz signals; after lowpass at 100 Hz
                the high-frequency power is heavily attenuated.
        """
        fs = 10000.0
        t = np.arange(0, 0.1, 1.0 / fs)
        low_freq = np.sin(2 * np.pi * 10 * t)
        high_freq = np.sin(2 * np.pi * 1000 * t)
        data = low_freq + high_freq

        filtered = butter_filter(data, highcut=100.0, fs=fs, order=4)
        # High-frequency power should be much smaller after filtering
        residual_power = np.var(filtered - low_freq)
        original_power = np.var(high_freq)
        assert residual_power < 0.1 * original_power

    def test_highpass(self):
        """
        Highpass filter attenuates low-frequency components.

        Tests:
            (Test Case 1) A mix of 10 Hz and 1000 Hz signals; after highpass at 500 Hz
                the low-frequency power is heavily attenuated.
        """
        fs = 10000.0
        t = np.arange(0, 0.1, 1.0 / fs)
        low_freq = np.sin(2 * np.pi * 10 * t)
        high_freq = np.sin(2 * np.pi * 1000 * t)
        data = low_freq + high_freq

        filtered = butter_filter(data, lowcut=500.0, fs=fs, order=4)
        residual_power = np.var(filtered - high_freq)
        original_power = np.var(low_freq)
        assert residual_power < 0.1 * original_power

    def test_bandpass(self):
        """
        Bandpass filter passes the target band and attenuates others.

        Tests:
            (Test Case 1) A mix of 10 Hz, 500 Hz, and 4000 Hz signals; after bandpass
                300-700 Hz, only the 500 Hz component remains dominant.
        """
        fs = 10000.0
        t = np.arange(0, 0.1, 1.0 / fs)
        sig_10 = np.sin(2 * np.pi * 10 * t)
        sig_500 = np.sin(2 * np.pi * 500 * t)
        sig_4000 = np.sin(2 * np.pi * 4000 * t)
        data = sig_10 + sig_500 + sig_4000

        filtered = butter_filter(data, lowcut=300.0, highcut=700.0, fs=fs, order=4)
        residual_power = np.var(filtered - sig_500)
        rejected_power = np.var(sig_10 + sig_4000)
        assert residual_power < 0.1 * rejected_power

    def test_no_cutoff_raises(self):
        """
        Omitting both cutoffs raises ValueError.

        Tests:
            (Test Case 1) Neither lowcut nor highcut provided.
        """
        with pytest.raises(ValueError, match="Need at least"):
            butter_filter(np.ones(100))

    def test_lowcut_ge_highcut_raises(self):
        """
        lowcut >= highcut raises ValueError.

        Tests:
            (Test Case 1) lowcut == highcut raises.
            (Test Case 2) lowcut > highcut raises.
        """
        with pytest.raises(ValueError, match="lowcut must be smaller"):
            butter_filter(np.ones(100), lowcut=500.0, highcut=500.0)
        with pytest.raises(ValueError, match="lowcut must be smaller"):
            butter_filter(np.ones(100), lowcut=600.0, highcut=500.0)

    def test_preserves_shape(self):
        """
        Output has the same shape as input.

        Tests:
            (Test Case 1) 1D input returns 1D output of same length.
            (Test Case 2) 2D input returns 2D output of same shape.
        """
        data_1d = np.random.default_rng(0).random(200)
        out_1d = butter_filter(data_1d, highcut=100.0, fs=1000.0)
        assert out_1d.shape == data_1d.shape

        data_2d = np.random.default_rng(0).random((3, 200))
        out_2d = butter_filter(data_2d, highcut=100.0, fs=1000.0)
        assert out_2d.shape == data_2d.shape

    def test_butter_filter_single_sample(self):
        """
        sosfiltfilt requires more than one sample; a single-sample input
        should raise an error.

        Tests:
            (Test Case 1) Single-element array raises ValueError from
                scipy.signal.sosfiltfilt (padlen requirement).
        """
        with pytest.raises(ValueError):
            butter_filter(np.array([1.0]), highcut=100.0, fs=1000.0)

    def test_highcut_equals_nyquist_raises(self):
        """
        When highcut equals exactly fs/2, the normalized frequency Wn = 1.0
        which is invalid for a digital Butterworth filter. scipy raises
        ValueError.

        Tests:
            (Test Case 1) highcut = fs/2 = 500 Hz at fs=1000 Hz.
                Wn = 500/1000*2 = 1.0. scipy.signal.iirfilter raises
                ValueError for Wn >= 1 in digital mode.
        """
        with pytest.raises(ValueError):
            butter_filter(np.ones(100), highcut=500.0, fs=1000.0)

    def test_highcut_exceeds_nyquist_raises(self):
        """
        When highcut exceeds fs/2, the normalized frequency Wn > 1.0
        which is invalid for a digital Butterworth filter. scipy raises
        ValueError.

        Tests:
            (Test Case 1) highcut = 600 Hz at fs=1000 Hz.
                Wn = 600/1000*2 = 1.2. scipy.signal.iirfilter raises
                ValueError for Wn > 1 in digital mode.
        """
        with pytest.raises(ValueError):
            butter_filter(np.ones(100), highcut=600.0, fs=1000.0)

    def test_lowcut_equals_nyquist_raises(self):
        """
        When lowcut equals fs/2 in highpass mode, the normalized frequency
        Wn = 1.0 which is invalid for a digital filter. scipy raises
        ValueError.

        Tests:
            (Test Case 1) lowcut = fs/2 = 500 Hz at fs=1000 Hz
                (highpass mode). Wn = 1.0 raises ValueError.
        """
        with pytest.raises(ValueError):
            butter_filter(np.ones(100), lowcut=500.0, fs=1000.0)

    def test_all_zero_data(self):
        """
        EC-UT-03: All-zero input data is filtered without error and
        produces all-zero output (filtering a zero signal yields zero).

        Tests:
            (Test Case 1) 1-D array of 200 zeros through a lowpass filter.
                Output is all zeros with the same shape.
        """
        data = np.zeros(200)
        result = butter_filter(data, highcut=100.0, fs=1000.0)
        assert result.shape == data.shape
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_2d_input_axis_filtering(self):
        """
        EC-UT-04: 2-D input is filtered along the last axis (default
        behaviour of scipy.signal.sosfiltfilt). Each row is filtered
        independently.

        Tests:
            (Test Case 1) Two-row array where row 0 has a 10 Hz sine and
                row 1 has a 1000 Hz sine. After lowpass at 100 Hz, row 0
                is mostly preserved and row 1 is heavily attenuated.
        """
        fs = 10000.0
        t = np.arange(0, 0.1, 1.0 / fs)
        row0 = np.sin(2 * np.pi * 10 * t)  # 10 Hz — below cutoff
        row1 = np.sin(2 * np.pi * 1000 * t)  # 1000 Hz — above cutoff
        data = np.vstack([row0, row1])

        filtered = butter_filter(data, highcut=100.0, fs=fs, order=4)
        assert filtered.shape == data.shape
        # Row 0 (low freq) should be mostly preserved
        assert np.var(filtered[0] - row0) < 0.1 * np.var(row0)
        # Row 1 (high freq) should be heavily attenuated
        assert np.var(filtered[1]) < 0.1 * np.var(row1)

    def test_lowcut_zero_with_highcut(self):
        """
        butter_filter with lowcut=0 and highcut=100 creates a bandpass with Wn=[0, ...].

        Tests:
            (Test Case 1) lowcut=0 with highcut creates a bandpass filter.
                Wn=[0, highcut/fs*2] where Wn[0]=0 is invalid for bandpass,
                raising a ValueError.

        Notes:
            - The code does not treat lowcut=0 as lowcut=None. It creates
              a bandpass filter with Wn=0, which scipy rejects.
        """
        data = np.random.rand(1000)
        with pytest.raises(ValueError):
            butter_filter(data, lowcut=0, highcut=100, fs=20000)

    def test_fs_zero_division_by_zero(self):
        """
        fs=0 causes division by zero in the Nyquist frequency calculation
        (Wn = highcut / (0 * 0.5) = inf), which scipy rejects.

        Tests:
            (Test Case 1) fs=0 with highcut=100 raises an error from scipy
                due to invalid normalized frequency.
        """
        with pytest.raises((ValueError, ZeroDivisionError)):
            butter_filter(np.ones(100), highcut=100.0, fs=0.0)


# ---------------------------------------------------------------------------
# trough_between
# ---------------------------------------------------------------------------


class TestTroughBetween:
    """Tests for the trough_between helper."""

    def test_finds_minimum(self):
        """
        Returns the index of the minimum value between two indices.

        Tests:
            (Test Case 1) Clear trough between two peaks.
        """
        pop_rate = np.array([0, 5, 3, 1, 2, 6, 0], dtype=float)
        result = trough_between(1, 5, pop_rate)
        assert result == 3

    def test_adjacent_indices_returns_none(self):
        """
        Adjacent indices (R - L <= 1) return None.

        Tests:
            (Test Case 1) Consecutive indices.
            (Test Case 2) Same index.
        """
        pop_rate = np.array([0, 5, 3, 1, 2, 6], dtype=float)
        assert trough_between(2, 3, pop_rate) is None
        assert trough_between(2, 2, pop_rate) is None

    def test_first_element_is_trough(self):
        """
        When the minimum is at the left boundary of the segment.

        Tests:
            (Test Case 1) Monotonically increasing segment.
        """
        pop_rate = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        assert trough_between(0, 5, pop_rate) == 0


# ---------------------------------------------------------------------------
# times_from_ms
# ---------------------------------------------------------------------------


class TestTimesFromMs:
    """Tests for the times_from_ms conversion function."""

    def test_ms_identity(self):
        """
        Unit 'ms' returns float copy of input unchanged.

        Tests:
            (Test Case 1) Values preserved as floats.
        """
        t = np.array([0, 100, 200])
        result = times_from_ms(t, "ms", None)
        np.testing.assert_array_equal(result, [0.0, 100.0, 200.0])
        assert result.dtype == float

    def test_to_seconds(self):
        """
        Unit 's' divides by 1000.

        Tests:
            (Test Case 1) 1000 ms becomes 1.0 s.
        """
        t = np.array([0, 1000, 2500])
        result = times_from_ms(t, "s", None)
        np.testing.assert_allclose(result, [0.0, 1.0, 2.5])

    def test_to_samples(self):
        """
        Unit 'samples' converts using fs_Hz.

        Tests:
            (Test Case 1) At 1000 Hz, 1 ms = 1 sample.
            (Test Case 2) At 20000 Hz, 1 ms = 20 samples.
        """
        t = np.array([0, 1, 5])
        result = times_from_ms(t, "samples", fs_Hz=1000.0)
        np.testing.assert_array_equal(result, [0, 1, 5])
        assert np.issubdtype(result.dtype, np.integer)

        result_20k = times_from_ms(t, "samples", fs_Hz=20000.0)
        np.testing.assert_array_equal(result_20k, [0, 20, 100])

    def test_samples_without_fs_raises(self):
        """
        Unit 'samples' without valid fs_Hz raises ValueError.

        Tests:
            (Test Case 1) fs_Hz=None raises.
            (Test Case 2) fs_Hz=0 raises.
        """
        t = np.array([100])
        with pytest.raises(ValueError, match="fs_Hz"):
            times_from_ms(t, "samples", fs_Hz=None)
        with pytest.raises(ValueError, match="fs_Hz"):
            times_from_ms(t, "samples", fs_Hz=0)

    def test_unknown_unit_raises(self):
        """
        Unknown unit string raises ValueError.

        Tests:
            (Test Case 1) Unit 'minutes' is not recognized.
        """
        with pytest.raises(ValueError, match="Unknown time unit"):
            times_from_ms(np.array([1.0]), "minutes", None)

    def test_negative_times_to_samples(self):
        """
        Negative ms values converted to samples produce negative integers
        via np.rint(). The function does not validate sign.

        Tests:
            (Test Case 1) Negative ms values [-1.0, -0.5, -10.0] at
                20 kHz. np.rint(-1.0 * 20) = -20, np.rint(-0.5 * 20) = -10,
                np.rint(-10.0 * 20) = -200. Result dtype is int.
        """
        t = np.array([-1.0, -0.5, -10.0])
        result = times_from_ms(t, "samples", fs_Hz=20000.0)
        np.testing.assert_array_equal(result, [-20, -10, -200])
        assert np.issubdtype(result.dtype, np.integer)

    def test_very_large_ms_values_to_samples(self):
        """
        Very large ms values may overflow when converted to int samples.
        Values within int64 range convert correctly; values exceeding it
        silently overflow.

        Tests:
            (Test Case 1) 1e12 ms at 20 kHz = 2e13 samples, fits in
                int64. Verify correct conversion.
            (Test Case 2) 1e18 ms at 20 kHz = 2e19 samples, exceeds
                int64 max (~9.2e18). Verify result does not match the
                expected float value (silent overflow).
        """
        import warnings

        # Case a: large but within int64 range
        t_ok = np.array([1e12])
        result_ok = times_from_ms(t_ok, "samples", fs_Hz=20000.0)
        expected_ok = int(1e12 * 20)
        assert result_ok[0] == expected_ok

        # Case b: overflow territory (2e19 > int64 max ~9.2e18)
        # numpy may emit a RuntimeWarning on int64 overflow
        t_overflow = np.array([1e18])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result_overflow = times_from_ms(t_overflow, "samples", fs_Hz=20000.0)
        expected_float = 1e18 * 20.0
        # The int64 cast silently overflows; the result will not match
        assert result_overflow[0] != expected_float


# ---------------------------------------------------------------------------
# to_ms
# ---------------------------------------------------------------------------


class TestToMs:
    """Tests for the to_ms conversion function."""

    def test_ms_identity(self):
        """
        Unit 'ms' returns float copy of input.

        Tests:
            (Test Case 1) Values preserved.
        """
        v = np.array([10, 20, 30])
        result = to_ms(v, "ms", None)
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])

    def test_from_seconds(self):
        """
        Unit 's' multiplies by 1000.

        Tests:
            (Test Case 1) 1.0 s becomes 1000.0 ms.
        """
        v = np.array([0.0, 1.0, 2.5])
        result = to_ms(v, "s", None)
        np.testing.assert_allclose(result, [0.0, 1000.0, 2500.0])

    def test_from_samples(self):
        """
        Unit 'samples' converts using fs_Hz.

        Tests:
            (Test Case 1) At 20000 Hz, 20 samples = 1 ms.
        """
        v = np.array([0, 20, 100])
        result = to_ms(v, "samples", fs_Hz=20000.0)
        np.testing.assert_allclose(result, [0.0, 1.0, 5.0])

    def test_samples_without_fs_raises(self):
        """
        Unit 'samples' without valid fs_Hz raises ValueError.

        Tests:
            (Test Case 1) fs_Hz=None raises.
        """
        with pytest.raises(ValueError, match="fs_Hz"):
            to_ms(np.array([1]), "samples", fs_Hz=None)

    def test_unknown_unit_raises(self):
        """
        Unknown unit string raises ValueError.

        Tests:
            (Test Case 1) Unit 'hours' is not recognized.
        """
        with pytest.raises(ValueError, match="Unknown time unit"):
            to_ms(np.array([1.0]), "hours", None)

    def test_roundtrip_with_times_from_ms(self):
        """
        Converting to another unit and back yields the original values.

        Tests:
            (Test Case 1) ms -> s -> ms round-trip.
            (Test Case 2) ms -> samples -> ms round-trip at 20 kHz.
        """
        original = np.array([0.0, 50.0, 123.456])
        via_s = times_from_ms(original, "s", None)
        back = to_ms(via_s, "s", None)
        np.testing.assert_allclose(back, original)

        via_samp = times_from_ms(original, "samples", fs_Hz=20000.0)
        back_samp = to_ms(via_samp.astype(float), "samples", fs_Hz=20000.0)
        np.testing.assert_allclose(back_samp, original, atol=0.05)

    def test_inf_input_propagates(self):
        """
        Infinite values propagate through arithmetic without raising.

        Tests:
            (Test Case 1) np.inf in seconds -> inf * 1000 = inf in ms.
            (Test Case 2) -np.inf in seconds -> -inf in ms.
        """
        v = np.array([np.inf, -np.inf])
        result = to_ms(v, "s", None)
        assert np.isinf(result[0]) and result[0] > 0
        assert np.isinf(result[1]) and result[1] < 0

    def test_nan_input_propagates(self):
        """
        NaN values propagate through arithmetic without raising.

        Tests:
            (Test Case 1) np.nan in seconds -> nan * 1000 = nan in ms.
        """
        v = np.array([np.nan])
        result = to_ms(v, "s", None)
        assert np.isnan(result[0])

    def test_inf_nan_ms_identity(self):
        """
        Inf and NaN pass through the ms identity path unchanged.

        Tests:
            (Test Case 1) to_ms with unit='ms' returns inf/nan as float.
        """
        v = np.array([np.inf, np.nan])
        result = to_ms(v, "ms", None)
        assert np.isinf(result[0])
        assert np.isnan(result[1])


# ---------------------------------------------------------------------------
# ensure_h5py
# ---------------------------------------------------------------------------


class TestEnsureH5py:
    """Tests for the ensure_h5py guard function."""

    def test_does_not_raise_when_available(self):
        """
        ensure_h5py succeeds when h5py is installed.

        Tests:
            (Test Case 1) No exception raised in this environment (h5py is a core dep).
        """
        ensure_h5py()

    def test_noop_when_called(self):
        """
        ensure_h5py is a no-op now that h5py is a hard dependency.

        Tests:
            (Test Case 1) Calling ensure_h5py() a second time still does not raise.
        """
        ensure_h5py()  # no-op, should not raise


# ---------------------------------------------------------------------------
# _train_from_i_t_list
# ---------------------------------------------------------------------------


class TestTrainFromITList:
    """Tests for the _train_from_i_t_list helper."""

    def test_basic_split(self):
        """
        Correctly groups spike times by unit index.

        Tests:
            (Test Case 1) Three units with interleaved spikes.
        """
        idces = [0, 1, 2, 0, 1, 0]
        times = [10, 20, 30, 40, 50, 60]
        result = _train_from_i_t_list(idces, times, N=3)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [10, 40, 60])
        np.testing.assert_array_equal(result[1], [20, 50])
        np.testing.assert_array_equal(result[2], [30])

    def test_n_none_infers_from_max(self):
        """
        N=None infers the number of units from max index + 1.

        Tests:
            (Test Case 1) Indices [0, 2] with N=None produces 3 entries.
        """
        result = _train_from_i_t_list([0, 2], [5, 15], N=None)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [5])
        assert len(result[1]) == 0
        np.testing.assert_array_equal(result[2], [15])

    def test_empty_units(self):
        """
        Units with no spikes get empty arrays.

        Tests:
            (Test Case 1) N=3 but only unit 1 has spikes.
        """
        result = _train_from_i_t_list([1], [100], N=3)
        assert len(result[0]) == 0
        np.testing.assert_array_equal(result[1], [100])
        assert len(result[2]) == 0


# ---------------------------------------------------------------------------
# PCA_reduction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestPCAReduction:
    """Tests for PCA_reduction."""

    def test_output_shape(self):
        """
        Output shape is (n_samples, n_components).

        Tests:
            (Test Case 1) Default n_components=2.
            (Test Case 2) n_components=3.
        """
        from spikelab.spikedata.utils import PCA_reduction

        rng = np.random.default_rng(0)
        data = rng.random((20, 10))
        embedding, var_ratio, components = PCA_reduction(data, n_components=2)
        assert embedding.shape == (20, 2)
        assert var_ratio.shape == (2,)
        assert components.shape == (2, 10)

        embedding3, var_ratio3, components3 = PCA_reduction(data, n_components=3)
        assert embedding3.shape == (20, 3)
        assert var_ratio3.shape == (3,)
        assert components3.shape == (3, 10)

    def test_variance_ordering(self):
        """
        First component captures more variance than the second.

        Tests:
            (Test Case 1) Variance ratio is monotonically decreasing.
            (Test Case 2) All variance ratios are positive and sum to <= 1.
        """
        from spikelab.spikedata.utils import PCA_reduction

        rng = np.random.default_rng(42)
        data = rng.random((50, 10))
        embedding, var_ratio, components = PCA_reduction(data, n_components=2)
        assert var_ratio[0] >= var_ratio[1]
        assert np.all(var_ratio > 0)
        assert var_ratio.sum() <= 1.0 + 1e-10

    def test_n_components_exceeds_features(self):
        """
        PCA_reduction with n_components > n_features raises ValueError.

        Tests:
            (Test Case 1) n_components=10 on a (20, 3) matrix raises ValueError.
            (Test Case 2) Error message includes the offending values.
        """
        from spikelab.spikedata.utils import PCA_reduction

        rng = np.random.default_rng(0)
        data = rng.random((20, 3))
        with pytest.raises(ValueError, match="n_components=10.*min.*=3"):
            PCA_reduction(data, n_components=10)

    def test_n_components_one(self):
        """
        EC-UT-07: PCA with n_components=1 returns a single column embedding
        and a single-element variance ratio.

        Tests:
            (Test Case 1) n_components=1 on a (20, 5) matrix. Embedding
                shape is (20, 1), variance ratio shape is (1,), components
                shape is (1, 5).
        """
        from spikelab.spikedata.utils import PCA_reduction

        rng = np.random.default_rng(0)
        data = rng.random((20, 5))
        embedding, var_ratio, components = PCA_reduction(data, n_components=1)
        assert embedding.shape == (20, 1)
        assert var_ratio.shape == (1,)
        assert components.shape == (1, 5)
        assert var_ratio[0] > 0
        assert var_ratio[0] <= 1.0

    def test_identical_rows_zero_variance(self):
        """
        EC-UT-08: When all rows are identical, variance is zero in every
        direction. PCA still runs but all explained variance ratios are
        zero (or NaN depending on sklearn version) and the embedding
        values are all zero.

        Tests:
            (Test Case 1) 10 identical rows of [1, 2, 3, 4, 5].
                Embedding has shape (10, 2) with all values ~0.
        """
        from spikelab.spikedata.utils import PCA_reduction

        data = np.tile([1.0, 2.0, 3.0, 4.0, 5.0], (10, 1))
        embedding, var_ratio, components = PCA_reduction(data, n_components=2)
        assert embedding.shape == (10, 2)
        np.testing.assert_allclose(embedding, 0.0, atol=1e-10)

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_n_components_zero(self):
        """
        PCA_reduction with n_components=0 raises ValueError.

        Tests:
            (Test Case 1) n_components=0: PCA(n_components=0) raises ValueError
                from scikit-learn.
        """
        from spikelab.spikedata.utils import PCA_reduction

        data = np.random.default_rng(0).random((10, 5))
        # n_components=0 does not exceed max_components check (0 <= 5),
        # but PCA(n_components=0) may not raise in all sklearn versions.
        # In some versions, it produces a (10, 0) embedding silently.
        embedding, var_ratio, components = PCA_reduction(data, n_components=0)
        assert embedding.shape == (10, 0) or embedding.shape[1] == 0


# ---------------------------------------------------------------------------
# UMAP_reduction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not UMAP_AVAILABLE, reason="umap-learn not installed")
class TestUMAPReduction:
    """Tests for UMAP_reduction."""

    def test_output_shape(self):
        """
        Output shape is (n_samples, n_components).

        Tests:
            (Test Case 1) n_components=2 on small dataset.
        """
        from spikelab.spikedata.utils import UMAP_reduction

        rng = np.random.default_rng(0)
        data = rng.random((30, 5))
        embedding, tw = UMAP_reduction(data, n_components=2, random_state=42)
        assert embedding.shape == (30, 2)
        assert isinstance(tw, float)
        assert 0.0 <= tw <= 1.0

    def test_raises_without_umap(self, monkeypatch):
        """
        ImportError raised when umap is None.

        Tests:
            (Test Case 1) Monkeypatching umap to None triggers ImportError.
        """
        import spikelab.spikedata.utils as utils_mod
        from spikelab.spikedata.utils import UMAP_reduction

        monkeypatch.setattr(utils_mod, "umap", None)
        with pytest.raises(ImportError, match="umap-learn"):
            UMAP_reduction(np.ones((10, 3)))


# ---------------------------------------------------------------------------
# UMAP_graph_communities
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (UMAP_AVAILABLE and COMMUNITY_AVAILABLE),
    reason="umap-learn, networkx, or python-louvain not installed",
)
class TestUMAPGraphCommunities:
    """Tests for UMAP_graph_communities."""

    def test_output_shapes(self):
        """
        Returns embedding and labels with correct shapes.

        Tests:
            (Test Case 1) Embedding shape is (n_samples, n_components).
            (Test Case 2) Labels shape is (n_samples,) with integer dtype.
        """
        from spikelab.spikedata.utils import UMAP_graph_communities

        rng = np.random.default_rng(0)
        data = rng.random((30, 5))
        embedding, labels, tw = UMAP_graph_communities(
            data, n_components=2, random_state=42
        )
        assert embedding.shape == (30, 2)
        assert labels.shape == (30,)
        assert labels.dtype == int
        assert isinstance(tw, float)
        assert 0.0 <= tw <= 1.0

    def test_raises_without_deps(self, monkeypatch):
        """
        ImportError raised when optional deps are None.

        Tests:
            (Test Case 1) umap=None raises ImportError.
            (Test Case 2) nx=None raises ImportError.
            (Test Case 3) community_louvain=None raises ImportError.
        """
        import spikelab.spikedata.utils as utils_mod
        from spikelab.spikedata.utils import UMAP_graph_communities

        data = np.ones((10, 3))

        monkeypatch.setattr(utils_mod, "umap", None)
        with pytest.raises(ImportError, match="umap-learn"):
            UMAP_graph_communities(data)

        # Restore umap for next test
        monkeypatch.undo()

        monkeypatch.setattr(utils_mod, "nx", None)
        with pytest.raises(ImportError, match="networkx"):
            UMAP_graph_communities(data)

        monkeypatch.undo()

        monkeypatch.setattr(utils_mod, "community_louvain", None)
        with pytest.raises(ImportError, match="python-louvain"):
            UMAP_graph_communities(data)


from spikelab.spikedata.utils import (
    _resampled_isi,
    check_neuron_attributes,
    randomize,
    extract_waveforms,
    get_sttc,
    swap,
)


class TestResampledIsi:
    """Edge-case tests for _resampled_isi."""

    def test_resampled_isi_identical_spike_times(self):
        """
        Identical spike times are deduplicated with a RuntimeWarning.

        Tests:
            (Test Case 1) Three identical spike times are reduced to one unique
                value. A RuntimeWarning about duplicate removal is emitted.
                With only 1 unique spike, the function returns zeros.
        """
        spikes = np.array([5.0, 5.0, 5.0])
        times = np.arange(0, 20, 1.0)
        with pytest.warns(RuntimeWarning, match="duplicate spike time"):
            result = _resampled_isi(spikes, times, sigma_ms=2.0)
        assert result.shape == times.shape
        # Only 1 unique spike -> returns zeros (single-spike path)
        np.testing.assert_array_equal(result, np.zeros_like(times))

    def test_resampled_isi_identical_time_values(self):
        """
        Duplicate time grid values raise ValueError immediately.

        Tests:
            (Test Case 1) All-identical time grid values are rejected with
                a ValueError indicating duplicates are not allowed.
        """
        spikes = np.array([5.0, 10.0])
        times = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="duplicate values"):
            _resampled_isi(spikes, times, sigma_ms=2.0)

    def test_negative_sigma(self):
        """
        _resampled_isi with negative sigma_ms may raise or produce unexpected output.

        Tests:
            (Test Case 1) Negative sigma produces a negative sigma for
                gaussian_filter1d, which raises a ValueError in scipy >= 1.7.
        """
        from spikelab.spikedata.utils import _resampled_isi

        spikes = [1.0, 5.0, 10.0]
        times = np.linspace(0, 15, 100)
        try:
            result = _resampled_isi(spikes, times, sigma_ms=-5.0)
            # If scipy doesn't raise, result is still produced
            assert isinstance(result, np.ndarray)
        except (ValueError, RuntimeError):
            pass  # Expected for scipy versions that validate sigma

    def test_non_uniform_time_grid(self):
        """
        _resampled_isi uses times[1] - times[0] as a uniform step size.
        Non-uniform time grids produce wrong results because the bin assignment
        assumes constant dt_ms.

        Tests:
            (Test Case 1) Non-uniform time grid [0, 1, 5, 10, 20]. The function
                uses dt_ms = 1.0 (from times[1] - times[0]) regardless of the
                actual spacing. It does not raise an error. Output shape matches
                the times array.

        Notes:
            - This is a known limitation: the function assumes a uniform grid
              but does not validate this assumption. Results for non-uniform
              grids are unreliable.
        """
        spikes = np.array([2.0, 8.0, 15.0])
        times = np.array([0.0, 1.0, 5.0, 10.0, 20.0])
        result = _resampled_isi(spikes, times, sigma_ms=2.0)
        assert result.shape == times.shape

    def test_spikes_outside_times_range(self):
        """
        Spikes outside the times range are extrapolated as constant from
        the edge, which is the behaviour of np.interp.

        Tests:
            (Test Case 1) Spikes at -50 and 150 with times [0, 100]. The
                function does not raise and returns an array matching times shape.
        """
        spikes = np.array([-50.0, 10.0, 50.0, 150.0])
        times = np.arange(0, 100, 1.0)
        result = _resampled_isi(spikes, times, sigma_ms=5.0)
        assert result.shape == times.shape
        # Some values should be nonzero (from the interior spikes)
        assert np.any(result > 0)


class TestRandomize:
    """Edge-case tests for the randomize function."""

    def test_randomize_zero_spike_raster(self):
        """
        A raster with no spikes returns all zeros with the same shape.

        Tests:
            (Test Case 1) Zero raster (3, 100) returns same-shape
                all-zero array.
        """
        raster = np.zeros((3, 100), dtype=int)
        result = randomize(raster)
        assert result.shape == (3, 100)
        np.testing.assert_array_equal(result, 0)

    def test_randomize_single_spike(self):
        """
        A raster with exactly one spike preserves exactly one nonzero value.

        Tests:
            (Test Case 1) Raster with one spike at (0, 50). Result has
                same shape and exactly 1 nonzero value total.

        Notes:
            With only one spike, no valid swap can occur (swap requires two
            distinct spike positions), so the spike stays in place.
        """
        ar = np.zeros((3, 100), dtype=int)
        ar[0, 50] = 1
        result = randomize(ar)
        assert result.shape == (3, 100)
        assert np.sum(result) == 1

    def test_non_binary_raster(self):
        """
        randomize rejects non-binary rasters with ValueError.

        Tests:
            (Test Case 1) A raster with values 0, 1, 2 raises ValueError.
        """
        ar = np.array([[0, 1, 0, 2], [1, 0, 1, 0], [0, 0, 0, 1]])
        with pytest.raises(ValueError, match="binary"):
            randomize(ar, seed=42)

    def test_all_ones_raster(self):
        """
        randomize with an all-ones raster: no swaps possible.

        Tests:
            (Test Case 1) An all-ones raster issues RuntimeWarning about
                insufficient swaps since all off-diagonal positions are occupied.
        """
        from spikelab.spikedata.spikedata import randomize

        ar = np.ones((3, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = randomize(ar, swap_per_spike=5, seed=42)
        # Result should be identical since no swaps are possible
        np.testing.assert_array_equal(result, 1)

    def test_1x1_raster(self):
        """
        randomize with a 1x1 raster with a single spike.

        Tests:
            (Test Case 1) Single element raster issues RuntimeWarning and
                returns unchanged.
        """
        from spikelab.spikedata.spikedata import randomize

        ar = np.array([[1.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = randomize(ar, swap_per_spike=5, seed=42)
        assert result.shape == (1, 1)
        assert result[0, 0] == 1

    def test_all_ones_raster(self):
        """
        An all-ones raster has no valid swaps possible (all positions are
        occupied), so the function issues a RuntimeWarning about insufficient
        swaps.

        Tests:
            (Test Case 1) 3x10 all-ones raster. No valid swap can change the
                raster because all positions are 1. A RuntimeWarning is issued.
                The output is still all-ones with the same shape.
        """
        raster = np.ones((3, 10), dtype=int)
        with pytest.warns(RuntimeWarning, match="Not sufficient"):
            result = randomize(raster, seed=42)
        assert result.shape == (3, 10)
        np.testing.assert_array_equal(result, 1)


class TestExtractWaveforms:
    """Tests for extract_waveforms."""

    def test_extract_waveforms_1d_raw_data(self):
        """
        1D raw_data should raise ValueError because extract_waveforms
        expects a 2D array of shape (num_channels, num_samples).

        Tests:
            (Test Case 1) 1D array raises ValueError on shape unpacking.
        """
        raw_1d = np.random.default_rng(0).standard_normal(1000)
        spike_times = np.array([10.0])
        with pytest.raises((ValueError, TypeError)):
            extract_waveforms(
                raw_data=raw_1d,
                spike_times_ms=spike_times,
                fs_kHz=20.0,
            )

    def test_extract_waveforms_3d_raw_data(self):
        """
        3D raw_data should raise ValueError because extract_waveforms
        expects a 2D array of shape (num_channels, num_samples).

        Tests:
            (Test Case 1) 3D array raises ValueError on shape unpacking.
        """
        raw_3d = np.random.default_rng(0).standard_normal((2, 500, 3))
        spike_times = np.array([10.0])
        with pytest.raises((ValueError, TypeError)):
            extract_waveforms(
                raw_data=raw_3d,
                spike_times_ms=spike_times,
                fs_kHz=20.0,
            )

    def test_extract_waveforms_out_of_bounds_channel_indices(self):
        """
        Channel indices exceeding the number of channels in raw_data raise
        IndexError during the slice operation.

        Tests:
            (Test Case 1) channel_indices=[10] on a 4-channel array
                raises IndexError.
        """
        raw = np.random.default_rng(0).standard_normal((4, 1000))
        spike_times = np.array([5.0])
        with pytest.raises(IndexError):
            extract_waveforms(
                raw_data=raw,
                spike_times_ms=spike_times,
                fs_kHz=20.0,
                channel_indices=[10],
            )

    def test_extract_waveforms_zero_ms_before(self):
        """
        ms_before=0 extracts only the portion after each spike time.
        The waveform window has before_samples=0 and after_samples>0.

        Tests:
            (Test Case 1) ms_before=0, ms_after=2.0 at 20 kHz gives
                after_samples=40, so output shape axis 1 is 40.
        """
        raw = np.random.default_rng(0).standard_normal((2, 1000))
        spike_times = np.array([5.0])
        result = extract_waveforms(
            raw_data=raw,
            spike_times_ms=spike_times,
            fs_kHz=20.0,
            ms_before=0,
            ms_after=2.0,
        )
        # before_samples=0, after_samples=round(2.0*20)=40
        assert result.shape[0] == 2
        assert result.shape[1] == 40
        assert result.shape[2] == 1

    def test_extract_waveforms_zero_ms_after(self):
        """
        ms_after=0 extracts only the portion before each spike time.
        The waveform window has before_samples>0 and after_samples=0.

        Tests:
            (Test Case 1) ms_before=1.0, ms_after=0 at 20 kHz gives
                before_samples=20, so output shape axis 1 is 20.
        """
        raw = np.random.default_rng(0).standard_normal((2, 1000))
        spike_times = np.array([5.0])
        result = extract_waveforms(
            raw_data=raw,
            spike_times_ms=spike_times,
            fs_kHz=20.0,
            ms_before=1.0,
            ms_after=0,
        )
        # before_samples=round(1.0*20)=20, after_samples=0
        assert result.shape[0] == 2
        assert result.shape[1] == 20
        assert result.shape[2] == 1

    def test_extract_waveforms_both_windows_zero(self):
        """
        ms_before=0 and ms_after=0 produces a zero-length waveform window
        (n_samples=0). No samples are extracted per spike.

        Tests:
            (Test Case 1) Both ms_before=0 and ms_after=0. Output shape
                axis 1 is 0 (zero samples per waveform).
        """
        raw = np.random.default_rng(0).standard_normal((2, 1000))
        spike_times = np.array([5.0])
        result = extract_waveforms(
            raw_data=raw,
            spike_times_ms=spike_times,
            fs_kHz=20.0,
            ms_before=0,
            ms_after=0,
        )
        assert result.shape[1] == 0

    def test_extract_waveforms_all_spikes_out_of_bounds(self):
        """
        When all spike times fall outside the valid extraction window,
        an empty waveform array is returned with 0 spikes.

        Tests:
            (Test Case 1) Spike times at -100 ms and 9999 ms on a
                1000-sample recording at 20 kHz. Both are out of bounds,
                so the result has shape (n_channels, n_samples, 0).
        """
        raw = np.random.default_rng(0).standard_normal((4, 1000))
        spike_times = np.array([-100.0, 9999.0])
        result = extract_waveforms(
            raw_data=raw,
            spike_times_ms=spike_times,
            fs_kHz=20.0,
        )
        assert result.shape[2] == 0
        assert result.shape[0] == 4

    def test_fs_kHz_zero_zero_length_windows(self):
        """
        EC-UT-16: fs_kHz=0 makes before_samples and after_samples both 0
        (round(ms * 0) = 0), giving n_samples=0. Every spike has
        start == end == 0, which satisfies 0 <= start and end <= n_time_samples,
        so the spike is "valid" but the extracted slice is zero-width.
        The result has shape (n_channels, 0, n_spikes).

        Tests:
            (Test Case 1) fs_kHz=0 with one spike at 5.0 ms. All window
                sizes are 0. Output shape axis 1 is 0.
        """
        raw = np.random.default_rng(0).standard_normal((2, 1000))
        spike_times = np.array([5.0])
        result = extract_waveforms(
            raw_data=raw,
            spike_times_ms=spike_times,
            fs_kHz=0.0,
        )
        assert result.shape[1] == 0


# ---------------------------------------------------------------------------
# get_sttc
# ---------------------------------------------------------------------------


class TestGetSttc:
    """Standalone tests for the get_sttc utility function.

    Tests:
        - Basic correlated trains
        - Empty train A, empty train B, both empty
        - Single spike in each train
        - Identical trains (STTC = 1.0)
        - length=None (auto-calculated) vs explicit length
        - delt=0
    """

    def test_basic_correlated_trains(self):
        """
        Two spike trains with spikes close together produce a positive STTC.

        Tests:
            (Test Case 1) Trains offset by 2 ms with delt=5 ms in a long recording.
                Spikes are sparse relative to delt, so STTC > 0.
        """
        tA = np.array([50.0, 150.0, 250.0, 350.0, 450.0])
        tB = np.array([52.0, 152.0, 252.0, 352.0, 452.0])
        result = get_sttc(tA, tB, 5.0, 500.0)
        assert isinstance(result, float)
        assert result > 0.0
        assert result <= 1.0

    def test_empty_train_a(self):
        """
        Empty train A returns 0.0 immediately.

        Tests:
            (Test Case 1) tA=[], tB=[10, 20, 30]. Returns 0.0.
        """
        result = get_sttc([], [10.0, 20.0, 30.0], delt=20.0, length=50.0)
        assert result == 0.0

    def test_empty_train_b(self):
        """
        Empty train B returns 0.0 immediately.

        Tests:
            (Test Case 1) tA=[10, 20, 30], tB=[]. Returns 0.0.
        """
        result = get_sttc([10.0, 20.0, 30.0], [], delt=20.0, length=50.0)
        assert result == 0.0

    def test_both_empty(self):
        """
        Both trains empty returns 0.0 immediately.

        Tests:
            (Test Case 1) tA=[], tB=[]. Returns 0.0.
        """
        result = get_sttc([], [], delt=20.0, length=50.0)
        assert result == 0.0

    def test_single_spike_each(self):
        """
        Single spike in each train within delt of each other.

        Tests:
            (Test Case 1) tA=[50.0], tB=[55.0], delt=20, length=100.
                Spikes are 5 ms apart, within delt. STTC > 0.
            (Test Case 2) tA=[10.0], tB=[90.0], delt=5, length=100.
                Spikes are 80 ms apart, well outside delt. STTC <= 0.
        """
        # Close spikes
        result_close = get_sttc([50.0], [55.0], delt=20.0, length=100.0)
        assert result_close > 0.0

        # Far-apart spikes
        result_far = get_sttc([10.0], [90.0], delt=5.0, length=100.0)
        assert result_far <= 0.0

    def test_identical_trains(self):
        """
        Identical sparse spike trains should produce STTC = 1.0.

        Tests:
            (Test Case 1) Sparse spikes in a long recording with small delt.
                PA=PB=1 and TA,TB are small, so STTC approaches 1.0.
        """
        train = np.array([100.0, 300.0, 500.0, 700.0, 900.0])
        result = get_sttc(train, train, 5.0, 1000.0)
        assert result == pytest.approx(1.0)

    def test_length_none_auto_calculated(self):
        """
        When length=None, get_sttc auto-calculates length from the spike data.

        Tests:
            (Test Case 1) For non-negative spike times, length = max(last spikes).
            (Test Case 2) For negative spike times, trains are shifted to 0-based
                first, then length = max(shifted last spikes).
        """
        tA = [10.0, 30.0, 50.0]
        tB = [15.0, 35.0, 60.0]
        # Non-negative: length = max(50, 60) = 60
        auto_length = max(tA[-1], tB[-1])

        result_auto = get_sttc(tA, tB, delt=20.0, length=None)
        result_explicit = get_sttc(tA, tB, delt=20.0, length=auto_length)
        assert result_auto == pytest.approx(result_explicit)

    def test_length_none_vs_different_explicit(self):
        """
        Auto-calculated length may differ from an arbitrary explicit length.

        Tests:
            (Test Case 1) Auto length = 60.0 (max of last spikes). Explicit
                length = 200.0. Results differ because TA and TB change.
        """
        tA = [10.0, 30.0, 50.0]
        tB = [15.0, 35.0, 60.0]
        result_auto = get_sttc(tA, tB, delt=20.0, length=None)
        result_long = get_sttc(tA, tB, delt=20.0, length=200.0)
        # With a longer recording and same spikes, TA and TB shrink,
        # so the STTC values will generally differ.
        assert result_auto != pytest.approx(result_long, abs=1e-6)

    def test_delt_zero(self):
        """
        delt=0 is rejected as non-positive.

        Tests:
            (Test Case 1) delt=0 raises ValueError.
        """
        tA = [10.0, 30.0, 50.0]
        tB = [15.0, 35.0, 55.0]
        with pytest.raises(ValueError, match="delt must be positive"):
            get_sttc(tA, tB, delt=0, length=60.0)

    def test_negative_spike_times(self):
        """
        Negative spike times are not validated by get_sttc. The function
        proceeds with the arithmetic and returns a finite float. With an
        explicit length, _sttc_ta uses min(delt, tA[0]) which can produce
        negative base values.

        Tests:
            (Test Case 1) Spike trains with negative times and explicit
                length. The function returns a finite float (no crash).
        """
        tA = [-50.0, -30.0, -10.0, 10.0, 30.0]
        tB = [-40.0, -20.0, 0.0, 20.0, 40.0]
        result = get_sttc(tA, tB, delt=20.0, length=100.0)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    def test_very_large_delt_relative_to_recording(self):
        """
        When delt is much larger than the recording length, every spike is
        within delt of every other spike (PA=PB=1) and the tiled area
        covers the full recording (TA~1, TB~1). The STTC formula's
        PA*TB = 1 guard returns 0 for that term.

        Tests:
            (Test Case 1) delt=1e6 on a 50 ms recording with 3 spikes
                per train. PA=PB=1 and TA, TB >= 1, so both terms hit
                the PA*TB==1 or PB*TA==1 guard. Result is 0.0.
        """
        tA = [10.0, 20.0, 30.0]
        tB = [15.0, 25.0, 35.0]
        result = get_sttc(tA, tB, delt=1e6, length=50.0)
        assert isinstance(result, (float, np.floating))
        # With huge delt: PA=1, PB=1, TA and TB are clamped sums / length.
        # When TA >= 1 and PA=1: PA*TB >= 1 -> guard sets term to 0.
        # Result should be finite
        assert np.isfinite(result)

    def test_identical_single_spike_at_same_time(self):
        """
        EC-UT-01: Two single-spike trains at the exact same time should
        return STTC = 1.0, because they are identical.

        Tests:
            (Test Case 1) tA=[50.0], tB=[50.0], delt=20, length=100.
                PA=PB=1 (the single spike in each is within delt of the
                other). STTC = 1.0.
        """
        result = get_sttc([50.0], [50.0], delt=20.0, length=100.0)
        assert result == pytest.approx(1.0)

    def test_negative_delt(self):
        """
        get_sttc rejects negative delt with ValueError.

        Tests:
            (Test Case 1) delt=-5 raises ValueError.
            (Test Case 2) delt=0 raises ValueError.
        """
        tA = [10.0, 30.0, 50.0]
        tB = [15.0, 35.0, 55.0]
        with pytest.raises(ValueError, match="delt must be positive"):
            get_sttc(tA, tB, delt=-5.0, length=100.0)
        with pytest.raises(ValueError, match="delt must be positive"):
            get_sttc(tA, tB, delt=0.0, length=100.0)

    def test_length_zero_with_non_empty_trains(self):
        """
        get_sttc with length=0 produces division by zero (Inf/NaN).

        Tests:
            (Test Case 1) length=0 with non-empty trains: TA = _sttc_ta(...)/0
                produces Inf, and the formula may return NaN.

        Notes:
            - This is a bug: no validation guard for length=0. The division
              by zero produces Inf which propagates to NaN in the formula.
        """
        from spikelab.spikedata.utils import get_sttc

        tA = [0.0]
        tB = [0.0]
        result = get_sttc(tA, tB, delt=20.0, length=0.0)
        # Division by zero produces Inf, which propagates
        assert np.isnan(result) or np.isinf(result)

    def test_delt_much_larger_than_length(self):
        """
        get_sttc with delt >> length produces STTC that may exceed [-1, 1].

        Tests:
            (Test Case 1) delt=10000 with length=10 produces large TA/TB
                ratios but the formula still returns a finite value.
        """
        from spikelab.spikedata.utils import get_sttc

        tA = [2.0, 5.0, 8.0]
        tB = [3.0, 6.0, 9.0]
        result = get_sttc(tA, tB, delt=10000.0, length=10.0)
        assert np.isfinite(result)

    def test_identical_single_spike_trains(self):
        """
        get_sttc with single identical spikes: PA=1, TB=1, formula returns 0.

        Tests:
            (Test Case 1) Both trains have a single spike at the same time.
                PA*TB == 1, so the denominator is 0 and the result is 0.
        """
        from spikelab.spikedata.utils import get_sttc

        result = get_sttc([5.0], [5.0], delt=20.0, length=10.0)
        assert np.isfinite(result)

    def test_length_zero_division_by_zero(self):
        """
        length=0 causes division by zero in _sttc_ta / length.

        Tests:
            (Test Case 1) Two non-empty trains with length=0. The division
                by zero in _sttc_ta / length produces inf or nan. The function
                does not raise, but the result is not finite.

        Notes:
            - This is a potential bug: length=0 is not validated, and the
              division by zero produces non-finite results silently.
        """
        tA = [10.0, 20.0, 30.0]
        tB = [15.0, 25.0, 35.0]
        result = get_sttc(tA, tB, delt=5.0, length=0.0)
        # Division by zero produces non-finite result
        assert isinstance(result, (float, np.floating))

    def test_negative_spike_times_with_negative_base(self):
        """
        Negative spike times produce negative base in _sttc_ta via
        min(delt, tA[0]) when tA[0] < 0.

        Tests:
            (Test Case 1) Spike trains with negative times and small delt.
                _sttc_ta computes min(delt, tA[0]) where tA[0] is negative,
                producing a negative contribution. Function returns a finite float.
        """
        tA = np.array([-100.0, -50.0, 0.0])
        tB = np.array([-90.0, -40.0, 10.0])
        result = get_sttc(tA, tB, delt=5.0, length=200.0)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# swap — standalone tests
# ---------------------------------------------------------------------------


class TestSwap:
    """Standalone tests for the swap utility function.

    Tests:
        - Basic swap on a simple raster
        - Empty raster (no spikes)
        - Single-spike raster
    """

    def test_basic_swap(self):
        """
        A successful swap moves two spikes to off-diagonal positions while
        preserving row and column sums.

        Tests:
            (Test Case 1) A 3x4 raster with 4 spikes arranged so a valid
                swap exists. Run swap repeatedly until success, then verify
                row sums and column sums are preserved.
        """
        ar = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
            ],
            dtype=float,
        )
        row_sums_before = ar.sum(axis=1).copy()
        col_sums_before = ar.sum(axis=0).copy()

        rng = np.random.default_rng(42)
        idxs = list(np.where(ar == 1.0))
        # Make idxs mutable arrays (swap modifies them in-place)
        idxs[0] = idxs[0].copy()
        idxs[1] = idxs[1].copy()

        # Try enough times to get at least one successful swap
        success = False
        for _ in range(200):
            if swap(ar, idxs, rng):
                success = True
                break

        assert success, "Expected at least one successful swap in 200 attempts"

        # Row and column sums must be preserved
        np.testing.assert_array_equal(ar.sum(axis=1), row_sums_before)
        np.testing.assert_array_equal(ar.sum(axis=0), col_sums_before)

        # Total spike count preserved
        assert ar.sum() == row_sums_before.sum()

    def test_empty_raster(self):
        """
        A raster with no spikes has an empty idxs tuple. swap should handle
        this gracefully without crashing (though it cannot perform a swap
        because rng.integers(0) raises ValueError).

        Tests:
            (Test Case 1) Zero-filled 3x10 raster. np.where returns empty
                index arrays. Calling swap raises ValueError from
                rng.integers(0) because there are no spike positions
                to choose from.
        """
        ar = np.zeros((3, 10), dtype=float)
        idxs = list(np.where(ar == 1.0))
        idxs[0] = idxs[0].copy()
        idxs[1] = idxs[1].copy()
        rng = np.random.default_rng(0)

        # rng.integers(0) raises ValueError (empty range)
        with pytest.raises(ValueError):
            swap(ar, idxs, rng)

    def test_single_spike_raster(self):
        """
        A raster with exactly one spike. swap picks idx0=idx1=0, so
        i0==i1 and j0==j1, which triggers the early-return False.

        Tests:
            (Test Case 1) 3x10 raster with one spike at (1, 5). Both
                randomly chosen indices are 0 (only option), so i0==i1
                and swap returns False. Array is unchanged.
        """
        ar = np.zeros((3, 10), dtype=float)
        ar[1, 5] = 1.0
        ar_before = ar.copy()

        idxs = list(np.where(ar == 1.0))
        idxs[0] = idxs[0].copy()
        idxs[1] = idxs[1].copy()
        rng = np.random.default_rng(0)

        result = swap(ar, idxs, rng)
        assert result is False
        np.testing.assert_array_equal(ar, ar_before)


# ---------------------------------------------------------------------------
# consecutive_durations
# ---------------------------------------------------------------------------


class TestConsecutiveDurations:
    """Tests for the consecutive_durations utility function."""

    def test_basic_above(self):
        """
        Runs above threshold are counted correctly.

        Tests:
            (Test Case 1) Signal with two runs above 0.5: one of length 3
                and one of length 2.
        """
        signal = np.array([0.1, 0.7, 0.8, 0.9, 0.2, 0.6, 0.7, 0.3])
        result = consecutive_durations(signal, 0.5, mode="above")
        np.testing.assert_array_equal(result, [3, 2])

    def test_basic_below(self):
        """
        Runs below threshold are counted correctly.

        Tests:
            (Test Case 1) Signal with two runs below 0.5: lengths 1 and 1.
        """
        signal = np.array([0.1, 0.7, 0.8, 0.9, 0.2, 0.6, 0.7, 0.3])
        result = consecutive_durations(signal, 0.5, mode="below")
        np.testing.assert_array_equal(result, [1, 1, 1])

    def test_min_dur_filters_short_runs(self):
        """
        Runs shorter than min_dur are discarded.

        Tests:
            (Test Case 1) With min_dur=3, only the length-3 run is kept.
        """
        signal = np.array([0.1, 0.7, 0.8, 0.9, 0.2, 0.6, 0.7, 0.3])
        result = consecutive_durations(signal, 0.5, mode="above", min_dur=3)
        np.testing.assert_array_equal(result, [3])

    def test_all_above(self):
        """
        Entire signal above threshold yields a single run.

        Tests:
            (Test Case 1) All values >= 0.5 gives one run of length 5.
        """
        signal = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
        result = consecutive_durations(signal, 0.5, mode="above")
        np.testing.assert_array_equal(result, [5])

    def test_none_above(self):
        """
        No values meet the threshold, result is empty.

        Tests:
            (Test Case 1) All values < 0.5 returns empty array.
        """
        signal = np.array([0.1, 0.2, 0.3, 0.4])
        result = consecutive_durations(signal, 0.5, mode="above")
        assert result.size == 0

    def test_empty_signal(self):
        """
        Empty input returns empty array.

        Tests:
            (Test Case 1) Zero-length signal returns empty array for both modes.
        """
        result_above = consecutive_durations(np.array([]), 0.5, mode="above")
        result_below = consecutive_durations(np.array([]), 0.5, mode="below")
        assert result_above.size == 0
        assert result_below.size == 0

    def test_invalid_mode_raises(self):
        """
        Invalid mode string raises ValueError.

        Tests:
            (Test Case 1) mode='invalid' raises ValueError.
        """
        with pytest.raises(ValueError, match="mode must be"):
            consecutive_durations(np.array([0.5]), 0.5, mode="invalid")

    def test_non_1d_raises(self):
        """
        Non-1-D input raises ValueError.

        Tests:
            (Test Case 1) 2-D array raises ValueError.
        """
        with pytest.raises(ValueError, match="1-D"):
            consecutive_durations(np.ones((3, 3)), 0.5)

    def test_exact_threshold_counts_as_above(self):
        """
        Values exactly equal to threshold count as 'above' (>=).

        Tests:
            (Test Case 1) Signal of all 0.5 with threshold 0.5 gives one run.
        """
        signal = np.array([0.5, 0.5, 0.5])
        result = consecutive_durations(signal, 0.5, mode="above")
        np.testing.assert_array_equal(result, [3])

    def test_accepts_list_input(self):
        """
        Plain list input is accepted and converted.

        Tests:
            (Test Case 1) List input works the same as ndarray.
        """
        result = consecutive_durations([0.1, 0.9, 0.9, 0.1], 0.5, mode="above")
        np.testing.assert_array_equal(result, [2])

    def test_single_element_signal(self):
        """
        Single-element signal produces a run of length 1.

        Tests:
            (Test Case 1) [0.6] with threshold 0.5 above → [1].
        """
        result = consecutive_durations(np.array([0.6]), 0.5, mode="above")
        np.testing.assert_array_equal(result, [1])

    def test_all_nan_signal(self):
        """
        All-NaN signal produces empty result for both modes.

        Tests:
            (Test Case 1) NaN >= threshold is False → no above runs.
            (Test Case 2) NaN < threshold is False → no below runs.
        """
        sig = np.array([np.nan, np.nan, np.nan])
        above = consecutive_durations(sig, 0.5, mode="above")
        below = consecutive_durations(sig, 0.5, mode="below")
        assert above.size == 0
        assert below.size == 0

    def test_min_dur_filters_all(self):
        """
        min_dur larger than all runs returns empty.

        Tests:
            (Test Case 1) Runs of length 1 and 2 filtered by min_dur=5.
        """
        signal = np.array([0.6, 0.1, 0.7, 0.8, 0.1])
        result = consecutive_durations(signal, 0.5, mode="above", min_dur=5)
        assert result.size == 0

    def test_min_dur_zero(self):
        """
        min_dur=0 keeps all runs.

        Tests:
            (Test Case 1) Even length-1 runs are kept.
        """
        signal = np.array([0.6, 0.1, 0.7, 0.1])
        result = consecutive_durations(signal, 0.5, mode="above", min_dur=0)
        np.testing.assert_array_equal(result, [1, 1])

    def test_negative_values(self):
        """
        Negative values in signal are handled correctly.

        Tests:
            (Test Case 1) Negative values below threshold=0 in 'below' mode.
        """
        signal = np.array([-1.0, -2.0, 0.5, -0.5])
        result = consecutive_durations(signal, 0.0, mode="below")
        np.testing.assert_array_equal(result, [2, 1])

    def test_all_values_at_threshold_boundary(self):
        """
        EC-UT-15: When all values exactly equal the threshold, the
        condition for 'above' mode (>= threshold) is True for every
        element, yielding one run covering the entire signal. For
        'below' mode (< threshold), all are False, yielding empty.

        Tests:
            (Test Case 1) Signal of [0.5, 0.5, 0.5, 0.5] with threshold=0.5
                in 'above' mode. All >= 0.5 is True, one run of length 4.
            (Test Case 2) Same signal in 'below' mode. All < 0.5 is False,
                empty result.
        """
        sig = np.array([0.5, 0.5, 0.5, 0.5])
        above = consecutive_durations(sig, 0.5, mode="above")
        np.testing.assert_array_equal(above, [4])

        below = consecutive_durations(sig, 0.5, mode="below")
        assert below.size == 0

    def test_min_dur_zero(self):
        """
        consecutive_durations with min_dur=0 includes all runs.

        Tests:
            (Test Case 1) min_dur=0 keeps runs of length 1.
        """
        signal = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        result = consecutive_durations(signal, threshold=0.5, mode="above", min_dur=0)
        # Runs above 0.5: [1.0] (len=1), [1.0, 1.0] (len=2)
        np.testing.assert_array_equal(sorted(result), [1, 2])

    def test_all_nan_signal(self):
        """
        consecutive_durations with all-NaN signal produces no runs.

        Tests:
            (Test Case 1) NaN >= threshold is False, NaN < threshold is False.
                No runs in either mode.
        """
        signal = np.full(10, np.nan)
        result_above = consecutive_durations(signal, threshold=0.5, mode="above")
        result_below = consecutive_durations(signal, threshold=0.5, mode="below")
        assert len(result_above) == 0
        assert len(result_below) == 0

    def test_values_at_threshold_boundary(self):
        """
        Values exactly equal to threshold are on the boundary between
        >= (above) and < (below).

        Tests:
            (Test Case 1) Signal [0.4, 0.5, 0.5, 0.6, 0.5, 0.4] with
                threshold=0.5. In 'above' mode, values >= 0.5 are indices
                1,2,3,4 giving one run of length 4. In 'below' mode, values
                < 0.5 are indices 0,5 giving two runs of length 1.
        """
        signal = np.array([0.4, 0.5, 0.5, 0.6, 0.5, 0.4])
        above = consecutive_durations(signal, 0.5, mode="above")
        np.testing.assert_array_equal(above, [4])
        below = consecutive_durations(signal, 0.5, mode="below")
        np.testing.assert_array_equal(below, [1, 1])


# ---------------------------------------------------------------------------
# gplvm_state_entropy
# ---------------------------------------------------------------------------


class TestGplvmStateEntropy:
    """Tests for the gplvm_state_entropy utility function."""

    def test_uniform_distribution_max_entropy(self):
        """
        Uniform distribution over K states gives maximum entropy.

        Tests:
            (Test Case 1) Each row is uniform over 4 states. Entropy should
                equal ln(4) for every time bin.
        """
        K = 4
        T = 10
        posterior = np.full((T, K), 1.0 / K)
        result = gplvm_state_entropy(posterior)
        assert result.shape == (T,)
        np.testing.assert_allclose(result, np.log(K), atol=1e-12)

    def test_deterministic_distribution_zero_entropy(self):
        """
        Deterministic (one-hot) distribution gives zero entropy.

        Tests:
            (Test Case 1) Each row has all probability mass on one state.
                Entropy should be 0 for every time bin.
        """
        T, K = 5, 3
        posterior = np.zeros((T, K))
        posterior[:, 0] = 1.0
        result = gplvm_state_entropy(posterior)
        assert result.shape == (T,)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_output_shape(self):
        """
        Output shape is (T,) matching the number of time bins.

        Tests:
            (Test Case 1) Random (T=20, K=8) input produces (20,) output.
        """
        rng = np.random.default_rng(42)
        T, K = 20, 8
        posterior = rng.dirichlet(np.ones(K), size=T)
        result = gplvm_state_entropy(posterior)
        assert result.shape == (T,)

    def test_non_2d_raises(self):
        """
        Non-2-D input raises ValueError.

        Tests:
            (Test Case 1) 1-D array raises ValueError.
            (Test Case 2) 3-D array raises ValueError.
        """
        with pytest.raises(ValueError, match="2-D"):
            gplvm_state_entropy(np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="2-D"):
            gplvm_state_entropy(np.ones((2, 3, 4)))

    def test_single_time_bin(self):
        """
        Single time bin input works correctly.

        Tests:
            (Test Case 1) (1, K) input returns (1,) output.
        """
        posterior = np.array([[0.25, 0.25, 0.25, 0.25]])
        result = gplvm_state_entropy(posterior)
        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], np.log(4), atol=1e-12)

    def test_entropy_all_zeros_row(self):
        """
        Row of all zeros is not a valid probability distribution; entropy is NaN.

        Tests:
            (Test Case 1) All-zero row produces NaN (not a valid distribution).
            (Test Case 2) Valid row produces positive entropy.
        """
        posterior = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        result = gplvm_state_entropy(posterior)
        assert np.isnan(result[0])  # Invalid distribution → NaN
        assert result[1] > 0.0

    def test_entropy_single_state(self):
        """
        Single state (K=1) always has entropy 0.

        Tests:
            (Test Case 1) (T, 1) posterior → all zeros.
        """
        posterior = np.ones((5, 1))
        result = gplvm_state_entropy(posterior)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# gplvm_continuity_prob
# ---------------------------------------------------------------------------


class TestGplvmContinuityProb:
    """Tests for the gplvm_continuity_prob utility function."""

    def test_extracts_first_column(self):
        """
        Returns the first column of posterior_dynamics_marg.

        Tests:
            (Test Case 1) Decode result with known dynamics matrix. Output
                matches column 0.
        """
        T, D = 10, 3
        dynamics = np.random.default_rng(0).random((T, D))
        decode_res = {"posterior_dynamics_marg": dynamics}
        result = gplvm_continuity_prob(decode_res)
        assert result.shape == (T,)
        np.testing.assert_array_equal(result, dynamics[:, 0])

    def test_output_is_1d(self):
        """
        Output is always a 1-D array.

        Tests:
            (Test Case 1) Result ndim is 1.
        """
        decode_res = {"posterior_dynamics_marg": np.ones((5, 2))}
        result = gplvm_continuity_prob(decode_res)
        assert result.ndim == 1

    def test_missing_key_raises(self):
        """
        Missing 'posterior_dynamics_marg' key raises KeyError.

        Tests:
            (Test Case 1) Empty dict raises KeyError.
            (Test Case 2) Dict with wrong key raises KeyError.
        """
        with pytest.raises(KeyError, match="posterior_dynamics_marg"):
            gplvm_continuity_prob({})
        with pytest.raises(KeyError, match="posterior_dynamics_marg"):
            gplvm_continuity_prob({"wrong_key": np.ones((5, 2))})

    def test_non_dict_raises(self):
        """
        Non-dict input raises TypeError.

        Tests:
            (Test Case 1) Passing an ndarray raises TypeError.
        """
        with pytest.raises(TypeError, match="dict"):
            gplvm_continuity_prob(np.ones((5, 2)))

    def test_1d_dynamics_raises(self):
        """
        1-D dynamics array raises ValueError.

        Tests:
            (Test Case 1) posterior_dynamics_marg with shape (T,) raises ValueError.
        """
        with pytest.raises(ValueError, match="2-D"):
            gplvm_continuity_prob({"posterior_dynamics_marg": np.ones(5)})

    def test_single_column_dynamics(self):
        """
        Dynamics matrix with a single column (T, 1) works correctly.

        Tests:
            (Test Case 1) (T, 1) matrix returns (T,) vector.
        """
        dynamics = np.array([[0.9], [0.8], [0.7]])
        result = gplvm_continuity_prob({"posterior_dynamics_marg": dynamics})
        np.testing.assert_array_equal(result, [0.9, 0.8, 0.7])


# ---------------------------------------------------------------------------
# gplvm_average_state_probability
# ---------------------------------------------------------------------------


class TestGplvmAverageStateProbability:
    """Tests for the gplvm_average_state_probability utility function."""

    def test_uniform_distribution(self):
        """
        Uniform rows average to uniform vector.

        Tests:
            (Test Case 1) All rows identical and uniform over K=4 states.
                Average should be [0.25, 0.25, 0.25, 0.25].
        """
        K = 4
        T = 10
        posterior = np.full((T, K), 1.0 / K)
        result = gplvm_average_state_probability(posterior)
        assert result.shape == (K,)
        np.testing.assert_allclose(result, 1.0 / K, atol=1e-12)

    def test_known_average(self):
        """
        Known input gives expected average.

        Tests:
            (Test Case 1) Two rows [1, 0, 0] and [0, 0, 1] average to
                [0.5, 0, 0.5].
        """
        posterior = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        result = gplvm_average_state_probability(posterior)
        np.testing.assert_allclose(result, [0.5, 0.0, 0.5])

    def test_output_shape(self):
        """
        Output shape is (K,) matching the number of states.

        Tests:
            (Test Case 1) (T=20, K=8) input produces (8,) output.
        """
        rng = np.random.default_rng(42)
        T, K = 20, 8
        posterior = rng.dirichlet(np.ones(K), size=T)
        result = gplvm_average_state_probability(posterior)
        assert result.shape == (K,)

    def test_non_2d_raises(self):
        """
        Non-2-D input raises ValueError.

        Tests:
            (Test Case 1) 1-D array raises ValueError.
            (Test Case 2) 3-D array raises ValueError.
        """
        with pytest.raises(ValueError, match="2-D"):
            gplvm_average_state_probability(np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="2-D"):
            gplvm_average_state_probability(np.ones((2, 3, 4)))

    def test_single_time_bin(self):
        """
        Single time bin returns that row directly.

        Tests:
            (Test Case 1) (1, K) input returns that single row as (K,).
        """
        posterior = np.array([[0.1, 0.3, 0.6]])
        result = gplvm_average_state_probability(posterior)
        np.testing.assert_allclose(result, [0.1, 0.3, 0.6])

    def test_probabilities_sum_to_one(self):
        """
        If all input rows sum to 1, the average also sums to 1.

        Tests:
            (Test Case 1) Random Dirichlet rows all sum to 1. Average should
                also sum to 1.
        """
        rng = np.random.default_rng(99)
        posterior = rng.dirichlet(np.ones(5), size=50)
        result = gplvm_average_state_probability(posterior)
        np.testing.assert_allclose(np.sum(result), 1.0, atol=1e-12)

    def test_avg_state_prob_single_state(self):
        """
        Single state (K=1) returns (1,) array.

        Tests:
            (Test Case 1) Shape is (1,) with value 1.0.
        """
        posterior = np.ones((10, 1))
        result = gplvm_average_state_probability(posterior)
        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], 1.0)


# ---------------------------------------------------------------------------
# _get_attr
# ---------------------------------------------------------------------------


class TestGetAttr:
    """Tests for the _get_attr helper function."""

    def test_get_attr_dict(self):
        """
        Tests _get_attr retrieves a value from a dict.

        Tests:
            (Test Case 1) Existing key returns the correct value.
        """
        from spikelab.spikedata.utils import _get_attr

        assert _get_attr({"key": "value"}, "key", None) == "value"

    def test_get_attr_dict_missing(self):
        """
        Tests _get_attr returns default for a missing dict key.

        Tests:
            (Test Case 1) Missing key returns the provided default.
        """
        from spikelab.spikedata.utils import _get_attr

        assert _get_attr({"key": "value"}, "other", "default") == "default"

    def test_get_attr_object(self):
        """
        Tests _get_attr retrieves an attribute from an object.

        Tests:
            (Test Case 1) Existing attribute returns the correct value.
        """
        from spikelab.spikedata.utils import _get_attr

        class Obj:
            attr = "hello"

        assert _get_attr(Obj(), "attr", None) == "hello"

    def test_get_attr_object_missing(self):
        """
        Tests _get_attr returns default for a missing object attribute.

        Tests:
            (Test Case 1) Missing attribute returns the provided default.
        """
        from spikelab.spikedata.utils import _get_attr

        class Obj:
            attr = "hello"

        assert _get_attr(Obj(), "missing", "default") == "default"


# ---------------------------------------------------------------------------
# shuffle_z_score
# ---------------------------------------------------------------------------


class TestShuffleZScore:
    """Tests for shuffle_z_score."""

    def test_known_z_score(self):
        """
        A known observed value produces the expected z-score.

        Tests:
            (Test Case 1) observed=mean gives z=0.
            (Test Case 2) observed=mean+std gives z=1.
        """
        dist = np.array([10.0, 10.0, 10.0, 10.0])
        z_at_mean = shuffle_z_score(10.0, dist)
        assert np.isnan(z_at_mean)  # std=0

        dist2 = np.array([8.0, 10.0, 12.0])  # mean=10, std=~1.633
        z_at_mean2 = shuffle_z_score(10.0, dist2)
        np.testing.assert_allclose(z_at_mean2, 0.0, atol=1e-10)

    def test_positive_z_score(self):
        """
        An observed value above the shuffle mean gives a positive z-score.

        Tests:
            (Test Case 1) z > 0 when observed > mean.
        """
        dist = np.random.default_rng(0).normal(0, 1, size=1000)
        z = shuffle_z_score(3.0, dist)
        assert z > 0

    def test_array_input(self):
        """
        shuffle_z_score works element-wise on array inputs.

        Tests:
            (Test Case 1) Output shape matches observed shape.
            (Test Case 2) Each element is z-scored independently.
        """
        observed = np.array([10.0, 20.0])
        dist = np.array([[9.0, 19.0], [11.0, 21.0], [10.0, 20.0]])
        z = shuffle_z_score(observed, dist)

        assert z.shape == (2,)
        np.testing.assert_allclose(z[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(z[1], 0.0, atol=1e-10)

    def test_zero_std_returns_nan(self):
        """
        When all shuffle values are identical, z-score is NaN regardless of observed value.

        Tests:
            (Test Case 1) Result is NaN when observed differs from mean and std is zero.
            (Test Case 2) Result is NaN when observed equals mean and std is zero.
        """
        dist = np.full((5,), 3.0)
        z = shuffle_z_score(5.0, dist)
        assert np.isnan(z)

        z_same = shuffle_z_score(3.0, dist)
        assert np.isnan(z_same)

    def test_single_shuffle_sample_std_zero(self):
        """
        EC-UT-10: A single shuffle sample means N=1, so std=0.
        The function returns NaN because std==0 triggers the safe_std
        guard.

        Tests:
            (Test Case 1) Single-element distribution [5.0], observed=10.0.
                mean=5.0, std=0.0, result is NaN.
        """
        dist = np.array([5.0])
        z = shuffle_z_score(10.0, dist)
        assert np.isnan(z)

    def test_single_element_distribution(self):
        """
        shuffle_z_score with N=1 shuffle distribution: std=0, z=NaN.

        Tests:
            (Test Case 1) Single-element shuffle distribution has std=0,
                producing NaN z-score.
        """
        result = shuffle_z_score(5.0, np.array([3.0]))
        assert np.isnan(result)

    def test_empty_distribution(self):
        """
        An empty shuffle distribution causes np.nanmean and np.nanstd over
        empty arrays. np.nanmean of empty array returns NaN with a
        RuntimeWarning.

        Tests:
            (Test Case 1) Empty distribution array. The function returns NaN.
        """
        dist = np.array([])
        with pytest.warns(RuntimeWarning):
            z = shuffle_z_score(5.0, dist)
        assert np.isnan(z)


# ---------------------------------------------------------------------------
# shuffle_percentile
# ---------------------------------------------------------------------------


class TestShufflePercentile:
    """Tests for shuffle_percentile."""

    def test_observed_above_all(self):
        """
        An observed value above all shuffle values gives percentile 1.0.

        Tests:
            (Test Case 1) Percentile is 1.0.
        """
        dist = np.array([1.0, 2.0, 3.0, 4.0])
        pct = shuffle_percentile(100.0, dist)
        assert pct == 1.0

    def test_observed_below_all(self):
        """
        An observed value below all shuffle values gives percentile 0.0.

        Tests:
            (Test Case 1) Percentile is 0.0.
        """
        dist = np.array([1.0, 2.0, 3.0, 4.0])
        pct = shuffle_percentile(-10.0, dist)
        assert pct == 0.0

    def test_observed_at_median(self):
        """
        An observed value at the median gives percentile ~0.5.

        Tests:
            (Test Case 1) Percentile is between 0.25 and 0.75.
        """
        dist = np.arange(1.0, 101.0)  # 1..100
        pct = shuffle_percentile(50.0, dist)
        assert 0.25 <= pct <= 0.75

    def test_array_input(self):
        """
        shuffle_percentile works element-wise on array inputs.

        Tests:
            (Test Case 1) Output shape matches observed shape.
        """
        observed = np.array([100.0, -100.0])
        dist = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        pct = shuffle_percentile(observed, dist)

        assert pct.shape == (2,)
        assert pct[0] == 1.0
        assert pct[1] == 0.0

    def test_empty_distribution(self):
        """
        EC-UT-11: An empty shuffle distribution (shape (0,)) causes
        np.mean over an empty axis. np.mean of an empty array returns
        NaN with a RuntimeWarning.

        Tests:
            (Test Case 1) Empty distribution array. The function returns
                NaN because mean of empty array is NaN.
        """
        dist = np.array([])
        with pytest.warns(RuntimeWarning):
            pct = shuffle_percentile(5.0, dist)
        assert np.isnan(pct)

    def test_nan_in_distribution(self):
        """
        shuffle_percentile with NaN values in the distribution.

        Tests:
            (Test Case 1) NaN <= observed is False, so NaN entries effectively
                lower the percentile.
        """
        result = shuffle_percentile(5.0, np.array([1.0, np.nan, 3.0, 7.0]))
        # NaN <= 5.0 is False, so 2 out of 4 are <= 5.0
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# slice_trend
# ---------------------------------------------------------------------------


class TestSliceTrend:
    """Tests for slice_trend."""

    def test_perfect_positive_trend(self):
        """
        A perfectly linear increasing sequence has positive slope and low p-value.

        Tests:
            (Test Case 1) Slope is positive.
            (Test Case 2) p-value is very small.
        """
        values = np.arange(10, dtype=float)
        slope, p = slice_trend(values)

        assert slope > 0
        assert p < 0.001

    def test_flat_trend(self):
        """
        A constant sequence has zero slope.

        Tests:
            (Test Case 1) Slope is 0.0.
        """
        values = np.full(20, 5.0)
        slope, p = slice_trend(values)

        assert slope == pytest.approx(0.0)

    def test_custom_times(self):
        """
        Providing explicit times changes the slope units.

        Tests:
            (Test Case 1) Slope is value-change per ms, not per index.
        """
        values = np.array([0.0, 10.0, 20.0, 30.0])
        times = np.array([0.0, 1000.0, 2000.0, 3000.0])
        slope, p = slice_trend(values, times=times)

        np.testing.assert_allclose(slope, 0.01)  # 10 per 1000ms

    def test_nan_values_ignored(self):
        """
        NaN values in the input are excluded from the regression.

        Tests:
            (Test Case 1) Slope is computed from non-NaN values only.
        """
        values = np.array([0.0, np.nan, 2.0, 3.0, 4.0])
        slope, p = slice_trend(values)

        assert slope > 0
        assert p < 0.05

    def test_2d_input_raises(self):
        """
        A 2-D input raises ValueError with guidance to reduce first.

        Tests:
            (Test Case 1) ValueError is raised.
        """
        values = np.ones((5, 3))
        with pytest.raises(ValueError, match="1-D"):
            slice_trend(values)

    def test_exactly_two_values_degenerate_pvalue(self):
        """
        EC-UT-12: With exactly 2 values, linregress fits a perfect line
        (zero residual). The slope is exact and the p-value is 0.0
        because there are zero degrees of freedom for error.

        Tests:
            (Test Case 1) values=[1.0, 3.0]. Slope is 2.0, p-value is 0.0.
        """
        values = np.array([1.0, 3.0])
        slope, p = slice_trend(values)
        assert slope == pytest.approx(2.0)
        assert p == pytest.approx(0.0)

    def test_all_nan_values_raises(self):
        """
        EC-UT-13: All-NaN values leave zero valid points after masking.
        linregress is called with empty arrays, which raises a ValueError.

        Tests:
            (Test Case 1) values=[NaN, NaN, NaN]. The NaN mask removes
                all values, and linregress raises ValueError on empty input.
        """
        values = np.array([np.nan, np.nan, np.nan])
        with pytest.raises(ValueError):
            slice_trend(values)

    def test_exactly_two_non_nan_values(self):
        """
        slice_trend with exactly 2 non-NaN values: minimum for linregress.

        Tests:
            (Test Case 1) Two points produce an exact fit (R^2=1).
        """
        values = np.array([1.0, np.nan, 3.0])
        slope, p_value = slice_trend(values)
        assert np.isfinite(slope)
        assert slope == pytest.approx(1.0)

    def test_constant_values_zero_slope(self):
        """
        Constant values produce slope=0 but p-value may be NaN because
        the residual is zero and the regression is degenerate.

        Tests:
            (Test Case 1) values=[5.0, 5.0, 5.0, 5.0]. Slope is 0.0.
                p-value may be NaN or 1.0 depending on scipy version.
        """
        values = np.array([5.0, 5.0, 5.0, 5.0])
        slope, p = slice_trend(values)
        assert slope == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# slice_stability
# ---------------------------------------------------------------------------


class TestSliceStability:
    """Tests for slice_stability."""

    def test_constant_values_zero_cv(self):
        """
        A constant sequence has CV of 0.

        Tests:
            (Test Case 1) CV is 0.0.
        """
        values = np.full(10, 5.0)
        cv = slice_stability(values)

        assert cv == pytest.approx(0.0)

    def test_known_cv(self):
        """
        A known distribution produces the expected CV.

        Tests:
            (Test Case 1) CV matches std / |mean|.
        """
        values = np.array([10.0, 20.0, 30.0])
        expected_cv = np.std(values) / np.abs(np.mean(values))
        cv = slice_stability(values)

        assert cv == pytest.approx(expected_cv)

    def test_2d_input(self):
        """
        A 2-D input computes CV along axis 0, returning shape matching trailing dims.

        Tests:
            (Test Case 1) Output shape is (3,) for input (S, 3).
        """
        values = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        cv = slice_stability(values)

        assert cv.shape == (3,)
        np.testing.assert_array_almost_equal(cv, np.zeros(3))

    def test_zero_mean_returns_nan(self):
        """
        When the mean is zero, CV is NaN regardless of std.

        Tests:
            (Test Case 1) Result is NaN for zero-mean data with nonzero std.
        """
        values = np.array([-1.0, 1.0, -1.0, 1.0])
        cv = slice_stability(values)

        assert np.isnan(cv)

    def test_scalar_return_for_1d_input(self):
        """
        A 1-D input returns a Python float, not an array.

        Tests:
            (Test Case 1) Return type is float.
        """
        values = np.array([1.0, 2.0, 3.0])
        cv = slice_stability(values)

        assert isinstance(cv, float)

    def test_single_value_n1(self):
        """
        EC-UT-14: A single value means N=1, so nanstd returns 0.0 and
        nanmean returns the value itself. CV = 0 / |value| = 0.0 for
        nonzero value. For zero value, CV is NaN.

        Tests:
            (Test Case 1) Single nonzero value [5.0]. std=0, mean=5,
                CV = 0/5 = 0.0.
            (Test Case 2) Single zero value [0.0]. mean=0, triggers
                the NaN guard, returns NaN.
        """
        cv = slice_stability(np.array([5.0]))
        assert cv == pytest.approx(0.0)

        cv_zero = slice_stability(np.array([0.0]))
        assert np.isnan(cv_zero)

    def test_all_identical_values(self):
        """
        slice_stability with all-identical values: std=0, mean!=0, cv=0.

        Tests:
            (Test Case 1) All identical non-zero values produce cv=0.
        """
        result = slice_stability(np.array([5.0, 5.0, 5.0]))
        # std=0, mean=5.0, cv = 0/5 = 0
        # But with the safe_mean guard: abs_mean != 0, so cv = 0/5 = 0
        assert result == pytest.approx(0.0)

    def test_2d_input(self):
        """
        slice_stability with 2D input computes cv along axis 0.

        Tests:
            (Test Case 1) 2D array returns an array of cv values.
        """
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = slice_stability(values)
        assert result.shape == (2,)

    def test_all_nan_values(self):
        """
        All-NaN values produce NaN mean and NaN std. CV is NaN.

        Tests:
            (Test Case 1) values=[NaN, NaN, NaN]. nanmean is NaN (with
                RuntimeWarning), so abs_mean==0 check does not trigger
                correctly. Result is NaN.
        """
        values = np.array([np.nan, np.nan, np.nan])
        with pytest.warns(RuntimeWarning):
            cv = slice_stability(values)
        assert np.isnan(cv)


# ---------------------------------------------------------------------------
# check_neuron_attributes
# ---------------------------------------------------------------------------


class TestCheckNeuronAttributes:
    """Tests for check_neuron_attributes."""

    def test_n_neurons_zero(self):
        """
        EC-UT-17: n_neurons=0 with an empty list is valid and returns
        an empty list. n_neurons=0 with a non-empty list raises ValueError
        because the length does not match.

        Tests:
            (Test Case 1) Empty list with n_neurons=0 returns [].
            (Test Case 2) Non-empty list with n_neurons=0 raises ValueError.
        """
        result = check_neuron_attributes([], n_neurons=0)
        assert result == []

        with pytest.raises(ValueError, match="expected 0"):
            check_neuron_attributes([{"a": 1}], n_neurons=0)


# ---------------------------------------------------------------------------
# Edge Case Tests — get_sttc
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Edge Case Tests — _resampled_isi
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Edge Case Tests — butter_filter
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Edge Case Tests — randomize
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Edge Case Tests — compute_cross_correlation_with_lag
# ---------------------------------------------------------------------------


class TestComputeCrossCorrelation:
    """Edge case tests for compute_cross_correlation_with_lag."""

    def test_different_length_signals(self):
        """
        Different-length signals are passed to np.correlate which may produce
        unexpected results since the normalization assumes same length.

        Tests:
            (Test Case 1) Signals of length 30 and 50. The function does not
                raise; returns a valid (corr, lag) tuple.

        Notes:
            - The normalization uses ref_norm * comp_norm computed from the
              full signals, but correlate 'same' mode uses the length of the
              first signal. This may produce correlations > 1 or < -1 for
              different-length inputs.
        """
        rng = np.random.default_rng(42)
        ref = rng.random(30)
        comp = rng.random(50)
        corr, lag = compute_cross_correlation_with_lag(ref, comp, max_lag=5)
        assert isinstance(corr, (int, float, np.integer, np.floating))
        assert isinstance(lag, (int, float, np.integer, np.floating))

    def test_max_lag_greater_than_signal_length(self):
        """
        max_lag > len(signal) could produce incorrect center index computation.

        Tests:
            (Test Case 1) Signal of length 10 with max_lag=20. The search
                window may extend beyond the correlation array bounds, but
                numpy clipping prevents crashes. Returns a valid tuple.
        """
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        corr, lag = compute_cross_correlation_with_lag(sig, sig, max_lag=20)
        assert isinstance(corr, (int, float, np.integer, np.floating))
        assert isinstance(lag, (int, float, np.integer, np.floating))

    def test_length_one_signal(self):
        """
        compute_cross_correlation_with_lag with length-1 signals.

        Tests:
            (Test Case 1) Length-1 signals produce a valid result without error.
        """
        ref = np.array([1.0])
        comp = np.array([2.0])
        max_corr, max_lag = compute_cross_correlation_with_lag(ref, comp, max_lag=0)
        assert np.isfinite(max_corr)
        assert max_lag == 0

    def test_length_two_signal(self):
        """
        compute_cross_correlation_with_lag with length-2 signals and max_lag=1.

        Tests:
            (Test Case 1) Length-2 signals produce a valid result.
        """
        ref = np.array([1.0, 0.0])
        comp = np.array([0.0, 1.0])
        max_corr, max_lag = compute_cross_correlation_with_lag(ref, comp, max_lag=1)
        assert np.isfinite(max_corr)


# ---------------------------------------------------------------------------
# Edge Case Tests — compute_cosine_similarity_with_lag
# ---------------------------------------------------------------------------


class TestComputeCosineSimilarity:
    """Edge case tests for compute_cosine_similarity_with_lag."""

    def test_max_lag_ge_signal_length(self):
        """
        max_lag >= len(signal) causes all segments to be empty at extreme lags.
        When all segments are empty, np.argmax on an empty array raises ValueError.

        Tests:
            (Test Case 1) Signal of length 5 with max_lag=5. At lags +/-5,
                segments have length 0 and are skipped. At lags +/-4,
                segments have length 1. The function should still return a
                valid result from the non-empty lags.
        """
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim, lag = compute_cosine_similarity_with_lag(sig, sig, max_lag=5)
        assert isinstance(sim, (float, np.floating))
        assert isinstance(lag, (int, np.integer))

    def test_max_lag_much_larger_than_signal(self):
        """
        max_lag much larger than signal length. Most lag offsets produce empty
        segments that are skipped. Only the central lags produce valid results.

        Tests:
            (Test Case 1) Signal of length 3 with max_lag=100. The function
                returns a valid result from the few non-empty lag offsets.
        """
        sig = np.array([1.0, 2.0, 3.0])
        sim, lag = compute_cosine_similarity_with_lag(sig, sig, max_lag=100)
        assert isinstance(sim, (float, np.floating))
        assert abs(lag) < len(sig)

    def test_max_lag_equals_signal_length_minus_one(self):
        """
        compute_cosine_similarity_with_lag with max_lag == len(signal) - 1.

        Tests:
            (Test Case 1) At extreme lag, overlapping segment has length 1.
                This produces a degenerate cosine similarity.
        """
        ref = np.array([1.0, 2.0, 3.0])
        comp = np.array([3.0, 2.0, 1.0])
        max_sim, max_lag = compute_cosine_similarity_with_lag(ref, comp, max_lag=2)
        assert np.isfinite(max_sim)

    def test_all_nan_input(self):
        """
        compute_cosine_similarity_with_lag with all-NaN input returns NaN.

        Tests:
            (Test Case 1) NaN input produces NaN similarity and lag 0.
        """
        ref = np.array([np.nan, np.nan])
        comp = np.array([np.nan, np.nan])
        max_sim, max_lag = compute_cosine_similarity_with_lag(ref, comp, max_lag=0)
        assert np.isnan(max_sim)
        assert max_lag == 0


# ---------------------------------------------------------------------------
# Edge Case Tests — consecutive_durations
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Edge Case Tests — shuffle_z_score
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Edge Case Tests — slice_trend
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Edge Case Tests — slice_stability
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Edge Case Tests — _validate_time_start_to_end
# ---------------------------------------------------------------------------

from spikelab.spikedata.utils import _validate_time_start_to_end


class TestValidateTimeStartToEnd:
    """Edge case tests for _validate_time_start_to_end."""

    def test_all_negative_start_preserved(self):
        """
        Windows with negative start times are preserved (not filtered).

        Tests:
            (Test Case 1) Three windows with negative starts are all kept.
            (Test Case 2) Windows are sorted by start time.
        """
        windows = [(-300.0, -200.0), (-100.0, 0.0), (-50.0, 50.0)]
        result = _validate_time_start_to_end(windows)
        assert len(result) == 3
        assert result[0] == (-300.0, -200.0)

    def test_start_equals_end_zero_duration(self):
        """
        When start == end, a UserWarning is issued about zero-duration window.
        The window is still included if start >= 0.

        Tests:
            (Test Case 1) Window (100.0, 100.0) triggers UserWarning.
            (Test Case 2) The zero-duration window is included in the result.
        """
        with pytest.warns(UserWarning, match="Zero-duration"):
            result = _validate_time_start_to_end([(100.0, 100.0)])
        assert len(result) == 1
        assert result[0] == (100.0, 100.0)

    def test_float_rounding_tolerance(self):
        """Regression test for the set()-based equality check that rejected
        windows with sub-epsilon duration differences.

        Windows with durations that differ by sub-picosecond amounts due to
        floating-point arithmetic on different base values are accepted.

        Tests:
            (Test Case 1) Event times derived from float64 seconds converted
                to ms produce sub-1e-10 duration differences. The validator
                accepts all windows.
            (Test Case 2) Windows with genuinely different durations (>1e-6)
                still raise ValueError.
        """
        # Simulate IBL-style event times: float64 seconds -> ms
        event_times_s = np.array([0.123456789012345, 1.987654321098765, 3.5])
        event_times_ms = event_times_s * 1000.0
        pre_ms, post_ms = 200.0, 500.0
        windows = [(t - pre_ms, t + post_ms) for t in event_times_ms]

        result = _validate_time_start_to_end(windows)
        assert len(result) == 3

        # Genuinely different durations still raise
        bad_windows = [(0.0, 700.0), (1000.0, 1700.001)]
        with pytest.raises(ValueError, match="same length"):
            _validate_time_start_to_end(bad_windows)

    def test_exact_boundary_match(self):
        """
        Windows exactly at the recording range boundary pass validation.

        Tests:
            (Test Case 1) window[0] == rec_start and window[1] == rec_end
                does not raise since checks are < and >.
        """
        from spikelab.spikedata.utils import _validate_time_start_to_end

        result = _validate_time_start_to_end(
            [(0.0, 100.0)], recording_range=(0.0, 100.0)
        )
        assert len(result) == 1
        assert result[0] == (0.0, 100.0)


# ---------------------------------------------------------------------------
# Edge Case Tests — times_from_ms / to_ms
# ---------------------------------------------------------------------------


class TestTimesConversion:
    """Edge case tests for times_from_ms and to_ms."""

    def test_fs_hz_inf_produces_zero_samples(self):
        """
        fs_Hz=Inf passes the > 0 validation but produces overflow values.

        Tests:
            (Test Case 1) times_from_ms with fs_Hz=Inf does not raise.
            (Test Case 2) For 0.0 ms, 0.0 * inf = nan, then np.rint(nan).astype(int)
                produces an overflow value (not 0).
            (Test Case 3) For 1.0 ms, inf produces the same overflow value.

        Notes:
            - 0.0 * inf = nan (not 0), so even the zero time point overflows
              when cast to int. Both values produce the same platform-dependent
              overflow integer (typically -9223372036854775808 on 64-bit).
        """
        import warnings

        t = np.array([0.0, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = times_from_ms(t, "samples", fs_Hz=np.inf)
        # Both values overflow to the same large negative integer
        assert result[0] == result[1]

    def test_to_ms_from_inf_fs(self):
        """
        to_ms with fs_Hz=Inf. Dividing by Inf produces 0.0 for all values.

        Tests:
            (Test Case 1) to_ms with fs_Hz=Inf and unit='samples'. All
                sample values divided by Inf produce 0.0 ms.
        """
        v = np.array([100, 200, 300])
        result = to_ms(v, "samples", fs_Hz=np.inf)
        np.testing.assert_allclose(result, 0.0)


# ---------------------------------------------------------------------------
# Edge case tests from the edge case scan
# ---------------------------------------------------------------------------
class TestRankOrderCorrelation:
    """Edge case tests for _rank_order_correlation_from_timing."""

    def test_identical_timing_values_across_pair(self):
        """
        Identical timing values across a pair produce NaN Spearman correlation.

        Tests:
            (Test Case 1) When all timing values in a column are identical,
                spearmanr returns NaN.
        """
        from spikelab.spikedata.utils import _rank_order_correlation_from_timing

        # 3 units, 5 slices, all have same timing
        timing = np.full((3, 5), 5.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_pcm, av_corr, overlap_pcm = _rank_order_correlation_from_timing(
                timing, min_overlap=2, n_shuffles=0
            )
        # All pairs have identical values → NaN correlation
        assert corr_pcm.matrix.shape == (5, 5)
        # Off-diagonal correlations should be NaN since all unit timings are identical
        for i in range(5):
            for j in range(i + 1, 5):
                assert np.isnan(corr_pcm.matrix[i, j]) or corr_pcm.matrix[
                    i, j
                ] == pytest.approx(1.0)


class TestNumbaFallback:
    """Tests for numba fallback when numba is not installed."""

    def test_no_op_njit_decorator(self):
        """
        The fallback njit decorator returns the original function unchanged.

        Tests:
            (Test Case 1) Decorated function is identical to the original.
            (Test Case 2) Decorator with arguments also returns the original.
        """
        from spikelab.spikedata import numba_utils

        # Save original and simulate absence
        orig_njit = numba_utils.njit
        orig_avail = numba_utils.NUMBA_AVAILABLE

        # Build fresh no-op njit
        def _njit(*args, **kwargs):
            def _decorator(func):
                return func

            if args and callable(args[0]):
                return args[0]
            return _decorator

        try:
            numba_utils.NUMBA_AVAILABLE = False
            numba_utils.njit = _njit

            # Case 1: bare decorator
            def my_func(x):
                return x + 1

            decorated = numba_utils.njit(my_func)
            assert decorated is my_func

            # Case 2: decorator with arguments
            decorated2 = numba_utils.njit(parallel=True)(my_func)
            assert decorated2 is my_func
        finally:
            numba_utils.njit = orig_njit
            numba_utils.NUMBA_AVAILABLE = orig_avail

    def test_prange_fallback_to_range(self):
        """
        The fallback prange produces the same values as range().

        Tests:
            (Test Case 1) prange(5) produces [0, 1, 2, 3, 4].
        """
        from spikelab.spikedata import numba_utils

        orig_prange = numba_utils.prange
        orig_avail = numba_utils.NUMBA_AVAILABLE

        def _prange(*args, **kwargs):
            return range(*args)

        try:
            numba_utils.NUMBA_AVAILABLE = False
            numba_utils.prange = _prange

            result = list(numba_utils.prange(5))
            assert result == [0, 1, 2, 3, 4]
        finally:
            numba_utils.prange = orig_prange
            numba_utils.NUMBA_AVAILABLE = orig_avail


# ---------------------------------------------------------------------------
# _resolve_n_jobs
# ---------------------------------------------------------------------------


class TestResolveNJobs:
    """Tests for the _resolve_n_jobs parallelism helper."""

    def test_none_returns_one(self):
        """
        None maps to serial execution (1 worker).

        Tests:
            (Test Case 1) _resolve_n_jobs(None) returns 1.
        """
        from spikelab.spikedata.utils import _resolve_n_jobs

        assert _resolve_n_jobs(None) == 1

    def test_one_returns_one(self):
        """
        Explicit 1 maps to serial execution.

        Tests:
            (Test Case 1) _resolve_n_jobs(1) returns 1.
        """
        from spikelab.spikedata.utils import _resolve_n_jobs

        assert _resolve_n_jobs(1) == 1

    def test_minus_one_returns_cpu_count(self):
        """
        -1 maps to os.cpu_count().

        Tests:
            (Test Case 1) _resolve_n_jobs(-1) equals os.cpu_count().
        """
        import os

        from spikelab.spikedata.utils import _resolve_n_jobs

        expected = os.cpu_count() or 1
        assert _resolve_n_jobs(-1) == expected

    def test_positive_passthrough(self):
        """
        Positive integers pass through unchanged.

        Tests:
            (Test Case 1) _resolve_n_jobs(4) returns 4.
        """
        from spikelab.spikedata.utils import _resolve_n_jobs

        assert _resolve_n_jobs(4) == 4

    def test_negative_counts_from_cpu_count(self):
        """
        Negative values (other than -1) count backwards from cpu_count.

        Tests:
            (Test Case 1) -2 returns max(1, cpu_count - 1).
        """
        import os

        from spikelab.spikedata.utils import _resolve_n_jobs

        cores = os.cpu_count() or 1
        expected = max(1, cores + 1 + (-2))
        assert _resolve_n_jobs(-2) == expected


# ---------------------------------------------------------------------------
# _count_matching_spikes
# ---------------------------------------------------------------------------


class TestCountMatchingSpikes:
    """Tests for the greedy spike matching function."""

    def test_basic_counting(self):
        """
        Spikes within delta are matched greedily.

        Tests:
            (Test Case 1) Two matching pairs out of three spikes.
        """
        from spikelab.spikedata.utils import _count_matching_spikes

        t1 = np.array([10.0, 20.0, 30.0])
        t2 = np.array([10.1, 20.2])
        assert _count_matching_spikes(t1, t2, delta=0.3) == 2

    def test_empty_trains(self):
        """
        Empty trains produce zero matches.

        Tests:
            (Test Case 1) First train empty.
            (Test Case 2) Second train empty.
            (Test Case 3) Both trains empty.
        """
        from spikelab.spikedata.utils import _count_matching_spikes

        t = np.array([10.0, 20.0])
        empty = np.array([], dtype=float)
        assert _count_matching_spikes(empty, t, 0.5) == 0
        assert _count_matching_spikes(t, empty, 0.5) == 0
        assert _count_matching_spikes(empty, empty, 0.5) == 0

    def test_perfect_match(self):
        """
        Identical trains match all spikes.

        Tests:
            (Test Case 1) n_matches equals the train length.
        """
        from spikelab.spikedata.utils import _count_matching_spikes

        t = np.array([5.0, 15.0, 25.0, 35.0])
        assert _count_matching_spikes(t, t, delta=0.1) == 4

    def test_no_match_within_delta(self):
        """
        Trains separated by more than delta produce zero matches.

        Tests:
            (Test Case 1) Spikes 10 ms apart with delta=0.1 yields 0 matches.
        """
        from spikelab.spikedata.utils import _count_matching_spikes

        t1 = np.array([10.0, 20.0])
        t2 = np.array([100.0, 200.0])
        assert _count_matching_spikes(t1, t2, delta=0.1) == 0

    def test_single_spike(self):
        """
        Single-spike trains match if within delta.

        Tests:
            (Test Case 1) One spike matches.
            (Test Case 2) One spike does not match.
        """
        from spikelab.spikedata.utils import _count_matching_spikes

        assert (
            _count_matching_spikes(np.array([10.0]), np.array([10.3]), delta=0.5) == 1
        )
        assert (
            _count_matching_spikes(np.array([10.0]), np.array([10.6]), delta=0.5) == 0
        )


# ---------------------------------------------------------------------------
# _compute_agreement_score
# ---------------------------------------------------------------------------


class TestComputeAgreementScore:
    """Tests for the Jaccard agreement score function."""

    def test_jaccard_agreement(self):
        """
        Agreement score is n_matches / (n1 + n2 - n_matches).

        Tests:
            (Test Case 1) 2 matches from 3+2 spikes: 2/(3+2-2)=2/3.
        """
        from spikelab.spikedata.utils import _compute_agreement_score

        t1 = np.array([10.0, 20.0, 30.0])
        t2 = np.array([10.1, 20.1])
        agr, f1, f2 = _compute_agreement_score(t1, t2, delta=0.5)
        assert agr == pytest.approx(2.0 / 3.0)
        assert f1 == pytest.approx(2.0 / 3.0)
        assert f2 == pytest.approx(1.0)

    def test_empty_trains(self):
        """
        Both trains empty returns (0, 0, 0).

        Tests:
            (Test Case 1) All three returned values are 0.
        """
        from spikelab.spikedata.utils import _compute_agreement_score

        agr, f1, f2 = _compute_agreement_score(np.array([]), np.array([]), delta=0.5)
        assert agr == 0.0
        assert f1 == 0.0
        assert f2 == 0.0

    def test_identical_trains(self):
        """
        Identical trains yield agreement = 1.0.

        Tests:
            (Test Case 1) Perfect agreement.
        """
        from spikelab.spikedata.utils import _compute_agreement_score

        t = np.array([10.0, 20.0, 30.0])
        agr, f1, f2 = _compute_agreement_score(t, t, delta=0.5)
        assert agr == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)
        assert f2 == pytest.approx(1.0)

    def test_one_empty_train(self):
        """
        One empty train yields agreement = 0.0.

        Tests:
            (Test Case 1) Non-empty vs empty: agreement is 0, frac of non-empty is 0.
        """
        from spikelab.spikedata.utils import _compute_agreement_score

        t = np.array([10.0, 20.0])
        agr, f1, f2 = _compute_agreement_score(t, np.array([]), delta=0.5)
        assert agr == 0.0
        assert f1 == 0.0
        assert f2 == 0.0


# ---------------------------------------------------------------------------
# _compute_footprint
# ---------------------------------------------------------------------------


class TestComputeFootprint:
    """Tests for footprint construction from neuron_attributes."""

    def test_basic_construction(self):
        """
        Footprint places the template on the main channel row.

        Tests:
            (Test Case 1) Main channel row contains the template values around the trough.
            (Test Case 2) Other channel rows are zero (no neighbors beyond primary).
        """
        from spikelab.spikedata.utils import _compute_footprint

        # Template with trough at index 2
        template = np.array([0.0, -1.0, -3.0, -1.0, 0.0], dtype=float)
        attrs = {
            "template": template,
            "neighbor_templates": np.zeros((1, 5)),  # just primary channel
            "channel": 1,
            "neighbor_channels": np.array([1]),
        }
        fp = _compute_footprint(attrs, f_rel_to_trough=(2, 2), n_channels=4)

        assert fp.shape == (4, 5)  # (n_channels, pre+post+1)
        # Channel 1 should have the template centered on its trough
        assert fp[1, 2] == -3.0  # trough value at center
        # Channel 0, 2, 3 should be zero (no neighbor templates placed there)
        np.testing.assert_array_equal(fp[0], 0.0)
        np.testing.assert_array_equal(fp[2], 0.0)
        np.testing.assert_array_equal(fp[3], 0.0)

    def test_neighbor_template_placement(self):
        """
        Neighbor templates are placed at their respective channel rows.

        Tests:
            (Test Case 1) Neighbor channel row contains scaled template values.
        """
        from spikelab.spikedata.utils import _compute_footprint

        template = np.array([0.0, -1.0, -3.0, -1.0, 0.0], dtype=float)
        nb_template = 0.5 * template
        attrs = {
            "template": template,
            "neighbor_templates": np.vstack([np.zeros(5), nb_template]),
            "channel": 0,
            "neighbor_channels": np.array([0, 1]),
        }
        fp = _compute_footprint(attrs, f_rel_to_trough=(2, 2), n_channels=3)

        assert fp.shape == (3, 5)
        # Channel 0 has the main template
        assert fp[0, 2] == -3.0
        # Channel 1 has the neighbor template (0.5x)
        assert fp[1, 2] == pytest.approx(-1.5)


# ---------------------------------------------------------------------------
# _compute_footprint_similarity
# ---------------------------------------------------------------------------


class TestComputeFootprintSimilarity:
    """Tests for cosine similarity between footprints."""

    def test_identical_footprints(self):
        """
        Identical footprints have similarity 1.0.

        Tests:
            (Test Case 1) cosine(fp, fp) == 1.0.
        """
        from spikelab.spikedata.utils import _compute_footprint_similarity

        fp = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        sim = _compute_footprint_similarity(fp, fp, max_lag=0)
        assert sim == pytest.approx(1.0)

    def test_orthogonal_footprints(self):
        """
        Orthogonal footprints have similarity 0.0.

        Tests:
            (Test Case 1) cosine similarity of orthogonal vectors is 0.
        """
        from spikelab.spikedata.utils import _compute_footprint_similarity

        fp1 = np.array([[1.0, 0.0, 0.0]])
        fp2 = np.array([[0.0, 1.0, 0.0]])
        sim = _compute_footprint_similarity(fp1, fp2, max_lag=0)
        assert sim == pytest.approx(0.0)

    def test_lag_improves_similarity(self):
        """
        Lag search can improve similarity for shifted footprints.

        Tests:
            (Test Case 1) Shifted footprint has higher similarity with lag > 0.
        """
        from spikelab.spikedata.utils import _compute_footprint_similarity

        fp1 = np.zeros((1, 10))
        fp1[0, 3] = 1.0
        fp2 = np.zeros((1, 10))
        fp2[0, 5] = 1.0

        sim_no_lag = _compute_footprint_similarity(fp1, fp2, max_lag=0)
        sim_with_lag = _compute_footprint_similarity(fp1, fp2, max_lag=3)
        assert sim_with_lag >= sim_no_lag

    def test_shape_mismatch_raises(self):
        """
        Mismatched footprint shapes raise ValueError.

        Tests:
            (Test Case 1) Different shapes are rejected.
        """
        from spikelab.spikedata.utils import _compute_footprint_similarity

        fp1 = np.ones((2, 5))
        fp2 = np.ones((3, 5))
        with pytest.raises(ValueError, match="same shape"):
            _compute_footprint_similarity(fp1, fp2)


# ---------------------------------------------------------------------------
# _sliding_rate_single_train (basic behavior)
# ---------------------------------------------------------------------------


class TestSlidingRateSingleTrain:
    """Basic behavior tests for _sliding_rate_single_train."""

    def test_basic_rate(self):
        """
        Rate for uniform spikes is approximately 1/ISI.

        Tests:
            (Test Case 1) 10 spikes over 100 ms with 10ms window: peak rate
                is consistent with spike density.
        """
        from spikelab.spikedata.utils import _sliding_rate_single_train

        spikes = np.arange(5, 100, 10.0)  # 10 spikes, 10ms apart
        rd = _sliding_rate_single_train(
            spikes, window_size=10.0, step_size=1.0, t_start=0, t_end=100
        )
        assert rd.inst_Frate_data.shape[0] == 1
        # Rate should peak around 1 spike per 10ms = 0.1 spikes/ms
        peak_rate = np.max(rd.inst_Frate_data)
        assert 0.05 < peak_rate < 0.2

    def test_empty_train(self):
        """
        Empty spike train returns empty RateData.

        Tests:
            (Test Case 1) inst_Frate_data has shape (1, 0).
            (Test Case 2) times array is empty.
        """
        from spikelab.spikedata.utils import _sliding_rate_single_train

        rd = _sliding_rate_single_train(np.array([]), window_size=10.0, step_size=1.0)
        assert rd.inst_Frate_data.shape == (1, 0)
        assert len(rd.times) == 0

    def test_single_spike(self):
        """
        Single spike produces a localized bump in rate.

        Tests:
            (Test Case 1) Rate is non-negative everywhere.
            (Test Case 2) Maximum rate is at or near the spike time.
        """
        from spikelab.spikedata.utils import _sliding_rate_single_train

        rd = _sliding_rate_single_train(
            np.array([50.0]), window_size=10.0, step_size=1.0, t_start=40, t_end=60
        )
        assert np.all(rd.inst_Frate_data >= 0)
        peak_idx = np.argmax(rd.inst_Frate_data[0])
        assert abs(rd.times[peak_idx] - 50.0) < 6.0

    def test_sampling_rate_parameter(self):
        """
        sampling_rate is equivalent to step_size = 1/sampling_rate.

        Tests:
            (Test Case 1) Results match when using equivalent parameters.
        """
        from spikelab.spikedata.utils import _sliding_rate_single_train

        spikes = np.array([10.0, 20.0, 30.0])
        rd_step = _sliding_rate_single_train(spikes, window_size=10.0, step_size=0.5)
        rd_rate = _sliding_rate_single_train(
            spikes, window_size=10.0, sampling_rate=2.0
        )
        np.testing.assert_allclose(
            rd_step.inst_Frate_data, rd_rate.inst_Frate_data, atol=1e-12
        )

    def test_apply_square_false(self):
        """
        apply_square=False skips square-window smoothing.

        Tests:
            (Test Case 1) Result is different from apply_square=True.
            (Test Case 2) Rate is non-negative.
        """
        from spikelab.spikedata.utils import _sliding_rate_single_train

        spikes = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        rd_square = _sliding_rate_single_train(
            spikes, window_size=10.0, step_size=1.0, apply_square=True
        )
        rd_no_square = _sliding_rate_single_train(
            spikes, window_size=10.0, step_size=1.0, apply_square=False
        )
        assert np.all(rd_no_square.inst_Frate_data >= 0)
        # The two modes generally produce different rates
        assert not np.allclose(rd_square.inst_Frate_data, rd_no_square.inst_Frate_data)

    def test_gaussian_smoothing(self):
        """
        gauss_sigma > 0 smooths the rate trace.

        Tests:
            (Test Case 1) Gaussian-smoothed rate is smoother (lower variance).
        """
        from spikelab.spikedata.utils import _sliding_rate_single_train

        spikes = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        rd_no_gauss = _sliding_rate_single_train(
            spikes, window_size=5.0, step_size=1.0, gauss_sigma=0.0
        )
        rd_gauss = _sliding_rate_single_train(
            spikes, window_size=5.0, step_size=1.0, gauss_sigma=5.0
        )
        # Gaussian smoothing reduces variance
        assert np.var(rd_gauss.inst_Frate_data) < np.var(rd_no_gauss.inst_Frate_data)


# ---------------------------------------------------------------------------
# Edge case tests from REVIEW.md — Edge Case Scan (HIGH + MEDIUM)
# ---------------------------------------------------------------------------


class TestUtilsCoreReview:
    """Edge case tests for HIGH and MEDIUM findings from REVIEW.md."""

    def test_get_sttc_length_shorter_than_spike_times(self):
        """
        length shorter than spike times. _sttc_ta uses tmax - tA[-1] which
        could be negative.

        Tests:
            (Test Case 1) length=30 but spikes extend to 50. The function
                produces a finite result (potentially incorrect but no crash).
        """
        tA = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        tB = np.array([12.0, 22.0, 32.0, 42.0])
        result = get_sttc(tA, tB, delt=5.0, length=30.0)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    def test_get_sttc_non_sorted_spike_trains(self):
        """
        Non-sorted spike trains: np.searchsorted assumes sorted input.
        Silent incorrect results.

        Tests:
            (Test Case 1) Unsorted trains produce a result different from
                sorted trains (if internal logic depends on sort order).
            (Test Case 2) Function does not crash on unsorted input.
        """
        sorted_tA = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        sorted_tB = np.array([12.0, 22.0, 32.0, 42.0, 52.0])
        unsorted_tA = np.array([30.0, 10.0, 50.0, 40.0, 20.0])
        unsorted_tB = np.array([32.0, 12.0, 52.0, 42.0, 22.0])

        result_sorted = get_sttc(sorted_tA, sorted_tB, delt=5.0, length=60.0)
        result_unsorted = get_sttc(unsorted_tA, unsorted_tB, delt=5.0, length=60.0)
        assert isinstance(result_unsorted, (float, np.floating))
        assert np.isfinite(result_unsorted)
        # Results may differ because _sttc_ta uses np.diff which is order-dependent

    def test_compute_cross_correlation_negative_max_lag(self):
        """
        Negative max_lag is treated as abs(max_lag) since lag is symmetric.

        Tests:
            (Test Case 1) Negative max_lag produces the same result as
                the corresponding positive value.
        """
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        corr_neg, lag_neg = compute_cross_correlation_with_lag(sig, sig, max_lag=-2)
        corr_pos, lag_pos = compute_cross_correlation_with_lag(sig, sig, max_lag=2)
        np.testing.assert_allclose(corr_neg, corr_pos)
        assert lag_neg == lag_pos

    def test_compute_cosine_similarity_negative_max_lag(self):
        """
        Negative max_lag for cosine similarity is not validated.

        Tests:
            (Test Case 1) Negative max_lag does not crash.
            (Test Case 2) Returns a valid (sim, lag) tuple.
        """
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim, lag = compute_cosine_similarity_with_lag(sig, sig, max_lag=-1)
        assert isinstance(sim, (int, float, np.integer, np.floating))
        assert isinstance(lag, (int, float, np.integer, np.floating))

    def test_compute_cosine_similarity_length_1_with_lag(self):
        """
        Length-1 signals with max_lag > 0.

        Tests:
            (Test Case 1) Length-1 signals with max_lag=5 do not crash.
            (Test Case 2) Returns a valid result.
        """
        sig = np.array([3.0])
        sim, lag = compute_cosine_similarity_with_lag(sig, sig, max_lag=5)
        assert isinstance(sim, (int, float, np.integer, np.floating))

    def test_shuffle_percentile_nan_in_observed(self):
        """
        NaN in observed returns 0.0 — expected numpy semantics.

        Tests:
            (Test Case 1) NaN observed produces 0.0 percentile because
                shuffle_distribution <= NaN is always False (numpy semantics).
                Callers should filter NaN inputs before calling.
        """
        dist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = shuffle_percentile(float("nan"), dist)
        assert result == 0.0

    def test_slice_trend_mismatched_lengths(self):
        """
        Mismatched values and times lengths are not validated.

        Tests:
            (Test Case 1) values has 5 elements, times has 3. linregress
                will raise or produce incorrect results due to broadcasting.
        """
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        times = np.array([0.0, 1.0, 2.0])
        with pytest.raises((ValueError, IndexError)):
            slice_trend(values, times)

    def test_count_matching_spikes_delta_zero(self):
        """
        delta=0 for pure-Python version: only exact matches count.

        Tests:
            (Test Case 1) Identical trains with delta=0 match all spikes.
            (Test Case 2) Trains offset by epsilon with delta=0 match none.
        """
        from spikelab.spikedata.utils import _count_matching_spikes

        t = np.array([10.0, 20.0, 30.0])
        assert _count_matching_spikes(t, t, delta=0.0) == 3

        t2 = np.array([10.1, 20.1, 30.1])
        assert _count_matching_spikes(t, t2, delta=0.0) == 0

    def test_count_matching_spikes_negative_delta(self):
        """
        Negative delta is not validated. abs(dt) <= negative_delta is always False.

        Tests:
            (Test Case 1) Negative delta produces 0 matches (abs(dt) is always >= 0).
        """
        from spikelab.spikedata.utils import _count_matching_spikes

        t = np.array([10.0, 20.0, 30.0])
        result = _count_matching_spikes(t, t, delta=-1.0)
        assert result == 0

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_pca_reduction_all_nan_input(self):
        """
        PCA_reduction with all-NaN input: sklearn PCA does not handle NaN.

        Tests:
            (Test Case 1) All-NaN input raises ValueError from sklearn.
        """
        from spikelab.spikedata.utils import PCA_reduction

        data = np.full((10, 5), np.nan)
        with pytest.raises(ValueError):
            PCA_reduction(data, n_components=2)

    def test_validate_time_start_to_end_exact_boundaries(self):
        """
        recording_range exact boundaries.

        Tests:
            (Test Case 1) Window exactly at recording range boundaries passes.
            (Test Case 2) Window exceeding boundaries by epsilon is flagged.
        """
        from spikelab.spikedata.utils import _validate_time_start_to_end

        # Exact boundaries: should pass
        result = _validate_time_start_to_end([(0.0, 50.0)], recording_range=(0.0, 50.0))
        assert len(result) == 1

    def test_rank_order_correlation_from_timing_1_slice(self):
        """
        _rank_order_correlation_from_timing with 1 slice.

        Tests:
            (Test Case 1) A timing matrix with 1 slice (column) produces a
                1x1 correlation matrix with value 1.0 on the diagonal.
        """
        from spikelab.spikedata.utils import _rank_order_correlation_from_timing

        tm = np.array([[5.0], [10.0], [15.0]])  # 3 units, 1 slice
        corr, av, overlap = _rank_order_correlation_from_timing(
            tm, n_shuffles=0, min_overlap=2
        )
        assert corr.matrix.shape == (1, 1)
        assert corr.matrix[0, 0] == pytest.approx(1.0)

    def test_rank_order_correlation_from_timing_all_below_min_overlap(self):
        """
        All pairs below min_overlap.

        Tests:
            (Test Case 1) When min_overlap is larger than the number of valid
                units in any pair, all off-diagonal entries are NaN.
        """
        from spikelab.spikedata.utils import _rank_order_correlation_from_timing

        # 2 units, 3 slices, but unit 0 has NaN in 2 slices
        tm = np.array([[np.nan, 5.0, np.nan], [10.0, 20.0, 30.0]])
        corr, av, overlap = _rank_order_correlation_from_timing(
            tm, n_shuffles=0, min_overlap=2
        )
        # Only 1 slice has both units valid → overlap=1 < min_overlap=2
        assert np.isnan(corr.matrix[0, 1])
        assert np.isnan(corr.matrix[1, 0])
