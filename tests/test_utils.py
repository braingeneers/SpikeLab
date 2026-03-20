"""
Tests for utility functions in spikedata/utils.py.

Covers: _cosine_sim, compute_cosine_similarity_with_lag,
compute_cross_correlation_with_lag, butter_filter, trough_between,
times_from_ms, to_ms, ensure_h5py, _train_from_i_t_list,
PCA_reduction, UMAP_reduction, UMAP_graph_communities.
"""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpikeLab.spikedata.utils import (
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

    def test_zero_vector_returns_zero(self):
        """
        A zero vector paired with any vector returns 0.0.

        Tests:
            (Test Case 1) Zero first argument returns 0.0.
            (Test Case 2) Zero second argument returns 0.0.
            (Test Case 3) Both zero returns 0.0.
        """
        a = np.array([1.0, 2.0, 3.0])
        z = np.zeros(3)
        assert _cosine_sim(z, a) == 0.0
        assert _cosine_sim(a, z) == 0.0
        assert _cosine_sim(z, z) == 0.0

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
        assert result.dtype == int

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

    def test_raises_when_missing(self, monkeypatch):
        """
        ensure_h5py raises ImportError when h5py is None.

        Tests:
            (Test Case 1) Patching h5py to None triggers ImportError.
        """
        import SpikeLab.spikedata.utils as utils_mod

        monkeypatch.setattr(utils_mod, "h5py", None)
        with pytest.raises(ImportError, match="h5py"):
            ensure_h5py()


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
        from SpikeLab.spikedata.utils import PCA_reduction

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
        from SpikeLab.spikedata.utils import PCA_reduction

        rng = np.random.default_rng(42)
        data = rng.random((50, 10))
        embedding, var_ratio, components = PCA_reduction(data, n_components=2)
        assert var_ratio[0] >= var_ratio[1]
        assert np.all(var_ratio > 0)
        assert var_ratio.sum() <= 1.0 + 1e-10


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
        from SpikeLab.spikedata.utils import UMAP_reduction

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
        import SpikeLab.spikedata.utils as utils_mod
        from SpikeLab.spikedata.utils import UMAP_reduction

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
        from SpikeLab.spikedata.utils import UMAP_graph_communities

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
        import SpikeLab.spikedata.utils as utils_mod
        from SpikeLab.spikedata.utils import UMAP_graph_communities

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


# ---------------------------------------------------------------------------
# Edge-case tests for core utils (Group 2)
# ---------------------------------------------------------------------------

from SpikeLab.spikedata.utils import (
    _resampled_isi,
    randomize,
    extract_waveforms,
    get_sttc,
    swap,
)


class TestButterFilterEdgeCases:
    """Edge-case tests for butter_filter."""

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


class TestExtractWaveformsEdgeCases:
    """Edge-case tests for extract_waveforms with invalid input shapes."""

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


# ---------------------------------------------------------------------------
# Edge-case tests for butter_filter (MED priority)
# ---------------------------------------------------------------------------


class TestButterFilterEdgeCasesMed:
    """MED-priority edge-case tests for butter_filter."""

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


# ---------------------------------------------------------------------------
# Edge-case tests for times_from_ms (MED priority)
# ---------------------------------------------------------------------------


class TestTimesFromMsEdgeCases:
    """MED-priority edge-case tests for times_from_ms."""

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
        assert result.dtype == int

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
# Edge-case tests for to_ms (MED priority)
# ---------------------------------------------------------------------------


class TestToMsEdgeCases:
    """MED-priority edge-case tests for to_ms."""

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
# Edge-case tests for get_sttc (MED priority)
# ---------------------------------------------------------------------------


class TestGetSttcEdgeCases:
    """MED-priority edge-case tests for get_sttc."""

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


# ---------------------------------------------------------------------------
# get_sttc — standalone tests
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
        When length=None, get_sttc auto-calculates length as max(tA[-1], tB[-1]).

        Tests:
            (Test Case 1) Compare auto-calculated length vs explicit length
                equal to max(tA[-1], tB[-1]). Results should match.
        """
        tA = [10.0, 30.0, 50.0]
        tB = [15.0, 35.0, 60.0]
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
        delt=0 means only exact spike-time matches count. For non-identical
        trains with no shared spike times, PA=PB=0.

        Tests:
            (Test Case 1) tA and tB with no shared times, delt=0.
                No spike in A is within 0 ms of any spike in B. STTC <= 0.
            (Test Case 2) Identical trains with delt=0. Every spike matches
                exactly, so STTC = 1.0.
        """
        tA = [10.0, 30.0, 50.0]
        tB = [15.0, 35.0, 55.0]
        result_diff = get_sttc(tA, tB, delt=0, length=60.0)
        assert result_diff <= 0.0

        # Identical trains: exact matches exist
        result_same = get_sttc(tA, tA, delt=0, length=60.0)
        assert result_same == pytest.approx(1.0)


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
        from SpikeLab.spikedata.utils import _get_attr

        assert _get_attr({"key": "value"}, "key", None) == "value"

    def test_get_attr_dict_missing(self):
        """
        Tests _get_attr returns default for a missing dict key.

        Tests:
            (Test Case 1) Missing key returns the provided default.
        """
        from SpikeLab.spikedata.utils import _get_attr

        assert _get_attr({"key": "value"}, "other", "default") == "default"

    def test_get_attr_object(self):
        """
        Tests _get_attr retrieves an attribute from an object.

        Tests:
            (Test Case 1) Existing attribute returns the correct value.
        """
        from SpikeLab.spikedata.utils import _get_attr

        class Obj:
            attr = "hello"

        assert _get_attr(Obj(), "attr", None) == "hello"

    def test_get_attr_object_missing(self):
        """
        Tests _get_attr returns default for a missing object attribute.

        Tests:
            (Test Case 1) Missing attribute returns the provided default.
        """
        from SpikeLab.spikedata.utils import _get_attr

        class Obj:
            attr = "hello"

        assert _get_attr(Obj(), "missing", "default") == "default"


# ---------------------------------------------------------------------------
# Edge case tests for new utility functions
# ---------------------------------------------------------------------------


class TestConsecutiveDurationsEdgeCases:
    """Edge case tests for consecutive_durations."""

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


class TestGplvmEdgeCases:
    """Edge case tests for GPLVM utility functions."""

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
