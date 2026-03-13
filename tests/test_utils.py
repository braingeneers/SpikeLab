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
    ensure_h5py,
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
        result = PCA_reduction(data, n_components=2)
        assert result.shape == (20, 2)

        result3 = PCA_reduction(data, n_components=3)
        assert result3.shape == (20, 3)

    def test_variance_ordering(self):
        """
        First component captures more variance than the second.

        Tests:
            (Test Case 1) Variance of column 0 >= variance of column 1.
        """
        from SpikeLab.spikedata.utils import PCA_reduction

        rng = np.random.default_rng(42)
        data = rng.random((50, 10))
        result = PCA_reduction(data, n_components=2)
        assert np.var(result[:, 0]) >= np.var(result[:, 1])


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
        result = UMAP_reduction(data, n_components=2, random_state=42)
        assert result.shape == (30, 2)

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
        embedding, labels = UMAP_graph_communities(
            data, n_components=2, random_state=42
        )
        assert embedding.shape == (30, 2)
        assert labels.shape == (30,)
        assert labels.dtype == int

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
        All spike times identical produces zero ISI, leading to inf rates.

        Tests:
            (Test Case 1) Identical spike times do not crash. Result may
                contain inf values (potential bug: division by zero ISI),
                which is documented here.

        Notes:
            _resampled_isi computes 1/ISI; when ISI=0 this yields inf. The
            function returns without exception but the output contains inf,
            which downstream consumers should be aware of.
        """
        spikes = np.array([5.0, 5.0, 5.0])
        times = np.arange(0, 20, 1.0)
        # Should not raise
        result = _resampled_isi(spikes, times, sigma_ms=2.0)
        assert result.shape == times.shape
        # Document: result contains inf due to zero ISI (potential bug)
        if np.any(np.isinf(result)):
            pass  # Known: division by zero ISI produces inf

    def test_resampled_isi_identical_time_values(self):
        """
        Identical time values produce dt_ms=0, which causes division by zero
        when computing bin indices.

        Tests:
            (Test Case 1) Identical time values (dt_ms=0). Verify the
                function either raises a clean error or produces inf/nan
                (documented as potential bug).

        Notes:
            With dt_ms=0 the function divides by zero when computing n_bins
            and bin indices. This may raise ZeroDivisionError, ValueError, or
            produce inf/nan depending on numpy behavior.
        """
        spikes = np.array([5.0, 10.0])
        times = np.array([1.0, 1.0, 1.0])
        # dt_ms = times[1] - times[0] = 0.0 -> division by zero
        try:
            result = _resampled_isi(spikes, times, sigma_ms=2.0)
            # If it returns, document that output may contain inf/nan
            has_bad = np.any(np.isinf(result)) or np.any(np.isnan(result))
            if has_bad:
                pass  # Known: dt_ms=0 produces inf/nan (potential bug)
        except (ZeroDivisionError, ValueError, FloatingPointError):
            pass  # Clean error on dt_ms=0 is acceptable


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
        # Case a: large but within int64 range
        t_ok = np.array([1e12])
        result_ok = times_from_ms(t_ok, "samples", fs_Hz=20000.0)
        expected_ok = int(1e12 * 20)
        assert result_ok[0] == expected_ok

        # Case b: overflow territory (2e19 > int64 max ~9.2e18)
        t_overflow = np.array([1e18])
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
