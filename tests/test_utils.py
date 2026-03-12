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
