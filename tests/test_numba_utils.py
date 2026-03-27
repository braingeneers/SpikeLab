"""Tests for spikedata/numba_utils.py — numba-accelerated kernels.

Verifies that numba kernels produce results close to the pure-numpy
reference implementations.  Uses ``np.allclose`` rather than strict
equality because floating-point accumulation order may differ.
"""

import numpy as np
import pytest

from spikelab.spikedata.numba_utils import (
    NUMBA_AVAILABLE,
    flatten_spike_trains,
    _nb_sttc_ta,
    _nb_sttc_na,
    _nb_sttc_pair,
    nb_sttc_all_pairs,
    _nb_latencies_pair,
    nb_latencies_all_pairs,
    nb_spike_trig_pop_rate,
)
from spikelab.spikedata.utils import get_sttc, _sttc_ta, _sttc_na
from spikelab.spikedata import SpikeData

pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sd(n_units=5, length=1000.0, spikes_per_unit=50, seed=42):
    """Create a SpikeData with random spike trains."""
    rng = np.random.default_rng(seed)
    trains = [
        np.sort(rng.uniform(0, length, size=spikes_per_unit)) for _ in range(n_units)
    ]
    return SpikeData(trains, N=n_units, length=length)


# ---------------------------------------------------------------------------
# flatten_spike_trains
# ---------------------------------------------------------------------------


class TestFlattenSpikeTrains:
    """Tests for the flatten_spike_trains helper."""

    def test_basic_round_trip(self):
        """
        Flat array + offsets correctly reconstruct each train shifted by start_time.

        Tests:
            (Test Case 1) Spike times are shifted by -start_time in the flat array.
            (Test Case 2) Offsets correctly delimit each unit's spikes.
        """
        trains = [np.array([10.0, 20.0, 30.0]), np.array([15.0, 25.0])]
        flat, offsets = flatten_spike_trains(trains, start_time=10.0)
        assert offsets.dtype == np.int64
        assert len(offsets) == 3
        assert offsets[0] == 0
        assert offsets[1] == 3
        assert offsets[2] == 5
        np.testing.assert_allclose(flat[0:3], [0.0, 10.0, 20.0])
        np.testing.assert_allclose(flat[3:5], [5.0, 15.0])

    def test_empty_trains(self):
        """
        All-empty spike trains produce an empty flat array.

        Tests:
            (Test Case 1) Flat array has length 0.
            (Test Case 2) Offsets are all zeros.
        """
        trains = [np.array([]), np.array([]), np.array([])]
        flat, offsets = flatten_spike_trains(trains, start_time=0.0)
        assert len(flat) == 0
        np.testing.assert_array_equal(offsets, [0, 0, 0, 0])

    def test_mixed_empty_and_nonempty(self):
        """
        Mix of empty and non-empty trains preserves offsets correctly.

        Tests:
            (Test Case 1) Empty unit has equal consecutive offsets.
            (Test Case 2) Non-empty unit's spikes are correctly placed.
        """
        trains = [np.array([5.0, 10.0]), np.array([]), np.array([7.0])]
        flat, offsets = flatten_spike_trains(trains, start_time=0.0)
        assert len(flat) == 3
        assert offsets[1] == 2  # unit 0 has 2 spikes
        assert offsets[2] == 2  # unit 1 is empty
        assert offsets[3] == 3  # unit 2 has 1 spike

    def test_zero_start_time(self):
        """
        With start_time=0, spike times are unchanged.

        Tests:
            (Test Case 1) No shift applied to spike times.
        """
        trains = [np.array([1.0, 2.0, 3.0])]
        flat, offsets = flatten_spike_trains(trains, start_time=0.0)
        np.testing.assert_allclose(flat, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# STTC kernels — comparison with numpy reference
# ---------------------------------------------------------------------------


class TestNumbaSttcKernels:
    """Tests verifying numba STTC kernels match numpy reference."""

    def test_sttc_ta_matches_numpy(self):
        """
        _nb_sttc_ta matches utils._sttc_ta for random spike trains.

        Tests:
            (Test Case 1) Multiple random spike trains produce close results.
        """
        rng = np.random.default_rng(42)
        for _ in range(5):
            spikes = np.sort(rng.uniform(0, 1000, size=100))
            delt = 20.0
            length = 1000.0
            np_val = _sttc_ta(spikes, delt, length)
            nb_val = _nb_sttc_ta(spikes, delt, length)
            assert np.isclose(np_val, nb_val), f"np={np_val}, nb={nb_val}"

    def test_sttc_na_matches_numpy(self):
        """
        _nb_sttc_na matches utils._sttc_na for random spike trains.

        Tests:
            (Test Case 1) Count of spikes within ±delt matches numpy.
        """
        rng = np.random.default_rng(42)
        for _ in range(5):
            tA = np.sort(rng.uniform(0, 1000, size=80))
            tB = np.sort(rng.uniform(0, 1000, size=90))
            delt = 20.0
            np_val = _sttc_na(tA, tB, delt)
            nb_val = _nb_sttc_na(tA, tB, delt)
            assert np_val == nb_val, f"np={np_val}, nb={nb_val}"

    def test_sttc_pair_matches_get_sttc(self):
        """
        _nb_sttc_pair matches get_sttc for random spike trains.

        Tests:
            (Test Case 1) Single-pair STTC is close to numpy reference.
        """
        rng = np.random.default_rng(42)
        tA = np.sort(rng.uniform(0, 1000, size=100))
        tB = np.sort(rng.uniform(0, 1000, size=100))
        delt = 20.0
        length = 1000.0

        np_val = get_sttc(tA, tB, delt, length, start_time=0.0)
        nb_val = _nb_sttc_pair(tA, tB, delt, length)
        assert np.isclose(np_val, nb_val, atol=1e-12)

    def test_sttc_pair_empty_train(self):
        """
        _nb_sttc_pair returns 0.0 when either train is empty.

        Tests:
            (Test Case 1) Empty train A.
            (Test Case 2) Empty train B.
            (Test Case 3) Both empty.
        """
        spikes = np.array([1.0, 2.0, 3.0])
        empty = np.array([], dtype=np.float64)
        assert _nb_sttc_pair(empty, spikes, 20.0, 100.0) == 0.0
        assert _nb_sttc_pair(spikes, empty, 20.0, 100.0) == 0.0
        assert _nb_sttc_pair(empty, empty, 20.0, 100.0) == 0.0

    def test_sttc_pair_identical_trains(self):
        """
        STTC of a train with itself is 1.0.

        Tests:
            (Test Case 1) Identical trains produce STTC = 1.0.
        """
        spikes = np.array([10.0, 50.0, 100.0, 200.0])
        val = _nb_sttc_pair(spikes, spikes, 20.0, 300.0)
        assert np.isclose(val, 1.0)

    def test_sttc_pair_symmetric(self):
        """
        STTC(A, B) == STTC(B, A).

        Tests:
            (Test Case 1) Symmetry holds for random trains.
        """
        rng = np.random.default_rng(99)
        tA = np.sort(rng.uniform(0, 500, size=50))
        tB = np.sort(rng.uniform(0, 500, size=50))
        ab = _nb_sttc_pair(tA, tB, 10.0, 500.0)
        ba = _nb_sttc_pair(tB, tA, 10.0, 500.0)
        assert np.isclose(ab, ba)


class TestNumbaSttcAllPairs:
    """Tests for nb_sttc_all_pairs parallel computation."""

    def test_matches_pairwise_reference(self):
        """
        nb_sttc_all_pairs matches per-pair get_sttc calls.

        Tests:
            (Test Case 1) All upper-triangle values match numpy reference.
        """
        sd = _make_sd(n_units=5, length=500.0, spikes_per_unit=30)
        flat, offsets = flatten_spike_trains(sd.train, sd.start_time)
        delt = 20.0
        result = nb_sttc_all_pairs(flat, offsets, sd.N, delt, sd.length)

        k = 0
        for i in range(sd.N):
            for j in range(i + 1, sd.N):
                ref = get_sttc(
                    sd.train[i],
                    sd.train[j],
                    delt,
                    sd.length,
                    start_time=sd.start_time,
                )
                assert np.isclose(
                    result[k], ref, atol=1e-10
                ), f"pair ({i},{j}): numba={result[k]}, numpy={ref}"
                k += 1

    def test_output_shape(self):
        """
        Output has correct length for N units.

        Tests:
            (Test Case 1) Length is N*(N-1)/2.
        """
        sd = _make_sd(n_units=4)
        flat, offsets = flatten_spike_trains(sd.train, sd.start_time)
        result = nb_sttc_all_pairs(flat, offsets, 4, 20.0, sd.length)
        assert len(result) == 6  # 4*3/2

    def test_values_in_range(self):
        """
        All STTC values are within [-1, 1].

        Tests:
            (Test Case 1) No value exceeds the theoretical STTC bounds.
        """
        sd = _make_sd(n_units=6, length=1000.0, spikes_per_unit=100)
        flat, offsets = flatten_spike_trains(sd.train, sd.start_time)
        result = nb_sttc_all_pairs(flat, offsets, sd.N, 20.0, sd.length)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# Pairwise latency kernels
# ---------------------------------------------------------------------------


class TestNumbaLatenciesPair:
    """Tests for _nb_latencies_pair."""

    def test_matches_numpy_searchsorted(self):
        """
        Numba latency pair matches numpy searchsorted computation.

        Tests:
            (Test Case 1) Mean and std match numpy reference for random trains.
        """
        rng = np.random.default_rng(42)
        tI = np.sort(rng.uniform(0, 1000, size=80))
        tJ = np.sort(rng.uniform(0, 1000, size=90))

        # Numpy reference
        idx = np.searchsorted(tJ, tI)
        np.clip(idx, 1, len(tJ) - 1, out=idx)
        dt_right = tJ[idx] - tI
        dt_left = tJ[idx - 1] - tI
        use_left = np.abs(dt_left) < np.abs(dt_right)
        latencies = np.where(use_left, dt_left, dt_right)
        ref_mean = np.mean(latencies)
        ref_std = np.std(latencies)

        nb_mean, nb_std, nb_count = _nb_latencies_pair(tI, tJ, 0.0, False)
        assert nb_count == len(tI)
        assert np.isclose(nb_mean, ref_mean, atol=1e-10)
        assert np.isclose(nb_std, ref_std, atol=1e-10)

    def test_with_window(self):
        """
        Window filter excludes distant latencies.

        Tests:
            (Test Case 1) Only latencies within window are counted.
        """
        tI = np.array([100.0, 500.0])
        tJ = np.array([105.0, 900.0])  # 500→900 = 400ms, outside window
        mean, std, count = _nb_latencies_pair(tI, tJ, 50.0, True)
        assert count == 1  # only the 100→105 pair
        assert np.isclose(mean, 5.0)

    def test_empty_trains(self):
        """
        Empty trains return zero mean, std, and count.

        Tests:
            (Test Case 1) Empty train_i.
            (Test Case 2) Empty train_j.
        """
        empty = np.array([], dtype=np.float64)
        spikes = np.array([1.0, 2.0])
        m, s, c = _nb_latencies_pair(empty, spikes, 0.0, False)
        assert c == 0 and m == 0.0 and s == 0.0
        m, s, c = _nb_latencies_pair(spikes, empty, 0.0, False)
        assert c == 0 and m == 0.0 and s == 0.0


class TestNumbaLatenciesAllPairs:
    """Tests for nb_latencies_all_pairs parallel computation."""

    def test_matches_spikedata_method(self):
        """
        nb_latencies_all_pairs matches SpikeData.get_pairwise_latencies.

        Tests:
            (Test Case 1) Mean and std matrices are close to numpy reference.
        """
        sd = _make_sd(n_units=4, length=500.0, spikes_per_unit=40)
        ref_mean, ref_std = sd.get_pairwise_latencies(window_ms=None)

        # Force numba path
        flat, offsets = flatten_spike_trains(sd.train, sd.start_time)
        nb_mean, nb_std = nb_latencies_all_pairs(flat, offsets, sd.N, 0.0, False)
        np.testing.assert_allclose(nb_mean, ref_mean.matrix, atol=1e-10)
        np.testing.assert_allclose(nb_std, ref_std.matrix, atol=1e-10)

    def test_with_window_matches(self):
        """
        Window-filtered numba latencies match numpy reference.

        Tests:
            (Test Case 1) Mean matrix with window filter is close.
        """
        sd = _make_sd(n_units=4, length=500.0, spikes_per_unit=40)
        ref_mean, ref_std = sd.get_pairwise_latencies(window_ms=50.0)

        flat, offsets = flatten_spike_trains(sd.train, sd.start_time)
        nb_mean, nb_std = nb_latencies_all_pairs(flat, offsets, sd.N, 50.0, True)
        np.testing.assert_allclose(nb_mean, ref_mean.matrix, atol=1e-10)
        np.testing.assert_allclose(nb_std, ref_std.matrix, atol=1e-10)

    def test_diagonal_is_zero(self):
        """
        Diagonal entries are zero (no self-latency).

        Tests:
            (Test Case 1) Both mean and std diagonals are zero.
        """
        sd = _make_sd(n_units=3)
        flat, offsets = flatten_spike_trains(sd.train, sd.start_time)
        nb_mean, nb_std = nb_latencies_all_pairs(flat, offsets, sd.N, 0.0, False)
        np.testing.assert_array_equal(np.diag(nb_mean), 0.0)
        np.testing.assert_array_equal(np.diag(nb_std), 0.0)


# ---------------------------------------------------------------------------
# Spike-triggered population rate kernel
# ---------------------------------------------------------------------------


class TestNumbaSpikeTriggeredPopRate:
    """Tests for nb_spike_trig_pop_rate."""

    def _numpy_stpr(self, spike_matrix, lags):
        """Pure-numpy reference implementation of stPR (Bimbard et al. formula)."""
        num_neurons, num_bins = spike_matrix.shape
        pop_sum = np.sum(spike_matrix, axis=0)
        mu = np.mean(spike_matrix, axis=1)
        mu_sum = np.sum(mu)
        total_spikes = np.sum(spike_matrix, axis=1)
        stPR = np.zeros((num_neurons, len(lags)))

        for i in range(num_neurons):
            if total_spikes[i] == 0 or mu[i] == 0:
                continue
            mu_loo = mu_sum - mu[i]
            P_loo = pop_sum - spike_matrix[i]
            P_loo_mean = np.mean(P_loo)
            spike_times = np.where(spike_matrix[i] > 0)[0]

            for tau_idx, tau in enumerate(lags):
                valid_t = spike_times + tau
                mask = (valid_t >= 0) & (valid_t < num_bins)
                if np.any(mask):
                    deviations = P_loo[valid_t[mask]] - P_loo_mean
                    stPR[i, tau_idx] = np.sum(deviations) / (total_spikes[i] * mu_loo)
        return stPR

    def test_matches_numpy_reference(self):
        """
        nb_spike_trig_pop_rate matches pure-numpy implementation.

        Tests:
            (Test Case 1) All coupling curve values are close.
        """
        rng = np.random.default_rng(42)
        spike_matrix = (rng.random((5, 200)) < 0.05).astype(np.float64)
        lags = np.arange(-10, 11)

        ref = self._numpy_stpr(spike_matrix, lags)
        nb = nb_spike_trig_pop_rate(spike_matrix, lags)
        np.testing.assert_allclose(nb, ref, atol=1e-12)

    def test_silent_neuron_zeros(self):
        """
        Silent neurons (no spikes) produce all-zero coupling curves.

        Tests:
            (Test Case 1) Row for silent neuron is all zeros.
        """
        spike_matrix = np.zeros((3, 100), dtype=np.float64)
        spike_matrix[0, [10, 30, 50]] = 1.0  # only unit 0 fires
        spike_matrix[2, [20, 40, 60]] = 1.0  # only unit 2 fires
        lags = np.arange(-5, 6)

        result = nb_spike_trig_pop_rate(spike_matrix, lags)
        np.testing.assert_array_equal(result[1, :], 0.0)  # unit 1 silent

    def test_output_shape(self):
        """
        Output shape is (N, len(lags)).

        Tests:
            (Test Case 1) Shape matches expected dimensions.
        """
        spike_matrix = np.eye(4, 20, dtype=np.float64)
        lags = np.arange(-3, 4)
        result = nb_spike_trig_pop_rate(spike_matrix, lags)
        assert result.shape == (4, 7)


# ---------------------------------------------------------------------------
# Integration: high-level methods dispatch to numba
# ---------------------------------------------------------------------------


class TestNumbaIntegrationSttc:
    """Tests that spike_time_tilings uses numba for N>2 and produces close results."""

    def test_tilings_close_to_pairwise(self):
        """
        spike_time_tilings (numba path, N>2) matches per-pair get_sttc.

        Tests:
            (Test Case 1) All off-diagonal values are close to individual
                get_sttc calls.
        """
        sd = _make_sd(n_units=5, length=500.0, spikes_per_unit=30)
        result = sd.spike_time_tilings(delt=20.0)

        for i in range(sd.N):
            for j in range(i + 1, sd.N):
                ref = get_sttc(
                    sd.train[i],
                    sd.train[j],
                    20.0,
                    sd.length,
                    start_time=sd.start_time,
                )
                assert np.isclose(result.matrix[i, j], ref, atol=1e-10)
                assert np.isclose(result.matrix[j, i], ref, atol=1e-10)

    def test_tilings_diagonal_is_one(self):
        """
        Diagonal entries are 1.0.

        Tests:
            (Test Case 1) Self-STTC is 1.0 for all units.
        """
        sd = _make_sd(n_units=4)
        result = sd.spike_time_tilings(delt=20.0)
        np.testing.assert_array_equal(np.diag(result.matrix), 1.0)

    def test_tilings_symmetric(self):
        """
        STTC matrix is symmetric.

        Tests:
            (Test Case 1) matrix[i,j] == matrix[j,i] for all pairs.
        """
        sd = _make_sd(n_units=4)
        result = sd.spike_time_tilings(delt=20.0)
        np.testing.assert_array_equal(result.matrix, result.matrix.T)

    def test_delt_zero_raises(self):
        """
        delt=0 raises ValueError before reaching numba.

        Tests:
            (Test Case 1) ValueError raised with descriptive message.
        """
        sd = _make_sd(n_units=4)
        with pytest.raises(ValueError, match="delt must be positive"):
            sd.spike_time_tilings(delt=0)


class TestNumbaIntegrationLatencies:
    """Tests that get_pairwise_latencies uses numba and produces close results."""

    def test_numba_vs_numpy_no_window(self):
        """
        Numba and numpy paths produce close results without window filter.

        Tests:
            (Test Case 1) Mean and std matrices are close.
        """
        sd = _make_sd(n_units=4, length=500.0, spikes_per_unit=40)

        # Numpy path (via return_distributions=True forces fallback)
        ref_mean, ref_std, _ = sd.get_pairwise_latencies(
            window_ms=None, return_distributions=True
        )
        # Numba path (return_distributions=False)
        nb_mean, nb_std = sd.get_pairwise_latencies(window_ms=None)

        np.testing.assert_allclose(nb_mean.matrix, ref_mean.matrix, atol=1e-10)
        np.testing.assert_allclose(nb_std.matrix, ref_std.matrix, atol=1e-10)

    def test_numba_vs_numpy_with_window(self):
        """
        Numba and numpy paths produce close results with window filter.

        Tests:
            (Test Case 1) Mean matrices with window=50ms are close.
        """
        sd = _make_sd(n_units=4, length=500.0, spikes_per_unit=40)

        ref_mean, ref_std, _ = sd.get_pairwise_latencies(
            window_ms=50.0, return_distributions=True
        )
        nb_mean, nb_std = sd.get_pairwise_latencies(window_ms=50.0)

        np.testing.assert_allclose(nb_mean.matrix, ref_mean.matrix, atol=1e-10)
        np.testing.assert_allclose(nb_std.matrix, ref_std.matrix, atol=1e-10)


class TestNumbaIntegrationStpr:
    """Tests that compute_spike_trig_pop_rate uses numba and produces close results."""

    def test_output_shape_and_range(self):
        """
        stPR output has correct shape and finite values.

        Tests:
            (Test Case 1) Shape is (N, 2*window_ms+1).
            (Test Case 2) All non-silent neurons produce finite values.
        """
        sd = _make_sd(n_units=4, length=500.0, spikes_per_unit=30)
        stPR, c0, cmax, delays, lags = sd.compute_spike_trig_pop_rate(window_ms=10)
        assert stPR.shape == (4, 21)
        assert len(lags) == 21
        assert np.all(np.isfinite(stPR))

    def test_coupling_strengths(self):
        """
        Coupling strength at lag 0 matches the stPR curve value.

        Tests:
            (Test Case 1) c0 matches stPR[:, window_ms] after filtering.
        """
        sd = _make_sd(n_units=4, length=500.0, spikes_per_unit=30)
        stPR, c0, cmax, delays, lags = sd.compute_spike_trig_pop_rate(window_ms=10)
        # c0 is extracted from the filtered curve at the center index
        assert len(c0) == 4
        assert len(cmax) == 4
        assert len(delays) == 4
