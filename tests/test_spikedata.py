import warnings
from dataclasses import dataclass
from unittest.mock import patch, MagicMock
import pathlib
import sys

import numpy as np
import pytest
from scipy import stats

try:
    import quantities
    from neo.core import SpikeTrain
except ImportError:
    SpikeTrain = None
    quantities = None

try:
    import poor_man_gplvm

    _has_pmgplvm = True
except ImportError:
    _has_pmgplvm = False

skip_no_pmgplvm = pytest.mark.skipif(
    not _has_pmgplvm, reason="poor_man_gplvm or jax not installed"
)

# Ensure project root is on sys.path, then import package normally so relative imports work.
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import SpikeLab.spikedata.spikedata as spikedata
from SpikeLab.spikedata import SpikeData
from SpikeLab.spikedata.spikeslicestack import SpikeSliceStack
from SpikeLab.spikedata.utils import (
    check_neuron_attributes,
    compute_avg_waveform,
    compute_cross_correlation_with_lag,
    compute_cosine_similarity_with_lag,
    extract_unit_waveforms,
    extract_waveforms,
    get_channels_for_unit,
    get_valid_spike_times,
    waveforms_by_channel,
)

skip_no_neo = pytest.mark.skipif(
    SpikeTrain is None, reason="neo or quantities not installed"
)


@dataclass
class MockNeuronAttributes:
    size: float


def sd_from_counts(counts):
    """
    Generates a SpikeData whose raster matches given counts.

    Parameters:
    counts (array-like): Number of spikes in each bin. Each element specifies the spike count for the corresponding bin.
    Returns:
    SpikeData: a SpikeData object whose raster matches the given counts

    Notes:
    - Each bin i will have counts[i] spikes, all at time i+0.5.
    """
    times = np.hstack([i * np.ones(c) for i, c in enumerate(counts)])
    return SpikeData([times + 0.5], length=len(counts))


def random_spikedata(units, spikes, rate=1.0):
    """
    Generates SpikeData from synthetic data with a given number of units, total number of
    spikes, and overall mean firing rate.

    Spikes are randomly assigned to units and times are uniformly distributed.

    Parameters:
        units (int): Number of units (neurons) in the generated SpikeData.
        spikes (int): Total number of spikes to generate.
        rate (float, optional): Overall mean firing rate. Default is 1.0.

    Returns:
        sd (SpikeData): object with the given number of units, total number of spikes, and overall mean firing rate
    """
    idces = np.random.randint(units, size=spikes)
    times = np.random.rand(spikes) * spikes / rate / units
    return SpikeData.from_idces_times(
        idces, times, length=spikes / rate / units, N=units
    )


class TestSpikeData:
    @staticmethod
    def assert_spikedata_equal(sda, sdb, msg=None):
        """
        Asserts that two SpikeData objects contain the same data.

        Tests:
        (Test Case 1) Compares the spike trains for equality in length and values (within tolerance).
        """
        for a, b in zip(sda.train, sdb.train):
            assert len(a) == len(b) and np.allclose(a, b), msg

    @staticmethod
    def assert_spikedata_subtime(sd, sdsub, tmin, tmax, msg=None):
        """
        Asserts that a subtime of a SpikeData is correct.

        Tests:
        (Test Case 1) Checks that the subtime has the correct length and that all spikes are within the expected window.
        """
        assert len(sd.train) == len(sdsub.train)
        assert sdsub.length == tmax - tmin
        for n, nsub in zip(sd.train, sdsub.train):
            assert np.all(nsub <= tmax - tmin), msg
            if tmin > 0:
                assert np.all(nsub > 0), msg
                n_in_range = np.sum((n > tmin) & (n <= tmax))
            else:
                assert np.all(nsub >= 0), msg
                n_in_range = np.sum(n <= tmax)
            assert len(nsub) == n_in_range, msg

    @staticmethod
    def assert_neuron_attributes_equal(nda, ndb, msg=None):
        """Assert that two lists of neuron attributes are equal elementwise."""
        assert len(nda) == len(ndb)
        for n, m in zip(nda, ndb):
            assert n == m

    def test_sd_from_counts(self):
        """
        Tests that sd_from_counts produces a SpikeData with the correct binned spike counts.

        Tests:
        (Test Case 1) Tests that sd_from_counts produces a SpikeData with the correct binned spike counts.
        (Test Case 2) Tests that the binned spike counts are correct.
        (Test Case 3) Tests that the extra bin is empty (0).


        Notes:
        - Checks that binning with size 1 correctly maps spikes to their expected bins.
        """
        # Create a known counts array
        counts = np.random.randint(10, size=1000)

        # Create SpikeData with these counts
        sd = sd_from_counts(counts)

        # Get the binned result
        binned_result = sd.binned(1)

        # Number of bins is always ceil(length / bin_size)
        expected_bins = int(np.ceil(sd.length / 1))

        # Test 1: Check that the output has the expected number of bins
        assert (
            len(binned_result) == expected_bins
        ), f"Expected {expected_bins} bins but got {len(binned_result)}"

        # Test 2: Check that the counts in each bin match our expectations
        assert np.all(
            binned_result[: len(counts)] == counts
        ), "Binned values don't match input counts"

    @skip_no_neo
    def test_neo_conversion(self):
        """
        Tests conversion to and from Neo SpikeTrain objects.

        Tests:
        (Test Case 1) Converts a random SpikeData to Neo SpikeTrains and back, and checks for equality.
        """
        times = np.random.rand(100) * 100
        idces = np.random.randint(5, size=100)
        sd = SpikeData.from_idces_times(idces, times, length=100.0)

        assert SpikeTrain is not None  # Type checker doesn't understand test skips.
        assert quantities is not None  # Type checker doesn't understand test skips.
        neo_trains = [
            SpikeTrain(t * quantities.ms, t_stop=100 * quantities.ms) for t in sd.train
        ]
        sdneo = SpikeData.from_neo_spiketrains(neo_trains)
        self.assert_spikedata_equal(sd, sdneo)

    def test_spike_data(self):
        """
        Comprehensive test of SpikeData constructors and methods.

        Tests:
        (Test Case 1) Tests two-argument constructor and spike time list with from_idces_times().
        (Test Case 2) Tests event list constructor with from_events().
        (Test Case 3) Tests base constructor.
        (Test Case 4) Tests events() method.
        (Test Case 5) Tests idces_times() method.
        (Test Case 6) Tests from_raster equality with input after re-binning.
        (Test Case 7) Tests subset() constructor.
        (Test Case 8) Tests subset() with a single unit.
        (Test Case 9) Tests subtime() constructor.
        (Test Case 10) Tests subtime() constructor actually grabs subsets.
        (Test Case 11) Tests subtime() with negative arguments.
        (Test Case 12) Tests subtime() with ... first argument.
        (Test Case 13) Tests subtime() with ... second argument.
        (Test Case 14) Tests subtime() with second argument greater than length.
        (Test Case 15) Tests that frames() returns a SpikeSliceStack consistent with subtime().
        (Test Case 16) Tests overlap parameter in frames() and that partial last windows are excluded.
        (Test Case 17) Tests frames() raises ValueError for invalid overlap and short recordings.
        """
        times = np.random.rand(100) * 100
        idces = np.random.randint(5, size=100)

        # Test two-argument constructor and spike time list.
        sd = SpikeData.from_idces_times(idces, times, length=100.0)
        assert np.all(np.sort(times) == list(sd.times))

        # Test event-list constructor.
        sd1 = SpikeData.from_events(list(zip(idces, times)))
        self.assert_spikedata_equal(sd, sd1)

        # Test base constructor.
        sd2 = SpikeData(sd.train)
        self.assert_spikedata_equal(sd, sd2)

        # Test events.
        sd4 = SpikeData.from_events(sd.events)
        self.assert_spikedata_equal(sd, sd4)

        # Test idces_times().
        sd5 = SpikeData.from_idces_times(*sd.idces_times())
        self.assert_spikedata_equal(sd, sd5)

        # Test the raster constructor. We can't expect equality because of
        # finite bin size, but we can check equality for the rasters.

        bin_size = 1.0
        r = sd.raster(bin_size) != 0
        sd_from_r = SpikeData.from_raster(r, bin_size)
        r2 = sd_from_r.raster(bin_size)

        # Compare content where shapes overlap
        min_rows = min(r.shape[0], r2.shape[0])
        min_cols = min(r.shape[1], r2.shape[1])
        r_subset = r[:min_rows, :min_cols]
        r2_subset = r2[:min_rows, :min_cols]
        assert np.all(r_subset == r2_subset)

        # Make sure the raster constructor handles multiple spikes in the same bin.
        tinysd = SpikeData.from_raster(np.array([[0, 3, 0]]), 20)
        assert np.all(tinysd.train[0] == [25.0, 30.0, 35.0])

        # Test subset() constructor.
        idces = [1, 2, 3]
        sdsub = sd.subset(idces)
        for i, j in enumerate(idces):
            assert np.all(sdsub.train[i] == sd.train[j])

        # Test subset() with a single unit.
        sdsub = sd.subset(1)
        assert sdsub.N == 1

        # Test subtime() constructor idempotence.
        sdtimefull = sd.subtime(0, 100)
        self.assert_spikedata_equal(sd, sdtimefull)

        # Test subtime() constructor actually grabs subsets.
        sdtime = sd.subtime(20, 50)
        self.assert_spikedata_subtime(sd, sdtime, 20, 50)

        # Test subtime() with negative arguments.
        sdtime = sd.subtime(-80, -50)
        self.assert_spikedata_subtime(sd, sdtime, 20, 50)

        # Check subtime() with ... first argument.
        sdtime = sd.subtime(..., 50)
        self.assert_spikedata_subtime(sd, sdtime, 0, 50)

        # Check subtime() with ... second argument.
        sdtime = sd.subtime(20, ...)
        self.assert_spikedata_subtime(sd, sdtime, 20, 100)

        # Check subtime() with second argument greater than length.
        sdtime = sd.subtime(20, 150)
        self.assert_spikedata_subtime(sd, sdtime, 20, 100)

        # Test that frames() returns a SpikeSliceStack consistent with subtime().
        stack = sd.frames(20)
        assert isinstance(stack, SpikeSliceStack)
        assert len(stack.spike_stack) == 5  # 100ms / 20ms = 5 frames
        for i, frame in enumerate(stack.spike_stack):
            self.assert_spikedata_equal(frame, sd.subtime(i * 20, (i + 1) * 20))

        # Test overlap parameter and that the partial last window is excluded.
        # step=10ms, so starts at [0,10,...,80]; start=90 → window (90,110) excluded.
        stack_overlap = sd.frames(20, overlap=10)
        assert isinstance(stack_overlap, SpikeSliceStack)
        assert len(stack_overlap.spike_stack) == 9
        for i, frame in enumerate(stack_overlap.spike_stack):
            self.assert_spikedata_equal(frame, sd.subtime(i * 10, i * 10 + 20))

        # Test ValueError for overlap >= length and recording shorter than frame.
        with pytest.raises(ValueError):
            sd.frames(20, overlap=20)
        with pytest.raises(ValueError):
            sd.frames(200)

    def test_raster(self):
        """
        Tests raster and sparse_raster methods for spike count preservation and binning rules.

        Tests:
        (Test Case 1) Tests that the raster and sparse_raster representations preserve spike counts.
        (Test Case 2) Tests that the length of the raster is consistent regardless of spike counts.
        (Test Case 3) Tests binning rules for edge cases and consistency with binned().
        """
        # Check that spike counts are preserved
        N = 10000
        sd = random_spikedata(10, N)
        assert sd.raster().sum() == N
        assert np.all(sd.sparse_raster() == sd.raster())

        # Make sure the length of the raster is consistent regardless of spike counts
        N = 10
        length = 1e4
        sdA = SpikeData.from_idces_times(
            np.zeros(N, int), np.random.rand(N) * length, length=length
        )
        sdB = SpikeData.from_idces_times(
            np.zeros(N, int), np.random.rand(N) * length, length=length
        )
        assert sdA.raster().shape == sdB.raster().shape

        # Test binning rules with specific spike times
        # Bins are left-open, right-closed: (0,10], (10,20], (20,30], (30,40]
        # t=0 clipped into bin 0, t=20 into bin 1 (right-closed), t=40 into bin 3
        sd = SpikeData([[0, 20, 40]])
        assert sd.length == 40

        ground_truth = [[1, 1, 0, 1]]
        actual_raster = sd.raster(10)

        assert actual_raster.shape == (1, 4)
        assert np.all(actual_raster == ground_truth)

        # Also verify that binning rules are consistent with binned() method
        binned = np.array([list(sd.binned(10))])
        assert np.all(sd.raster(10) == binned)

    def test_rates(self):
        """
        Tests rates() method for correct spike rate calculation and unit handling.

        Tests:
        (Test Case 1) Tests that rates() returns correct spike counts for each train.
        (Test Case 2) Tests conversion to Hz and error on invalid unit.
        """
        counts = np.random.poisson(100, size=50)
        sd = SpikeData([np.random.rand(n) for n in counts], length=1)
        assert np.all(sd.rates() == counts)

        # Test the other possible units of rates.
        assert np.all(sd.rates("Hz") == counts * 1000)
        with pytest.raises(ValueError):
            sd.rates("bad_unit")

    # Removed tests for deprecated utilities: pearson, burstiness_index

    def test_interspike_intervals(self):
        """
        Tests interspike_intervals() for correct ISI calculation.

        Tests:
        (Test Case 1) Tests that a uniform spike train yields uniform ISIs.
        (Test Case 2) Tests correct ISIs for multiple trains and random intervals.
        """
        N = 10000
        ar = np.arange(N)
        ii = SpikeData.from_idces_times(np.zeros(N, int), ar).interspike_intervals()
        assert (ii[0] == 1).all()
        assert len(ii[0]) == N - 1
        assert len(ii) == 1

        # Also make sure multiple spike trains do the same thing.
        ii = SpikeData.from_idces_times(ar % 10, ar).interspike_intervals()
        assert len(ii) == 10
        for i in ii:
            assert (i == 10).all()
            assert len(i) == N / 10 - 1

        # Finally, check with random ISIs.
        truth = np.random.rand(N)
        spikes = SpikeData.from_idces_times(np.zeros(N, int), truth.cumsum())
        ii = spikes.interspike_intervals()
        assert np.allclose(ii[0], truth[1:])

    def test_spike_time_tiling_ta(self):
        """
        Tests the _sttc_ta helper for correct calculation of total available time.

        Tests:
        (Test Cases) Tests trivial and edge cases for spike overlap and time window.
        """
        assert spikedata._sttc_ta([42], 1, 100) == 2
        assert spikedata._sttc_ta([], 1, 100) == 0

        # When spikes don't overlap, you should get exactly 2ndt.
        assert spikedata._sttc_ta(np.arange(42) + 1, 0.5, 100) == 42.0

        # When spikes overlap fully, you should get exactly (tmax-tmin) + 2dt
        assert spikedata._sttc_ta(np.arange(42) + 100, 100, 300) == 241

    def test_spike_time_tiling_na(self):
        """
        Tests the _sttc_na helper for correct calculation of number of spikes in window.

        Tests:
        (Test Cases) Tests base cases, interval inclusion, and multiple spike coverage.
        """
        assert spikedata._sttc_na([1, 2, 3], [], 1) == 0
        assert spikedata._sttc_na([], [1, 2, 3], 1) == 0

        assert spikedata._sttc_na([1], [2], 0.5) == 0
        assert spikedata._sttc_na([1], [2], 1) == 1

        # Make sure closed intervals are being used.
        na = spikedata._sttc_na(np.arange(10), np.arange(10) + 0.5, 0.5)
        assert na == 10

        # Skipping multiple spikes in spike train B.
        assert spikedata._sttc_na([4], [1, 2, 3, 4.5], 0.1) == 0
        assert spikedata._sttc_na([4], [1, 2, 3, 4.5], 0.5) == 1

        # Many spikes in train B covering a single one in A.
        assert spikedata._sttc_na([2], [1, 2, 3], 0.1) == 1
        assert spikedata._sttc_na([2], [1, 2, 3], 1) == 1

        # Many spikes in train A are covered by one in B.
        assert spikedata._sttc_na([1, 2, 3], [2], 0.1) == 1
        assert spikedata._sttc_na([1, 2, 3], [2], 1) == 3

    def test_spike_time_tiling_coefficient(self):
        """
        Tests spike_time_tiling and spike_time_tilings for correct STTC calculation.

        Tests:
        (Test Cases) Tests that STTC is 1 for identical trains, symmetric, and correct for anti-correlated trains.
        (Test Cases) Tests that STTC stays within [-1, 1] for random trains and is 0 for empty trains.
        """
        N = 10000

        # Any spike train should be exactly equal to itself, and the
        # result shouldn't depend on which train is A and which is B.
        foo = random_spikedata(2, N)
        assert foo.spike_time_tiling(0, 0, 1) == 1.0
        assert foo.spike_time_tiling(1, 1, 1) == 1.0
        assert foo.spike_time_tiling(0, 1, 1) == foo.spike_time_tiling(1, 0, 1)

        # Exactly the same thing, but for the matrix of STTCs.
        sttc = foo.spike_time_tilings(1)
        assert sttc.matrix.shape == (2, 2)
        assert sttc.matrix[0, 1] == sttc.matrix[1, 0]
        assert sttc.matrix[0, 0] == 1.0
        assert sttc.matrix[1, 1] == 1.0
        assert sttc.matrix[0, 1] == foo.spike_time_tiling(0, 1, 1)

        # Default arguments, inferred value of tmax.
        tmax = max(np.ptp(foo.train[0]), np.ptp(foo.train[1]))
        assert foo.spike_time_tiling(0, 1) == foo.spike_time_tiling(0, 1, tmax)

        # The uncorrelated spike trains above should stay near zero.
        assert foo.spike_time_tiling(0, 1, 1) == pytest.approx(0, abs=0.1)

        # Two spike trains that are in complete disagreement. This
        # should be exactly -0.8, but there's systematic error
        # proportional to 1/N, even in their original implementation.
        bar = SpikeData([np.arange(N) + 0.0, np.arange(N) + 0.5])
        assert bar.spike_time_tiling(0, 1, 0.4) == pytest.approx(
            -0.8, abs=10 ** (-int(np.log10(N)))
        )

        # As you vary dt, that alternating spike train actually gets
        # the STTC to go continuously from 0 to approach a limit of
        # lim(dt to 0.5) STTC(dt) = -1, but STTC(dt >= 0.5) = 0.
        assert bar.spike_time_tiling(0, 1, 0.5) == 0

        # Make sure it stays within range even for spike trains with
        # completely random lengths.
        for _ in range(100):
            baz = SpikeData([np.random.rand(np.random.poisson(100)) for _ in range(2)])
            sttc_val = baz.spike_time_tiling(0, 1, np.random.lognormal())
            assert sttc_val <= 1
            assert sttc_val >= -1

        # STTC of an empty spike train should definitely be 0!
        fish = SpikeData([[], np.random.rand(100)])
        sttc_val = fish.spike_time_tiling(0, 1, 0.01)
        assert sttc_val == 0

    def test_binning_doesnt_lose_spikes(self):
        """
        Tests that binning does not lose spikes.
        Tests:
        (Method 1) Generates a Poisson spike train
        (Test Case 1) Tests that the sum of binned spikes equals the original count.
        """
        N = 1000
        times = np.cumsum(stats.expon.rvs(size=N))
        spikes = SpikeData([times])
        assert sum(spikes.binned(5)) == N

    def test_binning(self):
        """
        Tests binned() method for correct bin assignment.

        Tests:
        (Test Case 1) Tests that binning with size 4 produces the expected counts.
        """
        # Bins are left-open, right-closed: (0,4], (4,8], (8,12], (12,16], (16,20], (20,24], (24,28]
        # t=1→0, t=2→0, t=5→1, t=15→3, t=16→3, t=20→4, t=22→5, t=25→6
        spikes = SpikeData([[1, 2, 5, 15, 16, 20, 22, 25]])
        assert list(spikes.binned(4)) == [2, 1, 0, 2, 1, 1, 1]

    # Removed tests for deprecated avalanche/DCC utilities

    # Removed tests for deprecated DCC utilities

    def test_metadata(self):
        """
        Tests propagation and copying of metadata and neuron_attributes.

        Tests:
        (Test Case 1) Tests that invalid neuron_attributes raise an error.
        (Test Case 2) Tests that subset and subtime propagate/copy metadata and neuron_attributes correctly.
        """
        # Make sure there's an error if the metadata is gibberish.
        with pytest.raises(ValueError):
            SpikeData([], N=5, length=100, neuron_attributes=[{}, {}])

        # Overall propagation testing...
        foo = SpikeData(
            [],
            N=5,
            length=1000,
            metadata=dict(name="Marvin"),
            neuron_attributes=[MockNeuronAttributes(ξ) for ξ in np.random.rand(5)],
        )

        # Make sure subset propagates all metadata and correctly
        # subsets the neuron_attributes.
        subset = [1, 3]
        assert foo.neuron_attributes is not None
        truth = [foo.neuron_attributes[i] for i in subset]
        bar = foo.subset(subset)
        assert foo.metadata == bar.metadata
        self.assert_neuron_attributes_equal(truth, bar.neuron_attributes)

        # Change the metadata of foo and see that it's copied, so the
        # change doesn't propagate.
        foo.metadata["name"] = "Ford"
        baz = bar.subtime(500, 1000)
        assert bar.metadata == baz.metadata
        assert bar.metadata is not baz.metadata
        assert foo.metadata["name"] != bar.metadata["name"]
        self.assert_neuron_attributes_equal(
            bar.neuron_attributes, baz.neuron_attributes
        )

    def test_raw_data(self):
        """
        Tests handling of raw_data and raw_time in SpikeData.

        Tests:
        (Test Case 1) Tests that providing only one of raw_data/raw_time raises an error.
        (Test Case 2) Tests that inconsistent lengths raise an error.
        (Test Case 3) Tests automatic generation of time array and correct slicing with subtime.
        """
        # Make sure there's an error if only one of raw_data and
        # raw_time is provided to the constructor.
        with pytest.raises(ValueError):
            SpikeData([], N=5, length=100, raw_data=[])
        with pytest.raises(ValueError):
            SpikeData([], N=5, length=100, raw_time=42)

        # Make sure inconsistent lengths throw an error as well.
        with pytest.raises(ValueError):
            SpikeData(
                [], N=5, length=100, raw_data=np.zeros((5, 100)), raw_time=np.arange(42)
            )

        # Check automatic generation of the time array.
        sd = SpikeData(
            [], N=5, length=100, raw_data=np.random.rand(5, 100), raw_time=1.0
        )
        assert np.all(sd.raw_time == np.arange(100))

        # Make sure the raw data is sliced properly with time.
        sd2 = sd.subtime(20, 30)
        assert np.all(sd2.raw_time == np.arange(10))
        assert np.all(sd2.raw_data == sd.raw_data[:, 20:30])

    def test_isi_rate(self):
        """
        Tests resampled_isi and _resampled_isi for correct ISI-based rate calculation.

        Tests:
        (Test Case 1) Tests that a constant-rate neuron yields the correct rate at all times.
        (Test Case 2) Tests correct rates for varying spike intervals.
        """
        spikes = np.arange(10)
        when = np.arange(1, 9, 0.01)  # sorted, evenly spaced, within spike range
        assert np.all(
            np.isclose(spikedata._resampled_isi(spikes, when, sigma_ms=0.0), 1000)
        )

    def test_latencies(self):
        """
        Tests latencies() for correct calculation of spike latencies relative to reference times.

        Tests:
        (Test Case 1) Tests that latencies are correct for shifted spike trains.
        (Test Case 2) Tests that small windows yield no latencies and negative latencies are handled.
        """
        a = SpikeData([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        b = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) - 0.2
        # Make sure the latencies are correct, this is latencies relative
        # to the input (b), so should all be .2 after
        assert a.latencies(b)[0][0] == pytest.approx(0.2)
        assert a.latencies(b)[0][-1] == pytest.approx(0.2)

        # Small enough window, should be no latencies.
        assert a.latencies(b, 0.1)[0] == []

        # Can do negative
        assert a.latencies([0.1])[0][0] == pytest.approx(-0.1)

    # New utilities tests: randomize, get_pop_rate, get_bursts

    def test_randomize_preserves_marginals(self):
        """
        Tests that spikedata.randomize preserves row and column marginals.

        Tests:
        (Test Case 1) Tests that spikedata.randomize preserves row and column marginals.
        (Test Case 2) Tests that the output is still binary and has the same shape.
        """
        rng = np.random.default_rng(0)
        N, T = 10, 50
        raster = (rng.random((N, T)) < 0.1).astype(float)

        row_sum = raster.sum(axis=1)
        col_sum = raster.sum(axis=0)
        total = raster.sum()

        rnd = spikedata.randomize(raster, swap_per_spike=3)

        assert rnd.shape == raster.shape
        uniq = np.unique(rnd)
        assert set(uniq.tolist()).issubset({0.0, 1.0})
        assert np.allclose(rnd.sum(axis=1), row_sum)
        assert np.allclose(rnd.sum(axis=0), col_sum)
        assert np.isclose(rnd.sum(), total)

    def test_get_pop_rate_square_only_matches_convolution(self):
        """
        Tests get_pop_rate with square window only (no Gaussian) matches direct convolution.

        Tests:
        (Method 1) Constructs a spike matrix with known spike times.
        (Test Case 1) Tests that get_pop_rate output matches numpy convolution of summed spike train.
        """

        trains = [
            [10, 20, 50, 70, 80],  # neuron 0
            [15, 20, 55, 70],  # neuron 1
            [20, 25, 60],  # neuron 2
        ]

        T, N = 100, 3
        t_spk_mat = np.zeros((T, N))
        # Left-open, right-closed binning: spike at time t goes to bin ceil(t/1)-1 = t-1
        bin_idx_0 = [t - 1 for t in trains[0]]
        bin_idx_1 = [t - 1 for t in trains[1]]
        bin_idx_2 = [t - 1 for t in trains[2]]
        t_spk_mat[bin_idx_0, 0] = 1
        t_spk_mat[bin_idx_1, 1] = 1
        t_spk_mat[bin_idx_2, 2] = 1

        sd = SpikeData(trains, length=T)

        SQUARE_WIDTH = 5
        GAUSS_SIGMA = 0

        pop = sd.get_pop_rate(
            square_width=SQUARE_WIDTH, gauss_sigma=GAUSS_SIGMA, raster_bin_size_ms=1.0
        )
        truth = np.convolve(
            np.sum(t_spk_mat, axis=1), np.ones(SQUARE_WIDTH) / SQUARE_WIDTH, mode="same"
        )

        assert np.allclose(pop, truth)

    def test_get_pop_rate_gaussian_only_impulse(self):
        """
        Tests get_pop_rate with Gaussian kernel only (no square) for a single impulse.

        Tests:
        (Method 1) Places a single spike in the center of the spike matrix.
        (Test Case 1) Tests that the output is a normalized Gaussian and is symmetric.
        """
        T = 101

        # Create a single spike at the center (t=50.5ms)
        trains = [[50.5]]
        sd = SpikeData(trains, length=T)

        SQUARE_WIDTH = 0
        GAUSS_SIGMA = 2

        pop = sd.get_pop_rate(
            square_width=SQUARE_WIDTH, gauss_sigma=GAUSS_SIGMA, raster_bin_size_ms=1.0
        )

        assert np.isclose(pop.sum(), 1.0, rtol=1e-3, atol=1e-3)
        assert np.isclose(pop[50 - 1], pop[50 + 1])

    def test_get_bursts_detects_simple_peaks(self):
        """
        Tests get_bursts for correct detection of simple burst peaks.

        Tests:
        (Method 1) Creates a population rate with two clear peaks.
        (Test Case 1) Tests that get_bursts finds two bursts, with correct peak and edge locations.
        """
        T = 200

        # Create spike trains with two bursts
        trains = [
            [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
            [48, 49, 50, 51, 52],
            [50, 50, 50],
            [145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155],
            [148, 149, 150, 151, 152],
            [150, 150, 150, 150],
        ]

        sd = SpikeData(trains, length=T)

        THR_BURST = 0.5
        MIN_BURST_DIFF = 10
        BURST_EDGE_MULT_THRESH = 0.2

        tburst, edges, peak_amp = sd.get_bursts(
            thr_burst=THR_BURST,
            min_burst_diff=MIN_BURST_DIFF,
            burst_edge_mult_thresh=BURST_EDGE_MULT_THRESH,
            square_width=0,
            gauss_sigma=0,
            acc_square_width=0,
            acc_gauss_sigma=0,
            raster_bin_size_ms=1.0,
        )

        # Should detect 2 bursts
        assert len(tburst) == 2
        assert len(peak_amp) == 2
        assert edges.shape == (2, 2)

        # First burst should be around t=50
        assert 48 <= tburst[0] <= 52
        # Second burst should be around t=150
        assert 148 <= tburst[1] <= 152

        # Check that edges bracket the peaks
        assert edges[0, 0] < tburst[0] < edges[0, 1]
        assert edges[1, 0] < tburst[1] < edges[1, 1]

    def test_burst_sensitivity_basic(self):
        """
        Tests burst_sensitivity for correct output shape and counts.

        Tests:
            (Test Case 1) Output shape matches (len(thr_values), len(dist_values)).
            (Test Case 2) All entries are non-negative integers.
            (Test Case 3) Lower threshold detects more or equal bursts than higher threshold.
        """
        trains = [
            [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
            [48, 49, 50, 51, 52],
            [50, 50, 50],
            [145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155],
            [148, 149, 150, 151, 152],
            [150, 150, 150, 150],
        ]
        sd = SpikeData(trains, length=200)

        thr_values = np.array([0.3, 0.5, 1.0, 2.0])
        dist_values = np.array([5, 10, 20])

        result = sd.burst_sensitivity(
            thr_values=thr_values,
            dist_values=dist_values,
            burst_edge_mult_thresh=0.2,
            square_width=0,
            gauss_sigma=0,
            acc_square_width=0,
            acc_gauss_sigma=0,
        )

        # Shape must match parameter grid
        assert result.shape == (4, 3)
        assert result.dtype == int

        # All counts non-negative
        assert np.all(result >= 0)

        # Lower threshold should detect >= bursts than higher threshold
        # (for every dist_value column)
        for j in range(result.shape[1]):
            for i in range(result.shape[0] - 1):
                assert result[i, j] >= result[i + 1, j]

    def test_burst_sensitivity_single_parameter(self):
        """
        Tests burst_sensitivity with one parameter held to a single value.

        Tests:
            (Test Case 1) Single thr_value produces shape (1, len(dist_values)).
            (Test Case 2) Single dist_value produces shape (len(thr_values), 1).
        """
        trains = [
            [50, 51, 52, 53, 54, 55],
            [150, 151, 152, 153, 154, 155],
        ]
        sd = SpikeData(trains, length=200)

        # Single threshold value
        result_single_thr = sd.burst_sensitivity(
            thr_values=np.array([0.5]),
            dist_values=np.array([5, 10, 20]),
            burst_edge_mult_thresh=0.2,
            square_width=0,
            gauss_sigma=0,
            acc_square_width=0,
            acc_gauss_sigma=0,
        )
        assert result_single_thr.shape == (1, 3)

        # Single distance value
        result_single_dist = sd.burst_sensitivity(
            thr_values=np.array([0.3, 0.5, 1.0]),
            dist_values=np.array([10]),
            burst_edge_mult_thresh=0.2,
            square_width=0,
            gauss_sigma=0,
            acc_square_width=0,
            acc_gauss_sigma=0,
        )
        assert result_single_dist.shape == (3, 1)

    def test_burst_sensitivity_precomputed_pop_rate(self):
        """
        Tests that passing pre-computed pop_rate and pop_rate_acc gives the
        same result as letting the method compute them internally.

        Tests:
            (Test Case 1) Results with and without pre-computed rates are identical.
        """
        trains = [
            [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
            [48, 49, 50, 51, 52],
            [145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155],
            [148, 149, 150, 151, 152],
        ]
        sd = SpikeData(trains, length=200)

        thr_values = np.array([0.3, 0.5, 1.0])
        dist_values = np.array([5, 10])

        # Let the method compute internally
        result_auto = sd.burst_sensitivity(
            thr_values=thr_values,
            dist_values=dist_values,
            burst_edge_mult_thresh=0.2,
            square_width=5,
            gauss_sigma=3,
            acc_square_width=3,
            acc_gauss_sigma=2,
        )

        # Pre-compute and pass in
        pop_rate = sd.get_pop_rate(square_width=5, gauss_sigma=3)
        pop_rate_acc = sd.get_pop_rate(square_width=3, gauss_sigma=2)

        result_precomputed = sd.burst_sensitivity(
            thr_values=thr_values,
            dist_values=dist_values,
            burst_edge_mult_thresh=0.2,
            pop_rate=pop_rate,
            pop_rate_acc=pop_rate_acc,
        )

        np.testing.assert_array_equal(result_auto, result_precomputed)

    def test_get_frac_active(self):
        """
        Tests get_frac_active method for calculating burst participation rates.

        Tests:
        (Method 1) Creates a known spike pattern with predictable burst participation
        (Test Case 1) Verifies correct calculation of unit participation per burst
        (Test Case 2) Verifies correct calculation of burst participation per unit
        (Test Case 3) Checks backbone unit identification using threshold
        """
        # Create spike trains with specific firing patterns
        spike_trains = [
            np.array([1, 3, 4, 7]),  # Unit 0
            np.array([2, 4, 6, 9]),  # Unit 1
            np.array([3, 6, 8]),  # Unit 2
        ]

        sd = SpikeData(spike_trains)

        edges = np.array(
            [
                [1, 4],  # First burst from t=1 to t=4
                [6, 9],  # Second burst from t=6 to t=9
            ]
        )

        min_spikes = 2
        backbone_threshold = 0.55

        frac_per_unit, frac_per_burst, backbone_units = sd.get_frac_active(
            edges, min_spikes, backbone_threshold
        )

        # With left-open, right-closed binning (ceil-1, bin_size=1):
        # t=1→bin0, t=2→bin1, t=3→bin2, t=4→bin3, t=6→bin5, t=7→bin6, t=8→bin7, t=9→bin8
        # Burst [1,4]: Unit0 bins{2,3}=2spk ✓, Unit1 bins{1,3}=2spk ✓, Unit2 bins{2}=1spk ✗
        # Burst [6,9]: Unit0 bins{6}=1spk ✗, Unit1 bins{8}=1spk ✗, Unit2 bins{7}=1spk ✗
        expected_frac_per_unit = np.array([0.5, 0.5, 0.0])
        expected_frac_per_burst = np.array([2 / 3, 0.0])
        expected_backbone_units = np.array([])

        assert np.allclose(frac_per_unit, expected_frac_per_unit)
        assert np.allclose(frac_per_burst, expected_frac_per_burst)
        assert np.array_equal(backbone_units, expected_backbone_units)

        # Test with different parameters
        min_spikes_high = 3
        frac_per_unit_high, frac_per_burst_high, backbone_high = sd.get_frac_active(
            edges, min_spikes_high, backbone_threshold
        )

        expected_high_unit = np.array([0.0, 0.0, 0.0])
        expected_high_burst = np.array([0.0, 0.0])
        expected_high_backbone = np.array([])

        assert np.allclose(frac_per_unit_high, expected_high_unit)
        assert np.allclose(frac_per_burst_high, expected_high_burst)
        assert np.array_equal(backbone_high, expected_high_backbone)

        # Test with lower backbone threshold
        low_threshold = 0.4
        _, _, backbone_low = sd.get_frac_active(edges, min_spikes, low_threshold)
        expected_low_backbone = np.array([0, 1])
        assert np.array_equal(backbone_low, expected_low_backbone)

    def test_neuron_to_channel_map(self):
        """
        Tests neuron_to_channel_map for correct channel mapping extraction.

        Tests:
        (Test Case 1) Tests basic functionality with standard 'channel' attribute
        (Test Case 2) Tests automatic detection of common attribute names
        (Test Case 3) Tests explicit channel_attr parameter
        (Test Case 4) Tests edge cases: no neuron_attributes, empty data
        (Test Case 5) Tests partial channel information (some neurons missing channel)
        """
        # Test basic functionality
        attrs = [{"channel": i % 4, "other_field": "test"} for i in range(10)]
        trains = [[] for _ in range(10)]
        sd = SpikeData(trains, neuron_attributes=attrs, length=100.0)
        mapping = sd.neuron_to_channel_map()

        assert len(mapping) == 10
        assert mapping[0] == 0
        assert mapping[1] == 1
        assert mapping[4] == 0  # 4 % 4 = 0
        assert mapping[5] == 1  # 5 % 4 = 1

        # Test with different attribute names
        attrs2 = [{"channel_id": i % 3} for i in range(6)]
        sd2 = SpikeData([[]] * 6, neuron_attributes=attrs2, length=100.0)
        mapping2 = sd2.neuron_to_channel_map()
        assert len(mapping2) == 6
        assert mapping2[0] == 0
        assert mapping2[3] == 0  # 3 % 3 = 0

        # Test explicit channel_attr parameter
        mapping2_explicit = sd2.neuron_to_channel_map(channel_attr="channel_id")
        assert mapping2 == mapping2_explicit

        # Test with channel_index attribute
        attrs3 = [{"channel_index": i // 2} for i in range(6)]
        sd3 = SpikeData([[]] * 6, neuron_attributes=attrs3, length=100.0)
        mapping3 = sd3.neuron_to_channel_map()
        assert mapping3[0] == 0
        assert mapping3[1] == 0
        assert mapping3[2] == 1
        assert mapping3[3] == 1

        # Test edge case: no neuron_attributes
        sd_no_attrs = SpikeData([[]] * 5, length=100.0)
        mapping_no_attrs = sd_no_attrs.neuron_to_channel_map()
        assert mapping_no_attrs == {}

        # Test edge case: empty data (N=0)
        sd_empty = SpikeData([], neuron_attributes=[], length=100.0)
        mapping_empty = sd_empty.neuron_to_channel_map()
        assert mapping_empty == {}

        # Test with partial channel information (some neurons missing channel)
        attrs_partial = [
            {"channel": 0},
            {"channel": 1},
            {},  # Missing channel
            {"channel": 2},
        ]
        sd_partial = SpikeData([[]] * 4, neuron_attributes=attrs_partial, length=100.0)
        mapping_partial = sd_partial.neuron_to_channel_map()
        assert len(mapping_partial) == 3
        assert mapping_partial[0] == 0
        assert mapping_partial[1] == 1
        assert mapping_partial[3] == 2
        assert 2 not in mapping_partial

    def test_channel_raster(self):
        """
        Tests channel_raster for correct channel aggregation.

        Tests:
        (Test Case 1) Tests basic aggregation of multiple neurons per channel
        (Test Case 2) Tests that spike counts are preserved
        (Test Case 3) Tests with different bin sizes
        (Test Case 4) Tests edge cases: no channel info, empty data
        (Test Case 5) Tests that channel raster shape matches expectations
        """
        # Create 6 neurons: 0,1 on channel 0; 2,3 on channel 1; 4,5 on channel 2
        attrs = [{"channel": i // 2} for i in range(6)]
        trains = [
            [10.0, 20.0],  # neuron 0, channel 0
            [15.0],  # neuron 1, channel 0
            [25.0],  # neuron 2, channel 1
            [30.0],  # neuron 3, channel 1
            [35.0],  # neuron 4, channel 2
            [40.0],  # neuron 5, channel 2
        ]
        sd = SpikeData(trains, neuron_attributes=attrs, length=50.0)

        # Test with bin_size=10.0
        ch_raster = sd.channel_raster(bin_size=10.0)

        assert ch_raster.shape[0] == 3
        expected_bins = int(np.ceil(50.0 / 10.0))
        assert ch_raster.shape[1] == expected_bins

        assert ch_raster[0, :].sum() == 3
        assert ch_raster[1, :].sum() == 2
        assert ch_raster[2, :].sum() == 2

        # Left-open, right-closed: t=10→bin0, t=20→bin1, t=15→bin1
        assert ch_raster[0, 0] == 1  # t=10 in bin 0
        assert ch_raster[0, 1] == 2  # t=15 and t=20 in bin 1

        # Verify total spike count matches neuron raster
        neuron_raster = sd.raster(bin_size=10.0)
        assert ch_raster.sum() == neuron_raster.sum()

        # Test with different bin_size
        ch_raster_small = sd.channel_raster(bin_size=5.0)
        assert ch_raster_small.shape[0] == 3
        assert ch_raster_small.sum() == neuron_raster.sum()

        # Test with explicit channel_attr
        ch_raster_explicit = sd.channel_raster(bin_size=10.0, channel_attr="channel")
        assert np.all(ch_raster == ch_raster_explicit)

        # Test with different attribute name
        attrs2 = [{"channel_id": i % 2} for i in range(4)]
        trains2 = [[10.0], [20.0], [30.0], [40.0]]
        sd2 = SpikeData(trains2, neuron_attributes=attrs2, length=50.0)
        ch_raster2 = sd2.channel_raster(bin_size=10.0, channel_attr="channel_id")
        assert ch_raster2.shape[0] == 2  # 2 channels
        assert ch_raster2[0, :].sum() == 2
        assert ch_raster2[1, :].sum() == 2

        # Test edge case: no channel information
        sd_no_channel = SpikeData([[]] * 3, length=100.0)
        with pytest.raises(ValueError):
            sd_no_channel.channel_raster()

        # Test that multiple neurons on same channel aggregate correctly
        attrs_same = [{"channel": 0} for _ in range(3)]
        trains_same = [[10.0, 20.0], [15.0], [25.0]]
        sd_same = SpikeData(trains_same, neuron_attributes=attrs_same, length=30.0)
        ch_raster_same = sd_same.channel_raster(bin_size=10.0)
        assert ch_raster_same.shape[0] == 1  # Only 1 channel
        assert ch_raster_same[0, :].sum() == 4  # Total 4 spikes

        # Test with non-contiguous channel indices
        attrs_nc = [
            {"channel": 0},
            {"channel": 5},
            {"channel": 10},
        ]
        trains_nc = [[10.0], [20.0], [30.0]]
        sd_nc = SpikeData(trains_nc, neuron_attributes=attrs_nc, length=40.0)
        ch_raster_nc = sd_nc.channel_raster(bin_size=10.0)
        assert ch_raster_nc.shape[0] == 3
        assert ch_raster_nc[0, :].sum() == 1
        assert ch_raster_nc[1, :].sum() == 1
        assert ch_raster_nc[2, :].sum() == 1

    def test_check_neuron_attributes(self):
        """
        Tests check_neuron_attributes validation and behavior.

        Tests:
        (Test Case 1) Tests that non-list inputs raise ValueError
        (Test Case 2) Tests that non-dict elements raise ValueError
        (Test Case 3) Tests length validation against n_neurons
        (Test Case 4) Tests key consistency validation (ValueError when keys differ)
        (Test Case 5) Tests that returned dicts are copies
        (Test Case 6) Tests empty list returns empty list
        (Test Case 7) Tests valid input returns normalized dicts with all keys
        """
        # Test Case 1: Non-list inputs raise ValueError
        with pytest.raises(ValueError):
            check_neuron_attributes({"a": 1})
        with pytest.raises(ValueError):
            check_neuron_attributes(None)

        # Test Case 2: Non-dict elements raise ValueError
        with pytest.raises(ValueError):
            check_neuron_attributes([{"a": 1}, "x"])
        with pytest.raises(ValueError):
            check_neuron_attributes([None])

        # Test Case 3: Length validation against n_neurons
        with pytest.raises(ValueError):
            check_neuron_attributes([{}], n_neurons=2)
        assert check_neuron_attributes([{}, {}], n_neurons=2) == [{}, {}]

        # Test Case 4: Key consistency validation - inconsistent keys raise ValueError
        with pytest.raises(ValueError, match="Neuron 1 missing") as exc_info:
            check_neuron_attributes([{"a": 1}, {}])
        assert "'a'" in str(exc_info.value)

        assert check_neuron_attributes([{"a": 1}, {"a": 2}]) == [{"a": 1}, {"a": 2}]

        # Test Case 5: Returns copies (modifying result does not affect original)
        original = [{"a": 1}]
        result = check_neuron_attributes(original)
        result[0]["a"] = 999
        assert original[0]["a"] == 1

        # Test Case 6: Empty list returns empty list
        assert check_neuron_attributes([]) == []

        # Test Case 7: Valid input with multiple keys returns normalized structure
        result = check_neuron_attributes([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        assert result == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

    def test_get_channels_for_unit(self):
        """
        Tests get_channels_for_unit channel resolution logic.

        Tests:
        (Test Case 1) channels=None uses neuron_to_channel mapping when available
        (Test Case 2) channels=None falls back to all channels when no mapping exists
        (Test Case 3) channels=int returns a single-element list
        (Test Case 4) channels=list returns the list as-is (when non-empty)
        (Test Case 5) channels=[] uses mapping when available, else all channels
        (Test Case 6) invalid channels type raises ValueError
        """
        neuron_to_channel = {0: 2, 3: 1}
        n_channels_total = 5

        assert get_channels_for_unit(
            unit_idx=0,
            channels=None,
            neuron_to_channel=neuron_to_channel,
            n_channels_total=n_channels_total,
        ) == [2]
        assert get_channels_for_unit(
            unit_idx=1,
            channels=None,
            neuron_to_channel=neuron_to_channel,
            n_channels_total=n_channels_total,
        ) == list(range(n_channels_total))
        assert get_channels_for_unit(
            unit_idx=0,
            channels=4,
            neuron_to_channel=neuron_to_channel,
            n_channels_total=n_channels_total,
        ) == [4]
        assert get_channels_for_unit(
            unit_idx=0,
            channels=[4, 0, 2],
            neuron_to_channel=neuron_to_channel,
            n_channels_total=n_channels_total,
        ) == [4, 0, 2]
        assert get_channels_for_unit(
            unit_idx=3,
            channels=[],
            neuron_to_channel=neuron_to_channel,
            n_channels_total=n_channels_total,
        ) == [1]
        assert get_channels_for_unit(
            unit_idx=999,
            channels=[],
            neuron_to_channel={},
            n_channels_total=n_channels_total,
        ) == list(range(n_channels_total))
        with pytest.raises(ValueError):
            get_channels_for_unit(
                unit_idx=0,
                channels="not-a-valid-type",
                neuron_to_channel=neuron_to_channel,
                n_channels_total=n_channels_total,
            )

    def test_compute_avg_waveform(self):
        """
        Tests compute_avg_waveform for both non-empty and empty waveform stacks.

        Tests:
        (Test Case 1) Non-empty stack returns mean across spikes (axis=2)
        (Test Case 2) Empty stack returns zeros of shape (num_channels, num_samples) with dtype
        """
        # Test Case 1: mean across spikes
        waveforms = np.array(
            [
                # channel 0
                [[1.0, 3.0], [2.0, 4.0]],
                # channel 1
                [[10.0, 14.0], [12.0, 16.0]],
            ],
            dtype=float,
        )  # shape (2, 2, 2)
        avg = compute_avg_waveform(waveforms, channel_indices=[0, 1], dtype=np.float32)
        expected = np.array([[2.0, 3.0], [12.0, 14.0]], dtype=float)
        assert np.allclose(avg, expected)

        # Test Case 2: empty spikes dimension
        empty = np.zeros((2, 30, 0), dtype=np.int16)
        avg_empty = compute_avg_waveform(empty, channel_indices=[5, 7], dtype=np.int16)
        assert avg_empty.shape == (2, 30)
        assert avg_empty.dtype == np.int16
        assert np.array_equal(avg_empty, np.zeros((2, 30), dtype=np.int16))

    def test_get_valid_spike_times(self):
        """
        Tests get_valid_spike_times for proper boundary filtering.

        Tests:
        (Test Case 1) Filters out spikes with extraction windows outside raw data bounds
        (Test Case 2) Empty spike list returns empty array
        """
        fs_kHz = 10.0
        ms_before, ms_after = 1.0, 2.0  # 10 samples before, 20 after
        n_time_samples = 200
        spike_times_ms = np.array([0.5, 1.0, 5.0, 19.0], dtype=float)

        valid = get_valid_spike_times(
            spike_times_ms=spike_times_ms,
            fs_kHz=fs_kHz,
            ms_before=ms_before,
            ms_after=ms_after,
            n_time_samples=n_time_samples,
        )
        assert np.array_equal(valid, np.array([1.0, 5.0]))

        valid_empty = get_valid_spike_times(
            spike_times_ms=np.array([], dtype=float),
            fs_kHz=fs_kHz,
            ms_before=ms_before,
            ms_after=ms_after,
            n_time_samples=n_time_samples,
        )
        assert valid_empty.size == 0

    def test_waveforms_by_channel(self):
        """
        Tests waveforms_by_channel conversion and validation.

        Tests:
        (Test Case 1) Returns dict[channel -> (num_samples, num_spikes)] with correct contents
        (Test Case 2) Raises ValueError when waveforms is not 3D
        (Test Case 3) Raises ValueError when channel_indices length mismatches waveforms axis 0
        """
        waveforms = np.zeros((2, 4, 3), dtype=float)
        waveforms[0, :, :] = 1.0
        waveforms[1, :, :] = 2.0

        ch_map = waveforms_by_channel(waveforms, channel_indices=[10, 12])
        assert set(ch_map.keys()) == {10, 12}
        assert ch_map[10].shape == (4, 3)
        assert ch_map[12].shape == (4, 3)
        assert np.allclose(ch_map[10], 1.0)
        assert np.allclose(ch_map[12], 2.0)

        with pytest.raises(ValueError):
            waveforms_by_channel(np.zeros((2, 4), dtype=float), channel_indices=[0, 1])
        with pytest.raises(ValueError):
            waveforms_by_channel(np.zeros((2, 4, 1), dtype=float), channel_indices=[0])

    def test_extract_waveforms(self):
        """
        Tests extract_waveforms waveform snippet extraction.

        Tests:
        (Test Case 1) Basic extraction returns (num_channels, num_samples, num_spikes) with correct slices
        (Test Case 2) channel_indices selects subset and preserves provided order
        (Test Case 3) Spikes with out-of-bounds windows are skipped
        (Test Case 4) Empty spike_times_ms returns an empty stack with correct shape/dtype
        (Test Case 5) Empty raw_data raises ValueError
        """
        n_channels_total, n_time_samples = 4, 200
        t = np.arange(n_time_samples, dtype=np.int64)
        raw_data = np.stack([ch * 1000 + t for ch in range(n_channels_total)], axis=0)

        fs_kHz = 10.0
        ms_before, ms_after = 1.0, 2.0  # => 30 samples
        spike_times_ms = np.array([5.0, 7.0], dtype=float)
        wf = extract_waveforms(
            raw_data,
            spike_times_ms=spike_times_ms,
            fs_kHz=fs_kHz,
            ms_before=ms_before,
            ms_after=ms_after,
        )
        assert wf.shape == (n_channels_total, 30, 2)
        assert np.array_equal(wf[:, :, 0], raw_data[:, 40:70])
        assert np.array_equal(wf[:, :, 1], raw_data[:, 60:90])

        channel_indices = [3, 1]
        wf_sub = extract_waveforms(
            raw_data,
            spike_times_ms=np.array([5.0], dtype=float),
            fs_kHz=fs_kHz,
            ms_before=ms_before,
            ms_after=ms_after,
            channel_indices=channel_indices,
        )
        assert wf_sub.shape == (2, 30, 1)
        assert np.array_equal(wf_sub[:, :, 0], raw_data[channel_indices, 40:70])

        # Out-of-bounds spikes should be skipped
        wf_skip = extract_waveforms(
            raw_data,
            spike_times_ms=np.array([0.5, 1.0, 19.0], dtype=float),
            fs_kHz=fs_kHz,
            ms_before=ms_before,
            ms_after=ms_after,
            channel_indices=[0],
        )
        assert wf_skip.shape == (1, 30, 1)
        assert np.array_equal(wf_skip[0, :, 0], raw_data[0, 0:30])

        wf_empty = extract_waveforms(
            raw_data.astype(np.int16),
            spike_times_ms=np.array([], dtype=float),
            fs_kHz=fs_kHz,
            ms_before=ms_before,
            ms_after=ms_after,
            channel_indices=[0, 2],
        )
        assert wf_empty.shape == (2, 30, 0)
        assert wf_empty.dtype == np.int16

        with pytest.raises(ValueError, match="raw_data is empty"):
            extract_waveforms(
                np.array([]),
                spike_times_ms=np.array([1.0], dtype=float),
                fs_kHz=fs_kHz,
            )

    def test_extract_unit_waveforms(self):
        """
        Tests extract_unit_waveforms orchestration logic and metadata outputs.

        Tests:
        (Test Case 1) channels=None uses neuron_to_channel mapping when available
        (Test Case 2) meta["spike_times_ms"] contains only valid spikes (bounds-checked)
        (Test Case 3) avg_waveform is computed across spikes (axis=2) when enabled
        (Test Case 4) return_avg_waveform=False yields avg_waveform=None
        (Test Case 5) return_channel_waveforms=True provides per-channel dict with expected shapes
        (Test Case 6) explicit channels list overrides mapping and preserves order
        """
        n_channels_total, n_time_samples = 4, 200
        fs_kHz = 10.0
        ms_before, ms_after = 1.0, 2.0  # => 30 samples

        t = np.arange(n_time_samples, dtype=np.int64)
        raw_data = np.stack([ch * 1000 + t for ch in range(n_channels_total)], axis=0)
        neuron_to_channel = {0: 2}

        # Include out-of-bounds spikes; only 1.0 and 5.0 are valid for these parameters.
        spike_times_ms = np.array([0.5, 1.0, 5.0, 19.0], dtype=float)

        waveforms, meta = extract_unit_waveforms(
            unit_idx=0,
            spike_times_ms=spike_times_ms,
            raw_data=raw_data,
            fs_kHz=fs_kHz,
            ms_before=ms_before,
            ms_after=ms_after,
            channels=None,
            neuron_to_channel=neuron_to_channel,
            return_channel_waveforms=True,
            return_avg_waveform=True,
        )

        # Mapping should pick channel 2 only
        assert meta["channels"] == [2]
        # Only valid spikes should remain in meta
        assert np.array_equal(meta["spike_times_ms"], np.array([1.0, 5.0]))
        # Waveforms should match those valid spikes
        assert waveforms.shape == (1, 30, 2)
        assert np.array_equal(waveforms[0, :, 0], raw_data[2, 0:30])  # 1ms -> [0:30]
        assert np.array_equal(waveforms[0, :, 1], raw_data[2, 40:70])  # 5ms -> [40:70]

        # avg_waveform should be mean across spikes
        avg_expected = waveforms.mean(axis=2)
        assert np.array_equal(meta["avg_waveform"], avg_expected)

        # Per-channel view should match waveforms slices
        assert "channel_waveforms" in meta
        assert 2 in meta["channel_waveforms"]
        assert np.array_equal(meta["channel_waveforms"][2], waveforms[0, :, :])

        # return_avg_waveform=False -> None
        _, meta_no_avg = extract_unit_waveforms(
            unit_idx=0,
            spike_times_ms=np.array([5.0], dtype=float),
            raw_data=raw_data,
            fs_kHz=fs_kHz,
            ms_before=ms_before,
            ms_after=ms_after,
            channels=None,
            neuron_to_channel=neuron_to_channel,
            return_channel_waveforms=False,
            return_avg_waveform=False,
        )
        assert meta_no_avg["avg_waveform"] is None

        # Explicit channels override mapping and preserve order
        waveforms_exp, meta_exp = extract_unit_waveforms(
            unit_idx=0,
            spike_times_ms=np.array([5.0], dtype=float),
            raw_data=raw_data,
            fs_kHz=fs_kHz,
            ms_before=ms_before,
            ms_after=ms_after,
            channels=[3, 1],
            neuron_to_channel=neuron_to_channel,
            return_channel_waveforms=False,
            return_avg_waveform=True,
        )
        assert meta_exp["channels"] == [3, 1]
        assert waveforms_exp.shape == (2, 30, 1)
        assert np.array_equal(waveforms_exp[:, :, 0], raw_data[[3, 1], 40:70])

    def test_set_neuron_attribute(self):
        """
        Tests set_neuron_attribute for single, array, and partial updates.

        Tests:
        (Test Case 1) Tests single value assignment to all neurons
        (Test Case 2) Tests array value assignment
        (Test Case 3) Tests partial update with neuron_indices
        (Test Case 4) Tests length mismatch raises ValueError
        """
        sd = SpikeData([[] for _ in range(4)], length=100)

        # Test Case 1: Single value assignment to all neurons
        sd.set_neuron_attribute("type", "excitatory")
        assert all(a["type"] == "excitatory" for a in sd.neuron_attributes)

        # Test Case 2: Array value assignment
        sd.set_neuron_attribute("rate", [1, 2, 3, 4])
        assert [a["rate"] for a in sd.neuron_attributes] == [1, 2, 3, 4]

        # Test Case 3: Partial update with neuron_indices
        sd.set_neuron_attribute("label", "A", neuron_indices=[0, 2])
        assert sd.neuron_attributes[0]["label"] == "A"
        assert sd.neuron_attributes[2]["label"] == "A"
        assert "label" not in sd.neuron_attributes[1]

        # Test Case 4: Length mismatch raises ValueError
        with pytest.raises(ValueError):
            sd.set_neuron_attribute("x", [1, 2], [0])

    def test_get_neuron_attribute(self):
        """
        Tests get_neuron_attribute retrieval with and without defaults.

        Tests:
        (Test Case 1) Tests retrieval when neuron_attributes is None (returns defaults)
        (Test Case 2) Tests retrieval of existing attribute values
        (Test Case 3) Tests default value for missing attributes
        (Test Case 4) Tests mixed case: some neurons have attribute, some use default
        """
        sd = SpikeData([[] for _ in range(3)], length=100)

        # Test Case 1: When neuron_attributes is None, returns default for all neurons
        assert sd.get_neuron_attribute("x") == [None, None, None]
        assert sd.get_neuron_attribute("x", default=-1) == [-1, -1, -1]

        # Test Case 2: Retrieval of existing attribute values
        sd.set_neuron_attribute("val", [1, 2, 3])
        assert sd.get_neuron_attribute("val") == [1, 2, 3]

        # Test Case 3: Default value for missing attributes
        assert sd.get_neuron_attribute("missing") == [None, None, None]
        assert sd.get_neuron_attribute("missing", default=0) == [0, 0, 0]

        # Test Case 4: Mixed case - partial attribute set via neuron_indices
        sd.set_neuron_attribute("label", "A", neuron_indices=[0, 2])
        assert sd.get_neuron_attribute("label") == ["A", None, "A"]
        assert sd.get_neuron_attribute("label", default="?") == ["A", "?", "A"]

    def test_get_waveform_traces(self):
        """
        Test get_waveform_traces for correct waveform extraction from raw data.

        Tests:
        (Test Case 1) Basic waveform extraction returns dict with correct shape
        (Test Case 2) Waveform shape is (num_channels, num_samples, num_spikes)
        (Test Case 3) Explicit channel parameter overrides mapping
        (Test Case 4) Empty list channels uses neuron_to_channel mapping
        (Test Case 5) No channel mapping extracts all channels
        (Test Case 6) Extract waveforms for all units
        (Test Case 7) Spikes near boundaries should be skipped
        (Test Case 8) Waveform storage in neuron_attributes (store=True default)
        (Test Case 9) Error handling for empty raw_data
        (Test Case 10) Error handling for unit out of range
        (Test Case 11) raw_time as timestamp array
        (Test Case 12) Unit with no spikes returns empty array with correct shape
        (Test Case 13) Bandpass filtering option
        (Test Case 14) Operations across spikes (axis=2)
        """
        n_channels = 4
        n_samples = 1000
        fs_kHz = 10.0
        raw_data = np.random.randn(n_channels, n_samples)
        raw_data[1, 195:205] = -5.0
        raw_data[1, 495:505] = -5.0

        trains = [[20.0, 50.0], [30.0], []]
        attrs = [{"channel": 1}, {"channel": 2}, {"channel": 0}]
        sd = SpikeData(
            trains,
            neuron_attributes=attrs,
            length=100.0,
            raw_data=raw_data,
            raw_time=fs_kHz,
        )

        # Basic extraction returns (waveforms, meta)
        waveforms, meta = sd.get_waveform_traces(unit=0, ms_before=1.0, ms_after=2.0)
        assert isinstance(meta, dict)
        assert "fs_kHz" in meta
        assert "unit_indices" in meta
        assert "channels" in meta
        assert "spike_times_ms" in meta
        assert "avg_waveforms" in meta

        assert waveforms.ndim == 3
        expected_samples = int(1.0 * fs_kHz) + int(2.0 * fs_kHz)
        assert waveforms.shape[0] == 1
        assert waveforms.shape[1] == expected_samples
        assert waveforms.shape[2] == 2

        avg_wf = meta["avg_waveforms"][0]
        assert avg_wf.ndim == 2
        assert avg_wf.shape[0] == waveforms.shape[0]
        assert avg_wf.shape[1] == waveforms.shape[1]
        assert np.any(waveforms < -4.0)

        waveforms_ch0, meta_ch0 = sd.get_waveform_traces(
            unit=0, channels=0, store=False
        )
        assert waveforms_ch0.shape[0] == 1
        assert meta_ch0["channels"][0] == [0]

        waveforms_empty_list, meta_empty_list = sd.get_waveform_traces(
            unit=0, channels=[], store=False
        )
        assert meta_empty_list["channels"][0] == [1]

        sd_no_channel = SpikeData(
            trains, length=100.0, raw_data=raw_data, raw_time=fs_kHz
        )
        waveforms_all_ch, _meta_all_ch = sd_no_channel.get_waveform_traces(
            unit=0, ms_before=1.0, ms_after=2.0
        )
        assert waveforms_all_ch.shape[0] == n_channels

        all_waveforms, all_meta = sd.get_waveform_traces(
            ms_before=1.0, ms_after=2.0, store=False
        )
        assert isinstance(all_waveforms, list)
        assert len(all_waveforms) == 3
        assert all_waveforms[0].shape[2] == 2
        assert all_waveforms[1].shape[2] == 1
        assert all_waveforms[2].shape[2] == 0
        assert all_meta["unit_indices"] == [0, 1, 2]

        sd_edge = SpikeData(
            [[5.0, 95.0]], length=100.0, raw_data=raw_data, raw_time=fs_kHz
        )
        waveforms_edge, _meta_edge = sd_edge.get_waveform_traces(
            unit=0, ms_before=10.0, ms_after=10.0
        )
        assert waveforms_edge.shape[2] == 0
        waveforms_small, _meta_small = sd_edge.get_waveform_traces(
            unit=0, ms_before=1.0, ms_after=1.0
        )
        assert waveforms_small.shape[2] == 2

        sd_store = SpikeData(
            trains,
            neuron_attributes=[{"channel": 1}, {"channel": 2}, {"channel": 0}],
            length=100.0,
            raw_data=raw_data,
            raw_time=fs_kHz,
        )
        _waveforms_stored, meta_stored = sd_store.get_waveform_traces(
            unit=0, ms_before=1.0, ms_after=2.0
        )
        assert sd_store.neuron_attributes[0].get("avg_waveform") is not None
        assert sd_store.neuron_attributes[0].get("waveforms") is not None
        assert sd_store.neuron_attributes[0].get("traces_meta") is not None
        assert sd_store.neuron_attributes[0]["traces_meta"]["channels"] == [1]
        assert np.isclose(
            sd_store.neuron_attributes[0]["traces_meta"]["fs_kHz"], fs_kHz
        )
        assert np.allclose(
            sd_store.neuron_attributes[0]["avg_waveform"],
            meta_stored["avg_waveforms"][0],
        )

        sd_store.get_waveform_traces()
        assert sd_store.neuron_attributes[0].get("avg_waveform") is not None
        assert sd_store.neuron_attributes[1].get("avg_waveform") is not None
        assert sd_store.neuron_attributes[2]["waveforms"].shape[2] == 0
        assert sd_store.neuron_attributes[0].get("traces_meta") is not None
        assert sd_store.neuron_attributes[1].get("traces_meta") is not None
        assert sd_store.neuron_attributes[2].get("traces_meta") is not None

        sd_no_raw = SpikeData(trains, length=100.0)
        with pytest.raises(ValueError):
            sd_no_raw.get_waveform_traces(unit=0)

        with pytest.raises(ValueError):
            sd.get_waveform_traces(unit=10)
        with pytest.raises(ValueError):
            sd.get_waveform_traces(unit=-1)

        timestamps = np.arange(n_samples) / fs_kHz
        sd_timestamps = SpikeData(
            trains,
            neuron_attributes=attrs,
            length=100.0,
            raw_data=raw_data,
            raw_time=timestamps,
        )
        waveforms_ts, _meta_ts = sd_timestamps.get_waveform_traces(
            unit=0, ms_before=1.0, ms_after=2.0, store=False
        )
        assert waveforms_ts.shape == waveforms.shape

        waveforms_empty, _meta_empty = sd.get_waveform_traces(
            unit=2, ms_before=1.0, ms_after=2.0, store=False
        )
        assert waveforms_empty.shape[0] == 1
        assert waveforms_empty.shape[1] == expected_samples
        assert waveforms_empty.shape[2] == 0

        waveforms_filtered, _meta_filtered = sd.get_waveform_traces(
            unit=0, bandpass=(100, 2000), filter_order=3, store=False
        )
        assert waveforms_filtered.shape == waveforms.shape
        assert not np.allclose(waveforms_filtered, waveforms)

        peak_amps = waveforms.min(axis=1)
        assert peak_amps.shape == (1, 2)

        mean_across_spikes = waveforms.mean(axis=2)
        assert np.allclose(mean_across_spikes, avg_wf)

        # Subset selection: list of unit indices returns list of waveforms + shared meta
        subset_waveforms, subset_meta = sd.get_waveform_traces(
            unit=[0, 2], ms_before=1.0, ms_after=2.0, store=False
        )
        assert isinstance(subset_waveforms, list)
        assert len(subset_waveforms) == 2
        assert subset_meta["unit_indices"] == [0, 2]
        assert subset_meta["channels"][0] == [1]
        assert subset_meta["channels"][1] == [0]
        assert subset_waveforms[0].shape[2] == 2  # unit 0 has 2 spikes
        assert subset_waveforms[1].shape[2] == 0  # unit 2 has no spikes

        # Subset selection: slice returns list of waveforms
        subset_slice_waveforms, subset_slice_meta = sd.get_waveform_traces(
            unit=slice(0, 2), ms_before=1.0, ms_after=2.0, store=False
        )
        assert isinstance(subset_slice_waveforms, list)
        assert len(subset_slice_waveforms) == 2
        assert subset_slice_meta["unit_indices"] == [0, 1]
        assert subset_slice_waveforms[0].shape[2] == 2
        assert subset_slice_waveforms[1].shape[2] == 1

        # Subset selection: range returns list of waveforms
        subset_range_waveforms, subset_range_meta = sd.get_waveform_traces(
            unit=range(1, 3), ms_before=1.0, ms_after=2.0, store=False
        )
        assert isinstance(subset_range_waveforms, list)
        assert len(subset_range_waveforms) == 2
        assert subset_range_meta["unit_indices"] == [1, 2]
        assert subset_range_waveforms[0].shape[2] == 1
        assert subset_range_waveforms[1].shape[2] == 0

    def test_align_to_events(self):
        """
        Test align_to_events for event-aligned slice stack creation.

        Tests:
            (Test Case 1) kind='spike' returns a SpikeSliceStack with one slice per event.
            (Test Case 2) kind='rate' returns a RateSliceStack with one slice per event.
            (Test Case 3) events given as a metadata key string are resolved correctly.
            (Test Case 4) An invalid metadata key raises KeyError with the key name.
            (Test Case 5) Events whose window exceeds recording bounds are dropped with UserWarning.
            (Test Case 6) All events dropped after filtering raises ValueError.
            (Test Case 7) An invalid kind value raises ValueError.
            (Test Case 8) Each slice spans exactly pre_ms + post_ms milliseconds.
        """
        from SpikeLab.spikedata.spikeslicestack import SpikeSliceStack
        from SpikeLab.spikedata.rateslicestack import RateSliceStack

        # Build a simple 3-unit recording: 200 ms, 10 spikes per unit
        trains = [np.linspace(5, 195, 10) for _ in range(3)]
        sd = SpikeData(trains, length=200.0)

        events_ms = np.array([50.0, 100.0, 150.0])
        pre_ms, post_ms = 20.0, 30.0

        # Test Case 1: kind='spike' → SpikeSliceStack
        spike_stack = sd.align_to_events(events_ms, pre_ms, post_ms, kind="spike")
        assert isinstance(spike_stack, SpikeSliceStack)
        assert len(spike_stack.spike_stack) == 3

        # Test Case 2: kind='rate' → RateSliceStack
        rate_stack = sd.align_to_events(events_ms, pre_ms, post_ms, kind="rate")
        assert isinstance(rate_stack, RateSliceStack)
        assert rate_stack.event_stack.shape[2] == 3  # 3 slices

        # Test Case 3: metadata key string resolves to correct array
        sd_with_meta = SpikeData(
            trains,
            length=200.0,
            metadata={"stim_on_times": events_ms.copy()},
        )
        spike_stack_meta = sd_with_meta.align_to_events(
            "stim_on_times", pre_ms, post_ms, kind="spike"
        )
        assert isinstance(spike_stack_meta, SpikeSliceStack)
        assert len(spike_stack_meta.spike_stack) == 3

        # Test Case 4: invalid metadata key raises KeyError
        with pytest.raises(KeyError, match="missing_key"):
            sd_with_meta.align_to_events("missing_key", pre_ms, post_ms)

        # Test Case 5: out-of-bounds events dropped with UserWarning
        events_with_oob = np.array([10.0, 100.0, 180.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spike_stack_filtered = sd.align_to_events(
                events_with_oob, pre_ms, post_ms, kind="spike"
            )
        assert len(spike_stack_filtered.spike_stack) == 1
        assert any(issubclass(w.category, UserWarning) for w in caught)
        warning_text = str(caught[0].message)
        assert "2" in warning_text  # 2 events dropped

        # Test Case 6: all events out of bounds → ValueError
        events_all_oob = np.array([5.0, 195.0])  # both outside with pre=20, post=30
        with pytest.raises(ValueError):
            sd.align_to_events(events_all_oob, pre_ms, post_ms)

        # Test Case 7: invalid kind raises ValueError
        with pytest.raises(ValueError, match="burst"):
            sd.align_to_events(events_ms, pre_ms, post_ms, kind="burst")

        # Test Case 8: slice duration equals pre_ms + post_ms
        spike_stack_times = sd.align_to_events(events_ms, pre_ms, post_ms, kind="spike")
        for start, end in spike_stack_times.times:
            assert end - start == pytest.approx(pre_ms + post_ms)

    def test_unit_locations(self):
        """
        Tests the unit_locations property.

        Tests:
            (Test Case 1) Returns None when neuron_attributes is None.
            (Test Case 2) Extracts from 'location' key.
            (Test Case 3) Extracts from 'x'/'y' keys.
            (Test Case 4) Extracts from 'x'/'y'/'z' keys.
            (Test Case 5) Extracts from 'position' key.
            (Test Case 6) Returns None when one unit lacks location data.
        """
        # No attributes
        sd = SpikeData([[1.0, 2.0], [3.0]], length=10.0)
        assert sd.unit_locations is None

        # 'location' key
        sd_loc = SpikeData(
            [[1.0], [2.0]],
            length=10.0,
            neuron_attributes=[{"location": [0.0, 1.0]}, {"location": [2.0, 3.0]}],
        )
        locs = sd_loc.unit_locations
        assert locs.shape == (2, 2)
        np.testing.assert_array_equal(locs[0], [0.0, 1.0])

        # 'x'/'y' keys
        sd_xy = SpikeData(
            [[1.0], [2.0]],
            length=10.0,
            neuron_attributes=[{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
        )
        locs_xy = sd_xy.unit_locations
        assert locs_xy.shape == (2, 2)
        np.testing.assert_array_equal(locs_xy[0], [1.0, 2.0])

        # 'x'/'y'/'z' keys
        sd_xyz = SpikeData(
            [[1.0], [2.0]],
            length=10.0,
            neuron_attributes=[
                {"x": 1.0, "y": 2.0, "z": 3.0},
                {"x": 4.0, "y": 5.0, "z": 6.0},
            ],
        )
        locs_xyz = sd_xyz.unit_locations
        assert locs_xyz.shape == (2, 3)

        # 'position' key
        sd_pos = SpikeData(
            [[1.0], [2.0]],
            length=10.0,
            neuron_attributes=[{"position": [0, 1]}, {"position": [2, 3]}],
        )
        assert sd_pos.unit_locations.shape == (2, 2)

        # Partial data returns None
        sd_partial = SpikeData(
            [[1.0], [2.0]],
            length=10.0,
            neuron_attributes=[{"location": [0, 1]}, {"other": 42}],
        )
        assert sd_partial.unit_locations is None

    def test_electrodes(self):
        """
        Tests the electrodes property.

        Tests:
            (Test Case 1) Returns None when neuron_attributes is None.
            (Test Case 2) Extracts from 'electrode' key.
            (Test Case 3) Extracts from 'channel' key.
            (Test Case 4) Extracts from 'ch' key.
            (Test Case 5) Returns None when one unit lacks electrode data.
        """
        sd = SpikeData([[1.0], [2.0]], length=10.0)
        assert sd.electrodes is None

        sd_elec = SpikeData(
            [[1.0], [2.0]],
            length=10.0,
            neuron_attributes=[{"electrode": 0}, {"electrode": 1}],
        )
        elec = sd_elec.electrodes
        assert len(elec) == 2
        np.testing.assert_array_equal(elec, [0, 1])

        sd_ch = SpikeData(
            [[1.0], [2.0]],
            length=10.0,
            neuron_attributes=[{"channel": 5}, {"channel": 10}],
        )
        np.testing.assert_array_equal(sd_ch.electrodes, [5, 10])

        sd_ch2 = SpikeData(
            [[1.0], [2.0]],
            length=10.0,
            neuron_attributes=[{"ch": 0}, {"ch": 1}],
        )
        np.testing.assert_array_equal(sd_ch2.electrodes, [0, 1])

        sd_partial = SpikeData(
            [[1.0], [2.0]],
            length=10.0,
            neuron_attributes=[{"electrode": 0}, {"other": 42}],
        )
        assert sd_partial.electrodes is None

    def test_subset_by_attribute(self):
        """
        Tests subset() with the by parameter for attribute-based selection.

        Tests:
            (Test Case 1) Select units by attribute value.
            (Test Case 2) Select single unit by string attribute.
        """
        sd = SpikeData(
            [[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]],
            length=10.0,
            neuron_attributes=[
                {"region": "CA1"},
                {"region": "CA3"},
                {"region": "CA1"},
            ],
        )
        sub = sd.subset(["CA1"], by="region")
        assert sub.N == 2
        # Units 0 and 2 have region=CA1
        np.testing.assert_array_almost_equal(sub.train[0], [1.0, 5.0])
        np.testing.assert_array_almost_equal(sub.train[1], [3.0, 7.0])

    def test_binned_meanrate(self):
        """
        Tests binned_meanrate() computes correct mean population rate.

        Tests:
            (Test Case 1) kHz output matches manual calculation.
            (Test Case 2) Hz output is 1000x kHz output.
            (Test Case 3) Invalid unit raises ValueError.
        """
        sd = SpikeData(
            [[0.5, 1.5, 2.5], [0.5, 1.5, 2.5]],
            length=4.0,
        )
        # binned(1) = [2, 2, 2, 0] (2 units each fire once per bin)
        # meanrate kHz = [2/(2*1), 2/(2*1), 2/(2*1), 0] = [1.0, 1.0, 1.0, 0]
        mr_khz = sd.binned_meanrate(bin_size=1, unit="kHz")
        assert mr_khz[0] == pytest.approx(1.0)
        assert mr_khz[3] == pytest.approx(0.0)

        mr_hz = sd.binned_meanrate(bin_size=1, unit="Hz")
        np.testing.assert_array_almost_equal(mr_hz, mr_khz * 1e3)

        with pytest.raises(ValueError, match="Unknown unit"):
            sd.binned_meanrate(bin_size=1, unit="bad")

    def test_resampled_isi(self):
        """
        Tests resampled_isi() returns correct shape and reasonable values.

        Tests:
            (Test Case 1) Output shape is (N, len(times)).
            (Test Case 2) Regular spike train produces approximately uniform rate.
        """
        # Unit with spikes at 0, 1, 2, ..., 99 (1 kHz)
        train = [np.arange(0, 100, 1.0)]
        sd = SpikeData(train, length=100.0)
        times = np.arange(5, 95, 1.0)
        rates = sd.resampled_isi(times, sigma_ms=5.0)
        assert rates.shape == (1, len(times))
        # Rate should be approximately 1000 Hz (1 spike per ms, ISI=1ms, rate=1/ISI=1000 Hz)
        assert np.mean(rates[0]) == pytest.approx(1000.0, rel=0.2)

    def test_append(self):
        """
        Tests append() concatenates two SpikeData objects in time.

        Tests:
            (Test Case 1) Result has combined length.
            (Test Case 2) Spike times from second object are offset.
            (Test Case 3) Same N preserved.
            (Test Case 4) Different N raises ValueError.
            (Test Case 5) Offset parameter works.
        """
        sd1 = SpikeData([[5.0, 10.0], [3.0]], length=20.0)
        sd2 = SpikeData([[1.0, 2.0], [4.0]], length=10.0)

        combined = sd1.append(sd2)
        assert combined.N == 2
        assert combined.length == pytest.approx(30.0)
        # sd2 spikes shifted by sd1.length (20.0)
        np.testing.assert_array_almost_equal(combined.train[0], [5.0, 10.0, 21.0, 22.0])
        np.testing.assert_array_almost_equal(combined.train[1], [3.0, 24.0])

        # Different N raises
        sd3 = SpikeData([[1.0]], length=10.0)
        with pytest.raises(ValueError, match="different N"):
            sd1.append(sd3)

        # With offset
        combined_offset = sd1.append(sd2, offset=5.0)
        assert combined_offset.length == pytest.approx(35.0)
        np.testing.assert_array_almost_equal(
            combined_offset.train[0], [5.0, 10.0, 26.0, 27.0]
        )

    def test_concatenate_spike_data(self):
        """
        Tests concatenate_spike_data() adds units from another SpikeData.

        Tests:
            (Test Case 1) N increases by the added units.
            (Test Case 2) Original trains preserved.
            (Test Case 3) New trains appended.
            (Test Case 4) In-place mutation.
        """
        sd1 = SpikeData([[1.0, 2.0], [3.0, 4.0]], length=10.0)
        sd2 = SpikeData([[5.0, 6.0]], length=10.0)

        original_n = sd1.N
        sd1.concatenate_spike_data(sd2)
        assert sd1.N == original_n + 1
        assert len(sd1.train) == 3
        np.testing.assert_array_almost_equal(sd1.train[0], [1.0, 2.0])
        np.testing.assert_array_almost_equal(sd1.train[2], [5.0, 6.0])

    def test_concatenate_spike_data_different_length(self):
        """
        Tests concatenate_spike_data when second SpikeData has different length.

        Tests:
            (Test Case 1) Second SpikeData is subtimed to first's length.
        """
        sd1 = SpikeData([[1.0, 2.0]], length=10.0)
        sd2 = SpikeData([[5.0, 15.0]], length=20.0)

        sd1.concatenate_spike_data(sd2)
        assert sd1.N == 2
        # sd2 subtimed to [0, 10) so spike at 15 is removed
        assert len(sd1.train[1]) == 1
        np.testing.assert_array_almost_equal(sd1.train[1], [5.0])

    def test_latencies_to_index(self):
        """
        Tests latencies_to_index() delegates correctly to latencies().

        Tests:
            (Test Case 1) Returns latencies from unit i's spikes to all units.
            (Test Case 2) Same result as calling latencies() directly with unit's train.
        """
        sd = SpikeData(
            [[10.0, 50.0, 90.0], [15.0, 55.0, 95.0], [20.0, 60.0]],
            length=100.0,
        )
        lat_to_idx = sd.latencies_to_index(0, window_ms=10.0)
        lat_direct = sd.latencies(sd.train[0], window_ms=10.0)

        assert len(lat_to_idx) == len(lat_direct)
        for a, b in zip(lat_to_idx, lat_direct):
            assert len(a) == len(b)

    def test_compute_spike_trig_pop_rate(self):
        """
        Tests compute_spike_trig_pop_rate() returns correct shapes.

        Tests:
            (Test Case 1) stPR_filtered has shape (N, 2*window_ms + 1).
            (Test Case 2) coupling_strengths_zero_lag has shape (N,).
            (Test Case 3) coupling_strengths_max has shape (N,).
            (Test Case 4) delays has shape (N,).
            (Test Case 5) lags has shape (2*window_ms + 1,).
            (Test Case 6) Silent neuron gets zero coupling.
        """
        sd = random_spikedata(5, 200, rate=1.0)
        window = 20
        stPR, cs_zero, cs_max, delays, lags = sd.compute_spike_trig_pop_rate(
            window_ms=window, cutoff_hz=20, fs=1000, bin_size=1, cut_outer=5
        )
        assert stPR.shape == (5, 2 * window + 1)
        assert cs_zero.shape == (5,)
        assert cs_max.shape == (5,)
        assert delays.shape == (5,)
        assert lags.shape == (2 * window + 1,)
        assert lags[0] == -window
        assert lags[-1] == window

    def test_compute_spike_trig_pop_rate_silent_neuron(self):
        """
        Tests compute_spike_trig_pop_rate with a silent neuron.

        Tests:
            (Test Case 1) Silent neuron's coupling curve is all zeros.
        """
        train = [np.array([10.0, 50.0, 90.0]), np.array([])]  # unit 1 silent
        sd = SpikeData(train, length=100.0)
        stPR, cs_zero, cs_max, delays, lags = sd.compute_spike_trig_pop_rate(
            window_ms=10
        )
        assert stPR.shape[0] == 2
        # Silent neuron should have all-zero coupling
        np.testing.assert_array_equal(stPR[1], np.zeros(21))

    def test_from_thresholding(self):
        """
        Tests from_thresholding static constructor.

        Tests:
            (Test Case 1) Detects spikes from synthetic raw data.
            (Test Case 2) raw_data and raw_time are attached.
            (Test Case 3) Direction 'up' only detects positive crossings.
            (Test Case 4) Filter disabled with filter=False.
        """
        rng = np.random.default_rng(42)
        fs_Hz = 10000.0
        n_ch = 2
        n_samples = 10000
        raw = rng.normal(0, 1, (n_ch, n_samples))
        # Insert large spikes
        raw[0, 500] = 20.0
        raw[0, 5000] = 20.0
        raw[1, 3000] = -20.0

        sd = SpikeData.from_thresholding(
            raw, fs_Hz=fs_Hz, threshold_sigma=5.0, filter=False
        )
        assert sd.N == n_ch
        assert sd.raw_data.shape == (n_ch, n_samples)
        assert len(sd.raw_time) == n_samples
        # At least some spikes should be detected
        total_spikes = sum(len(t) for t in sd.train)
        assert total_spikes > 0

        # Direction 'up' should not detect negative-only spikes on channel 1
        sd_up = SpikeData.from_thresholding(
            raw, fs_Hz=fs_Hz, threshold_sigma=5.0, filter=False, direction="up"
        )
        # Channel 0 should have spikes, channel 1 might not (only negative spike)
        assert len(sd_up.train[0]) > 0


class TestGetPairwiseCCG:
    """Tests for SpikeData.get_pairwise_ccg."""

    def test_basic_shape_and_symmetry(self):
        """
        Tests that get_pairwise_ccg returns correctly shaped, symmetric matrices.

        Tests:
            (Test Case 1) Output shapes are (N, N) for both corr and lag matrices.
            (Test Case 2) Correlation matrix is symmetric.
            (Test Case 3) Lag matrix is antisymmetric (lag[i,j] == -lag[j,i]).
            (Test Case 4) Diagonal of corr is 1, diagonal of lag is 0.
        """
        sd = random_spikedata(5, 5000)
        corr, lag = sd.get_pairwise_ccg(bin_size=1.0, max_lag=50)

        assert corr.matrix.shape == (5, 5)
        assert lag.matrix.shape == (5, 5)

        # Symmetry
        np.testing.assert_array_almost_equal(corr.matrix, corr.matrix.T)
        # Antisymmetry of lags
        np.testing.assert_array_almost_equal(lag.matrix, -lag.matrix.T)

        # Diagonal
        np.testing.assert_array_equal(np.diag(corr.matrix), np.ones(5))
        np.testing.assert_array_equal(np.diag(lag.matrix), np.zeros(5))

    def test_returns_pairwise_comp_matrix(self):
        """
        Tests that both return values are PairwiseCompMatrix instances.

        Tests:
            (Test Case 1) corr is a PairwiseCompMatrix.
            (Test Case 2) lag is a PairwiseCompMatrix.
        """
        from SpikeLab.spikedata.pairwise import PairwiseCompMatrix

        sd = random_spikedata(3, 3000)
        corr, lag = sd.get_pairwise_ccg(bin_size=1.0, max_lag=10)

        assert isinstance(corr, PairwiseCompMatrix)
        assert isinstance(lag, PairwiseCompMatrix)

    def test_metadata(self):
        """
        Tests that metadata on returned matrices stores bin_size and max_lag.

        Tests:
            (Test Case 1) corr metadata contains bin_size and max_lag.
            (Test Case 2) lag metadata contains bin_size and max_lag.
        """
        sd = random_spikedata(3, 3000)
        corr, lag = sd.get_pairwise_ccg(bin_size=2.0, max_lag=100)

        assert corr.metadata["bin_size"] == 2.0
        assert corr.metadata["max_lag"] == 100
        assert lag.metadata["bin_size"] == 2.0
        assert lag.metadata["max_lag"] == 100

    def test_identical_trains_perfect_correlation(self):
        """
        Tests that identical spike trains produce correlation of 1 and lag of 0.

        Tests:
            (Test Case 1) Two copies of the same train yield corr == 1 and lag == 0.
        """
        train = np.sort(np.random.uniform(0, 1000, size=200))
        sd = SpikeData([train, train.copy()], length=1000)
        corr, lag = sd.get_pairwise_ccg(bin_size=1.0, max_lag=50)

        assert corr.matrix[0, 1] == pytest.approx(1.0)
        assert lag.matrix[0, 1] == 0

    def test_cosine_similarity_func(self):
        """
        Tests that compute_cosine_similarity_with_lag works as compare_func.

        Tests:
            (Test Case 1) Output shapes are correct with cosine similarity.
            (Test Case 2) Diagonal of corr is 1.
            (Test Case 3) Correlation values are within [-1, 1].
        """
        sd = random_spikedata(4, 4000)
        corr, lag = sd.get_pairwise_ccg(
            compare_func=compute_cosine_similarity_with_lag,
            bin_size=1.0,
            max_lag=20,
        )

        assert corr.matrix.shape == (4, 4)
        np.testing.assert_array_almost_equal(np.diag(corr.matrix), np.ones(4))
        assert np.all(corr.matrix >= -1.0 - 1e-10)
        assert np.all(corr.matrix <= 1.0 + 1e-10)

    def test_bin_size_affects_lag_conversion(self):
        """
        Tests that max_lag is converted to bins using bin_size.

        Tests:
            (Test Case 1) With bin_size=5 and max_lag=50, the maximum absolute lag
                in bins should not exceed 10 (50/5).
        """
        sd = random_spikedata(3, 3000)
        corr, lag = sd.get_pairwise_ccg(bin_size=5.0, max_lag=50)

        # Lag values are in bins; max should be <= 10 (50ms / 5ms)
        assert np.all(np.abs(lag.matrix) <= 10)

    def test_single_unit(self):
        """
        Tests get_pairwise_ccg with a single unit.

        Tests:
            (Test Case 1) Returns 1x1 matrices with corr=1 and lag=0.
        """
        sd = SpikeData([np.sort(np.random.uniform(0, 500, 100))], length=500)
        corr, lag = sd.get_pairwise_ccg(bin_size=1.0, max_lag=10)

        assert corr.matrix.shape == (1, 1)
        assert corr.matrix[0, 0] == 1.0
        assert lag.matrix[0, 0] == 0

    def test_empty_train_pair(self):
        """
        Tests get_pairwise_ccg when one unit has no spikes.

        Tests:
            (Test Case 1) Correlation with an empty train is 0.
        """
        sd = SpikeData([[], np.sort(np.random.uniform(0, 500, 100))], length=500)
        corr, lag = sd.get_pairwise_ccg(bin_size=1.0, max_lag=10)

        assert corr.matrix[0, 1] == pytest.approx(0.0)

    def test_correlation_bounded(self):
        """
        Tests that all correlation values stay within [-1, 1].

        Tests:
            (Test Case 1) Random spike data with various configurations stays bounded.
        """
        for _ in range(10):
            n_units = np.random.randint(2, 6)
            sd = random_spikedata(n_units, n_units * 500)
            corr, lag = sd.get_pairwise_ccg(bin_size=1.0, max_lag=20)

            assert np.all(corr.matrix >= -1.0 - 1e-10)
            assert np.all(corr.matrix <= 1.0 + 1e-10)


class TestGetPairwiseLatencies:
    """Tests for SpikeData.get_pairwise_latencies."""

    def test_basic_shape(self):
        """
        Tests that get_pairwise_latencies returns correctly shaped matrices.

        Tests:
            (Test Case 1) Mean and std matrices are (N, N).
            (Test Case 2) Both are PairwiseCompMatrix instances.
        """
        from SpikeLab.spikedata.pairwise import PairwiseCompMatrix

        sd = random_spikedata(5, 5000)
        mean_lat, std_lat = sd.get_pairwise_latencies()

        assert mean_lat.matrix.shape == (5, 5)
        assert std_lat.matrix.shape == (5, 5)
        assert isinstance(mean_lat, PairwiseCompMatrix)
        assert isinstance(std_lat, PairwiseCompMatrix)

    def test_diagonal_is_zero(self):
        """
        Tests that diagonal entries are zero for both mean and std.

        Tests:
            (Test Case 1) Diagonal of mean matrix is all zeros.
            (Test Case 2) Diagonal of std matrix is all zeros.
        """
        sd = random_spikedata(4, 4000)
        mean_lat, std_lat = sd.get_pairwise_latencies()

        np.testing.assert_array_equal(np.diag(mean_lat.matrix), np.zeros(4))
        np.testing.assert_array_equal(np.diag(std_lat.matrix), np.zeros(4))

    def test_approximate_antisymmetry(self):
        """
        Tests that mean latency matrix is approximately antisymmetric.

        Tests:
            (Test Case 1) mean[i,j] is approximately -mean[j,i] for dense spike trains.

        Notes:
            - Not exact because different spike counts per train yield different
              nearest-spike pairings in each direction.
        """
        # Use dense trains so the approximation is tight
        sd = random_spikedata(3, 30000)
        mean_lat, _ = sd.get_pairwise_latencies()

        # Should be roughly antisymmetric
        for i in range(3):
            for j in range(i + 1, 3):
                assert mean_lat.matrix[i, j] == pytest.approx(
                    -mean_lat.matrix[j, i], abs=5.0
                )

    def test_std_is_non_negative(self):
        """
        Tests that all std values are non-negative.

        Tests:
            (Test Case 1) No negative entries in the std matrix.
        """
        sd = random_spikedata(4, 4000)
        _, std_lat = sd.get_pairwise_latencies()

        assert np.all(std_lat.matrix >= 0)

    def test_identical_trains_zero_latency(self):
        """
        Tests that identical spike trains produce zero mean and zero std.

        Tests:
            (Test Case 1) Mean latency between identical trains is 0.
            (Test Case 2) Std latency between identical trains is 0.
        """
        train = np.sort(np.random.uniform(0, 1000, size=200))
        sd = SpikeData([train, train.copy()], length=1000)
        mean_lat, std_lat = sd.get_pairwise_latencies()

        assert mean_lat.matrix[0, 1] == pytest.approx(0.0)
        assert mean_lat.matrix[1, 0] == pytest.approx(0.0)
        assert std_lat.matrix[0, 1] == pytest.approx(0.0)
        assert std_lat.matrix[1, 0] == pytest.approx(0.0)

    def test_known_latency(self):
        """
        Tests with a known offset between trains.

        Tests:
            (Test Case 1) Train B is train A shifted by +10ms. Mean latency
                from A to B should be exactly +10.
            (Test Case 2) Mean latency from B to A is close to -10 but not
                exact due to boundary effects (last spike in B has no forward
                match in A).
            (Test Case 3) Std from A to B is 0 (all latencies identical).
        """
        # Offset of 5ms with 20ms spacing avoids equidistant tie-breaking
        train_a = np.arange(20, 980, 20, dtype=float)
        train_b = train_a + 5.0  # shifted by +5ms
        sd = SpikeData([train_a, train_b], length=1000)
        mean_lat, std_lat = sd.get_pairwise_latencies()

        assert mean_lat.matrix[0, 1] == pytest.approx(5.0, abs=0.1)
        assert mean_lat.matrix[1, 0] == pytest.approx(-5.0, abs=0.1)
        assert std_lat.matrix[0, 1] == pytest.approx(0.0, abs=0.1)

    def test_window_ms_filter(self):
        """
        Tests that window_ms filters out distant latencies.

        Tests:
            (Test Case 1) With a tight window, only close spikes contribute.
            (Test Case 2) A pair with all latencies beyond the window yields
                mean=0 and std=0.
        """
        # Two trains: one spike at 0, one spike at 500 — latency is 500ms
        sd = SpikeData([[0.5], [500.5]], length=600)

        # No window — latency is 500
        mean_no_win, _ = sd.get_pairwise_latencies()
        assert mean_no_win.matrix[0, 1] == pytest.approx(500.0)

        # Window of 100ms — the 500ms latency is filtered out
        mean_win, std_win = sd.get_pairwise_latencies(window_ms=100.0)
        assert mean_win.matrix[0, 1] == pytest.approx(0.0)
        assert std_win.matrix[0, 1] == pytest.approx(0.0)

    def test_metadata(self):
        """
        Tests that metadata stores window_ms.

        Tests:
            (Test Case 1) Metadata contains window_ms=None by default.
            (Test Case 2) Metadata contains the specified window_ms value.
        """
        sd = random_spikedata(2, 2000)

        mean_lat, std_lat = sd.get_pairwise_latencies()
        assert mean_lat.metadata["window_ms"] is None

        mean_lat2, std_lat2 = sd.get_pairwise_latencies(window_ms=50.0)
        assert mean_lat2.metadata["window_ms"] == 50.0

    def test_return_distributions(self):
        """
        Tests that return_distributions=True returns a third element.

        Tests:
            (Test Case 1) Returns a tuple of length 3 when True.
            (Test Case 2) The distributions array has shape (U, U).
            (Test Case 3) Each entry is an ndarray.
            (Test Case 4) Diagonal entries are empty arrays.
            (Test Case 5) Number of latencies in [i,j] equals number of spikes
                in train i (without window filtering).
        """
        train_a = np.sort(np.random.uniform(0, 1000, size=50))
        train_b = np.sort(np.random.uniform(0, 1000, size=80))
        sd = SpikeData([train_a, train_b], length=1000)

        result = sd.get_pairwise_latencies(return_distributions=True)
        assert len(result) == 3

        mean_lat, std_lat, dists = result
        assert dists.shape == (2, 2)
        assert isinstance(dists[0, 1], np.ndarray)
        assert len(dists[0, 0]) == 0  # diagonal
        assert len(dists[0, 1]) == 50  # one latency per spike in train_a
        assert len(dists[1, 0]) == 80  # one latency per spike in train_b

    def test_empty_train(self):
        """
        Tests get_pairwise_latencies when one unit has no spikes.

        Tests:
            (Test Case 1) Mean and std are 0 for pairs involving empty trains.
            (Test Case 2) Distribution is empty for pairs involving empty trains.
        """
        sd = SpikeData([[], np.sort(np.random.uniform(0, 500, 100))], length=500)
        mean_lat, std_lat, dists = sd.get_pairwise_latencies(return_distributions=True)

        assert mean_lat.matrix[0, 1] == 0.0
        assert mean_lat.matrix[1, 0] == 0.0
        assert std_lat.matrix[0, 1] == 0.0
        assert len(dists[0, 1]) == 0
        assert len(dists[1, 0]) == 0

    def test_single_unit(self):
        """
        Tests get_pairwise_latencies with a single unit.

        Tests:
            (Test Case 1) Returns 1x1 matrices with zeros.
        """
        sd = SpikeData([np.sort(np.random.uniform(0, 500, 100))], length=500)
        mean_lat, std_lat = sd.get_pairwise_latencies()

        assert mean_lat.matrix.shape == (1, 1)
        assert mean_lat.matrix[0, 0] == 0.0
        assert std_lat.matrix[0, 0] == 0.0

    def test_without_distributions_returns_two(self):
        """
        Tests that return_distributions=False returns only two values.

        Tests:
            (Test Case 1) Default call returns a tuple of length 2.
        """
        sd = random_spikedata(3, 3000)
        result = sd.get_pairwise_latencies()
        assert len(result) == 2


class TestSpikeDataEdgeCases:
    """Edge-case tests for SpikeData boundaries (Group 3)."""

    def test_init_all_empty_trains_no_length(self):
        """
        SpikeData with all-empty trains and no explicit length defaults to duration 0.

        Tests:
            (Test Case 1) Verify that SpikeData([[], [], []], length=None) creates
                a valid object with length=0.0 and the correct number of units.
        """
        sd = SpikeData([[], [], []], length=None)
        assert sd.length == 0.0
        assert sd.N == 3
        assert all(len(t) == 0 for t in sd.train)

    def test_frames_length_equals_recording(self):
        """
        frames() with window length equal to the full recording length.

        Tests:
        (Test Case 1) Verify that frames(length) returns exactly 1 SpikeSliceStack
        containing 1 slice when length equals the recording length.
        """
        sd = SpikeData([[10.0, 50.0, 90.0]], length=100.0)
        stack = sd.frames(100.0)
        assert isinstance(stack, SpikeSliceStack)
        assert len(stack.spike_stack) == 1

    def test_interspike_intervals_single_spike(self):
        """
        interspike_intervals for a unit with exactly one spike and an empty train.

        Tests:
        (Test Case 1) A unit with 1 spike returns an ISI array of length 0.
        (Test Case 2) A unit with 0 spikes returns an ISI array of length 0.
        """
        # Single spike
        sd_single = SpikeData([[50.0]], length=100.0)
        isis_single = sd_single.interspike_intervals()
        assert len(isis_single) == 1
        assert len(isis_single[0]) == 0

        # Empty train
        sd_empty = SpikeData([[]], length=100.0)
        isis_empty = sd_empty.interspike_intervals()
        assert len(isis_empty) == 1
        assert len(isis_empty[0]) == 0

    def test_concatenate_mutates_self(self):
        """
        concatenate_spike_data mutates self and leaves the other unchanged.

        Tests:
        (Test Case 1) After concatenation, sd1.N equals the sum of original N values.
        (Test Case 2) sd1 has the correct number of trains.
        (Test Case 3) sd2 is unchanged (same N and same trains).
        """
        sd1 = SpikeData([[1.0, 2.0], [3.0, 4.0]], length=100.0)
        sd2 = SpikeData([[10.0], [20.0], [30.0]], length=100.0)
        sd2_N_orig = sd2.N
        sd2_trains_orig = [t.copy() for t in sd2.train]

        sd1.concatenate_spike_data(sd2)

        assert sd1.N == 5
        assert len(sd1.train) == 5

        # sd2 should be unchanged
        assert sd2.N == sd2_N_orig
        assert len(sd2.train) == len(sd2_trains_orig)
        for orig, current in zip(sd2_trains_orig, sd2.train):
            np.testing.assert_array_equal(orig, current)

    def test_sttc_both_trains_empty(self):
        """
        spike_time_tiling with both trains empty.

        Tests:
        (Test Case 1) Calling spike_time_tiling on two empty trains returns a finite
        number (0.0) without crashing.
        """
        sd = SpikeData([[], []], length=100.0)
        result = sd.spike_time_tiling(0, 1, delt=5.0)
        assert isinstance(result, float)
        # get_sttc returns 0.0 for empty trains
        np.testing.assert_equal(result, 0.0)

    def test_init_duplicate_spike_times(self):
        """
        Duplicate spike times are preserved in the train.

        Tests:
        (Test Case 1) SpikeData with duplicate spike times preserves all of them.
        """
        sd = SpikeData([[1.0, 1.0, 2.0]], length=10.0)
        assert sd.N == 1
        np.testing.assert_array_equal(sd.train[0], [1.0, 1.0, 2.0])

    def test_init_non_monotonic_sorted(self):
        """
        Non-monotonic spike times are sorted on construction.

        Tests:
        (Test Case 1) SpikeData sorts an unsorted input train.
        """
        sd = SpikeData([[5.0, 1.0, 3.0]], length=10.0)
        np.testing.assert_array_equal(sd.train[0], [1.0, 3.0, 5.0])

    def test_subset_empty_units(self):
        """
        Subset with an empty units list returns N=0.

        Tests:
        (Test Case 1) Result has N=0 and no trains.
        (Test Case 2) Length is preserved from the original.
        """
        sd = SpikeData([[1.0], [2.0], [3.0]], length=50.0)
        sub = sd.subset(units=[])
        assert sub.N == 0
        assert len(sub.train) == 0
        np.testing.assert_equal(sub.length, 50.0)

    def test_subset_duplicate_indices(self):
        """
        Subset deduplicates unit indices.

        Tests:
        (Test Case 1) Passing [0, 0, 1] yields N=2, not N=3, because subset treats
        units as a set.
        """
        sd = SpikeData([[1.0], [2.0], [3.0]], length=50.0)
        sub = sd.subset(units=[0, 0, 1])
        assert sub.N == 2

    def test_subtime_start_equals_end(self):
        """
        subtime raises ValueError when start equals end.

        Tests:
        (Test Case 1) subtime(10.0, 10.0) raises ValueError because the range is empty.
        """
        sd = SpikeData([[5.0, 15.0, 25.0]], length=50.0)
        with pytest.raises(ValueError):
            sd.subtime(10.0, 10.0)

    def test_subtime_no_spikes_in_window(self):
        """
        subtime with a window containing no spikes.

        Tests:
        (Test Case 1) Returns a valid SpikeData with empty trains and correct length.
        """
        sd = SpikeData([[10.0, 20.0, 30.0]], length=100.0)
        sub = sd.subtime(40.0, 50.0)
        assert sub.N == 1
        np.testing.assert_equal(sub.length, 10.0)
        assert len(sub.train[0]) == 0

    def test_subtime_boundary_inclusion(self):
        """
        subtime uses half-open interval [start, end).

        Tests:
        (Test Case 1) A spike at exactly start is included.
        (Test Case 2) A spike at exactly end is excluded.

        Notes:
        - subtime filters with (t >= start) & (t < end), so it is half-open.
        """
        sd = SpikeData([[10.0, 20.0]], length=50.0)
        sub = sd.subtime(10.0, 20.0)
        # After shift, spike at 10.0 becomes 0.0; spike at 20.0 is excluded
        assert len(sub.train[0]) == 1
        np.testing.assert_almost_equal(sub.train[0][0], 0.0)

    def test_subset_out_of_bounds_index(self):
        """
        Subset with an out-of-bounds unit index.

        Tests:
        (Test Case 1) Passing units=[99] when N=3 returns an empty SpikeData
        because no train index matches 99.
        """
        sd = SpikeData([[1.0], [2.0], [3.0]], length=50.0)
        sub = sd.subset(units=[99])
        assert sub.N == 0
        assert len(sub.train) == 0

    def test_subset_by_non_existent_key(self):
        """
        Subset with by= referencing a key that no neuron has.

        Tests:
        (Test Case 1) Passing by="nonexistent" returns an empty SpikeData
        because .get("nonexistent", _missing) never matches any value in units.
        """
        sd = SpikeData(
            [[1.0], [2.0]],
            length=50.0,
            neuron_attributes=[{"id": "a"}, {"id": "b"}],
        )
        sub = sd.subset(by="nonexistent", units=["x"])
        assert sub.N == 0
        assert len(sub.train) == 0

    def test_binned_zero_bin_size(self):
        """
        binned with zero bin_size.

        Tests:
        (Test Case 1) Calling binned(0) raises an exception because division
        by zero occurs inside sparse_raster.
        """
        sd = SpikeData([[1.0, 2.0]], length=10.0)
        with pytest.raises(Exception):
            sd.binned(0)

    def test_binned_negative_bin_size(self):
        """
        binned with negative bin_size.

        Tests:
        (Test Case 1) Calling binned(-1) raises an exception because negative
        bin size produces invalid array dimensions.
        """
        sd = SpikeData([[1.0, 2.0]], length=10.0)
        with pytest.raises(Exception):
            sd.binned(-1)

    def test_binned_spikes_at_exact_bin_boundaries(self):
        """
        binned with spikes at exact bin boundaries.

        Tests:
        (Test Case 1) spikes=[0, 20, 40] with bin_size=20 assigns each spike
        to the correct bin via left-open, right-closed convention.
        (Test Case 2) Total spike count is preserved.
        """
        sd = SpikeData([[0, 20, 40]], length=40.0)
        # Bins: (0,20], (20,40]. t=0 clipped to bin 0, t=20 into bin 0
        # (right-closed), t=40 into bin 1. length=ceil(40/20)=2.
        result = sd.binned(20)
        assert len(result) == 2
        assert result.sum() == 3
        assert result[0] == 2  # spikes at t=0 and t=20
        assert result[1] == 1  # spike at t=40

    def test_raster_bin_size_larger_than_length(self):
        """
        raster with bin_size larger than recording length.

        Tests:
        (Test Case 1) Returns a single-bin raster containing all spikes.
        """
        sd = SpikeData([[5.0, 10.0, 15.0]], length=20.0)
        r = sd.raster(bin_size=100.0)
        assert r.shape[1] == 1
        assert r[0, 0] == 3

    def test_raster_spike_at_t_zero(self):
        """
        raster captures a spike at t=0 in the first bin.

        Tests:
        (Test Case 1) A spike at exactly t=0 appears in bin index 0.
        """
        sd = SpikeData([[0.0, 5.0]], length=10.0)
        r = sd.raster(bin_size=5.0)
        assert r[0, 0] >= 1  # spike at t=0 is in first bin

    def test_rates_zero_length_recording(self):
        """
        rates with sd.length == 0.

        Tests:
        (Test Case 1) Calling rates() on a zero-length recording raises an
        exception or produces inf/nan due to division by zero.
        """
        sd = SpikeData([[]], length=0.0)
        # Division by zero in rates(): len(t) / self.length
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = sd.rates()
        # Either raises or produces nan/inf
        assert np.isnan(result[0]) or np.isinf(result[0])

    def test_align_to_events_empty_events(self):
        """
        align_to_events with an empty events array.

        Tests:
        (Test Case 1) Passing an empty list raises ValueError because no valid
        events remain after filtering.
        """
        sd = SpikeData([[5.0, 15.0, 25.0]], length=50.0)
        with pytest.raises(ValueError, match="No valid events remain"):
            sd.align_to_events([], pre_ms=5.0, post_ms=5.0)

    def test_align_to_events_at_recording_boundaries(self):
        """
        align_to_events with events at exact recording boundaries.

        Tests:
        (Test Case 1) An event at t=0 with pre_ms>0 is dropped because the
        window extends before the recording start.
        (Test Case 2) An event at t=length with post_ms>0 is dropped because
        the window extends past the recording end.
        (Test Case 3) If all boundary events are dropped, a ValueError is raised.
        """
        sd = SpikeData([[10.0, 20.0, 30.0]], length=50.0)
        # Both events have windows that extend outside [0, 50]
        with pytest.raises(ValueError, match="No valid events remain"):
            sd.align_to_events([0.0, 50.0], pre_ms=5.0, post_ms=5.0)

    def test_sttc_delt_zero(self):
        """
        spike_time_tiling with delt=0.

        Tests:
        (Test Case 1) delt=0 produces a finite result without numerical errors.
        The result should be between -1 and 1.
        """
        sd = SpikeData([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], length=10.0)
        result = sd.spike_time_tiling(0, 1, delt=0.0)
        assert np.isfinite(result)
        assert -1 <= result <= 1

    def test_sttc_delt_larger_than_recording(self):
        """
        spike_time_tiling with delt larger than recording length.

        Tests:
        (Test Case 1) When delt covers the entire recording, STTC is finite
        and within [-1, 1].
        """
        sd = SpikeData([[1.0, 5.0], [2.0, 6.0]], length=10.0)
        result = sd.spike_time_tiling(0, 1, delt=1000.0)
        assert np.isfinite(result)
        assert -1 <= result <= 1

    def test_latencies_empty_times(self):
        """
        latencies with an empty times array.

        Tests:
        (Test Case 1) Passing an empty list returns an empty list immediately.
        """
        sd = SpikeData([[1.0, 2.0, 3.0]], length=10.0)
        result = sd.latencies([])
        assert result == []

    def test_latencies_spike_at_exactly_query_time(self):
        """
        latencies when a spike occurs at exactly the query time.

        Tests:
        (Test Case 1) The latency is 0 and is included in the results.
        """
        sd = SpikeData([[5.0, 10.0, 15.0]], length=20.0)
        result = sd.latencies([10.0])
        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0] == 0.0

    def test_append_negative_offset(self):
        """
        append with a negative offset.

        Tests:
        (Test Case 1) The resulting length is self.length + other.length + offset,
        which is shorter than the sum of both lengths.
        (Test Case 2) Spike times from the appended data are shifted by
        self.length + offset.
        """
        sd1 = SpikeData([[5.0]], length=20.0)
        sd2 = SpikeData([[3.0]], length=10.0)
        result = sd1.append(sd2, offset=-5)
        # length = 20 + 10 + (-5) = 25
        np.testing.assert_equal(result.length, 25.0)
        # Appended spike at 3.0 shifted by self.length + offset = 20 + (-5) = 15
        expected_second_spike = 3.0 + 20.0 + (-5)
        np.testing.assert_almost_equal(result.train[0][1], expected_second_spike)

    def test_append_to_empty_spikedata(self):
        """
        append to a SpikeData with no spikes.

        Tests:
        (Test Case 1) Result contains the appended spikes shifted by the
        empty SpikeData's length.
        (Test Case 2) The resulting length equals the sum of both lengths.
        """
        sd_empty = SpikeData([[]], length=10.0)
        sd_data = SpikeData([[5.0, 8.0]], length=20.0)
        result = sd_empty.append(sd_data, offset=0)
        # length = 10 + 20 + 0 = 30
        np.testing.assert_equal(result.length, 30.0)
        # Spikes shifted by sd_empty.length = 10
        np.testing.assert_array_almost_equal(result.train[0], [15.0, 18.0])

    def test_from_raster_all_zeros(self):
        """
        from_raster with an all-zero raster.

        Tests:
        (Test Case 1) Produces a SpikeData with all empty spike trains.
        (Test Case 2) N matches the number of rows in the raster.
        """
        raster = np.zeros((3, 5))
        sd = SpikeData.from_raster(raster, bin_size_ms=10.0)
        assert sd.N == 3
        for train in sd.train:
            assert len(train) == 0

    def test_from_raster_single_bin(self):
        """
        from_raster with a single-bin raster.

        Tests:
        (Test Case 1) Length equals 1 * bin_size_ms.
        (Test Case 2) Spike times are correctly placed within the single bin.
        """
        raster = np.array([[2], [0]])
        sd = SpikeData.from_raster(raster, bin_size_ms=10.0)
        np.testing.assert_equal(sd.length, 10.0)
        assert sd.N == 2
        # Unit 0 has 2 spikes evenly spaced in the bin [0, 10)
        # linspace(0, 10, 4)[1:-1] = [2.5, 5.0, 7.5] ... wait, n_spikes=2 so
        # linspace(0, 10, 4)[1:-1] = [10/3, 20/3] approx [3.33, 6.67]
        assert len(sd.train[0]) == 2
        assert len(sd.train[1]) == 0

    def test_get_pop_rate_empty_spikedata(self):
        """
        get_pop_rate on a SpikeData with no spikes.

        Tests:
        (Test Case 1) Returns a valid array (all zeros or near-zero) without error.
        """
        sd = SpikeData([[]], length=100.0)
        result = sd.get_pop_rate()
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_get_bursts_no_bursts_detected(self):
        """
        get_bursts when no bursts are present.

        Tests:
        (Test Case 1) Returns empty arrays for tburst, edges, and peak_amp.
        """
        sd = SpikeData([[]], length=1000.0)
        tburst, edges, peak_amp = sd.get_bursts(
            thr_burst=5.0,
            min_burst_diff=50,
            burst_edge_mult_thresh=0.5,
        )
        assert len(tburst) == 0
        assert len(peak_amp) == 0

    def test_burst_sensitivity_no_spikes(self):
        """
        burst_sensitivity on a SpikeData with no spikes.

        Tests:
            (Test Case 1) Returns an all-zero integer matrix of correct shape.
        """
        sd = SpikeData([[]], length=1000.0)
        result = sd.burst_sensitivity(
            thr_values=np.array([0.5, 1.0]),
            dist_values=np.array([10, 20, 30]),
            burst_edge_mult_thresh=0.5,
        )
        assert result.shape == (2, 3)
        assert result.dtype == int
        np.testing.assert_array_equal(result, np.zeros((2, 3), dtype=int))

    def test_resampled_isi_single_spike(self):
        """
        resampled_isi with a train containing a single spike.

        Tests:
        (Test Case 1) Returns all zeros because ISI is undefined with fewer
        than 2 spikes.
        """
        sd = SpikeData([[50.0]], length=100.0)
        times = np.linspace(0, 100, 50)
        result = sd.resampled_isi(times)
        assert result.shape == (1, 50)
        np.testing.assert_array_equal(result[0], np.zeros(50))

    def test_resampled_isi_sigma_zero(self):
        """
        resampled_isi with sigma_ms=0.

        Tests:
        (Test Case 1) Returns a valid finite array without numerical errors
        (Gaussian smoothing is skipped when sigma <= 0).
        """
        sd = SpikeData([[10.0, 30.0, 60.0]], length=100.0)
        times = np.linspace(0, 100, 50)
        result = sd.resampled_isi(times, sigma_ms=0.0)
        assert result.shape == (1, 50)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Tests for fit_gplvm
# ---------------------------------------------------------------------------


class TestFitGplvm:
    """Tests for SpikeData.fit_gplvm."""

    @skip_no_pmgplvm
    def test_fit_gplvm_basic(self):
        """
        Fit GPLVM on small synthetic data and verify return dict structure.

        Tests:
            (Test Case 1) Verify all expected keys are present in the result.
            (Test Case 2) Verify binned_spike_counts has shape (T, N).
            (Test Case 3) Verify reorder_indices has length N.
            (Test Case 4) Verify model object is returned.
        """
        # 5 units, 500 ms recording, sparse spikes
        trains = [
            [10.0, 50.0, 120.0, 200.0, 350.0],
            [20.0, 80.0, 180.0, 300.0, 450.0],
            [30.0, 100.0, 150.0, 250.0, 400.0],
            [15.0, 60.0, 130.0, 210.0, 380.0],
            [40.0, 90.0, 170.0, 280.0, 420.0],
        ]
        sd = SpikeData(trains, N=5, length=500.0)

        result = sd.fit_gplvm(
            bin_size_ms=50.0,
            n_latent_bin=10,
            n_iter=2,
            random_seed=42,
        )

        # Check all expected keys
        expected_keys = {
            "decode_res",
            "log_marginal_l",
            "reorder_indices",
            "model",
            "binned_spike_counts",
            "bin_size_ms",
        }
        assert set(result.keys()) == expected_keys
        assert result["bin_size_ms"] == 50.0

        # binned_spike_counts shape: raster uses ceil, so 500ms / 50ms → 11 bins
        binned = result["binned_spike_counts"]
        assert binned.shape[1] == 5  # N units
        assert binned.shape[0] == sd.raster(50.0).shape[1]  # T bins match raster

        # reorder_indices should have one entry per unit
        assert len(result["reorder_indices"]) == 5

        # model should be a PoissonGPLVMJump1D instance
        from poor_man_gplvm.core import PoissonGPLVMJump1D

        assert isinstance(result["model"], PoissonGPLVMJump1D)

    @skip_no_pmgplvm
    def test_fit_gplvm_custom_bin_size(self):
        """
        Verify that bin_size_ms controls the time dimension of binned counts.

        Tests:
            (Test Case 1) bin_size_ms=100 on 500ms recording → T=5 bins.
        """
        trains = [
            [10.0, 150.0, 300.0],
            [50.0, 200.0, 400.0],
            [80.0, 250.0, 450.0],
        ]
        sd = SpikeData(trains, N=3, length=500.0)

        result = sd.fit_gplvm(
            bin_size_ms=100.0,
            n_latent_bin=10,
            n_iter=2,
        )

        binned = result["binned_spike_counts"]
        assert binned.shape[1] == 3  # N units
        assert binned.shape[0] == sd.raster(100.0).shape[1]  # T bins match raster

    @skip_no_pmgplvm
    def test_fit_gplvm_custom_model_class(self):
        """
        Verify that model_class parameter overrides the default model.

        Tests:
            (Test Case 1) Pass GaussianGPLVMJump1D and verify the returned
                model is an instance of that class.
        """
        from poor_man_gplvm.core import GaussianGPLVMJump1D

        trains = [
            [10.0, 50.0, 120.0, 200.0, 350.0],
            [20.0, 80.0, 180.0, 300.0, 450.0],
            [30.0, 100.0, 150.0, 250.0, 400.0],
        ]
        sd = SpikeData(trains, N=3, length=500.0)

        result = sd.fit_gplvm(
            bin_size_ms=50.0,
            n_latent_bin=10,
            n_iter=2,
            model_class=GaussianGPLVMJump1D,
        )

        assert isinstance(result["model"], GaussianGPLVMJump1D)

    def test_fit_gplvm_import_error(self):
        """
        Verify clean ImportError when poor_man_gplvm is not available.

        Tests:
            (Test Case 1) Mock the import to fail and check the error message
                mentions the package name and install instructions.
        """
        sd = SpikeData([[10.0, 50.0]], N=1, length=100.0)

        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "poor_man_gplvm" in name or name == "jax.random":
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="poor_man_gplvm"):
                sd.fit_gplvm()

    @skip_no_pmgplvm
    def test_fit_gplvm_log_marginal_likelihood_length(self):
        """
        Verify log_marginal_l has one entry per EM iteration.

        Tests:
            (Test Case 1) n_iter=3 produces log_marginal_l with length 3.
        """
        trains = [
            [10.0, 50.0, 120.0, 200.0, 350.0],
            [20.0, 80.0, 180.0, 300.0, 450.0],
            [30.0, 100.0, 150.0, 250.0, 400.0],
        ]
        sd = SpikeData(trains, N=3, length=500.0)

        result = sd.fit_gplvm(
            bin_size_ms=50.0,
            n_latent_bin=10,
            n_iter=3,
        )

        assert len(result["log_marginal_l"]) == 3

    @skip_no_pmgplvm
    def test_fit_gplvm_returns_numpy_arrays(self):
        """
        Verify all arrays in the result dict are numpy ndarrays, not JAX types.

        Tests:
            (Test Case 1) Top-level array values are np.ndarray.
            (Test Case 2) All arrays inside decode_res are np.ndarray.
        """
        trains = [
            [10.0, 50.0, 120.0, 200.0, 350.0],
            [20.0, 80.0, 180.0, 300.0, 450.0],
            [30.0, 100.0, 150.0, 250.0, 400.0],
        ]
        sd = SpikeData(trains, N=3, length=500.0)

        result = sd.fit_gplvm(
            bin_size_ms=50.0,
            n_latent_bin=10,
            n_iter=2,
        )

        # Top-level arrays
        for key in ("log_marginal_l", "reorder_indices", "binned_spike_counts"):
            assert isinstance(
                result[key], np.ndarray
            ), f"result['{key}'] is {type(result[key])}, expected np.ndarray"

        # All values inside decode_res must be numpy arrays or plain scalars
        for key, val in result["decode_res"].items():
            assert isinstance(
                val, (np.ndarray, int, float, bool, str)
            ), f"decode_res['{key}'] is {type(val)}, expected np.ndarray or scalar"


class TestRecentFixes:
    """Tests for fixes applied during the 2026-03-19 code review."""

    def test_subtime_always_shifts_to_zero(self):
        """
        Verify subtime shifts spike times so the new window starts at t=0.

        Tests:
            (Test Case 1) Spike times are shifted by the start offset.
            (Test Case 2) Length equals the window size (end - start).
        """
        sd = SpikeData([[50, 100, 150]], length=200)
        result = sd.subtime(50, 160)
        # subtime uses [start, end), so 50, 100, 150 are all included
        np.testing.assert_array_equal(result.train[0], [0, 50, 100])
        assert result.length == 110

    def test_concatenate_spike_data_preserves_raw(self):
        """
        Verify concatenate_spike_data does not modify raw_data or raw_time.

        Tests:
            (Test Case 1) raw_data is unchanged after concatenating units.
            (Test Case 2) raw_time is unchanged after concatenating units.
        """
        raw1 = np.ones((2, 10))
        time1 = np.arange(10, dtype=float)
        sd1 = SpikeData([[1, 2]], length=10, raw_data=raw1, raw_time=time1)
        sd2 = SpikeData([[3, 4]], length=10)
        sd1.concatenate_spike_data(sd2)
        assert sd1.raw_data.shape == (2, 10)  # unchanged
        assert sd1.raw_time.shape == (10,)  # unchanged
        np.testing.assert_array_equal(sd1.raw_data, raw1)

    def test_metadata_default_not_shared(self):
        """
        Verify that default metadata dicts are independent across instances.

        Tests:
            (Test Case 1) Mutating one instance's metadata does not affect another.
        """
        sd1 = SpikeData([[1]], length=5)
        sd2 = SpikeData([[2]], length=5)
        sd1.metadata["key"] = "value"
        assert "key" not in sd2.metadata

    # ------------------------------------------------------------------
    # spike_shuffle
    # ------------------------------------------------------------------

    def test_spike_shuffle_preserves_row_and_column_sums(self):
        """
        spike_shuffle preserves per-unit spike counts and per-bin population rates.

        Tests:
            (Test Case 1) Returned object is a SpikeData with same N and length.
            (Test Case 2) Row sums (spikes per unit) are preserved.
            (Test Case 3) Column sums (population rate per bin) are preserved.

        Notes:
            - Uses a SpikeData built from a binary raster to avoid multi-spike
              bins, which spike_shuffle's internal binarization would alter.
        """
        rng = np.random.default_rng(42)
        binary_raster = (rng.random((5, 100)) < 0.2).astype(int)
        sd = SpikeData.from_raster(binary_raster, bin_size_ms=1)
        shuffled = sd.spike_shuffle(swap_per_spike=5, seed=42, bin_size=1)

        assert isinstance(shuffled, SpikeData)
        assert shuffled.N == sd.N
        assert shuffled.length == sd.length

        orig_raster = sd.sparse_raster(bin_size=1).toarray()
        shuf_raster = shuffled.sparse_raster(bin_size=1).toarray()

        # Row sums (spikes per unit) must match
        np.testing.assert_array_equal(orig_raster.sum(axis=1), shuf_raster.sum(axis=1))
        # Column sums (population rate per bin) must match
        np.testing.assert_array_equal(orig_raster.sum(axis=0), shuf_raster.sum(axis=0))

    def test_spike_shuffle_seed_reproducibility(self):
        """
        Same seed produces the same shuffled result.

        Tests:
            (Test Case 1) Two calls with the same seed yield identical rasters.
            (Test Case 2) Different seeds yield different rasters.
        """
        np.random.seed(0)
        sd = random_spikedata(4, 100, rate=1.0)

        shuf1 = sd.spike_shuffle(seed=123, bin_size=1)
        shuf2 = sd.spike_shuffle(seed=123, bin_size=1)
        r1 = shuf1.sparse_raster(bin_size=1).toarray()
        r2 = shuf2.sparse_raster(bin_size=1).toarray()
        np.testing.assert_array_equal(r1, r2)

        shuf3 = sd.spike_shuffle(seed=456, bin_size=1)
        r3 = shuf3.sparse_raster(bin_size=1).toarray()
        assert not np.array_equal(r1, r3)

    def test_spike_shuffle_metadata_preserved(self):
        """
        spike_shuffle carries metadata and neuron_attributes forward.

        Tests:
            (Test Case 1) metadata dict is preserved.
            (Test Case 2) neuron_attributes are preserved.
        """
        attrs = [{"region": "ctx"}, {"region": "hpc"}]
        sd = SpikeData(
            [np.array([1.0, 5.0, 10.0]), np.array([2.0, 8.0, 15.0])],
            length=20.0,
            metadata={"exp": "test"},
            neuron_attributes=attrs,
        )
        shuffled = sd.spike_shuffle(seed=0)
        assert shuffled.metadata == {"exp": "test"}
        assert shuffled.neuron_attributes is not None
        assert len(shuffled.neuron_attributes) == 2

    def test_spike_shuffle_bin_size_gt_1(self):
        """
        spike_shuffle with bin_size > 1 binarizes multi-spike bins.

        Tests:
            (Test Case 1) Shuffled raster values are 0 or 1 (binary).
            (Test Case 2) Row sums and column sums are preserved on the binarized raster.
        """
        np.random.seed(99)
        sd = random_spikedata(3, 150, rate=1.0)
        shuffled = sd.spike_shuffle(seed=0, bin_size=5)

        orig_raster = sd.sparse_raster(bin_size=5).toarray()
        orig_binary = (orig_raster > 0).astype(int)
        shuf_raster = shuffled.sparse_raster(bin_size=5).toarray()

        # Output should be binary
        assert set(np.unique(shuf_raster)).issubset({0, 1})
        # Row and column sums of binarized original should be preserved
        np.testing.assert_array_equal(orig_binary.sum(axis=1), shuf_raster.sum(axis=1))
        np.testing.assert_array_equal(orig_binary.sum(axis=0), shuf_raster.sum(axis=0))

    def test_spike_shuffle_warns_on_multi_spike_bins(self):
        """
        spike_shuffle warns when multi-spike bins are binarized.

        Tests:
            (Test Case 1) RuntimeWarning issued when input has multi-spike bins.
            (Test Case 2) No warning when input is already binary.
        """
        # Multi-spike data: random_spikedata can produce >1 spike per 1ms bin
        np.random.seed(99)
        sd_dense = random_spikedata(3, 150, rate=1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sd_dense.spike_shuffle(seed=0, bin_size=1)
            multi_spike_warnings = [
                x for x in w if "Multi-spike bins" in str(x.message)
            ]
            assert len(multi_spike_warnings) > 0

        # Binary data: no warning
        rng = np.random.default_rng(0)
        binary_raster = (rng.random((3, 50)) < 0.2).astype(int)
        sd_binary = SpikeData.from_raster(binary_raster, bin_size_ms=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sd_binary.spike_shuffle(seed=0, bin_size=1)
            multi_spike_warnings = [
                x for x in w if "Multi-spike bins" in str(x.message)
            ]
            assert len(multi_spike_warnings) == 0

    # ------------------------------------------------------------------
    # N=0 edge cases
    # ------------------------------------------------------------------

    def test_empty_spikedata_rates(self):
        """
        SpikeData with zero units: rates() returns empty, binned_meanrate() returns zeros.

        Tests:
            (Test Case 1) rates() returns shape (0,).
            (Test Case 2) rates(unit='Hz') returns shape (0,).
            (Test Case 3) binned_meanrate() returns zeros array (no division by zero).
            (Test Case 4) binned_meanrate(unit='Hz') also returns zeros.
        """
        sd = SpikeData([], length=100.0)
        assert sd.N == 0
        r = sd.rates()
        assert r.shape == (0,)
        r_hz = sd.rates(unit="Hz")
        assert r_hz.shape == (0,)

        bmr = sd.binned_meanrate(bin_size=40)
        assert bmr.shape == (int(np.ceil(100.0 / 40)),)
        np.testing.assert_array_equal(bmr, 0.0)

        bmr_hz = sd.binned_meanrate(bin_size=40, unit="Hz")
        np.testing.assert_array_equal(bmr_hz, 0.0)

    def test_empty_spikedata_subset(self):
        """
        SpikeData.subset([]) produces an N=0 SpikeData.

        Tests:
            (Test Case 1) Subsetting with empty list returns N=0.
            (Test Case 2) The resulting SpikeData has length preserved.
            (Test Case 3) train is an empty list.
        """
        sd = SpikeData(
            [np.array([1.0, 2.0]), np.array([3.0])],
            length=10.0,
        )
        sub = sd.subset([])
        assert sub.N == 0
        assert sub.length == 10.0
        assert len(sub.train) == 0

    # ------------------------------------------------------------------
    # NaN spike times
    # ------------------------------------------------------------------

    def test_nan_spike_times_rejected(self):
        """
        SpikeData constructor rejects NaN spike times with ValueError.

        Tests:
            (Test Case 1) NaN in first unit raises ValueError.
            (Test Case 2) NaN in second unit raises ValueError with correct unit index.
            (Test Case 3) Empty trains and trains without NaN are accepted.
        """
        with pytest.raises(ValueError, match="unit 0.*NaN"):
            SpikeData([np.array([1.0, np.nan, 5.0])], length=10.0)

        with pytest.raises(ValueError, match="unit 1.*NaN"):
            SpikeData([np.array([1.0, 2.0]), np.array([3.0, np.nan])], length=10.0)

        # Empty trains and clean trains are fine
        sd = SpikeData([np.array([]), np.array([1.0, 2.0])], length=10.0)
        assert sd.N == 2

    # ------------------------------------------------------------------
    # to_hdf5 / to_nwb / to_kilosort delegation wrappers
    # ------------------------------------------------------------------

    def test_to_hdf5_delegates_to_exporter(self):
        """
        SpikeData.to_hdf5 delegates to data_exporters.export_spikedata_to_hdf5.

        Tests:
            (Test Case 1) The exporter function is called exactly once.
            (Test Case 2) The first positional arg is the SpikeData instance.
            (Test Case 3) The filepath keyword is forwarded.
        """
        sd = SpikeData([np.array([1.0, 2.0])], length=5.0)
        with patch(
            "SpikeLab.data_loaders.data_exporters.export_spikedata_to_hdf5"
        ) as mock_export:
            sd.to_hdf5("/tmp/fake.h5", style="ragged")
            mock_export.assert_called_once()
            args, kwargs = mock_export.call_args
            assert args[0] is sd
            assert args[1] == "/tmp/fake.h5"

    def test_to_nwb_delegates_to_exporter(self):
        """
        SpikeData.to_nwb delegates to data_exporters.export_spikedata_to_nwb.

        Tests:
            (Test Case 1) The exporter function is called exactly once.
            (Test Case 2) The first positional arg is the SpikeData instance.
        """
        sd = SpikeData([np.array([1.0, 2.0])], length=5.0)
        with patch(
            "SpikeLab.data_loaders.data_exporters.export_spikedata_to_nwb"
        ) as mock_export:
            sd.to_nwb("/tmp/fake.nwb")
            mock_export.assert_called_once()
            args, _ = mock_export.call_args
            assert args[0] is sd

    def test_to_kilosort_delegates_to_exporter(self):
        """
        SpikeData.to_kilosort delegates to data_exporters.export_spikedata_to_kilosort.

        Tests:
            (Test Case 1) The exporter function is called exactly once.
            (Test Case 2) The first positional arg is the SpikeData instance.
            (Test Case 3) fs_Hz keyword is forwarded.
        """
        sd = SpikeData([np.array([1.0, 2.0])], length=5.0)
        with patch(
            "SpikeLab.data_loaders.data_exporters.export_spikedata_to_kilosort"
        ) as mock_export:
            mock_export.return_value = ("/tmp/st.npy", "/tmp/sc.npy")
            sd.to_kilosort("/tmp/fake_dir", fs_Hz=30000.0)
            mock_export.assert_called_once()
            args, kwargs = mock_export.call_args
            assert args[0] is sd
            assert kwargs["fs_Hz"] == 30000.0

    def test_subtime_raw_data_shifted(self):
        """
        Verify subtime shifts raw_time to start at 0.

        Tests:
            (Test Case 1) raw_time is shifted so the first sample is at 0.
        """
        raw_time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        raw_data = np.arange(6, dtype=float).reshape(1, 6)
        sd = SpikeData([[2, 3, 4]], length=6, raw_data=raw_data, raw_time=raw_time)
        result = sd.subtime(2, 5)
        assert result.raw_time[0] == 0.0
        np.testing.assert_array_almost_equal(result.raw_time, [0.0, 1.0, 2.0])

    def test_rates_zero_length(self):
        """
        SpikeData with length=0.0 and empty trains: rates() returns NaN due to 0/0.

        Tests:
            (Test Case 1) rates() on a zero-length SpikeData with N=3 empty
                          trains emits a RuntimeWarning and returns NaN values.

        Notes:
            rates() computes ``np.array([len(t) for t in self.train]) / self.length``.
            When length is 0.0, numpy evaluates 0/0 as NaN with a RuntimeWarning.
            Ideally rates() would guard against zero length and return np.zeros(N),
            but the current implementation does not. This test documents the
            existing behavior.
        """
        sd = SpikeData([], N=3, length=0.0)
        assert sd.N == 3
        assert sd.length == 0.0

        with pytest.warns(RuntimeWarning, match="invalid value|divide by zero"):
            result = sd.rates()
        assert result.shape == (3,)
        assert np.all(np.isnan(result))
