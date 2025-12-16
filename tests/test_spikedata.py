import unittest
from dataclasses import dataclass
import pathlib
import sys

import numpy as np
from scipy import stats

try:
    import quantities
    from neo.core import SpikeTrain
except ImportError:
    SpikeTrain = None
    quantities = None

# Ensure project root is on sys.path, then import package normally so relative imports work.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import spikedata.spikedata as spikedata
from spikedata import SpikeData


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


class SpikeDataTest(unittest.TestCase):
    def assertSpikeDataEqual(self, sda, sdb, msg=None):
        """
        Asserts that two SpikeData objects contain the same data.

        Tests:
        (Test Case 1) Compares the spike trains for equality in length and values (within tolerance).
        """
        for a, b in zip(sda.train, sdb.train):
            self.assertTrue(len(a) == len(b) and np.allclose(a, b), msg=msg)

    def assertSpikeDataSubtime(self, sd, sdsub, tmin, tmax, msg=None):
        """
        Asserts that a subtime of a SpikeData is correct.

        Tests:
        (Test Case 1) Checks that the subtime has the correct length and that all spikes are within the expected window.
        """
        self.assertEqual(len(sd.train), len(sdsub.train))
        self.assertEqual(sdsub.length, tmax - tmin)
        for n, nsub in zip(sd.train, sdsub.train):
            self.assertAll(nsub <= tmax - tmin, msg=msg)
            if tmin > 0:
                self.assertAll(nsub > 0, msg=msg)
                n_in_range = np.sum((n > tmin) & (n <= tmax))
            else:
                self.assertAll(nsub >= 0, msg=msg)
                n_in_range = np.sum(n <= tmax)
            self.assertTrue(len(nsub) == n_in_range, msg=msg)

    def assertAll(self, bools, msg=None):
        """
        Asserts that all elements in a boolean array are True.

        Tests:
        (Test Case 1) Checks that all elements in the boolean array are True.
        """
        self.assertTrue(np.all(bools), msg=msg)

    def assertClose(self, a, b, msg=None, **kw):
        """
        Asserts that two arrays are equal within tolerance.

        Tests:
        (Test Case 1) Checks that the two arrays are equal within tolerance.
        """
        self.assertTrue(np.allclose(a, b, **kw), msg=msg)

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

        # Calculate expected output size based on our binning logic
        expected_bins = len(counts)
        if sd.length % 1 == 0:
            # If length is exactly divisible by bin_size, expect an extra bin
            expected_bins += 1

        # Test 1: Check that the output has the expected number of bins
        self.assertEqual(
            len(binned_result),
            expected_bins,
            f"Expected {expected_bins} bins but got {len(binned_result)}",
        )

        # Test 2: Check that the counts in each bin match our expectations
        self.assertAll(
            binned_result[: len(counts)] == counts,
            f"Binned values don't match input counts",
        )

        # Test 3: If there's an extra bin, it should be empty (0)
        if expected_bins > len(counts):
            self.assertEqual(
                binned_result[-1],
                0,
                f"Expected empty extra bin but got {binned_result[-1]}",
            )

    @unittest.skipIf(SpikeTrain is None, "neo or quantities not installed")
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
        self.assertSpikeDataEqual(sd, sdneo)

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
        (Test Case 15) Tests consistency between subtime() and frames().
        (Test Case 16) Tests overlap parameter in frames().
        """
        times = np.random.rand(100) * 100
        idces = np.random.randint(5, size=100)

        # Test two-argument constructor and spike time list.
        sd = SpikeData.from_idces_times(idces, times, length=100.0)
        self.assertAll(np.sort(times) == list(sd.times))

        # Test event-list constructor.
        sd1 = SpikeData.from_events(list(zip(idces, times)))
        self.assertSpikeDataEqual(sd, sd1)

        # Test base constructor.
        sd2 = SpikeData(sd.train)
        self.assertSpikeDataEqual(sd, sd2)

        # Test events.
        sd4 = SpikeData.from_events(sd.events)
        self.assertSpikeDataEqual(sd, sd4)

        # Test idces_times().
        sd5 = SpikeData.from_idces_times(*sd.idces_times())
        self.assertSpikeDataEqual(sd, sd5)

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
        self.assertAll(r_subset == r2_subset)

        # Make sure the raster constructor handles multiple spikes in the same bin.
        tinysd = SpikeData.from_raster(np.array([[0, 3, 0]]), 20)
        self.assertAll(tinysd.train[0] == [25.0, 30.0, 35.0])

        # Test subset() constructor.
        idces = [1, 2, 3]
        sdsub = sd.subset(idces)
        for i, j in enumerate(idces):
            self.assertAll(sdsub.train[i] == sd.train[j])

        # Test subset() with a single unit.
        sdsub = sd.subset(1)
        self.assertEqual(sdsub.N, 1)

        # Test subtime() constructor idempotence.
        sdtimefull = sd.subtime(0, 100)
        self.assertSpikeDataEqual(sd, sdtimefull)

        # Test subtime() constructor actually grabs subsets.
        sdtime = sd.subtime(20, 50)
        self.assertSpikeDataSubtime(sd, sdtime, 20, 50)

        # Test subtime() with negative arguments.
        sdtime = sd.subtime(-80, -50)
        self.assertSpikeDataSubtime(sd, sdtime, 20, 50)

        # Check subtime() with ... first argument.
        sdtime = sd.subtime(..., 50)
        self.assertSpikeDataSubtime(sd, sdtime, 0, 50)

        # Check subtime() with ... second argument.
        sdtime = sd.subtime(20, ...)
        self.assertSpikeDataSubtime(sd, sdtime, 20, 100)

        # Check subtime() with second argument greater than length.
        sdtime = sd.subtime(20, 150)
        self.assertSpikeDataSubtime(sd, sdtime, 20, 100)

        # Test consistency between subtime() and frames().
        for i, frame in enumerate(sd.frames(20)):
            self.assertSpikeDataEqual(frame, sd.subtime(i * 20, (i + 1) * 20))

        # Regression test for overlap parameter of frames().
        for i, frame in enumerate(sd.frames(20, overlap=10)):
            self.assertSpikeDataEqual(frame, sd.subtime(i * 10, i * 10 + 20))

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
        self.assertEqual(sd.raster().sum(), N)
        self.assertAll(sd.sparse_raster() == sd.raster())

        # Make sure the length of the raster is consistent regardless of spike counts
        N = 10
        length = 1e4
        sdA = SpikeData.from_idces_times(
            np.zeros(N, int), np.random.rand(N) * length, length=length
        )
        sdB = SpikeData.from_idces_times(
            np.zeros(N, int), np.random.rand(N) * length, length=length
        )
        self.assertEqual(sdA.raster().shape, sdB.raster().shape)

        # Test binning rules with specific spike times
        # With bin_size=10:
        # - spike at t=0 goes to bin 0 (floor(0/10) = 0)
        # - spike at t=20 goes to bin 2 (floor(20/10) = 2)
        # - spike at t=40 goes to bin 4 (floor(40/10) = 4)
        # - Since length=40 is divisible by bin_size=10, we add an extra bin
        sd = SpikeData([[0, 20, 40]])
        self.assertEqual(sd.length, 40)

        # With our new binning logic, this should create 5 bins
        ground_truth = [[1, 0, 1, 0, 1]]
        actual_raster = sd.raster(10)

        # Verify raster shape and values
        self.assertEqual(actual_raster.shape, (1, 5))
        self.assertAll(actual_raster == ground_truth)

        # Also verify that binning rules are consistent with binned() method
        binned = np.array([list(sd.binned(10))])
        self.assertAll(sd.raster(10) == binned)

    def test_rates(self):
        """
        Tests rates() method for correct spike rate calculation and unit handling.

        Tests:
        (Test Case 1) Tests that rates() returns correct spike counts for each train.
        (Test Case 2) Tests conversion to Hz and error on invalid unit.
        """
        counts = np.random.poisson(100, size=50)
        sd = SpikeData([np.random.rand(n) for n in counts], length=1)
        self.assertAll(sd.rates() == counts)

        # Test the other possible units of rates.
        self.assertAll(sd.rates("Hz") == counts * 1000)
        self.assertRaises(ValueError, lambda: sd.rates("bad_unit"))

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
        self.assertTrue((ii[0] == 1).all())
        self.assertEqual(len(ii[0]), N - 1)
        self.assertEqual(len(ii), 1)

        # Also make sure multiple spike trains do the same thing.
        ii = SpikeData.from_idces_times(ar % 10, ar).interspike_intervals()
        self.assertEqual(len(ii), 10)
        for i in ii:
            self.assertTrue((i == 10).all())
            self.assertEqual(len(i), N / 10 - 1)

        # Finally, check with random ISIs.
        truth = np.random.rand(N)
        spikes = SpikeData.from_idces_times(np.zeros(N, int), truth.cumsum())
        ii = spikes.interspike_intervals()
        self.assertClose(ii[0], truth[1:])

    def test_spike_time_tiling_ta(self):
        """
        Tests the _sttc_ta helper for correct calculation of total available time.

        Tests:
        (Test Cases) Tests trivial and edge cases for spike overlap and time window.
        """
        self.assertEqual(spikedata._sttc_ta([42], 1, 100), 2)
        self.assertEqual(spikedata._sttc_ta([], 1, 100), 0)

        # When spikes don't overlap, you should get exactly 2ndt.
        self.assertEqual(spikedata._sttc_ta(np.arange(42) + 1, 0.5, 100), 42.0)

        # When spikes overlap fully, you should get exactly (tmax-tmin) + 2dt
        self.assertEqual(spikedata._sttc_ta(np.arange(42) + 100, 100, 300), 241)

    def test_spike_time_tiling_na(self):
        """
        Tests the _sttc_na helper for correct calculation of number of spikes in window.

        Tests:
        (Test Cases) Tests base cases, interval inclusion, and multiple spike coverage.
        """
        self.assertEqual(spikedata._sttc_na([1, 2, 3], [], 1), 0)
        self.assertEqual(spikedata._sttc_na([], [1, 2, 3], 1), 0)

        self.assertEqual(spikedata._sttc_na([1], [2], 0.5), 0)
        self.assertEqual(spikedata._sttc_na([1], [2], 1), 1)

        # Make sure closed intervals are being used.
        na = spikedata._sttc_na(np.arange(10), np.arange(10) + 0.5, 0.5)
        self.assertEqual(na, 10)

        # Skipping multiple spikes in spike train B.
        self.assertEqual(spikedata._sttc_na([4], [1, 2, 3, 4.5], 0.1), 0)
        self.assertEqual(spikedata._sttc_na([4], [1, 2, 3, 4.5], 0.5), 1)

        # Many spikes in train B covering a single one in A.
        self.assertEqual(spikedata._sttc_na([2], [1, 2, 3], 0.1), 1)
        self.assertEqual(spikedata._sttc_na([2], [1, 2, 3], 1), 1)

        # Many spikes in train A are covered by one in B.
        self.assertEqual(spikedata._sttc_na([1, 2, 3], [2], 0.1), 1)
        self.assertEqual(spikedata._sttc_na([1, 2, 3], [2], 1), 3)

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
        self.assertEqual(foo.spike_time_tiling(0, 0, 1), 1.0)
        self.assertEqual(foo.spike_time_tiling(1, 1, 1), 1.0)
        self.assertEqual(
            foo.spike_time_tiling(0, 1, 1),
            foo.spike_time_tiling(1, 0, 1),
        )

        # Exactly the same thing, but for the matrix of STTCs.
        sttc = foo.spike_time_tilings(1)
        self.assertEqual(sttc.shape, (2, 2))
        self.assertEqual(sttc[0, 1], sttc[1, 0])
        self.assertEqual(sttc[0, 0], 1.0)
        self.assertEqual(sttc[1, 1], 1.0)
        self.assertEqual(sttc[0, 1], foo.spike_time_tiling(0, 1, 1))

        # Default arguments, inferred value of tmax.
        tmax = max(np.ptp(foo.train[0]), np.ptp(foo.train[1]))
        self.assertEqual(
            foo.spike_time_tiling(0, 1),
            foo.spike_time_tiling(0, 1, tmax),
        )

        # The uncorrelated spike trains above should stay near zero.
        # I'm not sure how many significant figures to expect with the
        # randomness, though, so it's really easy to pass.
        self.assertAlmostEqual(foo.spike_time_tiling(0, 1, 1), 0, 1)

        # Two spike trains that are in complete disagreement. This
        # should be exactly -0.8, but there's systematic error
        # proportional to 1/N, even in their original implementation.
        bar = SpikeData([np.arange(N) + 0.0, np.arange(N) + 0.5])
        self.assertAlmostEqual(bar.spike_time_tiling(0, 1, 0.4), -0.8, int(np.log10(N)))

        # As you vary dt, that alternating spike train actually gets
        # the STTC to go continuously from 0 to approach a limit of
        # lim(dt to 0.5) STTC(dt) = -1, but STTC(dt >= 0.5) = 0.
        self.assertEqual(bar.spike_time_tiling(0, 1, 0.5), 0)

        # Make sure it stays within range even for spike trains with
        # completely random lengths.
        for _ in range(100):
            baz = SpikeData([np.random.rand(np.random.poisson(100)) for _ in range(2)])
            sttc = baz.spike_time_tiling(0, 1, np.random.lognormal())
            self.assertLessEqual(sttc, 1)
            self.assertGreaterEqual(sttc, -1)

        # STTC of an empty spike train should definitely be 0!
        fish = SpikeData([[], np.random.rand(100)])
        sttc = fish.spike_time_tiling(0, 1, 0.01)
        self.assertEqual(sttc, 0)

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
        self.assertEqual(sum(spikes.binned(5)), N)

    def test_binning(self):
        """
        Tests binned() method for correct bin assignment.

        Tests:
        (Test Case 1) Tests that binning with size 4 produces the expected counts.
        """
        spikes = SpikeData([[1, 2, 5, 15, 16, 20, 22, 25]])
        self.assertListEqual(list(spikes.binned(4)), [2, 1, 0, 1, 1, 2, 1])

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
        with self.assertRaises(ValueError):
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
        self.assertDictEqual(foo.metadata, bar.metadata)
        self.assertNeuronAttributesEqual(truth, bar.neuron_attributes)

        # Change the metadata of foo and see that it's copied, so the
        # change doesn't propagate.
        foo.metadata["name"] = "Ford"
        baz = bar.subtime(500, 1000)
        self.assertDictEqual(bar.metadata, baz.metadata)
        self.assertIsNot(bar.metadata, baz.metadata)
        self.assertNotEqual(foo.metadata["name"], bar.metadata["name"])
        self.assertNeuronAttributesEqual(bar.neuron_attributes, baz.neuron_attributes)

    def assertNeuronAttributesEqual(self, nda, ndb, msg=None):
        """Assert that two lists of neuron attributes are equal elementwise."""
        self.assertEqual(len(nda), len(ndb))
        for n, m in zip(nda, ndb):
            self.assertEqual(n, m)

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
        self.assertRaises(
            ValueError, lambda: SpikeData([], N=5, length=100, raw_data=[])
        )
        self.assertRaises(
            ValueError, lambda: SpikeData([], N=5, length=100, raw_time=42)
        )

        # Make sure inconsistent lengths throw an error as well.
        self.assertRaises(
            ValueError,
            lambda: SpikeData(
                [], N=5, length=100, raw_data=np.zeros((5, 100)), raw_time=np.arange(42)
            ),
        )

        # Check automatic generation of the time array.
        sd = SpikeData(
            [], N=5, length=100, raw_data=np.random.rand(5, 100), raw_time=1.0
        )
        self.assertAll(sd.raw_time == np.arange(100))

        # Make sure the raw data is sliced properly with time.
        sd2 = sd.subtime(20, 30)
        self.assertAll(sd2.raw_time == np.arange(11))
        self.assertAll(sd2.raw_data == sd.raw_data[:, 20:31])

    def test_isi_rate(self):
        """
        Tests resampled_isi and _resampled_isi for correct ISI-based rate calculation.

        Tests:
        (Test Case 1) Tests that a constant-rate neuron yields the correct rate at all times.
        (Test Case 2) Tests correct rates for varying spike intervals.
        """
        # For a neuron that fires at a constant rate, any sample time should
        # give you exactly the correct rate, here 1 kHz.
        spikes = np.arange(10)
        when = np.random.rand(1000) * 12 - 1
        self.assertAll(spikedata._resampled_isi(spikes, when, sigma_ms=0.0) == 1)

        # Also check that the rate is correctly calculated for some varying
        # examples.
        sd = SpikeData([[0, 1 / k, 10 + 1 / k] for k in np.arange(1, 100)])
        self.assertAll(sd.resampled_isi(0).round(2) == np.arange(1, 100))
        self.assertAll(sd.resampled_isi(10).round(2) == 0.1)

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
        self.assertAlmostEqual(a.latencies(b)[0][0], 0.2)
        self.assertAlmostEqual(a.latencies(b)[0][-1], 0.2)

        # Small enough window, should be no latencies.
        self.assertEqual(a.latencies(b, 0.1)[0], [])

        # Can do negative
        self.assertAlmostEqual(a.latencies([0.1])[0][0], -0.1)

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

        self.assertEqual(rnd.shape, raster.shape)
        uniq = np.unique(rnd)
        self.assertTrue(set(uniq.tolist()).issubset({0.0, 1.0}))
        self.assertTrue(np.allclose(rnd.sum(axis=1), row_sum))
        self.assertTrue(np.allclose(rnd.sum(axis=0), col_sum))
        self.assertTrue(np.isclose(rnd.sum(), total))

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
        t_spk_mat = np.zeros(
            (T + 1, N)
        )  # T+1 because get_pop_rate adds a bin when len%bin_size==0
        t_spk_mat[trains[0], 0] = 1
        t_spk_mat[trains[1], 1] = 1
        t_spk_mat[trains[2], 2] = 1

        sd = SpikeData(trains, length=T)

        SQUARE_WIDTH = 5
        GAUSS_SIGMA = 0

        pop = sd.get_pop_rate(
            square_width=SQUARE_WIDTH, gauss_sigma=GAUSS_SIGMA, raster_bin_size_ms=1.0
        )
        truth = np.convolve(
            np.sum(t_spk_mat, axis=1), np.ones(SQUARE_WIDTH) / SQUARE_WIDTH, mode="same"
        )

        self.assertTrue(np.allclose(pop, truth))

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

        self.assertTrue(np.isclose(pop.sum(), 1.0, rtol=1e-3, atol=1e-3))
        self.assertTrue(np.isclose(pop[50 - 1], pop[50 + 1]))

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
        self.assertEqual(len(tburst), 2)
        self.assertEqual(len(peak_amp), 2)
        self.assertEqual(edges.shape, (2, 2))

        # First burst should be around t=50
        self.assertTrue(48 <= tburst[0] <= 52)
        # Second burst should be around t=150
        self.assertTrue(148 <= tburst[1] <= 152)

        # Check that edges bracket the peaks
        self.assertTrue(edges[0, 0] < tburst[0] < edges[0, 1])
        self.assertTrue(edges[1, 0] < tburst[1] < edges[1, 1])

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
        # Unit 0 fires at times 1, 3, 4, 7 (active in burst 1 only)
        # Unit 1 fires at times 2, 4, 6, 9 (active in both bursts)
        # Unit 2 fires at times 3, 6, 8 (active in burst 2 only)
        spike_trains = [
            np.array([1, 3, 4, 7]),  # Unit 0
            np.array([2, 4, 6, 9]),  # Unit 1
            np.array([3, 6, 8]),  # Unit 2
        ]

        # Create SpikeData object
        sd = SpikeData(spike_trains)

        # Define burst edges
        edges = np.array(
            [
                [1, 4],  # First burst from t=1 to t=4
                [6, 9],  # Second burst from t=6 to t=9
            ]
        )

        # Set parameters
        min_spikes = 2
        backbone_threshold = 0.55

        # Call the method
        frac_per_unit, frac_per_burst, backbone_units = sd.get_frac_active(
            edges, min_spikes, backbone_threshold
        )

        # Expected values based on our spike pattern:
        # Unit 0: active in 1/2 bursts (has 3 spikes in burst 1, 1 in burst 2)
        # Unit 1: active in 2/2 bursts (has 2 spikes in burst 1, 2 in burst 2)
        # Unit 2: active in 1/2 bursts (has 1 spike in burst 1, 2 in burst 2)
        # Burst 1: 2/3 units active (units 0 and 1)
        # Burst 2: 2/3 units active (units 1 and 2)
        # Only unit 1 is a backbone unit (>= 0.55 participation rate)

        expected_frac_per_unit = np.array([0.5, 1.0, 0.5])
        expected_frac_per_burst = np.array([2 / 3, 2 / 3])
        expected_backbone_units = np.array([1])

        # Verify results
        self.assertClose(frac_per_unit, expected_frac_per_unit)
        self.assertClose(frac_per_burst, expected_frac_per_burst)
        self.assertTrue(np.array_equal(backbone_units, expected_backbone_units))

        # Test with different parameters
        # Increase min_spikes to require 3 spikes per burst
        min_spikes_high = 3
        frac_per_unit_high, frac_per_burst_high, backbone_high = sd.get_frac_active(
            edges, min_spikes_high, backbone_threshold
        )

        # Now only unit 0 is active in burst 1, no units active in burst 2
        expected_high_unit = np.array([0.5, 0.0, 0.0])
        expected_high_burst = np.array([1 / 3, 0.0])
        expected_high_backbone = np.array([])

        self.assertClose(frac_per_unit_high, expected_high_unit)
        self.assertClose(frac_per_burst_high, expected_high_burst)
        self.assertTrue(np.array_equal(backbone_high, expected_high_backbone))

        # Test with lower backbone threshold
        low_threshold = 0.4
        _, _, backbone_low = sd.get_frac_active(edges, min_spikes, low_threshold)

        # With lower threshold, all units should be backbone
        expected_low_backbone = np.array([0, 1, 2])

        self.assertTrue(np.array_equal(backbone_low, expected_low_backbone))

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
        from dataclasses import dataclass

        @dataclass
        class NeuronAttrs:
            channel: int
            other_field: str = "test"

        # Test basic functionality
        attrs = [NeuronAttrs(channel=i % 4) for i in range(10)]
        trains = [[] for _ in range(10)]
        sd = SpikeData(trains, neuron_attributes=attrs, length=100.0)
        mapping = sd.neuron_to_channel_map()

        # Should have all 10 neurons mapped
        self.assertEqual(len(mapping), 10)
        # Check a few mappings
        self.assertEqual(mapping[0], 0)
        self.assertEqual(mapping[1], 1)
        self.assertEqual(mapping[4], 0)  # 4 % 4 = 0
        self.assertEqual(mapping[5], 1)  # 5 % 4 = 1

        # Test with different attribute names
        @dataclass
        class NeuronAttrsChannelId:
            channel_id: int

        attrs2 = [NeuronAttrsChannelId(channel_id=i % 3) for i in range(6)]
        sd2 = SpikeData([[]] * 6, neuron_attributes=attrs2, length=100.0)
        mapping2 = sd2.neuron_to_channel_map()
        self.assertEqual(len(mapping2), 6)
        self.assertEqual(mapping2[0], 0)
        self.assertEqual(mapping2[3], 0)  # 3 % 3 = 0

        # Test explicit channel_attr parameter
        mapping2_explicit = sd2.neuron_to_channel_map(channel_attr="channel_id")
        self.assertEqual(mapping2, mapping2_explicit)

        # Test with channel_index attribute
        @dataclass
        class NeuronAttrsChannelIndex:
            channel_index: int

        attrs3 = [NeuronAttrsChannelIndex(channel_index=i // 2) for i in range(6)]
        sd3 = SpikeData([[]] * 6, neuron_attributes=attrs3, length=100.0)
        mapping3 = sd3.neuron_to_channel_map()
        self.assertEqual(mapping3[0], 0)
        self.assertEqual(mapping3[1], 0)
        self.assertEqual(mapping3[2], 1)
        self.assertEqual(mapping3[3], 1)

        # Test edge case: no neuron_attributes
        sd_no_attrs = SpikeData([[]] * 5, length=100.0)
        mapping_no_attrs = sd_no_attrs.neuron_to_channel_map()
        self.assertEqual(mapping_no_attrs, {})

        # Test edge case: empty data (N=0)
        sd_empty = SpikeData([], neuron_attributes=[], length=100.0)
        mapping_empty = sd_empty.neuron_to_channel_map()
        self.assertEqual(mapping_empty, {})

        # Test with partial channel information (some neurons missing channel)
        @dataclass
        class NeuronAttrsPartial:
            channel: int = None

        attrs_partial = [
            NeuronAttrsPartial(channel=0),
            NeuronAttrsPartial(channel=1),
            NeuronAttrsPartial(),  # Missing channel
            NeuronAttrsPartial(channel=2),
        ]
        sd_partial = SpikeData([[]] * 4, neuron_attributes=attrs_partial, length=100.0)
        mapping_partial = sd_partial.neuron_to_channel_map()
        # Should only include neurons 0, 1, 3 (neuron 2 has no channel)
        self.assertEqual(len(mapping_partial), 3)
        self.assertEqual(mapping_partial[0], 0)
        self.assertEqual(mapping_partial[1], 1)
        self.assertEqual(mapping_partial[3], 2)
        self.assertNotIn(2, mapping_partial)

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
        from dataclasses import dataclass

        @dataclass
        class NeuronAttrs:
            channel: int

        # Create 6 neurons: 0,1 on channel 0; 2,3 on channel 1; 4,5 on channel 2
        attrs = [NeuronAttrs(channel=i // 2) for i in range(6)]
        # Create spike trains with known patterns:
        # Channel 0: neuron 0 has spikes at 10, 20; neuron 1 has spike at 15
        # Channel 1: neuron 2 has spike at 25; neuron 3 has spike at 30
        # Channel 2: neuron 4 has spike at 35; neuron 5 has spike at 40
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

        # Should have 3 channels
        self.assertEqual(ch_raster.shape[0], 3)
        # Should have 6 bins (0-10, 10-20, 20-30, 30-40, 40-50, plus extra bin)
        expected_bins = int(np.ceil(50.0 / 10.0))
        if 50.0 % 10.0 == 0:
            expected_bins += 1
        self.assertEqual(ch_raster.shape[1], expected_bins)

        # Channel 0 should have 3 spikes total (2 from neuron 0, 1 from neuron 1)
        self.assertEqual(ch_raster[0, :].sum(), 3)
        # Channel 1 should have 2 spikes total
        self.assertEqual(ch_raster[1, :].sum(), 2)
        # Channel 2 should have 2 spikes total
        self.assertEqual(ch_raster[2, :].sum(), 2)

        # Verify specific bins for channel 0
        # Bin 1 (10-20): should have 2 spikes (neuron 0 at 10, neuron 1 at 15)
        # Bin 2 (20-30): should have 1 spike (neuron 0 at 20)
        self.assertEqual(ch_raster[0, 1], 2)  # bin 1 (10-20)
        self.assertEqual(ch_raster[0, 2], 1)  # bin 2 (20-30)

        # Verify total spike count matches neuron raster
        neuron_raster = sd.raster(bin_size=10.0)
        self.assertEqual(ch_raster.sum(), neuron_raster.sum())

        # Test with different bin_size
        ch_raster_small = sd.channel_raster(bin_size=5.0)
        self.assertEqual(ch_raster_small.shape[0], 3)
        # Total spikes should still match
        self.assertEqual(ch_raster_small.sum(), neuron_raster.sum())

        # Test with explicit channel_attr
        ch_raster_explicit = sd.channel_raster(bin_size=10.0, channel_attr="channel")
        self.assertAll(ch_raster == ch_raster_explicit)

        # Test with different attribute name
        @dataclass
        class NeuronAttrsChannelId:
            channel_id: int

        attrs2 = [NeuronAttrsChannelId(channel_id=i % 2) for i in range(4)]
        trains2 = [[10.0], [20.0], [30.0], [40.0]]
        sd2 = SpikeData(trains2, neuron_attributes=attrs2, length=50.0)
        ch_raster2 = sd2.channel_raster(bin_size=10.0, channel_attr="channel_id")
        self.assertEqual(ch_raster2.shape[0], 2)  # 2 channels
        # Channel 0: neurons 0, 2 (spikes at 10, 30)
        # Channel 1: neurons 1, 3 (spikes at 20, 40)
        self.assertEqual(ch_raster2[0, :].sum(), 2)
        self.assertEqual(ch_raster2[1, :].sum(), 2)

        # Test edge case: no channel information
        sd_no_channel = SpikeData([[]] * 3, length=100.0)
        with self.assertRaises(ValueError):
            sd_no_channel.channel_raster()

        # Test that multiple neurons on same channel aggregate correctly
        @dataclass
        class NeuronAttrsSameChannel:
            channel: int

        # All neurons on channel 0
        attrs_same = [NeuronAttrsSameChannel(channel=0) for _ in range(3)]
        trains_same = [[10.0, 20.0], [15.0], [25.0]]
        sd_same = SpikeData(trains_same, neuron_attributes=attrs_same, length=30.0)
        ch_raster_same = sd_same.channel_raster(bin_size=10.0)
        self.assertEqual(ch_raster_same.shape[0], 1)  # Only 1 channel
        self.assertEqual(ch_raster_same[0, :].sum(), 4)  # Total 4 spikes

        # Test with non-contiguous channel indices
        @dataclass
        class NeuronAttrsNonContiguous:
            channel: int

        attrs_nc = [
            NeuronAttrsNonContiguous(channel=0),
            NeuronAttrsNonContiguous(channel=5),
            NeuronAttrsNonContiguous(channel=10),
        ]
        trains_nc = [[10.0], [20.0], [30.0]]
        sd_nc = SpikeData(trains_nc, neuron_attributes=attrs_nc, length=40.0)
        ch_raster_nc = sd_nc.channel_raster(bin_size=10.0)
        # Should have 3 channels (0, 5, 10)
        self.assertEqual(ch_raster_nc.shape[0], 3)
        # Each channel should have 1 spike
        self.assertEqual(ch_raster_nc[0, :].sum(), 1)
        self.assertEqual(ch_raster_nc[1, :].sum(), 1)
        self.assertEqual(ch_raster_nc[2, :].sum(), 1)
