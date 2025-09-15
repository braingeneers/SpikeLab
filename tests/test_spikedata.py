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
    "Generate a SpikeData whose raster matches given counts."
    times = np.hstack([i * np.ones(c) for i, c in enumerate(counts)])
    return SpikeData([times + 0.5], length=len(counts))


def random_spikedata(units, spikes, rate=1.0):
    """
    Generate SpikeData with a given number of units, total number of
    spikes, and overall mean firing rate.
    """
    idces = np.random.randint(units, size=spikes)
    times = np.random.rand(spikes) * spikes / rate / units
    return SpikeData.from_idces_times(
        idces, times, length=spikes / rate / units, N=units
    )


class SpikeDataTest(unittest.TestCase):
    def assertSpikeDataEqual(self, sda, sdb, msg=None):
        "Assert that two SpikeData objects contain the same data."
        for a, b in zip(sda.train, sdb.train):
            self.assertTrue(len(a) == len(b) and np.allclose(a, b), msg=msg)

    def assertSpikeDataSubtime(self, sd, sdsub, tmin, tmax, msg=None):
        "Assert that a subtime of a SpikeData is correct."
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
        "Assert that two arrays are equal elementwise."
        self.assertTrue(np.all(bools), msg=msg)

    def assertClose(self, a, b, msg=None, **kw):
        "Assert that two arrays are equal within tolerance."
        self.assertTrue(np.allclose(a, b, **kw), msg=msg)

    def test_sd_from_counts(self):
        # Just double-check that this helper method works...
        counts = np.random.randint(10, size=1000)
        sd = sd_from_counts(counts)
        self.assertAll(sd.binned(1) == counts)

    @unittest.skipIf(SpikeTrain is None, "neo or quantities not installed")
    def test_neo_conversion(self):
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
        # Generate a bunch of random spike times and indices.
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
        r2 = SpikeData.from_raster(r, bin_size).raster(bin_size)
        self.assertAll(r == r2)

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
        # Generate Poisson spike trains and make sure no spikes are
        # lost in translation.
        N = 10000
        sd = random_spikedata(10, N)

        # Try both a sparse and a dense raster.
        self.assertEqual(sd.raster().sum(), N)
        self.assertAll(sd.sparse_raster() == sd.raster())

        # Make sure the length of the raster is going to be consistent
        # no matter how many spikes there are.
        N = 10
        length = 1e4
        sdA = SpikeData.from_idces_times(
            np.zeros(N, int), np.random.rand(N) * length, length=length
        )
        sdB = SpikeData.from_idces_times(
            np.zeros(N, int), np.random.rand(N) * length, length=length
        )
        self.assertEqual(sdA.raster().shape, sdB.raster().shape)

        # Corner cases of raster binning rules: spikes exactly at
        # 0 end up in the first bin, but other bins should be
        # lower-open and upper-closed.
        ground_truth = [[1, 1, 0, 1]]
        sd = SpikeData([[0, 20, 40]])
        self.assertEqual(sd.length, 40)
        self.assertAll(sd.raster(10) == ground_truth)

        # Also verify that binning rules are consistent between the
        # raster and other binning methods.
        binned = np.array([list(sd.binned(10))])
        self.assertAll(sd.raster(10) == binned)

    def test_rates(self):
        # Generate random spike trains of varying lengths and
        # therefore rates to calculate.
        counts = np.random.poisson(100, size=50)
        sd = SpikeData([np.random.rand(n) for n in counts], length=1)
        self.assertAll(sd.rates() == counts)

        # Test the other possible units of rates.
        self.assertAll(sd.rates("Hz") == counts * 1000)
        self.assertRaises(ValueError, lambda: sd.rates("bad_unit"))

    # Removed tests for deprecated utilities: pearson, burstiness_index

    def test_interspike_intervals(self):
        # Uniform spike train: uniform ISIs. Also make sure it returns
        # a list of just the one array.
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

    # Removed tests for deprecated utilities: fano_factors

    def test_spike_time_tiling_ta(self):
        # Trivial base cases.
        self.assertEqual(spikedata._sttc_ta([42], 1, 100), 2)
        self.assertEqual(spikedata._sttc_ta([], 1, 100), 0)

        # When spikes don't overlap, you should get exactly 2ndt.
        self.assertEqual(spikedata._sttc_ta(np.arange(42) + 1, 0.5, 100), 42.0)

        # When spikes overlap fully, you should get exactly (tmax-tmin) + 2dt
        self.assertEqual(spikedata._sttc_ta(np.arange(42) + 100, 100, 300), 241)

    def test_spike_time_tiling_na(self):
        # Trivial base cases.
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
        # Examples to use in different cases.
        N = 10000

        # Any spike train should be exactly equal to itself, and the
        # result shouldn't depend on which train is A and which is B.
        foo = random_spikedata(2, N)
        self.assertEqual(foo.spike_time_tiling(0, 0, 1), 1.0)
        self.assertEqual(foo.spike_time_tiling(1, 1, 1), 1.0)
        self.assertEqual(foo.spike_time_tiling(0, 1, 1), foo.spike_time_tiling(1, 0, 1))

        # Exactly the same thing, but for the matrix of STTCs.
        sttc = foo.spike_time_tilings(1)
        self.assertEqual(sttc.shape, (2, 2))
        self.assertEqual(sttc[0, 1], sttc[1, 0])
        self.assertEqual(sttc[0, 0], 1.0)
        self.assertEqual(sttc[1, 1], 1.0)
        self.assertEqual(sttc[0, 1], foo.spike_time_tiling(0, 1, 1))

        # Default arguments, inferred value of tmax.
        tmax = max(np.ptp(foo.train[0]), np.ptp(foo.train[1]))
        self.assertEqual(foo.spike_time_tiling(0, 1), foo.spike_time_tiling(0, 1, tmax))

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
        # Generate the times of a Poisson spike train, and ensure that
        # no spikes are lost in translation.
        N = 1000
        times = np.cumsum(stats.expon.rvs(size=N))
        spikes = SpikeData([times])
        self.assertEqual(sum(spikes.binned(5)), N)

    def test_binning(self):
        # Here's a totally arbitrary list of spike times to bin.
        spikes = SpikeData([[1, 2, 5, 15, 16, 20, 22, 25]])
        self.assertListEqual(list(spikes.binned(4)), [2, 1, 0, 2, 1, 1, 1])

    # Removed tests for deprecated avalanche/DCC utilities

    # Removed tests for deprecated DCC utilities

    def test_metadata(self):
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
        self.assertEqual(len(nda), len(ndb))
        for n, m in zip(nda, ndb):
            self.assertEqual(n, m)

    def test_raw_data(self):
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
        # Calculate the firing rate of a single neuron using the inverse ISI.

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
        T, N = 100, 3
        t_spk_mat = np.zeros((T, N))
        t_spk_mat[[10, 20, 50, 70, 80], 0] = 1
        t_spk_mat[[15, 20, 55, 70], 1] = 1
        t_spk_mat[[20, 25, 60], 2] = 1

        SQUARE_WIDTH = 5
        GAUSS_SIGMA = 0

        pop = spikedata.get_pop_rate(t_spk_mat, SQUARE_WIDTH, GAUSS_SIGMA)
        truth = np.convolve(
            np.sum(t_spk_mat, axis=1), np.ones(SQUARE_WIDTH) / SQUARE_WIDTH, mode="same"
        )
        self.assertTrue(np.allclose(pop, truth))

    def test_get_pop_rate_gaussian_only_impulse(self):
        T, N = 101, 1
        t_spk_mat = np.zeros((T, N))
        t_spk_mat[T // 2, 0] = 1

        SQUARE_WIDTH = 0
        GAUSS_SIGMA = 2

        pop = spikedata.get_pop_rate(t_spk_mat, SQUARE_WIDTH, GAUSS_SIGMA)

        self.assertTrue(np.isclose(pop.sum(), 1.0, rtol=1e-3, atol=1e-3))
        self.assertTrue(np.isclose(pop[T // 2 - 1], pop[T // 2 + 1]))

    def test_get_bursts_detects_simple_peaks(self):
        T = 200
        pop_rate = np.zeros(T)
        pop_rate[45:56] = np.array([0, 2, 4, 6, 8, 10, 8, 6, 4, 2, 0])
        pop_rate[145:156] = np.array([0, 3, 6, 9, 12, 15, 12, 9, 6, 3, 0])

        pop_rate_acc = []
        THR_BURST = 0.5
        MIN_BURST_DIFF = 10
        BURST_EDGE_MULT_THRESH = 0.2

        tburst, edges, peak_amp = spikedata.get_bursts(
            pop_rate, pop_rate_acc, THR_BURST, MIN_BURST_DIFF, BURST_EDGE_MULT_THRESH
        )

        self.assertEqual(len(tburst), 2)
        self.assertEqual(len(peak_amp), 2)
        self.assertEqual(edges.shape, (2, 2))
        self.assertTrue(48 <= tburst[0] <= 52)
        self.assertTrue(148 <= tburst[1] <= 152)
        self.assertTrue(edges[0, 0] < tburst[0] < edges[0, 1])
        self.assertTrue(edges[1, 0] < tburst[1] < edges[1, 1])

    # Removed tests for deprecated randomization and sampling utilities
