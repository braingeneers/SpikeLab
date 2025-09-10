import unittest
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
from scipy import sparse, stats

try:
    import quantities
    from neo.core import SpikeTrain
except ImportError:
    SpikeTrain = None
    quantities = None

# Import the module by path instead of going through the __init__ logic so we can access
# all the hidden internal methods.
import spikedata.spikedata as spikedata
from spikedata import (
    SpikeData,
    best_effort_sample,
    fano_factors,
    pearson,
    randomize_raster_greedy,
)

Neuron = namedtuple("Neuron", "spike_time fs")


class MockSpikeRecorder:
    def __init__(self, idces, times):
        self.events = dict(senders=idces, times=times)


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

        # Test 'list of Neuron()s' constructor.
        fs = 10
        ns = [Neuron(spike_time=ts * fs, fs=fs * 1e3) for ts in sd.train]
        sd3 = SpikeData.from_mbt_neurons(ns)
        self.assertSpikeDataEqual(sd, sd3)

        # Test events.
        sd4 = SpikeData.from_events(sd.events)
        self.assertSpikeDataEqual(sd, sd4)

        # Test idces_times().
        sd5 = SpikeData.from_idces_times(*sd.idces_times())
        self.assertSpikeDataEqual(sd, sd5)

        # Test 'NEST SpikeRecorder' constructor, passing in an arange to
        # take the place of the NodeCollection you would usually use.
        recorder = MockSpikeRecorder(idces + 1, times)
        sd6 = SpikeData.from_nest(recorder, 1 + np.arange(5))
        self.assertSpikeDataEqual(sd, sd6)

        # Make sure the NEST constructor can combine ranges, and adds nest_id to
        # existing neuron attributes without affecting the values already there.
        attrs = [MockNeuronAttributes(ξ) for ξ in np.random.rand(5)]
        sd7 = SpikeData.from_nest(
            recorder, 1 + np.arange(3), 4 + np.arange(2), neuron_attributes=attrs
        )
        self.assertSpikeDataEqual(sd, sd7)
        assert sd7.neuron_attributes is not None
        for i, (attr, neuron_attrs) in enumerate(zip(attrs, sd7.neuron_attributes)):
            self.assertEqual(attr.size, neuron_attrs.size)
            self.assertEqual(neuron_attrs.nest_id, i + 1)

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

        # Make sure you can subset by neuron_data, not just raw index, and that order
        # does not matter for this operation.
        sdsub = sd6.subset([1, 2])
        sdsub2 = sd6.subset([3, 2], by="nest_id")
        self.assertSpikeDataEqual(sdsub, sdsub2)

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

    def test_pearson(self):
        # These four cells are constructed so that A is perfectly
        # correlated with B, perfectly anticorrelated with C, and
        # uncorrelated with D.
        cellA = [1, 0, 1, 1, 0, 0]
        cellB = cellA
        cellC = [0, 1, 0, 0, 1, 1]
        cellD = [1, 1, 1, 0, 0, 1]

        # Construct the true raster, use it to produce times and
        # indices using numpy methods, and ensure that the sparse
        # matrix generated is correct.
        ground_truth = np.stack((cellA, cellB, cellC, cellD))
        times, idces = np.where(ground_truth.T)
        sd = SpikeData.from_idces_times(idces, times + 0.5)
        raster = sd.sparse_raster(bin_size=1)
        self.assertAll(raster == ground_truth)

        # Finally, check the calculated Pearson coefficients to ensure
        # they're numerically close enough to the intended values.
        true_pearson = [[1, 1, -1, 0], [1, 1, -1, 0], [-1, -1, 1, 0], [0, 0, 0, 1]]
        sparse_pearson = pearson(raster)
        self.assertClose(sparse_pearson, true_pearson)

        # Test on dense matrices (fallback to np.pearson).
        dense_pearson = pearson(raster.todense())
        np_pearson = np.corrcoef(raster.todense())
        self.assertClose(dense_pearson, np_pearson)

        # Also check the calculations.
        self.assertEqual(dense_pearson.shape, sparse_pearson.shape)
        self.assertClose(dense_pearson, sparse_pearson)

    def test_burstiness_index(self):
        # Something completely uniform should have zero burstiness,
        # but ensure there's no spike at time zero.
        uniform = SpikeData([0.5 + np.arange(1000)])
        self.assertEqual(uniform.burstiness_index(10), 0)

        # All spikes at the same time is technically super bursty,
        # just make sure that they happen late enough that there are
        # actually several bins to count.
        atonce = SpikeData([[1]] * 1000)
        self.assertEqual(atonce.burstiness_index(0.01), 1)

        # Added code to deal with a corner case so it's really ALWAYS
        # in the zero to one range. I think this only happens with
        # very small values.
        self.assertEqual(SpikeData([[1]]).burstiness_index(), 1)

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

    def test_fano_factors(self):
        N = 10000

        # If there's no variance, Fano factors should be zero, for
        # both sparse and dense implementations. Also use todense()
        # next to  toarray() to show that both np.matrix and np.array
        # spike rasters are acceptable. Note that the numerical issues
        # in the sparse version mean that it's not precisely zero, so
        # we use assertAlmostEqual() in this case.
        ones = sparse.csr_matrix(np.ones(N))
        self.assertAlmostEqual(fano_factors(ones)[0], 0)
        self.assertEqual(fano_factors(ones.todense())[0], 0)
        self.assertEqual(fano_factors(ones.toarray())[0], 0)

        # Poisson spike trains should have Fano factors about 1.
        # This is only rough because random, but the sparse and dense
        # versions should both be equal to each other.
        foo = random_spikedata(1, N).sparse_raster(1)
        f_sparse = fano_factors(foo)[0]
        f_dense = fano_factors(foo.toarray())[0]
        self.assertAlmostEqual(f_sparse, 1, 1)
        self.assertAlmostEqual(f_dense, 1, 1)
        self.assertAlmostEqual(f_sparse, f_dense)

        # Make sure the sparse and dense are equal when there are
        # multiple spike trains as well.
        foo = random_spikedata(10, N).sparse_raster(10)
        f_sparse = fano_factors(foo)
        f_dense = fano_factors(foo.toarray())
        self.assertClose(f_sparse, f_dense)

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

    def test_avalanches(self):
        # The simple case where there are avalanches in the middle.
        sd = sd_from_counts([1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0])
        self.assertListEqual([len(av) for av in sd.avalanches(1, bin_size=1)], [5, 3])
        self.assertListEqual(
            [sum(av) for av in sd.avalanches(1, bin_size=1)],
            [2 + 3 + 4 + 3 + 2, 2 + 3 + 2],
        )

        # Also the duration-size lists of the same data.
        durations, sizes = sd.avalanche_duration_size(1, bin_size=1)
        self.assertListEqual(list(durations), [5, 3])
        self.assertListEqual(list(sizes), [2 + 3 + 4 + 3 + 2, 2 + 3 + 2])

        # Ensure that avalanches coinciding with the start and end of
        # recording don't get counted because there's no way to know
        # how long they are.
        sd = sd_from_counts([2, 5, 3, 0, 1, 0, 0, 2, 2, 0, 0, 0, 0, 4, 3, 4, 0, 42])
        self.assertListEqual([len(av) for av in sd.avalanches(1, bin_size=1)], [2, 3])

        # Corner cases where there are no avalanches: no transitions
        # because threshold too low, because threshold too high,
        # because only crosses once downwards, and because only
        # crosses once upwards.
        sd = sd_from_counts([1, 2, 3, 4, 5])
        self.assertListEqual(sd.avalanches(0, bin_size=1), [])
        self.assertListEqual(sd.avalanches(10, bin_size=1), [])
        self.assertListEqual(sd.avalanches(3, bin_size=1), [])
        sd = sd_from_counts([5, 4, 3, 2, 1])
        self.assertListEqual(sd.avalanches(3, bin_size=1), [])

    def test_dcc(self):
        # Make sure complete Poisson gibberish doesn't result in
        # anything that looks like a power law.
        sd = random_spikedata(1, 10000)
        dcc = sd.deviation_from_criticality(bin_size=1)
        self.assertTrue(dcc.p_size < 0.05 or dcc.p_duration < 0.05)

        # Corner case: DCC with no avalanches doesn't error.
        sd = sd_from_counts([1, 2, 3, 4, 5])
        sd.deviation_from_criticality()

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

    def test_okun_randomization(self):
        r = np.random.rand(100, 1000) < 0.1
        rr = spikedata.randomize_raster_okun(r)
        self.assertAll(r.sum(0) == rr.sum(0))
        self.assertAll(r.sum(1) == rr.sum(1))

    def test_base_randomization(self):
        r = np.random.rand(100, 1000) < 0.1
        rr = randomize_raster_greedy(r)
        self.assertAll(r.sum(0) == rr.sum(0))
        self.assertAll(r.sum(1) == rr.sum(1))

    def test_best_effort_sample(self):
        # Make sure it crashes if the requested sample is impossible.
        weights = np.arange(100) == 42
        with self.assertRaises(ValueError):
            best_effort_sample(weights, 10)

        # Make sure the same thing works if there are enough entries.
        weights = 10 * (np.arange(100) == 42)
        sample = best_effort_sample(weights, 10)
        self.assertEqual(len(sample), 10)
        self.assertAll(sample == 42)

        # Make sure it doesn't use replacement if it doesn't have to.
        weights = np.ones(100)
        sample = best_effort_sample(weights, 100)
        self.assertEqual(len(sample), 100)
        self.assertAll(np.sort(sample) == np.arange(100))

    def test_randomization_issue_13(self):
        # Having this rate too high is causing a ValueError.
        rates = np.linspace(0.0, 0.5, 100)
        r = np.random.rand(len(rates), 1000) < rates[:, np.newaxis]
        rr = randomize_raster_greedy(r)
        self.assertAll(r.sum(0) == rr.sum(0))
        self.assertAll(r.sum(1) == rr.sum(1))

        # Also make sure it works on rasters with multiple spikes per bin.
        r = np.random.randint(10, size=(100, 1000))
        rr = randomize_raster_greedy(r)
        self.assertAll(r.sum(0) == rr.sum(0))
        self.assertAll(r.sum(1) == rr.sum(1))
