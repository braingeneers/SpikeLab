import contextlib
import heapq
import itertools
import os
import warnings
from collections import namedtuple
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal, sparse, stats

__all__ = [
    "DCCResult",
    "SpikeData",
    "best_effort_sample",
    "burst_detection",
    "cumulative_moving_average",
    "fano_factors",
    "pearson",
    "population_firing_rate",
    "randomize_raster",
    "randomize_raster_greedy",
    "spike_time_tiling",
]

DCCResult = namedtuple("DCCResult", "dcc p_size p_duration")


@dataclass
class NestIDNeuronAttributes:
    """
    Neuron attributes containing nothing but the NEST ID of each unit from a simulation.
    """

    nest_id: int


class SpikeData:
    """
    Class for handling and manipulating neuronal spike data.

    This class provides a way to load, process, and analyze spike data from different
    input types, including NEST spike recorder, lists of indices and times, lists of
    channel-time pairs, lists of Neuron objects, or even prebuilt spike trains.

    Each instance of SpikeData has the following attributes:

    - train: The main data attribute. This is a list of numpy arrays, where each array
      contains the spike times for a particular neuron.

    - N: The number of neurons in the dataset.

    - length: The length of the spike train, defaults to the time of the last spike.

    - neuron_attributes: A list of attribute objects for each neuron. Each item should
      be a dataclass containing a consistent set of fields.

    - metadata: A dictionary containing any additional information or metadata about the
      spike data.

    - raw_data: If provided, this numpy array contains the raw time series data.

    - raw_time: This is either a numpy array of sample times, or a single float
      representing a sample rate in kHz.

    In addition to these data attributes, the SpikeData class also provides some useful
    methods for working with spike data, such as iterating through spike times or
    (index, time) pairs for all units in time order.

    Note that SpikeData expects spike times to be in units of milliseconds, unless a
    list of Neuron objects is given; these have spike times in units of samples, which
    are converted to milliseconds using the sample rate saved in the Neuron object.
    """

    @staticmethod
    def from_idces_times(idces, times, N=None, **kwargs):
        """
        Create a SpikeData object with N total units based on lists of unit indices and
        spike times. If N is not provided, it is set to one more than the maximum index.

        All metadata parameters of the regular constructor are accepted.
        """
        return SpikeData(_train_from_i_t_list(idces, times, N), N=N, **kwargs)

    @staticmethod
    def from_raster(raster, bin_size_ms, **kwargs):
        """
        Create a SpikeData object based on a spike raster matrix with shape (N, T),
        where T is a number of time bins.

        To make it clear which bin each spike belongs to, the generated spike times are
        evenly spaced within each bin. For example, if a unit fires 3 times in a 10 ms
        bin, those events go at 2.5, 5, and 7.5 ms after the start of the bin.

        All metadata parameters of the regular constructor are accepted.
        """
        N, T = raster.shape
        train = [[] for _ in range(N)]
        for i, t in zip(*raster.nonzero()):
            n_spikes = raster[i, t]
            times = t * bin_size_ms + np.linspace(0, bin_size_ms, n_spikes + 2)[1:-1]
            train[i].extend(times)

        kwargs.setdefault("length", T * bin_size_ms)
        return SpikeData(train, **kwargs)

    @staticmethod
    def from_nest(spike_recorder, *nodeses, neuron_attributes=None, **kwargs):
        """
        Create a SpikeData object from a NEST spike recorder. The remaining positional
        arguments should each be an iterable of integers (such as a nest.NodeCollection)
        indicating which units to include.

        If neuron_attributes is provided, each item will be mutated to add a field
        `nest_id`. Otherwise, a new list of neuron attributes objects will be created,
        each containing only the NEST ID of the corresponding unit. All other metadata
        parameters of the regular constructor are also accepted.
        """
        # These are indices and times, but since nodes usually subset the indices, we
        # can't just use the idces+times constructor.
        idces = spike_recorder.events["senders"]
        times = spike_recorder.events["times"]
        cells = np.hstack(nodeses)
        maxcell = cells.max()
        cellrev = np.zeros(maxcell + 1, int)
        cellrev[cells] = np.arange(len(cells))

        cellset = set(cells)
        train = [[] for _ in cells]
        for i, t in zip(idces, times):
            if i in cellset:
                train[cellrev[i]].append(t)

        if not neuron_attributes:
            neuron_attributes = [NestIDNeuronAttributes(i) for i in cells]
        else:
            for i, attrs in enumerate(neuron_attributes):
                attrs.nest_id = cells[i]
        return SpikeData(train, neuron_attributes=neuron_attributes, **kwargs)

    @staticmethod
    def from_events(events, N=None, **kwargs):
        """
        Create a SpikeData object with N total units based on a list of
        events, each an (index, time) pair. If N is not provided, it is
        set to one more than the maximum index.

        All metadata parameters of the regular constructor are accepted.
        """
        idces, times = [], []
        for i, t in events:
            idces.append(i)
            times.append(t)
        return SpikeData.from_idces_times(idces, times, N, **kwargs)

    @staticmethod
    def from_neo_spiketrains(spiketrains, **kwargs):
        """
        Create a SpikeData object from a list of neo.SpikeTrain objects. The spike times
        can be in any units, as they will be converted to regular np.arrays in units of
        milliseconds.
        """
        # This is done in a weird way that involves an extra copy of the data because
        # there's no way to convert the units without modifying the object or importing
        # Quantities. So we copy and in-place change units.
        trains = [st.copy() for st in spiketrains]
        for st in trains:
            st.units = "ms"
        # This on the other hand is NOT a copy, it just allocates new wrapper objects
        # wihle leaving the data buffers intact. This is necessary because some key
        # numpy ufuncs like np.sort() will not work on the Quantity objects.
        return SpikeData([np.asarray(st) for st in spiketrains], **kwargs)

    @staticmethod
    def from_mbt_neurons(neurons, **kwargs):
        """
        Create a SpikeData object from a list of Neuron objects as in the
        MuscleBeachTools package by extracting their list of spike times and
        converting the units to milliseconds.

        All metadata parameters of the regular constructor are accepted.
        """
        return SpikeData(
            [np.asarray(n.spike_time) / n.fs * 1e3 for n in neurons], **kwargs
        )

    @staticmethod
    def from_thresholding(
        data: NDArray,
        fs_Hz=20e3,
        threshold_sigma=5.0,
        filter: dict | bool = True,
        hysteresis=True,
        direction: Literal["both", "up", "down"] = "both",
    ):
        """
        Create a SpikeData object from raw data by filtering and thresholding raw
        electrophysiological data formatted as an array with shape (channels, time).

        If filter is True (default), filter the data using a third-order Butterworth
        filter with passband 300 Hz to 6 kHz. To use different filter parameters, pass a
        dictionary, which will be passed as keyword arguments to butter_filter(). If
        filter is falsy, no filtering is done.
        """
        if filter:
            if filter is True:
                filter = dict(lowcut=300.0, highcut=6e3, order=3)
            data = butter_filter(data, fs=fs_Hz, **filter)

        threshold = threshold_sigma * np.std(data, axis=1, keepdims=True)

        if direction == "both":
            raster = (data > threshold) | (data < -threshold)
        elif direction == "up":
            raster = data > threshold
        elif direction == "down":
            raster = data < -threshold

        if hysteresis:
            raster = np.diff(np.array(raster, dtype=int), axis=1) == 1

        return SpikeData.from_raster(
            raster, 1e3 / fs_Hz, raw_data=data, raw_time=fs_Hz / 1e3
        )

    def __init__(
        self,
        train,
        *,
        N=None,
        length=None,
        neuron_attributes=None,
        metadata={},
        raw_data=None,
        raw_time: NDArray | float | None = None,
    ):
        """
        Initialize a SpikeData object using a list of spike trains, each a
        list of spike times in milliseconds.

        Arbitrary raw timeseries data, not associated with particular units,
        can be passed in as `raw_data`, an array whose last dimension
        corresponds to the times given in `raw_time`. The `raw_time` argument
        can also be a sample rate in kHz, in which case it is generated
        assuming that the start of the raw data corresponds with t=0.
        """
        # Make sure each individual spike train is sorted. As a side effect,
        # also copy each array to avoid aliasing.
        self.train = [np.sort(times) for times in train]

        # The length of the spike train defaults to the last spike
        # time it contains.
        if length is None:
            length = max((t[-1] for t in self.train if len(t) > 0))
        self.length = length

        # If a number of units was provided, make the list of spike
        # trains consistent with that number.
        if N is not None and len(self.train) < N:
            self.train += [np.array([], float) for _ in range(N - len(self.train))]
        self.N = len(self.train)

        # Add the raw data if present, including generating raw time.
        if raw_data is not None and raw_time is not None:
            self.raw_data = np.asarray(raw_data)
            self.raw_time = np.asarray(raw_time)
            if np.ndim(self.raw_time) == 0:
                self.raw_time = np.arange(self.raw_data.shape[-1]) / raw_time
            elif self.raw_data.shape[-1:] != self.raw_time.shape:
                raise ValueError("Length of `raw_data` and " "`raw_time` must match.")
        elif raw_data is None and raw_time is None:
            self.raw_data = np.zeros((0, 0))
            self.raw_time = np.zeros((0,))
        else:
            raise ValueError(
                "Must provide both or neither of " "`raw_data` and `raw_time`."
            )

        # Add metadata and neuron_attributes, then validate that neuron_attributes
        # contains the right number of neurons.
        #
        # Note that if there is no metadata, it should be an empty dict, because that
        # way arbitrary fields can be added later, but null neuron_attributes requires
        # storing None so we don't get misaligned by concatenating an empty list later.
        self.metadata = metadata.copy()
        self.neuron_attributes = None
        if neuron_attributes:
            self.neuron_attributes = neuron_attributes.copy()
            if len(neuron_attributes) != self.N:
                raise ValueError(
                    f"neuron_attributes has {len(neuron_attributes)} "
                    f"instead of {self.N} items."
                )

    @property
    def times(self):
        "Iterate spike times for all units in time order."
        return heapq.merge(*self.train)

    @property
    def events(self):
        "Iterate (index,time) pairs for all units in time order."
        return heapq.merge(
            *[zip(itertools.repeat(i), t) for (i, t) in enumerate(self.train)],
            key=lambda x: x[1],
        )

    def idces_times(self):
        """
        Generate a matched pair of numpy arrays containing unit indices and times for
        all events.

        This is not a property unlike `times` and `events` because the lists must
        actually be constructed in memory.
        """
        idces, times = [], []
        for i, t in self.events:
            idces.append(i)
            times.append(t)
        return np.array(idces), np.array(times)

    def frames(self, length, overlap=0):
        """
        Iterate new SpikeData objects corresponding to subwindows of a given `length`
        with a fixed `overlap`.
        """
        for start in np.arange(0, self.length, length - overlap):
            yield self.subtime(start, start + length)

    def binned(self, bin_size=40.0):
        """
        Quantize time into intervals of bin_size and counts the number of events in
        each bin, considered as a lower half-open interval of times, with the exception
        that events at time precisely zero will be included in the first bin.
        """
        return self.sparse_raster(bin_size).sum(0)  # type: ignore

    def binned_meanrate(self, bin_size=40, unit="kHz"):
        """
        Calculate the mean firing rate across the population in each time bin.

        The rate is calculated as the number of events in each bin divided by the bin
        size and number of units. The unit may be either `Hz` or `kHz` (default).
        """
        binned_rate = self.binned(bin_size) / self.N / bin_size
        if unit == "Hz":
            return 1e3 * binned_rate
        elif unit == "kHz":
            return binned_rate
        else:
            raise ValueError(f"Unknown unit {unit} (try Hz or kHz)")

    def rates(self, unit="kHz"):
        """
        Calculate the mean firing rate of each neuron as an average number of events per
        time over the length of the data. The unit may be `Hz` or `kHz` (default).
        """
        rates = np.array([len(t) for t in self.train]) / self.length
        if unit == "Hz":
            return 1e3 * rates
        elif unit == "kHz":
            return rates
        else:
            raise ValueError(f"Unknown unit {unit} (try Hz or kHz)")

    def resampled_isi(self, times, sigma_ms=10.0):
        """
        Calculate firing rate of each unit at the given times by calculating the
        interspike intervals and interpolating their inverse.
        """
        return np.array([_resampled_isi(t, times, sigma_ms) for t in self.train])

    def subset(self, units, by=None):
        """
        Return a new SpikeData with spike times for only some units, selected either by
        their indices or by an ID stored under a given key in the neuron_attributes.

        Units are included in the output according to their order in self.train, not the
        order in the unit list (which is treated as a set).

        If IDs are not unique, every neuron which matches is included in the output.
        Neurons whose neuron_attributes entry does not have the key are always excluded.
        """
        if isinstance(units, int):
            units = [units]
        units = set(units)
        if by is not None:
            if self.neuron_attributes is None:
                raise ValueError("can't use `by` without `neuron_attributes`")
            _missing = object()
            units = {
                i
                for i in range(self.N)
                if getattr(self.neuron_attributes[i], by, _missing) in units
            }

        train = []
        neuron_attributes = []
        for i, ts in enumerate(self.train):
            if i in units:
                train.append(ts)
                if self.neuron_attributes is not None:
                    neuron_attributes.append(self.neuron_attributes[i])

        return SpikeData(
            train,
            length=self.length,
            N=len(train),
            neuron_attributes=neuron_attributes or None,
            metadata=self.metadata,
            raw_time=self.raw_time,
            raw_data=self.raw_data,
        )

    def subtime(self, start, end):
        """
        Return a new SpikeData with only spikes in a time range, closed on top but open
        on the bottom unless the lower bound is zero, consistent with the binning
        methods. This is to ensure no overlap between adjacent slices.

        Start and end can be negative, in which case they are counted backwards from the
        end. They can also be None or Ellipsis, in which case that end of the data is
        not truncated. All metadata and neuron data are propagated, while raw data is
        sliced to the same range of times, including all samples in the closed interval.
        """
        if start is None or start is Ellipsis:
            start = 0
        elif start < 0:
            start += self.length

        if end is None or end is Ellipsis:
            end = self.length
        elif end < 0:
            end += self.length
        elif end > self.length:
            end = self.length

        # Special case out the start=0 case by nopping the comparison.
        lower = start if start > 0 else -np.inf

        # Subset the spike train by time.
        train = [t[(t > lower) & (t <= end)] - start for t in self.train]

        # Subset and propagate the raw data.
        rawmask = (self.raw_time >= lower) & (self.raw_time <= end)
        return SpikeData(
            train,
            length=end - start,
            N=self.N,
            neuron_attributes=self.neuron_attributes,
            metadata=self.metadata,
            raw_time=self.raw_time[rawmask] - start,
            raw_data=self.raw_data[..., rawmask],
        )

    def __getitem__(self, key):
        """
        If a slice is provided, it is taken in time as with self.subtime(), but if an
        iterable is provided, it is taken as a list of neuron indices to select using
        self.subset().
        """
        if isinstance(key, slice):
            return self.subtime(key.start, key.stop)
        else:
            return self.subset(key)

    def append(self, spikeData, offset=0):
        """
        Append the spike times from another SpikeData object to this one, optionally
        offsetting them by a given amount from the end of the current data.

        The two SpikeData objects must have the same number of neurons.
        """
        if self.N != spikeData.N:
            raise ValueError("Cannot concatenate SpikeData with different N")
        train = [
            np.hstack([tr1, tr2 + self.length + offset])
            for tr1, tr2 in zip(self.train, spikeData.train)
        ]
        raw_data = np.concatenate((self.raw_data, spikeData.raw_data), axis=1)
        raw_time = np.concatenate((self.raw_time, spikeData.raw_time))
        length = self.length + spikeData.length + offset
        return SpikeData(
            train,
            length=length,
            N=self.N,
            neuron_attributes=self.neuron_attributes,
            raw_time=raw_time,
            raw_data=raw_data,
            metadata={
                **spikeData.metadata,
                **self.metadata,
            },  # Append the dicts together
        )

    def sparse_raster(self, bin_size=20.0):
        """
        Bin all spike times and create a sparse array where entry (i,j) is the number of
        times unit i fired in bin j.

        Bins are left-open and right-closed intervals except the first, which will
        capture any spikes occurring exactly at t=0.
        """
        indices = np.hstack([np.ceil(ts / bin_size) - 1 for ts in self.train]).astype(
            int
        )
        units = np.hstack([0] + [len(ts) for ts in self.train])
        indptr = np.cumsum(units)
        values = np.ones_like(indices)
        length = int(np.ceil(self.length / bin_size))
        np.clip(indices, 0, length - 1, out=indices)
        ret = sparse.csr_array((values, indices, indptr), shape=(self.N, length))
        return ret

    def raster(self, bin_size=20.0):
        """
        Bin all spike times and create a dense array where entry (i,j) is the number of
        times cell i fired in bin j.

        Bins are left-open and right-closed intervals except the first, which will
        capture any spikes occurring exactly at t=0.
        """
        return self.sparse_raster(bin_size).toarray()

    def interspike_intervals(self):
        "Produce a list of arrays of interspike intervals per unit."
        return [np.diff(ts) for ts in self.train]

    def isi_skewness(self):
        "Calculate the skewness of the interspike interval distribution for each unit."
        intervals = self.interspike_intervals()
        return [stats.skew(intl) for intl in intervals]

    def isi_log_histogram(self, bin_num=300):
        """
        Histogram of interspike intervals with logarithmic bin spacing. Returns both the
        histogram and the generated bin edges.
        """
        intervals = self.interspike_intervals()
        ret = []
        ret_logbins = []
        for ts in intervals:
            log_bins = np.geomspace(min(ts), max(ts), bin_num + 1)
            hist, _ = np.histogram(ts, log_bins)
            ret.append(hist)
            ret_logbins.append(log_bins)
        return ret, ret_logbins

    def isi_threshold_cma(self, hist, bins, coef=1):
        """
        Calculate interspike interval threshold from cumulative moving average [1]. The
        threshold is the bin that has the maximum CMA on the interspike interval
        histogram. Histogram and bins are logarithmic by default. `coef` is an input
        variable for threshold.

        [1] Kapucu, et al. Frontiers in Computational Neuroscience 6:38 (2012)
        """
        isi_thr = []
        for n in range(len(hist)):
            h = hist[n]
            max_idx = 0
            cma = 0
            cma_list = []
            for i in range(len(h)):
                cma = (cma * i + h[i]) / (i + 1)
                cma_list.append(cma)
            max_idx = np.argmax(cma_list)
            thr = (bins[n][max_idx + 1]) * coef
            isi_thr.append(thr)
        return isi_thr

    def burstiness_index(self, bin_size=40.0):
        """
        Compute the burstiness index [1], a number from 0 to 1 which quantifies
        synchronization of activity in neural cultures.

        Spikes are binned, and the fraction of spikes accounted for by the top 15% is
        calculated. This will be 0.15 if activity is fully asynchronous, and 1.0 if
        activity is fully synchronized into just a few bins. The result is rescaled to
        the range 0-1 for clearer interpretation.

        [1] Wagenaar, Madhavan, Pine & Potter. Controlling bursting in cortical cultures
            with closed-loop multi-electrode stimulation. Journal of Neuroscience 25:3,
            680–688 (2005).
        """
        binned = self.binned(bin_size)
        binned.sort()
        N85 = int(np.round(len(binned) * 0.85))

        # Special case to avoid an error when there is only one bin.
        if N85 == len(binned):
            return 1.0
        else:
            f15 = binned[N85:].sum() / binned.sum()
            return (f15 - 0.15) / 0.85

    def concatenate_spike_data(self, sd):
        """
        Add the units from another SpikeData object to this one. The new units are
        assigned indices starting from the end of the current data. If the new units
        have a longer spike train, it is truncated to the length of the current data.
        """
        if sd.length != self.length:
            sd = sd.subtime(0, self.length)
        self.train += sd.train
        self.N += sd.N
        self.raw_data += sd.raw_data
        self.raw_time += sd.raw_time
        self.metadata.update(sd.metadata)
        if self.neuron_attributes and sd.neuron_attributes:
            self.neuron_attributes += sd.neuron_attributes
        elif self.neuron_attributes or sd.neuron_attributes:
            warnings.warn(
                "Concatenating SpikeData where one has no neuron_attributes.",
                RuntimeWarning,
            )

    def spike_time_tilings(self, delt=20.0):
        """
        Compute the full spike time tiling coefficient matrix. STTC is a metric for
        correlation between spike trains with some improved intuitive properties
        compared to the Pearson correlation coefficient. Spike trains are lists of spike
        times sorted in ascending order.

        [1] Cutts & Eglen. Detecting pairwise correlations in spike trains: An objective
            comparison of methods and application to the study of retinal waves. Jouranl
            of Neuroscience 34:43, 14288–14303 (2014).
        """
        T = self.length
        ts = [_sttc_ta(ts, delt, T) / T for ts in self.train]

        ret = np.diag(np.ones(self.N))
        for i in range(self.N):
            for j in range(i + 1, self.N):
                ret[i, j] = ret[j, i] = _spike_time_tiling(
                    self.train[i], self.train[j], ts[i], ts[j], delt
                )
        return ret

    def spike_time_tiling(self, i, j, delt=20.0):
        """
        Calculate the spike time tiling coefficient between two units within
        this SpikeData. STTC is a metric for correlation between spike trains with some
        improved intuitive properties compared to the Pearson correlation coefficient.
        Spike trains are lists of spike times sorted in ascending order.

        [1] Cutts & Eglen. Detecting pairwise correlations in spike trains: An objective
            comparison of methods and application to the study of retinal waves. Jouranl
            of Neuroscience 34:43, 14288–14303 (2014).
        """
        return spike_time_tiling(self.train[i], self.train[j], delt, self.length)

    def avalanches(self, thresh, bin_size=40.0):
        """
        Bin the spikes in this data, and group the result into lists corresponding to
        avalanches, defined as deviations above a given threshold spike count.
        """
        counts = self.binned(bin_size)
        active = counts > thresh
        toggles = np.where(np.diff(active))[0]

        # If we start inactive, the first toggle begins the first
        # avalanche. Otherwise, we have to ignore it because we don't
        # know how long the system was active before.
        if active[0]:
            ups = toggles[1::2]
            downs = toggles[2::2]
        else:
            ups = toggles[::2]
            downs = toggles[1::2]

        # Now batch up the transitions and create a list of spike
        # counts in between them.
        return [counts[up + 1 : down + 1] for up, down in zip(ups, downs)]

    def avalanche_duration_size(self, thresh, bin_size=40):
        """
        Collect the avalanches in this data and regroup them into a pair of lists:
        durations and sizes.
        """
        durations, sizes = [], []
        for avalanche in self.avalanches(thresh, bin_size):
            durations.append(len(avalanche))
            sizes.append(sum(avalanche))
        return np.array(durations), np.array(sizes)

    def deviation_from_criticality(
        self, quantile=0.35, bin_size=40, N=1000, pval_truncated=0.05
    ):
        """
        Calculates the deviation from criticality according to the method of Ma et al.
        (2019), who used the relationship of the dynamical critical exponent to the
        exponents of the separate power laws corresponding to the avalanche size and
        duration distributions as a metric for suboptimal cortical function following
        monocular deprivation.

        The returned DCCResult struct contains not only the DCC metric itself but also
        the significance of the hypothesis that the size and duration distributions of
        the extracted avalanches are poorly fit by power laws.

        [1] Ma, Z., Turrigiano, G. G., Wessel, R. & Hengen, K. B. Cortical circuit
            dynamics are homeostatically tuned to criticality in vivo. Neuron 104,
            655-664.e4 (2019).
        """
        # Calculate the spike count threshold corresponding to
        # the given quantile.
        thresh = np.quantile(self.binned(bin_size), quantile)

        # Gather durations and sizes. If there are no avalanches, we
        # very much can't say the system is critical.
        durations, sizes = self.avalanche_duration_size(thresh, bin_size)
        if len(durations) == 0:
            return DCCResult(dcc=np.inf, p_size=1.0, p_duration=1.0)

        # Call out to all the actual statistics.
        p_size, alpha_size = _p_and_alpha(sizes, N, pval_truncated)
        p_dur, alpha_dur = _p_and_alpha(durations, N, pval_truncated)

        # Fit and predict the dynamical critical exponent.
        τ_fit = np.polyfit(np.log(durations), np.log(sizes), 1)[0]
        τ_pred = (alpha_dur - 1) / (alpha_size - 1)
        dcc = abs(τ_pred - τ_fit)

        # Return the DCC value and significance.
        return DCCResult(dcc=dcc, p_size=p_size, p_duration=p_dur)

    def latencies(self, times, window_ms=100.0):
        """
        Given a sorted list of times, compute the latencies from that time to each spike
        in each spike train within a window.

        :param times: list of times
        :param window_ms: window in ms
        :return: 2d list, each row is a list of latencies
                        from a time to each spike in the train
        """
        latencies = []
        if len(times) == 0:
            return latencies

        for train in self.train:
            cur_latencies = []
            if len(train) == 0:
                latencies.append(cur_latencies)
                continue
            for time in times:
                # Subtract time from all spikes in the train
                # and take the absolute value
                abs_diff_ind = np.argmin(np.abs(train - time))

                # Calculate the actual latency
                latency = np.array(train) - time
                latency = latency[abs_diff_ind]

                abs_diff = np.abs(latency)
                if abs_diff <= window_ms:
                    cur_latencies.append(latency)
            latencies.append(cur_latencies)
        return latencies

    def latencies_to_index(self, i, window_ms=100.0):
        """
        Compute the latency from one unit to all other units via self.latencies().

        :param i: index of the unit
        :param window_ms: window in ms
        :return: 2d list, each row is a list of latencies per neuron
        """
        return self.latencies(self.train[i], window_ms)

    def randomized(self, bin_size_ms=1.0, seed=None, method="poprate_greedy"):
        """
        Create a new SpikeData object which preserves the population rate and mean
        firing rate of each neuron in an existing SpikeData by randomly reallocating all
        spike times to different neurons at a resolution given by dt.

        See `spikedata.randomize_raster()` for a list of supported methods.
        """
        return SpikeData.from_raster(
            randomize_raster(self.sparse_raster(bin_size_ms), seed, method),
            bin_size_ms,
            length=self.length,
            metadata=self.metadata,
            neuron_attributes=self.neuron_attributes,
        )

    def population_firing_rate(self, bin_size=10, w=5, average=False):
        """
        Calculate binned population firing rate, smoothed by a moving average filter of
        width w bins. If average is True, divide the result by N to yield the population
        average firing rate instead of the total population rate (default).
        """
        bins, pop_rate = population_firing_rate(
            self.train, self.length, bin_size, w, average
        )
        return bins, pop_rate


def population_firing_rate(
    trains: list[NDArray] | NDArray,
    rec_length=None,
    bin_size=10,
    w=5,
    average=False,
):
    """
    Calculate population firing rate for given spike trains.

    :param trains: a list of spike trains, or a single combined spike train.
    :param rec_length: length of the recording.
                       If None, the maximum spike time is used.
    :param bin_size: binning width
    :param w: kernel width for smoothing
    :param average: If True, the result is averaged by the number of units.
                    Otherwise, the result is return as it is.
    :return: An array of the bins and an array of the frequency
             for the given units' spiking activity
    """
    if isinstance(trains, np.ndarray):
        trains = [trains]

    N = len(trains)
    trains = np.hstack(trains)

    if rec_length is None:
        rec_length = np.max(trains)

    bin_num = int(rec_length // bin_size) + 1
    bins = np.linspace(0, rec_length, bin_num)
    fr = np.histogram(trains, bins)[0] / bin_size
    fr_pop = np.convolve(fr, np.ones(w), "same") / w
    if average:
        fr_pop /= N
    return bins, fr_pop


def spike_time_tiling(tA, tB, delt=20.0, length: float | None = None):
    """
    Calculate the spike time tiling coefficient [1] between two spike trains. STTC is a
    metric for correlation between spike trains with some improved intuitive properties
    compared to the Pearson correlation coefficient. Spike trains are lists of spike
    times sorted in ascending order.

    [1] Cutts & Eglen. Detecting pairwise correlations in spike trains: An objective
        comparison of methods and application to the study of retinal waves. Journal of
        Neuroscience 34:43, 14288–14303 (2014).
    """
    if length is None:
        length = float(max(tA[-1], tB[-1]))

    if len(tA) == 0 or len(tB) == 0:
        return 0.0

    TA = _sttc_ta(tA, delt, length) / length
    TB = _sttc_ta(tB, delt, length) / length
    return _spike_time_tiling(tA, tB, TA, TB, delt)


def _spike_time_tiling(tA, tB, TA, TB, delt):
    "Internal helper method for the second half of STTC calculation."
    if len(tA) == 0 or len(tB) == 0:
        return 0
    PA = _sttc_na(tA, tB, delt) / len(tA)
    PB = _sttc_na(tB, tA, delt) / len(tB)

    aa = (PA - TB) / (1 - PA * TB) if PA * TB != 1 else 0
    bb = (PB - TA) / (1 - PB * TA) if PB * TA != 1 else 0
    return (aa + bb) / 2


def best_effort_sample(counts, M, rng: np.random.Generator | None = None):
    """
    Given a discrete distribution over the integers 0...N-1 in the form of an array of N
    counts, sample M elements from the distribution without replacement if possible. If
    not possible, sample with replacement but without exceeding the counts.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(counts)
    try:
        return rng.choice(N, size=M, replace=False, p=counts / counts.sum())
    except ValueError:
        pigeonhole = np.arange(len(counts))[counts > 0]
        new_counts = np.maximum(counts - 1, 0)
        if new_counts.sum() == 0:
            raise
        choices = best_effort_sample(new_counts, M - len(pigeonhole), rng)
        ret = np.concatenate((pigeonhole, choices))
        rng.shuffle(ret)
        return ret


def randomize_raster(raster, seed: int | None = None, method="poprate_greedy"):
    """
    Generate a randomized version of a spike raster by reassigning all its spike times
    to new bins, while preserving the total number of spikes for each unit.

    Supported randomization methods:
    - `poprate_greedy` preserves the population rate with an efficient algorithm, but
      may in some cases assign multiple spikes to a single bin.
    - `poprate_okun` preserves population rate and does not create spurious multi-spike
      bins, but only works when there is at most one spike per unit per bin.
    """
    if method == "poprate_greedy":
        return randomize_raster_greedy(raster, seed)
    elif method == "poprate_okun":
        return randomize_raster_okun(raster, seed)
    else:
        raise ValueError(f"Unsupported randomization method `{method}`.")


def _okun_swap(ar, idxs, rng):
    idx0 = rng.randint(len(idxs[0]))
    idx1 = rng.randint(len(idxs[0]))
    i0, j0 = idxs[0][idx0], idxs[1][idx0]
    i1, j1 = idxs[0][idx1], idxs[1][idx1]
    if i0 == i1 or j0 == j1 or ar[i0, j1] == 1.0 or ar[i1, j0] == 1.0:
        return False
    ar[i0, j0] = ar[i1, j1] = 0.0
    ar[i0, j1] = ar[i1, j0] = 1.0
    idxs[0][idx0], idxs[1][idx0] = i0, j1
    idxs[0][idx1], idxs[1][idx1] = i1, j0
    return True


def randomize_raster_okun(raster, seed: int | None = None, swap_per_spike=5):
    """
    Generate a randomized version of a spike raster, preserving population rate. The
    input raster MUST have at most one spike per neuron per bin!

    Algorithm due to Okun et al. (2015) "Diverse coupling of neurons to populations in
    sensory cortex," implementation by TJ van der Molen from "Protosequences in human
    cortical organoids model intrinsic states in the developing cortex."
    """
    rng = np.random.RandomState(seed)
    raster = raster.copy()
    idxs = np.where(raster == 1.0)
    cnt_swap = 0
    for _ in range(int((swap_per_spike + 1) * np.sum(raster))):
        if _okun_swap(raster, idxs, rng):
            cnt_swap += 1

    if cnt_swap < swap_per_spike * np.sum(raster):
        for _ in range(int((swap_per_spike + 1) * np.sum(raster))):
            if _okun_swap(raster, idxs, rng):
                cnt_swap += 1

    if cnt_swap < swap_per_spike * np.sum(raster):
        warnings.warn(
            "Insufficient successful swaps, only {} of {} required".format(
                cnt_swap, swap_per_spike * np.sum(raster)
            )
        )

    return raster


def randomize_raster_greedy(raster, seed: int | None = None):
    """
    Generate a randomized version of a spike raster, preserving population rate.

    This method greedily reassigns all spikes (going from the most to least active bin),
    meaning that it is relatively fast, but in some cases, it may have to allocate
    multiple spikes to the same neuron in one time bin, even if that was not true in the
    original input.
    """
    rsm = np.zeros(raster.shape, int)
    weights = raster.sum(1)

    # Iterate over the bins in order of how many spikes they have.
    n_spikeses = raster.sum(0)
    bin_order = np.argsort(n_spikeses)[::-1]
    bin_order = bin_order[n_spikeses[bin_order] > 0]

    # Choose which units to assign spikes to in each bin.
    rng = np.random.default_rng(seed)
    for bin in bin_order:
        for unit in best_effort_sample(weights, n_spikeses[bin], rng):
            weights[unit] -= 1
            rsm[unit, bin] += 1

    return rsm


def _resampled_isi(spikes, times, sigma_ms):
    """
    Helper method for calculating the firing rate of a spike train at specific times,
    based on the reciprocal inter-spike interval. It is assumed to have been sampled
    halfway between any two given spikes, interpolated, and then smoothed by a Gaussian
    kernel with the given width.
    """
    if len(spikes) == 0:
        return np.zeros_like(times)
    elif len(spikes) == 1:
        return np.ones_like(times) / spikes[0]
    else:
        x = 0.5 * (spikes[:-1] + spikes[1:])
        y = 1 / np.diff(spikes)
        fr = np.interp(times, x, y)
        if len(np.atleast_1d(fr)) < 2:
            return fr

        dt_ms = times[1] - times[0]
        sigma = sigma_ms / dt_ms
        if sigma > 0:
            return ndimage.gaussian_filter1d(fr, sigma)
        else:
            return fr


def _p_and_alpha(data, N_surrogate=1000, pval_truncated=0.0):
    """
    Helper method for DCC that performs a power-law fit to some data, and returns a
    p-value for the hypothesis that this fit is poor, together with just the exponent of
    the fit.

    A positive value of `pval_truncated` means to allow the hypothesis of a truncated
    power law, which must be better than the plain power law with the given significance
    under powerlaw's default nested hypothesis comparison test.

    The returned significance value is computed by sampling N surrogate datasets and
    counting what fraction are further from the fitted distribution according to the
    one-sample Kolmogorov-Smirnoff test.
    """
    # Perform the fits and compare the distributions with IO
    # silenced because there's no option to disable printing
    # in this library...
    try:
        from powerlaw import Fit
    except ImportError:
        raise ImportError("The powerlaw library is required to compute DCC.")

    with (
        open(os.devnull, "w") as f,
        contextlib.redirect_stdout(f),
        contextlib.redirect_stderr(f),
    ):
        fit = Fit(data)
        stat, p = fit.distribution_compare(
            "power_law", "truncated_power_law", nested=True
        )

    # If the truncated power law is a significantly better
    # explanation of the data, use it.
    if stat < 0 and p < pval_truncated:
        dist = fit.truncated_power_law
    else:
        dist = fit.power_law

    # The p-value of the fit is the fraction of surrogate
    # datasets which it fits worse than the input dataset.
    ks = stats.ks_1samp(data, dist.cdf)
    p = np.mean(
        [
            stats.ks_1samp(dist.generate_random(len(data)), dist.cdf) > ks
            for _ in range(N_surrogate)
        ]
    )
    return p, dist.alpha


def _train_from_i_t_list(idces, times, N):
    """
    Helper method for SpikeData constructors: given lists of spike times and indices,
    produce a list whose ith entry is a list of the spike times of the ith unit.
    """
    idces, times = np.asarray(idces), np.asarray(times)
    if N is None:
        N = idces.max() + 1

    ret = []
    for i in range(N):
        ret.append(times[idces == i])
    return ret


def fano_factors(raster: NDArray):
    """
    Given arrays of spike times and the corresponding units which produced them,
    computes the Fano factor of the corresponding spike raster.

    If a unit doesn't fire, a Fano factor of 1 is returned because in the limit of
    events happening at a rate ε->0, either as a Bernoulli process or in the many-bins
    limit of a single event, the Fano factor converges to 1.
    """
    if sparse.issparse(raster):
        mean = np.array(raster.mean(1)).ravel()
        prod = raster.multiply(raster)
        moment = np.array(prod.mean(1)).ravel()  # type: ignore

        # Silly numbers to make the next line return f=1 for a unit that never spikes.
        moment[mean == 0] = 2
        mean[mean == 0] = 1

        # This is the variance/mean ratio computed in a sparse-friendly way. This
        # algorithm is numerically unstable, but it's the best we can do.
        return moment / mean - mean

    else:
        mean: NDArray = np.asarray(raster).mean(1)
        var: NDArray = np.asarray(raster).var(1)
        mean[mean == 0] = var[mean == 0] = 1.0
        return var / mean


def _sttc_ta(tA, delt: float, tmax: float) -> float:
    """
    Helper function for spike time tiling coefficients: calculate the total amount of
    time within a range delt of spikes within the given sorted list of spike times tA.
    """
    if len(tA) == 0:
        return 0.0

    base = min(delt, tA[0]) + min(delt, tmax - tA[-1])
    return base + np.minimum(np.diff(tA), 2 * delt).sum()


def _sttc_na(tA, tB, delt: float) -> int:
    """
    Helper function for spike time tiling coefficients: given two sorted lists of spike
    times, calculate the number of spikes in spike train A within delt of any spike in
    spike train B.
    """
    if len(tB) == 0:
        return 0
    tA, tB = np.asarray(tA), np.asarray(tB)

    # Find the closest spike in B after spikes in A.
    iB = np.searchsorted(tB, tA)

    # Clip to ensure legal indexing, then check the spike at that
    # index and its predecessor to see which is closer.
    np.clip(iB, 1, len(tB) - 1, out=iB)
    dt_left = np.abs(tB[iB] - tA)
    dt_right = np.abs(tB[iB - 1] - tA)

    # Return how many of those spikes are actually within delt.
    return (np.minimum(dt_left, dt_right) <= delt).sum()


def pearson(spikes: NDArray) -> NDArray[np.floating]:
    """
    Compute a Pearson correlation coefficient matrix for a spike raster. Includes a
    sparse-friendly method for very large spike rasters, but falls back on np.corrcoef
    otherwise because this method can be numerically unstable.
    """
    if not sparse.issparse(spikes):
        return np.corrcoef(spikes)

    Exy = (spikes @ spikes.T) / spikes.shape[1]
    Ex = spikes.mean(axis=1)
    Ex2 = (spikes**2).mean(axis=1)
    σx = np.sqrt(Ex2 - Ex**2)

    # Some cells won't fire in the whole observation window. To get their
    # correlation coefficients to zero, give them infinite σ.
    σx[σx == 0] = np.inf

    # This is by the formula, but there's also a hack to deal with the
    # numerical issues that break the invariant that every variable
    # should have a Pearson autocorrelation of 1.
    Exx = np.multiply.outer(Ex, Ex)
    σxx = np.multiply.outer(σx, σx)
    corr = np.array(Exy - Exx) / σxx
    np.fill_diagonal(corr, 1)
    return corr


def cumulative_moving_average(hist):
    "The cumulative moving average for a histogram. Return a list of CMA."
    ret = []
    for h in hist:
        cma = 0
        cma_list = []
        for i in range(len(h)):
            cma = (cma * i + h[i]) / (i + 1)
            cma_list.append(cma)
        ret.append(cma_list)
    return ret


def burst_detection(spike_times, burst_threshold, spike_num_thr=3):
    """
    Detect burst from spike times with a interspike interval threshold (burst_threshold)
    and a spike number threshold (spike_num_thr).

    Returns:
        spike_num_list -- a list of burst features
          [index of burst start point, number of spikes in this burst]
        burst_set -- a list of spike times of all the bursts.
    """
    spike_num_burst = 1
    spike_num_list = []
    for i in range(len(spike_times) - 1):
        if spike_times[i + 1] - spike_times[i] <= burst_threshold:
            spike_num_burst += 1
        else:
            if spike_num_burst >= spike_num_thr:
                spike_num_list.append([i - spike_num_burst + 1, spike_num_burst])
                spike_num_burst = 1
            else:
                spike_num_burst = 1
    burst_set = []
    for loc in spike_num_list:
        for i in range(loc[1]):
            burst_set.append(spike_times[loc[0] + i])
    return spike_num_list, burst_set


def butter_filter(
    data,
    lowcut: float | None = None,
    highcut: float | None = None,
    fs=20000.0,
    order=5,
):
    """
    A digital butterworth filter. Type is based on input value.

    Inputs:
        data: array_like data to be filtered
        lowcut: low cutoff frequency. If None or 0, highcut must be a number.
                Filter is lowpass.
        highcut: high cutoff frequency. If None, lowpass must be a non-zero number.
                 Filter is highpass.
        If lowcut and highcut are both give, this filter is bandpass.
        In this case, lowcut must be smaller than highcut.
        fs: sample rate
        order: order of the filter

    Returns:
        The filtered output with the same shape as data
    """
    match (lowcut, highcut):
        case None, None:
            raise ValueError(
                "Need at least a low cutoff (lowcut) or high cutoff (highcut) frequency!"
            )
        case None, _:
            filter_type = "lowpass"
            Wn = highcut / fs * 2
        case _, None:
            filter_type = "highpass"
            Wn = lowcut / fs * 2
        case _, _:
            if lowcut >= highcut:
                raise ValueError("lowcut must be smaller than highcut")
            filter_type = "bandpass"
            band = [lowcut, highcut]
            Wn = [e / fs * 2 for e in band]

    filter_coeff = signal.iirfilter(
        order, Wn, analog=False, btype=filter_type, output="sos"
    )
    filtered_traces = signal.sosfiltfilt(filter_coeff, data)
    return filtered_traces
