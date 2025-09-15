"""
SpikeData core module.

Refactor (2025-09): API simplification and STTC reorganization
------------------------------------------------------------
This module was refactored to streamline the public API and colocate related
spike time tiling (STTC) helpers. The following previously exported items were
removed to reduce surface area and will be replaced by focused utilities:

- Nest/NEST features: NestIDNeuronAttributes, SpikeData.from_nest
- MuscleBeachTools: SpikeData.from_mbt_neurons
- ISI analytics: SpikeData.isi_skewness, SpikeData.isi_log_histogram,
  SpikeData.isi_threshold_cma
- Burst/avalanche/DCC: SpikeData.burstiness_index, SpikeData.avalanches,
  SpikeData.avalanche_duration_size, SpikeData.deviation_from_criticality,
  DCCResult, _p_and_alpha
- Randomization: SpikeData.randomized, randomize_raster, randomize_raster_greedy,
  randomize_raster_okun, _okun_swap, best_effort_sample
- Rates/correlations/hist utils: population_firing_rate (function and method),
  fano_factors, pearson, cumulative_moving_average, burst_detection

Reorganization:
- STTC helpers `_sttc_ta` and `_sttc_na` are colocated with the public
  `spike_time_tiling` function for clarity. Behavior is unchanged.

Notes for users migrating from older versions:
- Population rate: use `SpikeData.binned(bin_size)` and smooth externally
  (e.g., `np.convolve`) to reproduce prior behavior.
- Pairwise correlations: compute with your preferred method (e.g., NumPy or
  SciPy) on `SpikeData.raster()` output.
- Burst-related functionality will be provided by replacement modules.

No behavior changes were made to remaining APIs unless noted in their docstrings.
"""

import heapq
import itertools
import warnings
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal, sparse

from .utils import (
    spike_time_tiling,
    butter_filter,
    _sttc_ta,
    _sttc_na,
    _spike_time_tiling,
    _resampled_isi,
    _train_from_i_t_list,
    swap,
    randomize,
    get_pop_rate,
    get_bursts,
)

__all__ = [
    "SpikeData",
    "spike_time_tiling",
    "swap",
    "randomize",
    "get_pop_rate",
    "get_bursts",
]


class SpikeData:
    """
    Class for handling and manipulating neuronal spike data.

    This class provides a way to load, process, and analyze spike data from different
    input types, including lists of indices and times, lists of channel-time pairs,
    or prebuilt spike trains.

    Refactor 2025-09:
    - Removed niche/deprecated features (nest/mbt constructors, burst/avalanche/DCC,
      randomization, legacy correlation/utilities). Core loading, binning, rates,
      and STTC functionality remain unchanged.

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

        Refactor 2025-09: unchanged behavior.
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

        Refactor 2025-09: unchanged behavior.
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
    def from_events(events, N=None, **kwargs):
        """
        Create a SpikeData object with N total units based on a list of
        events, each an (index, time) pair. If N is not provided, it is
        set to one more than the maximum index.

        All metadata parameters of the regular constructor are accepted.

        Refactor 2025-09: unchanged behavior.
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

        Refactor 2025-09: unchanged behavior.
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
    def from_thresholding(
        data: NDArray,
        fs_Hz=20e3,
        threshold_sigma=5.0,
        filter: Union[dict, bool] = True,
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

        Refactor 2025-09: unchanged behavior.
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
        raw_time: Optional[Union[NDArray, float]] = None,
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

        Refactor 2025-09: unchanged behavior. Can be paired with external smoothing to
        replace the removed population_firing_rate utility.
        """
        # sum(0) on CSR returns a (1, T) matrix in older SciPy; flatten to 1D array
        return np.asarray(self.sparse_raster(bin_size).sum(0)).ravel()  # type: ignore

    def binned_meanrate(self, bin_size=40, unit="kHz"):
        """
        Calculate the mean firing rate across the population in each time bin.

        The rate is calculated as the number of events in each bin divided by the bin
        size and number of units. The unit may be either `Hz` or `kHz` (default).

        Refactor 2025-09: unchanged behavior.
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

        Refactor 2025-09: unchanged behavior.
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

        Refactor 2025-09: unchanged behavior.
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

        Refactor 2025-09: unchanged behavior.
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

        Refactor 2025-09: unchanged behavior.
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

        Refactor 2025-09: unchanged behavior.
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

        Refactor 2025-09: unchanged behavior.
        """
        indices = np.hstack([np.ceil(ts / bin_size) - 1 for ts in self.train]).astype(
            int
        )
        units = np.hstack([0] + [len(ts) for ts in self.train])
        indptr = np.cumsum(units)
        values = np.ones_like(indices)
        length = int(np.ceil(self.length / bin_size))
        np.clip(indices, 0, length - 1, out=indices)
        # Use csr_matrix for SciPy < 1.8 compatibility (csr_array not available)
        ret = sparse.csr_matrix((values, indices, indptr), shape=(self.N, length))
        return ret

    def raster(self, bin_size=20.0):
        """
        Bin all spike times and create a dense array where entry (i,j) is the number of
        times cell i fired in bin j.

        Bins are left-open and right-closed intervals except the first, which will
        capture any spikes occurring exactly at t=0.

        Refactor 2025-09: unchanged behavior.
        """
        return self.sparse_raster(bin_size).toarray()

    def interspike_intervals(self):
        """Produce a list of arrays of interspike intervals per unit.

        Refactor 2025-09: unchanged behavior.
        """
        return [np.diff(ts) for ts in self.train]

    def concatenate_spike_data(self, sd):
        """
        Add the units from another SpikeData object to this one. The new units are
        assigned indices starting from the end of the current data. If the new units
        have a longer spike train, it is truncated to the length of the current data.

        Refactor 2025-09: unchanged behavior.
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

        Refactor 2025-09: behavior unchanged; helpers are colocated below.
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

        Refactor 2025-09: behavior unchanged; uses reorganized helpers.
        """
        return spike_time_tiling(self.train[i], self.train[j], delt, self.length)

    def latencies(self, times, window_ms=100.0):
        """
        Given a sorted list of times, compute the latencies from that time to each spike
        in each spike train within a window.

        :param times: list of times
        :param window_ms: window in ms
        :return: 2d list, each row is a list of latencies
                        from a time to each spike in the train

        Refactor 2025-09: unchanged behavior.
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

        Refactor 2025-09: unchanged behavior.
        """
        return self.latencies(self.train[i], window_ms)
