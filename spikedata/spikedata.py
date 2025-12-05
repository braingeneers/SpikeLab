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
from typing import Literal, Optional, Union, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal, sparse
from scipy.stats import norm

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
    trough_between,
)

__all__ = ["SpikeData", "spike_time_tiling", "swap", "randomize"]


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

    def neuron_to_channel_map(
        self, channel_attr: Optional[str] = None
    ) -> dict[int, int]:
        """
        Return a mapping from neuron indices to channel indices.

        Extracts channel information from neuron_attributes. If channel_attr is not
        specified, attempts to find channel information using common attribute names:
        'channel', 'channel_id', 'channel_index', 'ch', 'channel_idx'.

        Args:
            channel_attr: Optional name of the attribute in neuron_attributes that
                contains the channel index. If None, searches for common attribute names.

        Returns:
            dict mapping neuron index (int) to channel index (int). If neuron_attributes
            is None or no channel information is found, returns an empty dict.

        Raises:
            ValueError: If neuron_attributes is None and channel information is required,
                or if the specified channel_attr doesn't exist for all neurons.

        Example:
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class NeuronAttrs:
            ...     channel: int
            >>> attrs = [NeuronAttrs(channel=i % 4) for i in range(10)]
            >>> sd = SpikeData([[]] * 10, neuron_attributes=attrs, length=100.0)
            >>> mapping = sd.neuron_to_channel_map()
            >>> mapping[0]  # neuron 0 -> channel 0
            0
            >>> mapping[5]  # neuron 5 -> channel 1
            1
        """
        if self.neuron_attributes is None or self.N == 0:
            return {}

        # Common attribute names to try if channel_attr is not specified
        common_names = ["channel", "channel_id", "channel_index", "ch", "channel_idx"]

        # Determine which attribute to use
        attr_name = channel_attr
        if attr_name is None:
            # Try to find a channel attribute automatically
            for name in common_names:
                if hasattr(self.neuron_attributes[0], name):
                    attr_name = name
                    break
            if attr_name is None:
                return {}

        # Build the mapping
        mapping = {}
        _missing = object()
        for i in range(self.N):
            channel_val = getattr(self.neuron_attributes[i], attr_name, _missing)
            if channel_val is not _missing and channel_val is not None:
                mapping[i] = int(channel_val)

        return mapping

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
        # indices = np.hstack([np.ceil(ts / bin_size) - 1 for ts in self.train]).astype(
        #     int
        # )
        indices = np.hstack([np.floor(ts / bin_size) for ts in self.train]).astype(int)
        units = np.hstack([0] + [len(ts) for ts in self.train])
        indptr = np.cumsum(units)
        values = np.ones_like(indices)
        length = int(np.ceil(self.length / bin_size))
        if (self.length % bin_size) == 0:
            # if length is 40 and bin size is 10, you don't want 40 falling into bin that
            # should only contain times 30-39. The other bins only have 10 times considered in each
            # So its best to show this spike in a new bin
            length += 1

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

    def channel_raster(self, bin_size=20.0, channel_attr: Optional[str] = None):
        """
        Create a raster aggregated by channel instead of neuron.

        Returns a dense array where entry (c,j) is the total number of spikes from all
        neurons on channel c in bin j. Channels are determined from neuron_attributes
        using the same logic as neuron_to_channel_map().

        Args:
            bin_size: Bin size in milliseconds (same as raster()).
            channel_attr: Optional name of the attribute in neuron_attributes that
                contains the channel index. If None, searches for common attribute names.
                See neuron_to_channel_map() for details.

        Returns:
            numpy.ndarray of shape (n_channels, n_bins) where n_channels is the number
            of unique channels found.

        Raises:
            ValueError: If neuron_attributes is None or no channel information can be found.

        Example:
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class NeuronAttrs:
            ...     channel: int
            >>> # Create 6 neurons: 0,1 on channel 0; 2,3 on channel 1; 4,5 on channel 2
            >>> attrs = [NeuronAttrs(channel=i // 2) for i in range(6)]
            >>> trains = [[10.0, 20.0], [15.0], [25.0], [30.0], [35.0], [40.0]]
            >>> sd = SpikeData(trains, neuron_attributes=attrs, length=50.0)
            >>> ch_raster = sd.channel_raster(bin_size=10.0)
            >>> ch_raster.shape  # (3 channels, time bins)
            (3, ...)
            >>> ch_raster[0, :].sum()  # Channel 0 should have 3 spikes total
            3
        """
        # Get neuron-to-channel mapping
        neuron_to_channel = self.neuron_to_channel_map(channel_attr)
        if not neuron_to_channel:
            raise ValueError(
                "No channel information found in neuron_attributes. "
                "Ensure neuron_attributes contains channel information or specify channel_attr."
            )

        # Get the neuron raster
        neuron_raster = self.raster(bin_size)

        # Find unique channels and create reverse mapping (channel -> position)
        unique_channels = sorted(set(neuron_to_channel.values()))
        n_channels = len(unique_channels)
        n_bins = neuron_raster.shape[1]
        channel_to_pos = {ch: pos for pos, ch in enumerate(unique_channels)}

        # Initialize channel raster
        channel_raster = np.zeros((n_channels, n_bins), dtype=neuron_raster.dtype)

        # Aggregate spikes by channel
        for neuron_idx, channel_idx in neuron_to_channel.items():
            if neuron_idx < neuron_raster.shape[0]:
                channel_pos = channel_to_pos[channel_idx]
                channel_raster[channel_pos, :] += neuron_raster[neuron_idx, :]

        return channel_raster

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

    def get_frac_active(self, edges, MIN_SPIKES, backbone_threshold):
        """
        Inputs:

        t_spk_mat : numpy.ndarray
            Spike matrix of shape (N, T) where T is time bins and N is units
            This computed by turning self.train into sparse spike matrix via self.sparse_raster()
        edges : numpy.ndarray
            Array of shape (B, 2) containing [start, end] indices for each burst
        MIN_SPIKES : int
            Minimum number of spikes required for a unit to be considered active in a burst
        BACKBONE_THRESHOLD : float between 0-1
            Minimum fraction of bursts a unit must be active in to be considered a backbone unit

        Returns:

        frac_per_unit : numpy.ndarray
            - 1D array where each value represents a neuron and the fraction of burtsts that involve that neuron.
            - Example) A value of .75 in the array means Neuron A is active in 75% of bursts

        frac_per_burst : numpy.ndarray
            - 1D array where each value represents a burst and the fraction of neurons that are active
            in that burst.
            - Example) A value of .75 in the array means Burst A involves 75% of neurons.

        backbone_units : numpy.ndarray
            1D array of the neuron/unit indices that are backbone units.
        """
        t_spk_mat = self.sparse_raster(bin_size=1).toarray()

        # initiate result array
        spikes_per_burst = np.zeros((t_spk_mat.shape[0], edges.shape[0]))

        # for each unit
        for unit in range(t_spk_mat.shape[0]):

            # obtain spike time indices. +1 since these are 1 indexes
            unit_spk_times = np.where(t_spk_mat[unit, :])[0]

            # for each burst
            for burst in range(edges.shape[0]):

                # obtain all spike times within burst
                burst_times = unit_spk_times[
                    (unit_spk_times >= edges[burst, 0])
                    & (unit_spk_times <= edges[burst, 1])
                ]

                # store number of spikes in burst
                spikes_per_burst[unit, burst] = len(burst_times)

        # determine bursts above MIN_SPIKES
        above_thresh = spikes_per_burst >= MIN_SPIKES

        # compute fraction of bursts above threshold per unit
        frac_per_unit = np.sum(above_thresh, axis=1) / edges.shape[0]
        frac_per_burst = np.sum(above_thresh, axis=0) / t_spk_mat.shape[0]

        backbone_units = np.where(frac_per_unit >= backbone_threshold)[0]
        return frac_per_unit, frac_per_burst, backbone_units

    # ----------------------------
    # Exporters
    # ----------------------------

    def to_hdf5(
        self,
        filepath: str,
        *,
        style: "Literal['raster','ragged','group','paired']" = "ragged",
        raster_dataset: str = "raster",
        raster_bin_size_ms: Optional[float] = None,
        spike_times_dataset: str = "spike_times",
        spike_times_index_dataset: str = "spike_times_index",
        spike_times_unit: "Literal['ms','s','samples']" = "s",
        fs_Hz: Optional[float] = None,
        group_per_unit: str = "units",
        group_time_unit: "Literal['ms','s','samples']" = "s",
        idces_dataset: str = "idces",
        times_dataset: str = "times",
        times_unit: "Literal['ms','s','samples']" = "ms",
        raw_dataset: Optional[str] = None,
        raw_time_dataset: Optional[str] = None,
        raw_time_unit: "Literal['ms','s','samples']" = "ms",
    ) -> None:
        """Export this SpikeData to an HDF5 file with flexible formatting options.

        Supports four different storage styles to accommodate various analysis workflows:
        1. 'raster': Dense 2D array (units × time bins) for binned spike counts
        2. 'ragged': Flat spike times with index array (efficient for sparse data)
        3. 'group': Separate dataset per unit within a group (easy unit access)
        4. 'paired': Two parallel arrays of unit indices and spike times

        Args:
            filepath: Path to the output HDF5 file
            style: Storage format style. Defaults to 'ragged' for efficiency.

            # Raster style parameters
            raster_dataset: Dataset name for raster data (style='raster')
            raster_bin_size_ms: Bin size in milliseconds for rasterization.
                Required for 'raster' style.

            # Ragged style parameters
            spike_times_dataset: Dataset name for flat spike times (style='ragged')
            spike_times_index_dataset: Dataset name for cumulative spike counts
                per unit (style='ragged')
            spike_times_unit: Time unit for spike times in ragged format

            # Time conversion parameters
            fs_Hz: Sampling frequency in Hz, required when converting to 'samples' unit

            # Group style parameters
            group_per_unit: Group name containing per-unit datasets (style='group')
            group_time_unit: Time unit for individual unit datasets

            # Paired style parameters
            idces_dataset: Dataset name for unit indices (style='paired')
            times_dataset: Dataset name for spike times (style='paired')
            times_unit: Time unit for paired times

            # Optional raw data parameters (unused in current implementation)
            raw_dataset: Reserved for future raw data export
            raw_time_dataset: Reserved for future raw time axis export
            raw_time_unit: Time unit for raw data timestamps

        Note:
            All spike times are stored internally in milliseconds and converted
            to the requested output unit. When using 'samples' unit, fs_Hz must
            be provided for proper conversion.
        """
        # Import locally to avoid import cycles at module import time
        from data_loaders.data_exporters import export_spikedata_to_hdf5

        # Delegate to the standalone exporter function with all parameters
        export_spikedata_to_hdf5(
            self,
            filepath,
            style=style,  # type: ignore[arg-type]
            raster_dataset=raster_dataset,
            raster_bin_size_ms=raster_bin_size_ms,
            spike_times_dataset=spike_times_dataset,
            spike_times_index_dataset=spike_times_index_dataset,
            spike_times_unit=spike_times_unit,  # type: ignore[arg-type]
            fs_Hz=fs_Hz,
            group_per_unit=group_per_unit,
            group_time_unit=group_time_unit,  # type: ignore[arg-type]
            idces_dataset=idces_dataset,
            times_dataset=times_dataset,
            times_unit=times_unit,  # type: ignore[arg-type]
            raw_dataset=raw_dataset,
            raw_time_dataset=raw_time_dataset,
            raw_time_unit=raw_time_unit,  # type: ignore[arg-type]
        )

    def to_nwb(
        self,
        filepath: str,
        *,
        spike_times_dataset: str = "spike_times",
        spike_times_index_dataset: str = "spike_times_index",
        group: str = "units",
    ) -> None:
        """Export this SpikeData to a minimal NWB-like file using h5py.

        Creates an NWB-compatible file structure with spike times stored in the
        standard '/units' group format. This produces a minimal but valid NWB
        file that can be round-tripped with the NWB loader.

        Args:
            filepath: Path to the output NWB file (.nwb extension recommended)
            spike_times_dataset: Name of the dataset containing flattened spike
                times in seconds. Standard NWB uses "spike_times".
            spike_times_index_dataset: Name of the dataset containing cumulative
                spike counts per unit for indexing into spike_times. Standard
                NWB uses "spike_times_index".
            group: Name of the HDF5 group to contain the spike data. Standard
                NWB uses "units" for the units table.

        Note:
            - Spike times are automatically converted from internal milliseconds
              to seconds as required by the NWB standard
            - The output file contains only the essential spike timing data,
              not the full NWB metadata structure
            - Compatible with both pynwb and h5py-based NWB readers
        """
        # Import locally to avoid circular imports
        from data_loaders.data_exporters import export_spikedata_to_nwb

        # Delegate to the standalone NWB exporter
        export_spikedata_to_nwb(
            self,
            filepath,
            spike_times_dataset=spike_times_dataset,
            spike_times_index_dataset=spike_times_index_dataset,
            group=group,
        )

    def to_kilosort(
        self,
        folder: str,
        *,
        fs_Hz: float,
        spike_times_file: str = "spike_times.npy",
        spike_clusters_file: str = "spike_clusters.npy",
        time_unit: "Literal['samples','ms','s']" = "samples",
        cluster_ids: Optional[List[int]] = None,
    ) -> Tuple[str, str]:
        """Export this SpikeData to a KiloSort/Phy-like folder structure.

        Creates the standard KiloSort output format with two numpy arrays:
        - spike_times.npy: All spike times across units
        - spike_clusters.npy: Corresponding cluster/unit ID for each spike

        This format is compatible with Phy for manual curation and other
        spike sorting analysis tools.

        Args:
            folder: Output directory path. Will be created if it doesn't exist.
            fs_Hz: Sampling frequency in Hz. Required for time unit conversion,
                especially when using 'samples' (the KiloSort default).
            spike_times_file: Filename for the spike times array. Standard
                KiloSort uses "spike_times.npy".
            spike_clusters_file: Filename for the cluster assignments array.
                Standard KiloSort uses "spike_clusters.npy".
            time_unit: Output time unit for spike times:
                - 'samples': Sample indices (requires fs_Hz for conversion)
                - 'ms': Milliseconds (matches internal SpikeData format)
                - 's': Seconds
            cluster_ids: Optional list of cluster IDs to assign to each unit.
                Must have length equal to self.N. If None, uses sequential
                integers [0, 1, 2, ...].

        Returns:
            tuple[str, str]: Paths to the created (spike_times_file, spike_clusters_file)

        Raises:
            ValueError: If fs_Hz <= 0 or cluster_ids length doesn't match self.N

        Note:
            - Empty units (no spikes) are skipped in the output arrays
            - Cluster IDs are mapped to units in order, so cluster_ids[i]
              corresponds to unit i in the SpikeData
            - The 'samples' time unit is most common for KiloSort workflows
        """
        # Import locally to avoid circular imports
        from data_loaders.data_exporters import export_spikedata_to_kilosort

        # Delegate to the standalone KiloSort exporter and return file paths
        return export_spikedata_to_kilosort(
            self,
            folder,
            fs_Hz=fs_Hz,
            spike_times_file=spike_times_file,
            spike_clusters_file=spike_clusters_file,
            time_unit=time_unit,  # type: ignore[arg-type]
            cluster_ids=cluster_ids,
        )

    def get_pop_rate(self, square_width, gauss_sigma, raster_bin_size_ms=1.0):
        """
        Compute population firing rate by smoothing the summed spike counts using
        a moving-average (square) window, then a Gaussian smoothing window.

        Parameters:
        square_width (int): Width of square smoothing window in bins
        gauss_sigma (int): Sigma of Guassian smoothing window in bins
        raster_bin_size_ms (float): Size of raster bins in ms

        Returns:
        pop_rate (np.ndarray[float64]): Smoothed population spiking data in spikes per bin
        """
        t_spk_mat = self.sparse_raster(
            raster_bin_size_ms
        )  # Shape: (neurons, time_bins)
        summed_spikes = np.asarray(
            t_spk_mat.sum(axis=0)
        ).flatten()  # Sum once across neurons dimension

        # Pass square window
        if square_width > 0:
            square_smooth_summed_spike = np.convolve(
                summed_spikes,
                np.ones(square_width) / square_width,
                mode="same",
            )
        else:
            square_smooth_summed_spike = summed_spikes

        # Pass gaussian window
        if gauss_sigma > 0:
            gauss_window = norm.pdf(
                np.arange(-3 * gauss_sigma, 3 * gauss_sigma + 1), 0, gauss_sigma
            )
            pop_rate = np.convolve(
                square_smooth_summed_spike,
                gauss_window / np.sum(gauss_window),
                mode="same",
            )
        else:
            pop_rate = square_smooth_summed_spike

        return pop_rate

    def get_bursts(
        self,
        thr_burst,
        min_burst_diff,
        burst_edge_mult_thresh,
        square_width=100,
        gauss_sigma=20,
        acc_square_width=5,
        acc_gauss_sigma=5,
        raster_bin_size_ms=1.0,
        peak_to_trough=True,
        pop_rate=None,
        pop_rate_acc=None,
    ):
        """
        Detect bursts from a population rate vector using thresholded peak finding and
        amplitude-scaled edge detection.

        Parameters:
        thr_burst (float): Threshold multiplier for burst peak detection
        min_burst_diff (int): Minimum time between detected bursts (in bins)
        burst_edge_mult_thresh (float): Threshold multiplier for burst edge detection
        square_width (int): Square window width for calculating pop_rate (in bins)
        gauss_sigma (int): Gaussian window sigma for calculating pop_rate (in bins)
        acc_square_width (int): Square window width for calculating pop_rate_acc (in bins)
        acc_gauss_sigma (int): Gaussian window sigma for calculating pop_rate_acc (in bins)
        raster_bin_size_ms (int): Time bin size for calculating population rate in ms
        peak_to_trough (bool): Flag to calculate bursts peak-to-trough (True) or peak-to-zero (False)
        pop_rate (np.ndarray[float64], optional): Pre-calculated smoothed population spiking data in spikes per bin
        pop_rate_acc (np.ndarray[float64], optional): Pre-calculated accurate smoothed population spiking data in spikes per bin

        Returns:
        tburst (np.ndarray[float64]): Time bin indices of detected bursts
        edges (np.ndarray[float64]): Time bin indices of burst edges (Shape = (N,2))
        peak_amp (np.ndarray[float64]): Amplitudes of bursts at corresponding array indices

        Note:
        - Will use pop_rate and pop_rate_acc if provided, otherwise will calculate using squared widths and sigmas
        - Using the peak-to-zero calculations may result in several bursts being detected at one peak
        - In the case that duplicate bursts are detected, prints an error with potential fixes
        """
        # Get pop rates and rms
        if pop_rate is None:
            pop_rate = self.get_pop_rate(
                square_width, gauss_sigma, raster_bin_size_ms=raster_bin_size_ms
            )
        if pop_rate_acc is None:
            pop_rate_acc = self.get_pop_rate(
                acc_square_width, acc_gauss_sigma, raster_bin_size_ms=raster_bin_size_ms
            )
        pop_rms = np.sqrt(np.mean(np.square(pop_rate)))

        # Find peaks
        peaks, _ = signal.find_peaks(
            pop_rate, height=pop_rms * thr_burst, distance=min_burst_diff
        )
        peak_amp = pop_rate[peaks]

        edges = np.full((len(peaks), 2), np.nan)
        tburst = np.full(len(peaks), np.nan)

        for burst in range(len(peaks)):
            pk = int(peaks[burst])
            pk_val = float(pop_rate[pk])

            # Determine baseline
            if peak_to_trough:  # Peak-to-trough case
                # Find troughs to left and right
                tL = (
                    trough_between(peaks[burst - 1], pk, pop_rate)
                    if burst > 0
                    else None
                )
                tR = (
                    trough_between(pk, peaks[burst + 1], pop_rate)
                    if burst < len(peaks) - 1
                    else None
                )

                # If only one trough is found, use it
                if tL is None and tR is None:
                    continue
                elif tL is None:
                    ti_val = float(pop_rate[tR])
                elif tR is None:
                    ti_val = float(pop_rate[tL])
                # If two troughs are found, use higher one
                # This is expected except at the edges
                else:
                    tL_val = float(pop_rate[tL])
                    tR_val = float(pop_rate[tR])
                    ti_val = max(tL_val, tR_val)
            else:  # Peak-to-zero case
                ti_val = 0.0

            # Calculate edge threshold
            delta = max(0.0, pk_val - ti_val)
            edge_level = ti_val + burst_edge_mult_thresh * delta

            # Find edges where signal crosses threshold
            frames_below_thresh = np.where(pop_rate < edge_level)[0]
            rel_frames = pk - frames_below_thresh

            if (
                len(rel_frames) == 0
                or len(rel_frames[rel_frames > 0]) == 0
                or len(rel_frames[rel_frames < 0]) == 0
            ):
                continue

            rel_burst_start = np.min(rel_frames[rel_frames > 0])
            rel_burst_end = np.max(rel_frames[rel_frames < 0])

            edges[burst, :] = [
                peaks[burst] - rel_burst_start,
                peaks[burst] - rel_burst_end,
            ]

            # Refine peak location using accurate population rate
            if len(pop_rate_acc) == len(pop_rate):
                segment = pop_rate_acc[int(edges[burst, 0]) : int(edges[burst, 1])]
                acc_peak = np.argmax(segment)
                peak_val = np.max(segment)
                tburst[burst] = acc_peak + edges[burst, 0]
                peak_amp[burst] = peak_val
            else:
                tburst[burst] = peaks[burst]

        # Filter out invalid bursts
        edges = edges[~np.isnan(tburst), :]
        peak_amp = peak_amp[~np.isnan(tburst)]
        tburst = tburst[~np.isnan(tburst)]

        # Check for duplicate bursts
        unique_bursts, counts = np.unique(tburst, return_counts=True)
        duplicates = unique_bursts[counts > 1]
        if len(duplicates) != 0:
            print(
                f"\n{len(tburst) - len(unique_bursts)} duplicate bursts were detected across the following times: {list(duplicates)}.\n"
                + f"This is likely due identifying bursts using peak-to-zero calculations. Consider setting the PEAK-TO-TROUGH flag to True.\n"
                + f"Otherwise, consider increasing burst_edge_mult_thresh if this burst duration is longer than you would expect for your data.\n"
                + f"Alternatively, increase min_burst_diff to prevent two bursts from being detected too close to each other.\n"
            )

        return tburst, edges, peak_amp
