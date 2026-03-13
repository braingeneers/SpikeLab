"""
SpikeData core module.
"""

import heapq
import itertools
import warnings
from typing import Literal, Optional, Union, List, Tuple, Sequence
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal, sparse
from .ratedata import RateData
from .pairwise import PairwiseCompMatrix
from scipy.stats import norm


from .utils import (
    get_sttc,
    butter_filter,
    extract_waveforms,
    _sttc_ta,
    _sttc_na,
    _spike_time_tiling,
    _resampled_isi,
    _train_from_i_t_list,
    swap,
    randomize,
    trough_between,
    extract_unit_waveforms,
)

__all__ = [
    "SpikeData",
    "get_sttc",
    "swap",
    "randomize",
]


class SpikeData:
    """
    Class for handling and manipulating neuronal spike data with functionality
    for loading, processing, and analyzing spike data from different sources.


    Each instance of SpikeData has the following attributes:

    train: The main data attribute. This is a list of numpy arrays, where each array
      contains the spike times for a particular neuron.

    N: The number of neurons in the dataset.

    length: The length of the spike train, defaults to the time of the last spike.

    neuron_attributes: A list of dictionaries containing information on each neuron.

    metadata: A dictionary containing any additional information or metadata about the
      spike data.

    raw_data: If provided, this numpy array contains the raw time series data.

    raw_time: This is either a numpy array of sample times, or a single float
      representing a sample rate in kHz.

    """

    @staticmethod
    def from_idces_times(idces, times, N=None, **kwargs):
        """
        Create a SpikeData object based on a list of N units indices and
        spike times.

        Parameters:
        idces (list): List of unit indices
        times (list): List of spike times
        N (int): Number of units (optional)
        **kwargs: Additional keyword arguments for the SpikeData constructor

        Returns:
        spike_data (SpikeData): A new SpikeData object with the given unit indices and spike times.

        Notes:
        - This method is a wrapper around the _train_from_i_t_list helper function.
        """
        return SpikeData(_train_from_i_t_list(idces, times, N), N=N, **kwargs)

    @staticmethod
    def from_raster(raster, bin_size_ms, **kwargs):
        """
        Create a SpikeData object based on a spike raster matrix with shape (N [units], T [time bins])

        Parameters:
        raster (numpy.ndarray): Spike raster matrix with shape (N [units], T [time bins])
        bin_size_ms (float): Size of each time bin in milliseconds
        **kwargs: Additional keyword arguments for the SpikeData constructor

        Returns:
        sd (SpikeData): Object with the given spike raster.

        Notes:
        - The generated spike times are evenly spaced within each time bin. For example, if a unit fires
            3 times in a 10 ms time bin, those events go at 2.5, 5, and 7.5 ms after the start of the bin.
        - All metadata parameters of the regular constructor are accepted.
        """
        raster = raster.astype(int)
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
        Create a SpikeData object based on a list of (unit index, time) pairs.

        Parameters:
        events (list): List of (index, time) pairs
        N (int): Number of units (default: maximum index in the events)
        **kwargs: Additional keyword arguments for the SpikeData constructor

        Returns:
        sd (SpikeData): Object with the given events.

        Notes:
        - This method is a wrapper around the from_idces_times helper function.
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
        Create a SpikeData object from a list of neo.SpikeTrain objects.
        Parameters:
        spiketrains (list): List of neo.SpikeTrain objects
        **kwargs: Additional keyword arguments for the SpikeData constructor

        Returns:
        sd (SpikeData): Object with the given spike trains in milliseconds.
        """
        trains = [st.copy() for st in spiketrains]
        for st in trains:
            st.units = "ms"

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

        Parameters:
        data (numpy.ndarray): Raw data with shape (channels, time)
        fs_Hz (float): Sampling frequency in Hz
        threshold_sigma (float): Threshold in units of per-channel standard deviation
        filter (dict or bool): Filter configuration or False to disable filtering; if True, a third-order Butterworth filter with passband 300 Hz to 6 kHz is used.
        hysteresis (bool): Use hysteresis for thresholding
        direction (str): Direction of thresholding ('both', 'up', 'down')

        Returns:
        sd (SpikeData): Object with the given raw data.

        Notes:
        -  To use different filter parameters, pass a dictionary, which will be passed as keyword arguments to butter_filter().
        -  If filter is False, no filtering is done.
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


        Parameters:
        train (list): List of spike trains, each a list of spike times in milliseconds
        N (int): Number of units (optional)
        length (float): Length of the spike train in milliseconds (optional)
        neuron_attributes (list): List of neuron attributes (optional)
        metadata (dict): Dictionary of metadata (optional)
        raw_data (numpy.ndarray): Raw timeseries data with shape (channels, time) (optional)
        raw_time (numpy.ndarray or float): Raw time vector with shape (time) or sample rate in kHz (optional)


        Notes:
        - Arbitrary raw timeseries data, not associated with particular units,
        can be passed in as `raw_data` (an array with shape (channels, time)).
        - The `raw_time` argument can also be a sample rate in kHz, in which case it is generated
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
        # storing None so we don't break concatenation semantics.
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
        """Iterate spike times for all units in time order."""
        return heapq.merge(*self.train)

    @property
    def events(self):
        """Iterate (index,time) pairs for all units in time order."""
        return heapq.merge(
            *[zip(itertools.repeat(i), t) for (i, t) in enumerate(self.train)],
            key=lambda x: x[1],
        )

    def idces_times(self):
        """
        Generate a matched pair of numpy arrays containing unit indices and times for
        all events.

        Returns:
        idces (numpy.ndarray): Array of unit indices
        times (numpy.ndarray): Array of times for all events.

        Notes:
        - This method is not a property unlike `times` and `events` because the lists must
        actually be constructed in memory.
        """
        idces, times = [], []
        for i, t in self.events:
            idces.append(i)
            times.append(t)
        return np.array(idces), np.array(times)

    @property
    def unit_locations(self) -> Optional[np.ndarray]:
        """
        Get unit locations as an (U, D) array where D is the spatial dimension.


        Returns:
        locations (numpy.ndarray): Array of unit locations, shape (N, D)
        - None if any unit lacks location data

        Notes:
        - Extracts from neuron_attributes 'location', 'x'/'y'/'z' (x and y required), or 'position' keys.
        """
        if self.neuron_attributes is None:
            return None

        locations = []
        for attr in self.neuron_attributes:
            if "location" in attr:
                locations.append(np.asarray(attr["location"]))
            elif "x" in attr and "y" in attr:
                loc = [attr["x"], attr["y"]]
                if "z" in attr:
                    loc.append(attr["z"])
                locations.append(np.asarray(loc))
            elif "position" in attr:
                locations.append(np.asarray(attr["position"]))
            else:
                return None  # Missing location for at least one unit

        if not locations:
            return None
        return np.array(locations)

    @property
    def electrodes(self) -> Optional[np.ndarray]:
        """
        Get electrode/channel indices for each unit as a 1D array.

        Extracts from neuron_attributes 'electrode', 'channel', or 'ch' keys.
        Returns None if any unit lacks electrode data.
        """
        if self.neuron_attributes is None:
            return None

        electrodes = []
        for attr in self.neuron_attributes:
            if "electrode" in attr:
                electrodes.append(attr["electrode"])
            elif "channel" in attr:
                electrodes.append(attr["channel"])
            elif "ch" in attr:
                electrodes.append(attr["ch"])
            else:
                return None  # Missing electrode for at least one unit

        if not electrodes:
            return None
        return np.array(electrodes)

    def frames(self, length, overlap=0):
        """
        Split the recording into a SpikeSliceStack of fixed-length windows.

        Parameters:
            length (float): Length of each window in milliseconds.
            overlap (float): Overlap between consecutive windows in milliseconds. Default 0.

        Returns:
            stack (SpikeSliceStack): Stack of SpikeData windows, one per frame.

        Notes:
            - Windows that would extend past the end of the recording are excluded.
            - overlap must be strictly less than length.
        """
        from .spikeslicestack import SpikeSliceStack

        step = length - overlap
        if step <= 0:
            raise ValueError("overlap must be less than length")
        times = [
            (float(start), float(start) + length)
            for start in np.arange(0, self.length - length + 1e-9, step)
        ]
        if not times:
            raise ValueError(
                f"Recording length ({self.length} ms) is shorter than frame length ({length} ms)"
            )
        return SpikeSliceStack(self, times_start_to_end=times)

    def binned(self, bin_size=40.0):
        """
        Quantize time into intervals of bin_size and counts the number of events in
        each bin, considered as a lower half-open interval of times, with the exception
        that events at time precisely zero will be included in the first bin.

        Parameters:
        bin_size (float): Size of the time bin in milliseconds

        Returns:
        binned_raster (numpy.ndarray): Array of the number of events in each bin.
        """
        # sum(0) on CSR returns a (1, T) matrix in older SciPy; flatten to 1D array
        return np.asarray(self.sparse_raster(bin_size).sum(0)).ravel()  # type: ignore

    def binned_meanrate(self, bin_size=40, unit="kHz"):
        """
        Calculate the mean firing rate across the population in each time bin.

        Parameters:
        bin_size (float): Size of the time bin in milliseconds
        unit (str): Unit of the firing rate ('Hz' or 'kHz')

        Returns:
        binned_meanrate (numpy.ndarray): Array of the mean firing rate in each bin.

        Notes:
        - The rate is calculated as the number of events in each bin divided by the bin size and number of units.
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
        time over the length of the data.

        Parameters:
        unit (str): Unit of the firing rate ('Hz' or 'kHz')

        Returns:
        rates (numpy.ndarray): Array of the firing rate of each neuron.
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
        Calculate instantaneous firing rate of each unit at the given times by calculating the
        interspike intervals and interpolating their inverse.

        Parameters:
        times (numpy.ndarray): Array of times to resample the firing rate to
        sigma_ms (float): Standard deviation of the Gaussian kernel in milliseconds

        Returns:
        rates (numpy.ndarray): Array of the resampled firing rate.
        """
        return np.array([_resampled_isi(t, times, sigma_ms) for t in self.train])

    def set_neuron_attribute(self, key: str, values, neuron_indices=None):
        """
        Set an attribute: a list of dictionaries, each with a key/value pair with length equal to the number of neurons.

        Parameters:
        key (str):  Name of a particular attribute.
        values (single value or list): Single value (applied to all) or list/array matching neuron_indices length for each neuron.
        neuron_indices (list): Neurons to update. If None, updates all.
        """
        if self.neuron_attributes is None:
            self.neuron_attributes = [{} for _ in range(self.N)]
        indices = range(self.N) if neuron_indices is None else neuron_indices
        if hasattr(values, "__len__") and not isinstance(values, str):
            indices = list(indices)
            if len(values) != len(indices):
                raise ValueError(
                    f"values length {len(values)} != indices length {len(indices)}"
                )
            for i, val in zip(indices, values):
                self.neuron_attributes[i][key] = val
        else:
            for i in indices:
                self.neuron_attributes[i][key] = values

    def get_neuron_attribute(self, key: str, default=None):
        """
        Get an attribute across all neurons.

        Parameters:
            key: Attribute name.
            default: Value if neuron lacks the attribute.

        Returns:
            List of values, one per neuron.
        """
        if self.neuron_attributes is None:
            return [default] * self.N
        return [attr.get(key, default) for attr in self.neuron_attributes]

    def subset(self, units, by=None):
        """
        Return a new SpikeData with spike times for only some units, selected either by
        their indices or by an ID stored under a given key in the neuron_attributes. Index-based if by = None.

        Parameters:
        units (list): List of unit indices to select
        by (str): Key to select units by in the neuron_attributes. Index-based if by = None.

        Returns:
        sd (SpikeData): New SpikeData object with the selected units.

        Notes:
        - Units are included in the output according to their order in self.train, not the
        order in the unit list (which is treated as a set).
        - If IDs are not unique, every neuron which matches is included in the output.
        - Neurons whose neuron_attributes entry does not have the key are always excluded.
        """
        if isinstance(units, int):
            units = [units]
        # For case where user inputs a single string for units when using by option
        if isinstance(units, str):
            units = [units]
        units = set(units)
        if by is not None:
            if self.neuron_attributes is None:
                raise ValueError("can't use `by` without `neuron_attributes`")
            _missing = object()
            units = {
                i
                for i in range(self.N)
                if self.neuron_attributes[i].get(by, _missing) in units
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

        Parameters:
            channel_attr: Optional name of the attribute in neuron_attributes that
                contains the channel index. If None, searches for common attribute names.

        Returns:
        mapping (dict): Mapping from neuron index (int) to channel index (int). Returns an empty dict if neuron_attributes is None.

        Notes:
        - If neuron_attributes is None and channel information is required,
        or if the specified channel_attr doesn't exist for all neurons, a ValueError is raised.
        - If channel_attr is not specified, attempts to find channel information using common attribute names:
        'channel', 'channel_id', 'channel_index', 'ch', 'channel_idx'.
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
                if name in self.neuron_attributes[0]:
                    attr_name = name
                    break
            if attr_name is None:
                return {}

        # Build the mapping
        mapping = {}
        _missing = object()
        for i in range(self.N):
            channel_val = self.neuron_attributes[i].get(attr_name, _missing)
            if channel_val is not _missing and channel_val is not None:
                mapping[i] = int(channel_val)

        return mapping

    def subtime(self, start, end, shift_time=True):
        """
        Extract a subset of time points from spikedata using time values.

        Parameters:
        start (int/float): Starting time value (inclusive)
        end (int/float): Ending time value (exclusive)
        shift_time (bool): If True, this will make the new output spike data object where the times are shifted so
                           relative to 0 (input start time becomes 0 for new spikedata)
                           If False, preserve original time values (spikes retain their original timestamps).
                           Example) shift_time=False
                                        subtime(1.0, 4.0, shift_time=False)
                                        Result: train[0] = [1.2, 2.3, 3.7]  Original timestamps preserved.
                                    shift_time=True (default)
                                        subtime(1.0, 4.0, shift_time=True)
                                        Result: train[0] = [0.2, 1.3, 2.7]  Shifted by -1.0 (the start value)

        Returns:
        SpikeData: New SpikeData object containing only the specified time range

        Notes:
        - Start and end can be negative, in which case they are counted backwards from the
        end.
        - They can also be None or Ellipsis, in which case that end of the data is
        not truncated.
        - All metadata and neuron data are propagated
        """
        if start is None or start is Ellipsis:
            start = 0
        elif start < 0:
            start += self.length
            if start < 0:
                raise ValueError(
                    f"start ({start - self.length}) is too negative. "
                    f"Minimum allowed is -{self.length} (recording length)"
                )
        elif start > self.length:
            start = self.length

        if end is None or end is Ellipsis:
            end = self.length
        elif end < 0:
            end += self.length
        elif end > self.length:
            end = self.length

        if start >= end:
            raise ValueError(
                f"start ({start}) must be less than end ({end}). "
                f"Cannot create subtime with invalid range."
            )

        time_shift = start if shift_time else 0

        # Subset the spike train by time
        train = [t[(t >= start) & (t < end)] - time_shift for t in self.train]

        # Subset and propagate the raw data
        rawmask = (self.raw_time >= start) & (self.raw_time < end)

        return SpikeData(
            train,
            length=end - start,
            N=self.N,
            neuron_attributes=self.neuron_attributes,
            metadata=self.metadata,
            raw_time=self.raw_time[rawmask] - time_shift,
            raw_data=self.raw_data[..., rawmask],
        )

    def __getitem__(self, key):
        """
        If a slice is provided, it is interpreted as a time range and handled using self.subtime().
        If an iterable is provided, it is interpreted as a list of neuron indices and handled
        using self.subset().

        Parameters:
        key (slice or iterable): Slice or iterable of neuron indices to select

        Returns:
        sd (SpikeData): New SpikeData object with the selected units.
        """
        if isinstance(key, slice):
            return self.subtime(key.start, key.stop)
        else:
            return self.subset(key)

    def append(self, spikeData, offset=0):
        """
        Append the spike times from another SpikeData object to this one, optionally
        offsetting them by a given amount from the end of the current data.

        Parameters:
        spikeData (SpikeData): SpikeData object to append
        offset (float): Offset in milliseconds to add to the spike times of the appended data

        Returns:
        sd (SpikeData): New SpikeData object with the appended data.

        Notes:
        - The two SpikeData objects must have the same number of neurons.
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
            },
        )

    def sparse_raster(self, bin_size=20.0):
        """
        Bin all spike times and create a sparse array where entry (i,j) is the number of
        times unit i fired in bin j.

        Parameters:
        bin_size (float): Size of the time bin in milliseconds

        Returns:
        raster (sparse.csr_matrix): Sparse array where entry (i,j) is the number of
        times unit i fired in bin j.

        Notes:
        - Bins are left-open and right-closed intervals except the first, which will
        capture any spikes occurring exactly at t=0.
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
        return sparse.csr_matrix((values, indices, indptr), shape=(self.N, length))

    def raster(self, bin_size=20.0):
        """
        Bin all spike times and create a dense array where entry (i,j) is the number of
        times cell i fired in bin j.

        Parameters:
        bin_size (float): Size of the time bin in milliseconds

        Returns:
        raster (numpy.ndarray): Dense array where entry (i,j) is the number of
        times unit i fired in bin j.

        Notes:
        - Bins are left-open and right-closed intervals except the first, which will
        capture any spikes occurring exactly at t=0.
        """
        return self.sparse_raster(bin_size).toarray()

    def channel_raster(self, bin_size=20.0, channel_attr: Optional[str] = None):
        """
        Create a raster aggregated by channel instead of neuron.

        Parameters:
        bin_size (float): Size of the time bin in milliseconds
        channel_attr (str): Name of the attribute in neuron_attributes that
        contains the channel index. If None, searches for common attribute names.

        Returns:
        channel_raster (numpy.ndarray): Dense array where entry (c,j) is the total number of spikes from all
        neurons on channel c in bin j.

        Notes:
        - Channels are determined from neuron_attributes using the same logic as neuron_to_channel_map().
        - If neuron_attributes is None or no channel information can be found, a ValueError is raised.
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

    def get_waveform_traces(
        self,
        unit: Optional[Union[int, slice, Sequence[int]]] = None,
        ms_before: float = 1.0,
        ms_after: float = 2.0,
        channels: Optional[Union[int, List[int]]] = None,
        bandpass: Optional[tuple] = None,
        filter_order: int = 3,
        store: bool = True,
        return_channel_waveforms: bool = False,
        return_avg_waveform: bool = True,
    ) -> Tuple[Union[np.ndarray, List[np.ndarray]], Dict[str, Any]]:
        """
        Extract raw voltage waveforms around spike times from raw data.

        Parameters:
            unit: Unit index selection.
                - int: extract a single unit (returns a single waveform array)
                - slice / list-like / array-like / range: extract a subset (returns a list of waveform arrays)
                - None: extract all units (returns a list of waveform arrays)
            ms_before: Milliseconds before each spike time (default: 1.0).
            ms_after: Milliseconds after each spike time (default: 2.0).
            channels: Channel(s) to extract. None uses neuron_to_channel_map or all
                channels; int for single channel; list for multiple; [] for mapped channel.
            bandpass: Optional (lowcut_Hz, highcut_Hz) for bandpass filtering.
            filter_order: Butterworth filter order (default: 3).
            store: If True (default), stores waveforms and avg_waveform in neuron_attributes.
            return_channel_waveforms: If True, include a per-channel dict
                `channel_waveforms[channel] -> (num_samples, num_spikes)` in the return.
            return_avg_waveform: If False, skip computing/returning `avg_waveform` (it will be None).

        Returns:
            (waveforms, meta)

            - waveforms:
                - If unit is an int: a single 3D array shaped (num_channels, num_samples, num_spikes)
                - Otherwise: a list of 3D arrays, one per selected unit
            - meta (dict):
                - fs_kHz: sampling rate in kHz
                - unit_indices: list of unit indices corresponding to the returned waveforms
                - channels: list of List[int], channels used per unit
                - spike_times_ms: list of np.ndarray, valid spike times per unit
                - avg_waveforms: optional list of 2D arrays (num_channels, num_samples) per unit
                - channel_waveforms: optional list of dicts (channel -> (num_samples, num_spikes)) per unit
        """
        # Validate that raw voltage data exists
        if self.raw_data.size == 0:
            raise ValueError("raw_data is empty")

        # If raw_time is a scalar, it's the sampling rate (kHz) directly, otherwise compute rate from median time delta
        if np.ndim(self.raw_time) == 0 or self.raw_time.shape == ():
            fs_kHz = float(self.raw_time)
        else:
            fs_kHz = 1.0 / np.median(np.diff(self.raw_time))

        # Get mapping of neuron indices to their recording channels using extract_unit_waveforms to determine default channels per unit
        neuron_to_channel = self.neuron_to_channel_map()

        # Normalize `unit` into an explicit list of indices to extract, while preserving
        # the historical behavior that passing a single int returns a single dict.
        return_single = False
        if unit is None:
            unit_indices = list(range(self.N))
        elif isinstance(unit, (int, np.integer)):
            u = int(unit)
            if u < 0 or u >= self.N:
                raise ValueError(f"Unit index {u} out of range (0 to {self.N - 1})")
            unit_indices = [u]
            return_single = True
        elif isinstance(unit, slice):
            unit_indices = list(range(self.N)[unit])
        else:
            try:
                unit_indices = [int(u) for u in unit]  # type: ignore[iteration-over-optional]
            except TypeError as e:
                raise ValueError(
                    "unit must be an int, slice, or sequence of ints (or None)"
                ) from e
            for u in unit_indices:
                if u < 0 or u >= self.N:
                    raise ValueError(f"Unit index {u} out of range (0 to {self.N - 1})")

        # Extract for each selected unit, optionally store, return (waveforms, meta).
        waveforms_out: List[np.ndarray] = []
        channels_out: List[List[int]] = []
        spike_times_out: List[np.ndarray] = []
        avg_waveforms_out: Optional[List[np.ndarray]] = (
            [] if return_avg_waveform else None
        )
        channel_waveforms_out: Optional[List[dict]] = (
            [] if return_channel_waveforms else None
        )

        for unit_idx in unit_indices:
            spike_times_ms = np.asarray(self.train[unit_idx])
            waveforms, unit_meta = extract_unit_waveforms(
                unit_idx=unit_idx,
                spike_times_ms=spike_times_ms,
                raw_data=self.raw_data,
                fs_kHz=fs_kHz,
                ms_before=ms_before,
                ms_after=ms_after,
                channels=channels,
                neuron_to_channel=neuron_to_channel,
                bandpass=bandpass,
                filter_order=filter_order,
                return_channel_waveforms=return_channel_waveforms,
                return_avg_waveform=return_avg_waveform,
            )
            if store and self.neuron_attributes is not None:
                self.neuron_attributes[unit_idx]["waveforms"] = waveforms
                if return_avg_waveform:
                    self.neuron_attributes[unit_idx]["avg_waveform"] = unit_meta[
                        "avg_waveform"
                    ]
                # Store per-unit trace metadata (kept out of the return payload).
                # This is useful for downstream analysis without duplicating it per-result dict.
                self.neuron_attributes[unit_idx]["traces_meta"] = {
                    "fs_kHz": fs_kHz,
                    "ms_before": ms_before,
                    "ms_after": ms_after,
                    "bandpass": bandpass,
                    "filter_order": filter_order,
                    "channels": unit_meta["channels"],
                    "spike_times_ms": unit_meta["spike_times_ms"],
                }

            waveforms_out.append(waveforms)
            channels_out.append(unit_meta["channels"])
            spike_times_out.append(unit_meta["spike_times_ms"])
            if return_avg_waveform and avg_waveforms_out is not None:
                avg_waveforms_out.append(unit_meta["avg_waveform"])
            if return_channel_waveforms and channel_waveforms_out is not None:
                channel_waveforms_out.append(unit_meta["channel_waveforms"])

        meta: Dict[str, Any] = {
            "fs_kHz": fs_kHz,
            "unit_indices": unit_indices,
            "channels": channels_out,
            "spike_times_ms": spike_times_out,
        }
        if return_avg_waveform and avg_waveforms_out is not None:
            # Always return as a list for consistency (one element per unit)
            meta["avg_waveforms"] = [
                np.asarray(a).reshape(a.shape[0], -1) for a in avg_waveforms_out
            ]
        if return_channel_waveforms:
            meta["channel_waveforms"] = channel_waveforms_out

        return (
            (waveforms_out[0] if return_single else waveforms_out),
            meta,
        )

    def interspike_intervals(self):
        """
        Produce a list of arrays of interspike intervals per unit.

        Formula:
        ISI = t_n - t_(n-1)

        Returns:
        isis (list): List of arrays of interspike intervals per unit.
        """
        return [np.diff(ts) for ts in self.train]

    def concatenate_spike_data(self, sd):
        """
        Add the units from another SpikeData object to this one.

        Parameters:
        sd (SpikeData): SpikeData object to append

        Returns:
        sd (SpikeData): New SpikeData object with the appended data.

        Notes:
        - New units are assigned indices starting from the end of the current data.
        - If the new units have a longer spike train, it is truncated to the length of the current data.
        """

        # Subtime the second SpikeData object to the length of the first
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
        Compute the spike time tiling coefficient matrix.

        Parameters:
        delt (float): Time window in milliseconds (default: 20.0)

        Returns:
        ret (numpy.ndarray): Spike time tiling coefficient matrix.

        [1] Cutts & Eglen. Detecting pairwise correlations in spike trains: An objective
            comparison of methods and application to the study of retinal waves. Journal of
            Neuroscience 34:43, 14288–14303 (2014).
        """
        T = self.length
        ts = [_sttc_ta(ts, delt, T) / T for ts in self.train]

        ret = np.diag(np.ones(self.N))
        for i in range(self.N):
            for j in range(i + 1, self.N):
                ret[i, j] = ret[j, i] = _spike_time_tiling(
                    self.train[i], self.train[j], ts[i], ts[j], delt
                )
        return PairwiseCompMatrix(matrix=ret, metadata={"delt": delt})

    def spike_time_tiling(self, i, j, delt=20.0):
        """
        Calculate the spike time tiling coefficient between two units within this SpikeData.

        Parameters:
        i (int): Index of the first unit
        j (int): Index of the second unit
        delt (float): Time window in milliseconds (default: 20.0)

        Returns:
        ret (float): Spike time tiling coefficient between the two units.

        [1] Cutts & Eglen. Detecting pairwise correlations in spike trains: An objective
            comparison of methods and application to the study of retinal waves. Journal of
            Neuroscience 34:43, 14288–14303 (2014).
        """
        return get_sttc(self.train[i], self.train[j], delt, self.length)

    def latencies(self, times, window_ms=100.0):
        """
        Given a sorted list of times, compute the latencies from that time to each spike
        in each spike train within a window.

        Parameters:
        times (list): List of times
        window_ms (float): Window in milliseconds (default: 100.0)

        Returns:
        latencies (list): 2d list, each row is a list of latencies from a time to each spike in the train
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

        Parameters:
        i (int): Index of the unit
        window_ms (float): Window in milliseconds (default: 100.0)

        Returns:
        latencies (list): 2d list, each row is a list of latencies per neuron
        """
        return self.latencies(self.train[i], window_ms)

    def get_frac_active(self, edges, MIN_SPIKES, backbone_threshold):
        """
        Computes fraction of total units active in per burst, fraction of bursts
        in which each unit is active and assigns backbone identity based on fraction of active bursts

        Parameters:
        edges (numpy.ndarray): Array of shape (B, 2) containing [start, end] indices for each burst
        MIN_SPIKES (int): Minimum number of spikes required for a unit to be considered active in a burst
        backbone_threshold (float [0, 1]): Minimum fraction of bursts a unit must be active in to be considered a backbone unit

        Returns:
        frac_per_unit (numpy.ndarray): 1D array where each value represents a neuron and the fraction of bursts in which the neuron was active
        frac_per_burst (numpy.ndarray): 1D array where each value represents a burst and the fraction of neurons that are active in that burst.
        backbone_units (numpy.ndarray): 1D array of the neuron/unit indices that are backbone units.
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

    def spike_shuffle(self, swap_per_spike=5, seed=None, bin_size=1):
        """
        Shuffles the underlying spike matrix of a SpikeData object using degree-preserving double-edge swaps.

        Parameters:
        -----------
        - swap_per_spike (int): Determines total number of swaps: num_spikes * swap_per_spike (optional, default=5).
        - seed (int): Set the random seed number for repeatability of results, None means no seed is set (optional, default=None).
        - bin_size(int): The number of individual time steps per bin. If bin_size > 1 (not recommended), bins with multiple
                       spikes are binarized to 1. In other words, the number of spikes within a bin is NOT preserved (optional, default=1).
        Returns:
        --------
        - shuffled_spike_data (SpikeData): SpikeData object where the underlying spike train matrix is now shuffled.

        Notes:
        -----
        - Shuffling is done in a manner where each neuron's average firing rate is preserved, but the specific time_bin in it spikes is shuffled.
        - Shuffling is done in a manner where each time bin's population rate is preserved, but the specific units active in each time bin are shuffled.
        - Ever spike swap involves 2 different spikes so on average, ever spike will get swapped 2*swap_per_spike times

        Ref:
        ----
        - Okun, M. et al. Population rate dynamics and multineuron firing patterns in sensory cortex. J. Neurosci. 32, 17108–17119 (2012)
        """
        spk_mat = self.sparse_raster(bin_size=bin_size).toarray()
        if bin_size != 1:
            binary_mat = spk_mat > 0
        else:
            binary_mat = spk_mat
        shuffled_mat = randomize(binary_mat, swap_per_spike=swap_per_spike, seed=seed)
        shuffled_spike_data = SpikeData.from_raster(
            shuffled_mat,
            bin_size,
            length=self.length,
            metadata=self.metadata,
            neuron_attributes=self.neuron_attributes,
        )
        return shuffled_spike_data

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
        """
        Export this SpikeData to an HDF5 file with flexible formatting options.

        Supports four different storage styles to accommodate various analysis workflows:
        1. 'raster': Dense 2D array (units × time bins) for binned spike counts
        2. 'ragged': Flat spike times with index array (efficient for sparse data)
        3. 'group': Separate dataset per unit within a group (easy unit access)
        4. 'paired': Two parallel arrays of unit indices and spike times

        Parameters:

        filepath (str): Path to the output HDF5 file
        style (Literal['raster','ragged','group','paired']): Storage format style. Defaults to 'ragged' for efficiency.

        # Raster style parameters
        raster_dataset (str): Dataset name for raster data (style='raster')
        raster_bin_size_ms (float): Bin size in milliseconds for rasterization. Required for 'raster' style.

        # Ragged style parameters
        spike_times_dataset (str): Dataset name for flat spike times (style='ragged')
        spike_times_index_dataset (str): Dataset name for cumulative spike counts per unit (style='ragged')
        spike_times_unit (Literal['ms','s','samples']): Time unit for spike times in ragged format

        # Time conversion parameters
        fs_Hz (float): Sampling frequency in Hz, required when converting to 'samples' unit

        # Group style parameters
        group_per_unit (str): Group name containing per-unit datasets (style='group')
        group_time_unit (Literal['ms','s','samples']): Time unit for individual unit datasets

        # Paired style parameters
        idces_dataset (str): Dataset name for unit indices (style='paired')
        times_dataset (str): Dataset name for spike times (style='paired')
        times_unit (Literal['ms','s','samples']): Time unit for paired times

        # Optional raw data parameters (unused in current implementation)
        raw_dataset (str): Reserved for future raw data export
        raw_time_dataset (str): Reserved for future raw time axis export
        raw_time_unit (Literal['ms','s','samples']): Time unit for raw data timestamps

        Notes:
        - All spike times are stored internally in milliseconds and converted to the requested output unit.
        - When using 'samples' unit, fs_Hz must be provided for proper conversion.
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
        """
        Export this SpikeData to a minimal NWB-compatible file using h5py, storing spike times in the standard '/units' group format for round-tripping with the NWB loader.

        Parameters:
        filepath (str): Path to the output NWB file (.nwb extension recommended)
        spike_times_dataset (str): Name of the dataset containing flattened spike times in seconds. Standard NWB uses "spike_times".
        spike_times_index_dataset (str): Name of the dataset containing cumulative spike counts per unit for indexing into spike_times. Standard NWB uses "spike_times_index".
        group (str): Name of the HDF5 group to contain the spike data. Standard NWB uses "units" for the units table.

        Notes:
        - Spike times are automatically converted from internal milliseconds to seconds as required by the NWB standard
        - The output file contains only the essential spike timing data, not the full NWB metadata structure
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
        """
        Export this SpikeData to a KiloSort/Phy-compatible folder structure with spike_times.npy and spike_clusters.npy arrays for use with Phy.

        Parameters:
        folder (str): Output directory path. Will be created if it doesn't exist.
        fs_Hz (float): Sampling frequency in Hz. Required for time unit conversion, especially when using 'samples' (the KiloSort default).
        spike_times_file (str): Filename for the spike times array. Standard KiloSort uses "spike_times.npy".
        spike_clusters_file (str): Filename for the cluster assignments array. Standard KiloSort uses "spike_clusters.npy".
        time_unit (Literal['samples','ms','s']): Output time unit for spike times.
        cluster_ids (Optional[List[int]]): Optional list of cluster IDs to assign to each unit. Must have length equal to self.N. If None, uses sequential integers [0, 1, 2, ...].

        Returns:
        tuple[str, str]: Paths to the created (spike_times_file, spike_clusters_file)

        Notes:
        - Empty units (no spikes) are skipped in the output arrays
        - Cluster IDs are mapped to units in order, so cluster_ids[i] corresponds to unit i in the SpikeData
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

    def get_pop_rate(self, square_width=20, gauss_sigma=100, raster_bin_size_ms=1.0):
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

    def plot_raster(
        self,
        time_range: Optional[Tuple[float, float]] = None,
        raster_bin_size_ms: float = 1.0,
        fr_rates: Optional[Union[np.ndarray, "RateData"]] = None,
        fr_rate_bin_ms: Optional[float] = None,
        pop_rate: Optional[np.ndarray] = None,
        pop_rate_square_width: int = 5,
        pop_rate_gauss_sigma: int = 5,
        sort_indices: Optional[Union[np.ndarray, List[int]]] = None,
        event_times_ms: Optional[List[float]] = None,
        event_periods: Optional[List[Tuple[float, float]]] = None,
        figsize: Tuple[float, float] = (12, 6),
        height_ratios: Optional[List[float]] = None,
        font_size: int = 14,
        xlabel: str = "Time (ms)",
        time_axis_absolute: bool = True,
        fr_vmin: Optional[float] = None,
        fr_vmax: Optional[float] = None,
        save_path: Optional[str] = None,
        show_fig: bool = True,
        **kwargs: Any,
    ) -> Tuple[Optional[Any], List[Any]]:
        """
        Plot spike raster with optional population rate and per-unit rate heatmap.
        Original TODO (raster): MAKE THIS A SPIKEDATA METHOD (SCRIPT IN plot_utils.py).

        Uses plot_utils.plot_raster. Pass precomputed pop_rate or fr_rates, or let
        them be computed (pop_rate via get_pop_rate; fr_rates from binned rates if
        fr_rate_bin_ms is set).

        Parameters
        ----------
        time_range : (float, float) or None
            (start_ms, end_ms) to display.
        raster_bin_size_ms : float
            Bin width in ms for the spike raster (default 1 ms).
        fr_rates : np.ndarray or RateData or None
            Per-unit rate matrix (units x time) for bottom heatmap. If RateData,
            uses .inst_Frate_data (must match raster time bins or be sliced).
        fr_rate_bin_ms : float or None
            If fr_rates is None and this is set, compute fr_rates as binned spike
            counts converted to Hz (counts / (bin_ms/1000)).
        pop_rate : np.ndarray 1D or None
            Precomputed population rate (spikes per bin). If None, computed with
            get_pop_rate(pop_rate_square_width, pop_rate_gauss_sigma, raster_bin_size_ms).
        pop_rate_square_width, pop_rate_gauss_sigma : int
            Used only when pop_rate is None.
        sort_indices : array-like or None
            Reorder units for raster and fr_rates.
        event_times_ms, event_periods : optional
            Passed to plot_utils.plot_raster.
        **kwargs
            Passed to plot_utils.plot_raster (e.g. figsize, height_ratios, font_size).

        Returns
        -------
        fig, axes
        """
        from .plot_utils import plot_raster as _plot_raster

        spk_mat = self.sparse_raster(bin_size=raster_bin_size_ms).toarray()
        if sort_indices is not None:
            sort_indices = np.asarray(sort_indices)

        if pop_rate is None:
            pop_rate = self.get_pop_rate(
                square_width=pop_rate_square_width,
                gauss_sigma=pop_rate_gauss_sigma,
                raster_bin_size_ms=raster_bin_size_ms,
            )

        fr_arr: Optional[np.ndarray] = None
        if fr_rates is not None:
            if hasattr(fr_rates, "inst_Frate_data"):
                fr_arr = np.asarray(fr_rates.inst_Frate_data)
            else:
                fr_arr = np.asarray(fr_rates)
        elif fr_rate_bin_ms is not None:
            # Binned counts -> Hz
            binned = self.raster(bin_size=fr_rate_bin_ms)
            fr_arr = binned / (fr_rate_bin_ms / 1000.0)

        return _plot_raster(
            spk_mat,
            bin_size_ms=raster_bin_size_ms,
            time_range=time_range,
            pop_rate=pop_rate,
            fr_rates=fr_arr,
            sort_indices=sort_indices,
            event_times_ms=event_times_ms,
            event_periods=event_periods,
            figsize=figsize,
            height_ratios=height_ratios,
            font_size=font_size,
            xlabel=xlabel,
            time_axis_absolute=time_axis_absolute,
            fr_vmin=fr_vmin,
            fr_vmax=fr_vmax,
            save_path=save_path,
            show_fig=show_fig,
            **kwargs,
        )

    def compute_spike_trig_pop_rate(
        self, window_ms=80, cutoff_hz=20, fs=1000, bin_size=1, cut_outer=10
    ):
        """
        Compute spike-triggered population rate (stPR).

        Implementation of the stPR measure from "Invariant activity sequences
        across the mouse brain" (Bhatt et al.).

        Computes c_{i,τ} = 100 × Σ_{t: f_i(t+τ)>0} [P(t) - μ_i] / ||f_i||

        For each neuron *i* and lag *τ*, the coupling curve measures how
        much the population rate deviates from its mean whenever neuron *i*
        fires, normalised by the neuron's total spike count.

        Parameters
        ----------
        window_ms (int): Half-width of the lag window in ms (window from -window_ms to +window_ms )
        cutoff_hz (float): Low-pass Butterworth filter cutoff in Hz applied to the coupling curves.
        fs (float): Sampling rate in Hz used for filter design.
        bin_size (float): Bin size in ms for the spike raster.
        cut_outer (int): Number of outer lag bins to ignore.

        Returns
        -------
        stPR_filtered (np.ndarray, shape (N, 2*window_ms + 1)): Low-pass filtered coupling curves for every neuron.
        coupling_strengths_zero_lag (np.ndarray, shape (N,)): Coupling strength at lag 0 (c_{i,0})
        coupling_strengths_max (np.ndarray, shape (N,)): Peak coupling strength within the trimmed lag window.
        delays (np.ndarray, shape (N,)): Lag (in bins) at which peak coupling occurs, relative to lag 0, within the trimmed window.
        lags (np.ndarray): Array of lag values from -window_ms to +window_ms.
        """
        # Bin spike data to a spike matrix
        spike_matrix = self.sparse_raster(bin_size=bin_size).toarray()

        # Get dimensions
        num_neurons, num_bins = spike_matrix.shape

        # Prepare lags: τ values from −window_ms to +window_ms
        lags = np.arange(-window_ms, window_ms + 1)

        # P(t) = (1/N) Σ_j f_j(t) — population mean rate per time bin
        P = np.mean(spike_matrix, axis=0)

        # μ_i = average firing rate of neuron i
        mu = np.mean(spike_matrix, axis=1)

        # ||f_i|| = total number of spikes fired by neuron i
        total_spikes = np.sum(spike_matrix, axis=1)

        # c_{i,τ} for all neurons, lags
        stPR = np.zeros((num_neurons, len(lags)))

        for i in range(num_neurons):
            # Skip silent neurons
            if total_spikes[i] == 0:
                continue

            # All spike times for neuron i: {s | f_i(s) > 0}
            spike_times = np.where(spike_matrix[i] > 0)[0]

            # Accumulator for Σ[P(t) - μ_i]
            sum_deviations = np.zeros(len(lags))

            for tau_idx, tau in enumerate(lags):
                # For lag τ, find {t | f_i(t + τ) > 0}
                # Let s = t + τ → t = s − τ where f_i(s) > 0

                # Map spike time to population time
                valid_t = spike_times - tau

                # Only use valid population times t ∈ [0, num_bins)
                mask = (valid_t >= 0) & (valid_t < num_bins)
                if np.any(mask):
                    # Compute [P(t) - μ_i] for valid trigger times
                    deviations = P[valid_t[mask]] - mu[i]
                    # Sum over trigger times
                    sum_deviations[tau_idx] = np.sum(deviations)

            # c_{i,τ} = 100 × Σ[P(t) − μ_i] / ||f_i||
            stPR[i] = 100 * sum_deviations / total_spikes[i]

        # Low-pass filter coupling curves with 20 Hz cutoff
        nyquist = fs / 2
        b, a = signal.butter(2, cutoff_hz / nyquist, btype="low")
        stPR_filtered = np.array(
            [signal.filtfilt(b, a, stPR[i]) for i in range(num_neurons)]
        )

        # Coupling strength = c_{i,0} (lag 0) for chorister/soloist classification
        coupling_strengths_zero_lag = stPR_filtered[:, window_ms]

        # Get peak coupling strength and delay (ignoring for lags in first and last cut_outer)
        trimmed = stPR_filtered[:, cut_outer:-cut_outer]
        coupling_strengths_max = np.max(trimmed, axis=1)
        peak_indices = np.argmax(trimmed, axis=1)
        delays = peak_indices - ((stPR_filtered.shape[1] - 1) / 2 - cut_outer)

        return (
            stPR_filtered,
            coupling_strengths_zero_lag,
            coupling_strengths_max,
            delays,
            lags,
        )

    def get_bursts(
        self,
        thr_burst,
        min_burst_diff,
        burst_edge_mult_thresh,
        square_width=20,
        gauss_sigma=100,
        acc_square_width=5,
        acc_gauss_sigma=5,
        raster_bin_size_ms=1.0,
        peak_to_trough=True,
        pop_rate=None,
        pop_rate_acc=None,
        pop_rms_override=None,
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
        pop_rms_override (float, optional): RMS to override burst threshold baseline for normalizing accross separate datasets

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
        if pop_rms_override is None:
            pop_rms = np.sqrt(np.mean(np.square(pop_rate)))
        else:
            pop_rms = pop_rms_override

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
