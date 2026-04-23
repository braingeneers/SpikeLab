"""SpikeData core module."""

import heapq
import itertools
import warnings
from typing import Literal, Optional, Union, List, Tuple, Sequence
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal, sparse
from scipy.stats import norm
from .ratedata import RateData
from .pairwise import PairwiseCompMatrix


from .utils import (
    get_sttc,
    butter_filter,
    extract_waveforms,
    _sttc_ta,
    _sttc_na,
    _spike_time_tiling,
    _resampled_isi,
    _sliding_rate_single_train,
    _train_from_i_t_list,
    swap,
    randomize,
    trough_between,
    extract_unit_waveforms,
    _get_attr,
    compute_cross_correlation_with_lag,
    compute_cosine_similarity_with_lag,
    _resolve_n_jobs,
    _compute_agreement_score,
    _compute_footprint,
    _compute_footprint_similarity,
)
from concurrent.futures import ThreadPoolExecutor

__all__ = [
    "SpikeData",
    "get_sttc",
]


class SpikeData:
    """Neuronal spike data with functionality for loading, processing, and analyzing.

    Attributes:
        train (list[numpy.ndarray]): List of numpy arrays, where each array
            contains the spike times for a particular neuron.
        N (int): The number of neurons in the dataset.
        length (float): The total duration of the time window in milliseconds.
        start_time (float): The time origin in milliseconds (default 0.0).
            For event-centered data, this is typically -pre_ms. Spike times
            fall within [start_time, start_time + length].
        neuron_attributes (list[dict] or None): A list of dictionaries
            containing information on each neuron.
        metadata (dict): A dictionary containing any additional information
            or metadata about the spike data.
        raw_data (numpy.ndarray): If provided, this numpy array contains the
            raw time series data.
        raw_time (numpy.ndarray or float): Either a numpy array of sample
            times, or a single float representing a sample rate in kHz.
    """

    @staticmethod
    def from_idces_times(idces, times, N=None, **kwargs):
        """Create a SpikeData object from lists of unit indices and spike times.

        Parameters:
            idces (list): List of unit indices.
            times (list): List of spike times.
            N (int): Number of units (optional).
            **kwargs: Additional keyword arguments for the SpikeData
                constructor.

        Returns:
            spike_data (SpikeData): A new SpikeData object with the given
                unit indices and spike times.

        Notes:
            - This method is a wrapper around the _train_from_i_t_list
              helper function.
            - When ``idces`` is empty and ``N`` is None, defaults to 0 units
              and ``length=0``.
        """
        idces = np.asarray(idces)
        if idces.size == 0:
            kwargs.setdefault("length", 0)
            N = 0 if N is None else N
        return SpikeData(_train_from_i_t_list(idces, times, N), N=N, **kwargs)

    @staticmethod
    def from_raster(raster, bin_size_ms, **kwargs):
        """Create a SpikeData object from a spike raster matrix with shape (N, T).

        Parameters:
            raster (numpy.ndarray): Spike raster matrix with shape
                (N [units], T [time bins]).
            bin_size_ms (float): Size of each time bin in milliseconds.
            **kwargs: Additional keyword arguments for the SpikeData
                constructor.

        Returns:
            sd (SpikeData): Object with the given spike raster.

        Notes:
            - The generated spike times are evenly spaced within each time
              bin. For example, if a unit fires 3 times in a 10 ms time bin,
              those events go at 2.5, 5, and 7.5 ms after the start of the
              bin.
            - All metadata parameters of the regular constructor are accepted.
            - For event-centered rasters (where bin 0 corresponds to
              ``start_time``, not t=0), pass ``start_time`` in kwargs so that
              spike times are correctly offset. Without it, bin 0 maps to t=0.
        """
        raster = np.asarray(raster)
        if raster.ndim != 2:
            raise ValueError(f"raster must be 2D (N x T), got {raster.ndim}D array")
        raster = raster.astype(int)
        N, T = raster.shape
        # Offset generated spike times by start_time so bin 0 maps to
        # start_time (not 0) when reconstructing event-centered data.
        t_offset = kwargs.get("start_time", 0.0)
        train = [[] for _ in range(N)]
        for i, t in zip(*raster.nonzero()):
            n_spikes = raster[i, t]
            times = (
                t_offset
                + t * bin_size_ms
                + np.linspace(0, bin_size_ms, n_spikes + 2)[1:-1]
            )
            train[i].extend(times)

        kwargs.setdefault("length", T * bin_size_ms)
        return SpikeData(train, **kwargs)

    @staticmethod
    def from_events(events, N=None, **kwargs):
        """Create a SpikeData object from a list of (unit index, time) pairs.

        Parameters:
            events (list): List of (index, time) pairs.
            N (int): Number of units (default: maximum index in the events).
            **kwargs: Additional keyword arguments for the SpikeData
                constructor.

        Returns:
            sd (SpikeData): Object with the given events.

        Notes:
            - This method is a wrapper around the from_idces_times helper
              function. All metadata parameters of the regular constructor
              are accepted.
        """
        idces, times = [], []
        for i, t in events:
            idces.append(i)
            times.append(t)
        return SpikeData.from_idces_times(idces, times, N, **kwargs)

    @staticmethod
    def from_neo_spiketrains(spiketrains, **kwargs):
        """Create a SpikeData object from a list of neo.SpikeTrain objects.

        Parameters:
            spiketrains (list): List of neo.SpikeTrain objects.
            **kwargs: Additional keyword arguments for the SpikeData
                constructor.

        Returns:
            sd (SpikeData): Object with the given spike trains in
                milliseconds.
        """
        trains = [st.copy() for st in spiketrains]
        for st in trains:
            st.units = "ms"

        return SpikeData([np.asarray(st) for st in trains], **kwargs)

    @staticmethod
    def from_thresholding(
        data: NDArray,
        fs_Hz=20e3,
        threshold_sigma=5.0,
        filter: Union[dict, bool] = True,
        hysteresis=True,
        direction: Literal["both", "up", "down"] = "both",
    ):
        """Create a SpikeData object by filtering and thresholding raw electrophysiological data.

        Parameters:
            data (numpy.ndarray): Raw data with shape (channels, time).
            fs_Hz (float): Sampling frequency in Hz.
            threshold_sigma (float): Threshold in units of per-channel
                standard deviation.
            filter (dict or bool): Filter configuration or False to disable
                filtering; if True, a third-order Butterworth filter with
                passband 300 Hz to 6 kHz is used.
            hysteresis (bool): Use hysteresis for thresholding.
            direction (str): Direction of thresholding ('both', 'up',
                'down').

        Returns:
            sd (SpikeData): Object with the given raw data.

        Notes:
            - To use different filter parameters, pass a dictionary, which
              will be passed as keyword arguments to butter_filter().
            - If filter is False, no filtering is done.
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
        else:
            raise ValueError(
                f"direction must be 'both', 'up', or 'down', got {direction!r}"
            )

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
        start_time=0.0,
        neuron_attributes=None,
        metadata=None,
        raw_data=None,
        raw_time: Optional[Union[NDArray, float]] = None,
    ):
        """Initialize a SpikeData object from a list of spike trains.

        Parameters:
            train (list): List of spike trains, each a list of spike times
                in milliseconds. Spike times can be negative for
                event-centered data (e.g. -200 to +300 around a stimulus
                event).
            N (int): Number of units (optional).
            length (float): Total duration of the time window in
                milliseconds (optional). For event-centered data with times
                from -200 to +300, length is 500. Defaults to the span from
                start_time to the latest spike time.
            start_time (float): Time of the first bin in milliseconds
                (default 0.0). For event-centered data, this is typically
                ``-pre_ms`` (e.g. -200.0). Spike times must fall within
                ``[start_time, start_time + length]``.
            neuron_attributes (list): List of neuron attributes (optional).
            metadata (dict): Dictionary of metadata (optional).
            raw_data (numpy.ndarray): Raw timeseries data with shape
                (channels, time) (optional).
            raw_time (numpy.ndarray or float): Raw time vector with shape
                (time) or sample rate in kHz (optional).

        Notes:
            - Arbitrary raw timeseries data, not associated with particular
              units, can be passed in as ``raw_data`` (an array with shape
              (channels, time)).
            - The ``raw_time`` argument can also be a sample rate in kHz, in
              which case it is generated assuming that the start of the raw
              data corresponds with t=0.
        """
        # Make sure each individual spike train is sorted. As a side effect,
        # also copy each array to avoid aliasing.
        self.train = [np.sort(times) for times in train]

        # Reject NaN spike times — they propagate silently and corrupt
        # downstream computations (rates, rasters, correlations).
        for i, t in enumerate(self.train):
            if len(t) > 0 and np.isnan(t).any():
                raise ValueError(f"spike times for unit {i} contain NaN values")

        # Store the time origin.
        self.start_time = float(start_time)

        # The length of the spike train defaults to the span from
        # start_time to the latest spike time.
        if length is None:
            max_spike = max(
                (t[-1] for t in self.train if len(t) > 0),
                default=self.start_time,
            )
            length = max_spike - self.start_time
        if np.isnan(length):
            raise ValueError("length must not be NaN")
        if length < 0:
            raise ValueError(f"length must be non-negative, got {length}")
        self.length = length

        # Validate that all spike times fall within [start_time, start_time + length].
        end_time = self.start_time + self.length
        for i, t in enumerate(self.train):
            if len(t) == 0:
                continue
            if t[0] < self.start_time:
                raise ValueError(
                    f"Unit {i}: spike time {t[0]:.4f} ms is before start_time "
                    f"({self.start_time}). Spike times must fall within "
                    f"[{self.start_time}, {end_time}]."
                )
            if t[-1] > end_time:
                raise ValueError(
                    f"Unit {i}: spike time {t[-1]:.4f} ms exceeds end of time "
                    f"window ({end_time}). If spike times are absolute, "
                    f"subtract the start time from each train before "
                    f"constructing SpikeData. To trim an existing SpikeData, "
                    f"use subtime()."
                )

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
        if metadata is None:
            metadata = {}
        self.metadata = metadata.copy()
        self.neuron_attributes = None
        if neuron_attributes is not None:
            self.neuron_attributes = neuron_attributes.copy()
            if len(neuron_attributes) != self.N:
                raise ValueError(
                    f"neuron_attributes has {len(neuron_attributes)} "
                    f"instead of {self.N} items."
                )

    def __repr__(self) -> str:
        return (
            f"SpikeData(N={self.N}, length={self.length:.1f}, "
            f"start_time={self.start_time:.1f})"
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
        """Generate matched arrays of unit indices and times for all events.

        Returns:
            idces (numpy.ndarray): Array of unit indices.
            times (numpy.ndarray): Array of times for all events.

        Notes:
            - This method is not a property unlike ``times`` and ``events``
              because the lists must actually be constructed in memory.
        """
        idces, times = [], []
        for i, t in self.events:
            idces.append(i)
            times.append(t)
        return np.array(idces), np.array(times)

    @property
    def unit_locations(self) -> Optional[np.ndarray]:
        """Get unit locations as an (U, D) array where D is the spatial dimension.

        Returns:
            locations (numpy.ndarray or None): Array of unit locations, shape
                (N, D). None if any unit lacks location data.

        Notes:
            - Extracts from neuron_attributes 'location', 'x'/'y'/'z'
              (x and y required), or 'position' keys.
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
        """Get electrode/channel indices for each unit as a 1D array.

        Returns:
            electrodes (numpy.ndarray or None): 1D array of electrode
                indices. None if any unit lacks electrode data.

        Notes:
            - Extracts from neuron_attributes 'electrode', 'channel', or
              'ch' keys.
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
        """Split the recording into a SpikeSliceStack of fixed-length windows.

        Parameters:
            length (float): Length of each window in milliseconds.
            overlap (float): Overlap between consecutive windows in
                milliseconds (default: 0).

        Returns:
            stack (SpikeSliceStack): Stack of SpikeData windows, one per
                frame.

        Notes:
            - Windows that would extend past the end of the recording are
              excluded.
            - overlap must be strictly less than length.
        """
        from .spikeslicestack import SpikeSliceStack

        step = length - overlap
        if step <= 0:
            raise ValueError("overlap must be less than length")
        end_time = self.start_time + self.length
        times = [
            (float(start), float(start) + length)
            for start in np.arange(self.start_time, end_time - length + 1e-9, step)
        ]
        if not times:
            raise ValueError(
                f"Recording length ({self.length} ms) is shorter than frame length ({length} ms)"
            )
        return SpikeSliceStack(self, times_start_to_end=times)

    def align_to_events(
        self,
        events,
        pre_ms,
        post_ms,
        *,
        kind="spike",
        bin_size_ms=1.0,
        sigma_ms=10,
    ):
        """Align spike trains to events and return an event-aligned slice stack.

        Parameters:
            events (array-like or str): Event times in milliseconds, or a
                string key into ``self.metadata`` whose value is an array
                of event times in ms.
            pre_ms (float): Window duration before each event in
                milliseconds.
            post_ms (float): Window duration after each event in
                milliseconds.
            kind (str): ``"spike"`` to return a ``SpikeSliceStack``, or
                ``"rate"`` to return a ``RateSliceStack``. Default
                ``"spike"``.
            bin_size_ms (float): Time bin width in milliseconds. Only used
                when ``kind="rate"``. Default 1.0.
            sigma_ms (float): Gaussian smoothing sigma in milliseconds for
                ISI-based firing rate estimation. Only used when
                ``kind="rate"``. Default 10.

        Returns:
            stack (SpikeSliceStack or RateSliceStack): Event-aligned slice
                stack with one slice per event. Events whose window extends
                outside ``[start_time, start_time + length]`` are dropped
                with a warning.

        Notes:
            - When ``events`` is a metadata key, the corresponding array
              must already be in milliseconds (as stored by
              ``load_spikedata_from_ibl``).
        """
        import warnings

        from .spikeslicestack import SpikeSliceStack
        from .rateslicestack import RateSliceStack

        if kind not in ("spike", "rate"):
            raise ValueError(f"kind must be 'spike' or 'rate', got {kind!r}")

        # Resolve metadata key to array.
        if isinstance(events, str):
            if self.metadata is None or events not in self.metadata:
                raise KeyError(
                    f"Metadata key {events!r} not found. "
                    f"Available keys: {list(self.metadata or {})}"
                )
            event_times = np.asarray(self.metadata[events], dtype=float)
        else:
            event_times = np.asarray(events, dtype=float)

        # Drop events whose window would extend outside [start_time, start_time + length].
        rec_start = self.start_time
        rec_end = self.start_time + self.length
        valid_mask = (event_times - pre_ms >= rec_start) & (
            event_times + post_ms <= rec_end
        )
        n_dropped = int(np.sum(~valid_mask))
        if n_dropped > 0:
            warnings.warn(
                f"{n_dropped} event(s) dropped because their "
                f"[{-pre_ms}, +{post_ms}] ms window extends outside the recording "
                f"bounds [{rec_start:.1f}, {rec_end:.1f}] ms.",
                UserWarning,
                stacklevel=2,
            )
        event_times = event_times[valid_mask]

        if len(event_times) == 0:
            raise ValueError(
                "No valid events remain after filtering for recording bounds."
            )

        time_bounds = (pre_ms, post_ms)

        if kind == "spike":
            return SpikeSliceStack(
                self, time_peaks=event_times.tolist(), time_bounds=time_bounds
            )
        else:
            return RateSliceStack(
                self,
                time_peaks=event_times.tolist(),
                time_bounds=time_bounds,
                sigma_ms=sigma_ms,
                step_size=bin_size_ms,
            )

    def binned(self, bin_size=40.0):
        """Count the number of events in each time bin across all units.

        Bins are relative to ``start_time``: ``(start_time, start_time +
        bin_size]``, ``(start_time + bin_size, start_time + 2*bin_size]``,
        etc. A spike at exactly ``start_time`` is included in bin 0.

        Parameters:
            bin_size (float): Size of the time bin in milliseconds.

        Returns:
            binned_raster (numpy.ndarray): Array of the number of events in
                each bin.
        """
        # sum(0) on CSR returns a (1, T) matrix in older SciPy; flatten to 1D array
        return np.asarray(self.sparse_raster(bin_size).sum(0)).ravel()  # type: ignore

    def binned_meanrate(self, bin_size=40, unit="kHz"):
        """Calculate the mean firing rate across the population in each time bin.

        Parameters:
            bin_size (float): Size of the time bin in milliseconds.
            unit (str): Unit of the firing rate ('Hz' or 'kHz').

        Returns:
            binned_meanrate (numpy.ndarray): Array of the mean firing rate
                in each bin.

        Notes:
            - The rate is calculated as the number of events in each bin
              divided by the bin size and number of units.
        """
        if self.N == 0:
            return np.zeros(int(np.ceil(self.length / bin_size)))
        binned_rate = self.binned(bin_size) / self.N / bin_size
        if unit == "Hz":
            return 1e3 * binned_rate
        elif unit == "kHz":
            return binned_rate
        else:
            raise ValueError(f"Unknown unit {unit} (try Hz or kHz)")

    def rates(self, unit="kHz"):
        """Calculate the mean firing rate of each neuron over the recording.

        Parameters:
            unit (str): Unit of the firing rate ('Hz' or 'kHz').

        Returns:
            rates (numpy.ndarray): Array of the firing rate of each neuron.
        """
        if self.length == 0:
            return np.zeros(self.N)
        rates = np.array([len(t) for t in self.train]) / self.length
        if unit == "Hz":
            return 1e3 * rates
        elif unit == "kHz":
            return rates
        else:
            raise ValueError(f"Unknown unit {unit} (try Hz or kHz)")

    def resampled_isi(self, times, sigma_ms=10.0):
        """Calculate instantaneous firing rate of each unit at the given times.

        Computes interspike intervals and interpolates their inverse.

        Parameters:
            times (numpy.ndarray): Array of times to resample the firing
                rate to.
            sigma_ms (float): Standard deviation of the Gaussian kernel in
                milliseconds.

        Returns:
            RateData: Object with inst_Frate_data (N, T) and times;
                units: Hz (spikes/s).
        """
        times = np.atleast_1d(times)
        rate_array = np.array([_resampled_isi(t, times, sigma_ms) for t in self.train])
        if rate_array.ndim == 1:
            rate_array = rate_array[:, np.newaxis]
        return RateData(
            inst_Frate_data=rate_array,
            times=times,
            neuron_attributes=self.neuron_attributes,
            rate_unit="Hz",
        )

    def sliding_rate(
        self,
        window_size,
        step_size=None,
        sampling_rate=None,
        t_start=None,
        t_end=None,
        gauss_sigma=0.0,
        apply_square=True,
    ):
        """
        Compute continuous firing rate of each unit using a sliding-window average.

        For each time bin t, counts spikes in the centered window [t - W/2, t + W/2]
        and returns rate R(t) = N / W (spikes per time unit, e.g. kHz).

        Parameters:
            window_size (float): Width of the sliding window in ms. Centered
                window [t - W/2, t + W/2].
            step_size (float, optional): Advance step for time bins in ms.
                Mutually exclusive with sampling_rate.
            sampling_rate (float, optional): Samples per ms;
                step_size = 1 / sampling_rate. Mutually exclusive with step_size.
            t_start (float, optional): Start of output time range in ms.
                Default: start_time - window_size/2.
            t_end (float, optional): End of output time range in ms.
                Default: start_time + length + window_size/2.
            gauss_sigma (float, optional): Gaussian smoothing sigma in ms.
                If 0, Gaussian smoothing is disabled.
            apply_square (bool, optional): If True, applies the square-window
                smoothing defined by window_size. If False, computes per-bin
                rates first and then applies optional Gaussian smoothing.

        Returns:
            RateData: Object with inst_Frate_data (N, T) and times;
                units: spikes/ms (kHz).
        """
        # --- Validate parameters ---
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if step_size is None and sampling_rate is None:
            raise ValueError("Must provide either step_size or sampling_rate")
        if step_size is not None and sampling_rate is not None:
            raise ValueError(
                "step_size and sampling_rate are mutually exclusive; "
                "provide one, not both"
            )
        if step_size is None:
            if sampling_rate <= 0:
                raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
            step_size = 1.0 / sampling_rate
        elif step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if gauss_sigma < 0:
            raise ValueError(f"gauss_sigma must be non-negative, got {gauss_sigma}")

        # --- Time range defaults ---
        if t_start is None:
            t_start = self.start_time - window_size / 2
        if t_end is None:
            t_end = self.start_time + self.length + window_size / 2
        if t_end <= t_start:
            raise ValueError(
                f"t_end must be greater than t_start "
                f"(got t_start={t_start}, t_end={t_end})"
            )

        # --- Compute bin edges and time vector ---
        span = t_end - t_start
        n_bins = int(np.ceil(span / step_size))
        remainder = span % step_size
        if remainder < 1e-12 or abs(remainder - step_size) < 1e-12:
            n_bins += 1
        bin_edges = t_start + np.arange(n_bins + 1) * step_size
        time_vector = (bin_edges[:-1] + bin_edges[1:]) / 2

        # --- Histogram all units at once ---
        rate_array = np.zeros((self.N, n_bins), dtype=float)
        for i, ts in enumerate(self.train):
            if len(ts) == 0:
                continue
            hist, _ = np.histogram(ts, bins=bin_edges)
            rate_array[i] = hist

        # --- Smoothing ---
        if apply_square:
            window_bins = min(max(1, int(round(window_size / step_size))), n_bins)
            effective_window = window_bins * step_size
            kernel = np.ones(window_bins)
            for i in range(self.N):
                counts = np.convolve(rate_array[i], kernel, mode="same")
                rate_array[i] = counts / effective_window
        else:
            rate_array /= step_size

        if gauss_sigma > 0:
            sigma_bins = gauss_sigma / step_size
            for i in range(self.N):
                rate_array[i] = ndimage.gaussian_filter1d(
                    rate_array[i], sigma=sigma_bins
                )

        return RateData(
            inst_Frate_data=rate_array,
            times=time_vector,
            neuron_attributes=self.neuron_attributes,
            rate_unit="kHz",
        )

    def set_neuron_attribute(self, key: str, values, neuron_indices=None):
        """Set an attribute across neurons in neuron_attributes.

        Parameters:
            key (str): Name of the attribute.
            values (single value or list): Single value (applied to all) or
                list/array matching neuron_indices length for each neuron.
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
        """Get an attribute across all neurons.

        Parameters:
            key (str): Attribute name.
            default (any): Value if neuron lacks the attribute.

        Returns:
            values (list): List of values, one per neuron.
        """
        if self.neuron_attributes is None:
            return [default] * self.N
        return [attr.get(key, default) for attr in self.neuron_attributes]

    def subset(self, units, by=None):
        """Return a new SpikeData with only the selected units.

        Units are selected either by their indices or by an ID stored under
        a given key in the neuron_attributes.

        Parameters:
            units (list): List of unit indices to select.
            by (str): Key to select units by in the neuron_attributes.
                Index-based if None.

        Returns:
            sd (SpikeData): New SpikeData object with the selected units.

        Notes:
            - Units are included in the output according to their order in
              self.train, not the order in the unit list (which is treated
              as a set).
            - raw_data and raw_time are not propagated to the subset -- they
              remain on the original SpikeData object.
            - If IDs are not unique, every neuron which matches is included
              in the output.
            - Neurons whose neuron_attributes entry does not have the key
              are always excluded.
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
                if _get_attr(self.neuron_attributes[i], by, _missing) in units
            }

        train = []
        neuron_attributes = []
        for i, ts in enumerate(self.train):
            if i in units:
                train.append(ts)
                if self.neuron_attributes is not None:
                    neuron_attributes.append(self.neuron_attributes[i])

        # raw_data/raw_time are not propagated to subsets — they remain
        # on the original SpikeData object and can be accessed there.
        return SpikeData(
            train,
            length=self.length,
            start_time=self.start_time,
            N=len(train),
            neuron_attributes=neuron_attributes or None,
            metadata=self.metadata,
        )

    def neuron_to_channel_map(
        self, channel_attr: Optional[str] = None
    ) -> dict[int, int]:
        """Return a mapping from neuron indices to channel indices.

        Parameters:
            channel_attr (str or None): Name of the attribute in
                neuron_attributes that contains the channel index. If None,
                searches for common attribute names.

        Returns:
            mapping (dict): Mapping from neuron index (int) to channel index
                (int). Returns an empty dict if neuron_attributes is None.

        Notes:
            - If neuron_attributes is None and channel information is
              required, or if the specified channel_attr doesn't exist for
              all neurons, a ValueError is raised.
            - If channel_attr is not specified, attempts to find channel
              information using common attribute names: 'channel',
              'channel_id', 'channel_index', 'ch', 'channel_idx'.
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

    def subtime(self, start, end, shift_to=None):
        """Extract a subset of time points from the spike data.

        Spike times are shifted so that ``shift_to`` becomes t=0 in the new
        SpikeData. By default ``shift_to=start``, so subtime(100, 200)
        produces spikes in the range [0, 100). For event-centered slicing,
        pass ``shift_to=event_time`` to produce spikes from
        ``-(event - start)`` to ``+(end - event)``.

        Parameters:
            start (float): Starting time value (inclusive).
            end (float): Ending time value (exclusive).
            shift_to (float or None): The time value that becomes t=0 in the
                output. Defaults to ``start`` (standard behavior). For
                event-centered output, pass the event time so that t=0
                corresponds to the event.

        Returns:
            sd (SpikeData): New SpikeData object containing only the
                specified time range.

        Notes:
            - For standard data (``start_time >= 0``), negative start/end
              values are counted backwards from the end of the recording.
              For event-centered data (``start_time < 0``), negative values
              are treated as literal times.
            - Start and end can also be None or Ellipsis, in which case that
              end of the data is not truncated.
            - All metadata and neuron data are propagated.
            - The output SpikeData has ``start_time = start - shift_to`` and
              ``length = end - start``.
        """
        end_time = self.start_time + self.length

        if start is None or start is Ellipsis:
            start = self.start_time
        elif start < 0 and self.start_time >= 0:
            # Backward-counting from end (only for standard 0-based data;
            # for event-centered data negative values are literal times).
            start += end_time
            if start < self.start_time:
                raise ValueError(
                    f"start ({start - end_time}) is too negative. "
                    f"Minimum allowed is -{self.length} (recording length)"
                )
        elif start > end_time:
            raise ValueError(
                f"start ({start}) exceeds recording end ({end_time}). "
                f"Recording range is [{self.start_time}, {end_time}]."
            )

        if end is None or end is Ellipsis:
            end = end_time
        elif end < 0 and self.start_time >= 0:
            end += end_time
        elif end > end_time:
            raise ValueError(
                f"end ({end}) exceeds recording end ({end_time}). "
                f"Recording range is [{self.start_time}, {end_time}]."
            )

        if start >= end:
            raise ValueError(
                f"start ({start}) must be less than end ({end}). "
                f"Cannot create subtime with invalid range."
            )

        # Default shift_to is start (standard 0-based behavior)
        if shift_to is None:
            shift_to = start

        # Subset the spike train by time, shifting by shift_to
        train = [t[(t >= start) & (t < end)] - shift_to for t in self.train]
        new_start_time = start - shift_to

        # Subset and propagate the raw data
        rawmask = (self.raw_time >= start) & (self.raw_time < end)

        return SpikeData(
            train,
            length=end - start,
            start_time=new_start_time,
            N=self.N,
            neuron_attributes=self.neuron_attributes,
            metadata=self.metadata,
            raw_time=self.raw_time[rawmask] - shift_to,
            raw_data=self.raw_data[..., rawmask],
        )

    def __getitem__(self, key):
        """Index by time slice or by unit indices.

        A slice is interpreted as a time range via ``subtime()``. An
        iterable is interpreted as unit indices via ``subset()``.

        Parameters:
            key (slice or iterable): Slice or iterable of neuron indices to
                select.

        Returns:
            sd (SpikeData): New SpikeData object with the selected units.
        """
        if isinstance(key, slice):
            return self.subtime(key.start, key.stop)
        else:
            return self.subset(key)

    def append(self, spikeData, offset=0):
        """Append spike times from another SpikeData object to this one.

        Parameters:
            spikeData (SpikeData): SpikeData object to append.
            offset (float): Offset in milliseconds to add to the spike times
                of the appended data.

        Returns:
            sd (SpikeData): New SpikeData object with the appended data.

        Notes:
            - The two SpikeData objects must have the same number of neurons.
            - On metadata key collision, values from ``self`` take precedence.
        """
        if self.N != spikeData.N:
            raise ValueError("Cannot concatenate SpikeData with different N")
        # Shift appended spikes from their own time base to follow self's time range.
        # Subtract spikeData.start_time to normalize to 0-based, then add the
        # concatenation point (self's end time + gap).
        concat_point = self.start_time + self.length + offset
        shift = concat_point - spikeData.start_time
        train = [
            np.hstack([tr1, tr2 + shift])
            for tr1, tr2 in zip(self.train, spikeData.train)
        ]
        if self.raw_data.size > 0 and spikeData.raw_data.size > 0:
            raw_data = np.concatenate((self.raw_data, spikeData.raw_data), axis=1)
            raw_time = np.concatenate((self.raw_time, spikeData.raw_time + shift))
        elif spikeData.raw_data.size > 0:
            raw_data = spikeData.raw_data.copy()
            raw_time = spikeData.raw_time + shift
        else:
            raw_data = self.raw_data
            raw_time = self.raw_time
        length = self.length + spikeData.length + offset
        return SpikeData(
            train,
            length=length,
            start_time=self.start_time,
            N=self.N,
            neuron_attributes=self.neuron_attributes,
            raw_time=raw_time,
            raw_data=raw_data,
            metadata={
                **spikeData.metadata,
                **self.metadata,
            },  # self.metadata takes precedence on key collision
        )

    def sparse_raster(self, bin_size=1.0, time_offset=0.0):
        """Bin spike times into a sparse (units, bins) matrix.

        Entry (i, j) is the number of times unit i fired in bin j. Spike
        times are shifted by ``-start_time`` before binning so that bin 0
        corresponds to ``start_time``.

        Parameters:
            bin_size (float): Size of the time bin in milliseconds.
            time_offset (float): Additional offset added to all spike times
                before binning (default 0.0). Use this to place spikes at
                their absolute recording position, e.g. ``time_offset=500``
                to shift all spikes by 500 ms in the raster.

        Returns:
            raster (sparse.csr_matrix): Sparse array where entry (i, j) is
                the number of times unit i fired in bin j.

        Notes:
            - Bins are left-open, right-closed intervals relative to
              start_time.
            - A spike at exactly start_time is clipped into bin 0.
            - The number of bins is always
              ceil((length + time_offset) / bin_size).
        """
        # Shift spike times so start_time → 0 before binning
        shift = -self.start_time + time_offset
        indices = np.hstack(
            [np.ceil((ts + shift) / bin_size) - 1 for ts in self.train]
        ).astype(int)
        units = np.hstack([0] + [len(ts) for ts in self.train])
        indptr = np.cumsum(units)
        values = np.ones_like(indices)
        length = int(np.ceil((self.length + time_offset) / bin_size))
        np.clip(indices, 0, length - 1, out=indices)
        # Use csr_matrix for SciPy < 1.8 compatibility (csr_array not available)
        return sparse.csr_matrix((values, indices, indptr), shape=(self.N, length))

    def raster(self, bin_size=1.0, time_offset=0.0):
        """Bin spike times into a dense (units, bins) array.

        Entry (i, j) is the number of times unit i fired in bin j.

        Parameters:
            bin_size (float): Size of the time bin in milliseconds.
            time_offset (float): Additional offset added to spike times
                before binning (default 0.0).

        Returns:
            raster (numpy.ndarray): Dense array where entry (i, j) is the
                number of times unit i fired in bin j.

        Notes:
            - Bins are left-open, right-closed intervals relative to
              start_time.
            - A spike at exactly start_time is clipped into bin 0.
        """
        return self.sparse_raster(bin_size, time_offset=time_offset).toarray()

    def channel_raster(self, bin_size=1.0, channel_attr: Optional[str] = None):
        """Create a raster aggregated by channel instead of neuron.

        Parameters:
            bin_size (float): Size of the time bin in milliseconds.
            channel_attr (str): Name of the attribute in neuron_attributes
                that contains the channel index. If None, searches for
                common attribute names.

        Returns:
            channel_raster (numpy.ndarray): Dense array where entry (c, j)
                is the total number of spikes from all neurons on channel c
                in bin j.

        Notes:
            - Channels are determined from neuron_attributes using the same
              logic as neuron_to_channel_map().
            - If neuron_attributes is None or no channel information can be
              found, a ValueError is raised.
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
        """Extract raw voltage waveforms around spike times from raw data.

        Parameters:
            unit (int, slice, list, or None): Unit index selection. int
                extracts a single unit (returns a single waveform array);
                slice/list-like/range extracts a subset (returns a list);
                None extracts all units (returns a list).
            ms_before (float): Milliseconds before each spike time
                (default: 1.0).
            ms_after (float): Milliseconds after each spike time
                (default: 2.0).
            channels (int, list, or None): Channel(s) to extract. None uses
                neuron_to_channel_map or all channels; int for single
                channel; list for multiple; [] for mapped channel.
            bandpass (tuple or None): Optional (lowcut_Hz, highcut_Hz) for
                bandpass filtering.
            filter_order (int): Butterworth filter order (default: 3).
            store (bool): If True (default), stores waveforms and
                avg_waveform in neuron_attributes.
            return_channel_waveforms (bool): If True, include a per-channel
                dict in the return.
            return_avg_waveform (bool): If False, skip computing/returning
                avg_waveform (it will be None).

        Returns:
            waveforms (numpy.ndarray or list): If unit is an int, a single
                3D array shaped (num_channels, num_samples, num_spikes).
                Otherwise, a list of 3D arrays, one per selected unit.
            meta (dict): Dictionary with keys: fs_kHz, unit_indices,
                channels, spike_times_ms, avg_waveforms (optional),
                channel_waveforms (optional).
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
        """Produce a list of arrays of interspike intervals per unit.

        Returns:
            isis (list): List of arrays of interspike intervals per unit.
        """
        return [np.diff(ts) for ts in self.train]

    def concatenate_spike_data(self, sd):
        """Combine units from another SpikeData object with this one.

        Returns a new SpikeData with the units of both objects. If the other
        SpikeData has a different time range, it is subtimed to match this
        one.

        Parameters:
            sd (SpikeData): SpikeData object whose units will be added.

        Returns:
            combined (SpikeData): New SpikeData with units from both objects.

        Notes:
            - raw_data and raw_time are carried over from self only.
            - If only one object has neuron_attributes, a RuntimeWarning is
              issued and the attributes from both are not merged.
        """

        # Subtime the second SpikeData object to the time range of the first
        if sd.length != self.length or sd.start_time != self.start_time:
            end_time = self.start_time + self.length
            sd = sd.subtime(self.start_time, end_time)

        new_train = [t.copy() for t in self.train] + [t.copy() for t in sd.train]
        merged_metadata = {**self.metadata, **sd.metadata}

        new_attrs = None
        if self.neuron_attributes is not None and sd.neuron_attributes is not None:
            new_attrs = self.neuron_attributes + sd.neuron_attributes
        elif self.neuron_attributes is not None or sd.neuron_attributes is not None:
            warnings.warn(
                "Concatenating SpikeData where one has no neuron_attributes. "
                "Dropping attributes from the result.",
                RuntimeWarning,
            )

        return SpikeData(
            new_train,
            length=self.length,
            start_time=self.start_time,
            neuron_attributes=new_attrs,
            metadata=merged_metadata,
            raw_data=self.raw_data,
            raw_time=self.raw_time,
        )

    def spike_time_tilings(self, delt=20.0):
        """Compute the spike time tiling coefficient matrix.

        Parameters:
            delt (float): Time window in milliseconds (default: 20.0).

        Returns:
            ret (PairwiseCompMatrix): Spike time tiling coefficient matrix.

        Notes:
            - When ``numba`` is installed, computation is parallelised across
              all unit pairs using numba's ``prange``.
            - Reference: Cutts & Eglen. Detecting pairwise correlations in
              spike trains. J. Neurosci. 34:43, 14288-14303 (2014).
        """
        if delt <= 0:
            raise ValueError(f"delt must be positive, got {delt}")

        from .numba_utils import NUMBA_AVAILABLE

        # Use numba only when N > 2 (more than 1 pair); for N <= 2
        # the JIT compilation overhead exceeds the serial computation.
        if NUMBA_AVAILABLE and self.N > 2:
            from .numba_utils import flatten_spike_trains, nb_sttc_all_pairs

            flat, offsets = flatten_spike_trains(self.train, self.start_time)
            length = self.length
            if length is None:
                length = float(np.max(flat)) if len(flat) > 0 else 0.0
            upper = nb_sttc_all_pairs(flat, offsets, self.N, delt, length)
            # Unpack upper-triangle vector into symmetric matrix
            ret = np.diag(np.ones(self.N))
            k = 0
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    ret[i, j] = ret[j, i] = upper[k]
                    k += 1
            return PairwiseCompMatrix(matrix=ret, metadata={"delt": delt})

        ret = np.diag(np.ones(self.N))
        for i in range(self.N):
            for j in range(i + 1, self.N):
                ret[i, j] = ret[j, i] = get_sttc(
                    self.train[i],
                    self.train[j],
                    delt,
                    self.length,
                    start_time=self.start_time,
                )
        return PairwiseCompMatrix(matrix=ret, metadata={"delt": delt})

    def spike_time_tiling(self, i, j, delt=20.0):
        """Calculate the spike time tiling coefficient between two units.

        Parameters:
            i (int): Index of the first unit.
            j (int): Index of the second unit.
            delt (float): Time window in milliseconds (default: 20.0).

        Returns:
            ret (float): Spike time tiling coefficient between the two units.

        Notes:
            - Reference: Cutts & Eglen. Detecting pairwise correlations in
              spike trains. J. Neurosci. 34:43, 14288-14303 (2014).
        """
        return get_sttc(
            self.train[i],
            self.train[j],
            delt,
            self.length,
            start_time=self.start_time,
        )

    def get_pairwise_ccg(
        self,
        compare_func=compute_cross_correlation_with_lag,
        bin_size=1.0,
        max_lag=350,
        n_jobs=-1,
    ):
        """Compute pairwise cross-correlogram matrices from binned spike arrays.

        Bins the spike trains into a binary raster and computes the pairwise
        similarity between all unit pairs using lagged cross-correlation
        (default) or lagged cosine similarity.

        Parameters:
            compare_func (callable): Comparison function from utils. Must
                accept (ref_signal, comp_signal, max_lag=int) and return
                (score, lag). Default is
                compute_cross_correlation_with_lag.
            bin_size (float): Bin size in milliseconds for the binary raster
                (default: 1.0).
            max_lag (float): Maximum lag in milliseconds to search for the
                peak correlation (default: 350). Converted to bins
                internally.
            n_jobs (int): Number of threads for parallel computation. -1
                uses all cores (default), 1 disables parallelism, None is
                serial.

        Returns:
            corr_matrix (PairwiseCompMatrix): Matrix of maximum correlation
                coefficients between all unit pairs. Diagonal is always 1.
            lag_matrix (PairwiseCompMatrix): Matrix of time lags in bins at
                which maximum correlation occurs. Positive lag means unit j
                leads unit i. Diagonal is always 0.
        """
        raster_matrix = self.raster(bin_size)
        num_units = raster_matrix.shape[0]
        max_lag_bins = int(round(max_lag / bin_size))

        corr_matrix = np.full((num_units, num_units), np.nan)
        lag_matrix = np.full((num_units, num_units), np.nan)

        pairs = [(n1, n2) for n1 in range(num_units) for n2 in range(n1, num_units)]

        def _compute_pair(pair):
            n1, n2 = pair
            return pair, compare_func(
                raster_matrix[n1, :], raster_matrix[n2, :], max_lag=max_lag_bins
            )

        n_workers = _resolve_n_jobs(n_jobs)
        if n_workers > 1 and len(pairs) > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = pool.map(_compute_pair, pairs)
        else:
            results = map(_compute_pair, pairs)

        for (n1, n2), (max_corr, max_lag_idx) in results:
            corr_matrix[n1, n2] = max_corr
            lag_matrix[n1, n2] = max_lag_idx
            corr_matrix[n2, n1] = max_corr
            lag_matrix[n2, n1] = -max_lag_idx

        return PairwiseCompMatrix(
            matrix=corr_matrix,
            metadata={"bin_size": bin_size, "max_lag": max_lag},
        ), PairwiseCompMatrix(
            matrix=lag_matrix,
            metadata={"bin_size": bin_size, "max_lag": max_lag},
        )

    def latencies(self, times, window_ms=100.0):
        """Compute latencies from each time to the nearest spike per unit.

        Parameters:
            times (list): List of times.
            window_ms (float): Window in milliseconds (default: 100.0).

        Returns:
            latencies (list): 2D list, each row is a list of latencies from
                a time to the nearest spike in the train.
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

    def get_pairwise_latencies(self, window_ms=None, return_distributions=False):
        """Compute pairwise nearest-spike latency distributions between all unit pairs.

        For each ordered pair (i, j), and for each spike in train i, finds
        the closest spike in train j and records the signed latency
        (t_j - t_i). Both directions are computed independently.

        Parameters:
            window_ms (float or None): If not None, discard latencies where
                the absolute distance exceeds this value (default: None, no
                filtering).
            return_distributions (bool): If True, also return a (U, U)
                numpy object array where entry [i, j] is an ndarray of all
                individual signed latencies from unit i to unit j
                (default: False).

        Returns:
            mean_latency (PairwiseCompMatrix): Matrix of mean signed
                latencies in milliseconds. Entry [i, j] is the average
                latency from each spike in unit i to the nearest spike in
                unit j. Diagonal is 0.
            std_latency (PairwiseCompMatrix): Matrix of latency standard
                deviations. Entry [i, j] is the std of latencies from
                unit i to unit j. Diagonal is 0.
            distributions (numpy.ndarray): Only returned when
                return_distributions is True. A (U, U) object array where
                [i, j] is an ndarray of all signed latencies from unit i
                to unit j.
        """
        N = self.N

        # --- Numba fast path (when distributions are not requested) ---
        from .numba_utils import NUMBA_AVAILABLE

        if NUMBA_AVAILABLE and not return_distributions and N > 1:
            from .numba_utils import flatten_spike_trains, nb_latencies_all_pairs

            flat, offsets = flatten_spike_trains(self.train, self.start_time)
            has_window = window_ms is not None
            w = window_ms if has_window else 0.0
            mean_matrix, std_matrix = nb_latencies_all_pairs(
                flat, offsets, N, w, has_window
            )
            meta = {"window_ms": window_ms}
            return (
                PairwiseCompMatrix(matrix=mean_matrix, metadata=meta),
                PairwiseCompMatrix(matrix=std_matrix, metadata=meta),
            )

        # --- Pure-numpy fallback ---
        mean_matrix = np.zeros((N, N))
        std_matrix = np.zeros((N, N))

        if return_distributions:
            dist_matrix = np.empty((N, N), dtype=object)

        for i in range(N):
            train_i = np.asarray(self.train[i])
            for j in range(N):
                if i == j:
                    if return_distributions:
                        dist_matrix[i, j] = np.array([], dtype=np.float64)
                    continue

                train_j = np.asarray(self.train[j])

                if len(train_i) == 0 or len(train_j) == 0:
                    if return_distributions:
                        dist_matrix[i, j] = np.array([], dtype=np.float64)
                    continue

                # For each spike in train_i, find the closest spike in train_j
                idx = np.searchsorted(train_j, train_i)
                np.clip(idx, 1, len(train_j) - 1, out=idx)

                # Check both the candidate and its predecessor
                dt_right = train_j[idx] - train_i
                dt_left = train_j[idx - 1] - train_i

                # Pick whichever is closer in absolute value
                use_left = np.abs(dt_left) < np.abs(dt_right)
                latencies = np.where(use_left, dt_left, dt_right)

                # Apply window filter
                if window_ms is not None:
                    mask = np.abs(latencies) <= window_ms
                    latencies = latencies[mask]

                if return_distributions:
                    dist_matrix[i, j] = latencies

                if len(latencies) > 0:
                    mean_matrix[i, j] = np.mean(latencies)
                    std_matrix[i, j] = np.std(latencies)

        meta = {"window_ms": window_ms}
        result = (
            PairwiseCompMatrix(matrix=mean_matrix, metadata=meta),
            PairwiseCompMatrix(matrix=std_matrix, metadata=meta),
        )
        if return_distributions:
            return result + (dist_matrix,)
        return result

    def latencies_to_index(self, i, window_ms=100.0):
        """Compute the latency from one unit to all other units.

        Parameters:
            i (int): Index of the unit.
            window_ms (float): Window in milliseconds (default: 100.0).

        Returns:
            latencies (list): 2D list, each row is a list of latencies per
                neuron.
        """
        return self.latencies(self.train[i], window_ms)

    def get_frac_active(self, edges, MIN_SPIKES, backbone_threshold, bin_size=1.0):
        """Compute fraction of units active per burst and backbone identity.

        Parameters:
            edges (numpy.ndarray): Array of shape (B, 2) containing
                [start, end] indices for each burst. Indices are in raster
                bin coordinates (bin index = time_ms / bin_size).
            MIN_SPIKES (int): Minimum number of spikes required for a unit
                to be considered active in a burst.
            backbone_threshold (float): Minimum fraction of bursts a unit
                must be active in to be considered a backbone unit (0 to 1).
            bin_size (float): Raster bin size in milliseconds (default 1.0).
                Must match the bin size used to compute ``edges``.

        Returns:
            frac_per_unit (numpy.ndarray): 1D array where each value is the
                fraction of bursts in which the neuron was active.
            frac_per_burst (numpy.ndarray): 1D array where each value is the
                fraction of neurons active in that burst.
            backbone_units (numpy.ndarray): 1D array of the neuron/unit
                indices that are backbone units.
        """
        t_spk_mat = self.sparse_raster(bin_size=bin_size).toarray()

        # Sanity check: edges must fit within the raster dimensions
        raster_bins = t_spk_mat.shape[1]
        if edges.size > 0 and int(edges.max()) > raster_bins:
            raise ValueError(
                f"Edge index {int(edges.max())} exceeds raster size "
                f"({raster_bins} bins at bin_size={bin_size} ms). Ensure "
                f"bin_size matches the value used in "
                f"get_bursts(raster_bin_size_ms=...)."
            )

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
        n_bursts = edges.shape[0]
        if n_bursts == 0:
            frac_per_unit = np.zeros(t_spk_mat.shape[0])
            frac_per_burst = np.array([])
            backbone_units = np.array([], dtype=int)
            return frac_per_unit, frac_per_burst, backbone_units
        frac_per_unit = np.sum(above_thresh, axis=1) / n_bursts
        frac_per_burst = np.sum(above_thresh, axis=0) / t_spk_mat.shape[0]

        backbone_units = np.where(frac_per_unit >= backbone_threshold)[0]
        return frac_per_unit, frac_per_burst, backbone_units

    def get_frac_spikes_in_burst(self, edges, bin_size=1.0):
        """Compute the fraction of each unit's spikes that fall inside burst windows.

        Parameters:
            edges (numpy.ndarray): Array of shape (B, 2) containing
                [start, end] indices for each burst. Indices are in raster
                bin coordinates (bin index = time_ms / bin_size).
            bin_size (float): Raster bin size in milliseconds (default 1.0).
                Must match the bin size used to compute ``edges``.

        Returns:
            frac_spikes_in_burst (numpy.ndarray): 1D array of shape (N,)
                where each value is the fraction of the unit's total spikes
                that fall inside any burst window. NaN for units with zero
                spikes.
        """
        t_spk_mat = self.sparse_raster(bin_size=bin_size).toarray()
        n_units = t_spk_mat.shape[0]
        n_bursts = edges.shape[0]

        total_spikes = t_spk_mat.sum(axis=1)
        frac = np.full(n_units, np.nan)

        if n_bursts == 0:
            return frac

        spikes_in_burst = np.zeros(n_units)
        for unit in range(n_units):
            unit_spk_times = np.where(t_spk_mat[unit, :])[0]
            for burst in range(n_bursts):
                in_burst = unit_spk_times[
                    (unit_spk_times >= edges[burst, 0])
                    & (unit_spk_times <= edges[burst, 1])
                ]
                spikes_in_burst[unit] += len(in_burst)

        has_spikes = total_spikes > 0
        frac[has_spikes] = spikes_in_burst[has_spikes] / total_spikes[has_spikes]
        return frac

    def spike_shuffle(self, swap_per_spike=5, seed=None, bin_size=1):
        """Shuffle the spike matrix using degree-preserving double-edge swaps.

        Parameters:
            swap_per_spike (int): Determines total number of swaps:
                num_spikes * swap_per_spike (default: 5).
            seed (int or None): Random seed for repeatability. None means no
                seed is set (default: None).
            bin_size (int): Number of individual time steps per bin. Bins
                with multiple spikes are binarized to 1. A RuntimeWarning
                is issued when multi-spike bins are detected (default: 1).

        Returns:
            shuffled_spike_data (SpikeData): SpikeData object with shuffled
                spike train matrix.

        Notes:
            - Each neuron's average firing rate is preserved, but the
              specific time bin in which it spikes is shuffled.
            - Each time bin's population rate is preserved, but the specific
              units active in each time bin are shuffled.
            - Every spike swap involves 2 different spikes so on average,
              every spike will get swapped 2*swap_per_spike times.
            - Reference: Okun, M. et al. Population rate dynamics and
              multineuron firing patterns in sensory cortex. J. Neurosci.
              32, 17108-17119 (2012).
        """
        if self.N == 0:
            return SpikeData(
                [],
                length=self.length,
                start_time=self.start_time,
                metadata=self.metadata,
            )

        spk_mat = self.sparse_raster(bin_size=bin_size).toarray()
        if (spk_mat > 1).any():
            warnings.warn(
                "Multi-spike bins detected; binarizing before shuffle "
                "(spike counts not preserved)",
                RuntimeWarning,
            )
        binary_mat = spk_mat > 0
        shuffled_mat = randomize(binary_mat, swap_per_spike=swap_per_spike, seed=seed)
        shuffled_spike_data = SpikeData.from_raster(
            shuffled_mat,
            bin_size,
            length=self.length,
            start_time=self.start_time,
            metadata=self.metadata,
            neuron_attributes=self.neuron_attributes,
        )
        return shuffled_spike_data

    def spike_shuffle_stack(self, n_shuffles, seed=None, swap_per_spike=5, bin_size=1):
        """Generate multiple shuffled copies as a SpikeSliceStack.

        Each shuffle is an independent call to ``spike_shuffle``. The
        resulting stack can be used with ``SpikeSliceStack.apply`` to build
        null distributions for statistical testing.

        Parameters:
            n_shuffles (int): Number of shuffled datasets to generate.
            seed (int or None): Base random seed. Each shuffle uses
                ``seed + i`` for reproducibility. None means no seed.
            swap_per_spike (int): Forwarded to ``spike_shuffle``
                (default: 5).
            bin_size (int): Forwarded to ``spike_shuffle`` (default: 1).

        Returns:
            stack (SpikeSliceStack): Stack of n_shuffles shuffled SpikeData
                objects. All slices share the same time bounds.
        """
        if n_shuffles < 1:
            raise ValueError("n_shuffles must be at least 1.")

        from .spikeslicestack import SpikeSliceStack

        shuffled = []
        for i in range(n_shuffles):
            s = seed + i if seed is not None else None
            shuffled.append(
                self.spike_shuffle(
                    swap_per_spike=swap_per_spike, seed=s, bin_size=bin_size
                )
            )

        times = [(self.start_time, self.start_time + self.length)] * n_shuffles
        return SpikeSliceStack(
            spike_stack=shuffled,
            times_start_to_end=times,
            neuron_attributes=self.neuron_attributes,
        )

    def subset_stack(self, n_subsets, units_per_subset, seed=None):
        """Generate multiple random unit subsets as a SpikeSliceStack.

        Each subset is drawn by sampling units_per_subset unit indices
        without replacement from the full unit set. Draws are independent
        across subsets (with replacement across draws), so the same unit
        may appear in multiple subsets.

        Parameters:
            n_subsets (int): Number of random subsets to generate.
            units_per_subset (int): Number of units in each subset.
            seed (int or None): Random seed for reproducibility.

        Returns:
            stack (SpikeSliceStack): Stack of n_subsets subsetted SpikeData
                objects. All slices share the same time bounds.

        Notes:
            - The stack-level ``neuron_attributes`` is ``None`` because each
              subset contains a different set of units. Individual
              ``SpikeData`` objects within the stack carry their own
              subsetted attributes.
        """
        if n_subsets < 1:
            raise ValueError("n_subsets must be at least 1.")

        from .spikeslicestack import SpikeSliceStack

        if units_per_subset > self.N:
            raise ValueError(
                f"units_per_subset ({units_per_subset}) exceeds number of "
                f"units ({self.N})"
            )

        rng = np.random.default_rng(seed)
        subsets = []
        for _ in range(n_subsets):
            indices = sorted(rng.choice(self.N, size=units_per_subset, replace=False))
            subsets.append(self.subset(indices))

        times = [(self.start_time, self.start_time + self.length)] * n_subsets
        return SpikeSliceStack(
            spike_stack=subsets,
            times_start_to_end=times,
            drop_slice_attributes=False,
        )

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
        """Export this SpikeData to an HDF5 file with flexible formatting.

        Supports four storage styles: 'raster' (dense 2D array), 'ragged'
        (flat spike times with index array), 'group' (separate dataset per
        unit), and 'paired' (parallel index and time arrays).

        Parameters:
            filepath (str): Path to the output HDF5 file.
            style (str): Storage format style ('raster', 'ragged', 'group',
                or 'paired'). Defaults to 'ragged'.
            raster_dataset (str): Dataset name for raster data
                (style='raster').
            raster_bin_size_ms (float): Bin size in milliseconds for
                rasterization. Required for 'raster' style.
            spike_times_dataset (str): Dataset name for flat spike times
                (style='ragged').
            spike_times_index_dataset (str): Dataset name for cumulative
                spike counts per unit (style='ragged').
            spike_times_unit (str): Time unit for spike times in ragged
                format ('ms', 's', or 'samples').
            fs_Hz (float): Sampling frequency in Hz, required when
                converting to 'samples' unit.
            group_per_unit (str): Group name containing per-unit datasets
                (style='group').
            group_time_unit (str): Time unit for individual unit datasets
                ('ms', 's', or 'samples').
            idces_dataset (str): Dataset name for unit indices
                (style='paired').
            times_dataset (str): Dataset name for spike times
                (style='paired').
            times_unit (str): Time unit for paired times ('ms', 's', or
                'samples').
            raw_dataset (str or None): Reserved for future raw data export.
            raw_time_dataset (str or None): Reserved for future raw time
                axis export.
            raw_time_unit (str): Time unit for raw data timestamps ('ms',
                's', or 'samples').

        Notes:
            - All spike times are stored internally in milliseconds and
              converted to the requested output unit.
            - When using 'samples' unit, fs_Hz must be provided for proper
              conversion.
        """
        # Import locally to avoid import cycles at module import time
        from ..data_loaders.data_exporters import export_spikedata_to_hdf5

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
        """Export this SpikeData to a minimal NWB-compatible HDF5 file.

        Stores spike times in the standard '/units' group format for
        round-tripping with the NWB loader.

        Parameters:
            filepath (str): Path to the output NWB file (.nwb extension
                recommended).
            spike_times_dataset (str): Name of the dataset containing
                flattened spike times in seconds. Standard NWB uses
                "spike_times".
            spike_times_index_dataset (str): Name of the dataset containing
                cumulative spike counts per unit for indexing into
                spike_times. Standard NWB uses "spike_times_index".
            group (str): Name of the HDF5 group to contain the spike data.
                Standard NWB uses "units" for the units table.

        Notes:
            - Spike times are automatically converted from internal
              milliseconds to seconds as required by the NWB standard.
            - The output file contains only the essential spike timing data,
              not the full NWB metadata structure.
            - Compatible with both pynwb and h5py-based NWB readers.
        """
        # Import locally to avoid circular imports
        from ..data_loaders.data_exporters import export_spikedata_to_nwb

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
        """Export this SpikeData to a KiloSort/Phy-compatible folder.

        Creates spike_times.npy and spike_clusters.npy arrays for use with
        Phy.

        Parameters:
            folder (str): Output directory path. Will be created if it
                doesn't exist.
            fs_Hz (float): Sampling frequency in Hz. Required for time unit
                conversion, especially when using 'samples'.
            spike_times_file (str): Filename for the spike times array.
                Standard KiloSort uses "spike_times.npy".
            spike_clusters_file (str): Filename for the cluster assignments
                array. Standard KiloSort uses "spike_clusters.npy".
            time_unit (str): Output time unit for spike times ('samples',
                'ms', or 's').
            cluster_ids (list[int] or None): Optional list of cluster IDs
                to assign to each unit. Must have length equal to self.N.
                If None, uses sequential integers [0, 1, 2, ...].

        Returns:
            paths (tuple[str, str]): Paths to the created
                (spike_times_file, spike_clusters_file).

        Notes:
            - Empty units (no spikes) are skipped in the output arrays.
            - Cluster IDs are mapped to units in order, so cluster_ids[i]
              corresponds to unit i in the SpikeData.
            - The 'samples' time unit is most common for KiloSort workflows.
        """
        # Import locally to avoid circular imports
        from ..data_loaders.data_exporters import export_spikedata_to_kilosort

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
        """Compute smoothed population firing rate.

        Smooths the summed spike counts using a moving-average (square)
        window, then a Gaussian smoothing window.

        Parameters:
            square_width (float): Width of square smoothing window in
                milliseconds.
            gauss_sigma (float): Sigma of Gaussian smoothing window in
                milliseconds.
            raster_bin_size_ms (float): Size of raster bins in ms.

        Returns:
            pop_rate (numpy.ndarray): Smoothed population spiking data in
                spikes per bin.

        Notes:
            - ``square_width`` and ``gauss_sigma`` are specified in
              milliseconds and converted to bin counts internally using
              ``raster_bin_size_ms``. With the default
              ``raster_bin_size_ms=1.0``, 1 ms = 1 bin.
            - The returned array index corresponds to raster bin index. For
              event-centered data (start_time < 0), bin 0 corresponds to
              start_time, not t=0. To find the bin for t=0 (the event), use
              ``event_bin = int(-start_time / raster_bin_size_ms)``.
        """
        if gauss_sigma < 0:
            raise ValueError(f"gauss_sigma must be non-negative, got {gauss_sigma}")
        if square_width < 0:
            raise ValueError(f"square_width must be non-negative, got {square_width}")

        # Convert ms to bins
        square_width_bins = max(0, int(round(square_width / raster_bin_size_ms)))
        gauss_sigma_bins = gauss_sigma / raster_bin_size_ms

        if square_width > 0 and square_width_bins < 1:
            warnings.warn(
                f"square_width ({square_width} ms) is smaller than "
                f"raster_bin_size_ms ({raster_bin_size_ms} ms) — "
                f"square smoothing will have no effect.",
                UserWarning,
            )
        if gauss_sigma > 0 and gauss_sigma_bins < 1:
            warnings.warn(
                f"gauss_sigma ({gauss_sigma} ms) is smaller than "
                f"raster_bin_size_ms ({raster_bin_size_ms} ms) — "
                f"Gaussian smoothing will have minimal effect.",
                UserWarning,
            )

        t_spk_mat = self.sparse_raster(
            raster_bin_size_ms
        )  # Shape: (neurons, time_bins)
        summed_spikes = np.asarray(
            t_spk_mat.sum(axis=0)
        ).flatten()  # Sum once across neurons dimension

        # Pass square window
        if square_width_bins > 0:
            square_smooth_summed_spike = np.convolve(
                summed_spikes,
                np.ones(square_width_bins) / square_width_bins,
                mode="same",
            )
        else:
            square_smooth_summed_spike = summed_spikes

        # Pass gaussian window
        if gauss_sigma_bins > 0:
            gauss_window = norm.pdf(
                np.arange(-3 * gauss_sigma_bins, 3 * gauss_sigma_bins + 1),
                0,
                gauss_sigma_bins,
            )
            pop_rate = np.convolve(
                square_smooth_summed_spike,
                gauss_window / np.sum(gauss_window),
                mode="same",
            )
        else:
            pop_rate = square_smooth_summed_spike

        return pop_rate

    def compute_spike_trig_pop_rate(
        self, window_ms=80, cutoff_hz=20, fs=1000, bin_size=1, cut_outer=10
    ):
        """Compute spike-triggered population rate (stPR).

        Implementation of the stPR measure from Bimbard et al., building on
        Okun et al. (Nature, 2015). For each neuron i and lag tau, the
        leave-one-out population rate is computed excluding neuron i. The
        coupling curve measures how much the other neurons' activity
        deviates from their temporal mean at times offset by tau from
        neuron i's spikes.

        Parameters:
            window_ms (int): Half-width of the lag window in ms (window
                from -window_ms to +window_ms).
            cutoff_hz (float): Low-pass Butterworth filter cutoff in Hz
                applied to the coupling curves.
            fs (float): Sampling rate in Hz used for filter design.
            bin_size (float): Bin size in ms for the spike raster.
            cut_outer (int): Number of outer lag bins to ignore.

        Returns:
            stPR_filtered (numpy.ndarray): Low-pass filtered coupling
                curves for every neuron, shape (N, 2*window_ms + 1).
            coupling_strengths_zero_lag (numpy.ndarray): Coupling strength
                at lag 0, shape (N,).
            coupling_strengths_max (numpy.ndarray): Peak coupling strength
                within the trimmed lag window, shape (N,).
            delays (numpy.ndarray): Lag (in ms) at which peak coupling
                occurs, shape (N,). Positive means neuron leads population.
            lags (numpy.ndarray): Array of lag values from -window_ms to
                +window_ms.
        """
        if window_ms < 1:
            raise ValueError("window_ms must be at least 1.")
        if self.N < 2:
            raise ValueError("compute_spike_trig_pop_rate requires at least 2 units.")

        # Bin spike data to a spike matrix
        spike_matrix = self.sparse_raster(bin_size=bin_size).toarray()

        # Get dimensions
        num_neurons, num_bins = spike_matrix.shape

        # Prepare lags: τ values from −window_ms to +window_ms
        lags = np.arange(-window_ms, window_ms + 1)

        # --- Numba fast path ---
        from .numba_utils import NUMBA_AVAILABLE

        if NUMBA_AVAILABLE:
            from .numba_utils import nb_spike_trig_pop_rate

            spike_f64 = spike_matrix.astype(np.float64)
            stPR = nb_spike_trig_pop_rate(spike_f64, lags)
        else:
            # --- Pure-numpy fallback ---
            # Total population spike count per bin (used for leave-one-out)
            pop_sum = np.sum(spike_matrix, axis=0)

            # μ_i = average firing rate of neuron i (spikes per bin)
            mu = np.mean(spike_matrix, axis=1)
            mu_sum = np.sum(mu)

            # ||f_i|| = total number of spikes fired by neuron i
            total_spikes = np.sum(spike_matrix, axis=1)

            # c_{i,τ} for all neurons, lags
            stPR = np.zeros((num_neurons, len(lags)))

            for i in range(num_neurons):
                # Skip silent neurons or neurons with zero mean rate
                if total_spikes[i] == 0 or mu[i] == 0:
                    continue

                # Leave-one-out population rate: P_{-i}(t)
                P_loo = pop_sum - spike_matrix[i]

                # Temporal mean of leave-one-out population rate: P̄_{-i}
                P_loo_mean = np.mean(P_loo)

                # Σ_{j≠i} μ_j = leave-one-out sum of mean rates
                mu_loo = mu_sum - mu[i]

                # All spike times for neuron i: {s | f_i(s) > 0}
                spike_times = np.where(spike_matrix[i] > 0)[0]

                # Accumulator for Σ[P_{-i}(t) - P̄_{-i}]
                sum_deviations = np.zeros(len(lags))

                for tau_idx, tau in enumerate(lags):
                    valid_t = spike_times + tau
                    mask = (valid_t >= 0) & (valid_t < num_bins)
                    if np.any(mask):
                        deviations = P_loo[valid_t[mask]] - P_loo_mean
                        sum_deviations[tau_idx] = np.sum(deviations)

                # c_{i,τ} = Σ[P_{-i}(t) − P̄_{-i}] / (||f_i|| × Σ_{j≠i} μ_j)
                stPR[i] = sum_deviations / (total_spikes[i] * mu_loo)

        # Low-pass filter coupling curves
        stPR_filtered = np.array(
            [
                butter_filter(stPR[i], highcut=cutoff_hz, fs=fs, order=2)
                for i in range(num_neurons)
            ]
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
        acc_square_width=8,
        acc_gauss_sigma=8,
        raster_bin_size_ms=1.0,
        peak_to_trough=True,
        pop_rate=None,
        pop_rate_acc=None,
        pop_rms_override=None,
    ):
        """Detect bursts using thresholded peak finding and edge detection.

        Parameters:
            thr_burst (float): Threshold multiplier for burst peak detection.
            min_burst_diff (int): Minimum time between detected bursts
                (in bins).
            burst_edge_mult_thresh (float): Threshold multiplier for burst
                edge detection.
            square_width (float): Square window width for calculating pop_rate
                (in milliseconds).
            gauss_sigma (float): Gaussian window sigma for calculating
                pop_rate (in milliseconds).
            acc_square_width (float): Square window width for calculating
                pop_rate_acc (in milliseconds).
            acc_gauss_sigma (float): Gaussian window sigma for calculating
                pop_rate_acc (in milliseconds).
            raster_bin_size_ms (float): Time bin size for calculating
                population rate in ms.
            peak_to_trough (bool): Flag to calculate bursts peak-to-trough
                (True) or peak-to-zero (False).
            pop_rate (numpy.ndarray or None): Pre-calculated smoothed
                population spiking data in spikes per bin.
            pop_rate_acc (numpy.ndarray or None): Pre-calculated accurate
                smoothed population spiking data in spikes per bin.
            pop_rms_override (float or None): RMS override for burst
                threshold baseline.

        Returns:
            tburst (numpy.ndarray): Time bin indices of detected bursts.
            edges (numpy.ndarray): Burst edge indices, shape ``(N, 2)``.
            peak_amp (numpy.ndarray): Amplitudes of bursts at corresponding
                array indices.

        Notes:
            - Will use pop_rate and pop_rate_acc if provided, otherwise will
              calculate using squared widths and sigmas.
            - Using the peak-to-zero calculations may result in several
              bursts being detected at one peak.
            - Returned time bin indices are relative to bin 0 of the raster.
              For event-centered data (``start_time < 0``), convert to
              event-relative ms via
              ``tburst * raster_bin_size_ms + start_time``.
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
            if pop_rms_override <= 0:
                raise ValueError(
                    f"pop_rms_override must be positive, got {pop_rms_override}"
                )
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
            warnings.warn(
                f"{len(tburst) - len(unique_bursts)} duplicate bursts were detected across the following times: {list(duplicates)}. "
                f"This is likely due to identifying bursts using peak-to-zero calculations. Consider setting the PEAK-TO-TROUGH flag to True. "
                f"Otherwise, consider increasing burst_edge_mult_thresh if this burst duration is longer than you would expect for your data. "
                f"Alternatively, increase min_burst_diff to prevent two bursts from being detected too close to each other.",
                RuntimeWarning,
            )

        return tburst, edges, peak_amp

    def burst_sensitivity(
        self,
        thr_values,
        dist_values,
        burst_edge_mult_thresh,
        square_width=20,
        gauss_sigma=100,
        acc_square_width=8,
        acc_gauss_sigma=8,
        raster_bin_size_ms=1.0,
        peak_to_trough=True,
        pop_rate=None,
        pop_rate_acc=None,
        pop_rms_override=None,
    ):
        """Sweep burst detection parameters and return burst counts.

        Calls ``get_bursts`` for every combination of ``thr_values`` and
        ``dist_values``, holding ``burst_edge_mult_thresh`` constant.

        Parameters:
            thr_values (array-like): 1-D array of ``thr_burst`` values to
                sweep.
            dist_values (array-like): 1-D array of ``min_burst_diff`` values
                (in bins) to sweep.
            burst_edge_mult_thresh (float): Held constant during the sweep.
            square_width (int): Square window width for pop_rate (in bins).
            gauss_sigma (int): Gaussian window sigma for pop_rate (in bins).
            acc_square_width (int): Square window width for pop_rate_acc
                (in bins).
            acc_gauss_sigma (int): Gaussian window sigma for pop_rate_acc
                (in bins).
            raster_bin_size_ms (float): Time bin size for population rate
                in ms.
            peak_to_trough (bool): Peak-to-trough (True) or peak-to-zero
                (False) burst detection.
            pop_rate (numpy.ndarray or None): Pre-computed smoothed
                population rate.
            pop_rate_acc (numpy.ndarray or None): Pre-computed accurate
                smoothed population rate.
            pop_rms_override (float or None): RMS override for burst
                threshold baseline.

        Returns:
            burst_counts (numpy.ndarray): Integer array of shape
                (len(thr_values), len(dist_values)) with the number of
                detected bursts for each parameter combination.

        Notes:
            - Either ``thr_values`` or ``dist_values`` can have length 1 to
              focus the sensitivity analysis on a single parameter.
            - Pre-computing ``pop_rate`` and ``pop_rate_acc`` and passing
              them in avoids redundant smoothing inside the loop.
        """
        thr_values = np.asarray(thr_values)
        dist_values = np.asarray(dist_values)

        # Pre-compute population rates once if not provided
        if pop_rate is None:
            pop_rate = self.get_pop_rate(
                square_width, gauss_sigma, raster_bin_size_ms=raster_bin_size_ms
            )
        if pop_rate_acc is None:
            pop_rate_acc = self.get_pop_rate(
                acc_square_width, acc_gauss_sigma, raster_bin_size_ms=raster_bin_size_ms
            )

        burst_counts = np.empty((len(thr_values), len(dist_values)), dtype=int)

        for i, thr in enumerate(thr_values):
            for j, dist in enumerate(dist_values):
                tburst, _, _ = self.get_bursts(
                    thr_burst=float(thr),
                    min_burst_diff=int(dist),
                    burst_edge_mult_thresh=burst_edge_mult_thresh,
                    peak_to_trough=peak_to_trough,
                    pop_rate=pop_rate,
                    pop_rate_acc=pop_rate_acc,
                    pop_rms_override=pop_rms_override,
                )
                burst_counts[i, j] = len(tburst)

        return burst_counts

    def fit_gplvm(
        self,
        bin_size_ms=50.0,
        movement_variance=1.0,
        tuning_lengthscale=10.0,
        n_latent_bin=100,
        n_iter=20,
        n_time_per_chunk=10000,
        random_seed=3,
        model_class=None,
        **model_kwargs,
    ):
        """Fit a Gaussian Process Latent Variable Model to binned spike counts.

        Bins the spike data into a spike count matrix, fits a GPLVM model
        via expectation-maximisation, decodes latent states, and returns
        the results together with a unit reordering based on tuning peaks.

        Parameters:
            bin_size_ms (float): Bin width in milliseconds for spike count
                matrix.
            movement_variance (float): Movement variance hyperparameter for
                the GPLVM transition kernel.
            tuning_lengthscale (float): Lengthscale hyperparameter for the
                tuning curve kernel.
            n_latent_bin (int): Number of latent bins (discretisation of
                the latent space).
            n_iter (int): Number of EM iterations.
            n_time_per_chunk (int): Number of time bins per chunk for
                chunked inference (controls memory usage).
            random_seed (int): Random seed for JAX PRNG key.
            model_class (class or None): Model class to use. Defaults to
                ``PoissonGPLVMJump1D`` from ``poor_man_gplvm``.
            **model_kwargs: Additional keyword arguments passed to the model
                constructor (e.g. ``p_move_to_jump``, ``basis_type``).

        Returns:
            result (dict): Dictionary with keys: ``"decode_res"`` (decoded
                latent state dictionary), ``"log_marginal_l"`` (array of
                log marginal likelihoods per EM iteration),
                ``"reorder_indices"`` (unit reordering indices based on
                tuning curve peaks), ``"model"`` (the fitted model object),
                ``"binned_spike_counts"`` (the (T, N) binned spike count
                matrix), and ``"bin_size_ms"`` (the bin width used).

        Notes:
            - Requires ``poor_man_gplvm`` and ``jax``. Install with
              ``pip install poor-man-gplvm jax jaxlib``.
            - The binned spike count matrix has shape ``(T, N)`` where T is
              the number of time bins and N is the number of units.
            - To compute metrics from the fitted model, see the GPLVM
              utility functions in ``spikedata.utils``.
        """
        try:
            import poor_man_gplvm as pmg
            import poor_man_gplvm.utils as pmg_utils
            import jax.random as jr
        except ImportError as e:
            raise ImportError(
                "fit_gplvm requires 'poor_man_gplvm' and 'jax'. "
                "Install with: pip install poor-man-gplvm jax jaxlib jaxopt optax"
            ) from e

        if model_class is None:
            model_class = pmg.PoissonGPLVMJump1D

        # Build (T, N) binned spike count matrix
        binned_spk_mat = self.raster(bin_size_ms).T

        # Initialise model
        model = model_class(
            n_neuron=binned_spk_mat.shape[1],
            n_latent_bin=n_latent_bin,
            movement_variance=movement_variance,
            tuning_lengthscale=tuning_lengthscale,
            **model_kwargs,
        )

        # Fit via EM
        em_res = model.fit_em(
            binned_spk_mat,
            key=jr.PRNGKey(random_seed),
            n_iter=n_iter,
            n_time_per_chunk=n_time_per_chunk,
        )

        log_marginal_l = np.asarray(em_res["log_marginal_l"])

        # Decode latent states
        decode_res = model.decode_latent(binned_spk_mat)

        # Convert decode_res values from JAX arrays to numpy
        decode_res = {
            k: np.asarray(v) if hasattr(v, "shape") else v
            for k, v in decode_res.items()
        }

        # Get unit reordering by tuning curve peaks
        sort_res = pmg_utils.post_fit_sort_neuron(em_res)
        reorder_indices = np.asarray(sort_res["argsort"])

        return {
            "decode_res": decode_res,
            "log_marginal_l": log_marginal_l,
            "reorder_indices": reorder_indices,
            "model": model,
            "binned_spike_counts": np.asarray(binned_spk_mat),
            "bin_size_ms": bin_size_ms,
        }

    def plot(self, **kwargs):
        """Assemble a multi-panel column figure from this SpikeData object.

        Thin wrapper around ``plot_utils.plot_recording(self, **kwargs)``.
        See ``plot_recording`` for the full list of parameters.

        Parameters:
            **kwargs: All keyword arguments are forwarded to
                ``plot_recording``.

        Returns:
            fig (matplotlib.Figure): The assembled figure.
        """
        from .plot_utils import plot_recording

        return plot_recording(self, **kwargs)

    def plot_spatial_network(
        self,
        ax,
        matrix,
        edge_threshold=None,
        top_pct=None,
        node_size_range=(2, 20),
        node_cmap="viridis",
        node_linewidth=0.2,
        edge_color="red",
        edge_linewidth=0.6,
        edge_alpha_range=(0.15, 1.0),
        scale_bar_um=500,
        font_size=None,
    ):
        """Plot units at their MEA positions with pairwise edges.

        Unit positions are extracted from ``neuron_attributes`` (requires
        ``'x'`` and ``'y'`` keys). Thin wrapper around
        ``plot_utils.plot_spatial_network``.

        Parameters:
            ax (matplotlib.axes.Axes): Target axes.
            matrix (numpy.ndarray): Symmetric (N, N) pairwise matrix (e.g.
                correlation). Diagonal values are ignored.
            edge_threshold (float or None): Minimum matrix value to draw an
                edge.
            top_pct (float or None): Percentage of top edges to draw.
            node_size_range (tuple): (min_size, max_size) in points squared
                for scatter markers.
            node_cmap (str): Matplotlib colourmap for node colour.
            node_linewidth (float): Outline width of node markers.
            edge_color (str): Colour for network edges.
            edge_linewidth (float): Line width for network edges.
            edge_alpha_range (tuple): (min_alpha, max_alpha) for edge
                transparency.
            scale_bar_um (float): Scale bar length in micrometres (0 to
                omit).
            font_size (int or None): Font size for scale bar label.

        Returns:
            scatter (matplotlib.collections.PathCollection): The scatter
                artist, useful for adding a colorbar.
        """
        from .plot_utils import plot_spatial_network

        if self.neuron_attributes is None:
            raise ValueError(
                "neuron_attributes is None — cannot extract unit positions."
            )
        positions = self.unit_locations
        if positions is None:
            raise ValueError(
                "neuron_attributes must contain 'x'/'y', 'location', or 'position' "
                "keys for spatial plotting."
            )
        return plot_spatial_network(
            ax,
            positions,
            matrix,
            edge_threshold=edge_threshold,
            top_pct=top_pct,
            node_size_range=node_size_range,
            node_cmap=node_cmap,
            node_linewidth=node_linewidth,
            edge_color=edge_color,
            edge_linewidth=edge_linewidth,
            edge_alpha_range=edge_alpha_range,
            scale_bar_um=scale_bar_um,
            font_size=font_size,
        )

    def plot_aligned_pop_rate(
        self,
        events=None,
        pre_ms=250,
        post_ms=500,
        ax=None,
        pop_rate=None,
        pop_rate_params=None,
        color=None,
        label=None,
        linewidth=1.5,
        show_individual=False,
        individual_alpha=0.15,
        individual_linewidth=0.5,
        burst_edges=None,
        edge_percentile=None,
        xlabel="Time from event (ms)",
        ylabel="Pop. rate",
        font_size=None,
    ):
        """Plot the average population rate aligned to events.

        Cuts the population rate around each event and plots the mean trace.
        When ``events`` is None, burst peaks are auto-detected via
        ``self.get_bursts()``.

        Parameters:
            events (array-like or None): Event times in ms. If None, burst
                peaks are auto-detected.
            pre_ms (float): Window duration before each event in ms.
            post_ms (float): Window duration after each event in ms.
            ax (matplotlib.axes.Axes or None): Target axes. If None, a new
                figure is created.
            pop_rate (numpy.ndarray or None): Pre-computed population rate
                (1-ms bins, full recording). If None, computed via
                ``self.get_pop_rate()``.
            pop_rate_params (dict or None): Keyword arguments forwarded to
                ``self.get_pop_rate()`` when ``pop_rate`` is None. Defaults:
                ``square_width=5, gauss_sigma=5``.
            color (str or None): Colour for the mean trace. If None, uses
                the first colour from the default colour cycle.
            label (str or None): Legend label for the mean trace.
            linewidth (float): Line width for the mean trace.
            show_individual (bool): If True, plot each individual event's
                pop-rate trace as a thin line behind the average.
            individual_alpha (float): Alpha for individual traces.
            individual_linewidth (float): Line width for individual traces.
            burst_edges (numpy.ndarray or None): Per-event [start_ms,
                end_ms] boundaries, shape (B, 2).
            edge_percentile (float or None): Percentile (0-100) controlling
                how conservatively the edge markers are placed. When set,
                vertical dashed lines are drawn at the resulting positions.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            font_size (int or None): Font size for labels and ticks. If
                None, uses current rcParams.

        Returns:
            avg_rate (numpy.ndarray): The mean population rate across
                events, shape (pre_ms + post_ms,).

        Notes:
            - To compare multiple conditions, call this method once per
              condition on the same ``ax``. Each call draws its own trace
              with its own ``color`` and ``label``; add ``ax.legend()``
              after the last call.
        """
        from .plot_utils import _import_matplotlib, _apply_font_size

        plt, _ = _import_matplotlib()

        # --- Auto-detect bursts if events not provided --------------------
        if events is None:
            tburst, auto_edges, _ = self.get_bursts(
                thr_burst=2.5,
                min_burst_diff=1000,
                burst_edge_mult_thresh=0.2,
            )
            events = tburst.astype(float)
            if edge_percentile is not None and burst_edges is None:
                burst_edges = auto_edges.astype(float)
        else:
            events = np.asarray(events, dtype=float).ravel()

        # --- Compute or validate pop_rate ---------------------------------
        if pop_rate is None:
            params = {"square_width": 5, "gauss_sigma": 5}
            if pop_rate_params is not None:
                params.update(pop_rate_params)
            pop_rate = self.get_pop_rate(**params)
        pop_rate = np.asarray(pop_rate, dtype=float).ravel()

        # --- Cut windows and collect slices --------------------------------
        window_len = int(pre_ms) + int(post_ms)
        slices = []
        for t in events:
            t0 = int(t) - int(pre_ms)
            t1 = int(t) + int(post_ms)
            if t0 >= 0 and t1 <= len(pop_rate):
                slices.append(pop_rate[t0:t1])

        if len(slices) == 0:
            raise ValueError(
                "No valid event windows found. Check that events fall within "
                "the recording and the window does not extend past boundaries."
            )

        slices = np.array(slices)  # (n_valid_events, window_len)
        avg_rate = np.mean(slices, axis=0)

        # --- Create axes if needed ----------------------------------------
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots()

        t_axis = np.arange(window_len) - int(pre_ms)

        # --- Resolve colour -----------------------------------------------
        if color is None:
            cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            color = cycle_colors[0]

        # --- Individual traces --------------------------------------------
        if show_individual:
            for s in slices:
                ax.plot(
                    t_axis,
                    s,
                    color=color,
                    alpha=individual_alpha,
                    linewidth=individual_linewidth,
                    zorder=1,
                )

        # --- Mean trace ---------------------------------------------------
        ax.plot(
            t_axis,
            avg_rate,
            color=color,
            linewidth=linewidth,
            label=label,
            zorder=2,
        )

        # --- Edge markers -------------------------------------------------
        if edge_percentile is not None:
            if burst_edges is None:
                raise ValueError(
                    "burst_edges is required when events are user-provided "
                    "and edge_percentile is not None."
                )
            burst_edges = np.asarray(burst_edges, dtype=float)
            starts_rel = burst_edges[:, 0] - events[: len(burst_edges)]
            ends_rel = burst_edges[:, 1] - events[: len(burst_edges)]
            start_marker = np.percentile(starts_rel, edge_percentile)
            end_marker = np.percentile(ends_rel, 100 - edge_percentile)
            ax.axvline(
                start_marker,
                color=color,
                linewidth=0.7,
                linestyle=":",
                alpha=0.7,
            )
            ax.axvline(
                end_marker,
                color=color,
                linewidth=0.7,
                linestyle=":",
                alpha=0.7,
            )

        # --- Axes formatting ----------------------------------------------
        ax.set_xlim(t_axis[0], t_axis[-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if font_size is not None:
            _apply_font_size(ax, font_size)

        return avg_rate

    # ------------------------------------------------------------------
    # Curation
    # ------------------------------------------------------------------

    def curate_by_min_spikes(self, min_spikes=30):
        """Remove units with fewer than *min_spikes* spikes.

        See ``spikelab.spikedata.curation.curate_by_min_spikes`` for
        full documentation.
        """
        from .curation import curate_by_min_spikes

        return curate_by_min_spikes(self, min_spikes=min_spikes)

    def curate_by_firing_rate(self, min_rate_hz=0.05):
        """Remove units whose firing rate is below *min_rate_hz*.

        See ``spikelab.spikedata.curation.curate_by_firing_rate`` for
        full documentation.
        """
        from .curation import curate_by_firing_rate

        return curate_by_firing_rate(self, min_rate_hz=min_rate_hz)

    def curate_by_isi_violations(
        self, max_violation=1.0, threshold_ms=1.5, min_isi_ms=0.0, method="percent"
    ):
        """Remove units with excessive ISI violations.

        See ``spikelab.spikedata.curation.curate_by_isi_violations``
        for full documentation.
        """
        from .curation import curate_by_isi_violations

        return curate_by_isi_violations(
            self,
            max_violation=max_violation,
            threshold_ms=threshold_ms,
            min_isi_ms=min_isi_ms,
            method=method,
        )

    def curate_by_snr(self, min_snr=5.0, ms_before=1.0, ms_after=2.0):
        """Remove units whose SNR is below *min_snr*.

        See ``spikelab.spikedata.curation.curate_by_snr`` for full
        documentation.
        """
        from .curation import curate_by_snr

        return curate_by_snr(
            self, min_snr=min_snr, ms_before=ms_before, ms_after=ms_after
        )

    def curate_by_std_norm(
        self,
        max_std_norm=1.0,
        at_peak=True,
        window_ms_before=0.5,
        window_ms_after=1.5,
        ms_before=1.0,
        ms_after=2.0,
    ):
        """Remove units whose normalized waveform STD exceeds *max_std_norm*.

        See ``spikelab.spikedata.curation.curate_by_std_norm`` for full
        documentation.
        """
        from .curation import curate_by_std_norm

        return curate_by_std_norm(
            self,
            max_std_norm=max_std_norm,
            at_peak=at_peak,
            window_ms_before=window_ms_before,
            window_ms_after=window_ms_after,
            ms_before=ms_before,
            ms_after=ms_after,
        )

    def compute_waveform_metrics(
        self,
        ms_before=1.0,
        ms_after=2.0,
        at_peak=True,
        window_ms_before=0.5,
        window_ms_after=1.5,
    ):
        """Compute average waveforms, SNR, and normalized STD for all units.

        Stores results in ``neuron_attributes``.  See
        ``spikelab.spikedata.curation.compute_waveform_metrics`` for
        full documentation.
        """
        from .curation import compute_waveform_metrics

        return compute_waveform_metrics(
            self,
            ms_before=ms_before,
            ms_after=ms_after,
            at_peak=at_peak,
            window_ms_before=window_ms_before,
            window_ms_after=window_ms_after,
        )

    def curate(self, **kwargs):
        """Apply multiple curation criteria in sequence (intersection).

        See ``spikelab.spikedata.curation.curate`` for full
        documentation and supported keyword arguments.
        """
        from .curation import curate

        return curate(self, **kwargs)

    def curate_by_merge_duplicates(self, **kwargs):
        """Remove duplicate units by merging nearby pairs with similar waveforms.

        See ``spikelab.spikedata.curation.curate_by_merge_duplicates`` for
        full documentation and supported keyword arguments.
        """
        from .curation import curate_by_merge_duplicates

        return curate_by_merge_duplicates(self, **kwargs)

    @staticmethod
    def build_curation_history(sd_original, sd_curated, results, parameters=None):
        """Translate curation results into a serializable history dict.

        See ``spikelab.spikedata.curation.build_curation_history`` for
        full documentation.
        """
        from .curation import build_curation_history

        return build_curation_history(
            sd_original,
            sd_curated,
            results,
            parameters=parameters,
        )

    def split_epochs(self):
        """Split a concatenated SpikeData into per-epoch SpikeData objects.

        Uses ``metadata["rec_chunks_ms"]`` (list of ``(start_ms, end_ms)``
        tuples) to slice this SpikeData via ``subtime``.  Each resulting
        SpikeData receives the corresponding epoch template from
        ``neuron_attributes["epoch_templates"]`` as its ``"template"``
        attribute.

        Parameters:
            None.  Epoch boundaries and names are read from ``metadata``.

        Returns:
            epochs (list[SpikeData]): One SpikeData per epoch, time-shifted
                so each starts at t=0.

        Notes:
            - Requires ``metadata["rec_chunks_ms"]``.  Raises ``ValueError``
              if not present.
            - ``metadata["rec_chunk_names"]`` (optional) is stored as
              ``metadata["source_file"]`` on each output SpikeData.
        """
        chunks_ms = self.metadata.get("rec_chunks_ms")
        if chunks_ms is None or len(chunks_ms) == 0:
            raise ValueError(
                "No epoch boundaries found in metadata['rec_chunks_ms']. "
                "This SpikeData was not created from concatenated recordings."
            )

        # Pre-read epoch templates from the original before subtime
        # shares the neuron_attributes dicts by reference.
        epoch_templates_per_unit = []
        if self.neuron_attributes is not None:
            for attrs in self.neuron_attributes:
                epoch_templates_per_unit.append(attrs.get("epoch_templates"))

        chunk_names = self.metadata.get("rec_chunk_names")
        epochs = []

        for i, (start_ms, end_ms) in enumerate(chunks_ms):
            sd_epoch = self.subtime(start_ms, end_ms)

            # Give each epoch its own copy of neuron_attributes and metadata
            # since subtime shares them by reference.
            if sd_epoch.neuron_attributes is not None:
                sd_epoch.neuron_attributes = [
                    dict(a) for a in sd_epoch.neuron_attributes
                ]
                for j, attrs in enumerate(sd_epoch.neuron_attributes):
                    if j < len(epoch_templates_per_unit):
                        et = epoch_templates_per_unit[j]
                        if et is not None and i < len(et):
                            attrs["template"] = et[i]
                    attrs.pop("epoch_templates", None)

            sd_epoch.metadata = dict(sd_epoch.metadata)
            # Remove concatenation metadata from individual epochs
            sd_epoch.metadata.pop("rec_chunks_ms", None)
            sd_epoch.metadata.pop("rec_chunks_frames", None)
            sd_epoch.metadata.pop("rec_chunk_names", None)

            if chunk_names is not None and i < len(chunk_names):
                sd_epoch.metadata["source_file"] = chunk_names[i]
            sd_epoch.metadata["epoch_index"] = i

            epochs.append(sd_epoch)

        return epochs

    def compare_sorter(
        self,
        other: "SpikeData",
        comparison_type: Literal["spike_times", "waveforms"] = "spike_times",
        delta_ms: float = 0.4,
        f_rel_to_trough: Tuple[int, int] = (20, 40),
        max_lag: int = 5,
        n_jobs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compare this sorter output against another SpikeData object.

        Implements the same comparison methodology as SpikeInterface's
        ``SymmetricSortingComparison`` (Buccino et al., eLife 2020): greedy
        spike matching within a temporal window followed by Jaccard agreement
        scoring. Use :meth:`best_match_assignment` on the returned agreement
        matrix to obtain the optimal unit mapping via the Hungarian algorithm
        (equivalent to SpikeInterface's ``get_best_unit_match``).

        Parameters:
            other (SpikeData): SpikeData from a different sorter to compare
                against.
            comparison_type: ``"spike_times"`` for spike-train agreement or
                ``"waveforms"`` for template footprint similarity.
            delta_ms (float): Maximum temporal distance (ms) for a spike
                match (spike_times only).
            f_rel_to_trough (tuple[int, int]): ``(pre, post)`` sample window
                around the trough for footprint construction (waveforms only).
            max_lag (int): Maximum lag in samples for footprint cosine
                similarity search (waveforms only).
            n_jobs (int or None): Number of parallel workers. ``None`` or 1
                for serial execution, -1 for all cores.

        Returns:
            result (dict): Comparison output dictionary containing:
                - ``labels_1`` / ``labels_2``: unit indices
                - ``metadata``: comparison settings
                - For ``spike_times``: ``agreement``, ``frac_1``, ``frac_2``
                - For ``waveforms``: ``similarity``

        References:
            Buccino et al., "SpikeInterface, a unified framework for spike
            sorting", eLife (2020). https://doi.org/10.7554/eLife.61834
        """
        labels_1 = list(range(self.N))
        labels_2 = list(range(other.N))
        n_workers = _resolve_n_jobs(n_jobs)

        if comparison_type == "spike_times":
            M, N = self.N, other.N

            from .numba_utils import NUMBA_AVAILABLE

            if NUMBA_AVAILABLE and M > 0 and N > 0:
                from .numba_utils import (
                    flatten_spike_trains,
                    nb_agreement_all_pairs,
                )

                flat1, offsets1 = flatten_spike_trains(self.train)
                flat2, offsets2 = flatten_spike_trains(other.train)
                agreement, frac_1, frac_2 = nb_agreement_all_pairs(
                    flat1, offsets1, M, flat2, offsets2, N, delta_ms
                )
            else:
                agreement = np.zeros((M, N))
                frac_1 = np.zeros((M, N))
                frac_2 = np.zeros((M, N))

                pairs = [(i, j) for i in range(M) for j in range(N)]

                if n_workers <= 1 or len(pairs) == 0:
                    for i, j in pairs:
                        a, r1, r2 = _compute_agreement_score(
                            self.train[i], other.train[j], delta_ms
                        )
                        agreement[i, j] = a
                        frac_1[i, j] = r1
                        frac_2[i, j] = r2
                else:
                    trains_self = self.train
                    trains_other = other.train

                    def _score_pair(pair):
                        i, j = pair
                        return (
                            i,
                            j,
                            *_compute_agreement_score(
                                trains_self[i], trains_other[j], delta_ms
                            ),
                        )

                    with ThreadPoolExecutor(max_workers=n_workers) as pool:
                        for i, j, a, r1, r2 in pool.map(_score_pair, pairs):
                            agreement[i, j] = a
                            frac_1[i, j] = r1
                            frac_2[i, j] = r2

            return {
                "labels_1": labels_1,
                "labels_2": labels_2,
                "agreement": agreement,
                "frac_1": frac_1,
                "frac_2": frac_2,
                "metadata": {
                    "comparison_type": "spike_times",
                    "delta_ms": delta_ms,
                },
            }

        elif comparison_type == "waveforms":
            M, N = self.N, other.N
            similarity = np.zeros((M, N))
            if M == 0 or N == 0:
                return {
                    "labels_1": labels_1,
                    "labels_2": labels_2,
                    "similarity": similarity,
                    "metadata": {
                        "comparison_type": "waveforms",
                        "f_rel_to_trough": f_rel_to_trough,
                        "max_lag": max_lag,
                    },
                }

            required = (
                "template",
                "neighbor_templates",
                "channel",
                "neighbor_channels",
            )
            for label, sd in [("self", self), ("other", other)]:
                if sd.neuron_attributes is None:
                    raise ValueError(
                        f"{label}.neuron_attributes is None. Waveform comparison "
                        "requires 'template', 'neighbor_templates', 'channel', "
                        "and 'neighbor_channels' per unit."
                    )
                for idx, attrs in enumerate(sd.neuron_attributes):
                    for key in required:
                        if key not in attrs:
                            raise ValueError(
                                f"{label}.neuron_attributes[{idx}] is missing "
                                f"required key '{key}'."
                            )

            all_channels: List[int] = []
            self_attrs: List[Dict[str, Any]] = self.neuron_attributes  # type: ignore[assignment]
            other_attrs: List[Dict[str, Any]] = other.neuron_attributes  # type: ignore[assignment]
            for attrs_list in (self_attrs, other_attrs):
                for attrs in attrs_list:
                    all_channels.append(int(attrs["channel"]))
                    all_channels.extend(
                        int(c) for c in np.asarray(attrs["neighbor_channels"])
                    )
            if not all_channels:
                raise ValueError(
                    "No channels found in neuron_attributes for waveform comparison."
                )
            n_channels = max(all_channels) + 1

            fp_cache_1 = [
                _compute_footprint(self_attrs[i], f_rel_to_trough, n_channels)
                for i in range(M)
            ]
            fp_cache_2 = [
                _compute_footprint(other_attrs[j], f_rel_to_trough, n_channels)
                for j in range(N)
            ]

            pairs = [(i, j) for i in range(M) for j in range(N)]

            if n_workers <= 1 or len(pairs) == 0:
                for i, j in pairs:
                    similarity[i, j] = _compute_footprint_similarity(
                        fp_cache_1[i], fp_cache_2[j], max_lag
                    )
            else:

                def _sim_pair(pair):
                    i, j = pair
                    return (
                        i,
                        j,
                        _compute_footprint_similarity(
                            fp_cache_1[i], fp_cache_2[j], max_lag
                        ),
                    )

                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    for i, j, sim in pool.map(_sim_pair, pairs):
                        similarity[i, j] = sim

            return {
                "labels_1": labels_1,
                "labels_2": labels_2,
                "similarity": similarity,
                "metadata": {
                    "comparison_type": "waveforms",
                    "f_rel_to_trough": f_rel_to_trough,
                    "max_lag": max_lag,
                },
            }

        else:
            raise ValueError(
                f"Unknown comparison_type '{comparison_type}'. "
                "Expected 'spike_times' or 'waveforms'."
            )

    @staticmethod
    def best_match_assignment(
        score_matrix: "NDArray[np.floating]",
        minimize: bool = False,
    ) -> Dict[str, Any]:
        """Compute optimal unit assignment from a pairwise score matrix.

        Uses the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``)
        to find the assignment of rows (sorter 1 units) to columns (sorter 2
        units) that maximizes (or minimizes) the total score. When the matrix
        is non-square, unmatched units are reported separately. This is
        equivalent to SpikeInterface's ``get_best_unit_match`` step.

        Parameters:
            score_matrix (np.ndarray): Pairwise score matrix of shape (M, N),
                e.g. the ``agreement`` or ``similarity`` matrix returned by
                :meth:`compare_sorter`.
            minimize (bool): If True, find the assignment that minimizes the
                total score (useful for distance matrices). Default is False
                (maximize).

        Returns:
            result (dict): Dictionary containing:
                - ``row_indices``: matched row indices (length = min(M, N))
                - ``col_indices``: matched column indices (length = min(M, N))
                - ``scores``: score for each matched pair
                - ``total_score``: sum of matched scores
                - ``unmatched_rows``: row indices with no match (if M > N)
                - ``unmatched_cols``: col indices with no match (if N > M)
                - ``row_order``: full row permutation array (length M) —
                  matched rows first, then unmatched. Apply to any (M, ...)
                  array via ``array[row_order]``.
                - ``col_order``: full column permutation array (length N) —
                  matched cols first, then unmatched. Apply to any (..., N)
                  array via ``array[:, col_order]``.
                - ``reordered_matrix``: score_matrix reordered so that matched
                  pairs lie along the diagonal (equivalent to
                  ``score_matrix[np.ix_(row_order, col_order)]``)
        """
        from scipy.optimize import linear_sum_assignment

        score_matrix = np.asarray(score_matrix, dtype=float)
        if score_matrix.ndim != 2:
            raise ValueError(
                f"score_matrix must be 2-D, got shape {score_matrix.shape}"
            )

        M, N = score_matrix.shape
        if M == 0 or N == 0:
            return {
                "row_indices": np.array([], dtype=int),
                "col_indices": np.array([], dtype=int),
                "scores": np.array([], dtype=float),
                "total_score": 0.0,
                "unmatched_rows": np.arange(M, dtype=int),
                "unmatched_cols": np.arange(N, dtype=int),
                "row_order": np.arange(M, dtype=int),
                "col_order": np.arange(N, dtype=int),
                "reordered_matrix": score_matrix.copy(),
            }

        cost = score_matrix if minimize else -score_matrix
        row_ind, col_ind = linear_sum_assignment(cost)

        scores = score_matrix[row_ind, col_ind]
        total_score = float(scores.sum())

        all_rows = set(range(M))
        all_cols = set(range(N))
        unmatched_rows = np.array(sorted(all_rows - set(row_ind.tolist())), dtype=int)
        unmatched_cols = np.array(sorted(all_cols - set(col_ind.tolist())), dtype=int)

        # Build reordered matrix: matched rows/cols first (in assignment order),
        # then unmatched rows/cols appended.
        row_order = np.concatenate([row_ind, unmatched_rows])
        col_order = np.concatenate([col_ind, unmatched_cols])
        reordered = score_matrix[np.ix_(row_order, col_order)]

        return {
            "row_indices": row_ind,
            "col_indices": col_ind,
            "scores": scores,
            "total_score": total_score,
            "unmatched_rows": unmatched_rows,
            "unmatched_cols": unmatched_cols,
            "row_order": row_order,
            "col_order": col_order,
            "reordered_matrix": reordered,
        }
