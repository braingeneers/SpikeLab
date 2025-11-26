"""
NeuronAttributes module for managing neuron metadata in SpikeData.

This module provides a DataFrame-based system for storing and manipulating
neuron-level attributes such as cluster IDs, electrode positions, firing rates,
and quality metrics.
"""

from typing import Optional, Union, List, Dict, Any
import warnings
import numpy as np
import pandas as pd


class NeuronAttributes:
    """
    Wrapper class for managing neuron attributes as a pandas DataFrame.

    This class provides a structured way to store and manipulate metadata
    for individual neurons in a SpikeData object. The DataFrame is indexed
    by neuron position (0 to N-1), matching the order in SpikeData.train.

    Standard column names (optional but recommended):
        Core attributes:
            - unit_id: Unique identifier for each neuron
            - electrode_id: Physical electrode identifier
            - channel: Recording channel number
            - firing_rate_hz: Mean firing rate in Hz

        Quality metrics:
            - snr: Signal-to-noise ratio
            - amplitude: Spike amplitude (μV or arbitrary units)
            - isolation_distance: Cluster isolation quality
            - isi_violations: ISI violation rate

        Spatial location:
            - unit_x, unit_y, unit_z: Unit position coordinates (μm)
            - electrode_x, electrode_y, electrode_z: Electrode coordinates (μm)

        Note: Store spatial coordinates as separate x, y, z columns for best
        compatibility with pandas operations. Combine using np.column_stack()
        when needed for analysis.

    Additional custom columns are supported and encouraged for analysis-specific
    metadata.

    Attributes
    ----------
    _df : pd.DataFrame
        The underlying DataFrame storing neuron attributes
    n_neurons : int
        Number of neurons (must match length of DataFrame)

    Examples
    --------
    >>> # Create from dictionary
    >>> attrs = NeuronAttributes.from_dict({
    ...     'unit_id': [101, 102, 103],
    ...     'firing_rate_hz': [5.2, 8.1, 3.4]
    ... }, n_neurons=3)
    >>>
    >>> # Access as DataFrame
    >>> df = attrs.to_dataframe()
    >>>
    >>> # Get specific attribute
    >>> rates = attrs.get_attribute('firing_rate_hz')
    >>>
    >>> # Set new attribute
    >>> attrs.set_attribute('quality', ['good', 'good', 'mua'])
    """

    # Standard column names for validation and documentation
    STANDARD_COLUMNS = {
        # Core attributes
        "unit_id",
        "electrode_id",
        "channel",
        "firing_rate_hz",
        # Quality metrics
        "snr",
        "amplitude",
        "isolation_distance",
        "isi_violations",
        # Spatial location (separate coordinate columns)
        "unit_x",
        "unit_y",
        "unit_z",
        "electrode_x",
        "electrode_y",
        "electrode_z",
        # ISI-based temporal patterns
        "mean_isi_ms",
        "median_isi_ms",
        "cv_isi",
        "isi_skewness",
        "burst_index",
        "pause_ratio",
        "refractory_violations",
        # Latency statistics
        "mean_latency_ms",
        "median_latency_ms",
        "latency_jitter_ms",
        # Burst participation
        "burst_participation",
        "is_backbone_unit",
        # Waveform attributes
        "mean_waveform",
        "waveform_channel",
        "waveform_n_spikes",
    }

    def __init__(self, df: pd.DataFrame, n_neurons: int, validate: bool = True):
        """
        Initialize NeuronAttributes from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing neuron attributes. Index should be
            integer-based (0 to n_neurons-1).
        n_neurons : int
            Expected number of neurons. Must match DataFrame length.
        validate : bool, optional
            If True, validate the DataFrame structure and column names.
            Default is True.

        Raises
        ------
        ValueError
            If n_neurons doesn't match DataFrame length or if validation fails.
        """
        self.n_neurons = n_neurons
        self._df = df.copy()

        # Ensure integer index starting from 0
        if len(self._df) > 0:
            self._df.index = pd.RangeIndex(len(self._df))

        if validate:
            self.validate_n_neurons(n_neurons)
            self.validate_columns()

    def validate_n_neurons(self, expected_n: int) -> None:
        """
        Validate that the DataFrame has the expected number of rows.

        Parameters
        ----------
        expected_n : int
            Expected number of neurons.

        Raises
        ------
        ValueError
            If the number of rows doesn't match expected_n.
        """
        if len(self._df) != expected_n:
            raise ValueError(
                f"NeuronAttributes has {len(self._df)} rows but "
                f"expected {expected_n} neurons."
            )

    def validate_columns(self) -> None:
        """
        Validate column names and warn about potential typos.

        Checks for columns that are similar to standard names but don't
        match exactly, which might indicate typos.
        """
        for col in self._df.columns:
            if col not in self.STANDARD_COLUMNS:
                # Check for potential typos in standard columns
                col_lower = col.lower().replace("_", "").replace("-", "")
                for std_col in self.STANDARD_COLUMNS:
                    std_lower = std_col.lower().replace("_", "").replace("-", "")
                    if col_lower == std_lower and col != std_col:
                        warnings.warn(
                            f"Column '{col}' looks like standard column '{std_col}'. "
                            f"Consider using the standard name for consistency.",
                            UserWarning,
                        )

    def set_attribute(
        self, column: str, values: Union[np.ndarray, List, pd.Series]
    ) -> None:
        """
        Set or update a neuron attribute column.

        Parameters
        ----------
        column : str
            Name of the attribute column.
        values : array-like
            Values for the attribute. Must have length equal to n_neurons.

        Raises
        ------
        ValueError
            If values length doesn't match n_neurons.
        """
        values = np.asarray(values)
        if len(values) != self.n_neurons:
            raise ValueError(
                f"Values length ({len(values)}) doesn't match "
                f"n_neurons ({self.n_neurons})."
            )
        self._df[column] = values

    def get_attribute(self, column: str) -> np.ndarray:
        """
        Get values for a specific attribute column.

        Parameters
        ----------
        column : str
            Name of the attribute column.

        Returns
        -------
        np.ndarray
            Array of attribute values.

        Raises
        ------
        KeyError
            If column doesn't exist.
        """
        if column not in self._df.columns:
            raise KeyError(f"Column '{column}' not found in neuron attributes.")
        return self._df[column].values

    def subset(self, indices: Union[List[int], np.ndarray]) -> "NeuronAttributes":
        """
        Select a subset of neurons by their indices.

        Parameters
        ----------
        indices : array-like of int
            Indices of neurons to select.

        Returns
        -------
        NeuronAttributes
            New NeuronAttributes object with selected neurons.
        """
        indices = np.asarray(indices)
        subset_df = self._df.iloc[indices].copy()
        subset_df.index = pd.RangeIndex(len(subset_df))
        return NeuronAttributes(subset_df, n_neurons=len(indices), validate=False)

    def concat(self, other: "NeuronAttributes") -> "NeuronAttributes":
        """
        Concatenate with another NeuronAttributes object.

        Columns that exist in one but not the other will be filled with NaN.

        Parameters
        ----------
        other : NeuronAttributes
            Another NeuronAttributes object to concatenate.

        Returns
        -------
        NeuronAttributes
            New NeuronAttributes with concatenated data.
        """
        combined_df = pd.concat([self._df, other._df], ignore_index=True)
        new_n = self.n_neurons + other.n_neurons
        return NeuronAttributes(combined_df, n_neurons=new_n, validate=False)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Get the underlying DataFrame.

        Returns
        -------
        pd.DataFrame
            Copy of the underlying DataFrame.
        """
        return self._df.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any], n_neurons: int) -> "NeuronAttributes":
        """
        Create NeuronAttributes from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary mapping column names to values.
        n_neurons : int
            Expected number of neurons.

        Returns
        -------
        NeuronAttributes
            New NeuronAttributes object.
        """
        df = pd.DataFrame(data)
        return cls(df, n_neurons=n_neurons, validate=True)

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, n_neurons: Optional[int] = None
    ) -> "NeuronAttributes":
        """
        Create NeuronAttributes from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing neuron attributes.
        n_neurons : int, optional
            Expected number of neurons. If None, inferred from DataFrame length.

        Returns
        -------
        NeuronAttributes
            New NeuronAttributes object.
        """
        if n_neurons is None:
            n_neurons = len(df)
        return cls(df, n_neurons=n_neurons, validate=True)

    def __len__(self) -> int:
        """Return the number of neurons."""
        return self.n_neurons

    def __repr__(self) -> str:
        """Return string representation."""
        return f"NeuronAttributes({self.n_neurons} neurons, {len(self._df.columns)} attributes)\n{self._df}"

    def __str__(self) -> str:
        """Return string representation."""
        return self.__repr__()

    def calculate_mean_waveforms(
        self,
        spikedata: "SpikeData",  # type: ignore
        ms_before: float = 1.0,
        ms_after: float = 2.0,
        max_spikes: Optional[int] = 1000,
        auto_save: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate mean waveforms for each neuron from raw data.

        Extracts the waveform from the channel with the largest amplitude for each unit.
        Automatically stores waveforms as 'mean_waveform' attribute and derived metrics
        in neuron_attributes.

        Parameters
        ----------
        spikedata : SpikeData
            The parent SpikeData object containing spike trains and raw data.
        ms_before : float
            Milliseconds before spike peak to include in waveform.
        ms_after : float
            Milliseconds after spike peak to include in waveform.
        max_spikes : int, optional
            Maximum number of spikes to use per neuron (for efficiency).
            If None, uses all spikes.
        auto_save : bool, optional
            If True (default), automatically saves mean waveforms as 'mean_waveform'
            attribute. Each neuron's waveform is stored as a 1D array (samples).

        Returns
        -------
        dict
            Dictionary with keys:
            - 'mean_waveforms': ndarray of shape (n_neurons, n_samples) - best channel per unit
            - 'std_waveforms': ndarray of shape (n_neurons, n_samples) - std on best channel
            - 'best_channels': ndarray of shape (n_neurons,) - channel index with largest amplitude
            - 'n_spikes_used': ndarray of shape (n_neurons,) - count of spikes used
            - 'time_ms': ndarray with time axis relative to spike (0 = spike peak)

        Raises
        ------
        ValueError
            If SpikeData doesn't have raw_data and raw_time.

        Examples
        --------
        >>> # Calculate and automatically save mean waveforms
        >>> wf_data = sd.neuron_attributes.calculate_mean_waveforms(sd)
        >>> # Access saved waveforms
        >>> mean_waveforms = sd.neuron_attributes.get_attribute('mean_waveform')
        >>> # Get waveform for neuron 0
        >>> neuron_0_waveform = mean_waveforms[0]  # Shape: (n_samples,)
        >>> # Get which channel was used
        >>> best_channel = sd.neuron_attributes.get_attribute('waveform_channel')[0]
        """
        # Check that raw data exists
        if not hasattr(spikedata, "raw_data") or spikedata.raw_data.size == 0:
            raise ValueError(
                "SpikeData must have raw_data and raw_time to extract waveforms. "
                "Load data with raw traces or use a recording loader."
            )

        raw_data = (
            spikedata.raw_data
        )  # Raw data is a 2D array of shape (channels, samples)
        raw_time = spikedata.raw_time  # Raw time is a 1D array of shape (samples)

        # Determine sampling rate in kHz
        if np.ndim(raw_time) == 0:
            # raw_time is already a sampling rate in kHz
            fs_khz = float(raw_time)
        else:
            # Calculate from time vector (assumes uniform sampling)
            dt_ms = np.mean(
                np.diff(raw_time)
            )  # dt_ms is the time between samples in milliseconds
            fs_khz = 1.0 / dt_ms  # fs_khz is the sampling rate in kHz

        # Convert time windows to samples
        before_samples = int(
            ms_before * fs_khz
        )  # before_samples is the number of samples before the spike
        after_samples = int(ms_after * fs_khz) + 1  # +1 to include the spike itself
        total_samples = before_samples + after_samples

        # Get dimensions
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(
                1, -1
            )  # If raw_data is a 1D array, reshape it to a 2D array

        # Initialize output arrays - now only storing best channel per unit
        mean_waveforms = np.zeros((self.n_neurons, total_samples))
        std_waveforms = np.zeros((self.n_neurons, total_samples))
        best_channels = np.zeros(self.n_neurons, dtype=int)
        n_spikes_used = np.zeros(self.n_neurons, dtype=int)

        # Extract waveforms for each neuron
        for neuron_idx in range(self.n_neurons):
            spike_times = spikedata.train[
                neuron_idx
            ]  # spike_times is a 1D array of shape (n_spikes)

            if len(spike_times) == 0:
                continue  # If there are no spikes, skip this neuron

            # Optionally subsample spikes - if max_spikes is not None, subsample the spikes to the maximum number of spikes
            if max_spikes is not None and len(spike_times) > max_spikes:
                indices = np.random.choice(len(spike_times), max_spikes, replace=False)
                spike_times = spike_times[indices]

            # Convert spike times to samples
            spike_samples = (spike_times * fs_khz).astype(
                int
            )  # spike_samples is a 1D array of shape (n_spikes)

            # Extract waveforms for this neuron (all channels)
            waveforms_list = []
            for spike_sample in spike_samples:
                start = spike_sample - before_samples
                end = spike_sample + after_samples

                # Skip if out of bounds
                if start < 0 or end > raw_data.shape[1]:
                    continue

                waveform = raw_data[:, start:end]
                waveforms_list.append(
                    waveform
                )  # waveform is a 2D array of shape (channels, samples)

            if len(waveforms_list) > 0:
                waveforms = np.array(
                    waveforms_list
                )  # waveforms is a 3D array of shape (n_spikes, n_channels, n_samples)
                mean_wf_all_ch = np.mean(waveforms, axis=0)  # (n_channels, n_samples)

                # Find channel with largest peak-to-peak amplitude
                amplitudes = np.max(mean_wf_all_ch, axis=1) - np.min(
                    mean_wf_all_ch, axis=1
                )
                best_ch = np.argmax(
                    amplitudes
                )  # best_ch is the channel index with the largest peak-to-peak amplitude
                best_channels[neuron_idx] = best_ch

                # Store only the best channel waveform
                mean_waveforms[neuron_idx] = mean_wf_all_ch[best_ch]
                std_waveforms[neuron_idx] = np.std(waveforms[:, best_ch, :], axis=0)
                n_spikes_used[neuron_idx] = len(waveforms_list)

        # Create time axis
        time_ms = np.linspace(-ms_before, ms_after, total_samples)

        # Automatically save to neuron_attributes if requested
        if auto_save:
            # Store mean waveforms as array of 1D arrays (one per neuron)
            # Each element is shape (n_samples,) from the best channel
            waveform_objects = np.empty(self.n_neurons, dtype=object)
            for i in range(self.n_neurons):
                waveform_objects[i] = mean_waveforms[i]

            self.set_attribute("mean_waveform", waveform_objects)
            self.set_attribute("waveform_channel", best_channels)
            self.set_attribute("waveform_n_spikes", n_spikes_used)

        return {
            "mean_waveforms": mean_waveforms,
            "std_waveforms": std_waveforms,
            "best_channels": best_channels,
            "n_spikes_used": n_spikes_used,
            "time_ms": time_ms,
        }

    def compute_isi_statistics(
        self, spikedata: "SpikeData", auto_save: bool = True  # type: ignore
    ) -> Dict[str, np.ndarray]:
        """
        Compute interspike interval (ISI) based statistics for temporal firing patterns.

        Calculates various ISI-based metrics that characterize the temporal structure
        of neuronal firing, including regularity (CV), burstiness, and quality metrics.

        Parameters
        ----------
        spikedata : SpikeData
            The parent SpikeData object containing spike trains.
        auto_save : bool, optional
            If True (default), automatically saves computed metrics to neuron_attributes.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'mean_isi_ms': Mean interspike interval in milliseconds
            - 'median_isi_ms': Median ISI (more robust to bursting)
            - 'cv_isi': Coefficient of variation (std/mean, firing regularity)
            - 'isi_skewness': Distribution skewness (positive = bursting)
            - 'burst_index': Fraction of ISIs < 10ms (burstiness measure)
            - 'pause_ratio': Fraction of ISIs > 100ms (long pauses)
            - 'refractory_violations': Count of ISIs < 2ms (quality metric)

        Notes
        -----
        - Units with no spikes will have NaN values for all metrics
        - CV (coefficient of variation): values near 0 indicate regular firing,
          values near 1 indicate Poisson-like, values > 1 indicate irregular/bursting
        - Burst index: higher values indicate more bursty firing
        - Refractory violations: should be 0 or very low for good quality units

        Examples
        --------
        >>> # Compute and save ISI statistics
        >>> isi_stats = sd.neuron_attributes.compute_isi_statistics(sd)
        >>> # Access CV values to find regular vs bursting neurons
        >>> cv_values = sd.neuron_attributes.get_attribute('cv_isi')
        >>> regular_neurons = np.where(cv_values < 0.5)[0]
        """
        from scipy import stats

        # Get ISIs from SpikeData
        isis = spikedata.interspike_intervals()

        # Initialize result arrays
        mean_isi = np.zeros(self.n_neurons)
        median_isi = np.zeros(self.n_neurons)
        cv_isi = np.zeros(self.n_neurons)
        isi_skewness = np.zeros(self.n_neurons)
        burst_index = np.zeros(self.n_neurons)
        pause_ratio = np.zeros(self.n_neurons)
        refractory_violations = np.zeros(self.n_neurons, dtype=int)

        # Compute metrics for each neuron
        for i, isi in enumerate(isis):
            if len(isi) == 0:
                # No spikes - set all to NaN
                mean_isi[i] = np.nan
                median_isi[i] = np.nan
                cv_isi[i] = np.nan
                isi_skewness[i] = np.nan
                burst_index[i] = np.nan
                pause_ratio[i] = np.nan
                refractory_violations[i] = 0
            else:
                # Basic statistics
                mean_isi[i] = np.mean(isi)
                median_isi[i] = np.median(isi)

                # Coefficient of variation (regularity)
                if mean_isi[i] > 0:
                    cv_isi[i] = np.std(isi) / mean_isi[i]
                else:
                    cv_isi[i] = np.nan

                # Skewness (using scipy for robust calculation)
                if len(isi) >= 3:  # Need at least 3 points for skewness
                    isi_skewness[i] = stats.skew(isi)
                else:
                    isi_skewness[i] = np.nan

                # Burst index: fraction of short ISIs (< 10ms)
                burst_index[i] = np.sum(isi < 10.0) / len(isi)

                # Pause ratio: fraction of long ISIs (> 100ms)
                pause_ratio[i] = np.sum(isi > 100.0) / len(isi)

                # Refractory violations: ISIs < 2ms (quality metric)
                refractory_violations[i] = np.sum(isi < 2.0)

        # Auto-save to neuron_attributes if requested
        if auto_save:
            self.set_attribute("mean_isi_ms", mean_isi)
            self.set_attribute("median_isi_ms", median_isi)
            self.set_attribute("cv_isi", cv_isi)
            self.set_attribute("isi_skewness", isi_skewness)
            self.set_attribute("burst_index", burst_index)
            self.set_attribute("pause_ratio", pause_ratio)
            self.set_attribute("refractory_violations", refractory_violations)

        return {
            "mean_isi_ms": mean_isi,
            "median_isi_ms": median_isi,
            "cv_isi": cv_isi,
            "isi_skewness": isi_skewness,
            "burst_index": burst_index,
            "pause_ratio": pause_ratio,
            "refractory_violations": refractory_violations,
        }

    def compute_latency_statistics(
        self,
        spikedata: "SpikeData",  # type: ignore
        reference_neuron: int,
        window_ms: float = 100.0,
        auto_save: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Compute latency statistics relative to a reference neuron.

        Calculates mean latency and jitter for each neuron relative to the spike
        times of a reference neuron, useful for identifying functional connectivity
        and response timing.

        Parameters
        ----------
        spikedata : SpikeData
            The parent SpikeData object containing spike trains.
        reference_neuron : int
            Index of the reference neuron to compute latencies from.
        window_ms : float, optional
            Maximum latency window in milliseconds. Default is 100.0.
        auto_save : bool, optional
            If True (default), automatically saves computed metrics to neuron_attributes.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'mean_latency_ms': Mean latency to reference neuron in milliseconds
            - 'median_latency_ms': Median latency (more robust to outliers)
            - 'latency_jitter_ms': Standard deviation of latencies (temporal precision)

        Notes
        -----
        - Neurons with no spikes within the window will have NaN values
        - Positive latencies indicate the neuron fires after the reference
        - Negative latencies indicate the neuron fires before the reference
        - Low jitter indicates precise temporal relationship

        Examples
        --------
        >>> # Compute latencies relative to neuron 0 (e.g., a pacemaker)
        >>> lat_stats = sd.neuron_attributes.compute_latency_statistics(sd, reference_neuron=0)
        >>> # Find neurons that consistently follow the reference
        >>> followers = np.where(
        ...     (sd.neuron_attributes.get_attribute('mean_latency_ms') > 0) &
        ...     (sd.neuron_attributes.get_attribute('latency_jitter_ms') < 5)
        ... )[0]
        """
        # Get latencies from SpikeData
        latencies = spikedata.latencies_to_index(reference_neuron, window_ms=window_ms)

        # Initialize result arrays
        mean_latency = np.zeros(self.n_neurons)
        median_latency = np.zeros(self.n_neurons)
        latency_jitter = np.zeros(self.n_neurons)

        # Compute statistics for each neuron
        for i, lat in enumerate(latencies):
            if len(lat) == 0:
                # No latencies within window
                mean_latency[i] = np.nan
                median_latency[i] = np.nan
                latency_jitter[i] = np.nan
            else:
                lat_array = np.array(lat)
                mean_latency[i] = np.mean(lat_array)
                median_latency[i] = np.median(lat_array)
                latency_jitter[i] = np.std(lat_array)

        # Auto-save to neuron_attributes if requested
        if auto_save:
            self.set_attribute("mean_latency_ms", mean_latency)
            self.set_attribute("median_latency_ms", median_latency)
            self.set_attribute("latency_jitter_ms", latency_jitter)

        return {
            "mean_latency_ms": mean_latency,
            "median_latency_ms": median_latency,
            "latency_jitter_ms": latency_jitter,
        }

    def compute_burst_participation(
        self,
        spikedata: "SpikeData",  # type: ignore
        burst_edges: np.ndarray,
        min_spikes: int = 5,
        backbone_threshold: float = 0.5,
        auto_save: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Compute burst participation metrics for network burst analysis.

        Uses SpikeData.get_frac_active() to calculate which neurons participate in
        detected network bursts and identifies "backbone" neurons that consistently
        participate.

        Parameters
        ----------
        spikedata : SpikeData
            The parent SpikeData object containing spike trains.
        burst_edges : np.ndarray
            Array of shape (n_bursts, 2) containing [start, end] time indices for each
            burst in milliseconds.
        min_spikes : int, optional
            Minimum number of spikes required for a neuron to be considered active
            in a burst. Default is 5.
        backbone_threshold : float, optional
            Minimum fraction of bursts (0-1) a neuron must participate in to be
            considered a backbone unit. Default is 0.5.
        auto_save : bool, optional
            If True (default), automatically saves computed metrics to neuron_attributes
            and metadata.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'burst_participation': Fraction of bursts each neuron participates in (0-1)
            - 'is_backbone_unit': Boolean array indicating backbone neurons
            - 'backbone_indices': Indices of neurons classified as backbone units
            - 'frac_per_burst': Fraction of neurons active in each burst

        Notes
        -----
        - Burst edges should be in milliseconds matching SpikeData time units
        - Backbone neurons are those that reliably participate in network activity
        - Results are also stored in spikedata.metadata if auto_save=True

        Examples
        --------
        >>> # First detect bursts (using external method or manual detection)
        >>> burst_edges = np.array([[100, 200], [500, 600], [1000, 1100]])  # ms
        >>>
        >>> # Compute participation
        >>> burst_stats = sd.neuron_attributes.compute_burst_participation(
        ...     sd, burst_edges, min_spikes=3, backbone_threshold=0.7
        ... )
        >>>
        >>> # Find backbone neurons
        >>> backbone = burst_stats['backbone_indices']
        >>> print(f"Found {len(backbone)} backbone neurons")
        >>>
        >>> # Access saved attributes
        >>> participation = sd.neuron_attributes.get_attribute('burst_participation')
        """
        # Get burst participation from SpikeData
        frac_per_unit, frac_per_burst, backbone_units = spikedata.get_frac_active(
            burst_edges, min_spikes, backbone_threshold
        )

        # Create boolean array for backbone units
        is_backbone = np.zeros(self.n_neurons, dtype=bool)
        is_backbone[backbone_units] = True

        # Auto-save to neuron_attributes if requested
        if auto_save:
            self.set_attribute("burst_participation", frac_per_unit)
            self.set_attribute("is_backbone_unit", is_backbone)

            # Also save burst-level info to metadata
            # Note: This modifies the parent SpikeData's metadata
            # Store as dict to make it JSON-serializable
            spikedata.metadata["burst_analysis"] = {
                "burst_edges": (
                    burst_edges.tolist()
                    if isinstance(burst_edges, np.ndarray)
                    else burst_edges
                ),
                "min_spikes": min_spikes,
                "backbone_threshold": backbone_threshold,
                "n_bursts": len(burst_edges),
                "n_backbone_units": len(backbone_units),
                "frac_per_burst": (
                    frac_per_burst.tolist()
                    if isinstance(frac_per_burst, np.ndarray)
                    else frac_per_burst
                ),
            }

        return {
            "burst_participation": frac_per_unit,
            "is_backbone_unit": is_backbone,
            "backbone_indices": backbone_units,
            "frac_per_burst": frac_per_burst,
        }
