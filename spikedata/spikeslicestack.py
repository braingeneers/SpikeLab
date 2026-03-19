import warnings

import numpy as np

from .pairwise import PairwiseCompMatrix, PairwiseCompMatrixStack
from .spikedata import SpikeData
from .utils import (
    _validate_time_start_to_end,
    _get_attr,
    get_sttc,
    compute_cross_correlation_with_lag,
)


class SpikeSliceStack:
    """
    Description:
    -----------
    A data structure where the underlying data is a list of SpikeData objects, one per slice.
    User inputs a single SpikeData object and specifies the slice times to split it, or
    directly provides a pre-built list of SpikeData objects.

        - U: Units (refers to neuron/neuron clusters)
        - S: Slices (can be bursts, events, etc)

    Parameters:
    -----------
    Option #1: data_obj
        data_obj (SpikeData): A SpikeData object to slice.
        There are 2 choices for time input:
            Choice A)
                times_start_to_end (list): Each entry must be a tuple. Each tuple is (start, end) and
                                           represents the start and end times of a desired slice.
                                           Each tuple must have same duration.
            Choice B) (both of the following must be input for this option)
                time_peaks (list): List of times as int or float where there is a burst peak or
                                   stimulation event.
                time_bounds (tuple): Single tuple (left_bound, right_bound). If you put (250, 500),
                                     then this means 250 ms before peak and 500 ms after peak.
        neuron_attributes (list or None): List of attribute dicts, one per unit. If None,
                                          inherited from data_obj.

    Option #2: spike_stack
        spike_stack (list): List of SpikeData objects, one per slice. All must have the same
                            number of units.
        times_start_to_end (list): Each entry must be a tuple (start, end). Length must equal
                                   len(spike_stack). If None, generated automatically from
                                   slice lengths concatenated end-to-end.
        neuron_attributes (list or None): List of attribute dicts, one per unit.

    Instance Variables:
    --------
    self.spike_stack (list): List of SpikeData objects, one per slice. Spike times within each
                             slice are preserved in absolute recording time (not shifted to 0).
                             Use self.times to know each slice's absolute time window.
                             Example)
                                spike_stack[0].train[neuron_0] = [810, 950, 1200, 1480]   # absolute ms
                                spike_stack[1].train[neuron_0] = [4800, 4900, 5100, 5450] # absolute ms
                                spike_stack[2].train[neuron_0] = [8800, 9050, 9300]       # absolute ms
    self.times (list of tuples): List of (start, end) time bounds for each slice in original
                                 recording time, sorted chronologically. Length equals S.
                                 Example: [(100, 350), (500, 750), (1000, 1250)]
    self.N (int): Number of units.
    self.neuron_attributes (list or None): List of attribute dicts, one per unit. None if not
                                           provided.
    """

    def __init__(
        self,
        data_obj=None,
        times_start_to_end=None,
        time_peaks=None,
        time_bounds=None,
        spike_stack=None,
        neuron_attributes=None,
    ):
        if data_obj is None and spike_stack is None:
            raise TypeError(
                "Must input either a SpikeData as data_obj (option 1) or spike_stack (option 2)"
            )
        if data_obj is not None and spike_stack is not None:
            warnings.warn(
                "User input both data_obj and spike_stack. "
                "Ignoring data_obj and using spike_stack instead.",
                UserWarning,
            )
            data_obj = None

        # Option 1: Using data_obj
        if data_obj is not None:
            if not isinstance(data_obj, SpikeData):
                raise TypeError("data_obj must be a SpikeData object")

            if times_start_to_end is None:
                if time_peaks is None or time_bounds is None:
                    raise ValueError(
                        "Must provide either times_start_to_end or "
                        "both time_peaks and time_bounds"
                    )
                if not isinstance(time_bounds, tuple) or len(time_bounds) != 2:
                    raise TypeError(
                        "time_bounds must be a tuple of (before, after) durations"
                    )
                before, after = time_bounds
                time_peaks = sorted(time_peaks)
                times_start_to_end = []
                for t in time_peaks:
                    times_start_to_end.append((t - before, t + after))

            times_start_to_end = _validate_time_start_to_end(times_start_to_end)

            self.times = times_start_to_end
            self.spike_stack = []
            for start, end in times_start_to_end:
                self.spike_stack.append(data_obj.subtime(start, end))

            if neuron_attributes is None:
                neuron_attributes = data_obj.neuron_attributes

        # Option 2: Using spike_stack directly
        if spike_stack is not None:
            if not isinstance(spike_stack, list):
                raise TypeError("spike_stack must be a list of SpikeData objects")
            for s in spike_stack:
                if not isinstance(s, SpikeData):
                    raise TypeError("spike_stack must be a list of SpikeData objects")
            if len(spike_stack) == 0:
                raise ValueError("spike_stack must not be empty")

            N = spike_stack[0].N
            for s in spike_stack:
                if s.N != N:
                    raise ValueError(
                        "All SpikeData objects in spike_stack must have the same number of units"
                    )

            if times_start_to_end is None:
                t = 0.0
                times_start_to_end = []
                for s in spike_stack:
                    times_start_to_end.append((t, t + s.length))
                    t += s.length
            else:
                times_start_to_end = _validate_time_start_to_end(times_start_to_end)
                if len(times_start_to_end) != len(spike_stack):
                    raise ValueError(
                        "times_start_to_end must have the same length as spike_stack"
                    )

            self.spike_stack = list(spike_stack)
            self.times = times_start_to_end

        self.N = self.spike_stack[0].N

        self.neuron_attributes = None
        if neuron_attributes is not None:
            self.neuron_attributes = neuron_attributes.copy()
            if len(self.neuron_attributes) != self.N:
                raise ValueError(
                    f"neuron_attributes has {len(self.neuron_attributes)} items "
                    f"but spike_stack has {self.N} units"
                )

    def subslice(self, slices):
        """
        Extract a subset of slices from the spike stack.

        Parameters:
        -----------
        slices (int or list): Slice index or list of slice indices to extract.

        Returns:
        --------
        SpikeSliceStack: New SpikeSliceStack containing only the specified slices.
                         Shape changes from S to S_trimmed. All units and
                         neuron_attributes are carried over.
        """
        S = len(self.spike_stack)
        if isinstance(slices, int):
            slices = [slices]
        for s in slices:
            if s >= S or s < -S:
                raise ValueError(f"One or more slice indices out of range for S={S}")
        slices = sorted(slices)
        new_spike_stack = []
        new_times = []
        for s in slices:
            new_spike_stack.append(self.spike_stack[s])
            new_times.append(self.times[s])
        return SpikeSliceStack(
            spike_stack=new_spike_stack,
            times_start_to_end=new_times,
            neuron_attributes=self.neuron_attributes,
        )

    def subset(self, units, by=None):
        """
        Extract a subset of units from every slice in the spike stack.

        Parameters:
        -----------
        units (int, str, or list): Unit indices to extract. If by=None, must be int(s).
                                   If by is set, values to match in neuron_attributes.
        by (str or None): If set, select units by this neuron_attribute key instead of
                          by index.

        Returns:
        --------
        SpikeSliceStack: New SpikeSliceStack containing only the specified units across
                         all slices. All slices and neuron_attributes are carried over.

        Notes:
        - Units are included in the output in the order they appear in the train
          (ascending index order), not the order listed in units.
        - If IDs are not unique (when using by), every matching neuron is included.
        """
        if isinstance(units, (int, str)):
            units = [units]

        # Resolve which indices will be kept so we can update neuron_attributes
        if by is not None:
            if self.neuron_attributes is None:
                raise ValueError("can't use `by` without `neuron_attributes`")
            _missing = object()
            unit_set = set(units)
            kept_indices = []
            for i in range(self.N):
                if _get_attr(self.neuron_attributes[i], by, _missing) in unit_set:
                    kept_indices.append(i)
        else:
            kept_indices = sorted(set(int(u) for u in units))

        new_spike_stack = []
        for sd in self.spike_stack:
            new_spike_stack.append(sd.subset(kept_indices))

        new_neuron_attributes = None
        if self.neuron_attributes is not None:
            new_neuron_attributes = []
            for i in kept_indices:
                new_neuron_attributes.append(self.neuron_attributes[i])

        return SpikeSliceStack(
            spike_stack=new_spike_stack,
            times_start_to_end=self.times,
            neuron_attributes=new_neuron_attributes,
        )

    def subtime_by_index(self, start_idx, end_idx):
        """
        Trim each slice to a sub-window specified by millisecond indices (1 index = 1 ms),
        measured from the start of each slice. Trims along the time axis while preserving
        all slices and units.

        Parameters:
        -----------
        start_idx (int): Start index in ms from slice start (inclusive). Supports negative indexing.
        end_idx (int): End index in ms from slice start (exclusive). Supports negative indexing.

        Returns:
        --------
        SpikeSliceStack: New SpikeSliceStack where each slice is trimmed to the corresponding
                         absolute time window. Absolute spike times are preserved (not shifted).
                         self.times is updated to reflect the new absolute time bounds.

        Notes:
        - Indices are relative to each slice's own start (index 0 = slice start ms).
          They are converted to absolute recording times internally before trimming.
        - Original absolute timestamps are preserved. If you want shifted-to-zero timestamps,
          simply make a new SpikeSliceStack.
        - All slices, neuron_attributes are carried over from the original.
        """
        slice_duration_ms = self.times[0][1] - self.times[0][0]
        T = int(round(slice_duration_ms))

        if start_idx < 0:
            start_idx += T
        if end_idx < 0:
            end_idx += T
        if start_idx < 0 or start_idx >= T:
            raise ValueError(f"start_idx {start_idx} out of range for T={T}")
        if end_idx <= start_idx or end_idx > T:
            raise ValueError(f"end_idx {end_idx} invalid for T={T}")

        new_spike_stack = []
        new_times = []
        for sd, t in zip(self.spike_stack, self.times):
            new_spike_stack.append(sd.subtime(float(start_idx), float(end_idx)))
            abs_start = t[0] + float(start_idx)
            abs_end = t[0] + float(end_idx)
            new_times.append((abs_start, abs_end))

        return SpikeSliceStack(
            spike_stack=new_spike_stack,
            times_start_to_end=new_times,
            neuron_attributes=self.neuron_attributes,
        )

    def to_raster_array(self, bin_size=1.0):
        """
        Convert the spike stack into a 3D raster array of shape (N, T, S).

        Each slice is rasterized with the given bin size, producing a spike count matrix
        where entry (n, t, s) is the number of spikes unit n fired in time bin t of slice s.
        Time bin 0 corresponds to the start of each slice (index 0 = slice start).

        Parameters:
        -----------
        bin_size (float): Time bin size in ms (default 1.0).

        Returns:
        --------
        raster_stack (np.ndarray): 3D array of shape (N, T, S) with non-negative integer
                                   spike counts.
        """
        dense_list = []
        for sd, (start, end) in zip(self.spike_stack, self.times):
            # Spike times are absolute so we manually shift to 0-based before
            # rasterizing. Calling sd.subtime(start, end) would fail because
            # sd.length == duration, not the absolute end time.
            duration = end - start
            shifted_train = []
            for spikes in sd.train:
                shifted_train.append(spikes - start)
            temp_sd = SpikeData(
                shifted_train,
                length=duration,
                N=sd.N,
                neuron_attributes=sd.neuron_attributes,
            )
            dense_list.append(temp_sd.sparse_raster(bin_size=bin_size).toarray())
        return np.stack(dense_list, axis=2)

    def unit_to_unit_comparison(
        self,
        metric="ccg",
        delt=20.0,
        bin_size=1.0,
        max_lag=350,
    ):
        """
        Compute pairwise unit-to-unit similarity within each slice using spike-based metrics.

        For each slice, computes a (U, U) similarity matrix between all unit pairs,
        then stacks the results into a ``PairwiseCompMatrixStack (U, U, S)``.

        Parameters:
            metric (str): Similarity metric to use. ``"ccg"`` for cross-correlogram
                on binned rasters (default), ``"sttc"`` for spike time tiling coefficient.
            delt (float): STTC time window in milliseconds (default: 20.0).
                Only used when metric is ``"sttc"``.
            bin_size (float): Bin size in milliseconds for the binary raster
                (default: 1.0). Only used when metric is ``"ccg"``.
            max_lag (float): Maximum lag in milliseconds to search for the peak
                correlation (default: 350). Only used when metric is ``"ccg"``.

        Returns:
            corr_stack (PairwiseCompMatrixStack): Pairwise similarity scores between
                all unit pairs for each slice. Shape is ``(U, U, S)``.
            lag_stack (PairwiseCompMatrixStack or None): Lag at which maximum
                similarity occurs for each pair per slice. Shape is ``(U, U, S)``.
                ``None`` when metric is ``"sttc"`` (STTC has no lag).
            av_corr (np.ndarray): Average similarity per slice across all unit
                pairs in the lower triangle. Shape is ``(S,)``.
            av_lag (np.ndarray or None): Average lag per slice. Shape is ``(S,)``.
                ``None`` when metric is ``"sttc"``.

        Notes:
            - Analogous to ``RateSliceStack.unit_to_unit_correlation`` but operates
              on raw spike trains instead of firing rate time series.
        """
        if metric not in ("sttc", "ccg"):
            raise ValueError(f"metric must be 'sttc' or 'ccg', got {metric!r}")

        num_units = self.N
        num_slices = len(self.spike_stack)

        if num_units < 2:
            warnings.warn(
                "Cannot compute unit-to-unit comparison with fewer than "
                "2 units. Returning NaN.",
                RuntimeWarning,
            )
            nan_stack = np.full((num_units, num_units, num_slices), np.nan)
            nan_avgs = np.full(num_slices, np.nan)
            return (
                PairwiseCompMatrixStack(stack=nan_stack, times=self.times),
                (
                    PairwiseCompMatrixStack(stack=nan_stack.copy(), times=self.times)
                    if metric == "ccg"
                    else None
                ),
                nan_avgs,
                nan_avgs.copy() if metric == "ccg" else None,
            )

        corr_matrices = []
        lag_matrices = []

        for sd in self.spike_stack:
            if metric == "sttc":
                pcm = sd.spike_time_tilings(delt=delt)
                corr_matrices.append(pcm.matrix)
            else:  # ccg
                corr_pcm, lag_pcm = sd.get_pairwise_ccg(
                    bin_size=bin_size, max_lag=max_lag
                )
                corr_matrices.append(corr_pcm.matrix)
                lag_matrices.append(lag_pcm.matrix)

        # Stack: list of (U, U) -> (S, U, U) -> transpose to (U, U, S)
        corr_array = np.moveaxis(np.stack(corr_matrices, axis=0), 0, 2)

        lower_tri = np.tril_indices(num_units, k=-1)
        av_corr = np.nanmean(corr_array[lower_tri[0], lower_tri[1], :], axis=0)

        corr_stack = PairwiseCompMatrixStack(stack=corr_array, times=self.times)

        if metric == "ccg":
            lag_array = np.moveaxis(np.stack(lag_matrices, axis=0), 0, 2)
            av_lag = np.nanmean(lag_array[lower_tri[0], lower_tri[1], :], axis=0)
            lag_stack = PairwiseCompMatrixStack(stack=lag_array, times=self.times)
        else:
            lag_stack = None
            av_lag = None

        return corr_stack, lag_stack, av_corr, av_lag

    def get_slice_to_slice_unit_comparison(
        self,
        metric="ccg",
        delt=20.0,
        bin_size=1.0,
        max_lag=350,
        min_spikes=3,
        min_frac=0.3,
    ):
        """
        Compute slice-to-slice similarity for each unit using spike-based metrics.

        For each unit independently, compares its spike train across every pair of
        slices. Asks: "Does unit X fire in the same temporal pattern across repeated
        events?" Returns a ``PairwiseCompMatrixStack (S, S, U)``.

        Parameters:
            metric (str): Similarity metric to use. ``"ccg"`` for cross-correlogram
                on binned rasters (default), ``"sttc"`` for spike time tiling coefficient.
            delt (float): STTC time window in milliseconds (default: 20.0).
                Only used when metric is ``"sttc"``.
            bin_size (float): Bin size in milliseconds for the binary raster
                (default: 1.0). Only used when metric is ``"ccg"``.
            max_lag (float): Maximum lag in milliseconds to search for the peak
                correlation (default: 350). Only used when metric is ``"ccg"``.
            min_spikes (int): Minimum number of spikes in a slice for a unit to
                be considered valid in that slice (default: 3).
            min_frac (float): Maximum fraction of slices that can be invalid before
                a unit's average is set to NaN (default: 0.3).

        Returns:
            all_corr (PairwiseCompMatrixStack): Pairwise similarity between all
                slice pairs for each unit. Shape is ``(S, S, U)``.
            all_lag (PairwiseCompMatrixStack or None): Lag at which maximum
                similarity occurs for each slice pair per unit. Shape is ``(S, S, U)``.
                ``None`` when metric is ``"sttc"``.
            av_corr (np.ndarray): Average similarity per unit across all valid
                slice pairs. Shape is ``(U,)``.
            av_lag (np.ndarray or None): Average lag per unit. Shape is ``(U,)``.
                ``None`` when metric is ``"sttc"``.

        Notes:
            - Analogous to ``RateSliceStack.get_slice_to_slice_unit_corr_from_stack``
              but operates on raw spike trains.
            - Spike times within each slice are shifted to start at 0 before
              comparison so that temporal patterns are aligned across slices.
        """
        if metric not in ("sttc", "ccg"):
            raise ValueError(f"metric must be 'sttc' or 'ccg', got {metric!r}")

        num_units = self.N
        num_slices = len(self.spike_stack)

        if num_slices < 2:
            warnings.warn(
                "Cannot compute slice-to-slice unit comparison with fewer than "
                "2 slices. Returning NaN.",
                RuntimeWarning,
            )
            av_corr = np.full(num_units, np.nan)
            nan_stack = np.full((num_slices, num_slices, num_units), np.nan)
            return (
                PairwiseCompMatrixStack(stack=nan_stack),
                (
                    PairwiseCompMatrixStack(stack=nan_stack.copy())
                    if metric == "ccg"
                    else None
                ),
                av_corr,
                av_corr.copy() if metric == "ccg" else None,
            )

        # Pre-compute shifted spike trains (shifted to 0-based per slice)
        # and per-slice rasters for CCG
        shifted_trains = []  # list of S lists, each containing U spike arrays
        slice_durations = []
        slice_rasters = []  # only populated for CCG

        for sd, (start, end) in zip(self.spike_stack, self.times):
            duration = end - start
            slice_durations.append(duration)
            trains = []
            for u in range(num_units):
                trains.append(np.asarray(sd.train[u]) - start)
            shifted_trains.append(trains)

            if metric == "ccg":
                # Build shifted SpikeData for raster computation
                temp_sd = SpikeData(trains, length=duration, N=num_units)
                slice_rasters.append(temp_sd.raster(bin_size))

        max_lag_bins = int(round(max_lag / bin_size)) if metric == "ccg" else 0

        # Initialize result arrays: (U, S, S), will transpose to (S, S, U) at end
        all_corr_scores = np.full((num_units, num_slices, num_slices), np.nan)
        all_lag_scores = (
            np.full((num_units, num_slices, num_slices), np.nan)
            if metric == "ccg"
            else None
        )
        av_corr = np.full(num_units, np.nan)
        av_lag = np.full(num_units, np.nan) if metric == "ccg" else None

        lower_tri = np.tril_indices(num_slices, k=-1)

        for unit in range(num_units):
            # Count invalid slices for this unit
            invalid_count = 0

            for ref_s in range(num_slices):
                ref_train = shifted_trains[ref_s][unit]
                if len(ref_train) < min_spikes:
                    invalid_count += 1
                    continue

                for comp_s in range(ref_s, num_slices):
                    comp_train = shifted_trains[comp_s][unit]
                    if len(comp_train) < min_spikes:
                        continue

                    if metric == "sttc":
                        length = max(slice_durations[ref_s], slice_durations[comp_s])
                        score = get_sttc(
                            ref_train, comp_train, delt=delt, length=length
                        )
                        all_corr_scores[unit, ref_s, comp_s] = score
                        all_corr_scores[unit, comp_s, ref_s] = score
                    else:  # ccg
                        ref_signal = slice_rasters[ref_s][unit, :]
                        comp_signal = slice_rasters[comp_s][unit, :]
                        score, lag = compute_cross_correlation_with_lag(
                            ref_signal, comp_signal, max_lag=max_lag_bins
                        )
                        all_corr_scores[unit, ref_s, comp_s] = score
                        all_corr_scores[unit, comp_s, ref_s] = score
                        all_lag_scores[unit, ref_s, comp_s] = lag
                        all_lag_scores[unit, comp_s, ref_s] = -lag

            # Compute average if enough slices were valid
            if invalid_count / num_slices <= min_frac:
                av_corr[unit] = np.nanmean(
                    all_corr_scores[unit, lower_tri[0], lower_tri[1]]
                )
                if metric == "ccg":
                    av_lag[unit] = np.nanmean(
                        all_lag_scores[unit, lower_tri[0], lower_tri[1]]
                    )

        # Transpose from (U, S, S) to (S, S, U)
        all_corr_scores = np.moveaxis(all_corr_scores, 0, 2)
        all_corr_stack = PairwiseCompMatrixStack(stack=all_corr_scores)

        if metric == "ccg":
            all_lag_scores = np.moveaxis(all_lag_scores, 0, 2)
            all_lag_stack = PairwiseCompMatrixStack(stack=all_lag_scores)
        else:
            all_lag_stack = None

        return all_corr_stack, all_lag_stack, av_corr, av_lag
