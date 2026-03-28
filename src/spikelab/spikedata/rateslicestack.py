import numpy as np

__all__ = ["RateSliceStack"]
from .ratedata import RateData
from .spikedata import SpikeData
import warnings
from .pairwise import PairwiseCompMatrix, PairwiseCompMatrixStack


from concurrent.futures import ThreadPoolExecutor

from .utils import (
    compute_cross_correlation_with_lag,
    compute_cosine_similarity_with_lag,
    _validate_time_start_to_end,
    _get_attr,
    _resolve_n_jobs,
)


class RateSliceStack:
    """A 3D firing rate matrix of shape (U, T, S) with correlation and similarity capabilities.

    U is units (neurons), T is time bins, and S is slices (bursts, events, etc).
    Construct from either a data_obj (SpikeData or RateData) with time
    specifications, or directly from a pre-built event_matrix. The instance
    variables are the same regardless of input option.

    Parameters:
        data_obj (SpikeData or RateData): A data object to slice. Provide
            either this or event_matrix, not both.
        times_start_to_end (list or None): Each entry is a tuple (start, end)
            representing the start and end times of a desired slice. Each
            tuple must have the same duration.
        time_peaks (list or None): List of times as int or float where there
            is a burst peak or stimulation event. Must be paired with
            time_bounds. Alternative to times_start_to_end.
        time_bounds (tuple or None): Single tuple (left_bound, right_bound).
            For example, (250, 500) means 250 ms before peak and 500 ms
            after peak. Must be paired with time_peaks.
        sigma_ms (float): Smoothing factor for computing ISI if you input a
            SpikeData object. Otherwise not used.
        event_matrix (np.ndarray or None): A 3D array of shape (U, T, S).
            Provide either this or data_obj, not both.
        step_size (float or None): Time resolution in milliseconds between
            consecutive time bins. If None, defaults to 1.0.
        neuron_attributes (list or None): List of attribute objects, one per
            unit, containing arbitrary metadata about each neuron.

    Attributes:
        event_stack (np.ndarray): 3D array of shape (U, T, S) where U is the
            number of units, T is the number of time bins, and S is the
            number of slices.
        times (list): List of (start, end) time bounds for each slice, sorted
            chronologically. Length equals S. Example:
            [(100, 200), (500, 600), (1000, 1100)].
        step_size (float): Time resolution in milliseconds between consecutive
            time bins. Inferred from input data. For SpikeData input, defaults
            to 1.0 ms. Example: 1.0 means time bins are at [100, 101, 102, ...]
            ms.
        neuron_attributes (list or None): List of attribute objects, one per
            unit, containing arbitrary metadata about each neuron.
    """

    def __init__(
        self,
        data_obj=None,  # Option 1
        times_start_to_end=None,
        time_peaks=None,
        time_bounds=None,
        sigma_ms=10,
        event_matrix=None,  # Option 2
        step_size=None,
        neuron_attributes=None,
    ):
        if (data_obj is None) and (event_matrix is None):
            raise ValueError(
                "Must input either data_obj(option 1) or event_matrix(option 2)"
            )
        if (data_obj is not None) and (event_matrix is not None):
            warnings.warn(
                "User input both data_obj and event_matrix. Ignoring data_obj and using event_matrix instead.",
                UserWarning,
            )
            data_obj = None

        if sigma_ms is not None and sigma_ms < 0:
            raise ValueError("sigma_ms must be non-negative")

        # Option 1: Using data_obj
        if data_obj is not None:
            if not isinstance(data_obj, (SpikeData, RateData)):
                raise TypeError(
                    "data_obj must either be a SpikeData object or RateData object"
                )

            # This is to check that one of the time options is selected
            if times_start_to_end is None:
                if time_peaks is None or time_bounds is None:
                    raise ValueError(
                        "Must provide either times_start_to_end or both times_peaks and time_bounds"
                    )

                # If we're using peaks+bounds, validate them
                if not isinstance(time_bounds, tuple) or len(time_bounds) != 2:
                    raise TypeError(
                        "time_bounds must be a tuple of (before, after) durations"
                    )

                # Convert peaks and bounds to start_to_end format
                before, after = time_bounds
                time_peaks = sorted(time_peaks)
                times_start_to_end = [(t - before, t + after) for t in time_peaks]

            # Now that everything is times_start_to_end format, checking if inputs are correct types
            # Determine recording range for validation.
            if isinstance(data_obj, SpikeData):
                rec_range = (
                    data_obj.start_time,
                    data_obj.start_time + data_obj.length,
                )
            elif len(data_obj.times) > 1:
                step = data_obj.times[1] - data_obj.times[0]
                rec_range = (data_obj.times[0], data_obj.times[-1] + step)
            elif len(data_obj.times) == 1:
                rec_range = (data_obj.times[0], data_obj.times[0] + 1)
            else:
                rec_range = None
            times_start_to_end = _validate_time_start_to_end(
                times_start_to_end, recording_range=rec_range
            )

            # Actual constructor

            if isinstance(data_obj, SpikeData):
                resolution = step_size if step_size is not None else 1.0
                all_times = np.arange(
                    data_obj.start_time,
                    data_obj.start_time + data_obj.length,
                    resolution,
                )
                inst_Frate_matrix = data_obj.resampled_isi(all_times, sigma_ms)
                data_obj = RateData(inst_Frate_matrix, all_times)

            if len(data_obj.times) > 1:
                self.step_size = data_obj.times[1] - data_obj.times[0]
            else:
                self.step_size = 1.0

            self.times = times_start_to_end
            event_stack = []
            if isinstance(data_obj, RateData):
                # I use subtime here to extract a burst event and its time value based subtime
                for time in times_start_to_end:
                    start = time[0]
                    end = time[1]
                    rate_obj_slice = data_obj.subtime(start, end)
                    slice_matrix = rate_obj_slice.inst_Frate_data
                    event_stack.append(slice_matrix)

            # Converts to a 3d array
            event_stack = np.stack(event_stack, axis=2)
            # This makes event stack be U x T x S
            self.event_stack = event_stack

        # Option 2: Using event matrx
        if event_matrix is not None:
            if not isinstance(event_matrix, np.ndarray):
                raise TypeError("event_matrix must be a numpy array")
            if event_matrix.ndim != 3:
                raise ValueError(
                    f"event_matrix must be 3D (U x T x S), got {event_matrix.ndim}D array"
                )
            if step_size is None:
                self.step_size = 1.0
            else:
                self.step_size = step_size
            if times_start_to_end is None:
                slice_duration = event_matrix.shape[1] * self.step_size
                times_start_to_end = []
                for i in range(event_matrix.shape[2]):
                    start = i * slice_duration
                    end = (i + 1) * slice_duration
                    tup = (start, end)
                    times_start_to_end.append(tup)
            else:
                times_start_to_end = _validate_time_start_to_end(times_start_to_end)
                # Make sure there is a (start,end) tuple for each slice
                if len(times_start_to_end) != event_matrix.shape[2]:
                    raise ValueError(
                        "times_start_to_end must have the same length as the last dimension of event_matrix"
                    )
            self.event_stack = event_matrix
            self.times = times_start_to_end

        if neuron_attributes is None and data_obj is not None:
            neuron_attributes = getattr(data_obj, "neuron_attributes", None)

        self.neuron_attributes = None
        if neuron_attributes is not None:
            self.neuron_attributes = neuron_attributes.copy()
            if len(neuron_attributes) != self.event_stack.shape[0]:
                raise ValueError(
                    f"neuron_attributes has {len(neuron_attributes)} items "
                    f"but event_stack has {self.event_stack.shape[0]} units"
                )

    def order_units_across_slices(
        self,
        agg_func,
        MIN_RATE_THRESHOLD=0.1,
        MIN_FRAC_ACTIVE=0.0,
        frac_active=None,
        timing_matrix=None,
    ):
        """Reorder units from earliest to latest peak firing rate across slices.

        Parameters:
            agg_func (str): Either ``"median"`` or ``"mean"``. Used for
                calculating the median/mean time when each unit has peak
                firing rate.
            MIN_RATE_THRESHOLD (float): Minimum peak firing rate for a slice
                to be included in the ordering calculation. Slices where a
                unit's max rate is below this threshold are excluded from
                that unit's typical peak time calculation. Ignored when
                timing_matrix is provided.
            MIN_FRAC_ACTIVE (float): Minimum fraction of slices a unit must
                be active in to be placed in the highly-active group.
                Default 0.0 means all units are in the first group, so the
                second array in each output tuple will be empty.
            frac_active (np.ndarray or None): Optional pre-computed
                fraction-active array of shape (U,) to use for the group
                split instead of the rate-based calculation. Compatible
                sources: SpikeSliceStack.compute_frac_active and
                SpikeData.get_frac_active (frac_per_unit output).
            timing_matrix (np.ndarray or None): Optional pre-computed (U, S)
                timing matrix from get_unit_timing_per_slice. When provided,
                MIN_RATE_THRESHOLD is ignored and this matrix is used
                directly.

        Returns:
            reordered_slice_matrices (tuple of arrays): Tuple of two 3D
                arrays from event_stack with the U dimension reordered
                temporally. The first array is the highly-active group and
                the second is the lower-activity group.
            unit_ids_in_order (tuple of arrays): Two arrays of original unit
                IDs in temporal order (highly-active, low-activity). For
                example, [3, 1, 0, 2] means unit 3 fires first. Use this
                to map back to original unit IDs.
            unit_std_indices (tuple of arrays): Two arrays of standard
                deviation of peak firing rate times (highly-active,
                low-activity). Lower values indicate more consistent timing
                across slices.
            unit_peak_times (tuple of arrays): Two arrays of median/mean peak
                firing time bin (highly-active, low-activity).
            unit_frac_active (tuple of arrays): Two arrays of the fraction of
                slices each unit was active in (highly-active, low-activity).

        Notes:
            - Call get_unit_timing_per_slice first to pre-compute the timing
              matrix if you want to reuse it across multiple calls (e.g.
              rank_order_correlation and this method).
        """
        # burst_matrices is U x T x S
        slice_matrices = self.event_stack
        num_units = slice_matrices.shape[0]
        num_slices = slice_matrices.shape[2]

        if timing_matrix is not None:
            unit_max_indices_matrix = np.asarray(timing_matrix, dtype=float)
            if unit_max_indices_matrix.shape != (num_units, num_slices):
                raise ValueError(
                    f"timing_matrix must have shape ({num_units}, {num_slices}), "
                    f"got {unit_max_indices_matrix.shape}"
                )
            # For frac_active fallback, derive mask from non-NaN entries
            mask = ~np.isnan(unit_max_indices_matrix)
        else:
            unit_max_indices_matrix = self.get_unit_timing_per_slice(
                MIN_RATE_THRESHOLD=MIN_RATE_THRESHOLD
            )
            mask = ~np.isnan(unit_max_indices_matrix)

        unit_std_indices = np.nanstd(unit_max_indices_matrix, axis=1)

        # This gives you a list of size N. Now you have median peak time for each neuron
        if agg_func == "median":
            unit_peak_times = np.nanmedian(unit_max_indices_matrix, axis=1)

        elif agg_func == "mean":
            unit_peak_times = np.nanmean(unit_max_indices_matrix, axis=1)
        else:
            raise ValueError(
                f"{agg_func} is not a valid input option. Must be either median or mean"
            )

        # Compute or validate frac_active only when splitting is requested
        num_units = slice_matrices.shape[0]
        skip_split = not MIN_FRAC_ACTIVE
        if skip_split:
            unit_frac_active = np.ones(num_units)
            highly_active_units = np.arange(num_units)
            low_active_units = np.array([], dtype=int)
        else:
            if frac_active is not None:
                frac_active = np.asarray(frac_active, dtype=float)
                if frac_active.shape != (num_units,):
                    raise ValueError(
                        f"frac_active must have shape ({num_units},), "
                        f"got {frac_active.shape}"
                    )
                unit_frac_active = frac_active
            else:
                unit_frac_active = np.sum(mask, axis=1) / mask.shape[1]
            highly_active_units = np.where(unit_frac_active >= MIN_FRAC_ACTIVE)[0]
            low_active_units = np.where(unit_frac_active < MIN_FRAC_ACTIVE)[0]

        highly_active_order = highly_active_units[
            np.argsort(unit_peak_times[highly_active_units])
        ]
        low_active_order = low_active_units[
            np.argsort(unit_peak_times[low_active_units])
        ]

        # Cast to int for output only after sorting is done.
        # NaN values (units with no active slices) become -1.
        unit_peak_times_int = np.full(unit_peak_times.shape, -1, dtype=int)
        valid = ~np.isnan(unit_peak_times)
        unit_peak_times_int[valid] = np.round(unit_peak_times[valid]).astype(int)

        reordered_slice_matrices = (
            slice_matrices[highly_active_order, :, :],
            slice_matrices[low_active_order, :, :],
        )
        unit_ids_in_order = (highly_active_order, low_active_order)
        unit_std_indices = (
            unit_std_indices[highly_active_order],
            unit_std_indices[low_active_order],
        )
        unit_peak_times = (
            unit_peak_times_int[highly_active_order],
            unit_peak_times_int[low_active_order],
        )
        unit_frac_active = (
            unit_frac_active[highly_active_order],
            unit_frac_active[low_active_order],
        )

        return (
            reordered_slice_matrices,
            unit_ids_in_order,
            unit_std_indices,
            unit_peak_times,
            unit_frac_active,
        )

    def get_slice_to_slice_unit_corr_from_stack(
        self,
        compare_func=compute_cross_correlation_with_lag,
        MIN_RATE_THRESHOLD=0.1,
        MIN_FRAC=0.3,
        max_lag=10,
        frac_active=None,
        n_jobs=-1,
    ):
        """Compute slice-to-slice similarity along the unit axis of event_stack (U, T, S).

        Output is a PairwiseCompMatrixStack of shape (S, S, U).

        Parameters:
            compare_func (callable): Comparison function from utils. Specify
                cross-correlation or cosine similarity. The default is cross
                correlation. See utils.py for details.
            MIN_RATE_THRESHOLD (float): Minimum mean firing rate to consider a
                slice valid for that neuron.
            MIN_FRAC (float): Maximum fraction of slices that can be skipped
                before a unit is deemed invalid (default 0.3 = 30%).
            max_lag (int): Maximum lag in frames to search for similarity. If
                None, lag is set to 0.
            frac_active (np.ndarray or None): Optional pre-computed
                fraction-active array of shape (U,) to override the internal
                rate-based validity check for computing averages. When
                provided, a unit's average is set to NaN if
                frac_active[u] < (1 - MIN_FRAC). MIN_RATE_THRESHOLD still
                controls which individual slice pairs are computed.
                Compatible sources: SpikeSliceStack.compute_frac_active and
                SpikeData.get_frac_active (frac_per_unit output).
            n_jobs (int): Number of threads for parallel computation. -1 uses
                all cores (default), 1 disables parallelism, None is serial.

        Returns:
            all_slice_corr_scores (PairwiseCompMatrixStack): Pairwise
                correlation scores between all slice pairs for each unit.
                Shape is (S, S, U) in the stack attribute.
            av_slice_corr_scores (np.ndarray): Average correlation per neuron
                across all valid slice pairs. Shape is (U,).
        """
        # Get dimensions
        event_stack = self.event_stack
        num_units = event_stack.shape[0]  # N
        num_time_bins = event_stack.shape[1]  # T
        num_slices = event_stack.shape[2]  # B

        # Validate frac_active override if provided
        if frac_active is not None:
            frac_active = np.asarray(frac_active, dtype=float)
            if frac_active.shape != (num_units,):
                raise ValueError(
                    f"frac_active must have shape ({num_units},), "
                    f"got {frac_active.shape}"
                )

        # Early return for single slice — pairwise comparison undefined (BUG-005)
        if num_slices < 2:
            warnings.warn(
                "Cannot compute slice-to-slice unit correlation with fewer than "
                "2 slices. Returning NaN.",
                RuntimeWarning,
            )
            av_slice_corr_scores = np.full(num_units, np.nan)
            all_slice_corr_scores = np.full((num_slices, num_slices, num_units), np.nan)
            return (
                PairwiseCompMatrixStack(stack=all_slice_corr_scores),
                av_slice_corr_scores,
            )

        # Initialize result matrices (compute in U x S x S, then transpose)
        av_slice_corr_scores = np.full(num_units, np.nan)
        all_slice_corr_scores = np.full((num_units, num_slices, num_slices), np.nan)

        lower_tri_indices = np.tril_indices(num_slices, k=-1)

        def _process_unit(unit):
            unit_corr = np.full((num_slices, num_slices), np.nan)
            counter = 0
            for ref_b in range(num_slices):
                ref_rate = event_stack[unit, :, ref_b]
                if np.mean(ref_rate) < MIN_RATE_THRESHOLD:
                    counter += 1
                    continue
                for comp_b in range(ref_b, num_slices):
                    comp_rate = event_stack[unit, :, comp_b]
                    if np.mean(comp_rate) < MIN_RATE_THRESHOLD:
                        continue
                    max_corr, _ = compare_func(ref_rate, comp_rate, max_lag)
                    unit_corr[comp_b, ref_b] = max_corr
                    unit_corr[ref_b, comp_b] = max_corr
            if frac_active is not None:
                unit_valid = frac_active[unit] >= (1 - MIN_FRAC)
            else:
                unit_valid = counter / num_slices <= MIN_FRAC
            av = (
                np.nanmean(unit_corr[lower_tri_indices[0], lower_tri_indices[1]])
                if unit_valid
                else np.nan
            )
            return unit, unit_corr, av

        n_workers = _resolve_n_jobs(n_jobs)
        if n_workers > 1 and num_units > 1:
            print(f"get_slice_to_slice_unit_corr: using thread pool ({n_workers} workers)")
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = pool.map(_process_unit, range(num_units))
        else:
            print("get_slice_to_slice_unit_corr: using serial backend")
            results = map(_process_unit, range(num_units))

        for unit, unit_corr, av in results:
            all_slice_corr_scores[unit] = unit_corr
            av_slice_corr_scores[unit] = av
        # Transpose from (U, S, S) to (S, S, U) for n×n×S convention
        all_slice_corr_scores = np.moveaxis(all_slice_corr_scores, 0, 2)
        # all_burst_corr_scores is now SxSxU and av_burst_corr_scores is U since its the mean correlation across all bursts.
        return (
            PairwiseCompMatrixStack(stack=all_slice_corr_scores),
            av_slice_corr_scores,
        )

    def get_slice_to_slice_time_corr_from_stack(
        self, compare_func=compute_cosine_similarity_with_lag, max_lag=0, n_jobs=-1
    ):
        """Compute slice-to-slice similarity along the time axis of event_stack (U, T, S).

        Output is a PairwiseCompMatrixStack of shape (S, S, T).

        Parameters:
            compare_func (callable): Comparison function from utils. Specify
                cross-correlation or cosine similarity. The default is cosine
                similarity. See utils.py for details.
            max_lag (int): Maximum lag in frames to search for similarity. If
                None, lag is set to 0.
            n_jobs (int): Number of threads for parallel computation. -1 uses
                all cores (default), 1 disables parallelism, None is serial.

        Returns:
            all_slice_corr_scores (PairwiseCompMatrixStack): Pairwise
                correlation scores between all slice pairs for each time bin.
                Shape is (S, S, T) in the stack attribute.
            av_slice_corr_scores (np.ndarray): Average correlation per time
                bin across all valid slice pairs. Shape is (T,).
        """
        # Get dimensions
        event_stack = self.event_stack
        num_time_bins = event_stack.shape[1]  # T
        num_slices = event_stack.shape[2]  # S

        # Early return for single slice — pairwise comparison undefined
        if num_slices < 2:
            warnings.warn(
                "Cannot compute slice-to-slice time correlation with fewer than "
                "2 slices. Returning NaN.",
                RuntimeWarning,
            )
            av_slice_corr_scores = np.full(num_time_bins, np.nan)
            all_slice_corr_scores = np.full(
                (num_slices, num_slices, num_time_bins), np.nan
            )
            return (
                PairwiseCompMatrixStack(stack=all_slice_corr_scores),
                av_slice_corr_scores,
            )

        # Initialize result matrices (compute in T x S x S, then transpose)
        av_slice_corr_scores = np.full(num_time_bins, np.nan)
        all_slice_corr_scores = np.full((num_time_bins, num_slices, num_slices), np.nan)

        lower_tri_indices = np.tril_indices(num_slices, k=-1)

        def _process_time(t):
            time_corr = np.full((num_slices, num_slices), np.nan)
            for ref_b in range(num_slices):
                ref_rate = event_stack[:, t, ref_b]
                for comp_b in range(ref_b, num_slices):
                    comp_rate = event_stack[:, t, comp_b]
                    max_corr, _ = compare_func(ref_rate, comp_rate, max_lag)
                    time_corr[comp_b, ref_b] = max_corr
                    time_corr[ref_b, comp_b] = max_corr
            av = np.nanmean(time_corr[lower_tri_indices[0], lower_tri_indices[1]])
            return t, time_corr, av

        n_workers = _resolve_n_jobs(n_jobs)
        if n_workers > 1 and num_time_bins > 1:
            print(f"get_slice_to_slice_time_corr: using thread pool ({n_workers} workers)")
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = pool.map(_process_time, range(num_time_bins))
        else:
            print("get_slice_to_slice_time_corr: using serial backend")
            results = map(_process_time, range(num_time_bins))

        for t, time_corr, av in results:
            all_slice_corr_scores[t] = time_corr
            av_slice_corr_scores[t] = av
        # Transpose from (T, S, S) to (S, S, T) for n×n×S convention
        all_slice_corr_scores = np.moveaxis(all_slice_corr_scores, 0, 2)
        # all_slice_corr_scores is now SxSxT and av_burst_corr_scores is T

        return (
            PairwiseCompMatrixStack(stack=all_slice_corr_scores),
            av_slice_corr_scores,
        )

    def convert_to_list_of_RateData(self):
        """Create a list of RateData objects from the 3D event_stack.

        Returns:
            output (list): List of RateData objects. Length equals S.
        """
        output = []
        # U x T x S
        for s_idx in range(self.event_stack.shape[2]):
            matrix = self.event_stack[:, :, s_idx]
            start, end = self.times[s_idx]
            time = start + np.arange(matrix.shape[1]) * self.step_size
            if time[-1] > end:
                # Extremely rare edge case with floating point calculation. Should never happen but just in case
                time = np.clip(time, start, end - np.finfo(float).eps)
            # time = np.arange(start, end, self.step_size)
            rate_obj = RateData(matrix, time, neuron_attributes=self.neuron_attributes)
            output.append(rate_obj)
        return output

    def unit_to_unit_correlation(
        self, compare_func=compute_cross_correlation_with_lag, max_lag=10, n_jobs=-1
    ):
        """Compute unit-to-unit similarity along the slice axis of event_stack (U, T, S).

        Output is a PairwiseCompMatrixStack of shape (U, U, S).

        Parameters:
            compare_func (callable): Comparison function from utils. Specify
                cross-correlation or cosine similarity. The default is cross
                correlation. See utils.py for details.
            max_lag (int): Maximum lag in frames to search for similarity. If
                None, lag is set to 0.
            n_jobs (int): Number of threads for parallel computation. -1 uses
                all cores (default), 1 disables parallelism, None is serial.

        Returns:
            max_corr_array (PairwiseCompMatrixStack): Pairwise correlation
                scores between all unit pairs for each slice. Shape is
                (U, U, S) in the stack attribute.
            max_corr_lag_array (PairwiseCompMatrixStack): Lag where
                correlation between each pair is at its maximum. Shape is
                (U, U, S) in the stack attribute.
            av_max_corr (np.ndarray): Average correlation per slice across
                all valid unit pairs. Shape is (S,).
            av_max_corr_lag (np.ndarray): Average lag where correlation
                between each pair is at its maximum. Shape is (S,).
        """
        num_units = self.event_stack.shape[0]
        num_slices = self.event_stack.shape[2]

        # Early return for single unit — pairwise comparison undefined (BUG-005)
        if num_units < 2:
            warnings.warn(
                "Cannot compute unit-to-unit correlation with fewer than "
                "2 units. Returning NaN.",
                RuntimeWarning,
            )
            nan_stack = np.full((num_units, num_units, num_slices), np.nan)
            nan_avgs = np.full(num_slices, np.nan)
            return (
                PairwiseCompMatrixStack(stack=nan_stack, times=self.times),
                PairwiseCompMatrixStack(stack=nan_stack.copy(), times=self.times),
                nan_avgs,
                nan_avgs.copy(),
            )

        max_corr_stack = []
        max_corr_lag_stack = []
        rate_data_stack = self.convert_to_list_of_RateData()
        for i in range(len(rate_data_stack)):
            rate_data = rate_data_stack[i]
            # This gives 2 UxU matrices
            max_corr_matrix, lag_corr_matrix = rate_data.get_pairwise_fr_corr(
                compare_func, max_lag, n_jobs=n_jobs
            )
            max_corr_stack.append(max_corr_matrix.matrix)
            max_corr_lag_stack.append(lag_corr_matrix.matrix)
        # Make the list of correlation matrices into a 3d matrix (S x U x U)
        max_corr_array = np.stack(max_corr_stack, axis=0)
        max_corr_lag_array = np.stack(max_corr_lag_stack, axis=0)

        num_units = max_corr_array.shape[1]
        lower_tri_indices = np.tril_indices(num_units, k=-1)

        # Find the averages to get a single dimension array of averages
        av_max_corr = np.nanmean(
            max_corr_array[:, lower_tri_indices[0], lower_tri_indices[1]], axis=(1)
        )  # shape (S,)
        av_max_corr_lag = np.nanmean(
            max_corr_lag_array[:, lower_tri_indices[0], lower_tri_indices[1]], axis=(1)
        )  # shape (S,)

        # Transpose from (S, U, U) to (U, U, S) for n×n×S convention
        max_corr_array = np.moveaxis(max_corr_array, 0, 2)
        max_corr_lag_array = np.moveaxis(max_corr_lag_array, 0, 2)

        return (
            PairwiseCompMatrixStack(stack=max_corr_array, times=self.times),
            PairwiseCompMatrixStack(stack=max_corr_lag_array, times=self.times),
            av_max_corr,
            av_max_corr_lag,
        )

    def subset(self, units, by=None):
        """Extract a subset of units/neurons from the rate slice stack.

        Parameters:
            units (list or array): Unit indices to extract. If by is None,
                this should always be a list of ints. If by is not None,
                the list can contain ints or strings.
            by (str or None): Neuron attribute key to match against. Only
                use this if you initialized the object with
                neuron_attributes. Set to the key that contains neuron_id
                values. None selects by index (default).

        Returns:
            result (RateSliceStack): New RateSliceStack object containing
                only the specified units.
        """
        N = self.event_stack.shape[0]
        if isinstance(units, int):
            units = [units]
        if isinstance(units, str):
            units = [units]
        units = set(units)
        if by is None:
            for u in units:
                if not isinstance(u, (int, np.integer)):
                    raise TypeError(f"Unit index must be an integer, got {type(u)}")
                if u < 0 or u >= N:
                    raise ValueError(f"Unit index {u} out of range for {N} units.")
        if by is not None:
            # VALUE-BASED: Look up by neuron_attribute
            if self.neuron_attributes is None:
                raise ValueError("can't use `by` without `neuron_attributes`")

            _missing = object()
            units = {
                i
                for i in range(N)
                if _get_attr(self.neuron_attributes[i], by, _missing) in units
            }
        units = sorted(units)
        neuron_attributes = None
        if self.neuron_attributes is not None:
            neuron_attributes = [self.neuron_attributes[i] for i in units]

        new_stack = self.event_stack[units, :, :]
        return RateSliceStack(
            event_matrix=new_stack,
            times_start_to_end=self.times,
            step_size=self.step_size,
            neuron_attributes=neuron_attributes,
        )

    def subtime_by_index(self, start_idx, end_idx):
        """Extract a subset of time bins from every slice using index values.

        Trims along the time axis (T dimension) while preserving all slices
        (S dimension).

        Parameters:
            start_idx (int): Starting time bin index (inclusive). Supports
                negative indexing.
            end_idx (int): Ending time bin index (exclusive). Supports
                negative indexing.

        Returns:
            result (RateSliceStack): New RateSliceStack where each slice
                contains only the specified time bins. Shape changes from
                (U, T, S) to (U, T_trimmed, S).

        Notes:
            - Original timestamps are preserved (not shifted to zero). To get
              shifted-to-zero timestamps, create a new RateSliceStack.
            - All slices, neuron_attributes, and step_size are carried over
              from the original.
        """
        T = self.event_stack.shape[1]
        if start_idx < 0:
            start_idx += T
        if end_idx < 0:
            end_idx += T
        if start_idx < 0 or start_idx >= T:
            raise ValueError(f"start_idx {start_idx} out of range for T={T}")
        if end_idx <= start_idx or end_idx > T:
            raise ValueError(f"end_idx {end_idx} invalid")

        new_stack = self.event_stack[:, start_idx:end_idx, :].copy()

        new_times = []
        for t in self.times:
            new_start = t[0] + start_idx * self.step_size
            new_end = t[0] + end_idx * self.step_size
            new_times.append((new_start, new_end))

        return RateSliceStack(
            event_matrix=new_stack,
            times_start_to_end=new_times,
            step_size=self.step_size,
            neuron_attributes=self.neuron_attributes,
        )

    def subslice(self, slices):
        """Extract a subset of slices from the event stack using index values.

        Trims along the slice axis (S dimension) while preserving all time
        bins (T dimension).

        Parameters:
            slices (int or list): Slice index or list of slice indices to
                extract.

        Returns:
            result (RateSliceStack): New RateSliceStack containing only the
                specified slices. Shape changes from (U, T, S) to
                (U, T, S_trimmed).

        Notes:
            - All units, neuron_attributes, and step_size are carried over
              from the original.
        """
        length = self.event_stack.shape[2]
        if isinstance(slices, int):
            slices = [slices]
        for s in slices:
            if s >= length or s < -length:
                raise ValueError(
                    f"One or more slice indices out of range for S={length}"
                )
        slices = sorted(slices)
        new_times = []
        for s in slices:
            new_times.append(self.times[s])
        new_stack = self.event_stack[:, :, slices]
        return RateSliceStack(
            event_matrix=new_stack,
            times_start_to_end=new_times,
            step_size=self.step_size,
            neuron_attributes=self.neuron_attributes,
        )

    def get_unit_timing_per_slice(self, MIN_RATE_THRESHOLD=0.1):
        """Compute the peak firing rate time bin for each unit in each slice.

        Returns a ``(U, S)`` matrix where entry ``[u, s]`` is the time bin
        index of the peak firing rate for unit u in slice s. Units whose
        peak rate falls below MIN_RATE_THRESHOLD in a slice are marked NaN.

        Parameters:
            MIN_RATE_THRESHOLD (float): Minimum peak firing rate for a unit
                to count as active in a slice (default: 0.1).

        Returns:
            timing_matrix (np.ndarray): Array of shape ``(U, S)`` with peak
                time bin indices (integers cast to float for NaN support).
                These can be used to index directly into the ``(U, T, S)``
                event stack. NaN where the unit is inactive.

        Notes:
            - Values are bin indices, not milliseconds. This differs from
              ``SpikeSliceStack.get_unit_timing_per_slice`` which returns
              milliseconds. Both representations preserve rank order, so
              ``rank_order_correlation`` produces identical results either way.
            - The returned matrix can be passed to ``rank_order_correlation``
              to compute Spearman rank correlations between slice pairs.
        """
        slice_matrices = self.event_stack
        unit_max_indices = np.argmax(slice_matrices, axis=1).astype(float)
        unit_max_rates = np.max(slice_matrices, axis=1)
        unit_max_indices[unit_max_rates < MIN_RATE_THRESHOLD] = np.nan
        return unit_max_indices

    def rank_order_correlation(
        self,
        timing_matrix=None,
        MIN_RATE_THRESHOLD=0.1,
        min_overlap=3,
        min_overlap_frac=None,
        n_shuffles=100,
        seed=1,
        n_jobs=-1,
    ):
        """Compute Spearman rank-order correlation of unit timing between all slice pairs.

        For each pair of slices, only units active in both slices (non-NaN in
        both columns of the timing matrix) are included. If the overlap falls
        below the required minimum, the pair is set to NaN.

        When ``n_shuffles > 0``, the rank orders are shuffled n_shuffles
        times for each pair to build a null distribution, and the raw
        correlation is z-score normalised against it.

        Parameters:
            timing_matrix (np.ndarray or None): Array of shape ``(U, S)`` with
                timing values per unit per slice. NaN entries mark inactive
                units. Typically produced by ``get_unit_timing_per_slice``.
                When None, computed automatically using MIN_RATE_THRESHOLD.
            MIN_RATE_THRESHOLD (float): Minimum peak firing rate threshold
                (default: 0.1). Only used when timing_matrix is None.
            min_overlap (int): Minimum number of units that must be active in
                both slices (default: 3).
            min_overlap_frac (float or None): Minimum fraction of total units
                that must be active in both slices (default: None). When
                provided, the effective threshold is
                ``max(min_overlap, ceil(min_overlap_frac * U))``.
            n_shuffles (int): Number of shuffle iterations for z-scoring
                (default: 100). Set to 0 to return raw Spearman correlations.
                Values between 1 and 4 are rejected (minimum 5 required for
                a meaningful null distribution).
            seed (int or None): Random seed for reproducibility of the shuffle
                (default: 1).
            n_jobs (int): Number of threads for parallel computation. -1 uses
                all cores (default), 1 disables parallelism, None is serial.

        Returns:
            corr_matrix (PairwiseCompMatrix): Spearman correlation matrix of
                shape ``(S, S)``. When ``n_shuffles > 0``, values are z-scores.
                When ``n_shuffles == 0``, values are raw Spearman correlations.
            av_corr (float): Average correlation (or z-score) across all valid
                lower-triangle pairs.
            overlap_matrix (PairwiseCompMatrix): Matrix of shape ``(S, S)``
                with fraction of units active in both slices.
        """
        from .utils import _rank_order_correlation_from_timing

        if timing_matrix is None:
            timing_matrix = self.get_unit_timing_per_slice(
                MIN_RATE_THRESHOLD=MIN_RATE_THRESHOLD
            )

        return _rank_order_correlation_from_timing(
            timing_matrix,
            min_overlap=min_overlap,
            min_overlap_frac=min_overlap_frac,
            n_shuffles=n_shuffles,
            seed=seed,
            n_jobs=n_jobs,
        )
