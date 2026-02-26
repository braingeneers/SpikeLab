import numpy as np
from scipy import signal
from .ratedata import RateData
from .spikedata import SpikeData


from .utils import (
    compute_cross_correlation_with_lag,
    compute_cosine_similarity_with_lag,
    extract_lower_triangle_features,
    PCA_reduction,
)


class RateSliceStack:
    """
    Description:
    -----------
    A data structure where the underlying data is a 3D matrix of shape (U,T,S) and allows the user to compute correlation
    matrices/cosine similarity matrices.
        - U: Units (refers to neuron/neuron clusters)
        - T: Time bins
        - S: Slices (can be bursts, events, etc)
    If the user inputs an event_matrix, then no other parameters are needed. Otherwise, all parameters are required.
    The instance variables are the same despite the input option and all methods will work the same.

    Parameters:
    -----------
    You can either input data_obj or event_matrix as your underlying data.
    Option #1: data_obj
        data_obj (SpikeData or RateData): A data object in either one of these forms.
        There are 2 choices for time input:
            Choice A)
                times_start_to_end (list): Each entry must be a tuple. Each tuple is (start, end) and
                                        represents the start and end times of a desired slice.
                                        Each tuple must have same duration.
            Choice B) (both of the following must be input for this option)
                time_peaks (list): List of times as int or float where there is a burst peak or stimulation event.
                                This variable must be pairedwith time bounds
                time_bounds (tuple): Single tuple (left_bound, right_bound). If you put (250,500), then this means
                                    250 ms before peak and 500 ms after peak.
        sigma_ms (float): Smoothing factor for computing isi if you input a SpikeData object. Otherwise, not used

    Option #2: event_matrix
        event_matrix (3D array): A 3D array of shape (U, T, S). If the user inputs this, then no other inputs are needed.
        times_start_to_end (list): Each entry must be a tuple. Each tuple is (start, end) and represents the start and
                                   end times of a desired slice. Each tuple must have same duration. Length must equal S.
                                   If none, will be automatically computed with step_size 1.0
        step_size (float): Time resolution in milliseconds between consecutive time bins. If None, becomes 1.0.

    Instance Variables:
    --------
    self.event_stack (array): 3D array of shape (U, T, S) where
        - U: Nuber of units (refers to neuron/neuron clusters)
        - T: Number of time bins
        - S: Number of slices (can be bursts, events, etc)
    self.times (list of tuples): List of (start, end) time bounds for each slice, sorted chronologically. Length equals S (number of slices).
                            Example: [(100, 200), (500, 600), (1000, 1100)]
    self.step_size (float): Time resolution in milliseconds between consecutive time bins. Inferred from input data. For SpikeData input, defaults to 1.0ms.
                            Example: 1.0 means time bins are at [100, 101, 102, ...] ms
    self.neuron_attributes (list or None): List of attribute objects, one per unit, containing arbitrary metadata about each neuron. None if not provided.

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
            raise ValueError(
                "Must input only one of data_obj (option 1) or event_matrix (option 2), not both"
            )
        # Option 1: Using data_obj
        if data_obj is not None:
            if not isinstance(data_obj, (SpikeData, RateData)):
                # CHECK IF ISINSTANCE WORKS WITH THESE TWO CLASSES
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
            times_start_to_end = self._validate_time_start_to_end(times_start_to_end)

            # Actual constructor

            if isinstance(data_obj, SpikeData):
                # Make it step_size 1
                all_times = np.arange(0, data_obj.length, 1.0)
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
                    rate_obj_slice = data_obj.subtime(start, end, shift_time=False)
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
                times_start_to_end = self._validate_time_start_to_end(
                    times_start_to_end
                )
                # Make sure there is a (start,end) tuple for each slice
                if len(times_start_to_end) != event_matrix.shape[2]:
                    raise ValueError(
                        "times_start_to_end must have the same length as the last dimension of event_matrix"
                    )
            self.event_stack = event_matrix
            self.times = times_start_to_end

        self.neuron_attributes = None
        if neuron_attributes is not None:
            self.neuron_attributes = neuron_attributes.copy()
            if len(neuron_attributes) != self.event_stack.shape[0]:
                raise ValueError(
                    f"neuron_attributes has {len(neuron_attributes)} items "
                    f"but event_stack has {self.event_stack.shape[0]} units"
                )

        

    def _validate_time_start_to_end(self, times_start_to_end):
        """
        Validates that the list of (start,end) tuples has the same duration of time_steps and are in proper format for object constructor.

        Parameters:
        -----------
        times_start_to_end (list): Each entry must be a tuple. Each tuple is (start, end) and represents the start and
                                   end times of a desired slice. Each tuple must have same duration. Length must equal S.

        Returns:
        --------
        valid_time_tuples (list): Sorted list of valid (start, end) tuples, with negative-start 
                                  windows removed.
        """
        if not isinstance(times_start_to_end, list):
            raise TypeError("times must be a list of tuples")
        time_diff_check = []
        valid_time_tuples = []
        times_start_to_end = sorted(times_start_to_end)
        for i, time_window in enumerate(times_start_to_end):
            if not isinstance(time_window, tuple):
                raise TypeError(f"Element {i} of times is not a tuple: {time_window}")
            if len(time_window) != 2:
                raise TypeError(
                    f"Element {i} of times must be a tuple of length 2 (start, end): {time_window}"
                )
            if not (
                isinstance(time_window[0], (int, float, np.number))
                and isinstance(time_window[1], (int, float, np.number))
            ):
                raise TypeError(
                    f"Start and end times in element {i} must be numbers: {time_window}"
                )
            if time_window[0] >= time_window[1]:
                raise ValueError(
                    f"Start time must be less than end time in element {i}: {time_window}"
                )

            # Check if any times are negative due to time_bounds and time_peaks operation
            if time_window[0] < 0:
                continue

            time_diff_check.append(time_window[1] - time_window[0])
            # We only want to address time windows that are above 0 (recording start) and below recording end
            valid_time_tuples.append(time_window)
            if len(set(time_diff_check)) > 1:
                raise ValueError("All time windows must have the same length")
        return valid_time_tuples

    def order_units_across_slices(self, agg_func, MIN_RATE_THRESHOLD=0.1):
        """
        Reorders the units across slices from earliest to latest peak firing rate in underlying 3D self.event_stack matrix
        that is UxTxS (units x time_bin x slice)

        Parameters:
        agg_func (string): This should be either "median" or "mean". This is for calculating the median/mean time when that
                           unit has peak firing rate.
        MIN_RATE_THRESHOLD (float): Minimum peak firing rate for a slice to be included in the ordering calculation.
                                    Slices where a unit's max rate < threshold are excluded from that unit's typical
                                    peak time calculation.


        Returns:
        --------
        reordered_slice_matrices (array): This is 3D self.event_stack but the 0th dimension U is reordered temporally
                                    Now, the first unit/neuron is the one that usually fires off first across all slices.
        unit_ids_in_order(array): Array of size U which is original unit/neuron indices sorted by their typical firing order.
                                  For example, [3, 1, 0, 2] means unit/neuron 3 fires first, then unit/neuron 1,
                                  then unit/neuron 0, then unit/neuron 2. So unit/neuron 3 is now the first unit/neuron in reordered_burst_matrices.
                                  Use this to map back to original unit/neuron IDs.
        unit_std_indices(array): Array of the size U. Shows the standard deviation of max firing rate times for units in original order.
                                 If a unit has lower standard deviation, it means it has a similar firing rate time across
                                 all bursts. For example, [.1,.5,.6] means that unit 0 has standard deviation of .1
        unit_peak_times(array): Array of size U. Contains the median/mean (depending on agg_func input) firing peak time_bins for each unit
                                in original order. For example, [2,8,5] means that unit 0's mean/median peak firing rate occurs in time_bin 2
        """
        # burst_matrices is U x T x S
        slice_matrices = self.event_stack

        # This is a matrix (UxS) where row is unit, and each column is a burst. Value is the time index
        # firing rate peak for unit U in slice S
        unit_max_indices_matrix = np.argmax(slice_matrices, axis=1)
        # This matrix is same size as one above, but instead of time_bins with max rates, the values are the max rates
        unit_max_rates = np.max(slice_matrices, axis=1)
        # Make mask for removing those below threshold
        mask = unit_max_rates >= MIN_RATE_THRESHOLD

        unit_max_indices_matrix = unit_max_indices_matrix.astype(float)
        unit_max_indices_matrix[~mask] = np.nan

        unit_std_indices = np.nanstd(unit_max_indices_matrix, axis=1)

        # This gives you a list of size N. Now you have median peak time for each neuron
        if agg_func == "median":
            unit_peak_times = np.round(
                np.nanmedian(unit_max_indices_matrix, axis=1)
            ).astype(int)
        elif agg_func == "mean":
            unit_peak_times = np.round(
                np.nanmean(unit_max_indices_matrix, axis=1)
            ).astype(int)
        else:
            raise ValueError(
                f"{agg_func} is not a valid input option. Must be either median or mean"
            )

        # arr = [5,2,9,1] means neuron 0 max firing at time 5, neuron 1 max firing at time 2.
        # So np.argsort(arr) returns [3, 1, 0, 2] which means neuron 3 has max firing first, then neuron 1, etc

        unit_ids_in_order = np.argsort(unit_peak_times)
        # Reorder the units in orginal slice_matrices so that they are in temporal order
        reordered_slice_matrices = slice_matrices[unit_ids_in_order, :, :]

        return (
            reordered_slice_matrices,
            unit_ids_in_order,
            unit_std_indices,
            unit_peak_times,
        )

    def get_slice_to_slice_unit_corr_from_stack(
        self,
        compare_func=compute_cross_correlation_with_lag,
        MIN_RATE_THRESHOLD=0.1,
        MIN_FRAC=0.3,
        max_lag=10,
    ):
        """
        Compute slice-to-slice (aka burst-to-burst) similarity along the 0th axis of self.event_stack (U x T x S)
        to give output size (U x S x S)

        Parameters:
        -----------
        compare_func (method in utils): Specify if you want to compare signals with correaltion or cosine similarity functions.
                                          The default is cross correlation. These functions can be insepcted further in utils.py
        MIN_RATE_THRESHOLD (float): Minimum mean firing rate to consider a slice valid for that neuron
        MIN_FRAC (float): Maximum fraction of slice that can be skipped before a unit is deemed invalid (default 0.3 = 30%)
        max_lag (int): Maximum lag in frames to search for similarity. If None, lag is set to 0.

        Returns:
        --------
        all_slice_corr_scores (array): Pairwise correlation scores between all slice pairs for each unit shape (U, S, S)
        av_slice_corr_scores (array): Average correlation per neuron across all valid slice pairs shape (U,)

        """
        # Get dimensions
        event_stack = self.event_stack
        num_units = event_stack.shape[0]  # N
        num_time_bins = event_stack.shape[1]  # T
        num_slices = event_stack.shape[2]  # B

        # Initialize result matrices
        av_slice_corr_scores = np.full(num_units, np.nan)
        all_slice_corr_scores = np.full((num_units, num_slices, num_slices), np.nan)

        lower_tri_indices = np.tril_indices(num_slices, k=-1)

        # For each neuron
        for unit in range(num_units):
            # Counter for skipped slices
            counter = 0

            # Loop through each slice. This is your reference signal
            for ref_b in range(num_slices):
                # Reference vector
                ref_rate = event_stack[unit, :, ref_b]

                # Check if mean firing rate is above threshold
                if np.mean(ref_rate) < MIN_RATE_THRESHOLD:
                    # Count each time a reference slice is inactive for this unit
                    counter += 1
                    continue

                for comp_b in range(ref_b, num_slices):
                    # Comp vector
                    comp_rate = event_stack[unit, :, comp_b]

                    # Check if mean firing rate is above threshold
                    if np.mean(comp_rate) < MIN_RATE_THRESHOLD:
                        continue

                    # Compute similarity, we only want one output, not the lagged one.
                    max_corr, _ = compare_func(ref_rate, comp_rate, max_lag)

                    # Store results
                    all_slice_corr_scores[unit, comp_b, ref_b] = max_corr
                    all_slice_corr_scores[unit, ref_b, comp_b] = max_corr

            # If less than MIN_FRAC bursts were skipped
            if counter / num_slices <= MIN_FRAC:
                # Average results over all pairs in lower triangle. Don't want to include diagonol in mean calculation.
                av_slice_corr_scores[unit] = np.nanmean(
                    all_slice_corr_scores[
                        unit, lower_tri_indices[0], lower_tri_indices[1]
                    ]
                )
        # all_burst_corr_scores is UxSxS and av_burst_corr_scores is N since its the mean correlation across all bursts.
        return all_slice_corr_scores, av_slice_corr_scores

    def get_slice_to_slice_time_corr_from_stack(
        self, compare_func=compute_cosine_similarity_with_lag, max_lag=0
    ):
        """
        Compute slice-to-slice similarity along the 1st axis of RateSliceStack self.event_stack (U x T x S)
        This is done along the time axis, making the output size be (T x S x S) which doesn't require thresholding.

        Parameters:
        -----------
        compare_func (method in utils): Specify if you want to compare signals with correaltion or cosine similarity functions.
                                        The default is cosine similarity. These functions can be insepcted further in utils.py
        max_lag (int): Maximum lag in frames to search for similarity. If None, lag is set to 0.

        Returns:
        --------
        all_slice_corr_scores (array): Pairwise correlation scores between all slice pairs for each time_bin shape (T, S, S)
        av_slice_corr_scores (array): Average correlation per time_bin across all valid slice pairs shape (T,)

        """
        # Get dimensions
        event_stack = self.event_stack
        num_units = event_stack.shape[0]  # N
        num_time_bins = event_stack.shape[1]  # T
        num_slices = event_stack.shape[2]  # B

        # Initialize result matrices
        av_slice_corr_scores = np.full(num_time_bins, np.nan)
        all_slice_corr_scores = np.full((num_time_bins, num_slices, num_slices), np.nan)

        lower_tri_indices = np.tril_indices(num_slices, k=-1)

        # For each neuron
        for time in range(num_time_bins):

            # For each reference burst
            for ref_b in range(num_slices):
                # Reference vector
                ref_rate = event_stack[:, time, ref_b]

                # Start at ref_b since the output must be symmetric, so we only need to do half the computation.
                for comp_b in range(ref_b, num_slices):
                    # Comparison vector
                    comp_rate = event_stack[:, time, comp_b]

                    # Compute similarity
                    max_corr, _ = compare_func(ref_rate, comp_rate, max_lag)

                    # Store results
                    all_slice_corr_scores[time, comp_b, ref_b] = max_corr

                    all_slice_corr_scores[time, ref_b, comp_b] = max_corr

            av_slice_corr_scores[time] = np.nanmean(
                all_slice_corr_scores[time, lower_tri_indices[0], lower_tri_indices[1]]
            )
        # all_slice_corr_scores is TxSxS and av_burst_corr_scores is T

        return all_slice_corr_scores, av_slice_corr_scores

    def PCA_on_lower_diagnol_corr_matrix(self, all_burst_corr_scores, n_components=2):
        """
        Apply PCA to reduce feature dimensions.

        Parameters:
        -----------
        feature_matrix : array, shape (B, F)
            Each row is a burst, each column is a feature

        n_components : int
            Number of components (default: 2)

        Returns:
        --------
        pca_result : array, shape (B, n_components)
            Reduced representation
        """
        # lower triangle is S x F (or U x F if you input U x S x S) (or T x F if you input T x S x S)
        lower_triangle = extract_lower_triangle_features(all_burst_corr_scores)
        pca_result = PCA_reduction(lower_triangle, n_components)

        return pca_result

    def convert_to_list_of_RateData(self):
        """
        Creates a stack of RateData objects from the 3D self.event_matrix

        Parameters:
        -----------
        No inputs, it just uses the underlying self.event_matrix

        Returns:
        --------
        output (list): List of RateData objects. Length of list = S
        """
        output = []
        # U x T x S
        for slice in range(self.event_stack.shape[2]):
            matrix = self.event_stack[:, :, slice]
            start, end = self.times[slice]
            time = start + np.arange(matrix.shape[1]) * self.step_size
            if time[-1] > end:
                # Extremely rare edge case with floating point calculation. Should never happen but just in case
                time = np.clip(time, start, end - np.finfo(float).eps)
            # time = np.arange(start, end, self.step_size)
            rate_obj = RateData(matrix, time)
            output.append(rate_obj)
        return output

    def unit_to_unit_correlation(
        self, compare_func=compute_cross_correlation_with_lag, max_lag=10
    ):
        """
        Compute unit-to-unit similarity along the last axis of RateSliceStack self.event_stack (U x T x S)
        This is done along the slice axis, making the output size be (S x U x U) which doesn't require thresholding.

        Parameters:
        -----------
        compare_func (method in utils): Specify if you want to compare signals with correaltion or cosine similarity functions.
                                          The default is cosine similarity. These functions can be insepcted further in utils.py
        max_lag (int): Maximum lag in frames to search for similarity. If None, lag is set to 0.

        Returns:
        --------
        max_corr_array (array): Pairwise correlation scores between all unit pairs for each slice. Shape is (S, U, U)
        max_corr_lag_array (array): Lag where correlation between pair is at max. Shape is (S, U, U)
        av_max_corr (array): Average correlation per time_bin across all valid slice pairs. Shape is (S,)
        av_max_corr_lag (array): Average lag where correlation between pair is at max. Shape is (S,)

        """
        max_corr_stack = []
        max_corr_lag_stack = []
        rate_data_stack = self.convert_to_list_of_RateData()
        for i in range(len(rate_data_stack)):
            rate_data = rate_data_stack[i]
            # This gives 2 UxU matrices
            max_corr_matrix, lag_corr_matrix = rate_data.get_pairwise_fr_corr(
                compare_func, max_lag
            )
            max_corr_stack.append(max_corr_matrix)
            max_corr_lag_stack.append(lag_corr_matrix)
        # Make the list of correlation matrices into a 3d matrix
        max_corr_array = np.stack(max_corr_stack, axis=0)
        max_corr_lag_array = np.stack(max_corr_lag_stack, axis=0)

        num_units = max_corr_array.shape[1]
        lower_tri_indices = np.tril_indices(num_units, k=-1)

        # Find the averages to get a single dimension array of averages
        av_max_corr = np.nanmean(
            max_corr_array[:, lower_tri_indices[0], lower_tri_indices[1]], axis=(1)
        )  # shape (B,)
        av_max_corr_lag = np.nanmean(
            max_corr_lag_array[:, lower_tri_indices[0], lower_tri_indices[1]], axis=(1)
        )  # shape (B,)
        return max_corr_array, max_corr_lag_array, av_max_corr, av_max_corr_lag

    def subset(self, units, by=None):
        """
        Extract a subset of units/neurons from the rateslicestack. Index-based if by = None.

        Parameters:
        units (list or array): Unit indices to extract. If by = None, then this should be always be a list of ints. 
                               If by != None, then the list can be a list of ints or strings.
        by (string): This is None by default. Only use this if you initialized object with neuron_attributes dictionary.
                     If you have neuron_attributes, set variable "by" to be the key that contains neuron_id values.

        Returns:
        RateSliceStack: New RateSliceStack object containing only the specified units
        """
        N = self.event_stack.shape[0]
        if isinstance(units, int):
            units = [units]
        if isinstance(units, str):
            units = [units]
        units = set(units)
        if by is not None:
            # VALUE-BASED: Look up by neuron_attribute
            if self.neuron_attributes is None:
                raise ValueError("can't use `by` without `neuron_attributes`")

            _missing = object()
            units = {
                i
                for i in range(N)
                if getattr(self.neuron_attributes[i], by, _missing) in units
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
