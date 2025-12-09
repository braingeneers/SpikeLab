import SpikeData
import RateData
import numpy as np
from scipy import signal
from .ratedata import *
from .spikedata import *


from .utils import (
    compute_cross_correlation_with_lag,
    compute_cosine_similarity_with_lag,
    extract_lower_triangle_features,
    PCA_reduction,
)


class RateSliceStack:
    """
    Parameters:
    -----------
    data_obj (SpikeData or RateData): A data object in either one of these forms.
    There are 2 options for time input:
        Option #1
            times_start_to_end (list): Each entry must be a tuple. Each tuple is (start, end) and
                                       represents the start and end times of a burst/stimulation event.
                                       Each tuple must have same duration.
        Option #2 (both of the following must be input for this option)
            time_peaks (list): List of times as int or float where there is a burst peak or stimulation event.
                               This variable must be pairedwith time bounds
            time_bounds (tuple): Single tuple (left_bound, right_bound). If you put (250,500), then this means
                                250 ms before peak and 500 ms after peak.
    sigma_ms (float): Smoothing factor for computing isi if you input a SpikeData object

    Instance Variables:
    --------
    self.event_stack (array): 3D array of shape (N, T, B) where
        - N: number of  units/neurons
        - T: number of time bins per slice
        - B: number of slices/bursts/events
    self.times (list of tuples): List of (start, end) time bounds for each slice, sorted chronologically. Length equals B (number of slices).
                            Example: [(100, 200), (500, 600), (1000, 1100)]
    self.step_size (float): Time resolution in milliseconds between consecutive time bins. Inferred from input data. For SpikeData input, defaults to 1.0ms.
                       Example: 1.0 means time bins are at [100, 101, 102, ...] ms

    """

    def __init__(
        self,
        data_obj,
        times_start_to_end=None,
        time_peaks=None,
        time_bounds=None,
        sigma_ms=10,
    ):
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
                isinstance(time_window[0], (int, float))
                and isinstance(time_window[1], (int, float))
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

        times_start_to_end = valid_time_tuples

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
            # I use subtime here to extract a burst event
            for time in times_start_to_end:
                start = time[0]
                end = time[1]
                rate_obj_slice = data_obj.subtime(start, end)
                event_matrix = rate_obj_slice.inst_Frate_data
                event_stack.append(event_matrix)

        # Converts to a 3d array
        event_stack = np.stack(event_stack, axis=2)
        # This makes event stack be N x T x B
        self.event_stack = event_stack

    def order_units_across_bursts(self):
        """
        Reorders the neurons in bursts from earliest to latest peak firing rate.

        Parameters:
        No inputs. It uses the underlying 3D self.event_stack matrix that is NxTxB (neuron x time_bin x burst/event)
        Returns:
        --------
        reordered_burst_matrices: This is 3D self.event_stack but the 0th dimension N is reordered temporally
                                    Now, the first neuron is the one that usually fires off across all bursts
        neurons_ids_in_order: Array of original neuron indices sorted by their typical firing order.
                                For example, [3, 1, 0, 2] means neuron 3 fires first, then neuron 1,
                                then neuron 0, then neuron 2. So neuron 3 is now the first neuron in reordered_burst_matrices.
                                Use this to map back to original neuron IDs.
        """
        # burst_matrices is N x T x B
        slice_matrices = self.event_stack

        # This is a matrix (NxB) where row is neuron, and each column is a burst/slice/event. Value is the time index
        # firing rate peak for neuron N in burst B
        unit_max_indices_array = np.argmax(slice_matrices, axis=1)

        # This gives you a list of size N. Now you have median peak time for each neuron
        unit_median_indices = np.median(unit_max_indices_array, axis=1)

        # arr = [5,2,9,1] means neuron 0 max firing at time 5, neuron 1 max firing at time 2.
        # So np.argsort(arr) returns [3, 1, 0, 2] which means neuron 3 has max firing first, then neuron 1, etc

        unit_ids_in_order = np.argsort(unit_median_indices)
        # Reorder the neurons in orginal slice_matrices so that they are in temporal order
        reordered_slice_matrices = slice_matrices[unit_ids_in_order, :, :]
        # Returns the reordered bursts/slices/events, the unit ids in order so you can see which unit fires when
        return reordered_slice_matrices, unit_ids_in_order

    def get_slice_to_slice_unit_corr_from_stack(
        self,
        compare_func=compute_cross_correlation_with_lag,
        MIN_RATE_THRESHOLD=0.1,
        MIN_FRAC=0.3,
        max_lag=350,
    ):
        """
        Compute slice-to-slice (aka burst-to-burst) similarity along the 0th axis of RateSliceStack self.event_stack (N x T x B)
        This is done along the unit axis, making the output size be (N x B x B) and requiring thresholding to make
        units and slices that are inactive be Nan.


        Parameters:
        -----------
        compare_func (method in utils): Specify if you want to compare signals with correaltion or cosine similarity functions.
                                          The default is cross correlation. These functions can be insepcted further in utils.py
        MIN_RATE_THRESHOLD (float): Minimum mean firing rate to consider a slice/burst/event valid for that neuron
        MIN_FRAC (float): Maximum fraction of slice/burst/events that can be skipped before a unit is deemed invalid (default 0.3 = 30%)
        max_lag (int): Maximum lag in frames to search for similarity. If None, searches all possible lags.

        Returns:
        --------
        all_slice_corr_scores (array): Pairwise correlation scores between all slice/burst/event pairs for each unit shape (N, B, B)
        av_slice_corr_scores (array): Average correlation per neuron across all valid slice/burst/event pairs shape (N,)

        """
        # Get dimensions
        event_stack = self.event_stack
        num_units = event_stack.shape[0]  # N
        num_time_bins = event_stack.shape[1]  # T
        num_slices = event_stack.shape[2]  # B

        # Initialize result matrices
        av_slice_corr_scores = np.full(num_units, np.nan)
        all_slice_corr_scores = np.full((num_units, num_slices, num_slices), np.nan)

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

                for comp_b in range(num_slices):
                    # Comp vector
                    comp_rate = event_stack[unit, :, comp_b]

                    # Check if mean firing rate is above threshold
                    if np.mean(comp_rate) < MIN_RATE_THRESHOLD:
                        continue

                    # Compute similarity, we only want one output, not the lagged one.
                    max_corr, _ = compare_func(ref_rate, comp_rate, max_lag)

                    # Store results
                    all_slice_corr_scores[unit, comp_b, ref_b] = max_corr

            # If less than MIN_FRAC bursts were skipped
            if counter / num_slices <= MIN_FRAC:
                # Average results over all pairs
                av_slice_corr_scores[unit] = np.nanmean(
                    all_slice_corr_scores[unit, :, :]
                )
        # all_burst_corr_scores is NxBxB and av_burst_corr_scores is N since its the mean correlation across all bursts.
        # Keep as is, make helper function for converting BxB into below diagnol
        return all_slice_corr_scores, av_slice_corr_scores

    def get_slice_to_slice_time_corr_from_stack(
        self, compare_func=compute_cosine_similarity_with_lag, max_lag=0
    ):
        """
        Compute slice-to-slice (aka burst-to-burst) similarity along the 1st axis of RateSliceStack self.event_stack (N x T x B)
        This is done along the time axis, making the output size be (T x B x B) which doesn't require thresholding.

        Parameters:
        -----------
        compare_func (method in utils): Specify if you want to compare signals with correaltion or cosine similarity functions.
                                          The default is cosine similarity. These functions can be insepcted further in utils.py
        max_lag (int): Maximum lag in frames to search for similarity. If None, searches all possible lags.

        Returns:
        --------
        all_slice_corr_scores (array): Pairwise correlation scores between all slice/burst/event pairs for each time_bin shape (T, B, B)
        av_slice_corr_scores (array): Average correlation per time_bin across all valid slice/burst/event pairs shape (T,)

        """
        # Get dimensions
        event_stack = self.event_stack
        num_units = event_stack.shape[0]  # N
        num_time_bins = event_stack.shape[1]  # T
        num_slices = event_stack.shape[2]  # B

        # Initialize result matrices
        av_slice_corr_scores = np.full(num_time_bins, np.nan)
        all_slice_corr_scores = np.full((num_time_bins, num_slices, num_slices), np.nan)

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

            av_slice_corr_scores[time] = np.nanmean(all_slice_corr_scores[time, :, :])
        # all_slice_corr_scores is TxBxB and av_burst_corr_scores is N since its the mean correlation across all bursts.

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
        # lower triangle is B x F (or N x F if you input N x B x B) (or T x F if you input T x Bx B)
        lower_triangle = extract_lower_triangle_features(all_burst_corr_scores)
        pca_result = PCA_reduction(lower_triangle, n_components)

        return pca_result

    def convert_to_list_of_RateData(self):
        output = []
        # N x T x B
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
        Compute unit-to-unit similarity along the last axis of RateSliceStack self.event_stack (N x T x B)
        This is done along the slice axis, making the output size be (B x N x N) which doesn't require thresholding.

        Parameters:
        -----------
        compare_func (method in utils): Specify if you want to compare signals with correaltion or cosine similarity functions.
                                          The default is cosine similarity. These functions can be insepcted further in utils.py
        max_lag (int): Maximum lag in frames to search for similarity. If None, searches all possible lags.

        Returns:
        --------
        max_corr_array (array): Pairwise correlation scores between all unit pairs for each slice. Shape is (B, N, N)
        max_corr_lag_array (array): Lag where correlation between pair is at max. Shape is (B, N, N)
        av_max_corr (array): Average correlation per time_bin across all valid slice/burst/event pairs shape (B,)
        av_max_corr_lag (array): Average lag where correlation between pair is at max.

        """
        max_corr_stack = []
        max_corr_lag_stack = []
        rate_data_stack = self.convert_to_list_of_RateData()
        for i in range(len(rate_data_stack)):
            rate_data = rate_data_stack[i]
            # This gives 2 NxN matrices
            max_corr_matrix, lag_corr_matrix = rate_data.get_pairwise_fr_corr(
                compare_func, max_lag
            )
            max_corr_stack.append(max_corr_matrix)
            max_corr_lag_stack.append(lag_corr_matrix)
        # Make the list of correlation matrices into a 3d matrix
        max_corr_array = np.stack(max_corr_stack, axis=0)
        max_corr_lag_array = np.stack(max_corr_lag_stack, axis=0)
        # Find the averages to get a single dimension array of averages
        av_max_corr = np.nanmean(max_corr_array, axis=(1, 2))  # shape (B,)
        av_max_corr_lag = np.nanmean(max_corr_lag_array, axis=(1, 2))  # shape (B,)
        return max_corr_array, max_corr_lag_array, av_max_corr, av_max_corr_lag
