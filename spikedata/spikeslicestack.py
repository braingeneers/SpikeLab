import numpy as np

from .spikedata import SpikeData


class SpikeSliceStack:
    """
    Description:
    -----------
    A data structure where the underlying data is a list of SpikeData objects, and each SpikeData
    object represents one slice. User inputs a single SpikeData object and specifies the slice times
    so they can split this one large SpikeData into a list of smaller ones.

    Parameters:
    -----------
    data_obj (SpikeData): A SpikeData object.
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

    Instance Variables:
    --------
    self.spike_stack (list): List of SpikeData objects, and each entry represents one slice.
    self.times (list of tuples): List of (start, end) time bounds for each slice, sorted chronologically. Length equals S (number of slices).
                                 Example: [(100, 200), (500, 600), (1000, 1100)]


    """

    def __init__(
        self, data_obj, times_start_to_end=None, time_peaks=None, time_bounds=None
    ):
        # Ensure SpikeData is used
        if not isinstance(data_obj, SpikeData):
            raise TypeError("data_obj must be a SpikeData object")

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

        self.times = times_start_to_end
        event_stack = []

        # Use subtime here to extract a burst event
        for time in times_start_to_end:
            start = time[0]
            end = time[1]
            spike_obj_slice = data_obj.subtime(start, end)

            event_stack.append(spike_obj_slice)

        self.spike_stack = event_stack

    def to_raster_array(self):
        """
        Convert the spike stack into a 3D raster array of shape (U, T, S).

        Each slice is rasterized with 1 ms bins, producing a spike count matrix
        where entry (u, t, s) is the number of spikes unit u fired in time bin t
        of slice s.

        Returns:
            raster_stack (np.ndarray): 3D array of shape (U, T, S) with
                non-negative integer spike counts.
        """
        dense_list = [
            spike_slice.sparse_raster(bin_size=1).toarray()
            for spike_slice in self.spike_stack
        ]
        return np.stack(dense_list, axis=2)
