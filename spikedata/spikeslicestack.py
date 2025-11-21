import SpikeData
import RateData
import numpy as np


class SpikeSliceStack:
    # This is an object that contains a collection of spike objects as a list
    # There are a few options of input time formats:
    # Option 1:
        # times_start_to_end: List of tuples. Each tuple is (start, end) and represents the start and end times
        # of a burst/stimulation event. Each tuple must have same duration

    # Option 2 (Note both of the following must be input to work):
        # time_peaks: List of times where there is a burst peak or stimulation event. This variable must be paired
        #      with time bounds
        # time_bounds: Single tuple (left_bound, right_bound).
        #       If you put (250,500), then this means 250 ms before peak and 500 ms after peak.

    # slice_or_rate_obj: Either SpikeData or RateData object. Constructor handles the rest
    # sigma_ms: Smoothing factor for computing isi if you input a SpikeData object
    # step_size: How many time_steps per unit?
    #   For example, (start,end) = (0,10) could be 10 timestemps if step_size =1, or 2 timesteps if step_size = 5

    def __init__(
        self,
        data_obj,
        times_start_to_end=None,
        time_peaks=None,
        time_bounds=None,
        sigma_ms=10,
        step_size=1,
    ):
        if not isinstance(data_obj, (SpikeData)):
            # CHECK IF ISINSTANCE WORKS WITH THESE TWO CLASSES
            raise TypeError(
                "data_obj must be a SpikeData object"
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

            # Check if any times are beyond the end of the recording
            # if time_peaks is not None:
            #     if time_window[1] > time_peaks[-1]:
            #         continue

            time_diff_check.append(time_window[1] - time_window[0])
            # We only want to address time windows that are above 0 (recording start) and below recording end
            valid_time_tuples.append(time_window)

        if len(set(time_diff_check)) > 1:
            raise ValueError("All time windows must have the same length")

        times_start_to_end = valid_time_tuples

        # Actual constructor this is where spikeslicestack should change
        self.times = times_start_to_end
        event_stack = []
        
        # I use subtime here to extract a burst event
        for time in times_start_to_end:
            start = time[0]
            end = time[1]
            spike_obj_slice = data_obj.subtime(start, end)
            event_matrix = spike_obj_slice.sparse_raster(bin_size = 1)
            event_stack.append(spike_obj_slice)

        # if isinstance(data_obj, SpikeData):
        #     # Here, I may be mistaken but I didn't see a need to do subtime because
        #     #I can specify times directly from resampled isi and extract burst events.
        #     for time in times:
        #         start = time[0]
        #         end = time[1]
        #         all_times = np.arange(start, end, step_size)
        #         #Look into times
        #         burst_matrix = data_obj.resampled_isi(all_times, sigma_ms)
        #         burst_stack.append(burst_matrix)
        
        #event_stack = np.stack(event_stack)
        self.spike_stack = event_stack

    def to_sparse_matrices(self):
        """
        Transforms the list of spike objects from self.spike_stack into a 3D sparse spike matrices of NxTxB (neuron x time_bin x burst/event)
        

        Parameters:
            - No input: It uses the underlying self.spike_stack.
        Returns:
            - sparse_stack: 3D sparse spike matrix of size NxTxB where each value is 
                            a 1 or 0 if there is a spike in that for neuron in a time_bin 
                            at a given burst.

        """
        sparse_list =[]
        for i in len(self.spike_stack):
            spike_obj_slice = self.spike_stack[i]
            event_matrix = spike_obj_slice.sparse_raster(bin_size = 1)

            sparse_list.append(event_matrix)
        sparse_stack = np.stack(sparse_list,axis=2)
        # Make sparse stack into N x T x B
        return sparse_stack
        
        
