import SpikeData
import RateData
import numpy as np

class SliceStack:
    # times: List of tuples. Each tuple is (start, end) and represents the start and end times
        #of a burst
    # slice_or_rate_obj: Either SpikeData or RateData object. Constructor handles the rest
    # sigma_ms: Smoothing factor for computing isi if you input a SpikeData object
    # step_size: How many time_steps per unit? 
    #   For example, (start,end) = (0,10) could be 10 timestemps if step_size =1, or 2 timesteps if step_size = 5

    def __init__(
            self,
            times,
            data_obj,
            sigma_ms = 10,
            step_size= 1,

    ):
        #Checking if inputs are correct types
        if not isinstance(times, list):
              raise TypeError("times must be a list of tuples")
        for i, time_window in enumerate(times):
            if not isinstance(time_window, tuple):
                raise TypeError(f"Element {i} of times is not a tuple: {time_window}")
            if len(time_window) != 2:
                raise TypeError(f"Element {i} of times must be a tuple of length 2 (start, end): {time_window}")
            if not (isinstance(time_window[0], (int, float)) and isinstance(time_window[1], (int, float))):
                raise TypeError(f"Start and end times in element {i} must be numbers: {time_window}")
            if time_window[0] >= time_window[1]:
                raise ValueError(f"Start time must be less than end time in element {i}: {time_window}")
            
        if not isinstance(data_obj,(SpikeData,RateData)):
            #CHECK IF ISINSTANCE WORKS WITH THESE TWO CLASSES
            raise TypeError("data_obj must either be a SpikeData object or RateData object")
        #Actual constructor
        self.times = times
        burst_stack = []
        if isinstance(data_obj, RateData):
            #I use subtime here to extract a burst event
            for time in times:
                start = time[0]
                end = time[1]
                rate_obj_slice = data_obj.subtime(start, end)
                burst_matrix = rate_obj_slice.inst_Frate_data
                burst_stack.append(burst_matrix)

        if isinstance(data_obj, SpikeData):
            # Here, I may be mistaken but I didn't see a need to do subtime because
            #I can specify times directly from resampled isi and extract burst events.
            for time in times:
                start = time[0]
                end = time[1]
                all_times = np.arange(start, end, step_size)
                #Look into times
                burst_matrix = data_obj.resampled_isi(all_times, sigma_ms)
                burst_stack.append(burst_matrix)
        #Converts to a 3d array
        burst_stack = np.stack(burst_matrix)
        self.burst_stack = burst_matrix


        