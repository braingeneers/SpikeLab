import numpy as np
from scipy import signal

from .utils import(
    compute_cross_correlation_with_lag
)

class RateData:
    # It's like SpikeData but its underlying data is instaneous firing rates, not
    # sparse spike matrices.

    # neuron_ids is a list of neuron_ids that each row index in inst_Frate_data represents
    # neuron_ids = [2,7,8] so inst_Frate_data has 3 rows, and row 0 represents neuron_id 2

    # times is a list of time values that each column index in inst_Frate_data represents
    # times = [5,10,15] so inst_Frate_data column 0 is 5 ms, column 1 is 10 ms, and column 2 is 15 ms
    def __init__(
        self,
        inst_Frate_data,
        times,
        neuron_attributes = None,
        # subset_neurons = [],
        # subset_time_range = [],
        N=None
        #length=None
    ):
        if inst_Frate_data.ndim != 2:
            raise ValueError(
                f"rates must be a 2D array, got shape {self.inst_Frate_data.shape}"
            )
        
        if len(times) != inst_Frate_data.shape[1]:
            raise ValueError(
                "Number of columns in inst_Frate_data must be the same as length of times"
            )
        

        if any(x < 0 for x in times):
            raise ValueError("No negative values are allowed in times.")
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        
        # if neuron_ids is None:
        #     neuron_ids = np.arange(inst_Frate_data.shape[0])

        # if not isinstance(neuron_ids, np.ndarray):
        #     neuron_ids = np.array(neuron_ids)
        

        self.inst_Frate_data = inst_Frate_data
        # self.neuron_ids = neuron_ids
        self.times = times

        self.N = inst_Frate_data.shape[0]
        self.neuron_attributes = None
        if neuron_attributes:
            self.neuron_attributes = neuron_attributes.copy()
            if len(neuron_attributes) != self.N:
                raise ValueError(
                    f"neuron_attributes has {len(neuron_attributes)} items "
                    f"but inst_Frate_data has {self.N} rows"
                )

        # self.length = times[-1] if len(times) > 0 else 0

        # self.subset_neurons = range(self.N)
        # # Time is 0 indexed. So if someone says they want time 8, this refers to time 7-8.
        # self.subset_time_range = (times[0], times[-1])

    def subset(self, units, by = None):
        """
        Extract a subset of neurons from the rate data.

        Parameters:
            units: List or array of neuron indices to extract 
            by = "id" allows you to use and track neuron_attributes

        Returns:
            RateData: New RateData object containing only the specified neurons
        """
        # neuron_indices = []

        # for i in range(len(units)):
        #     curr_neuron_id = units[i]
        #     for j in range(len(self.neuron_ids)):
        #         if self.neuron_ids[j] == curr_neuron_id:
        #             neuron_indices.append(j)
        # if len(neuron_indices) == 0:
        #     raise ValueError("Input Neuron_ids do not exist for this RateData Object")

        if isinstance(units, int):
            units = [units]
        units = set(units)
        if by is not None:
            # VALUE-BASED: Look up by neuron_attribute
            if self.neuron_attributes is None:
                raise ValueError("can't use `by` without `neuron_attributes`")
            
            _missing = object()
            units = {
                i
                for i in range(self.N)
                if getattr(self.neuron_attributes[i], by, _missing) in units
            }
        units = sorted(units)

        # if max(units) >= self.inst_Frate_data.shape[0]:
        #     raise ValueError("Unit out of range")
        

        output = self.inst_Frate_data[units, :]
        neuron_attributes = None
        if self.neuron_attributes is not None:
            neuron_attributes = [self.neuron_attributes[i] for i in units]

       #selected_ids = self.neuron_ids[units] 
        # for neuron in units:
        #     neuron_firing_rate = self.inst_Frate_data[neuron,:]
        #     output.append(neuron_firing_rate)
        #     neuron_ids.append(neuron)
        return RateData(inst_Frate_data=output, 
                        times=self.times,
                        neuron_attributes=neuron_attributes)

    def subtime(self, start, end, shift_time=True):
        """
        Extract a subset of time points from the rate data.

        Parameters:
            start: Starting time index (inclusive)
            end: Ending time index (exclusive)

            start: Starting time  (inclusive)
            end: Ending time  (exclusive)

        Returns:
            RateData: New RateData object containing only the specified time range
        """

        # output = self.inst_Frate_data[:, start:end]
        # return RateData(
        #     inst_Frate_data=output,
        #     subset_neurons=self.subset_neurons,
        #     subset_time_range=(start, end),
        # )
        length = self.times[-1] if len(self.times) > 0 else 0

        if start is None or start is Ellipsis:
            start = self.times[0] if len(self.times) > 0 else 0
        elif start < 0:
            start += length
            if start < 0:
                raise ValueError(
                    f"start ({start - length}) is too negative. "
                    f"Minimum allowed is -{length}"
                )
        
        # Handle end
        if end is None or end is Ellipsis:
            end = length
        elif end < 0:
            end += length
            if end < 0:
                raise ValueError(
                    f"end ({end - length}) is too negative. "
                    f"Minimum allowed is -{length}"
                )
        
        # Validate
        if start >= end:
            raise ValueError(
                f"start ({start}) must be less than end ({end})"
            )

        mask = (self.times >= start) & (self.times < end)

        # Check if start and end were in range
        if not np.any(mask):
            raise ValueError(
                f"No time points found in range [{start}, {end}). "
                f"The available range is [{self.times[0]}, {self.times[-1]}]"
            )

        output = self.inst_Frate_data[:, mask]
        new_times = self.times[mask]
        if shift_time:
            new_times = new_times - new_times[0]
        return RateData(
            inst_Frate_data=output,
            times=new_times
        )
    
    def subtime_by_index(self, start_idx, end_idx, shift_time=True):
        """Extract time range by INDICES"""

        if start_idx < 0 or start_idx >= len(self.times):
            raise ValueError(f"start_idx {start_idx} out of range")
        if end_idx <= start_idx or end_idx > len(self.times):
            raise ValueError(f"end_idx {end_idx} invalid")
        
        output = self.inst_Frate_data[:, start_idx:end_idx]
        new_times = self.times[start_idx:end_idx]
        
        if shift_time:
            new_times = new_times - new_times[0]
        
        return RateData(
            inst_Frate_data=output,
            times=new_times
        )
    
    def neuron_pairwise_fr_corr(self):

        rate_matrix = self.inst_Frate_data
        
        num_neurons = self.inst_Frate_data.shape[0]  # N
        num_time_bins = self.inst_Frate_data.shape[1]  # T
        corr_matrix_this_event = np.full((num_neurons, num_neurons), np.nan)
        lag_matrix_this_event = np.full((num_neurons, num_neurons), np.nan)

            
        for n1 in range(num_neurons):
            for n2 in range(num_neurons):
                reference_signal = rate_matrix[n1,:]
                compare_signal = rate_matrix[n2,:]
                max_corr, max_lag_idx = compute_cross_correlation_with_lag(reference_signal, compare_signal, max_lag = 350)

                corr_matrix_this_event[n1,n2] = max_corr
                lag_matrix_this_event[n1,n2] = max_lag_idx
                
        
        #Output is NxN

        return corr_matrix_this_event, lag_matrix_this_event
    
   
