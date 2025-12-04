import numpy as np
from scipy import signal

def compute_cross_correlation(ref_rate, comp_rate):
        """
        Compute normalized cross correlation between two firing rate signals.
        
        Parameters:
        -----------
        ref_rate : array
            Reference firing rate signal (1D)
        comp_rate : array
            Comparison firing rate signal (1D)
            
        Returns:
        --------
        max_corr : float
            Maximum correlation coefficient
        """
        # compute cross correlation
        r = signal.correlate(ref_rate, comp_rate, mode='same') / np.sqrt(
            signal.correlate(ref_rate, ref_rate, mode='same')[int(len(ref_rate) / 2)] *
            signal.correlate(comp_rate, comp_rate, mode='same')[int(len(comp_rate) / 2)])

        # obtain maximum correlation
        max_corr = np.max(r)
        
        return max_corr
def compute_cross_correlation_with_lag(ref_rate, comp_rate, max_lag=350):
        """
        Compute normalized cross correlation with lag information.
        
        This is the SAME correlation computation from get_burst_to_burst_corr,
        but enhanced to also return lag information.
        
        Parameters:
        -----------
        ref_rate : array
            Reference firing rate signal
        comp_rate : array
            Comparison firing rate signal
        max_lag : int, optional
            Maximum lag in frames to search for correlation.
            If None, searches all possible lags.
            
        Returns:
        --------
        r : array
            Full cross-correlation function at all lags
        max_corr : float
            Maximum correlation coefficient
        max_lag_idx : int
            Lag (in frames) at which maximum correlation occurs
        """
        # THIS IS THE EXACT SAME CORRELATION COMPUTATION FROM YOUR BOSS'S CODE
        r = signal.correlate(ref_rate, comp_rate, mode='same') / np.sqrt(
            signal.correlate(ref_rate, ref_rate, mode='same')[int(len(ref_rate) / 2)] *
            signal.correlate(comp_rate, comp_rate, mode='same')[int(len(comp_rate) / 2)])
        
        center = int(len(r) / 2)
        
        # ADDED: Restrict search to max_lag if specified
        if max_lag is not None:
            search_start = max(0, center - max_lag)
            search_end = min(len(r), center + max_lag + 1)
            search_window = r[search_start:search_end]
            
            max_corr = np.max(search_window)
            max_lag_idx = np.argmax(search_window) + search_start - center
        else:
            # ORIGINAL: Just get max from entire array
            max_corr = np.max(r)
            max_lag_idx = np.argmax(r) - center
        
        return r, max_corr, max_lag_idx


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
        neuron_ids = None,
        # subset_neurons = [],
        # subset_time_range = [],
        N=None,
        # length=None
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
        
        if neuron_ids is None:
            neuron_ids = np.arange(inst_Frate_data.shape[0])

        if not isinstance(neuron_ids, np.ndarray):
            neuron_ids = np.array(neuron_ids)

        self.inst_Frate_data = inst_Frate_data
        self.neuron_ids = neuron_ids
        self.times = times

        self.N = inst_Frate_data.shape[0]

        # self.length = times[-1] if len(times) > 0 else 0

        # self.subset_neurons = range(self.N)
        # # Time is 0 indexed. So if someone says they want time 8, this refers to time 7-8.
        # self.subset_time_range = (times[0], times[-1])

    def subset(self, units):
        """
        Extract a subset of neurons from the rate data.

        Parameters:
            units: List or array of neuron indices to extract (old)
            units: List or array of neuron_ids to extract

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
        
        if max(units) >= self.inst_Frate_data.shape[0]:
            raise ValueError("Unit out of range")
        

        output = self.inst_Frate_data[units, :]
        selected_ids = self.neuron_ids[units] 
        # for neuron in units:
        #     neuron_firing_rate = self.inst_Frate_data[neuron,:]
        #     output.append(neuron_firing_rate)
        #     neuron_ids.append(neuron)
        return RateData(inst_Frate_data=output, 
                        times=self.times,
                        neuron_ids=selected_ids)

    def subtime(self, start, end):
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

        mask = (self.times >= start) & (self.times < end)

        # Check if start and end were in range
        if not np.any(mask):
            raise ValueError(
                f"No time points found in range [{start}, {end}). "
                f"The available range is [{self.times[0]}, {self.times[-1]}]"
            )

        output = self.inst_Frate_data[:, mask]
        new_times = self.times[mask]
        return RateData(
            inst_Frate_data=output,
            times=new_times
        )
    def neuron_to_neuron_correlation_in_one_burst(self):

        rate_matrix = self.inst_Frate_data
        
        num_neurons = self.inst_Frate_data.shape[0]  # N
        num_time_bins = self.inst_Frate_data.shape[1]  # T
        corr_matrix_this_burst = np.full((num_neurons, num_neurons), np.nan)
        lag_matrix_this_burst = np.full((num_neurons, num_neurons), np.nan)

            
        for n1 in range(num_neurons):
            for n2 in range(num_neurons):
                reference_signal = rate_matrix[n1,:]
                compare_signal = rate_matrix[n2,:]
                _, max_corr, max_lag_idx = compute_cross_correlation_with_lag(reference_signal, compare_signal)

                corr_matrix_this_burst[n1,n2] = max_corr
                lag_matrix_this_burst[n1,n2] = max_lag_idx
                
        
        #Output is NxN

        return corr_matrix_this_burst, lag_matrix_this_burst
    
   
