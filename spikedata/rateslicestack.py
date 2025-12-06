import SpikeData
import RateData
import numpy as np
from scipy import signal


from .utils import(
    compute_cross_correlation_with_lag,
    compute_cosine_similarity_with_lag,
    extract_lower_triangle_features,
    PCA_reduction
)






class RateSliceStack:
    # This is an object that contains a collection of sparse matrices in 3D matrix form
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

        # Actual constructor

        if isinstance(data_obj, SpikeData):
            all_times = np.arange(0, data_obj.length)
            inst_Frate_matrix = data_obj.resampled_isi(all_times, sigma_ms)
            data_obj = RateData(inst_Frate_matrix)

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
        # Converts to a 3d array
        event_stack = np.stack(event_stack, axis=2)
        # This makes event stack be N x T x B
        self.event_stack = event_stack

    def order_neurons_across_bursts(self):
        """
        Reorders the neurons in bursts from earliest to latest peak firing rate.

        Parameters:
            - No inputs. It uses the underlying 3D self.event_stack matrix that is NxTxB (neuron x time_bin x burst/event)
        Returns:
            - reordered_burst_matrices: This is 3D self.event_stack but the 0th dimension N is reordered temporally
                                        Now, the first neuron is the one that usually fires off across all bursts
            - neurons_ids_in_order: Array of original neuron indices sorted by their typical firing order.
                                    For example, [3, 1, 0, 2] means neuron 3 fires first, then neuron 1,
                                    then neuron 0, then neuron 2. So neuron 3 is now the first neuron in reordered_burst_matrices.
                                    Use this to map back to original neuron IDs.

        """
        # burst_matrices is N x T x B
        burst_matrices = self.event_stack

        # This is a matrix (NxB) where row is neuron, and each column is a burst. Value is the time index
        # firing rate peak for neuron N in burst B
        neuron_max_indices_array = np.argmax(burst_matrices, axis=1)

        # This gives you a list of size N. Now you have median peak time for each neuron
        neuron_median_indices = np.median(neuron_max_indices_array, axis=1)

        # arr = [5,2,9,1] means neuron 0 max firing at time 5, neuron 1 max firing at time 2.
        # So np.argsort(arr) returns [3, 1, 0, 2] which means neuron 3 has max firing first, then neuron 1, etc

        neurons_ids_in_order = np.argsort(neuron_median_indices)
        # Reorder the neurons in orginal burst_matrices so that they are in temporal order
        reordered_burst_matrices = burst_matrices[neurons_ids_in_order, :, :]
        # Returns the reordered bursts, the neuron ids in order so you can see which neuron fires when
        return reordered_burst_matrices, neurons_ids_in_order

    # def order_neurons_across_bursts(self):
    #     #burst_matrices is B x N x T
    #     burst_matrices = self.event_stack
    #     #This list will be size bursts x neuron_id
    #     neuron_max_indices_across_bursts = []
    #     for burst_index in range(burst_matrices.shape[0]):
    #         burst = burst_matrices[burst_index,:,:]
    #         # I NEED TO GO BACK AND DO TESTS IN IPYNB WITH SHAPES BEFORE PROCEEDING
    #         # This list will be size N (number of neurons)
    #         neuron_max_indices_per_burst = []
    #         for neuron_index in range(burst.shape[0]):
    #             neuron = burst[neuron_index,:]
    #             max_index = np.argmax(neuron)
    #             #
    #             neuron_max_indices_per_burst.append(max_index)

    #         neuron_max_indices_across_bursts.append(neuron_max_indices_per_burst)
    #         #This is BxN and each value is the time bin that has max firing rate

    #     neuron_max_indices_array = np.array(neuron_max_indices_across_bursts)
    #     #This gives you a list of size N
    #     neuron_median_indices = np.median(neuron_max_indices_array, axis=0)

    #     #arr = [5,2,9,1] means neuron 0 max firing at time 5, neuron 1 max firing at time 2.
    #     #So np.argsort(arr) returns [3, 1, 0, 2] which means neuron 3 has max firing first, then neuron 1, etc

    #     neurons_ids_in_order = np.argsort(neuron_median_indices)
    #     #Reorder the neurons in orginal burst_matrices so that they are in temporal order
    #     reordered_burst_matrices = burst_matrices[:,neurons_ids_in_order,:]
    #     #Returns the reordered bursts, the neuron ids in order so you can see which neuron fires when
    #     return reordered_burst_matrices, neurons_ids_in_order

   


    def get_slice_to_slice_neuron_corr_from_stack(self,compare_func =compute_cross_correlation_with_lag,  MIN_RATE_THRESHOLD=0.1, MIN_FRAC=0.3, max_lag = 350):
        """
        Compute burst-to-burst correlation using RateSliceStack object.

        Using Min_rate since this doesn't have t_spike matrix
        
        Parameters:
        -----------
        rate_slice_stack : RateSliceStack
            Your RateSliceStack object containing event_stack (N x T x B)
        compare_func:
            Specify if you want to compare signals with correaltion or cosine similarity functions.
        MIN_RATE_THRESHOLD : float
            Minimum mean firing rate to consider a burst valid for that neuron
        MIN_FRAC : float
            Maximum fraction of bursts that can be skipped (default 0.3 = 30%)
            
        Returns:
        --------
        all_burst_corr_scores : array, shape (N, B, B)
            Pairwise correlation scores between all burst pairs for each neuron
        av_burst_corr_scores : array, shape (N,)
            Average correlation per neuron across all valid burst pairs
        """
        # Get dimensions
        event_stack = self.event_stack
        num_neurons = event_stack.shape[0]  # N
        num_time_bins = event_stack.shape[1]  # T
        num_bursts = event_stack.shape[2]  # B
        
        # Initialize result matrices
        av_burst_corr_scores = np.full(num_neurons, np.nan)
        all_burst_corr_scores = np.full((num_neurons, num_bursts, num_bursts), np.nan)
        
        # For each neuron
        for neuron in range(num_neurons):
            # Make list of comparison bursts
            comp_bursts = list(range(num_bursts))
            
            # Counter for skipped bursts
            counter = 0
            
            # For each reference burst
            for ref_b in range(num_bursts):
                # Remove ref burst from comp bursts
                comp_bursts = [b for b in comp_bursts if b != ref_b]
                
                # Extract firing rate for this neuron in this burst
                # Shape: (T,) - 1D array of firing rates over time
                ref_rate = event_stack[neuron, :, ref_b]
                
                # Check if mean firing rate is above threshold
                if np.mean(ref_rate) < MIN_RATE_THRESHOLD:
                    counter += 1
                    continue
                
                # For each comparison burst
                for comp_b in comp_bursts:
                    # Extract firing rate for this neuron in comparison burst
                    comp_rate = event_stack[neuron, :, comp_b]
                    
                    # Check if mean firing rate is above threshold
                    if np.mean(comp_rate) < MIN_RATE_THRESHOLD:
                        continue
                    
                    # Compute cross correlation. Only use 10 ms since we are comparing bursts, and bursts
                    # are centered around 0 which is burst peak. That means we expect that burst 1 and burst 2 will have 
                    # max acvitivty near 0
                    max_corr, _ = compare_func(ref_rate, comp_rate, max_lag)
                    
                    # Store results
                    all_burst_corr_scores[neuron, comp_b, ref_b] = max_corr
            
            # If less than MIN_FRAC bursts were skipped
            if counter / num_bursts <= MIN_FRAC:
                # Average results over all pairs
                av_burst_corr_scores[neuron] = np.nanmean(all_burst_corr_scores[neuron, :, :])
        #all_burst_corr_scores is NxBxB and av_burst_corr_scores is N since its the mean correlation across all bursts.
        # Keep as is, make helper function for converting BxB into below diagnol
        return all_burst_corr_scores, av_burst_corr_scores
    
    def neuron_to_neuron_correlation(self, compare_func =compute_cross_correlation_with_lag, max_lag=350):
        event_stack = self.event_stack
        num_neurons = event_stack.shape[0]  # N
        num_time_bins = event_stack.shape[1]  # T
        num_bursts = event_stack.shape[2]  # B

        num_values_under_diagnol = num_neurons * (num_neurons - 1) // 2
        # output_max_corr = np.full((num_bursts, num_values_under_diagnol), np.nan)
        # output_max_lag = np.full((num_bursts, num_values_under_diagnol), np.nan)
        output_max_corr = np.full((num_bursts, num_neurons, num_neurons), np.nan)
        output_max_lag = np.full((num_bursts, num_neurons, num_neurons), np.nan)
        av_output_max_corr = np.full(num_bursts, np.nan)
        av_output_max_lag = np.full(num_bursts, np.nan)


        

        for b in range(num_bursts):
            corr_matrix_this_burst = np.full((num_neurons, num_neurons), np.nan)
            lag_matrix_this_burst = np.full((num_neurons, num_neurons), np.nan)

            rate_matrix = event_stack[:,:,b]
            for n1 in range(num_neurons):
                for n2 in range(num_neurons):
                    reference_signal = rate_matrix[n1,:]
                    compare_signal = rate_matrix[n2,:]
                    max_corr, max_lag_idx = compare_func(reference_signal, compare_signal, max_lag)

                    corr_matrix_this_burst[n1,n2] = max_corr
                    lag_matrix_this_burst[n1,n2] = max_lag_idx
                    
            # output_max_corr.append(corr_matrix_this_burst)
            # output_max_lag.append(lag_matrix_this_burst)

            output_max_corr[b, :] = corr_matrix_this_burst
            output_max_lag[b, :] = lag_matrix_this_burst

            # lower_tri_idx = np.tril_indices(num_neurons, k=-1)
            # output_max_corr[b, :] = corr_matrix_this_burst[lower_tri_idx]
            # output_max_lag[b, :] = lag_matrix_this_burst[lower_tri_idx]


        # With full matrix, it is B x N x N. But we want lower triangle of NxN correlation
        # Outputs are B x F where F is the number of values under the diagnol in matrix.
        # Each burst is a 1xF, represting the neuron correaltions in that one burst

        # We should make a function where you can input n1 and n2, and it will extract 
        # the correaltion from a 1xF input
            av_output_max_corr[b] = np.nanmean(output_max_corr[b, :, :])
            av_output_max_lag[b] = np.nanmean(output_max_lag[b, :, :])
            

        return output_max_corr, av_output_max_corr, output_max_lag, av_output_max_lag
    

    
    def get_slice_to_slice_time_corr_from_stack(self, compare_func = compute_cosine_similarity_with_lag, max_lag = 0):
        """
        Compute burst-to-burst cosine similarity using RateSliceStack object.
        At which moments during the burst are all bursts most similar to each other?

        Using Min_rate since this doesn't have t_spike matrix
        
        Parameters:
        -----------
        rate_slice_stack : RateSliceStack
            Your RateSliceStack object containing event_stack (N x T x B)
        compare_func:
            Specify if you want to compare signals with correaltion or cosine similarity functions.
            
        Returns:
        --------
        all_burst_corr_scores : array, shape (N, B, B)
            Pairwise correlation scores between all burst pairs for each neuron
        av_burst_corr_scores : array, shape (N,)
            Average correlation per neuron across all valid burst pairs
        """
        # Get dimensions
        event_stack = self.event_stack
        num_neurons = event_stack.shape[0]  # N
        num_time_bins = event_stack.shape[1]  # T
        num_bursts = event_stack.shape[2]  # B
        
        # Initialize result matrices
        av_burst_corr_scores = np.full(num_time_bins, np.nan)
        all_burst_corr_scores = np.full((num_time_bins, num_bursts, num_bursts), np.nan)
        
        # For each neuron
        for time in range(num_time_bins):
            # Make list of comparison bursts
            comp_bursts = list(range(num_bursts))
            
            # Counter for skipped bursts
            counter = 0
            
            # For each reference burst
            for ref_b in range(num_bursts):
                # Remove ref burst from comp bursts
                comp_bursts = [b for b in comp_bursts if b != ref_b]
                
                # Extract firing rate for this neuron in this burst
                # Shape: (T,) - 1D array of firing rates over time
                ref_rate = event_stack[:, time, ref_b]
                
                # # Check if mean firing rate is above threshold
                # if np.mean(ref_rate) < MIN_RATE_THRESHOLD:
                #     counter += 1
                #     continue
                
                # For each comparison burst
                for comp_b in comp_bursts:
                    # Extract firing rate for this neuron in comparison burst
                    comp_rate = event_stack[:, time, comp_b]
                    
                    # # Check if mean firing rate is above threshold
                    # if np.mean(comp_rate) < MIN_RATE_THRESHOLD:
                    #     continue
                    
                    # Compute cross correlation. Only use 10 ms since we are comparing bursts, and bursts
                    # are centered around 0 which is burst peak. That means we expect that burst 1 and burst 2 will have 
                    # max acvitivty near 0
                    max_corr, _ = compare_func(ref_rate, comp_rate, max_lag)
                    
                    # Store results
                    all_burst_corr_scores[time, comp_b, ref_b] = max_corr
            
           
            av_burst_corr_scores[time] = np.nanmean(all_burst_corr_scores[time,:, :])
        #all_burst_corr_scores is TxBxB and av_burst_corr_scores is N since its the mean correlation across all bursts.

        return all_burst_corr_scores, av_burst_corr_scores
    
    
        
    def PCA_on_lower_diagnol_corr_matrix(self,all_burst_corr_scores, n_components =2):
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
         #lower triangle is B x F (or N x F if you input N x B x B) (or T x F if you input T x Bx B)
        lower_triangle= extract_lower_triangle_features(all_burst_corr_scores)
        pca_result = PCA_reduction(lower_triangle, n_components)

        
        return pca_result
    
    
    



        

    # Make a equation for converting 3D matrix into list of rate datas 
    # Have rate data have actual nxn function
    # Make a class for nxn matrices. This will have lower triangle function as well
    # And PCa function
    # So then output of get_slice_to_slice_neuron_corr_from_stack would instead be a list
    # of this nxn object instead of a 3d matrix
                


