# functional_connectivity_metric.py

from typing import Tuple, List
import numpy as np
import torch


def compute_functional_connectivity(spike_times: np.ndarray, 
                                   neuron_pairs: List[Tuple[int, int]]) -> np.ndarray:
    """
    Compute the functional connectivity metric based on spike timing correlations.

    Parameters:
    - spike_times: A 2D numpy array where each row represents spike times for a neuron.
    - neuron_pairs: A list of tuples where each tuple contains indices of neuron pairs.

    Returns:
    - A 1D numpy array containing the computed connectivity values for each neuron pair.
    """
    num_pairs = len(neuron_pairs)
    connectivity_values = np.zeros(num_pairs)
    
    for i, (neuron1, neuron2) in enumerate(neuron_pairs):
        # Extract spike trains for the neuron pair
        spikes_neuron1 = spike_times[neuron1]
        spikes_neuron2 = spike_times[neuron2]
        
        # Compute cross-correlation
        cross_corr = np.correlate(spikes_neuron1, spikes_neuron2, mode='full')
        
        # Compute the metric (e.g., peak of cross-correlation)
        connectivity_values[i] = np.max(cross_corr)
        
    return connectivity_values

# Example usage:
# spike_times_example = np.random.rand(10, 100)  # Example spike_times for 10 neurons with 100 spikes each
# neuron_pairs_example = [(0, 1), (0, 2), (1, 2)]  # Example list of neuron pairs
# connectivity = compute_functional_connectivity(spike_times_example, neuron_pairs_example)
# print(connectivity)