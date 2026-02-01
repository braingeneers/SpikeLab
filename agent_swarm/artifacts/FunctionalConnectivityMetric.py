# FunctionalConnectivityMetric.py

import numpy as np
from scipy.signal import correlate
from typing import List

class SpikeData:
    def __init__(self, spikes: List[np.ndarray]):
        self.spikes = spikes

    def calculate_functional_connectivity(self) -> np.ndarray:
        """
        Calculate the functional connectivity matrix based on spike timing correlation.

        Returns:
            np.ndarray: A 2D array representing the functional connectivity matrix.
        """
        num_neurons = len(self.spikes)
        connectivity_matrix = np.zeros((num_neurons, num_neurons))

        for i in range(num_neurons):
            for j in range(i + 1, num_neurons):
                # Compute the cross-correlation between the spike trains of neuron i and neuron j
                correlation = correlate(self.spikes[i], self.spikes[j], mode='full')
                max_correlation = np.max(np.abs(correlation))
                
                # Populate the symmetric connectivity matrix
                connectivity_matrix[i, j] = max_correlation
                connectivity_matrix[j, i] = max_correlation

        return connectivity_matrix

# Example usage
if __name__ == "__main__":
    # Hypothetical spike data for three neurons
    spike_data = SpikeData([
        np.random.rand(100),  # Neuron 1
        np.random.rand(100),  # Neuron 2
        np.random.rand(100),  # Neuron 3
    ])
    connectivity_matrix = spike_data.calculate_functional_connectivity()
    print("Functional Connectivity Matrix:\n", connectivity_matrix)