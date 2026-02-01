import numpy as np
dfrom scipy.signal import correlate

def compute_functional_connectivity(spike_trains: np.ndarray) -> np.ndarray:
    """
    Computes the functional connectivity metric based on spike timing correlations.
    
    :param spike_trains: 2D numpy array with shape (n_neurons, n_timepoints) where
                         each entry is 1 if the neuron fired at that timepoint and 0 otherwise.
    :return: 2D numpy array of shape (n_neurons, n_neurons) representing the strength of connections.
    """
    n_neurons = spike_trains.shape[0]
    connectivity_matrix = np.zeros((n_neurons, n_neurons))

    for i in range(n_neurons):
        for j in range(n_neurons):
            if i != j:
                connectivity_matrix[i, j] = np.corrcoef(spike_trains[i], spike_trains[j])[0, 1]
                
    return connectivity_matrix
