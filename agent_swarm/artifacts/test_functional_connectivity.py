import pytest
import numpy as np
from FunctionalConnectivityMetric import SpikeData

# Test cases for the SpikeData class

def test_spikedata_basic_functionality():
    # Basic functionality with small example
    spikes = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1]
    ])
    spike_data = SpikeData(spikes)
    result = spike_data.compute_connectivity()
    expected = np.array([
        [1, -0.5, 0.5],
        [-0.5, 1, 0],
        [0.5, 0, 1]
    ])
    assert np.allclose(result, expected), "Basic functionality test failed."

def test_spikedata_empty_input():
    # Edge case: empty input
    spikes = np.array([[]])
    spike_data = SpikeData(spikes)
    result = spike_data.compute_connectivity()
    expected = np.array([[]])
    assert np.array_equal(result, expected), "Empty input test failed."

def test_spikedata_nan_values():
    # Edge case: input with NaN values
    spikes = np.array([
        [np.nan, 1, 0, 0, 1],
        [1, 0, np.nan, 0, 0],
        [0, 0, 0, 1, np.nan]
    ])
    spike_data = SpikeData(spikes)
    result = spike_data.compute_connectivity()
    expected = np.nan * np.empty((3, 3))
    assert np.all(np.isnan(result) == np.isnan(expected)), "NaN values test failed."

def test_spikedata_large_scale():
    # Edge case: Large scale data
    np.random.seed(42)
    spikes = np.random.randint(0, 2, size=(1000, 1000))
    spike_data = SpikeData(spikes)
    result = spike_data.compute_connectivity()
    assert result.shape == (1000, 1000), "Large scale test failed due to incorrect shape."
    assert np.all(np.diag(result) == 1), "Large scale test failed due to incorrect diagonal values."
