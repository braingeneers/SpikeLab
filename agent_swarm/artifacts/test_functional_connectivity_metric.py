import pytest
import numpy as np
from functional_connectivity_metric import compute_connectivity


def test_basic_connectivity():
    # Test on simple spike timing arrays
    spikes_A = np.array([0, 0, 1, 1, 0, 0])
    spikes_B = np.array([0, 1, 1, 0, 0, 0])
    expected_connectivity = 0.5  # Define expected value based on your algorithm
    assert compute_connectivity(spikes_A, spikes_B) == pytest.approx(expected_connectivity)


def test_identical_spike_trains():
    # Two identical spike trains should have max connectivity
    spikes = np.array([0, 1, 0, 1])
    assert compute_connectivity(spikes, spikes) == pytest.approx(1.0)


def test_opposite_spike_trains():
    # Perfectly opposite spike trains
    spikes_A = np.array([0, 1, 0, 1])
    spikes_B = np.array([1, 0, 1, 0])
    assert compute_connectivity(spikes_A, spikes_B) == pytest.approx(0.0)


def test_handling_of_nan_values():
    # Should be able to handle NaN values without crashing
    spikes_A = np.array([0, np.nan, 1, 1, 0, 1])
    spikes_B = np.array([np.nan, 1, 1, 0, 0, 0])
    connectivity = compute_connectivity(spikes_A, spikes_B)
    assert not np.isnan(connectivity)


def test_empty_spike_trains():
    # Empty arrays should not cause a crash and return zero connectivity
    spikes_A = np.array([])
    spikes_B = np.array([])
    connectivity = compute_connectivity(spikes_A, spikes_B)
    assert connectivity == 0.0
