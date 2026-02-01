import pytest
import numpy as np
from FunctionalConnectivity import calculate_covariance, granger_causality

def test_calculate_covariance():
    # Test normal case
    data = np.random.rand(100, 5)
    covariance_matrix = calculate_covariance(data)
    assert covariance_matrix.shape == (5, 5)
    assert np.allclose(covariance_matrix, covariance_matrix.T)  # Symmetric

    # Edge cases
    data_empty = np.array([]).reshape(0, 5)
    with pytest.raises(ValueError):
        calculate_covariance(data_empty)

    data_nan = np.random.rand(100, 5)
    data_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        calculate_covariance(data_nan)

    data_large = np.random.rand(10000, 100)
    covariance_matrix_large = calculate_covariance(data_large)
    assert covariance_matrix_large.shape == (100, 100)


def test_granger_causality():
    # Test normal case
    data = np.random.rand(100, 5)
    gc_matrix = granger_causality(data, max_lag=5)
    assert gc_matrix.shape == (5, 5)

    # Edge cases
    data_empty = np.array([]).reshape(0, 5)
    with pytest.raises(ValueError):
        granger_causality(data_empty, max_lag=5)

    data_nan = np.random.rand(100, 5)
    data_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        granger_causality(data_nan, max_lag=5)

    data_large = np.random.rand(10000, 100)
    gc_matrix_large = granger_causality(data_large, max_lag=5)
    assert gc_matrix_large.shape == (100, 100)