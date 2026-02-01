from typing import Tuple
import numpy as np
from scipy import linalg

class FunctionalConnectivity:
    def __init__(self, time_series: np.ndarray):
        """
        Initialize the FunctionalConnectivity object with time series data.

        :param time_series: A 2D numpy array of time series data, where rows correspond to time points and columns correspond to different signals.
        """
        self.time_series = time_series

    def calculate_covariance(self) -> np.ndarray:
        """
        Calculate the covariance matrix of the time series.

        :return: Covariance matrix of the input time series.
        """
        return np.cov(self.time_series, rowvar=False)

    def granger_causality(self, max_lag: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Granger causality between the time series.

        :param max_lag: The maximum lag to consider for the model.
        :return: A tuple containing the F-statistic values and corresponding p-values in numpy arrays.
        """
        n, m = self.time_series.shape
        F_vals = np.zeros((m, m, max_lag))
        p_vals = np.zeros((m, m, max_lag))

        # Augment the time series with its lags up to max_lag
        X_lagged = np.zeros((n - max_lag, max_lag * m))
        for k in range(1, max_lag + 1):
            X_lagged[:, (k-1)*m:k*m] = self.time_series[max_lag-k:n-k, :]

        for i in range(m):
            for j in range(m):
                if i != j:
                    # Construct the model with/without the other time series
                    Y = self.time_series[max_lag:, i]
                    X_full = np.hstack([X_lagged, self.time_series[max_lag:, j:j+1]])
                    X_restricted = X_lagged

                    beta_full, _, _, _ = linalg.lstsq(X_full, Y)
                    rss_full = np.sum((Y - X_full @ beta_full)**2)

                    beta_restricted, _, _, _ = linalg.lstsq(X_restricted, Y)
                    rss_restricted = np.sum((Y - X_restricted @ beta_restricted)**2)

                    # Calculate F-statistic
                    num_params_excluded = X_full.shape[1] - X_restricted.shape[1]
                    F = ((rss_restricted - rss_full) / num_params_excluded) /
                        (rss_full / (Y.size - X_full.shape[1]))
                    F_vals[i, j, k - 1] = F

                    # Placeholder for p-values, requires additional statistical distribution

        return F_vals, p_vals