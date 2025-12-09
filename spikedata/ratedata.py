import numpy as np
from scipy import signal

from .utils import compute_cross_correlation_with_lag


class RateData:
    """
    Parameters:
    -----------
    inst_Frate_data (array): 2D array of shape (N, T). Each value is the instanteous firing rate.
        - N: number of  units/neurons
        - T: number of time bins
    times (list): List of time values that each column index in inst_Frate_data represents.
                  For example, times = [5,10,15] so inst_Frate_data column 0 is 5 ms, column
                  1 is 10 ms, and column 2 is 15 ms
    neuron_attributes: TBD

    Instance Variables:
    --------
    self.inst_Frate_data (array): 2D array of shape (N, T). Each value is the instanteous firing rate.
        - N: number of  units/neurons
        - T: number of time bins
    self.times (list): List of time values that each column index in inst_Frate_data represents.
                  For example, times = [5,10,15] so inst_Frate_data column 0 is 5 ms, column
                  1 is 10 ms, and column 2 is 15 ms
    self.neuron_attributes: TBD
    self.N (int): Number of units in self.inst_Frate_data
    """

    def __init__(self, inst_Frate_data, times, neuron_attributes=None, N=None):
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
        self.inst_Frate_data = inst_Frate_data
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

    def subset(self, units, by=None):
        """
        Extract a subset of neurons from the rate data.

        Parameters:
        units (list or array): Neuron indices to extract
        by : "id" allows you to use and track neuron_attributes

        Returns:
        RateData: New RateData object containing only the specified neurons
        """

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

        output = self.inst_Frate_data[units, :]
        neuron_attributes = None
        if self.neuron_attributes is not None:
            neuron_attributes = [self.neuron_attributes[i] for i in units]

        return RateData(
            inst_Frate_data=output,
            times=self.times,
            neuron_attributes=neuron_attributes,
        )

    def subtime(self, start, end, shift_time=True):
        """
        Extract a subset of time points from the rate data using time values.

        Parameters:
        start (int/float): Starting time value (inclusive)
        end (int/flot): Ending time value (exclusive)
        shift_time (bool): If true, this will make the new output rate object where self.times[0] = 0 (starting time is 0)
                      If false, this will make the new output rate object where self.times[0] = start (starting time is start input)

        Returns:
        RateData: New RateData object containing only the specified time range
        """

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
            raise ValueError(f"start ({start}) must be less than end ({end})")

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
            times=new_times,
            neuron_attributes=self.neuron_attributes,
        )

    def subtime_by_index(self, start_idx, end_idx, shift_time=True):
        """
        Extract a subset of time points from the rate data using time values.

        Parameters:
        start (int): Starting time index (inclusive)
        end (int): Ending time index (exclusive)
        shift_time (bool): If true, this will make the new output rate object where self.times[0] = 0 (starting time is 0)
                      If false, this will make the new output rate object where self.times[0] = start (starting time is start input)

        Returns:
        RateData: New RateData object containing only the specified time range
        """

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
            times=new_times,
            neuron_attributes=self.neuron_attributes,
        )

    def get_pairwise_fr_corr(
        self, compare_func=compute_cross_correlation_with_lag, max_lag=10
    ):
        """
        Takes the object's underlying firing rate matrix (N, T) and computes the unit to unit correlation

        Parameters:
        max_lag (int): Max number of lag steps around 0 user wants to be considered for finding the max correlation.

        Returns:
        corr_matrix_this_event (array): Matrix of maximum correlation coefficients between all neuron pairs.
                                          matrix[i, j] is the max correlation between neuron i and neuron j.
                                          Values range from -1 to 1. Diagonal is always 1 (self-correlation).
        lag_matrix_this_event (array): Matrix of time lags (in time bins) at which maximum correlation occurs.
                                         lag_matrix[i, j] is the lag where correlation between i and j is maximal.
                                         Positive lag means neuron j leads neuron i (j fires earlier).
                                         Negative lag means neuron i leads neuron j (i fires earlier).
        """

        rate_matrix = self.inst_Frate_data

        num_units = self.inst_Frate_data.shape[0]  # N
        num_time_bins = self.inst_Frate_data.shape[1]  # T
        corr_matrix_this_event = np.full((num_units, num_units), np.nan)
        lag_matrix_this_event = np.full((num_units, num_units), np.nan)

        for n1 in range(num_units):
            for n2 in range(n1, num_units):
                reference_signal = rate_matrix[n1, :]
                compare_signal = rate_matrix[n2, :]
                max_corr, max_lag_idx = compare_func(
                    reference_signal, compare_signal, max_lag=max_lag
                )

                corr_matrix_this_event[n1, n2] = max_corr
                lag_matrix_this_event[n1, n2] = max_lag_idx

                corr_matrix_this_event[n2, n1] = max_corr
                lag_matrix_this_event[n2, n1] = -max_lag_idx

        # Output is NxN

        return corr_matrix_this_event, lag_matrix_this_event
