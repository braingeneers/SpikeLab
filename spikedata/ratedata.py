import warnings

import numpy as np
from scipy import signal

from .utils import (
    compute_cross_correlation_with_lag,
    PCA_reduction,
    UMAP_reduction,
    UMAP_graph_communities,
)


class RateData:
    """
    Description:
    -----------
    A data structure where the underlying data is a 2D instantneous firing rate matrix. This object allows the user
    to perform a set of functions upon this data, including unit to unit correlation matrix computations.

    Parameters:
    -----------
    inst_Frate_data (array): 2D array of shape (U, T). Each value is the instanteous firing rate.
        - U: number of  units/neurons
        - T: number of time bins
    times (list): List of time values that each column index in inst_Frate_data represents.
                  For example, times = [5,10,15] so inst_Frate_data column 0 is 5 ms, column
                  1 is 10 ms, and column 2 is 15 ms
    neuron_attributes: TBD

    Instance Variables:
    --------
    self.inst_Frate_data (array): 2D array of shape (U, T). Each value is the instanteous firing rate.
        - U: number of  units/neurons
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
        Extract a subset of units/neurons from the rate data. Index-based if by = None.

        Parameters:
        units (list or array): Unit indices to extract.
                               If by is not None, then units are the neuron_id you want to extract.
        by (string): This is None by default. Only use this if you initialized object with neuron_attributes dictionary.
                     If you have neuron_attributes, set variable "by" to be the key that contains neuron_id values.

        Returns:
        RateData: New RateData object containing only the specified units
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
        Extract a subset of time points from the rate data using time index values.

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
        Takes the object's underlying firing rate matrix (U, T) and computes the unit to unit similarity,
        and the similarity metric is set with compare_func.

        Parameters:
        compare_func (method in utils): Specify if you want to compare signals with cross-correlation or cosine similarity functions.
                                        The default is cross correlation. These functions can be insepcted further in utils.py
        max_lag (int): Max number of lag steps around 0 user wants to be considered for finding the max correlation.
                       If None, lag is set to 0.


        Returns:
        corr_matrix_this_event (array): Matrix of maximum correlation coefficients between all unit/neuron pairs.
                                          matrix[i, j] is the max correlation between unit i and unit j.
                                          Values range from -1 to 1. Diagonal is always 1 (self-correlation).
        lag_matrix_this_event (array): Matrix of time lags (in time bins) at which maximum correlation occurs.
                                         lag_matrix[i, j] is the lag where correlation between i and j is maximal.
                                         Positive lag means unit j leads unit i (j fires earlier).
                                         Negative lag means unit i leads unit j (i fires earlier).
                                         Diagonal is always 0 (self-correlation is perfectly aligned, so max corr at 0 lag.)
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

        # Output is UxU
        return corr_matrix_this_event, lag_matrix_this_event

    def get_manifold(
        self,
        method: str = "PCA",
        n_components: int = 2,
        **kwargs,
    ):
        """
        Project the firing-rate data into a low-dimensional manifold using PCA or UMAP.

        Parameters:
        method (str): Which dimensionality reduction method to use. Either "PCA" (default) or "UMAP".
        n_components (int): Number of output dimensions to return (default=2).
        **kwargs: Additional options for UMAP. If method="UMAP", you can specify:
            - use_graph_communities (bool): If True, use UMAP's connectivity graph with Louvain community detection (default: False).
            - return_labels (bool): If True and use_graph_communities is True, return (embedding, labels) tuple (default: False).
            - Other UMAP-specific keyword arguments such as n_neighbors, min_dist, metric, or resolution.

        Returns:
        embedding (ndarray): Low-dimensional embedding, shape (T, n_components), where T is the number of time bins.
            Each row corresponds to a time bin in self.times.
        (embedding, labels) (tuple): If method="UMAP", use_graph_communities=True, and return_labels=True,
            returns both the embedding and an array of integer community labels for each time bin.
        """
        # Shape is (U, T); treat each time bin as a sample.
        data_T = self.inst_Frate_data.T  # (T, U)

        method_upper = method.upper()
        if method_upper == "PCA":
            if kwargs:
                print(
                    f"Additional keyword arguments {list(kwargs.keys())} are ignored for method='{method}'."
                )   
            return PCA_reduction(data_T, n_components=n_components)
        if method_upper == "UMAP":
            # Optional graph-based UMAP + Louvain communities.
            use_graph_communities = kwargs.pop("use_graph_communities", False)
            return_labels = kwargs.pop("return_labels", False)

            if return_labels and not use_graph_communities:
                warnings.warn(
                    "return_labels=True has no effect without use_graph_communities=True; "
                    "labels will not be returned.",
                    UserWarning,
                    stacklevel=2,
                )

            if use_graph_communities:
                embedding, labels = UMAP_graph_communities(
                    data_T,
                    n_components=n_components,
                    **kwargs,
                )
                if return_labels:
                    return embedding, labels
                return embedding

            # Default: plain UMAP embedding only.
            return UMAP_reduction(
                data_T,
                n_components=n_components,
                **kwargs,
            )

        raise ValueError(
            f"Unknown manifold method '{method}' (expected 'PCA' or 'UMAP')."
        )
