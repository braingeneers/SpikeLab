import warnings

import numpy as np

__all__ = ["RateData"]

from .pairwise import PairwiseCompMatrix
from concurrent.futures import ThreadPoolExecutor

from .utils import (
    compute_cross_correlation_with_lag,
    PCA_reduction,
    UMAP_reduction,
    UMAP_graph_communities,
    _get_attr,
    _resolve_n_jobs,
)


class RateData:
    """A 2D instantaneous firing rate matrix with unit-to-unit correlation capabilities.

    Parameters:
        inst_Frate_data (array): 2D array of shape (U, T). Each value is the
            instantaneous firing rate. U is the number of units/neurons and
            T is the number of time bins.
        times (list): List of time values that each column index in
            inst_Frate_data represents. For example, times = [5, 10, 15]
            so inst_Frate_data column 0 is 5 ms, column 1 is 10 ms, and
            column 2 is 15 ms.
        neuron_attributes (list or None): List of dicts, one per unit,
            containing arbitrary metadata about each neuron. None if not
            provided.

    Attributes:
        inst_Frate_data (array): 2D array of shape (U, T). Each value is the
            instantaneous firing rate. U is the number of units/neurons and
            T is the number of time bins.
        times (list): List of time values that each column index in
            inst_Frate_data represents. For example, times = [5, 10, 15]
            so inst_Frate_data column 0 is 5 ms, column 1 is 10 ms, and
            column 2 is 15 ms.
        neuron_attributes (list or None): List of dicts, one per unit,
            containing arbitrary metadata about each neuron. None if not
            provided.
        N (int): Number of units in inst_Frate_data.

    Notes:
        - ``times`` may contain negative values when the RateData represents an
          event-aligned window (e.g., times from -200 to +500 ms around a stimulus).
        - ``subtime`` always treats ``start``/``end`` as literal time values.
          Use ``subtime_by_index`` for index-based slicing with negative indexing.
    """

    def __init__(self, inst_Frate_data, times, neuron_attributes=None):
        if inst_Frate_data.ndim != 2:
            raise ValueError(
                f"rates must be a 2D array, got shape {inst_Frate_data.shape}"
            )

        if len(times) != inst_Frate_data.shape[1]:
            raise ValueError(
                "Number of columns in inst_Frate_data must be the same as length of times"
            )

        if not isinstance(times, np.ndarray):
            times = np.array(times)
        self.inst_Frate_data = inst_Frate_data
        self.times = times

        self.N = inst_Frate_data.shape[0]
        self.neuron_attributes = None
        if neuron_attributes is not None:
            self.neuron_attributes = neuron_attributes.copy()
            if len(neuron_attributes) != self.N:
                raise ValueError(
                    f"neuron_attributes has {len(neuron_attributes)} items "
                    f"but inst_Frate_data has {self.N} rows"
                )

    def __repr__(self) -> str:
        t0 = float(self.times[0]) if len(self.times) > 0 else 0.0
        t1 = float(self.times[-1]) if len(self.times) > 0 else 0.0
        return f"RateData(shape={self.inst_Frate_data.shape}, time_range=[{t0:.1f}, {t1:.1f}])"

    def subset(self, units, by=None):
        """Extract a subset of units/neurons from the rate data.

        Parameters:
            units (list or array): Unit indices to extract. If by is None,
                this should always be a list of ints. If by is not None,
                the list can contain ints or strings.
            by (str or None): Neuron attribute key to match against. Only
                use this if you initialized the object with
                neuron_attributes. Set to the key that contains neuron_id
                values. None selects by index (default).

        Returns:
            result (RateData): New RateData object containing only the
                specified units.
        """

        if isinstance(units, int):
            units = [units]
        # For case where user inputs a single string for units when using by option
        if isinstance(units, str):
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
                if _get_attr(self.neuron_attributes[i], by, _missing) in units
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

    def subtime(self, start, end):
        """Extract a subset of time points from the rate data using time values.

        Original time values are preserved in the output.

        Parameters:
            start (int or float): Starting time value (inclusive).
            end (int or float): Ending time value (exclusive).

        Returns:
            result (RateData): New RateData object containing only the
                specified time range.

        Notes:
            - Start and end are always treated as literal time values (not
              offsets from the end). To slice by array index with negative
              indexing support, use ``subtime_by_index(start_idx, end_idx)``.
        """

        # Handle start
        if start is None or start is Ellipsis:
            start = self.times[0] if len(self.times) > 0 else 0

        # Handle end — use a value just past the last time point so the
        # mask (times < end) includes the final bin.
        if end is None or end is Ellipsis:
            if len(self.times) > 1:
                end = self.times[-1] + (self.times[1] - self.times[0])
            elif len(self.times) == 1:
                end = self.times[-1] + 1
            else:
                end = 0

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
        return RateData(
            inst_Frate_data=output,
            times=new_times,
            neuron_attributes=self.neuron_attributes,
        )

    def subtime_by_index(self, start_idx, end_idx):
        """Extract a subset of time points from the rate data using time index values.

        Original time values are preserved in the output.

        Parameters:
            start_idx (int): Starting time index (inclusive).
            end_idx (int): Ending time index (exclusive).

        Returns:
            result (RateData): New RateData object containing only the
                specified time range.

        Notes:
            - Supports negative indexing (e.g., -5 selects 5 from the end).
            - To slice by time values instead of array indices, use
              ``subtime(start, end)``.
        """
        if start_idx < 0:
            start_idx += len(self.times)
        if end_idx < 0:
            end_idx += len(self.times)

        if start_idx < 0 or start_idx >= len(self.times):
            raise ValueError(f"start_idx {start_idx} out of range")
        if end_idx <= start_idx or end_idx > len(self.times):
            raise ValueError(f"end_idx {end_idx} invalid")

        output = self.inst_Frate_data[:, start_idx:end_idx]
        new_times = self.times[start_idx:end_idx]

        return RateData(
            inst_Frate_data=output,
            times=new_times,
            neuron_attributes=self.neuron_attributes,
        )

    def frames(self, length, overlap=0):
        """Split the rate data into a RateSliceStack of fixed-length windows.

        Parameters:
            length (float): Length of each window in milliseconds.
            overlap (float): Overlap between consecutive windows in milliseconds. Default 0.

        Returns:
            stack (RateSliceStack): Stack of rate data windows, one per frame.

        Notes:
            - Windows that would extend past the end of the recording are excluded.
            - overlap must be strictly less than length.
        """
        from .rateslicestack import RateSliceStack

        step = length - overlap
        if step <= 0:
            raise ValueError("overlap must be less than length")

        t0 = float(self.times[0])
        t_end = float(self.times[-1])
        step_size = float(self.times[1] - self.times[0]) if len(self.times) > 1 else 1.0

        upper = t_end - length + step_size + 1e-9
        times = [
            (float(start), float(start) + length)
            for start in np.arange(t0, upper, step)
        ]
        if not times:
            raise ValueError(
                f"Recording length ({t_end - t0 + step_size:.1f} ms) is shorter "
                f"than frame length ({length} ms)"
            )
        return RateSliceStack(self, times_start_to_end=times)

    def get_pairwise_fr_corr(
        self, compare_func=compute_cross_correlation_with_lag, max_lag=10, n_jobs=-1
    ):
        """Compute unit-to-unit similarity from the firing rate matrix (U, T).

        Parameters:
            compare_func (callable): Comparison function from utils. Specify
                cross-correlation or cosine similarity. The default is cross
                correlation. See utils.py for details.
            max_lag (int): Max number of lag steps around 0 to consider for
                finding the max correlation. If None, lag is set to 0.
            n_jobs (int): Number of threads for parallel computation. -1 uses
                all cores (default), 1 disables parallelism, None is serial.

        Returns:
            corr_matrix (PairwiseCompMatrix): Maximum correlation coefficients
                between all unit/neuron pairs. matrix[i, j] is the max
                correlation between unit i and unit j. Values range from -1
                to 1. Diagonal is always 1 (self-correlation).
            lag_matrix (PairwiseCompMatrix): Time lags (in time bins) at which
                maximum correlation occurs. lag_matrix[i, j] is the lag where
                correlation between i and j is maximal. Positive lag means
                unit j leads unit i (j fires earlier). Negative lag means
                unit i leads unit j (i fires earlier). Diagonal is always 0.
        """

        rate_matrix = self.inst_Frate_data

        num_units = self.inst_Frate_data.shape[0]  # N
        corr_matrix_this_event = np.full((num_units, num_units), np.nan)
        lag_matrix_this_event = np.full((num_units, num_units), np.nan)

        pairs = [(n1, n2) for n1 in range(num_units) for n2 in range(n1, num_units)]

        def _compute_pair(pair):
            n1, n2 = pair
            return pair, compare_func(
                rate_matrix[n1, :], rate_matrix[n2, :], max_lag=max_lag
            )

        n_workers = _resolve_n_jobs(n_jobs)
        if n_workers > 1 and len(pairs) > 1:
            print(f"get_pairwise_corr: using thread pool ({n_workers} workers)")
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = pool.map(_compute_pair, pairs)
        else:
            print("get_pairwise_corr: using serial backend")
            results = map(_compute_pair, pairs)

        for (n1, n2), (max_corr, max_lag_idx) in results:
            corr_matrix_this_event[n1, n2] = max_corr
            lag_matrix_this_event[n1, n2] = max_lag_idx
            corr_matrix_this_event[n2, n1] = max_corr
            lag_matrix_this_event[n2, n1] = -max_lag_idx

        # Output is UxU, wrapped in PairwiseCompMatrix for API consistency
        meta = {"compare_func": compare_func.__name__, "max_lag": max_lag}
        return (
            PairwiseCompMatrix(matrix=corr_matrix_this_event, metadata=meta),
            PairwiseCompMatrix(matrix=lag_matrix_this_event, metadata=meta),
        )

    def get_manifold(
        self,
        method: str = "PCA",
        n_components: int = 2,
        **kwargs,
    ):
        """Project the firing-rate data into a low-dimensional manifold using PCA or UMAP.

        Parameters:
            method (str): Which dimensionality reduction method to use.
                Either ``"PCA"`` (default) or ``"UMAP"``.
            n_components (int): Number of output dimensions to return
                (default 2).
            **kwargs: Additional options for UMAP. If method is ``"UMAP"``,
                you can specify use_graph_communities (bool), return_labels
                (bool), and other UMAP-specific keyword arguments such as
                n_neighbors, min_dist, metric, or resolution.

        Returns:
            result (tuple): Depends on method and options:
                If method is ``"PCA"``: ``(embedding, explained_variance_ratio,
                components)`` where embedding has shape (T, n_components),
                explained_variance_ratio has shape (n_components,), and
                components has shape (n_components, U).
                If method is ``"UMAP"``: ``(embedding, trustworthiness)`` where
                embedding has shape (T, n_components) and trustworthiness is
                a float from 0 to 1.
                If method is ``"UMAP"`` with use_graph_communities=True and
                return_labels=True: ``(embedding, labels, trustworthiness)``.

        Notes:
            - To visualise the resulting embedding, use
              :func:`~spikelab.spikedata.plot_utils.plot_manifold`. It
              accepts the embedding array directly and supports background
              masks, continuous colour values, and discrete group colouring.
        """
        # Shape is (U, T); treat each time bin as a sample.
        data_T = self.inst_Frate_data.T  # (T, U)

        method_upper = method.upper()
        if method_upper == "PCA":
            if kwargs:
                warnings.warn(
                    f"Additional keyword arguments {list(kwargs.keys())} are ignored for method='{method}'.",
                    UserWarning,
                )
            return PCA_reduction(
                data_T, n_components=n_components
            )  # (embedding, var_ratio, components)
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
                embedding, labels, tw = UMAP_graph_communities(
                    data_T,
                    n_components=n_components,
                    **kwargs,
                )
                if return_labels:
                    return embedding, labels, tw
                return embedding, tw

            # Default: plain UMAP embedding + trustworthiness.
            return UMAP_reduction(
                data_T,
                n_components=n_components,
                **kwargs,
            )  # (embedding, trustworthiness)

        raise ValueError(
            f"Unknown manifold method '{method}' (expected 'PCA' or 'UMAP')."
        )
