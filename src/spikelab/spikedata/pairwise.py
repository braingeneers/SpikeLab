from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Iterator
import numpy as np


@dataclass
class PairwiseCompMatrix:
    """A data class for n x n pairwise comparison matrices (e.g., correlation, STTC).

    Attributes:
        matrix (np.ndarray): The n x n comparison matrix.
        labels (list or None): Labels for the rows/columns (e.g., unit IDs).
        metadata (dict): Additional information about the matrix.

    Examples:
        Creating a PairwiseCompMatrix:

            >>> matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
            >>> pcm = PairwiseCompMatrix(matrix=matrix, labels=["A", "B"])

        Exporting to NetworkX:

            >>> G = pcm.to_networkx()
            >>> G = pcm.to_networkx(threshold=0.3)  # Only edges with |weight| > 0.3
            >>> G = pcm.to_networkx(invert_weights=True)  # For shortest path algorithms

        Getting a binary thresholded matrix:

            >>> binary_pcm = pcm.threshold(0.4)  # Values > 0.4 become 1, else 0
    """

    matrix: np.ndarray
    labels: Optional[List[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"Matrix must be n x n, got {self.matrix.shape}")

        if self.labels is not None and len(self.labels) != self.matrix.shape[0]:
            raise ValueError(
                f"Number of labels ({len(self.labels)}) must match matrix dimension ({self.matrix.shape[0]})"
            )

    def __repr__(self) -> str:
        return f"PairwiseCompMatrix(shape={self.matrix.shape}, labels={self.labels}, metadata={list(self.metadata.keys())})"

    def to_networkx(
        self,
        threshold: Optional[float] = None,
        invert_weights: bool = False,
    ):
        """Export the matrix to a NetworkX graph.

        Parameters:
            threshold (float or None): If provided, only edges with absolute
                weight > threshold will be included.
            invert_weights (bool): If True, edge weights are set to
                (1 - value) instead of value. This is useful for weighted
                network metrics like shortest path length, where strong
                correlations (e.g., 0.9) should represent short/cheap paths
                rather than long/expensive paths.

        Returns:
            G (networkx.Graph): The exported graph.

        Notes:
            When using NetworkX for weighted shortest path algorithms (e.g.,
            ``nx.shortest_path_length``), edge weights are interpreted as
            distances. For correlation matrices where high values indicate
            strong relationships, set ``invert_weights=True`` so that:
            - Strong correlation (0.9) -> weight 0.1 (short path)
            - Weak correlation (0.1) -> weight 0.9 (long path)
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX is required for to_networkx. Install with 'pip install networkx'"
            )

        G = nx.Graph()
        n = self.matrix.shape[0]

        # Add nodes
        for i in range(n):
            label = self.labels[i] if self.labels is not None else i
            G.add_node(i, label=label)

        # Add edges
        for i in range(n):
            for j in range(i + 1, n):
                weight = self.matrix[i, j]
                if threshold is None or abs(weight) > threshold:
                    if not np.isnan(weight):
                        edge_weight = (1.0 - weight) if invert_weights else weight
                        G.add_edge(i, j, weight=float(edge_weight))

        return G

    def threshold(self, threshold: float) -> "PairwiseCompMatrix":
        """Create a binary matrix based on a threshold.

        Parameters:
            threshold (float): Values with absolute value > threshold become
                1, otherwise 0.

        Returns:
            result (PairwiseCompMatrix): A new PairwiseCompMatrix with binary
                (0/1) values.

        Examples:
            >>> matrix = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.5], [0.2, 0.5, 1.0]])
            >>> pcm = PairwiseCompMatrix(matrix=matrix)
            >>> binary_pcm = pcm.threshold(0.4)
            >>> print(binary_pcm.matrix)
            [[1. 1. 0.]
             [1. 1. 1.]
             [0. 1. 1.]]
        """
        binary_matrix = (np.abs(self.matrix) > threshold).astype(float)
        return PairwiseCompMatrix(
            matrix=binary_matrix,
            labels=self.labels,
            metadata={**self.metadata, "threshold": threshold, "binary": True},
        )

    def normalize(
        self,
        method: str = "min_max",
        *,
        axis: Optional[str] = None,
    ) -> "PairwiseCompMatrix":
        """Return a normalized copy of this matrix.

        Parameters:
            method (str): Normalization method. One of
                ``"min_max"`` (scale to [0, 1]),
                ``"z_score"`` (subtract mean, divide by std),
                ``"row"`` (per-row min-max), or
                ``"col"`` (per-column min-max).
            axis (str or None): When set to ``"row"`` or ``"col"``,
                normalization is applied per-row or per-column instead
                of globally.  When None (default), the entire matrix
                is normalized at once.

        Returns:
            result (PairwiseCompMatrix): A new PairwiseCompMatrix with
                normalized values.

        Notes:
            - NaN values are ignored during computation and preserved
              in the output.
            - For ``"z_score"``, if the standard deviation is zero the
              result is filled with zeros (no division by zero).
        """
        if method in ("row", "col"):
            axis = method
            method = "min_max"

        mat = self.matrix.astype(np.float64)

        if method == "min_max":
            normalized = _min_max_normalize(mat, axis)
        elif method == "z_score":
            normalized = _z_score_normalize(mat, axis)
        else:
            raise ValueError(
                f"Unknown normalization method {method!r}; "
                "expected 'min_max', 'z_score', 'row', or 'col'."
            )

        return PairwiseCompMatrix(
            matrix=normalized,
            labels=self.labels,
            metadata={
                **self.metadata,
                "normalization": method,
                "normalization_axis": axis,
            },
        )

    _OPS = {
        "lt": np.less,
        "le": np.less_equal,
        "gt": np.greater,
        "ge": np.greater_equal,
        "eq": np.equal,
        "ne": np.not_equal,
    }

    def remove_by_condition(
        self,
        condition: "PairwiseCompMatrix",
        op: str,
        threshold: float,
        fill: float = np.nan,
    ) -> "PairwiseCompMatrix":
        """Return a copy with entries removed where a condition matrix satisfies a comparison.

        Entries where the comparison ``op(condition, threshold)`` evaluates to
        True are replaced by *fill*; all other entries keep their original value
        from *self*.

        Parameters:
            condition (PairwiseCompMatrix): Matrix to evaluate the comparison on.
                Must have the same shape as self.
            op (str): Comparison operator applied element-wise to the condition
                matrix. Standard: ``"lt"`` (<), ``"le"`` (<=), ``"gt"`` (>),
                ``"ge"`` (>=), ``"eq"`` (==), ``"ne"`` (!=). Absolute-value
                variants: ``"abs_lt"``, ``"abs_le"``, ``"abs_gt"``, ``"abs_ge"``
                — these compare ``|condition|`` against the threshold.
            threshold (float): Threshold value for the comparison.
            fill (float): Replacement value for removed entries (default: NaN).

        Returns:
            result (PairwiseCompMatrix): Copy of self where entries satisfying
                the condition are replaced by *fill*. Labels and metadata are
                preserved from self.
        """
        if not isinstance(condition, PairwiseCompMatrix):
            raise TypeError(
                f"condition must be a PairwiseCompMatrix, got {type(condition).__name__}"
            )
        if condition.matrix.shape != self.matrix.shape:
            raise ValueError(
                f"condition shape {condition.matrix.shape} does not match "
                f"self shape {self.matrix.shape}"
            )

        use_abs = op.startswith("abs_")
        base_op = op[4:] if use_abs else op

        if base_op not in self._OPS:
            raise ValueError(
                f"Unknown op {op!r}. Must be one of: "
                f"{', '.join(sorted(self._OPS))} or their abs_ variants."
            )

        cond_values = np.abs(condition.matrix) if use_abs else condition.matrix
        mask = self._OPS[base_op](cond_values, threshold)

        result_matrix = self.matrix.copy()
        result_matrix[mask] = fill

        return PairwiseCompMatrix(
            matrix=result_matrix,
            labels=self.labels,
            metadata={
                **self.metadata,
                "removed_by_condition": {
                    "op": op,
                    "threshold": threshold,
                    "fill": fill,
                },
            },
        )

    def extract_lower_triangle(self) -> np.ndarray:
        """Extract lower triangle (excluding diagonal) from this correlation matrix.

        Returns:
            values (np.ndarray): Lower triangle values as a 1D array with
                shape ``(F,)`` where F = n*(n-1)/2.
        """
        n = self.matrix.shape[0]
        lower_tri_idx = np.tril_indices(n, k=-1)
        return self.matrix[lower_tri_idx[0], lower_tri_idx[1]]

    @staticmethod
    def _is_diverging(matrix):
        """Check whether a matrix has both meaningful negative and positive values."""
        finite = matrix[np.isfinite(matrix)]
        if len(finite) == 0:
            return False
        return float(finite.min()) < 0 and float(finite.max()) > 0

    def plot(
        self,
        ax=None,
        cmap=None,
        vmin=None,
        vmax=None,
        colorbar_label="",
        font_size=14,
        tick_labels=None,
        save_path=None,
    ):
        """Plot the pairwise matrix as a heatmap.

        Parameters:
            ax (matplotlib.axes.Axes or None): Target axes. If None a standalone
                figure is created.
            cmap (str or None): Matplotlib colormap name. If None,
                auto-selects ``"RdBu_r"`` for diverging data (contains both
                negative and positive values) or ``"viridis"`` otherwise.
            vmin (float or None): Colormap minimum.
            vmax (float or None): Colormap maximum.
            colorbar_label (str): Label for the colorbar.
            font_size (int): Font size for labels and ticks.
            tick_labels (list[str] or None): Custom tick labels for both axes.
                If None, uses ``self.labels`` (or integer indices when labels
                are not set).
            save_path (str or None): If provided (and ``ax`` is None), save the
                figure to this path and close it.

        Returns:
            result: ``(fig, ax)`` when ``ax`` is None, otherwise just ``ax``.
        """
        from .plot_utils import plot_heatmap

        if cmap is None:
            cmap = "RdBu_r" if self._is_diverging(self.matrix) else "viridis"

        if tick_labels is None:
            tick_labels = (
                self.labels
                if self.labels is not None
                else [str(i) for i in range(self.matrix.shape[0])]
            )
        n = self.matrix.shape[0]
        ticks = (list(range(n)), tick_labels)

        return plot_heatmap(
            self.matrix,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            origin="upper",
            xlabel="",
            ylabel="",
            xticks=ticks,
            yticks=ticks,
            show_colorbar=True,
            colorbar_label=colorbar_label,
            font_size=font_size,
            save_path=save_path,
        )

    def extract_pairs_by_group(
        self,
        unit_labels,
    ) -> dict:
        """Extract upper-triangle pair values grouped by unit label combinations.

        Given a label array of length N (one per unit), splits the upper
        triangle of the matrix into groups based on each pair's label
        combination. For example, a boolean label ``is_lower`` yields three
        groups: ``(False, False)``, ``(False, True)``, ``(True, True)``.

        Parameters:
            unit_labels (array-like): Labels of length N assigning each unit
                to a group. Can be boolean, integer, or string values.

        Returns:
            groups (dict): Mapping from ``(label_a, label_b)`` tuples to 1D
                arrays of pair values. Keys are canonically ordered so that
                ``label_a <= label_b``. Only groups with at least one pair are
                included. The values within each group preserve the order
                produced by ``np.triu_indices``, making results from different
                matrices with the same labels directly alignable for paired
                tests.
        """
        unit_labels = np.asarray(unit_labels)
        n = self.matrix.shape[0]
        if len(unit_labels) != n:
            raise ValueError(
                f"unit_labels length ({len(unit_labels)}) must match "
                f"matrix dimension ({n})"
            )

        ri, ci = np.triu_indices(n, k=1)
        values = self.matrix[ri, ci]
        labels_r = unit_labels[ri]
        labels_c = unit_labels[ci]

        unique_labels = sorted(set(unit_labels.tolist()))
        groups = {}
        for i_lbl, la in enumerate(unique_labels):
            for lb in unique_labels[i_lbl:]:
                if la == lb:
                    mask = (labels_r == la) & (labels_c == la)
                else:
                    mask = ((labels_r == la) & (labels_c == lb)) | (
                        (labels_r == lb) & (labels_c == la)
                    )
                if mask.any():
                    groups[(la, lb)] = values[mask]

        return groups

    def plot_spatial_network(
        self,
        ax,
        positions,
        edge_threshold=None,
        top_pct=None,
        node_size_range=(2, 20),
        node_cmap="viridis",
        node_linewidth=0.2,
        edge_color="red",
        edge_linewidth=0.6,
        edge_alpha_range=(0.15, 1.0),
        scale_bar_um=500,
        font_size=None,
    ):
        """Plot this pairwise matrix as a spatial network on MEA positions.

        Unit positions must be supplied as *positions* -- extract them from
        ``SpikeData.neuron_attributes`` (e.g.
        ``np.array([[na['x'], na['y']] for na in sd.neuron_attributes])``).

        Thin wrapper around ``plot_utils.plot_spatial_network``.

        Parameters:
            ax (matplotlib.axes.Axes): Target axes.
            positions (np.ndarray): Unit positions, shape ``(N, 2)`` with
                columns ``[x, y]`` in micrometres.
            edge_threshold (float or None): Minimum matrix value to draw an
                edge.
            top_pct (float or None): Percentage of top edges to draw.
            node_size_range (tuple): ``(min_size, max_size)`` in points² for
                scatter markers.
            node_cmap (str): Matplotlib colourmap for node colour.
            node_linewidth (float): Outline width of node markers.
            edge_color (str): Colour for network edges.
            edge_linewidth (float): Line width for network edges.
            edge_alpha_range (tuple): ``(min_alpha, max_alpha)`` for edge
                transparency.
            scale_bar_um (float): Scale bar length in micrometres (0 to omit).
            font_size (int or None): Font size for scale bar label.

        Returns:
            scatter (matplotlib.collections.PathCollection): The scatter
                artist, useful for adding a colorbar.
        """
        from .plot_utils import plot_spatial_network

        return plot_spatial_network(
            ax,
            positions,
            self.matrix,
            edge_threshold=edge_threshold,
            top_pct=top_pct,
            node_size_range=node_size_range,
            node_cmap=node_cmap,
            node_linewidth=node_linewidth,
            edge_color=edge_color,
            edge_linewidth=edge_linewidth,
            edge_alpha_range=edge_alpha_range,
            scale_bar_um=scale_bar_um,
            font_size=font_size,
        )


@dataclass
class PairwiseCompMatrixStack:
    """A data class for a stack of n x n pairwise comparison matrices (e.g., across slices or time bins).

    Attributes:
        stack (np.ndarray): The n x n x S stack of comparison matrices, where
            S is the number of slices.
        labels (list or None): Labels for the rows/columns (e.g., unit IDs).
        times (list of tuple or None): Time windows (start, end) associated
            with each matrix in the stack.
        metadata (dict): Additional information about the stack.

    The stack supports flexible indexing:

    - Single index: Returns a PairwiseCompMatrix for that slice.

        >>> stack[0]  # First matrix as PairwiseCompMatrix

    - Slice: Returns a new PairwiseCompMatrixStack with the selected range.

        >>> stack[0:5]  # First 5 matrices as a new stack
        >>> stack[::2]  # Every other matrix

    - Iteration: Iterate over all matrices in the stack.

        >>> for matrix in stack:
        ...     print(matrix.matrix.shape)

    - subslice(): Select specific non-contiguous slices by index.

        >>> stack.subslice([0, 2, 5])  # Select slices 0, 2, and 5

    Examples:
        Creating a stack:

            >>> stack_data = np.random.rand(5, 5, 10)  # 5x5 matrices, 10 slices
            >>> stack = PairwiseCompMatrixStack(stack=stack_data)

        Slicing:

            >>> sub_stack = stack[0:3]  # Get first 3 slices
            >>> single_matrix = stack[5]  # Get 6th slice as PairwiseCompMatrix

        Binary thresholding:

            >>> binary_stack = stack.threshold(0.5)  # Threshold all matrices
    """

    stack: np.ndarray
    labels: Optional[List[Any]] = None
    times: Optional[List[tuple]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.stack.ndim != 3 or self.stack.shape[0] != self.stack.shape[1]:
            raise ValueError(f"Stack must be n x n x S, got {self.stack.shape}")

        if self.labels is not None and len(self.labels) != self.stack.shape[0]:
            raise ValueError(
                f"Number of labels ({len(self.labels)}) must match matrix dimension ({self.stack.shape[0]})"
            )

        if self.times is not None and len(self.times) != self.stack.shape[2]:
            raise ValueError(
                f"Number of times ({len(self.times)}) must match stack size ({self.stack.shape[2]})"
            )

    def __repr__(self) -> str:
        return f"PairwiseCompMatrixStack(matrix_shape={self.stack.shape[:2]}, size={self.stack.shape[2]}, labels={self.labels}, metadata={list(self.metadata.keys())})"

    def __getitem__(
        self, index
    ) -> Union[PairwiseCompMatrix, "PairwiseCompMatrixStack"]:
        """Get a single matrix or a sub-stack by index or slice.

        Parameters:
            index (int or slice): int returns the matrix at that slice index
                as PairwiseCompMatrix; slice returns a new
                PairwiseCompMatrixStack with the selected slices.

        Returns:
            result (PairwiseCompMatrix or PairwiseCompMatrixStack): Single
                matrix or sub-stack.

        Examples:
            >>> stack[0]      # Get first matrix as PairwiseCompMatrix
            >>> stack[0:5]    # Get first 5 matrices as new stack
            >>> stack[::2]    # Get every other matrix
        """
        if isinstance(index, (slice, np.ndarray, list)):
            return PairwiseCompMatrixStack(
                stack=self.stack[:, :, index],
                labels=self.labels,
                times=self.times[index] if self.times else None,
                metadata=self.metadata.copy(),
            )

        return PairwiseCompMatrix(
            matrix=self.stack[:, :, index],
            labels=self.labels,
            metadata={
                **self.metadata,
                "stack_index": index,
                "time": self.times[index] if self.times else None,
            },
        )

    def __iter__(self) -> Iterator[PairwiseCompMatrix]:
        """Iterate over each matrix in the stack."""
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        """Return the number of slices in the stack."""
        return self.stack.shape[2]

    def subslice(self, indices: List[int]) -> "PairwiseCompMatrixStack":
        """Select specific slices from the stack by their indices.

        Parameters:
            indices (list of int): List of slice indices to select.

        Returns:
            result (PairwiseCompMatrixStack): A new stack containing only the
                selected slices.

        Examples:
            >>> stack = PairwiseCompMatrixStack(stack=np.random.rand(5, 5, 10))
            >>> sub = stack.subslice([0, 2, 5, 9])  # Select specific slices
            >>> len(sub)  # 4
        """
        indices = list(indices)
        return PairwiseCompMatrixStack(
            stack=self.stack[:, :, indices],
            labels=self.labels,
            times=[self.times[i] for i in indices] if self.times else None,
            metadata=self.metadata.copy(),
        )

    def threshold(self, threshold: float) -> "PairwiseCompMatrixStack":
        """Create a binary stack based on a threshold.

        Parameters:
            threshold (float): Values with absolute value > threshold become
                1, otherwise 0.

        Returns:
            result (PairwiseCompMatrixStack): A new stack with binary (0/1)
                values.

        Examples:
            >>> stack = PairwiseCompMatrixStack(stack=np.random.rand(5, 5, 10))
            >>> binary_stack = stack.threshold(0.5)
        """
        binary_stack = (np.abs(self.stack) > threshold).astype(float)
        return PairwiseCompMatrixStack(
            stack=binary_stack,
            labels=self.labels,
            times=self.times,
            metadata={**self.metadata, "threshold": threshold, "binary": True},
        )

    def normalize(
        self,
        method: str = "min_max",
        *,
        axis: Optional[str] = None,
        per_slice: bool = False,
    ) -> "PairwiseCompMatrixStack":
        """Return a normalized copy of this stack.

        Parameters:
            method (str): Normalization method (``"min_max"``,
                ``"z_score"``, ``"row"``, or ``"col"``).  See
                ``PairwiseCompMatrix.normalize`` for details.
            axis (str or None): ``"row"`` or ``"col"`` for per-row /
                per-column normalization within each N x N slice, or
                None for global normalization.
            per_slice (bool): When True, each slice is normalized
                independently.  When False (default), statistics are
                computed across the entire stack.

        Returns:
            result (PairwiseCompMatrixStack): A new stack with
                normalized values.
        """
        if method in ("row", "col"):
            axis = method
            method = "min_max"

        if per_slice:
            slices = []
            for s in range(self.stack.shape[2]):
                mat = self.stack[:, :, s].astype(np.float64)
                if method == "min_max":
                    slices.append(_min_max_normalize(mat, axis))
                elif method == "z_score":
                    slices.append(_z_score_normalize(mat, axis))
                else:
                    raise ValueError(
                        f"Unknown normalization method {method!r}; "
                        "expected 'min_max', 'z_score', 'row', or 'col'."
                    )
            normalized = np.stack(slices, axis=2)
        else:
            if axis is not None:
                raise ValueError(
                    f"axis={axis!r} is only supported with per_slice=True. "
                    "Global (per_slice=False) normalization operates on the "
                    "entire stack — per-row/col normalization is ambiguous "
                    "across slices."
                )
            stk = self.stack.astype(np.float64)
            if method == "min_max":
                lo = np.nanmin(stk)
                hi = np.nanmax(stk)
                rng = hi - lo
                normalized = (stk - lo) / rng if rng != 0 else np.zeros_like(stk)
            elif method == "z_score":
                mu = np.nanmean(stk)
                sd = np.nanstd(stk)
                normalized = (stk - mu) / sd if sd != 0 else np.zeros_like(stk)
            else:
                raise ValueError(
                    f"Unknown normalization method {method!r}; "
                    "expected 'min_max', 'z_score', 'row', or 'col'."
                )

        return PairwiseCompMatrixStack(
            stack=normalized,
            labels=self.labels,
            times=self.times,
            metadata={
                **self.metadata,
                "normalization": method,
                "normalization_axis": axis,
                "normalization_per_slice": per_slice,
            },
        )

    _OPS = PairwiseCompMatrix._OPS

    def remove_by_condition(
        self,
        condition: Union[PairwiseCompMatrix, "PairwiseCompMatrixStack"],
        op: str,
        threshold: float,
        fill: float = np.nan,
    ) -> "PairwiseCompMatrixStack":
        """Return a copy with entries removed where a condition satisfies a comparison.

        Entries where ``op(condition, threshold)`` evaluates to True are
        replaced by *fill*; all other entries keep their original value from
        *self*. The condition is applied element-wise across all slices.

        Parameters:
            condition (PairwiseCompMatrix or PairwiseCompMatrixStack): Matrix or
                stack to evaluate the comparison on. A single
                ``PairwiseCompMatrix`` is broadcast across all slices. A
                ``PairwiseCompMatrixStack`` must have the same shape
                ``(N, N, S)`` as self.
            op (str): Comparison operator applied element-wise to the condition.
                Standard: ``"lt"`` (<), ``"le"`` (<=), ``"gt"`` (>),
                ``"ge"`` (>=), ``"eq"`` (==), ``"ne"`` (!=). Absolute-value
                variants: ``"abs_lt"``, ``"abs_le"``, ``"abs_gt"``, ``"abs_ge"``
                — these compare ``|condition|`` against the threshold.
            threshold (float): Threshold value for the comparison.
            fill (float): Replacement value for removed entries (default: NaN).

        Returns:
            result (PairwiseCompMatrixStack): Copy of self where entries
                satisfying the condition are replaced by *fill*. Labels, times,
                and metadata are preserved from self.
        """
        use_abs = op.startswith("abs_")
        base_op = op[4:] if use_abs else op

        if base_op not in self._OPS:
            raise ValueError(
                f"Unknown op {op!r}. Must be one of: "
                f"{', '.join(sorted(self._OPS))} or their abs_ variants."
            )

        if isinstance(condition, PairwiseCompMatrix):
            if condition.matrix.shape != self.stack.shape[:2]:
                raise ValueError(
                    f"condition shape {condition.matrix.shape} does not match "
                    f"stack matrix shape {self.stack.shape[:2]}"
                )
            # Broadcast (N, N) -> (N, N, S)
            cond_values = condition.matrix[:, :, np.newaxis]
            if use_abs:
                cond_values = np.abs(cond_values)
            mask = self._OPS[base_op](
                np.broadcast_to(cond_values, self.stack.shape), threshold
            )
        elif isinstance(condition, PairwiseCompMatrixStack):
            if condition.stack.shape != self.stack.shape:
                raise ValueError(
                    f"condition shape {condition.stack.shape} does not match "
                    f"self shape {self.stack.shape}"
                )
            cond_values = np.abs(condition.stack) if use_abs else condition.stack
            mask = self._OPS[base_op](cond_values, threshold)
        else:
            raise TypeError(
                f"condition must be a PairwiseCompMatrix or PairwiseCompMatrixStack, "
                f"got {type(condition).__name__}"
            )

        result_stack = self.stack.copy()
        result_stack[mask] = fill

        return PairwiseCompMatrixStack(
            stack=result_stack,
            labels=self.labels,
            times=self.times,
            metadata={
                **self.metadata,
                "removed_by_condition": {
                    "op": op,
                    "threshold": threshold,
                    "fill": fill,
                },
            },
        )

    def mean(self, ignore_nan: bool = True) -> PairwiseCompMatrix:
        """Compute the mean matrix across the stack.

        Parameters:
            ignore_nan (bool): Whether to use np.nanmean to ignore NaN values
                in the average.

        Returns:
            mean_matrix (PairwiseCompMatrix): The element-wise mean across all
                slices.
        """
        if ignore_nan:
            mean_matrix = np.nanmean(self.stack, axis=2)
        else:
            mean_matrix = np.mean(self.stack, axis=2)

        return PairwiseCompMatrix(
            matrix=mean_matrix,
            labels=self.labels,
            metadata={**self.metadata, "computed": "mean"},
        )

    def plot_mean(
        self,
        ax=None,
        ignore_nan=True,
        cmap=None,
        vmin=None,
        vmax=None,
        colorbar_label="",
        font_size=14,
        tick_labels=None,
        save_path=None,
    ):
        """Plot the mean matrix across all slices as a heatmap.

        Computes ``nanmean`` (or ``mean``) over the stack axis and delegates
        to ``PairwiseCompMatrix.plot()``.

        Parameters:
            ax (matplotlib.axes.Axes or None): Target axes. If None a standalone
                figure is created.
            ignore_nan (bool): Use ``np.nanmean`` to ignore NaN values.
            cmap (str or None): Matplotlib colormap name. If None,
                auto-selects based on whether the mean matrix is diverging.
            vmin (float or None): Colormap minimum.
            vmax (float or None): Colormap maximum.
            colorbar_label (str): Label for the colorbar.
            font_size (int): Font size for labels and ticks.
            tick_labels (list[str] or None): Custom tick labels for both axes.
                If None, uses ``self.labels`` (or integer indices when labels
                are not set).
            save_path (str or None): If provided (and ``ax`` is None), save the
                figure to this path and close it.

        Returns:
            result: ``(fig, ax)`` when ``ax`` is None, otherwise just ``ax``.
        """
        mean_pcm = self.mean(ignore_nan=ignore_nan)
        return mean_pcm.plot(
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar_label=colorbar_label,
            font_size=font_size,
            tick_labels=tick_labels,
            save_path=save_path,
        )

    def extract_lower_triangle_features(self) -> np.ndarray:
        """Extract lower triangle (excluding diagonal) from each correlation matrix in the stack.

        Returns:
            features (np.ndarray): 2D matrix of shape ``(S, F)`` where each
                row contains lower triangle values for that correlation
                matrix. F = n*(n-1)/2 (number of unique pairs).
        """
        matrix_3d = self.stack
        if matrix_3d.ndim != 3:
            raise ValueError(
                f"Stack must be a 3D array (n, n, S), got {matrix_3d.ndim}D"
            )
        if matrix_3d.shape[0] != matrix_3d.shape[1]:
            raise ValueError(
                "Stack must have shape (n, n, S) where the first two dimensions are equal."
            )
        num_items = matrix_3d.shape[0]
        lower_tri_idx = np.tril_indices(num_items, k=-1)
        # matrix_3d[lower_tri_idx[0], lower_tri_idx[1], :] gives (F, S), transpose to (S, F)
        features = matrix_3d[lower_tri_idx[0], lower_tri_idx[1], :].T
        return features

    def dim_red_on_lower_diagonal_corr_matrix(
        self,
        method: str = "PCA",
        n_components: int = 2,
        **kwargs,
    ) -> np.ndarray:
        """Apply dimensionality reduction (PCA or UMAP) to the lower triangle of each correlation matrix in the stack.

        Parameters:
            method (str): Dimensionality reduction method to use. ``"PCA"``
                (default) or ``"UMAP"``.
            n_components (int): Number of components (dimensions) in the
                output manifold.
            **kwargs: Additional keyword arguments passed through to UMAP when
                ``method='UMAP'`` (e.g., ``n_neighbors``, ``min_dist``,
                ``metric``).

        Returns:
            result (tuple): For PCA: a 3-tuple
                ``(embedding, explained_variance_ratio, components)`` with
                shapes ``(S, n_components)``, ``(n_components,)``, and
                ``(n_components, F)`` where ``F = N*(N-1)//2``.
                For UMAP: a 2-tuple ``(embedding, trustworthiness)`` with
                embedding shape ``(S, n_components)`` and trustworthiness
                a float in [0, 1].
        """
        from .utils import PCA_reduction, UMAP_reduction

        lower_triangle = self.extract_lower_triangle_features()

        method_upper = method.upper()
        if method_upper == "PCA":
            if kwargs:
                raise TypeError(
                    "Additional keyword arguments are only supported for UMAP; "
                    f"got kwargs {list(kwargs.keys())} for method='{method}'."
                )
            return PCA_reduction(lower_triangle, n_components=n_components)
        if method_upper == "UMAP":
            return UMAP_reduction(
                lower_triangle,
                n_components=n_components,
                **kwargs,
            )

        raise ValueError(
            f"Unknown manifold method '{method}' (expected 'PCA' or 'UMAP')."
        )


# ---------------------------------------------------------------------------
# Normalization helpers (used by PairwiseCompMatrix.normalize and
# PairwiseCompMatrixStack.normalize)
# ---------------------------------------------------------------------------


def _min_max_normalize(mat: np.ndarray, axis: Optional[str] = None) -> np.ndarray:
    """Min-max normalize a 2-D matrix to [0, 1].

    Parameters:
        mat (np.ndarray): ``(N, N)`` input matrix.
        axis (str or None): ``"row"``, ``"col"``, or None (global).

    Returns:
        normalized (np.ndarray): ``(N, N)`` normalized matrix.
    """
    if axis == "row":
        lo = np.nanmin(mat, axis=1, keepdims=True)
        hi = np.nanmax(mat, axis=1, keepdims=True)
    elif axis == "col":
        lo = np.nanmin(mat, axis=0, keepdims=True)
        hi = np.nanmax(mat, axis=0, keepdims=True)
    else:
        lo = np.nanmin(mat)
        hi = np.nanmax(mat)

    rng = hi - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(rng != 0, (mat - lo) / rng, 0.0)
    # Preserve NaN
    result[np.isnan(mat)] = np.nan
    return result


def _z_score_normalize(mat: np.ndarray, axis: Optional[str] = None) -> np.ndarray:
    """Z-score normalize a 2-D matrix (mean=0, std=1).

    Parameters:
        mat (np.ndarray): ``(N, N)`` input matrix.
        axis (str or None): ``"row"``, ``"col"``, or None (global).

    Returns:
        normalized (np.ndarray): ``(N, N)`` normalized matrix.
    """
    if axis == "row":
        mu = np.nanmean(mat, axis=1, keepdims=True)
        sd = np.nanstd(mat, axis=1, keepdims=True)
    elif axis == "col":
        mu = np.nanmean(mat, axis=0, keepdims=True)
        sd = np.nanstd(mat, axis=0, keepdims=True)
    else:
        mu = np.nanmean(mat)
        sd = np.nanstd(mat)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(sd != 0, (mat - mu) / sd, 0.0)
    result[np.isnan(mat)] = np.nan
    return result
