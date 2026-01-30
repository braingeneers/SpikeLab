from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Iterator
import numpy as np


@dataclass
class PairwiseCompMatrix:
    """
    A data class for n x n pairwise comparison matrices (e.g., correlation, STTC).

    Attributes:
    -----------
    matrix : np.ndarray
        The n x n comparison matrix.
    labels : list, optional
        Labels for the rows/columns (e.g., unit IDs).
    metadata : dict
        Additional information about the matrix.

    Examples:
    ---------
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
        """
        Export the matrix to a NetworkX graph.

        Parameters:
        -----------
        threshold : float, optional
            If provided, only edges with absolute weight > threshold will be included.
        invert_weights : bool, default False
            If True, edge weights are set to (1 - value) instead of value.
            This is useful for weighted network metrics like shortest path length,
            where strong correlations (e.g., 0.9) should represent short/cheap paths
            rather than long/expensive paths.

        Returns:
        --------
        G : networkx.Graph

        Notes:
        ------
        When using NetworkX for weighted shortest path algorithms (e.g.,
        `nx.shortest_path_length`), edge weights are interpreted as distances.
        For correlation matrices where high values indicate strong relationships,
        set `invert_weights=True` so that:
        - Strong correlation (0.9) → weight 0.1 (short path)
        - Weak correlation (0.1) → weight 0.9 (long path)
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
        """
        Create a binary matrix based on a threshold.

        Parameters:
        -----------
        threshold : float
            Values with absolute value > threshold become 1, otherwise 0.

        Returns:
        --------
        PairwiseCompMatrix
            A new PairwiseCompMatrix with binary (0/1) values.

        Examples:
        ---------
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


@dataclass
class PairwiseCompMatrixStack:
    """
    A data class for a stack of n x n pairwise comparison matrices (e.g., across slices or time bins).

    Attributes:
    -----------
    stack : np.ndarray
        The n x n x S stack of comparison matrices, where S is the number of slices.
    labels : list, optional
        Labels for the rows/columns (e.g., unit IDs).
    times : list of tuples, optional
        Time windows (start, end) associated with each matrix in the stack.
    metadata : dict
        Additional information about the stack.

    Indexing and Slicing:
    ---------------------
    The stack supports flexible indexing:

    - **Single index**: Returns a PairwiseCompMatrix for that slice.

        >>> stack[0]  # First matrix as PairwiseCompMatrix

    - **Slice**: Returns a new PairwiseCompMatrixStack with the selected range.

        >>> stack[0:5]  # First 5 matrices as a new stack
        >>> stack[::2]  # Every other matrix

    - **Iteration**: Iterate over all matrices in the stack.

        >>> for matrix in stack:
        ...     print(matrix.matrix.shape)

    - **subslice()**: Select specific non-contiguous slices by index.

        >>> stack.subslice([0, 2, 5])  # Select slices 0, 2, and 5

    Examples:
    ---------
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
        """
        Get a single matrix or a sub-stack by index or slice.

        Parameters:
        -----------
        index : int or slice
            - int: Returns the matrix at that slice index as PairwiseCompMatrix
            - slice: Returns a new PairwiseCompMatrixStack with the selected slices

        Returns:
        --------
        PairwiseCompMatrix or PairwiseCompMatrixStack

        Examples:
        ---------
        >>> stack[0]      # Get first matrix as PairwiseCompMatrix
        >>> stack[0:5]    # Get first 5 matrices as new stack
        >>> stack[::2]    # Get every other matrix
        """
        if isinstance(index, slice):
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
        """
        Select specific slices from the stack by their indices.

        Parameters:
        -----------
        indices : list of int
            List of slice indices to select.

        Returns:
        --------
        PairwiseCompMatrixStack
            A new stack containing only the selected slices.

        Examples:
        ---------
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
        """
        Create a binary stack based on a threshold.

        Parameters:
        -----------
        threshold : float
            Values with absolute value > threshold become 1, otherwise 0.

        Returns:
        --------
        PairwiseCompMatrixStack
            A new stack with binary (0/1) values.

        Examples:
        ---------
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

    def mean(self, ignore_nan: bool = True) -> PairwiseCompMatrix:
        """
        Compute the mean matrix across the stack.

        Parameters:
        -----------
        ignore_nan : bool, default True
            Whether to use np.nanmean to ignore NaN values in the average.

        Returns:
        --------
        mean_matrix : PairwiseCompMatrix
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
