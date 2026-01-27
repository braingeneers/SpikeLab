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

    def to_networkx(self, threshold: Optional[float] = None):
        """
        Export the matrix to a NetworkX graph.

        Parameters:
        -----------
        threshold: float, optional
            If provided, only edges with absolute weight > threshold will be included.

        Returns:
        --------
        G: networkx.Graph
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
                        G.add_edge(i, j, weight=float(weight))

        return G


@dataclass
class PairwiseCompMatrixStack:
    """
    A data class for a stack of n x n pairwise comparison matrices (e.g., across slices or time bins).

    Attributes:
    -----------
    stack : np.ndarray
        The S x n x n stack of comparison matrices.
    labels : list, optional
        Labels for the rows/columns (e.g., unit IDs).
    times : list of tuples, optional
        Time windows (start, end) associated with each matrix in the stack.
    metadata : dict
        Additional information about the stack.
    """

    stack: np.ndarray
    labels: Optional[List[Any]] = None
    times: Optional[List[tuple]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.stack.ndim != 3 or self.stack.shape[1] != self.stack.shape[2]:
            raise ValueError(f"Stack must be S x n x n, got {self.stack.shape}")

        if self.labels is not None and len(self.labels) != self.stack.shape[1]:
            raise ValueError(
                f"Number of labels ({len(self.labels)}) must match matrix dimension ({self.stack.shape[1]})"
            )

        if self.times is not None and len(self.times) != self.stack.shape[0]:
            raise ValueError(
                f"Number of times ({len(self.times)}) must match stack size ({self.stack.shape[0]})"
            )

    def __repr__(self) -> str:
        return f"PairwiseCompMatrixStack(size={self.stack.shape[0]}, matrix_shape={self.stack.shape[1:]}, labels={self.labels}, metadata={list(self.metadata.keys())})"

    def __getitem__(
        self, index
    ) -> Union[PairwiseCompMatrix, "PairwiseCompMatrixStack"]:
        if isinstance(index, slice):
            return PairwiseCompMatrixStack(
                stack=self.stack[index],
                labels=self.labels,
                times=self.times[index] if self.times else None,
                metadata=self.metadata.copy(),
            )

        return PairwiseCompMatrix(
            matrix=self.stack[index],
            labels=self.labels,
            metadata={
                **self.metadata,
                "stack_index": index,
                "time": self.times[index] if self.times else None,
            },
        )

    def __iter__(self) -> Iterator[PairwiseCompMatrix]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.stack.shape[0]

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
            mean_matrix = np.nanmean(self.stack, axis=0)
        else:
            mean_matrix = np.mean(self.stack, axis=0)

        return PairwiseCompMatrix(
            matrix=mean_matrix,
            labels=self.labels,
            metadata={**self.metadata, "computed": "mean"},
        )
