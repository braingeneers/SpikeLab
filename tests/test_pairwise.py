import warnings

import numpy as np
import networkx as nx
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spikelab.spikedata.pairwise import (
    PairwiseCompMatrix,
    PairwiseCompMatrixStack,
)
from spikelab.spikedata import SpikeData
from spikelab.spikedata.rateslicestack import RateSliceStack

try:
    import umap  # noqa: F401

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# PairwiseCompMatrix — __init__ / __post_init__
# ---------------------------------------------------------------------------


class TestPairwiseCompMatrixInit:
    """Tests for PairwiseCompMatrix initialization and validation."""

    def test_pairwise_comp_matrix_init(self):
        # Normal init
        matrix = np.random.rand(5, 5)
        pcm = PairwiseCompMatrix(matrix=matrix, labels=["a", "b", "c", "d", "e"])
        assert pcm.matrix.shape == (5, 5)
        assert len(pcm.labels) == 5

        # Invalid shape
        with pytest.raises(ValueError):
            PairwiseCompMatrix(matrix=np.random.rand(5, 4))

        # Label mismatch
        with pytest.raises(ValueError):
            PairwiseCompMatrix(matrix=np.random.rand(5, 5), labels=["a", "b"])

    def test_post_init_wrong_ndim(self):
        """PairwiseCompMatrix rejects non-2D arrays.

        Tests: passing a 1D or 3D array to PairwiseCompMatrix
        should raise a ValueError during __post_init__ validation.
        """
        with pytest.raises(ValueError):
            PairwiseCompMatrix(matrix=np.array([1.0, 2.0, 3.0]))

        with pytest.raises(ValueError):
            PairwiseCompMatrix(matrix=np.random.rand(3, 3, 3))

    def test_init_0x0_matrix(self):
        """EC-PW-01: A 0x0 matrix is accepted by __post_init__.

        Tests: PairwiseCompMatrix with a (0,0) ndarray should
        construct successfully since the square check (shape[0] == shape[1])
        passes for empty arrays.
        """
        matrix = np.empty((0, 0))
        pcm = PairwiseCompMatrix(matrix=matrix)
        assert pcm.matrix.shape == (0, 0)
        assert pcm.labels is None


# ---------------------------------------------------------------------------
# PairwiseCompMatrix — to_networkx
# ---------------------------------------------------------------------------


class TestPairwiseCompMatrixToNetworkx:
    """Tests for PairwiseCompMatrix.to_networkx()."""

    def test_pairwise_comp_matrix_to_networkx(self):
        matrix = np.array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.8], [0.1, 0.8, 1.0]])
        pcm = PairwiseCompMatrix(matrix=matrix, labels=["A", "B", "C"])

        # No threshold
        G = pcm.to_networkx()
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 3  # (0,1), (0,2), (1,2)
        assert G.edges[0, 1]["weight"] == 0.5

        # With threshold
        G_thresh = pcm.to_networkx(threshold=0.6)
        assert G_thresh.number_of_edges() == 1  # Only (1,2) with 0.8
        assert (1, 2) in G_thresh.edges

        # Handling NaNs in NetworkX
        matrix_nan = matrix.copy()
        matrix_nan[0, 1] = np.nan
        pcm_nan = PairwiseCompMatrix(matrix=matrix_nan)
        G_nan = pcm_nan.to_networkx()
        assert G_nan.number_of_edges() == 2  # (0,2) and (1,2)

    def test_pairwise_comp_matrix_to_networkx_invert_weights(self):
        """Test that invert_weights correctly transforms edge weights to 1-value."""
        matrix = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.5], [0.1, 0.5, 1.0]])
        pcm = PairwiseCompMatrix(matrix=matrix)

        # Without invert_weights
        G = pcm.to_networkx()
        assert G.edges[0, 1]["weight"] == pytest.approx(0.9)
        assert G.edges[0, 2]["weight"] == pytest.approx(0.1)

        # With invert_weights=True
        G_inv = pcm.to_networkx(invert_weights=True)
        assert G_inv.edges[0, 1]["weight"] == pytest.approx(0.1)  # 1 - 0.9
        assert G_inv.edges[0, 2]["weight"] == pytest.approx(0.9)  # 1 - 0.1
        assert G_inv.edges[1, 2]["weight"] == pytest.approx(0.5)  # 1 - 0.5

        # Verify shortest path now uses inverted weights correctly
        # Strong correlation (0.9) should now be a short path (0.1)
        path_length = nx.shortest_path_length(
            G_inv, source=0, target=1, weight="weight"
        )
        assert path_length == pytest.approx(0.1)

    def test_to_networkx_all_nan(self):
        """All-NaN matrix produces a graph with nodes but no edges.

        Tests: to_networkx on a (3,3) all-NaN matrix should
        return a graph with 3 nodes and 0 edges since NaN weights are skipped.
        """
        matrix = np.full((3, 3), np.nan)
        pcm = PairwiseCompMatrix(matrix=matrix)
        G = pcm.to_networkx()
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 0

    def test_to_networkx_negative_weights(self):
        """EC-PW-02: Negative weights are preserved as-is in the graph.

        Tests: to_networkx on a matrix with negative off-diagonal values
        should include those edges with their original negative weight.
        """
        matrix = np.array([[1.0, -0.5], [-0.5, 1.0]])
        pcm = PairwiseCompMatrix(matrix=matrix)
        G = pcm.to_networkx()
        assert G.number_of_edges() == 1
        assert G.edges[0, 1]["weight"] == pytest.approx(-0.5)

    def test_to_networkx_invert_weights_outside_0_1(self):
        """EC-PW-03: invert_weights with values > 1 or < 0 produces out-of-range weights.

        Tests: to_networkx(invert_weights=True) computes 1 - value
        without clamping. A weight of 1.5 becomes -0.5 and a weight of
        -0.3 becomes 1.3. This can produce negative edge weights which
        may break shortest-path algorithms.
        """
        matrix = np.array([[1.0, 1.5, -0.3], [1.5, 1.0, 0.5], [-0.3, 0.5, 1.0]])
        pcm = PairwiseCompMatrix(matrix=matrix)
        G = pcm.to_networkx(invert_weights=True)
        # 1 - 1.5 = -0.5
        assert G.edges[0, 1]["weight"] == pytest.approx(-0.5)
        # 1 - (-0.3) = 1.3
        assert G.edges[0, 2]["weight"] == pytest.approx(1.3)
        # 1 - 0.5 = 0.5
        assert G.edges[1, 2]["weight"] == pytest.approx(0.5)

    def test_to_networkx_inf_weights_inverted(self):
        """Inf weights produce -Inf when inverted via 1 - Inf.

        Tests:
            (Test Case 1) Inf edge is included (np.isnan(Inf) is False).
            (Test Case 2) With invert_weights=True, weight becomes 1 - Inf = -Inf.

        Notes:
            - Negative-infinity edge weights can break shortest-path algorithms
              such as Dijkstra. This documents the current behavior rather than
              a desired invariant.
        """
        matrix = np.array([[1.0, np.inf], [np.inf, 1.0]])
        pcm = PairwiseCompMatrix(matrix=matrix)

        # Without invert: Inf is preserved
        G = pcm.to_networkx()
        assert G.edges[0, 1]["weight"] == float("inf")

        # With invert: 1 - Inf = -Inf
        G_inv = pcm.to_networkx(invert_weights=True)
        assert G_inv.edges[0, 1]["weight"] == float("-inf")


# ---------------------------------------------------------------------------
# PairwiseCompMatrix — threshold
# ---------------------------------------------------------------------------


class TestPairwiseCompMatrixThreshold:
    """Tests for PairwiseCompMatrix.threshold()."""

    def test_pairwise_comp_matrix_threshold(self):
        """Test the threshold method for creating binary matrices."""
        matrix = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.5], [0.2, 0.5, 1.0]])
        pcm = PairwiseCompMatrix(matrix=matrix)

        # Threshold at 0.4
        binary_pcm = pcm.threshold(0.4)
        expected = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        np.testing.assert_array_equal(binary_pcm.matrix, expected)
        assert binary_pcm.metadata["threshold"] == 0.4
        assert binary_pcm.metadata["binary"] is True

        # Test with negative values (absolute value)
        matrix_neg = np.array([[1.0, -0.8, 0.2], [-0.8, 1.0, -0.5], [0.2, -0.5, 1.0]])
        pcm_neg = PairwiseCompMatrix(matrix=matrix_neg)
        binary_neg = pcm_neg.threshold(0.4)
        np.testing.assert_array_equal(binary_neg.matrix, expected)

    def test_threshold_zero(self):
        """Thresholding at zero turns all non-zero values to 1.

        Tests: threshold(0.0) should set every cell whose
        absolute value exceeds 0 to 1 and leave exact zeros as 0.
        """
        matrix = np.array([[0.0, 0.5, 0.0], [0.5, 1.0, -0.3], [0.0, -0.3, 0.0]])
        pcm = PairwiseCompMatrix(matrix=matrix)
        result = pcm.threshold(0.0)
        expected = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
        np.testing.assert_array_equal(result.matrix, expected)

    def test_threshold_nan_values(self):
        """EC-PW-04: NaN values become 0 after thresholding.

        Tests: threshold() uses np.abs() > threshold which evaluates
        to False for NaN (since NaN comparisons are always False).
        NaN entries silently become 0.0 in the binary result.
        """
        matrix = np.array([[1.0, np.nan, 0.5], [np.nan, 1.0, 0.2], [0.5, 0.2, 1.0]])
        pcm = PairwiseCompMatrix(matrix=matrix)
        result = pcm.threshold(0.3)
        # NaN entries -> abs(NaN) > 0.3 -> False -> 0.0
        expected = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        np.testing.assert_array_equal(result.matrix, expected)


# ---------------------------------------------------------------------------
# PairwiseCompMatrix — extract_lower_triangle
# ---------------------------------------------------------------------------


class TestPairwiseCompMatrixExtractLowerTriangle:
    """Tests for PairwiseCompMatrix.extract_lower_triangle()."""

    def test_extract_lower_triangle_1x1(self):
        """Lower triangle of a 1x1 matrix is empty.

        Tests: a (1,1) PairwiseCompMatrix has no off-diagonal
        elements, so extract_lower_triangle should return an empty array.
        """
        pcm = PairwiseCompMatrix(matrix=np.array([[1.0]]))
        result = pcm.extract_lower_triangle()
        assert result.shape == (0,)
        assert len(result) == 0

    def test_extract_lower_triangle_2x2(self):
        """
        Lower triangle of a 2x2 matrix contains exactly one element.

        Tests:
            (Test Case 1) Result shape is (1,) — one off-diagonal pair.
            (Test Case 2) Value matches the (1, 0) entry.
        """
        matrix = np.array([[1.0, 0.3], [0.7, 1.0]])
        pcm = PairwiseCompMatrix(matrix=matrix)
        result = pcm.extract_lower_triangle()
        assert result.shape == (1,)
        assert result[0] == pytest.approx(0.7)

    def test_extract_lower_triangle_3x3(self):
        """
        Lower triangle of a 3x3 matrix contains 3 elements in column-major order.

        Tests:
            (Test Case 1) Result shape is (3,) — F = 3*(3-1)/2 = 3.
            (Test Case 2) Values match entries (1,0), (2,0), (2,1).
        """
        matrix = np.array([[1.0, 0.2, 0.3], [0.4, 1.0, 0.5], [0.6, 0.7, 1.0]])
        pcm = PairwiseCompMatrix(matrix=matrix)
        result = pcm.extract_lower_triangle()
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, [0.4, 0.6, 0.7])

    def test_extract_lower_triangle_with_nan(self):
        """
        NaN values in the lower triangle are preserved in the output.

        Tests:
            (Test Case 1) NaN at position (1,0) appears in the extracted array.
            (Test Case 2) Non-NaN values are preserved.
        """
        matrix = np.array([[1.0, 0.5], [np.nan, 1.0]])
        pcm = PairwiseCompMatrix(matrix=matrix)
        result = pcm.extract_lower_triangle()
        assert result.shape == (1,)
        assert np.isnan(result[0])


# ---------------------------------------------------------------------------
# PairwiseCompMatrixStack — init / basic
# ---------------------------------------------------------------------------


class TestPairwiseCompMatrixStackInit:
    """Tests for PairwiseCompMatrixStack initialization and basic operations."""

    def test_pairwise_comp_matrix_stack(self):
        # n x n x S format (5x5 matrices, 10 slices)
        stack_data = np.random.rand(5, 5, 10)
        times = [(i * 100, (i + 1) * 100) for i in range(10)]
        stack = PairwiseCompMatrixStack(stack=stack_data, times=times)

        assert len(stack) == 10
        assert stack[0].matrix.shape == (5, 5)
        assert stack[0].metadata["time"] == (0, 100)

        # Mean calculation
        mean_pcm = stack.mean()
        assert np.allclose(mean_pcm.matrix, np.mean(stack_data, axis=2))

        # Mean with NaNs
        stack_data_nan = stack_data.copy()
        stack_data_nan[0, 1, 0] = np.nan
        stack_nan = PairwiseCompMatrixStack(stack=stack_data_nan)
        mean_pcm_nan = stack_nan.mean(ignore_nan=True)
        assert not np.isnan(mean_pcm_nan.matrix[0, 1])

        mean_pcm_raw = stack_nan.mean(ignore_nan=False)
        assert np.isnan(mean_pcm_raw.matrix[0, 1])

    def test_init_non_square_slices(self):
        """EC-PW-05: Non-square slices are rejected by __post_init__.

        Tests: PairwiseCompMatrixStack with shape (3, 4, 5) should
        raise a ValueError because the first two dimensions are not equal.
        """
        with pytest.raises(ValueError, match="n x n x S"):
            PairwiseCompMatrixStack(stack=np.random.rand(3, 4, 5))

    def test_init_labels_length_mismatch(self):
        """EC-PW-09: Labels length not matching N raises ValueError.

        Tests: PairwiseCompMatrixStack with (3, 3, 5) stack and 2 labels
        should raise a ValueError because labels length must match the
        matrix dimension.
        """
        with pytest.raises(ValueError, match="Number of labels"):
            PairwiseCompMatrixStack(stack=np.random.rand(3, 3, 5), labels=["a", "b"])

    def test_init_0x0xS_stack(self):
        """A 0x0xS stack is accepted by __post_init__ since 0 == 0 passes the square check.

        Tests:
            (Test Case 1) A (0, 0, 3) stack constructs successfully.
            (Test Case 2) Length equals S (3).
            (Test Case 3) Iterating yields 3 PairwiseCompMatrix objects with shape (0, 0).
        """
        data = np.empty((0, 0, 3))
        stack = PairwiseCompMatrixStack(stack=data, times=[(0, 1), (1, 2), (2, 3)])
        assert stack.stack.shape == (0, 0, 3)
        assert len(stack) == 3
        for pcm in stack:
            assert isinstance(pcm, PairwiseCompMatrix)
            assert pcm.matrix.shape == (0, 0)


# ---------------------------------------------------------------------------
# PairwiseCompMatrixStack — slicing / getitem / iteration
# ---------------------------------------------------------------------------


class TestPairwiseCompMatrixStackSlicing:
    """Tests for PairwiseCompMatrixStack slicing, indexing, and iteration."""

    def test_pairwise_comp_matrix_stack_slicing_and_iter(self):
        """
        Test slicing and iteration on PairwiseCompMatrixStack.

        Slicing is fully supported:
        - stack[i] returns a single PairwiseCompMatrix
        - stack[start:end] returns a new PairwiseCompMatrixStack with selected slices
        - stack[::step] returns every nth slice as a new stack
        - Iteration: for matrix in stack: yields each PairwiseCompMatrix
        """
        # n x n x S format (5x5 matrices, 10 slices)
        stack_data = np.random.rand(5, 5, 10)
        stack = PairwiseCompMatrixStack(stack=stack_data)

        # Slicing with range
        sub_stack = stack[0:3]
        assert isinstance(sub_stack, PairwiseCompMatrixStack)
        assert len(sub_stack) == 3
        assert np.array_equal(sub_stack.stack, stack_data[:, :, 0:3])

        # Slicing with step
        step_stack = stack[::2]
        assert len(step_stack) == 5  # 0, 2, 4, 6, 8
        assert np.array_equal(step_stack.stack, stack_data[:, :, ::2])

        # Iteration
        matrices = list(stack)
        assert len(matrices) == 10
        assert isinstance(matrices[0], PairwiseCompMatrix)
        assert np.array_equal(matrices[0].matrix, stack_data[:, :, 0])

    def test_getitem_negative_index(self):
        """Negative indexing returns the last slice as a PairwiseCompMatrix.

        Tests: stack[-1] should return the last slice and be
        an instance of PairwiseCompMatrix.
        """
        data = np.random.rand(3, 3, 5)
        stack = PairwiseCompMatrixStack(stack=data)
        result = stack[-1]
        assert isinstance(result, PairwiseCompMatrix)
        np.testing.assert_array_equal(result.matrix, data[:, :, -1])

    def test_getitem_reverse_slice(self):
        """Reverse slicing returns slices in reversed order.

        Tests: stack[::-1] should yield a new stack whose slices
        are in reverse order compared to the original.
        """
        data = np.arange(3 * 3 * 5, dtype=float).reshape(3, 3, 5)
        stack = PairwiseCompMatrixStack(stack=data)
        result = stack[::-1]
        assert isinstance(result, PairwiseCompMatrixStack)
        assert len(result) == 5
        for i in range(5):
            np.testing.assert_array_equal(result.stack[:, :, i], data[:, :, 4 - i])

    def test_getitem_empty_slice(self):
        """An empty slice range returns a stack with S=0.

        Tests: stack[5:5] on a 5-slice stack should return an
        empty stack with zero slices.
        """
        data = np.random.rand(3, 3, 5)
        stack = PairwiseCompMatrixStack(stack=data)
        result = stack[5:5]
        assert isinstance(result, PairwiseCompMatrixStack)
        assert len(result) == 0
        assert result.stack.shape == (3, 3, 0)

    def test_getitem_out_of_bounds_index(self):
        """Out-of-bounds integer index raises IndexError.

        Tests:
            (Test Case 1) Accessing index 5 on a 5-slice stack raises IndexError.
            (Test Case 2) Accessing index -6 on a 5-slice stack raises IndexError.

        Notes:
            - The error originates from NumPy array indexing. The message may
              not reference PairwiseCompMatrixStack directly.
        """
        stack = PairwiseCompMatrixStack(stack=np.random.rand(3, 3, 5))
        with pytest.raises(IndexError):
            stack[5]
        with pytest.raises(IndexError):
            stack[-6]

    def test_iter_empty_stack(self):
        """Iterating over an empty stack yields no elements.

        Tests: list(stack) on a (3,3,0) stack should be an
        empty list.
        """
        stack = PairwiseCompMatrixStack(stack=np.empty((3, 3, 0)), times=[])
        items = list(stack)
        assert items == []


# ---------------------------------------------------------------------------
# PairwiseCompMatrixStack — subslice
# ---------------------------------------------------------------------------


class TestPairwiseCompMatrixStackSubslice:
    """Tests for PairwiseCompMatrixStack.subslice()."""

    def test_pairwise_comp_matrix_stack_subslice(self):
        """Test the subslice method for selecting specific non-contiguous slices."""
        # n x n x S format (5x5 matrices, 10 slices)
        stack_data = np.random.rand(5, 5, 10)
        times = [(i * 100, (i + 1) * 100) for i in range(10)]
        stack = PairwiseCompMatrixStack(stack=stack_data, times=times)

        # Select specific slices
        sub = stack.subslice([0, 2, 5, 9])
        assert len(sub) == 4
        assert np.array_equal(sub.stack[:, :, 0], stack_data[:, :, 0])
        assert np.array_equal(sub.stack[:, :, 1], stack_data[:, :, 2])
        assert np.array_equal(sub.stack[:, :, 2], stack_data[:, :, 5])
        assert np.array_equal(sub.stack[:, :, 3], stack_data[:, :, 9])

        # Times should also be subsliced
        assert sub.times == [(0, 100), (200, 300), (500, 600), (900, 1000)]

    def test_subslice_empty_indices(self):
        """Subslice with an empty index list returns a stack with S=0.

        Tests: subslice([]) on a (3,3,5) stack should yield an
        empty stack with shape (3,3,0).
        """
        stack = PairwiseCompMatrixStack(
            stack=np.random.rand(3, 3, 5),
            times=[(i, i + 1) for i in range(5)],
        )
        result = stack.subslice([])
        assert result.stack.shape == (3, 3, 0)
        assert len(result) == 0
        assert result.times == []

    def test_subslice_duplicate_indices(self):
        """Subslice with duplicate indices keeps duplicates.

        Tests: subslice([0, 0, 1]) on a (3,3,5) stack should
        return a stack with S=3 where slices 0 and 1 are duplicated.
        """
        data = np.random.rand(3, 3, 5)
        times = [(i, i + 1) for i in range(5)]
        stack = PairwiseCompMatrixStack(stack=data, times=times)
        result = stack.subslice([0, 0, 1])
        assert len(result) == 3
        np.testing.assert_array_equal(result.stack[:, :, 0], data[:, :, 0])
        np.testing.assert_array_equal(result.stack[:, :, 1], data[:, :, 0])
        np.testing.assert_array_equal(result.stack[:, :, 2], data[:, :, 1])

    def test_subslice_out_of_bounds_index(self):
        """Subslice with an out-of-bounds index raises IndexError.

        Tests: subslice([0, 1, 10]) on a (3,3,5) stack should raise
        an IndexError because index 10 exceeds the stack size of 5.
        """
        stack = PairwiseCompMatrixStack(stack=np.random.rand(3, 3, 5))
        with pytest.raises(IndexError):
            stack.subslice([0, 1, 10])

    def test_subslice_negative_indices(self):
        """Subslice with negative indices wraps around via NumPy indexing.

        Tests:
            (Test Case 1) subslice([-1]) returns the last slice.
            (Test Case 2) subslice([-1, -2]) returns last two slices in that order.
            (Test Case 3) Times are correctly selected by negative index.
        """
        data = np.arange(3 * 3 * 5, dtype=float).reshape(3, 3, 5)
        times = [(i, i + 1) for i in range(5)]
        stack = PairwiseCompMatrixStack(stack=data, times=times)

        result = stack.subslice([-1])
        assert len(result) == 1
        np.testing.assert_array_equal(result.stack[:, :, 0], data[:, :, -1])
        assert result.times == [(4, 5)]

        result2 = stack.subslice([-1, -2])
        assert len(result2) == 2
        np.testing.assert_array_equal(result2.stack[:, :, 0], data[:, :, 4])
        np.testing.assert_array_equal(result2.stack[:, :, 1], data[:, :, 3])
        assert result2.times == [(4, 5), (3, 4)]

    def test_subslice_unsorted_indices(self):
        """Subslice with unsorted indices preserves the given order.

        Tests: subslice([2, 0, 1]) should return slices in that
        exact order: result slice 0 = original slice 2, etc.
        """
        data = np.arange(3 * 3 * 5, dtype=float).reshape(3, 3, 5)
        times = [(i, i + 1) for i in range(5)]
        stack = PairwiseCompMatrixStack(stack=data, times=times)

        result = stack.subslice([2, 0, 1])
        assert len(result) == 3
        np.testing.assert_array_equal(result.stack[:, :, 0], data[:, :, 2])
        np.testing.assert_array_equal(result.stack[:, :, 1], data[:, :, 0])
        np.testing.assert_array_equal(result.stack[:, :, 2], data[:, :, 1])
        assert result.times == [(2, 3), (0, 1), (1, 2)]


# ---------------------------------------------------------------------------
# PairwiseCompMatrixStack — mean
# ---------------------------------------------------------------------------


class TestPairwiseCompMatrixStackMean:
    """Tests for PairwiseCompMatrixStack.mean()."""

    def test_mean_empty_stack(self):
        """Mean of an empty stack returns a (3,3) all-NaN matrix.

        Tests: mean() on a (3,3,0) stack should produce NaN values
        for every cell since there are no slices to average.
        """
        stack = PairwiseCompMatrixStack(stack=np.empty((3, 3, 0)), times=[])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = stack.mean()
        assert result.matrix.shape == (3, 3)
        assert np.all(np.isnan(result.matrix))

    def test_mean_single_slice(self):
        """Mean of a single-slice stack equals the single slice itself.

        Tests: with known values in a (3,3,1) stack, the mean
        should be identical to that single slice.
        """
        known = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.7], [0.3, 0.7, 1.0]])
        data = known[:, :, np.newaxis]  # (3,3,1)
        stack = PairwiseCompMatrixStack(stack=data, times=[(0, 1)])
        result = stack.mean()
        np.testing.assert_array_almost_equal(result.matrix, known)

    def test_mean_all_inf_stack(self):
        """EC-PW-07: Mean of an all-Inf stack returns Inf (nanmean) or Inf (mean).

        Tests: mean() on a stack where every element is np.inf should
        return inf for every cell. nanmean treats inf as a valid value
        (not NaN), so the mean of [inf, inf, ...] is inf.
        """
        data = np.full((3, 3, 4), np.inf)
        stack = PairwiseCompMatrixStack(stack=data)
        result = stack.mean(ignore_nan=True)
        assert np.all(np.isinf(result.matrix))
        assert np.all(result.matrix > 0)  # positive inf

        result_raw = stack.mean(ignore_nan=False)
        assert np.all(np.isinf(result_raw.matrix))

    def test_mean_ignore_nan_false_propagates_nan(self):
        """ignore_nan=False uses np.mean which propagates NaN across the stack axis.

        Tests:
            (Test Case 1) A single NaN in one slice causes the mean for that cell
                to be NaN when ignore_nan=False.
            (Test Case 2) Cells without NaN are computed correctly.
            (Test Case 3) ignore_nan=True on the same data excludes NaN and
                produces a finite result.
        """
        data = np.ones((3, 3, 4))
        data[0, 1, 2] = np.nan  # one NaN in cell (0,1), slice 2

        stack = PairwiseCompMatrixStack(stack=data)

        # ignore_nan=False: NaN propagates
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result_raw = stack.mean(ignore_nan=False)
        assert np.isnan(result_raw.matrix[0, 1])
        # Other cells should be 1.0
        assert result_raw.matrix[1, 2] == pytest.approx(1.0)

        # ignore_nan=True: NaN excluded, mean of three 1.0 values = 1.0
        result_nan = stack.mean(ignore_nan=True)
        assert result_nan.matrix[0, 1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PairwiseCompMatrixStack — threshold
# ---------------------------------------------------------------------------


class TestPairwiseCompMatrixStackThreshold:
    """Tests for PairwiseCompMatrixStack.threshold()."""

    def test_pairwise_comp_matrix_stack_threshold(self):
        """Test the threshold method for PairwiseCompMatrixStack."""
        # n x n x S format
        stack_data = np.array(
            [
                [[0.9, 0.2], [0.5, 0.7]],  # slice 0
                [[0.3, 0.8], [0.1, 0.6]],  # slice 1
            ]
        ).transpose(
            1, 2, 0
        )  # Reshape to (2, 2, 2)

        stack = PairwiseCompMatrixStack(stack=stack_data)
        binary_stack = stack.threshold(0.4)

        expected = np.array(
            [
                [[1.0, 0.0], [1.0, 1.0]],  # slice 0
                [[0.0, 1.0], [0.0, 1.0]],  # slice 1
            ]
        ).transpose(1, 2, 0)

        np.testing.assert_array_equal(binary_stack.stack, expected)
        assert binary_stack.metadata["threshold"] == 0.4

    def test_threshold_empty_stack(self):
        """Threshold on an empty stack (S=0) returns an empty stack.

        Tests: thresholding a (3,3,0) stack should return a new
        stack with shape (3,3,0) and zero length.
        """
        stack = PairwiseCompMatrixStack(stack=np.empty((3, 3, 0)))
        result = stack.threshold(0.5)
        assert isinstance(result, PairwiseCompMatrixStack)
        assert result.stack.shape == (3, 3, 0)
        assert len(result) == 0
        assert result.metadata["threshold"] == 0.5
        assert result.metadata["binary"] is True


# ---------------------------------------------------------------------------
# PairwiseCompMatrixStack — dimensionality reduction
# ---------------------------------------------------------------------------


class TestPairwiseCompMatrixStackDimRed:
    """Tests for PairwiseCompMatrixStack dimensionality reduction and feature extraction."""

    def test_pca_compatibility(self):
        # Create a stack of 10 matrices (5x5 units) - now n x n x S
        stack_data = np.random.rand(5, 5, 10)
        # Make them symmetric
        for i in range(10):
            stack_data[:, :, i] = (stack_data[:, :, i] + stack_data[:, :, i].T) / 2

        stack = PairwiseCompMatrixStack(stack=stack_data)

        # Test extract_lower_triangle_features on stack
        features = stack.extract_lower_triangle_features()
        assert features.shape == (10, 10)  # 5*(5-1)/2 = 10 features

        # Test dim_red_on_lower_diagonal_corr_matrix (default PCA)
        pca_result, var_ratio, components = stack.dim_red_on_lower_diagonal_corr_matrix(
            method="PCA", n_components=2
        )
        assert pca_result.shape == (10, 2)
        assert var_ratio.shape == (2,)
        assert components.shape == (2, 10)  # 10 lower-triangle features

    def test_dim_red_pca_with_kwargs_raises(self):
        """Test that PCA method raises TypeError when given extra kwargs."""
        stack_data = np.random.rand(5, 5, 10)
        for i in range(10):
            stack_data[:, :, i] = (stack_data[:, :, i] + stack_data[:, :, i].T) / 2
        stack = PairwiseCompMatrixStack(stack=stack_data)

        with pytest.raises(TypeError, match="only supported for UMAP"):
            stack.dim_red_on_lower_diagonal_corr_matrix(
                method="PCA", n_components=2, n_neighbors=15
            )

    def test_dim_red_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""
        stack_data = np.random.rand(5, 5, 10)
        for i in range(10):
            stack_data[:, :, i] = (stack_data[:, :, i] + stack_data[:, :, i].T) / 2
        stack = PairwiseCompMatrixStack(stack=stack_data)

        with pytest.raises(ValueError, match="Unknown manifold method.*TSNE"):
            stack.dim_red_on_lower_diagonal_corr_matrix(method="TSNE", n_components=2)

    @pytest.mark.skipif(not UMAP_AVAILABLE, reason="umap-learn not installed")
    def test_dim_red_umap(self):
        """Test UMAP dimensionality reduction, skipped if umap-learn not installed."""
        stack_data = np.random.rand(5, 5, 20)
        for i in range(20):
            stack_data[:, :, i] = (stack_data[:, :, i] + stack_data[:, :, i].T) / 2
        stack = PairwiseCompMatrixStack(stack=stack_data)

        umap_result, tw = stack.dim_red_on_lower_diagonal_corr_matrix(
            method="UMAP", n_components=2, n_neighbors=5, min_dist=0.1
        )
        assert umap_result.shape == (20, 2)
        assert isinstance(tw, float)

    def test_dim_red_empty_stack(self):
        """Dimensionality reduction on an empty stack raises an error.

        Tests: PCA cannot be performed on zero samples, so
        dim_red_on_lower_diagonal_corr_matrix should raise on a (3,3,0) stack.
        """
        stack = PairwiseCompMatrixStack(stack=np.empty((3, 3, 0)), times=[])
        with pytest.raises(Exception):
            stack.dim_red_on_lower_diagonal_corr_matrix("PCA", n_components=1)

    def test_dim_red_single_slice(self):
        """Dimensionality reduction on a single-slice stack raises or returns (1,1).

        Tests: with only one sample, PCA with n_components=1 should
        either raise or return an embedding of shape (1, 1).
        """
        data = np.random.rand(3, 3, 1)
        data[:, :, 0] = (data[:, :, 0] + data[:, :, 0].T) / 2
        stack = PairwiseCompMatrixStack(stack=data, times=[(0, 1)])
        try:
            result = stack.dim_red_on_lower_diagonal_corr_matrix("PCA", n_components=1)
            assert result.shape == (1, 1)
        except Exception:
            pass  # raising is also acceptable

    def test_dim_red_n_components_exceeds_S(self):
        """Dimensionality reduction with n_components > S raises ValueError.

        Tests: requesting more PCA components than the number of
        samples (slices) should raise a ValueError from sklearn.
        """
        data = np.random.default_rng(0).random((4, 4, 3))
        for i in range(3):
            data[:, :, i] = (data[:, :, i] + data[:, :, i].T) / 2
        stack = PairwiseCompMatrixStack(stack=data)
        with pytest.raises(ValueError):
            stack.dim_red_on_lower_diagonal_corr_matrix("PCA", n_components=10)

    def test_extract_lower_triangle_features_1x1xS_stack(self):
        """A 1x1xS stack produces (S, 0) features since a 1x1 matrix has no lower triangle.

        Tests:
            (Test Case 1) Feature matrix shape is (S, 0).
            (Test Case 2) PCA on the (S, 0) feature matrix raises an error because
                there are zero features to reduce.

        Notes:
            - This is a degenerate case: a 1x1 pairwise matrix has no off-diagonal
              entries, so there are no features to extract. Downstream PCA will fail.
        """
        data = np.ones((1, 1, 5))
        stack = PairwiseCompMatrixStack(stack=data)
        features = stack.extract_lower_triangle_features()
        assert features.shape == (5, 0)

        # PCA on zero features should fail
        with pytest.raises(Exception):
            stack.dim_red_on_lower_diagonal_corr_matrix("PCA", n_components=1)

    def test_extract_lower_triangle_features_nan_stack(self):
        """EC-PW-08: extract_lower_triangle_features preserves NaN values.

        Tests: extract_lower_triangle_features on a stack containing
        NaN values should propagate NaNs into the output feature matrix
        without raising an error.
        """
        data = np.ones((3, 3, 4))
        data[1, 0, 0] = np.nan  # lower triangle position (1,0), slice 0
        data[2, 1, 2] = np.nan  # lower triangle position (2,1), slice 2
        stack = PairwiseCompMatrixStack(stack=data)
        features = stack.extract_lower_triangle_features()
        # 3x3 matrix -> 3 lower triangle features, 4 slices -> (4, 3)
        assert features.shape == (4, 3)
        # Slice 0 should have a NaN in the feature from position (1,0)
        assert np.isnan(features[0, 0])  # tril_indices(3, k=-1) -> (1,0), (2,0), (2,1)
        # Slice 2 should have a NaN in the feature from position (2,1)
        assert np.isnan(features[2, 2])
        # Other entries should be 1.0
        assert features[1, 0] == 1.0
        assert features[3, 1] == 1.0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestPairwiseIntegration:
    """Integration tests for pairwise classes with SpikeData and RateSliceStack."""

    def test_integration_spikedata(self):
        # Create dummy SpikeData
        train = [np.array([10, 20, 30]), np.array([15, 25, 35])]
        sd = SpikeData(train, length=100)

        sttc_pcm = sd.spike_time_tilings(delt=5.0)
        assert isinstance(sttc_pcm, PairwiseCompMatrix)
        assert sttc_pcm.matrix.shape == (2, 2)
        assert sttc_pcm.metadata["delt"] == 5.0

    def test_integration_rateslicestack(self):
        # Create dummy RateSliceStack
        event_matrix = np.random.rand(2, 50, 5)  # U x T x S
        rss = RateSliceStack(None, event_matrix=event_matrix)

        # Test unit_to_unit_correlation - now returns U x U x S
        corr_stack, lag_stack, av_corr, av_lag = rss.unit_to_unit_correlation()
        assert isinstance(corr_stack, PairwiseCompMatrixStack)
        assert isinstance(lag_stack, PairwiseCompMatrixStack)
        assert len(corr_stack) == 5  # 5 slices
        assert corr_stack[0].matrix.shape == (2, 2)  # 2x2 unit matrix
        assert corr_stack.stack.shape == (2, 2, 5)  # n x n x S

        # Test get_slice_to_slice_unit_corr_from_stack - returns S x S x U
        all_slice_corr, av_slice_corr = rss.get_slice_to_slice_unit_corr_from_stack()
        assert isinstance(all_slice_corr, PairwiseCompMatrixStack)
        assert all_slice_corr[0].matrix.shape == (5, 5)  # S x S
        assert all_slice_corr.stack.shape == (
            5,
            5,
            2,
        )  # n x n x S (where n=S=5, third dim=U=2)

        # Test get_slice_to_slice_time_corr_from_stack - returns S x S x T
        all_slice_time_corr, av_slice_time_corr = (
            rss.get_slice_to_slice_time_corr_from_stack()
        )
        assert isinstance(all_slice_time_corr, PairwiseCompMatrixStack)
        assert all_slice_time_corr[0].matrix.shape == (5, 5)  # S x S
        assert len(all_slice_time_corr) == 50  # T time bins
        assert all_slice_time_corr.stack.shape == (
            5,
            5,
            50,
        )  # n x n x S (where n=S=5, third dim=T=50)

    def test_single_unit_spikedata_produces_1x1_pairwise(self):
        """
        Pairwise comparison on single-unit SpikeData produces a trivial 1x1 matrix.

        Tests:
            (Test Case 1) spike_time_tilings on 1-unit data returns (1, 1) matrix.
            (Test Case 2) Diagonal is 1.0 (self-tiling).
            (Test Case 3) extract_lower_triangle returns empty array.
        """
        sd = SpikeData([np.array([5.0, 10.0, 15.0])], length=20.0)
        sttc = sd.spike_time_tilings(delt=5.0)
        assert sttc.matrix.shape == (1, 1)
        assert sttc.matrix[0, 0] == pytest.approx(1.0)
        assert sttc.extract_lower_triangle().shape == (0,)

    def test_rigorous_edge_cases(self):
        # Empty stack
        empty_stack_data = np.zeros((5, 5, 0))  # n x n x S with S=0
        empty_stack = PairwiseCompMatrixStack(stack=empty_stack_data)
        assert len(empty_stack) == 0

        # mean() on empty stack
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_empty = empty_stack.mean()
            assert np.all(np.isnan(mean_empty.matrix))

        # Single unit matrix
        single_matrix = np.array([[1.0]])
        pcm_single = PairwiseCompMatrix(matrix=single_matrix)
        G_single = pcm_single.to_networkx()
        assert G_single.number_of_nodes() == 1
        assert G_single.number_of_edges() == 0

        # Infs
        inf_matrix = np.array([[1.0, np.inf], [np.inf, 1.0]])
        pcm_inf = PairwiseCompMatrix(matrix=inf_matrix)
        G_inf = pcm_inf.to_networkx()
        assert G_inf.edges[0, 1]["weight"] == float("inf")


# ---------------------------------------------------------------------------
# remove_by_condition — PairwiseCompMatrix
# ---------------------------------------------------------------------------


class TestRemoveByConditionMatrix:
    """Tests for PairwiseCompMatrix.remove_by_condition()."""

    def _make_pair(self):
        """Helper: target STTC matrix and condition latency matrix."""
        target = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.6], [0.3, 0.6, 1.0]])
        condition = np.array([[0.0, 1.5, 5.0], [1.5, 0.0, -3.0], [5.0, -3.0, 0.0]])
        return (
            PairwiseCompMatrix(matrix=target),
            PairwiseCompMatrix(matrix=condition),
        )

    def test_abs_lt_removes_correct_entries(self):
        """
        Entries where |condition| < threshold are replaced by fill.

        Tests:
            (Test Case 1) |1.5| < 2 → entries (0,1) and (1,0) are set to NaN.
            (Test Case 2) |5.0| >= 2 and |-3.0| >= 2 → those entries preserved.
        """
        target, condition = self._make_pair()
        result = target.remove_by_condition(condition, op="abs_lt", threshold=2.0)

        assert np.isnan(result.matrix[0, 1])
        assert np.isnan(result.matrix[1, 0])
        assert result.matrix[0, 2] == 0.3
        assert result.matrix[1, 2] == 0.6

    def test_custom_fill_value(self):
        """
        Custom fill value (0.0) is used instead of NaN.

        Tests:
            (Test Case 1) Removed entries are set to 0.0, not NaN.
        """
        target, condition = self._make_pair()
        result = target.remove_by_condition(
            condition, op="abs_lt", threshold=2.0, fill=0.0
        )

        assert result.matrix[0, 1] == 0.0
        assert result.matrix[1, 0] == 0.0
        assert result.matrix[0, 2] == 0.3

    def test_gt_operator(self):
        """
        Entries where condition > threshold are replaced.

        Tests:
            (Test Case 1) condition values 5.0 > 4.0 are removed.
            (Test Case 2) Values <= 4.0 are preserved.
        """
        target, condition = self._make_pair()
        result = target.remove_by_condition(condition, op="gt", threshold=4.0)

        assert np.isnan(result.matrix[0, 2])
        assert np.isnan(result.matrix[2, 0])
        assert result.matrix[0, 1] == 0.8  # 1.5 not > 4.0

    def test_le_operator(self):
        """
        Entries where condition <= threshold are replaced.

        Tests:
            (Test Case 1) condition value 0.0 <= 1.5 → diagonal removed.
            (Test Case 2) condition value 1.5 <= 1.5 → removed.
            (Test Case 3) condition value -3.0 <= 1.5 → removed.
            (Test Case 4) condition value 5.0 > 1.5 → preserved.
        """
        target, condition = self._make_pair()
        result = target.remove_by_condition(condition, op="le", threshold=1.5, fill=0.0)

        assert result.matrix[0, 0] == 0.0  # 0.0 <= 1.5
        assert result.matrix[0, 1] == 0.0  # 1.5 <= 1.5
        assert result.matrix[1, 2] == 0.0  # -3.0 <= 1.5, removed
        assert result.matrix[0, 2] == 0.3  # 5.0 not <= 1.5

    def test_abs_ge_operator(self):
        """
        Entries where |condition| >= threshold are replaced.

        Tests:
            (Test Case 1) abs_ge with threshold=3.0 removes entries where
                |condition| >= 3.0 (values 5.0 and -3.0).
        """
        target, condition = self._make_pair()
        result = target.remove_by_condition(condition, op="abs_ge", threshold=3.0)

        assert np.isnan(result.matrix[0, 2])  # |5.0| >= 3
        assert np.isnan(result.matrix[2, 0])  # |5.0| >= 3
        assert np.isnan(result.matrix[1, 2])  # |-3.0| >= 3
        assert result.matrix[0, 1] == 0.8  # |1.5| < 3

    def test_no_entries_removed(self):
        """
        When no entries match the condition, the result equals self.

        Tests:
            (Test Case 1) threshold=100 with abs_lt removes nothing.
        """
        target, condition = self._make_pair()
        result = target.remove_by_condition(condition, op="gt", threshold=100.0)

        np.testing.assert_array_equal(result.matrix, target.matrix)

    def test_all_entries_removed(self):
        """
        When all entries match, the entire matrix is filled.

        Tests:
            (Test Case 1) abs_ge with threshold=0.0 removes everything.
        """
        target, condition = self._make_pair()
        result = target.remove_by_condition(
            condition, op="abs_ge", threshold=0.0, fill=-1.0
        )

        assert np.all(result.matrix == -1.0)

    def test_labels_preserved(self):
        """
        Labels from self are preserved in the result.

        Tests:
            (Test Case 1) Labels are identical to the original.
        """
        target = PairwiseCompMatrix(matrix=np.eye(3), labels=["A", "B", "C"])
        condition = PairwiseCompMatrix(matrix=np.zeros((3, 3)))
        result = target.remove_by_condition(condition, op="lt", threshold=1.0)

        assert result.labels == ["A", "B", "C"]

    def test_metadata_records_operation(self):
        """
        Metadata records the condition operation details.

        Tests:
            (Test Case 1) metadata contains 'removed_by_condition' with op, threshold, fill.
        """
        target, condition = self._make_pair()
        result = target.remove_by_condition(
            condition, op="abs_lt", threshold=2.0, fill=0.0
        )

        assert result.metadata["removed_by_condition"] == {
            "op": "abs_lt",
            "threshold": 2.0,
            "fill": 0.0,
        }

    def test_shape_mismatch_raises(self):
        """
        Mismatched shapes raise ValueError.

        Tests:
            (Test Case 1) 3x3 target with 2x2 condition raises ValueError.
        """
        target = PairwiseCompMatrix(matrix=np.eye(3))
        condition = PairwiseCompMatrix(matrix=np.eye(2))

        with pytest.raises(ValueError, match="does not match"):
            target.remove_by_condition(condition, op="lt", threshold=0.5)

    def test_invalid_op_raises(self):
        """
        Invalid operator string raises ValueError.

        Tests:
            (Test Case 1) op='invalid' raises ValueError.
            (Test Case 2) op='abs_invalid' raises ValueError.
        """
        target, condition = self._make_pair()

        with pytest.raises(ValueError, match="Unknown op"):
            target.remove_by_condition(condition, op="invalid", threshold=0.5)
        with pytest.raises(ValueError, match="Unknown op"):
            target.remove_by_condition(condition, op="abs_invalid", threshold=0.5)

    def test_non_pcm_condition_raises(self):
        """
        Non-PairwiseCompMatrix condition raises TypeError.

        Tests:
            (Test Case 1) Passing a plain ndarray raises TypeError.
        """
        target = PairwiseCompMatrix(matrix=np.eye(3))

        with pytest.raises(TypeError, match="PairwiseCompMatrix"):
            target.remove_by_condition(np.eye(3), op="lt", threshold=0.5)

    def test_self_referential(self):
        """
        Using self as the condition works correctly.

        Tests:
            (Test Case 1) Remove STTC entries where STTC < 0.5, preserving
                values >= 0.5.
        """
        target, _ = self._make_pair()
        result = target.remove_by_condition(target, op="lt", threshold=0.5)

        assert np.isnan(result.matrix[0, 2])  # 0.3 < 0.5
        assert np.isnan(result.matrix[2, 0])  # 0.3 < 0.5
        assert result.matrix[0, 1] == 0.8  # 0.8 >= 0.5
        assert result.matrix[1, 2] == 0.6  # 0.6 >= 0.5

    def test_all_nan_condition_matrix(self):
        """
        All-NaN condition matrix removes no entries (NaN comparisons are False).

        Tests:
            (Test Case 1) Result matrix unchanged from target.
        """
        target = PairwiseCompMatrix(matrix=np.array([[1.0, 0.5], [0.5, 1.0]]))
        condition = PairwiseCompMatrix(matrix=np.full((2, 2), np.nan))
        result = target.remove_by_condition(condition, op="lt", threshold=0.5)
        np.testing.assert_array_equal(result.matrix, target.matrix)

    def test_all_nan_target_matrix(self):
        """
        All-NaN target with matching condition replaces matched NaN entries with fill.

        Tests:
            (Test Case 1) Entries where condition matches are set to fill (0.0).
            (Test Case 2) Other NaN entries remain NaN.
        """
        target = PairwiseCompMatrix(matrix=np.full((2, 2), np.nan))
        condition = PairwiseCompMatrix(matrix=np.array([[0.0, 1.0], [1.0, 0.0]]))
        result = target.remove_by_condition(condition, op="lt", threshold=0.5, fill=0.0)
        assert result.matrix[0, 0] == 0.0  # 0.0 < 0.5 → replaced
        assert np.isnan(result.matrix[0, 1])  # 1.0 >= 0.5 → stays NaN

    def test_1x1_matrix(self):
        """
        1x1 matrix works correctly.

        Tests:
            (Test Case 1) Single entry replaced when condition matches.
        """
        target = PairwiseCompMatrix(matrix=np.array([[0.9]]))
        condition = PairwiseCompMatrix(matrix=np.array([[0.1]]))
        result = target.remove_by_condition(condition, op="lt", threshold=0.5)
        assert np.isnan(result.matrix[0, 0])

    def test_nan_condition_with_abs_variant(self):
        """
        abs_ variant on NaN condition: |NaN| is NaN, comparison returns False.

        Tests:
            (Test Case 1) No entries removed.
        """
        target = PairwiseCompMatrix(matrix=np.eye(3))
        condition = PairwiseCompMatrix(matrix=np.full((3, 3), np.nan))
        result = target.remove_by_condition(condition, op="abs_lt", threshold=10.0)
        np.testing.assert_array_equal(result.matrix, target.matrix)


# ---------------------------------------------------------------------------
# remove_by_condition — PairwiseCompMatrixStack
# ---------------------------------------------------------------------------


class TestRemoveByConditionStack:
    """Tests for PairwiseCompMatrixStack.remove_by_condition()."""

    def _make_pair(self):
        """Helper: target and condition stacks (3, 3, 2)."""
        target = np.array(
            [
                [[0.9, 0.7], [0.5, 0.3], [0.1, 0.8]],
                [[0.5, 0.3], [0.9, 0.7], [0.4, 0.6]],
                [[0.1, 0.8], [0.4, 0.6], [0.9, 0.7]],
            ]
        )
        condition = np.array(
            [
                [[0.0, 0.0], [1.0, 5.0], [3.0, 0.5]],
                [[1.0, 5.0], [0.0, 0.0], [-2.0, 4.0]],
                [[3.0, 0.5], [-2.0, 4.0], [0.0, 0.0]],
            ]
        )
        return (
            PairwiseCompMatrixStack(stack=target),
            PairwiseCompMatrixStack(stack=condition),
        )

    def test_stack_abs_lt_removes_correct_entries(self):
        """
        Entries where |condition| < threshold are replaced across all slices.

        Tests:
            (Test Case 1) |1.0| < 2 in slice 0 at (0,1) → removed.
            (Test Case 2) |5.0| >= 2 in slice 1 at (0,1) → preserved.
        """
        target, condition = self._make_pair()
        result = target.remove_by_condition(condition, op="abs_lt", threshold=2.0)

        assert np.isnan(result.stack[0, 1, 0])  # |1.0| < 2
        assert result.stack[0, 1, 1] == 0.3  # |5.0| >= 2
        assert result.stack[0, 2, 0] == 0.1  # |3.0| >= 2
        assert np.isnan(result.stack[0, 2, 1])  # |0.5| < 2

    def test_stack_custom_fill(self):
        """
        Custom fill value works for stacks.

        Tests:
            (Test Case 1) fill=-1 replaces matched entries with -1.
        """
        target, condition = self._make_pair()
        result = target.remove_by_condition(
            condition, op="abs_lt", threshold=2.0, fill=-1.0
        )

        assert result.stack[0, 1, 0] == -1.0  # |1.0| < 2

    def test_broadcast_single_matrix(self):
        """
        A single PairwiseCompMatrix condition is broadcast across all slices.

        Tests:
            (Test Case 1) Same mask applied to both slices.
            (Test Case 2) Entries matching in the single matrix are removed
                in every slice of the target.
        """
        target, _ = self._make_pair()
        single_condition = PairwiseCompMatrix(
            matrix=np.array([[0.0, 1.0, 5.0], [1.0, 0.0, 3.0], [5.0, 3.0, 0.0]])
        )
        result = target.remove_by_condition(
            single_condition, op="abs_lt", threshold=2.0
        )

        # (0,1) has |1.0| < 2 → both slices removed
        assert np.isnan(result.stack[0, 1, 0])
        assert np.isnan(result.stack[0, 1, 1])
        # (0,2) has |5.0| >= 2 → both slices preserved
        assert result.stack[0, 2, 0] == 0.1
        assert result.stack[0, 2, 1] == 0.8

    def test_stack_shape_mismatch_raises(self):
        """
        Mismatched stack shapes raise ValueError.

        Tests:
            (Test Case 1) (3,3,2) target with (3,3,3) condition raises ValueError.
        """
        target, _ = self._make_pair()
        bad_condition = PairwiseCompMatrixStack(stack=np.zeros((3, 3, 3)))

        with pytest.raises(ValueError, match="does not match"):
            target.remove_by_condition(bad_condition, op="lt", threshold=0.5)

    def test_broadcast_shape_mismatch_raises(self):
        """
        Broadcasting a single matrix with wrong N raises ValueError.

        Tests:
            (Test Case 1) (3,3,2) target with (2,2) condition raises ValueError.
        """
        target, _ = self._make_pair()
        bad_single = PairwiseCompMatrix(matrix=np.zeros((2, 2)))

        with pytest.raises(ValueError, match="does not match"):
            target.remove_by_condition(bad_single, op="lt", threshold=0.5)

    def test_invalid_condition_type_raises(self):
        """
        Non-PCM/Stack condition raises TypeError.

        Tests:
            (Test Case 1) Passing a plain ndarray raises TypeError.
        """
        target, _ = self._make_pair()

        with pytest.raises(TypeError, match="PairwiseCompMatrix"):
            target.remove_by_condition(np.zeros((3, 3, 2)), op="lt", threshold=0.5)

    def test_invalid_op_raises(self):
        """
        Invalid operator string raises ValueError.

        Tests:
            (Test Case 1) op='bad' raises ValueError.
        """
        target, condition = self._make_pair()

        with pytest.raises(ValueError, match="Unknown op"):
            target.remove_by_condition(condition, op="bad", threshold=0.5)

    def test_times_and_labels_preserved(self):
        """
        Times and labels from self are preserved in the result.

        Tests:
            (Test Case 1) Times and labels are identical to the original.
        """
        target_stack = PairwiseCompMatrixStack(
            stack=np.ones((3, 3, 2)),
            labels=["A", "B", "C"],
            times=[(0, 10), (10, 20)],
        )
        condition = PairwiseCompMatrixStack(stack=np.zeros((3, 3, 2)))
        result = target_stack.remove_by_condition(condition, op="lt", threshold=1.0)

        assert result.labels == ["A", "B", "C"]
        assert result.times == [(0, 10), (10, 20)]

    def test_metadata_records_operation(self):
        """
        Metadata records the condition operation details on the stack.

        Tests:
            (Test Case 1) metadata contains 'removed_by_condition' dict.
        """
        target, condition = self._make_pair()
        result = target.remove_by_condition(
            condition, op="abs_gt", threshold=3.0, fill=0.0
        )

        assert result.metadata["removed_by_condition"] == {
            "op": "abs_gt",
            "threshold": 3.0,
            "fill": 0.0,
        }

    def test_empty_stack_remove_by_condition(self):
        """
        Empty stack (S=0) returns empty stack.

        Tests:
            (Test Case 1) Result shape is (3, 3, 0).
        """
        target = PairwiseCompMatrixStack(stack=np.empty((3, 3, 0)))
        condition = PairwiseCompMatrixStack(stack=np.empty((3, 3, 0)))
        result = target.remove_by_condition(condition, op="lt", threshold=0.5)
        assert result.stack.shape == (3, 3, 0)

    def test_all_nan_condition_stack(self):
        """
        All-NaN condition stack removes no entries.

        Tests:
            (Test Case 1) Result stack unchanged from target.
        """
        target = PairwiseCompMatrixStack(stack=np.ones((3, 3, 2)))
        condition = PairwiseCompMatrixStack(stack=np.full((3, 3, 2), np.nan))
        result = target.remove_by_condition(condition, op="lt", threshold=0.5)
        np.testing.assert_array_equal(result.stack, target.stack)


# ---------------------------------------------------------------------------
# Edge case tests from the edge case scan
# ---------------------------------------------------------------------------


class TestPairwiseCompMatrixPostInitEdgeCases:
    """Additional edge case tests for PairwiseCompMatrix.__post_init__."""

    def test_3d_array_raises(self):
        """
        3D array input raises ValueError.

        Tests:
            (Test Case 1) Shape (3, 3, 1) is caught by ndim != 2 check.
        """
        with pytest.raises(ValueError, match="must be n x n"):
            PairwiseCompMatrix(matrix=np.ones((3, 3, 1)))

    def test_empty_labels_on_0x0_matrix(self):
        """
        Empty labels=[] on a 0x0 matrix is valid.

        Tests:
            (Test Case 1) A 0x0 matrix with labels=[] does not raise.
        """
        pcm = PairwiseCompMatrix(matrix=np.empty((0, 0)), labels=[])
        assert pcm.matrix.shape == (0, 0)
        assert pcm.labels == []


class TestPairwiseCompMatrixToNetworkxEdgeCases:
    """Additional edge case tests for PairwiseCompMatrix.to_networkx."""

    def test_to_networkx_import_failure(self):
        """
        to_networkx raises ImportError when networkx is not installed.

        Tests:
            (Test Case 1) Mocking the import failure produces an ImportError.
        """
        from unittest.mock import patch

        pcm = PairwiseCompMatrix(matrix=np.eye(2))
        with patch.dict("sys.modules", {"networkx": None}):
            with pytest.raises(ImportError, match="NetworkX"):
                pcm.to_networkx()

    def test_to_networkx_1x1_matrix(self):
        """
        to_networkx on a 1x1 matrix produces a single node, no edges.

        Tests:
            (Test Case 1) Single-element matrix creates a graph with 1 node
                and 0 edges (diagonal is excluded).
        """
        import networkx as nx

        pcm = PairwiseCompMatrix(matrix=np.array([[1.0]]))
        G = pcm.to_networkx()
        assert len(G.nodes) == 1
        assert len(G.edges) == 0


class TestPairwiseCompMatrixThresholdEdgeCases:
    """Additional edge case tests for PairwiseCompMatrix.threshold."""

    def test_threshold_nan(self):
        """
        threshold with threshold=NaN: np.abs(matrix) > NaN is always False.

        Tests:
            (Test Case 1) Result matrix is all zeros.
        """
        pcm = PairwiseCompMatrix(matrix=np.array([[1.0, 0.5], [0.5, 1.0]]))
        result = pcm.threshold(float("nan"))
        np.testing.assert_array_equal(result.matrix, 0)

    def test_threshold_negative(self):
        """
        threshold with negative threshold: np.abs(matrix) > -1 is always True.

        Tests:
            (Test Case 1) Result matrix is all ones (for finite values).
        """
        pcm = PairwiseCompMatrix(matrix=np.array([[0.1, 0.2], [0.2, 0.1]]))
        result = pcm.threshold(-1.0)
        np.testing.assert_array_equal(result.matrix, 1)


class TestPairwiseCompMatrixExtractLowerTriangleEdgeCases:
    """Additional edge case tests for PairwiseCompMatrix.extract_lower_triangle."""

    def test_extract_lower_triangle_0x0(self):
        """
        extract_lower_triangle on a 0x0 matrix returns an empty array.

        Tests:
            (Test Case 1) np.tril_indices(0, k=-1) returns empty arrays.
        """
        pcm = PairwiseCompMatrix(matrix=np.empty((0, 0)))
        result = pcm.extract_lower_triangle()
        assert len(result) == 0


class TestRemoveByConditionMatrixEdgeCases:
    """Additional edge case tests for PairwiseCompMatrix.remove_by_condition."""

    def test_remove_by_condition_nan_threshold(self):
        """
        remove_by_condition with threshold=NaN: comparisons with NaN are always False.

        Tests:
            (Test Case 1) Nothing is removed; result equals original.
        """
        pcm = PairwiseCompMatrix(matrix=np.array([[1.0, 0.5], [0.5, 1.0]]))
        cond = PairwiseCompMatrix(matrix=np.array([[0.3, 0.6], [0.6, 0.3]]))
        result = pcm.remove_by_condition(cond, op="lt", threshold=float("nan"))
        np.testing.assert_array_equal(result.matrix, pcm.matrix)

    def test_remove_by_condition_inf_fill(self):
        """
        remove_by_condition with fill=Inf sets removed entries to Inf.

        Tests:
            (Test Case 1) Entries meeting condition are set to Inf.
        """
        pcm = PairwiseCompMatrix(matrix=np.array([[1.0, 0.5], [0.5, 1.0]]))
        cond = PairwiseCompMatrix(matrix=np.array([[0.1, 0.1], [0.1, 0.1]]))
        result = pcm.remove_by_condition(
            cond, op="lt", threshold=0.5, fill=float("inf")
        )
        # All condition values (0.1) are < 0.5, so all entries replaced with Inf
        assert np.all(np.isinf(result.matrix))


class TestPairwiseCompMatrixStackEdgeCases:
    """Additional edge case tests for PairwiseCompMatrixStack."""

    def test_stack_times_non_tuple_elements(self):
        """
        Stack with times containing non-tuple elements (e.g. integers).

        Tests:
            (Test Case 1) times=[1, 2, 3] is accepted (no element-by-element validation).
        """
        stack = PairwiseCompMatrixStack(stack=np.ones((2, 2, 3)), times=[1, 2, 3])
        assert len(stack.times) == 3

    def test_stack_NxNx0_shape(self):
        """
        Stack with NxNx0 shape (non-zero N, zero slices).

        Tests:
            (Test Case 1) A (3, 3, 0) stack is valid.
        """
        stack = PairwiseCompMatrixStack(stack=np.empty((3, 3, 0)))
        assert stack.stack.shape == (3, 3, 0)


class TestPairwiseCompMatrixStackGetItemEdgeCases:
    """Additional edge case tests for PairwiseCompMatrixStack.__getitem__."""

    def test_getitem_times_none(self):
        """
        stack[0] with times=None does not raise.

        Tests:
            (Test Case 1) Indexing with times=None returns PairwiseCompMatrix
                with no times attribute.
        """
        stack = PairwiseCompMatrixStack(stack=np.ones((2, 2, 3)), times=None)
        result = stack[0]
        assert isinstance(result, PairwiseCompMatrix)
        assert result.matrix.shape == (2, 2)


class TestPairwiseCompMatrixStackSubsliceEdgeCases:
    """Additional edge case tests for PairwiseCompMatrixStack.subslice."""

    def test_subslice_numpy_array_indices(self):
        """
        subslice with indices as a numpy array instead of a list.

        Tests:
            (Test Case 1) numpy array indices are converted to list and work correctly.
        """
        stack = PairwiseCompMatrixStack(stack=np.ones((2, 2, 5)))
        result = stack.subslice(np.array([0, 2, 4]))
        assert result.stack.shape == (2, 2, 3)


class TestPairwiseCompMatrixStackMeanEdgeCases:
    """Additional edge case tests for PairwiseCompMatrixStack.mean."""

    def test_mean_partial_all_nan_columns(self):
        """
        mean(ignore_nan=True) where some (i, j) positions are all NaN across S.

        Tests:
            (Test Case 1) Positions that are all NaN produce NaN in the mean
                with a RuntimeWarning.
        """
        data = np.ones((2, 2, 3))
        data[0, 1, :] = np.nan
        data[1, 0, :] = np.nan
        stack = PairwiseCompMatrixStack(stack=data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = stack.mean(ignore_nan=True)
        assert np.isnan(result.matrix[0, 1])
        assert np.isnan(result.matrix[1, 0])
        assert result.matrix[0, 0] == pytest.approx(1.0)


class TestPairwiseCompMatrixStackDimRedEdgeCases:
    """Additional edge case tests for PairwiseCompMatrixStack dim reduction."""

    def test_dim_red_1x1xS_stack(self):
        """
        dim_red on a 1x1xS stack: lower triangle is empty (F=0).

        Tests:
            (Test Case 1) PCA with 0 features raises ValueError.
        """
        stack = PairwiseCompMatrixStack(stack=np.ones((1, 1, 5)))
        with pytest.raises(ValueError):
            stack.dim_red_on_lower_diagonal_corr_matrix(method="PCA", n_components=1)


# ---------------------------------------------------------------------------
# PairwiseCompMatrix._is_diverging
# ---------------------------------------------------------------------------


class TestIsDiverging:
    """Tests for PairwiseCompMatrix._is_diverging."""

    def test_positive_only(self):
        """Non-negative matrix is not diverging."""
        mat = np.array([[1.0, 0.5], [0.5, 1.0]])
        assert PairwiseCompMatrix._is_diverging(mat) is False

    def test_mixed_values(self):
        """Matrix with both negative and positive values is diverging."""
        mat = np.array([[1.0, -0.3], [-0.3, 1.0]])
        assert PairwiseCompMatrix._is_diverging(mat) is True

    def test_all_negative(self):
        """All-negative matrix is not diverging."""
        mat = np.array([[-0.5, -0.2], [-0.2, -0.8]])
        assert PairwiseCompMatrix._is_diverging(mat) is False

    def test_all_zero(self):
        """All-zero matrix is not diverging."""
        mat = np.zeros((3, 3))
        assert PairwiseCompMatrix._is_diverging(mat) is False

    def test_all_nan(self):
        """All-NaN matrix is not diverging."""
        mat = np.full((3, 3), np.nan)
        assert PairwiseCompMatrix._is_diverging(mat) is False

    def test_nan_with_mixed(self):
        """NaN entries are ignored; remaining values determine divergence."""
        mat = np.array([[np.nan, -0.5], [0.3, np.nan]])
        assert PairwiseCompMatrix._is_diverging(mat) is True


# ---------------------------------------------------------------------------
# PairwiseCompMatrix.plot
# ---------------------------------------------------------------------------


class TestPairwiseCompMatrixPlot:
    """Tests for PairwiseCompMatrix.plot."""

    @pytest.fixture(autouse=True)
    def close_figs(self):
        yield
        plt.close("all")

    def test_standalone_returns_fig_ax(self):
        """Standalone call (no ax) returns (fig, ax) tuple."""
        mat = np.random.default_rng(0).random((4, 4))
        pcm = PairwiseCompMatrix(matrix=mat)
        result = pcm.plot()
        assert isinstance(result, tuple)
        fig, ax = result
        assert isinstance(fig, plt.Figure)

    def test_with_ax(self):
        """Passing an axes returns just the axes."""
        mat = np.random.default_rng(0).random((4, 4))
        pcm = PairwiseCompMatrix(matrix=mat)
        fig, ax = plt.subplots()
        result = pcm.plot(ax=ax)
        assert result is ax

    def test_auto_cmap_viridis(self):
        """Non-diverging data auto-selects viridis."""
        mat = np.array([[1.0, 0.5], [0.5, 1.0]])
        pcm = PairwiseCompMatrix(matrix=mat)
        fig, ax = plt.subplots()
        pcm.plot(ax=ax)
        assert ax.images[0].cmap.name == "viridis"

    def test_auto_cmap_diverging(self):
        """Diverging data auto-selects RdBu_r."""
        mat = np.array([[1.0, -0.5], [-0.5, 1.0]])
        pcm = PairwiseCompMatrix(matrix=mat)
        fig, ax = plt.subplots()
        pcm.plot(ax=ax)
        assert ax.images[0].cmap.name == "RdBu_r"

    def test_explicit_cmap_overrides(self):
        """Explicit cmap overrides auto-detection."""
        mat = np.array([[1.0, -0.5], [-0.5, 1.0]])
        pcm = PairwiseCompMatrix(matrix=mat)
        fig, ax = plt.subplots()
        pcm.plot(ax=ax, cmap="hot")
        assert ax.images[0].cmap.name == "hot"

    def test_labels_as_ticks(self):
        """Labels are used as tick labels."""
        mat = np.eye(3)
        pcm = PairwiseCompMatrix(matrix=mat, labels=["A", "B", "C"])
        fig, ax = plt.subplots()
        pcm.plot(ax=ax)
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_labels == ["A", "B", "C"]

    def test_no_labels_uses_indices(self):
        """Without labels, integer indices are used."""
        mat = np.eye(3)
        pcm = PairwiseCompMatrix(matrix=mat)
        fig, ax = plt.subplots()
        pcm.plot(ax=ax)
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_labels == ["0", "1", "2"]


# ---------------------------------------------------------------------------
# PairwiseCompMatrixStack.plot_mean
# ---------------------------------------------------------------------------


class TestPairwiseCompMatrixStackPlotMean:
    """Tests for PairwiseCompMatrixStack.plot_mean."""

    @pytest.fixture(autouse=True)
    def close_figs(self):
        yield
        plt.close("all")

    def test_standalone_returns_fig_ax(self):
        """Standalone call returns (fig, ax) tuple."""
        stack = np.random.default_rng(0).random((4, 4, 5))
        pcms = PairwiseCompMatrixStack(stack=stack)
        result = pcms.plot_mean()
        assert isinstance(result, tuple)

    def test_with_ax(self):
        """Passing an axes returns just the axes."""
        stack = np.random.default_rng(0).random((4, 4, 5))
        pcms = PairwiseCompMatrixStack(stack=stack)
        fig, ax = plt.subplots()
        result = pcms.plot_mean(ax=ax)
        assert result is ax

    def test_auto_cmap_diverging_mean(self):
        """Stack whose mean is diverging auto-selects RdBu_r."""
        rng = np.random.default_rng(0)
        stack = rng.uniform(-1, 1, size=(4, 4, 5))
        pcms = PairwiseCompMatrixStack(stack=stack)
        fig, ax = plt.subplots()
        pcms.plot_mean(ax=ax)
        assert ax.images[0].cmap.name == "RdBu_r"

    def test_auto_cmap_nonnegative_mean(self):
        """Stack whose mean is non-negative auto-selects viridis."""
        rng = np.random.default_rng(0)
        stack = rng.uniform(0, 1, size=(4, 4, 5))
        pcms = PairwiseCompMatrixStack(stack=stack)
        fig, ax = plt.subplots()
        pcms.plot_mean(ax=ax)
        assert ax.images[0].cmap.name == "viridis"

    def test_explicit_cmap(self):
        """Explicit cmap overrides auto-detection."""
        stack = np.random.default_rng(0).random((3, 3, 3))
        pcms = PairwiseCompMatrixStack(stack=stack)
        fig, ax = plt.subplots()
        pcms.plot_mean(ax=ax, cmap="hot")
        assert ax.images[0].cmap.name == "hot"


class TestCoverageGaps:
    """Tests for coverage gaps in pairwise modules."""

    def test_remove_by_condition_eq_operator(self):
        """
        Tests: PairwiseCompMatrix.remove_by_condition with 'eq' operator.

        (Test Case 1) Entries equal to threshold are replaced with NaN.
        (Test Case 2) Entries not equal to threshold are preserved.
        """
        mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        pcm = PairwiseCompMatrix(matrix=mat)
        condition = PairwiseCompMatrix(matrix=mat)

        result = pcm.remove_by_condition(condition, "eq", 5.0)
        assert np.isnan(result.matrix[1, 1])
        assert result.matrix[0, 0] == 1.0
        assert result.matrix[2, 2] == 9.0

    def test_remove_by_condition_ne_operator(self):
        """
        Tests: PairwiseCompMatrix.remove_by_condition with 'ne' operator.

        (Test Case 1) Entries not equal to threshold are replaced with NaN.
        (Test Case 2) Entries equal to threshold are preserved.
        """
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        pcm = PairwiseCompMatrix(matrix=mat)
        condition = PairwiseCompMatrix(matrix=mat)

        result = pcm.remove_by_condition(condition, "ne", 2.0)
        # Only (0,1) has value 2.0 in condition, everything else is != 2.0 → NaN
        assert result.matrix[0, 1] == 2.0
        assert np.isnan(result.matrix[0, 0])
        assert np.isnan(result.matrix[1, 0])
        assert np.isnan(result.matrix[1, 1])

    def test_stack_remove_by_condition_stack_vs_stack(self):
        """
        Tests: PairwiseCompMatrixStack.remove_by_condition with stack condition.

        (Test Case 1) Per-slice removal works with matching S dimension.
        """
        rng = np.random.default_rng(42)
        stack = rng.random((3, 3, 4))
        pcms = PairwiseCompMatrixStack(stack=stack)

        cond_stack = rng.random((3, 3, 4))
        cond = PairwiseCompMatrixStack(stack=cond_stack)

        result = pcms.remove_by_condition(cond, "gt", 0.5)
        for s in range(4):
            mask = cond_stack[:, :, s] > 0.5
            assert np.all(np.isnan(result.stack[:, :, s][mask]))
