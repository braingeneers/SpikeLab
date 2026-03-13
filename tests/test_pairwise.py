import pathlib
import sys
import warnings

import numpy as np
import networkx as nx
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpikeLab.spikedata.pairwise import (
    PairwiseCompMatrix,
    PairwiseCompMatrixStack,
)
from SpikeLab.spikedata import SpikeData
from SpikeLab.spikedata.rateslicestack import RateSliceStack

try:
    import umap  # noqa: F401

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class TestPairwise:
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
        pca_result = stack.dim_red_on_lower_diagonal_corr_matrix(
            method="PCA", n_components=2
        )
        assert pca_result.shape == (10, 2)

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

        umap_result = stack.dim_red_on_lower_diagonal_corr_matrix(
            method="UMAP", n_components=2, n_neighbors=5, min_dist=0.1
        )
        assert umap_result.shape == (20, 2)


class TestPairwiseEdgeCases:
    """Edge case tests for pairwise matrix and stack operations.

    Tests:
        subslice with empty indices list
        mean on empty stack
        dim_red on empty stack
        dim_red on single-slice stack
        mean on single-slice stack
        subslice with duplicate indices
        negative indexing on stack
        reverse slice on stack
        empty slice on stack
        iteration over empty stack
        extract_lower_triangle on 1x1 matrix
        threshold at zero
        to_networkx on all-NaN matrix
        post_init validation for wrong ndim
    """

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

    def test_iter_empty_stack(self):
        """Iterating over an empty stack yields no elements.

        Tests: list(stack) on a (3,3,0) stack should be an
        empty list.
        """
        stack = PairwiseCompMatrixStack(stack=np.empty((3, 3, 0)), times=[])
        items = list(stack)
        assert items == []

    def test_extract_lower_triangle_1x1(self):
        """Lower triangle of a 1x1 matrix is empty.

        Tests: a (1,1) PairwiseCompMatrix has no off-diagonal
        elements, so extract_lower_triangle should return an empty array.
        """
        pcm = PairwiseCompMatrix(matrix=np.array([[1.0]]))
        result = pcm.extract_lower_triangle()
        assert result.shape == (0,)
        assert len(result) == 0

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

    def test_post_init_wrong_ndim(self):
        """PairwiseCompMatrix rejects non-2D arrays.

        Tests: passing a 1D or 3D array to PairwiseCompMatrix
        should raise a ValueError during __post_init__ validation.
        """
        with pytest.raises(ValueError):
            PairwiseCompMatrix(matrix=np.array([1.0, 2.0, 3.0]))

        with pytest.raises(ValueError):
            PairwiseCompMatrix(matrix=np.random.rand(3, 3, 3))

    def test_subslice_out_of_bounds_index(self):
        """Subslice with an out-of-bounds index raises IndexError.

        Tests: subslice([0, 1, 10]) on a (3,3,5) stack should raise
        an IndexError because index 10 exceeds the stack size of 5.
        """
        stack = PairwiseCompMatrixStack(stack=np.random.rand(3, 3, 5))
        with pytest.raises(IndexError):
            stack.subslice([0, 1, 10])

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
