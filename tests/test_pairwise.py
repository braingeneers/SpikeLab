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
