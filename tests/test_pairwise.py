import pathlib
import sys
import unittest
import numpy as np
import networkx as nx

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpikeLab.spikedata.pairwise import (
    PairwiseCompMatrix,
    PairwiseCompMatrixStack,
)
from SpikeLab.spikedata import SpikeData
from SpikeLab.spikedata.rateslicestack import RateSliceStack


class TestPairwise(unittest.TestCase):
    def test_pairwise_comp_matrix_init(self):
        """
        Tests PairwiseCompMatrix initialization and validation.

        Tests:
        (Test Case 1) Tests normal initialization with valid matrix and labels.
        (Test Case 2) Tests that non-square matrix raises ValueError.
        (Test Case 3) Tests that label count mismatch raises ValueError.
        """
        # Normal init
        matrix = np.random.rand(5, 5)
        pcm = PairwiseCompMatrix(matrix=matrix, labels=["a", "b", "c", "d", "e"])
        self.assertEqual(pcm.matrix.shape, (5, 5))
        self.assertEqual(len(pcm.labels), 5)

        # Invalid shape
        with self.assertRaises(ValueError):
            PairwiseCompMatrix(matrix=np.random.rand(5, 4))

        # Label mismatch
        with self.assertRaises(ValueError):
            PairwiseCompMatrix(matrix=np.random.rand(5, 5), labels=["a", "b"])

    def test_pairwise_comp_matrix_to_networkx(self):
        """
        Tests conversion of PairwiseCompMatrix to NetworkX graph.

        Tests:
        (Test Case 1) Tests basic conversion without threshold.
        (Test Case 2) Tests conversion with threshold filtering edges.
        (Test Case 3) Tests that NaN values are excluded from graph edges.
        """
        matrix = np.array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.8], [0.1, 0.8, 1.0]])
        pcm = PairwiseCompMatrix(matrix=matrix, labels=["A", "B", "C"])

        # No threshold
        G = pcm.to_networkx()
        self.assertEqual(G.number_of_nodes(), 3)
        self.assertEqual(G.number_of_edges(), 3)  # (0,1), (0,2), (1,2)
        self.assertEqual(G.edges[0, 1]["weight"], 0.5)

        # With threshold
        G_thresh = pcm.to_networkx(threshold=0.6)
        self.assertEqual(G_thresh.number_of_edges(), 1)  # Only (1,2) with 0.8
        self.assertIn((1, 2), G_thresh.edges)

        # Handling NaNs in NetworkX
        matrix_nan = matrix.copy()
        matrix_nan[0, 1] = np.nan
        pcm_nan = PairwiseCompMatrix(matrix=matrix_nan)
        G_nan = pcm_nan.to_networkx()
        self.assertEqual(G_nan.number_of_edges(), 2)  # (0,2) and (1,2)

    def test_pairwise_comp_matrix_to_networkx_invert_weights(self):
        """
        Tests that invert_weights correctly transforms edge weights to 1-value.

        Tests:
        (Test Case 1) Tests that weights remain unchanged without invert_weights.
        (Test Case 2) Tests that weights are inverted (1 - value) with invert_weights=True.
        (Test Case 3) Tests that shortest path uses inverted weights correctly.
        """
        matrix = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.5], [0.1, 0.5, 1.0]])
        pcm = PairwiseCompMatrix(matrix=matrix)

        # Without invert_weights
        G = pcm.to_networkx()
        self.assertAlmostEqual(G.edges[0, 1]["weight"], 0.9)
        self.assertAlmostEqual(G.edges[0, 2]["weight"], 0.1)

        # With invert_weights=True
        G_inv = pcm.to_networkx(invert_weights=True)
        self.assertAlmostEqual(G_inv.edges[0, 1]["weight"], 0.1)  # 1 - 0.9
        self.assertAlmostEqual(G_inv.edges[0, 2]["weight"], 0.9)  # 1 - 0.1
        self.assertAlmostEqual(G_inv.edges[1, 2]["weight"], 0.5)  # 1 - 0.5

        # Verify shortest path now uses inverted weights correctly
        # Strong correlation (0.9) should now be a short path (0.1)
        path_length = nx.shortest_path_length(
            G_inv, source=0, target=1, weight="weight"
        )
        self.assertAlmostEqual(path_length, 0.1)

    def test_pairwise_comp_matrix_threshold(self):
        """
        Tests the threshold method for creating binary matrices.

        Tests:
        (Test Case 1) Tests thresholding creates correct binary matrix.
        (Test Case 2) Tests that metadata includes threshold value and binary flag.
        (Test Case 3) Tests thresholding with negative values uses absolute value.
        """
        matrix = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.5], [0.2, 0.5, 1.0]])
        pcm = PairwiseCompMatrix(matrix=matrix)

        # Threshold at 0.4
        binary_pcm = pcm.threshold(0.4)
        expected = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        np.testing.assert_array_equal(binary_pcm.matrix, expected)
        self.assertEqual(binary_pcm.metadata["threshold"], 0.4)
        self.assertTrue(binary_pcm.metadata["binary"])

        # Test with negative values (absolute value)
        matrix_neg = np.array([[1.0, -0.8, 0.2], [-0.8, 1.0, -0.5], [0.2, -0.5, 1.0]])
        pcm_neg = PairwiseCompMatrix(matrix=matrix_neg)
        binary_neg = pcm_neg.threshold(0.4)
        np.testing.assert_array_equal(binary_neg.matrix, expected)

    def test_pairwise_comp_matrix_stack(self):
        """
        Tests PairwiseCompMatrixStack initialization and mean calculation.

        Tests:
        (Test Case 1) Tests stack initialization with correct shape and time metadata.
        (Test Case 2) Tests mean calculation across slices.
        (Test Case 3) Tests mean with ignore_nan=True excludes NaN values.
        (Test Case 4) Tests mean with ignore_nan=False propagates NaN values.
        """
        # n x n x S format (5x5 matrices, 10 slices)
        stack_data = np.random.rand(5, 5, 10)
        times = [(i * 100, (i + 1) * 100) for i in range(10)]
        stack = PairwiseCompMatrixStack(stack=stack_data, times=times)

        self.assertEqual(len(stack), 10)
        self.assertEqual(stack[0].matrix.shape, (5, 5))
        self.assertEqual(stack[0].metadata["time"], (0, 100))

        # Mean calculation
        mean_pcm = stack.mean()
        self.assertTrue(np.allclose(mean_pcm.matrix, np.mean(stack_data, axis=2)))

        # Mean with NaNs
        stack_data_nan = stack_data.copy()
        stack_data_nan[0, 1, 0] = np.nan
        stack_nan = PairwiseCompMatrixStack(stack=stack_data_nan)
        mean_pcm_nan = stack_nan.mean(ignore_nan=True)
        self.assertFalse(np.isnan(mean_pcm_nan.matrix[0, 1]))

        mean_pcm_raw = stack_nan.mean(ignore_nan=False)
        self.assertTrue(np.isnan(mean_pcm_raw.matrix[0, 1]))

    def test_pairwise_comp_matrix_stack_slicing_and_iter(self):
        """
        Tests slicing and iteration on PairwiseCompMatrixStack.

        Tests:
        (Test Case 1) Tests slice with range returns new PairwiseCompMatrixStack.
        (Test Case 2) Tests slice with step returns correct subset.
        (Test Case 3) Tests iteration yields PairwiseCompMatrix objects.

        Notes:
        - stack[i] returns a single PairwiseCompMatrix
        - stack[start:end] returns a new PairwiseCompMatrixStack with selected slices
        - stack[::step] returns every nth slice as a new stack
        """
        # n x n x S format (5x5 matrices, 10 slices)
        stack_data = np.random.rand(5, 5, 10)
        stack = PairwiseCompMatrixStack(stack=stack_data)

        # Slicing with range
        sub_stack = stack[0:3]
        self.assertIsInstance(sub_stack, PairwiseCompMatrixStack)
        self.assertEqual(len(sub_stack), 3)
        self.assertTrue(np.array_equal(sub_stack.stack, stack_data[:, :, 0:3]))

        # Slicing with step
        step_stack = stack[::2]
        self.assertEqual(len(step_stack), 5)  # 0, 2, 4, 6, 8
        self.assertTrue(np.array_equal(step_stack.stack, stack_data[:, :, ::2]))

        # Iteration
        matrices = list(stack)
        self.assertEqual(len(matrices), 10)
        self.assertIsInstance(matrices[0], PairwiseCompMatrix)
        self.assertTrue(np.array_equal(matrices[0].matrix, stack_data[:, :, 0]))

    def test_pairwise_comp_matrix_stack_subslice(self):
        """
        Tests the subslice method for selecting specific non-contiguous slices.

        Tests:
        (Test Case 1) Tests subslice returns correct number of slices.
        (Test Case 2) Tests subslice data matches original at selected indices.
        (Test Case 3) Tests times are correctly subsliced.
        """
        # n x n x S format (5x5 matrices, 10 slices)
        stack_data = np.random.rand(5, 5, 10)
        times = [(i * 100, (i + 1) * 100) for i in range(10)]
        stack = PairwiseCompMatrixStack(stack=stack_data, times=times)

        # Select specific slices
        sub = stack.subslice([0, 2, 5, 9])
        self.assertEqual(len(sub), 4)
        self.assertTrue(np.array_equal(sub.stack[:, :, 0], stack_data[:, :, 0]))
        self.assertTrue(np.array_equal(sub.stack[:, :, 1], stack_data[:, :, 2]))
        self.assertTrue(np.array_equal(sub.stack[:, :, 2], stack_data[:, :, 5]))
        self.assertTrue(np.array_equal(sub.stack[:, :, 3], stack_data[:, :, 9]))

        # Times should also be subsliced
        self.assertEqual(sub.times, [(0, 100), (200, 300), (500, 600), (900, 1000)])

    def test_pairwise_comp_matrix_stack_threshold(self):
        """
        Tests the threshold method for PairwiseCompMatrixStack.

        Tests:
        (Test Case 1) Tests threshold creates correct binary stack.
        (Test Case 2) Tests metadata includes threshold value.
        """
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
        self.assertEqual(binary_stack.metadata["threshold"], 0.4)

    def test_integration_spikedata(self):
        """
        Tests integration of PairwiseCompMatrix with SpikeData.

        Tests:
        (Test Case 1) Tests spike_time_tilings returns PairwiseCompMatrix.
        (Test Case 2) Tests correct shape and metadata from STTC calculation.
        """
        # Create dummy SpikeData
        train = [np.array([10, 20, 30]), np.array([15, 25, 35])]
        sd = SpikeData(train, length=100)

        sttc_pcm = sd.spike_time_tilings(delt=5.0)
        self.assertIsInstance(sttc_pcm, PairwiseCompMatrix)
        self.assertEqual(sttc_pcm.matrix.shape, (2, 2))
        self.assertEqual(sttc_pcm.metadata["delt"], 5.0)

    def test_integration_rateslicestack(self):
        """
        Tests integration of PairwiseCompMatrixStack with RateSliceStack.

        Tests:
        (Test Case 1) Tests unit_to_unit_correlation returns correct shape (U x U x S).
        (Test Case 2) Tests get_slice_to_slice_unit_corr_from_stack returns correct shape (S x S x U).
        (Test Case 3) Tests get_slice_to_slice_time_corr_from_stack returns correct shape (S x S x T).
        """
        # Create dummy RateSliceStack
        event_matrix = np.random.rand(2, 50, 5)  # U x T x S
        rss = RateSliceStack(None, event_matrix=event_matrix)

        # Test unit_to_unit_correlation - now returns U x U x S
        corr_stack, lag_stack, av_corr, av_lag = rss.unit_to_unit_correlation()
        self.assertIsInstance(corr_stack, PairwiseCompMatrixStack)
        self.assertIsInstance(lag_stack, PairwiseCompMatrixStack)
        self.assertEqual(len(corr_stack), 5)  # 5 slices
        self.assertEqual(corr_stack[0].matrix.shape, (2, 2))  # 2x2 unit matrix
        self.assertEqual(corr_stack.stack.shape, (2, 2, 5))  # n x n x S

        # Test get_slice_to_slice_unit_corr_from_stack - returns S x S x U
        all_slice_corr, av_slice_corr = rss.get_slice_to_slice_unit_corr_from_stack()
        self.assertIsInstance(all_slice_corr, PairwiseCompMatrixStack)
        self.assertEqual(all_slice_corr[0].matrix.shape, (5, 5))  # S x S
        self.assertEqual(
            all_slice_corr.stack.shape, (5, 5, 2)
        )  # n x n x S (where n=S=5, third dim=U=2)

        # Test get_slice_to_slice_time_corr_from_stack - returns S x S x T
        all_slice_time_corr, av_slice_time_corr = (
            rss.get_slice_to_slice_time_corr_from_stack()
        )
        self.assertIsInstance(all_slice_time_corr, PairwiseCompMatrixStack)
        self.assertEqual(all_slice_time_corr[0].matrix.shape, (5, 5))  # S x S
        self.assertEqual(len(all_slice_time_corr), 50)  # T time bins
        self.assertEqual(
            all_slice_time_corr.stack.shape, (5, 5, 50)
        )  # n x n x S (where n=S=5, third dim=T=50)

    def test_rigorous_edge_cases(self):
        """
        Tests edge cases for PairwiseCompMatrix and PairwiseCompMatrixStack.

        Tests:
        (Test Case 1) Tests empty stack has length 0.
        (Test Case 2) Tests mean on empty stack returns NaN matrix.
        (Test Case 3) Tests single unit matrix converts to graph with 1 node and 0 edges.
        (Test Case 4) Tests infinite values are preserved in graph edges.
        """
        # Empty stack
        empty_stack_data = np.zeros((5, 5, 0))  # n x n x S with S=0
        empty_stack = PairwiseCompMatrixStack(stack=empty_stack_data)
        self.assertEqual(len(empty_stack), 0)

        # mean() on empty stack
        with warnings_context():  # Avoid RuntimeWarning: Mean of empty slice
            mean_empty = empty_stack.mean()
            self.assertTrue(np.all(np.isnan(mean_empty.matrix)))

        # Single unit matrix
        single_matrix = np.array([[1.0]])
        pcm_single = PairwiseCompMatrix(matrix=single_matrix)
        G_single = pcm_single.to_networkx()
        self.assertEqual(G_single.number_of_nodes(), 1)
        self.assertEqual(G_single.number_of_edges(), 0)

        # Infs
        inf_matrix = np.array([[1.0, np.inf], [np.inf, 1.0]])
        pcm_inf = PairwiseCompMatrix(matrix=inf_matrix)
        G_inf = pcm_inf.to_networkx()
        self.assertEqual(G_inf.edges[0, 1]["weight"], float("inf"))

    def test_pca_compatibility(self):
        """
        Tests PCA dimensionality reduction on lower triangle features.

        Tests:
        (Test Case 1) Tests extract_lower_triangle_features returns correct shape.
        (Test Case 2) Tests dim_red_on_lower_diagonal_corr_matrix with PCA returns correct shape.
        """
        # Create a stack of 10 matrices (5x5 units) - now n x n x S
        stack_data = np.random.rand(5, 5, 10)
        # Make them symmetric
        for i in range(10):
            stack_data[:, :, i] = (stack_data[:, :, i] + stack_data[:, :, i].T) / 2

        stack = PairwiseCompMatrixStack(stack=stack_data)

        # Test extract_lower_triangle_features on stack
        features = stack.extract_lower_triangle_features()
        self.assertEqual(features.shape, (10, 10))  # 5*(5-1)/2 = 10 features

        # Test dim_red_on_lower_diagonal_corr_matrix (default PCA)
        pca_result = stack.dim_red_on_lower_diagonal_corr_matrix(
            method="PCA", n_components=2
        )
        self.assertEqual(pca_result.shape, (10, 2))

    def test_dim_red_pca_with_kwargs_raises(self):
        """
        Tests that PCA method raises TypeError when given extra kwargs.

        Tests:
        (Test Case 1) Tests that passing UMAP-specific kwargs to PCA raises TypeError.
        """
        stack_data = np.random.rand(5, 5, 10)
        for i in range(10):
            stack_data[:, :, i] = (stack_data[:, :, i] + stack_data[:, :, i].T) / 2
        stack = PairwiseCompMatrixStack(stack=stack_data)

        with self.assertRaises(TypeError) as ctx:
            stack.dim_red_on_lower_diagonal_corr_matrix(
                method="PCA", n_components=2, n_neighbors=15
            )
        self.assertIn("only supported for UMAP", str(ctx.exception))

    def test_dim_red_unknown_method_raises(self):
        """
        Tests that unknown method raises ValueError.

        Tests:
        (Test Case 1) Tests that passing unknown method name raises ValueError with method name in message.
        """
        stack_data = np.random.rand(5, 5, 10)
        for i in range(10):
            stack_data[:, :, i] = (stack_data[:, :, i] + stack_data[:, :, i].T) / 2
        stack = PairwiseCompMatrixStack(stack=stack_data)

        with self.assertRaises(ValueError) as ctx:
            stack.dim_red_on_lower_diagonal_corr_matrix(method="TSNE", n_components=2)
        self.assertIn("Unknown manifold method", str(ctx.exception))
        self.assertIn("TSNE", str(ctx.exception))

    def test_dim_red_umap(self):
        """
        Tests UMAP dimensionality reduction on lower triangle features.

        Tests:
        (Test Case 1) Tests dim_red_on_lower_diagonal_corr_matrix with UMAP returns correct shape.

        Notes:
        - Skipped if umap-learn is not installed.
        """
        try:
            import umap  # noqa: F401
        except ImportError:
            self.skipTest("umap-learn not installed")

        stack_data = np.random.rand(5, 5, 20)
        for i in range(20):
            stack_data[:, :, i] = (stack_data[:, :, i] + stack_data[:, :, i].T) / 2
        stack = PairwiseCompMatrixStack(stack=stack_data)

        umap_result = stack.dim_red_on_lower_diagonal_corr_matrix(
            method="UMAP", n_components=2, n_neighbors=5, min_dist=0.1
        )
        self.assertEqual(umap_result.shape, (20, 2))


def warnings_context():
    import warnings

    return warnings.catch_warnings()


if __name__ == "__main__":
    unittest.main()
