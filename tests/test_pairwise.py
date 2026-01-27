import unittest
import numpy as np
import networkx as nx
from spikedata.pairwise import PairwiseCompMatrix, PairwiseCompMatrixStack
from spikedata import SpikeData
from spikedata.rateslicestack import RateSliceStack


class TestPairwise(unittest.TestCase):
    def test_pairwise_comp_matrix_init(self):
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

    def test_pairwise_comp_matrix_stack(self):
        stack_data = np.random.rand(10, 5, 5)
        times = [(i * 100, (i + 1) * 100) for i in range(10)]
        stack = PairwiseCompMatrixStack(stack=stack_data, times=times)

        self.assertEqual(len(stack), 10)
        self.assertEqual(stack[0].matrix.shape, (5, 5))
        self.assertEqual(stack[0].metadata["time"], (0, 100))

        # Mean calculation
        mean_pcm = stack.mean()
        self.assertTrue(np.allclose(mean_pcm.matrix, np.mean(stack_data, axis=0)))

        # Mean with NaNs
        stack_data_nan = stack_data.copy()
        stack_data_nan[0, 0, 1] = np.nan
        stack_nan = PairwiseCompMatrixStack(stack=stack_data_nan)
        mean_pcm_nan = stack_nan.mean(ignore_nan=True)
        self.assertFalse(np.isnan(mean_pcm_nan.matrix[0, 1]))

        mean_pcm_raw = stack_nan.mean(ignore_nan=False)
        self.assertTrue(np.isnan(mean_pcm_raw.matrix[0, 1]))

    def test_pairwise_comp_matrix_stack_slicing_and_iter(self):
        stack_data = np.random.rand(10, 5, 5)
        stack = PairwiseCompMatrixStack(stack=stack_data)

        # Slicing
        sub_stack = stack[0:3]
        self.assertIsInstance(sub_stack, PairwiseCompMatrixStack)
        self.assertEqual(len(sub_stack), 3)
        self.assertTrue(np.array_equal(sub_stack.stack, stack_data[0:3]))

        # Iteration
        matrices = list(stack)
        self.assertEqual(len(matrices), 10)
        self.assertIsInstance(matrices[0], PairwiseCompMatrix)
        self.assertTrue(np.array_equal(matrices[0].matrix, stack_data[0]))

    def test_integration_spikedata(self):
        # Create dummy SpikeData
        train = [np.array([10, 20, 30]), np.array([15, 25, 35])]
        sd = SpikeData(train, length=100)

        sttc_pcm = sd.spike_time_tilings(delt=5.0)
        self.assertIsInstance(sttc_pcm, PairwiseCompMatrix)
        self.assertEqual(sttc_pcm.matrix.shape, (2, 2))
        self.assertEqual(sttc_pcm.metadata["delt"], 5.0)

    def test_integration_rateslicestack(self):
        # Create dummy RateSliceStack
        event_matrix = np.random.rand(2, 50, 5)  # U x T x S
        rss = RateSliceStack(None, event_matrix=event_matrix)

        # Test unit_to_unit_correlation
        corr_stack, lag_stack, av_corr, av_lag = rss.unit_to_unit_correlation()
        self.assertIsInstance(corr_stack, PairwiseCompMatrixStack)
        self.assertIsInstance(lag_stack, PairwiseCompMatrixStack)
        self.assertEqual(len(corr_stack), 5)
        self.assertEqual(corr_stack[0].matrix.shape, (2, 2))

        # Test get_slice_to_slice_unit_corr_from_stack
        all_slice_corr, av_slice_corr = rss.get_slice_to_slice_unit_corr_from_stack()
        self.assertIsInstance(all_slice_corr, PairwiseCompMatrixStack)
        self.assertEqual(all_slice_corr[0].matrix.shape, (5, 5))

        # Test get_slice_to_slice_time_corr_from_stack
        all_slice_time_corr, av_slice_time_corr = (
            rss.get_slice_to_slice_time_corr_from_stack()
        )
        self.assertIsInstance(all_slice_time_corr, PairwiseCompMatrixStack)
        self.assertEqual(all_slice_time_corr[0].matrix.shape, (5, 5))
        self.assertEqual(len(all_slice_time_corr), 50)  # T

    def test_rigorous_edge_cases(self):
        # Empty stack
        empty_stack_data = np.zeros((0, 5, 5))
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


def warnings_context():
    import warnings

    return warnings.catch_warnings()


if __name__ == "__main__":
    unittest.main()
