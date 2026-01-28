import unittest
import numpy as np
import sys
import pathlib
from unittest.mock import MagicMock

# Ensure project root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spikedata.rateslicestack import RateSliceStack
from spikedata.ratedata import RateData
from spikedata.spikedata import SpikeData


class TestRateSliceStack(unittest.TestCase):
    def setUp(self):
        # Create a basic RateData object
        self.data = np.random.rand(5, 100)  # 5 units, 100 time bins
        self.times = np.arange(100)
        self.rd = RateData(self.data, self.times)

        # Slices for Option 1
        self.slices = [(10, 20), (30, 40), (50, 60)]

    def test_init_invalid_data_obj(self):
        with self.assertRaises(TypeError):
            RateSliceStack("invalid")

    def test_init_option1_missing_times(self):
        with self.assertRaises(ValueError):
            RateSliceStack(self.rd)

    def test_init_option1_invalid_bounds(self):
        with self.assertRaises(TypeError):
            RateSliceStack(self.rd, time_peaks=[10], time_bounds=[5, 5])

    def test_init_option1_valid_peaks(self):
        stack = RateSliceStack(self.rd, time_peaks=[20, 50], time_bounds=(5, 5))
        self.assertEqual(len(stack.times), 2)
        self.assertEqual(stack.event_stack.shape[2], 2)

    def test_init_option1_single_time(self):
        # Coverage for line 114
        rd_single = RateData(np.array([[1.0]]), [10.0])
        stack = RateSliceStack(rd_single, times_start_to_end=[(10, 20)])
        self.assertEqual(stack.step_size, 1.0)

    def test_init_option1_with_spikedata(self):
        # Create a mock SpikeData that behaves correctly
        sd = MagicMock(spec=SpikeData)
        sd.length = 100.0
        # resampled_isi returns a matrix
        sd.resampled_isi.return_value = np.zeros((2, 100))

        stack = RateSliceStack(sd, times_start_to_end=[(10, 20)])
        self.assertEqual(stack.event_stack.shape[2], 1)

    def test_init_option2_invalid_matrix(self):
        with self.assertRaises(TypeError):
            RateSliceStack(None, event_matrix="invalid")
        with self.assertRaises(ValueError):
            RateSliceStack(None, event_matrix=np.zeros((10, 10)))  # 2D instead of 3D

    def test_init_option2_valid(self):
        matrix = np.random.rand(2, 10, 3)
        stack = RateSliceStack(None, event_matrix=matrix)
        self.assertEqual(stack.event_stack.shape, (2, 10, 3))
        self.assertEqual(len(stack.times), 3)
        self.assertEqual(stack.step_size, 1.0)

    def test_init_option2_with_step_size(self):
        # Coverage for line 143
        matrix = np.random.rand(2, 10, 3)
        stack = RateSliceStack(None, event_matrix=matrix, step_size=2.0)
        self.assertEqual(stack.step_size, 2.0)

    def test_init_option2_mismatched_times(self):
        matrix = np.random.rand(2, 10, 3)
        with self.assertRaises(ValueError):
            RateSliceStack(None, event_matrix=matrix, times_start_to_end=[(0, 10)])

    def test_init_no_options(self):
        with self.assertRaises(ValueError):
            RateSliceStack(None)

    def test_validate_times_errors(self):
        stack = RateSliceStack(None, event_matrix=np.zeros((2, 10, 1)))
        with self.assertRaises(TypeError):
            stack._validate_time_start_to_end("not a list")
        with self.assertRaises(TypeError):
            stack._validate_time_start_to_end([10])
        with self.assertRaises(TypeError):
            stack._validate_time_start_to_end([(10, 20, 30)])
        with self.assertRaises(TypeError):
            stack._validate_time_start_to_end([(10, "20")])
        with self.assertRaises(ValueError):
            stack._validate_time_start_to_end([(20, 10)])
        with self.assertRaises(ValueError):
            # Mismatched lengths
            stack._validate_time_start_to_end([(10, 20), (30, 45)])

    def test_validate_times_skip_negative(self):
        # Coverage for line 195
        matrix = np.zeros((2, 10, 1))
        stack = RateSliceStack(None, event_matrix=matrix)
        # One valid, one with negative start
        times = [(10, 20), (-10, 0)]
        valid = stack._validate_time_start_to_end(times)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0], (10, 20))

    def test_order_units_across_slices(self):
        # Create a matrix where unit 0 peaks at bin 2, unit 1 peaks at bin 8
        matrix = np.zeros((2, 10, 2))
        matrix[0, 2, :] = 1.0  # Unit 0 peaks at 2
        matrix[1, 8, :] = 1.0  # Unit 1 peaks at 8
        stack = RateSliceStack(None, event_matrix=matrix)

        # Test median
        reordered, ids, std, peaks = stack.order_units_across_slices("median")
        self.assertTrue(np.array_equal(ids, [0, 1]))

        # Test mean
        reordered, ids, std, peaks = stack.order_units_across_slices("mean")
        self.assertTrue(np.array_equal(ids, [0, 1]))

        # Test invalid agg_func
        with self.assertRaises(ValueError):
            stack.order_units_across_slices("invalid")

    def test_get_slice_to_slice_unit_corr(self):
        matrix = np.random.rand(2, 10, 3)
        stack = RateSliceStack(None, event_matrix=matrix)
        all_corr, av_corr = stack.get_slice_to_slice_unit_corr_from_stack(max_lag=1)
        self.assertEqual(all_corr.shape, (2, 3, 3))
        self.assertEqual(av_corr.shape, (2,))

    def test_get_slice_to_slice_unit_corr_threshold(self):
        # Coverage for line 325-326, 334
        matrix = np.zeros((1, 10, 2))
        matrix[0, :, 0] = 0.5  # Reference slice ABOVE threshold
        matrix[0, :, 1] = 0.05  # Comp slice BELOW threshold
        stack = RateSliceStack(None, event_matrix=matrix)
        all_corr, av_corr = stack.get_slice_to_slice_unit_corr_from_stack(
            MIN_RATE_THRESHOLD=0.1
        )
        # 1. ref_b=0 (rate 0.5) is OK.
        # 2. comp_b=1 (rate 0.05) triggers line 334 continue.
        self.assertTrue(np.isnan(all_corr[0, 0, 1]))

    def test_get_slice_to_slice_time_corr(self):
        matrix = np.random.rand(2, 10, 3)
        stack = RateSliceStack(None, event_matrix=matrix)
        all_corr, av_corr = stack.get_slice_to_slice_time_corr_from_stack(max_lag=0)
        self.assertEqual(all_corr.shape, (10, 3, 3))
        self.assertEqual(av_corr.shape, (10,))

    def test_pca_reduction(self):
        matrix = np.random.rand(2, 3, 3)  # simulate corr matrix
        stack = RateSliceStack(None, event_matrix=np.random.rand(5, 10, 3))
        # This will call extract_lower_triangle_features and PCA_reduction
        # We need more than 2 slices for tril_indices(3) to have enough features for n_components=2?
        # tril_indices(3, k=-1) is (1,0), (2,0), (2,1) -> 3 features.
        res = stack.PCA_on_lower_diagnol_corr_matrix(matrix, n_components=1)
        self.assertEqual(res.shape, (2, 1))

    def test_convert_to_list_of_ratedata(self):
        matrix = np.random.rand(2, 10, 3)
        stack = RateSliceStack(None, event_matrix=matrix)
        rd_list = stack.convert_to_list_of_RateData()
        self.assertEqual(len(rd_list), 3)
        self.assertIsInstance(rd_list[0], RateData)

    def test_unit_to_unit_correlation(self):
        matrix = np.random.rand(2, 10, 3)
        stack = RateSliceStack(None, event_matrix=matrix)
        corr, lag, av_corr, av_lag = stack.unit_to_unit_correlation(max_lag=1)
        self.assertEqual(corr.shape, (3, 2, 2))
        self.assertEqual(lag.shape, (3, 2, 2))
        self.assertEqual(av_corr.shape, (3,))


if __name__ == "__main__":
    unittest.main()
