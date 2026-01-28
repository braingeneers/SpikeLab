import sys
import unittest
from unittest.mock import MagicMock
import numpy as np
import pytest
import pathlib

# Ensure project root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# HACK: Fix broken imports in spikeslicestack.py
# The source code does `import SpikeData` and `import RateData`
# which fails in a standard package structure.
from spikedata import spikedata, ratedata

sys.modules["SpikeData"] = spikedata
sys.modules["RateData"] = ratedata

from spikedata.spikeslicestack import SpikeSliceStack
from spikedata import SpikeData


class TestSpikeSliceStack(unittest.TestCase):
    def setUp(self):
        # Create a standard SpikeData object for testing
        self.train = [np.array([10.0, 20.0, 30.0]), np.array([15.0, 25.0])]
        self.sd = SpikeData(self.train, length=100.0)

    def test_init_invalid_data_obj(self):
        with self.assertRaises(TypeError):
            SpikeSliceStack("not a spikedata")

    def test_init_missing_arguments(self):
        with self.assertRaises(ValueError):
            SpikeSliceStack(self.sd)

    def test_init_times_start_to_end_wrong_type(self):
        with self.assertRaises(TypeError):
            SpikeSliceStack(self.sd, times_start_to_end="not a list")

    def test_init_times_start_to_end_not_tuples(self):
        with self.assertRaises(TypeError):
            SpikeSliceStack(self.sd, times_start_to_end=[[10, 20]])

    def test_init_times_start_to_end_wrong_tuple_len(self):
        with self.assertRaises(TypeError):
            SpikeSliceStack(self.sd, times_start_to_end=[(10,)])

    def test_init_times_start_to_end_non_numeric(self):
        with self.assertRaises(TypeError):
            SpikeSliceStack(self.sd, times_start_to_end=[(10, "20")])

    def test_init_times_start_to_end_logic_error(self):
        with self.assertRaises(ValueError):
            SpikeSliceStack(self.sd, times_start_to_end=[(20, 10)])

    def test_init_times_different_lengths_raises(self):
        with self.assertRaises(ValueError):
            # All windows must have the same length
            SpikeSliceStack(self.sd, times_start_to_end=[(10, 20), (30, 45)])

    def test_init_time_peaks_bounds_wrong_type(self):
        with self.assertRaises(TypeError):
            SpikeSliceStack(self.sd, time_peaks=[10, 20], time_bounds=[250, 500])

    def test_init_time_peaks_bounds_wrong_len(self):
        with self.assertRaises(TypeError):
            SpikeSliceStack(self.sd, time_peaks=[10, 20], time_bounds=(250, 500, 100))

    def test_init_valid_times_start_to_end(self):
        # Valid Choice A
        times = [(10, 20), (30, 40), (50, 60)]
        stack = SpikeSliceStack(self.sd, times_start_to_end=times)
        self.assertEqual(len(stack.spike_stack), 3)
        self.assertEqual(stack.times, times)
        for i, (start, end) in enumerate(times):
            self.assertEqual(stack.spike_stack[i].length, end - start)

    def test_init_valid_peaks_bounds(self):
        # Valid Choice B
        peaks = [20, 50, 80]
        bounds = (5, 10)  # 5 before, 10 after
        stack = SpikeSliceStack(self.sd, time_peaks=peaks, time_bounds=bounds)
        # Expected times: (15, 30), (45, 60), (75, 90)
        expected = [(15, 30), (45, 60), (75, 90)]
        self.assertEqual(stack.times, expected)
        self.assertEqual(len(stack.spike_stack), 3)

    def test_init_negative_start_filtered(self):
        # If start < 0, the slice should be skipped (per line 88-89 of spikeslicestack.py)
        peaks = [2, 20]
        bounds = (5, 5)  # (2-5, 2+5) = (-3, 7) - negative start
        stack = SpikeSliceStack(self.sd, time_peaks=peaks, time_bounds=bounds)
        # Only (15, 25) should remain
        self.assertEqual(len(stack.times), 1)
        self.assertEqual(stack.times[0], (15, 25))

    def test_to_sparse_matrices_exposes_bug(self):
        """
        Exposes bug in SpikeSliceStack.to_sparse_matrices where it uses `for i in len(self.spike_stack):`
        instead of `range(len(...))` or `enumerate(...)`.
        """
        times = [(10, 20), (30, 40)]
        stack = SpikeSliceStack(self.sd, times_start_to_end=times)

        # This will fail with TypeError: 'int' object is not iterable
        try:
            mat = stack.to_sparse_matrices()
            self.assertEqual(mat.shape[0], self.sd.N)
            self.assertEqual(mat.shape[2], 2)
        except TypeError as e:
            if "'int' object is not iterable" in str(e):
                # Regression confirmed
                pass
            else:
                raise e


if __name__ == "__main__":
    unittest.main()
