"""
Tests for the RateData class (spikedata/ratedata.py).

Covers: constructor validation, subset, subtime, subtime_by_index,
frames, get_pairwise_fr_corr, and get_manifold.
"""

import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spikedata.ratedata import RateData
from spikedata.rateslicestack import RateSliceStack

try:
    import umap  # noqa: F401

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def make_ratedata(n_units=3, n_times=100, step=1.0, t0=0.0, seed=0):
    """
    Create a RateData with random firing rates on a uniform time grid.

    Parameters:
        n_units (int): Number of units.
        n_times (int): Number of time bins.
        step (float): Time step in milliseconds.
        t0 (float): Start time in milliseconds.
        seed (int): Random seed for reproducibility.

    Returns:
        rd (RateData): A RateData object with shape (n_units, n_times).
    """
    rng = np.random.default_rng(seed)
    times = np.arange(t0, t0 + n_times * step, step)
    data = rng.random((n_units, len(times)))
    return RateData(data, times)


class TestRateData(unittest.TestCase):
    def test_constructor(self):
        """
        Tests RateData constructor for valid inputs and validation errors.

        Tests:
            (Test Case 1) Valid construction stores correct attributes.
            (Test Case 2) Non-2D array raises ValueError.
            (Test Case 3) Mismatched times length raises ValueError.
            (Test Case 4) Negative time value raises ValueError.
        """
        times = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.ones((2, 4))

        rd = RateData(data, times)
        self.assertEqual(rd.N, 2)
        self.assertEqual(rd.inst_Frate_data.shape, (2, 4))
        self.assertTrue(np.array_equal(rd.times, times))

        # Non-2D array raises ValueError.
        with self.assertRaises(ValueError):
            RateData(np.ones((2, 4, 1)), times)

        # Times length mismatch raises ValueError.
        with self.assertRaises(ValueError):
            RateData(data, np.array([0.0, 1.0]))

        # Negative time raises ValueError.
        with self.assertRaises(ValueError):
            RateData(data, np.array([-1.0, 0.0, 1.0, 2.0]))

    def test_subset(self):
        """
        Tests that subset() returns a RateData with the correct units.

        Tests:
            (Test Case 1) List-based index selection returns correct rows and shape.
            (Test Case 2) Single int input is handled correctly.
            (Test Case 3) Times are preserved unchanged.
        """
        rd = make_ratedata(n_units=5, n_times=50)

        sub = rd.subset([0, 2, 4])
        self.assertEqual(sub.N, 3)
        self.assertEqual(sub.inst_Frate_data.shape, (3, 50))
        np.testing.assert_array_equal(sub.inst_Frate_data[0], rd.inst_Frate_data[0])
        np.testing.assert_array_equal(sub.inst_Frate_data[1], rd.inst_Frate_data[2])
        np.testing.assert_array_equal(sub.inst_Frate_data[2], rd.inst_Frate_data[4])
        np.testing.assert_array_equal(sub.times, rd.times)

        # Single int.
        sub_single = rd.subset(1)
        self.assertEqual(sub_single.N, 1)
        self.assertEqual(sub_single.inst_Frate_data.shape, (1, 50))

    def test_subtime(self):
        """
        Tests that subtime() slices correctly with both shift_time modes.

        Tests:
            (Test Case 1) Basic slice extracts correct time range.
            (Test Case 2) shift_time=True shifts times to start from 0.
            (Test Case 3) shift_time=False preserves original time values.
            (Test Case 4) No time points in range raises ValueError.
        """
        rd = make_ratedata(n_units=2, n_times=100, step=1.0)  # times: 0..99

        sub = rd.subtime(20.0, 40.0)
        # times in [20, 40) → 20 bins
        self.assertEqual(sub.inst_Frate_data.shape[1], 20)
        # shift_time=True (default): times start at 0
        self.assertAlmostEqual(float(sub.times[0]), 0.0)

        # shift_time=False: times retain original values
        sub_abs = rd.subtime(20.0, 40.0, shift_time=False)
        self.assertAlmostEqual(float(sub_abs.times[0]), 20.0)
        self.assertAlmostEqual(float(sub_abs.times[-1]), 39.0)

        # Data matches the original slice.
        np.testing.assert_array_equal(sub.inst_Frate_data, rd.inst_Frate_data[:, 20:40])

        # Out-of-range raises ValueError.
        with self.assertRaises(ValueError):
            rd.subtime(200.0, 300.0)

    def test_subtime_by_index(self):
        """
        Tests that subtime_by_index() slices by column index.

        Tests:
            (Test Case 1) Correct data and shape returned for valid indices.
            (Test Case 2) shift_time=True shifts times to start from 0.
            (Test Case 3) Invalid start or end index raises ValueError.
        """
        rd = make_ratedata(n_units=2, n_times=60, step=2.0)  # times: 0,2,4,...,118

        sub = rd.subtime_by_index(10, 30)
        self.assertEqual(sub.inst_Frate_data.shape, (2, 20))
        np.testing.assert_array_equal(sub.inst_Frate_data, rd.inst_Frate_data[:, 10:30])
        self.assertAlmostEqual(float(sub.times[0]), 0.0)  # shift_time=True default

        # shift_time=False preserves values.
        sub_abs = rd.subtime_by_index(10, 30, shift_time=False)
        self.assertAlmostEqual(float(sub_abs.times[0]), float(rd.times[10]))

        # Out-of-bounds indices raise ValueError.
        with self.assertRaises(ValueError):
            rd.subtime_by_index(-1, 10)
        with self.assertRaises(ValueError):
            rd.subtime_by_index(10, 100)

    def test_frames(self):
        """
        Tests that frames() returns a correctly shaped RateSliceStack.

        Tests:
            (Test Case 1) Returns a RateSliceStack instance.
            (Test Case 2) Frame count is correct for evenly divisible recording.
            (Test Case 3) Each frame's data matches the corresponding subtime slice.

        Notes:
            - times are [0..99] ms at 1 ms step; length=100 bins, frame=20 ms → 5 frames.
        """
        rd = make_ratedata(n_units=3, n_times=100, step=1.0)  # times: 0..99

        stack = rd.frames(20)
        self.assertIsInstance(stack, RateSliceStack)
        self.assertEqual(len(stack.times), 5)
        self.assertEqual(stack.event_stack.shape, (3, 20, 5))

        # Each frame's data must match the raw subtime slice.
        for i, (start, end) in enumerate(stack.times):
            expected = rd.subtime(start, end, shift_time=False).inst_Frate_data
            np.testing.assert_array_equal(stack.event_stack[:, :, i], expected)

    def test_frames_overlap(self):
        """
        Tests frames() with overlap and that partial last windows are excluded.

        Tests:
            (Test Case 1) Overlap produces more frames with correct step.
            (Test Case 2) Window that would extend past the last time bin is excluded.
            (Test Case 3) Data of overlapping frames is internally consistent.

        Notes:
            - times [0..99], frame=20, overlap=10 → step=10 → starts [0,10,...,80] = 9 frames.
              Start 90 gives window (90,110); 110 > 99+1 so it is excluded.
        """
        rd = make_ratedata(n_units=2, n_times=100, step=1.0)

        stack = rd.frames(20, overlap=10)
        self.assertIsInstance(stack, RateSliceStack)
        self.assertEqual(len(stack.times), 9)
        self.assertEqual(stack.event_stack.shape, (2, 20, 9))

        # Verify the last frame starts at 80 and ends at 100.
        last_start, last_end = stack.times[-1]
        self.assertAlmostEqual(last_start, 80.0)
        self.assertAlmostEqual(last_end, 100.0)

    def test_frames_errors(self):
        """
        Tests that frames() raises ValueError for invalid arguments.

        Tests:
            (Test Case 1) overlap equal to length raises ValueError.
            (Test Case 2) overlap greater than length raises ValueError.
            (Test Case 3) Frame length larger than the recording raises ValueError.
        """
        rd = make_ratedata(n_units=2, n_times=50, step=1.0)

        with self.assertRaises(ValueError):
            rd.frames(20, overlap=20)

        with self.assertRaises(ValueError):
            rd.frames(20, overlap=25)

        with self.assertRaises(ValueError):
            rd.frames(200)

    def test_get_pairwise_fr_corr(self):
        """
        Tests get_pairwise_fr_corr() for correct output shape and mathematical invariants.

        Tests:
            (Test Case 1) Returns two (U, U) matrices.
            (Test Case 2) Diagonal of correlation matrix is 1 (self-correlation).
            (Test Case 3) Identical rows produce perfect correlation of 1 and lag of 0.
            (Test Case 4) Both matrices are symmetric.
        """
        n_units, n_times = 4, 80
        rng = np.random.default_rng(42)
        data = rng.random((n_units, n_times))

        # Make rows 0 and 1 identical to ensure perfect correlation.
        data[1] = data[0]

        times = np.arange(n_times, dtype=float)
        rd = RateData(data, times)

        corr, lag = rd.get_pairwise_fr_corr(max_lag=5)

        self.assertEqual(corr.shape, (n_units, n_units))
        self.assertEqual(lag.shape, (n_units, n_units))

        # Diagonal must be 1.
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(n_units))
        # Diagonal lag must be 0.
        np.testing.assert_array_equal(np.diag(lag), np.zeros(n_units))

        # Identical rows → perfect correlation and zero lag.
        self.assertAlmostEqual(corr[0, 1], 1.0, places=5)
        self.assertAlmostEqual(lag[0, 1], 0.0, places=5)

        # Both matrices are symmetric.
        np.testing.assert_array_almost_equal(corr, corr.T)

    def test_get_manifold_pca(self):
        """
        Tests get_manifold() for correct output shape and error handling.

        Tests:
            (Test Case 1) PCA output has shape (T, n_components).
            (Test Case 2) n_components=3 produces correct shape.
            (Test Case 3) Unknown method raises ValueError.
        """
        rd = make_ratedata(n_units=5, n_times=60)

        embedding = rd.get_manifold(method="PCA", n_components=2)
        self.assertEqual(embedding.shape, (60, 2))

        embedding3 = rd.get_manifold(method="PCA", n_components=3)
        self.assertEqual(embedding3.shape, (60, 3))

        with self.assertRaises(ValueError):
            rd.get_manifold(method="TSNE")

    @unittest.skipUnless(UMAP_AVAILABLE, "umap-learn not installed")
    def test_get_manifold_umap(self):
        """
        Tests get_manifold() with UMAP produces correct output shape.

        Tests:
            (Test Case 1) UMAP output has shape (T, n_components).
        """
        rd = make_ratedata(n_units=5, n_times=60)

        embedding = rd.get_manifold(method="UMAP", n_components=2)
        self.assertEqual(embedding.shape, (60, 2))


if __name__ == "__main__":
    unittest.main()
