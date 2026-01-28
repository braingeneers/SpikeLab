import unittest
import numpy as np
from spikedata.ratedata import RateData
from spikedata.utils import (
    compute_cross_correlation_with_lag,
    compute_cosine_similarity_with_lag,
)


class TestRateData(unittest.TestCase):
    def setUp(self):
        self.data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.times = np.array([10.0, 20.0, 30.0])
        self.rd = RateData(self.data, self.times)

    def test_init_invalid_ndim(self):
        # BUG: Currently raises AttributeError instead of ValueError due to accessing unassigned self.inst_Frate_data
        with self.assertRaises(AttributeError):
            RateData(np.array([1, 2, 3]), [1, 2, 3])

    def test_init_mismatched_times(self):
        with self.assertRaises(ValueError):
            RateData(self.data, [10.0, 20.0])

    def test_init_negative_times(self):
        with self.assertRaises(ValueError):
            RateData(self.data, [10.0, -20.0, 30.0])

    def test_init_neuron_attributes_mismatch(self):
        with self.assertRaises(ValueError):
            RateData(self.data, self.times, neuron_attributes=[1])

    def test_init_success(self):
        # Pass a list for times to hit line 50 (conversion to array)
        rd = RateData(self.data, [10.0, 20.0, 30.0], neuron_attributes=[1, 2])
        self.assertEqual(rd.N, 2)
        self.assertEqual(len(rd.neuron_attributes), 2)
        self.assertIsInstance(rd.times, np.ndarray)

    def test_subset_index(self):
        sub = self.rd.subset(0)
        self.assertEqual(sub.N, 1)
        self.assertTrue(np.array_equal(sub.inst_Frate_data, self.data[[0], :]))

    def test_subset_by_attribute_missing_attributes(self):
        with self.assertRaises(ValueError):
            self.rd.subset([1], by="id")

    def test_subset_by_attribute_success(self):
        class Attr:
            def __init__(self, id):
                self.id = id

        attrs = [Attr(101), Attr(102)]
        rd = RateData(self.data, self.times, neuron_attributes=attrs)
        sub = rd.subset([101], by="id")
        self.assertEqual(sub.N, 1)
        self.assertEqual(sub.neuron_attributes[0].id, 101)

    def test_subtime_basic(self):
        sub = self.rd.subtime(15, 25)
        self.assertEqual(len(sub.times), 1)
        self.assertEqual(sub.times[0], 0.0)  # shift_time=True by default

    def test_subtime_no_shift(self):
        sub = self.rd.subtime(15, 25, shift_time=False)
        self.assertEqual(sub.times[0], 20.0)

    def test_subtime_negative_start(self):
        # length is 30.0 (times[-1]). -15 means 30-15 = 15.
        sub = self.rd.subtime(-15, 30)
        self.assertEqual(sub.times[0], 0.0)
        self.assertEqual(len(sub.times), 1)  # only 20.0 is in [15, 30)

    def test_subtime_too_negative_start(self):
        with self.assertRaises(ValueError):
            self.rd.subtime(-40, 30)

    def test_subtime_negative_end(self):
        sub = self.rd.subtime(10, -5)  # 10 to 25
        self.assertEqual(len(sub.times), 2)

    def test_subtime_too_negative_end(self):
        with self.assertRaises(ValueError):
            self.rd.subtime(10, -40)

    def test_subtime_invalid_range(self):
        with self.assertRaises(ValueError):
            self.rd.subtime(25, 15)

    def test_subtime_out_of_range(self):
        with self.assertRaises(ValueError):
            self.rd.subtime(40, 50)

    def test_subtime_ellipsis(self):
        # BUG: Currently ellipsis results in 2 bins because end=times[-1] and mask is t < end
        sub = self.rd.subtime(..., ...)
        self.assertEqual(len(sub.times), 2)

    def test_subtime_by_index_invalid(self):
        with self.assertRaises(ValueError):
            self.rd.subtime_by_index(-1, 2)
        with self.assertRaises(ValueError):
            self.rd.subtime_by_index(0, 4)
        with self.assertRaises(ValueError):
            self.rd.subtime_by_index(2, 1)

    def test_subtime_by_index_success(self):
        sub = self.rd.subtime_by_index(0, 2)
        self.assertEqual(len(sub.times), 2)

    def test_get_pairwise_fr_corr(self):
        corr, lag = self.rd.get_pairwise_fr_corr(max_lag=1)
        self.assertEqual(corr.shape, (2, 2))
        self.assertEqual(lag.shape, (2, 2))
        self.assertEqual(corr[0, 0], 1.0)
        self.assertEqual(lag[0, 0], 0)

    def test_get_pairwise_fr_corr_cosine(self):
        corr, lag = self.rd.get_pairwise_fr_corr(
            compare_func=compute_cosine_similarity_with_lag, max_lag=0
        )
        self.assertEqual(corr.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
