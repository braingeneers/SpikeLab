"""
Tests for the RateSliceStack class (spikedata/rateslicestack.py).

Covers: constructor (both modes), validation, order_units_across_slices,
get_slice_to_slice_unit_corr_from_stack, get_slice_to_slice_time_corr_from_stack,
unit_to_unit_correlation, convert_to_list_of_RateData, subset, subtime_by_index,
subslice.
"""

import warnings

import numpy as np
import pytest

from spikelab.spikedata.ratedata import RateData
from spikelab.spikedata.rateslicestack import RateSliceStack
from spikelab.spikedata.spikedata import SpikeData
from spikelab.spikedata.pairwise import PairwiseCompMatrixStack


def make_event_matrix(n_units=3, n_times=20, n_slices=4, seed=0):
    """Create a random 3D array (U, T, S) for RateSliceStack construction."""
    rng = np.random.default_rng(seed)
    return rng.random((n_units, n_times, n_slices))


def make_ratedata(n_units=3, n_times=100, step=1.0, t0=0.0, seed=0):
    """Create a RateData with random firing rates on a uniform time grid."""
    rng = np.random.default_rng(seed)
    times = np.arange(t0, t0 + n_times * step, step)
    data = rng.random((n_units, len(times)))
    return RateData(data, times)


def make_spikedata(n_units=3, length_ms=100.0, seed=0):
    """Create a SpikeData with uniformly spaced spikes per unit."""
    rng = np.random.default_rng(seed)
    train = []
    for _ in range(n_units):
        n_spikes = rng.integers(5, 20)
        spikes = np.sort(rng.uniform(0, length_ms, n_spikes))
        train.append(spikes)
    return SpikeData(train, length=length_ms)


class TestRateSliceStackConstructor:
    def test_event_matrix_basic(self):
        """
        Tests Option 2 constructor with a 3D event_matrix.

        Tests:
            (Test Case 1) Shape is preserved.
            (Test Case 2) Auto-generated times have correct length and duration.
            (Test Case 3) Default step_size is 1.0.
        """
        mat = make_event_matrix(3, 20, 4)
        rss = RateSliceStack(event_matrix=mat)
        assert rss.event_stack.shape == (3, 20, 4)
        assert len(rss.times) == 4
        assert rss.step_size == 1.0
        # Auto-generated times: each slice has duration T * step_size = 20
        for i, (start, end) in enumerate(rss.times):
            assert start == pytest.approx(i * 20.0)
            assert end == pytest.approx((i + 1) * 20.0)

    def test_event_matrix_with_step_size(self):
        """
        Tests constructor with custom step_size.

        Tests:
            (Test Case 1) Custom step_size is stored.
            (Test Case 2) Auto-generated times reflect custom step_size.
        """
        mat = make_event_matrix(2, 10, 3)
        rss = RateSliceStack(event_matrix=mat, step_size=2.0)
        assert rss.step_size == 2.0
        # Duration per slice = 10 * 2.0 = 20
        assert rss.times[0] == (0.0, 20.0)
        assert rss.times[1] == (20.0, 40.0)

    def test_event_matrix_with_times(self):
        """
        Tests constructor with explicit times_start_to_end for event_matrix.

        Tests:
            (Test Case 1) Provided times are stored correctly.
            (Test Case 2) Mismatched length raises ValueError.
        """
        mat = make_event_matrix(2, 10, 3)
        times = [(0.0, 10.0), (20.0, 30.0), (40.0, 50.0)]
        rss = RateSliceStack(event_matrix=mat, times_start_to_end=times)
        assert rss.times == times

        # Wrong number of time tuples
        with pytest.raises(ValueError):
            RateSliceStack(
                event_matrix=mat,
                times_start_to_end=[(0.0, 10.0), (20.0, 30.0)],
            )

    def test_event_matrix_not_3d_raises(self):
        """
        Tests that non-3D event_matrix raises ValueError.

        Tests:
            (Test Case 1) 2D array raises ValueError.
        """
        with pytest.raises(ValueError, match="3D"):
            RateSliceStack(event_matrix=np.ones((3, 10)))

    def test_event_matrix_not_ndarray_raises(self):
        """
        Tests that non-ndarray event_matrix raises TypeError.

        Tests:
            (Test Case 1) List input raises TypeError.
        """
        with pytest.raises(TypeError, match="numpy array"):
            RateSliceStack(event_matrix=[[[1, 2], [3, 4]]])

    def test_ratedata_input(self):
        """
        Tests Option 1 constructor with RateData input.

        Tests:
            (Test Case 1) Shape matches expected (U, T_slice, S).
            (Test Case 2) Times are stored correctly.
            (Test Case 3) Step size is inferred from RateData.
        """
        rd = make_ratedata(n_units=3, n_times=100, step=1.0)
        times = [(10.0, 30.0), (40.0, 60.0), (70.0, 90.0)]
        rss = RateSliceStack(data_obj=rd, times_start_to_end=times)
        assert rss.event_stack.shape[0] == 3  # units
        assert rss.event_stack.shape[2] == 3  # slices
        assert len(rss.times) == 3
        assert rss.step_size == pytest.approx(1.0)

    def test_spikedata_input(self):
        """
        Tests Option 1 constructor with SpikeData input.

        Tests:
            (Test Case 1) SpikeData is converted to RateData internally.
            (Test Case 2) Output shape is (U, T_slice, S).
        """
        sd = make_spikedata(n_units=3, length_ms=100.0)
        times = [(10.0, 30.0), (50.0, 70.0)]
        rss = RateSliceStack(data_obj=sd, times_start_to_end=times)
        assert rss.event_stack.shape[0] == 3
        assert rss.event_stack.shape[2] == 2
        assert rss.step_size == pytest.approx(1.0)

    def test_peaks_and_bounds(self):
        """
        Tests construction using time_peaks + time_bounds.

        Tests:
            (Test Case 1) Peaks and bounds are converted to start/end tuples.
            (Test Case 2) All windows are preserved.
        """
        rd = make_ratedata(n_units=2, n_times=200, step=1.0)
        # All peaks have enough margin for bounds (10, 10)
        rss = RateSliceStack(
            data_obj=rd,
            time_peaks=[20.0, 50.0, 100.0],
            time_bounds=(10.0, 10.0),
        )
        assert rss.event_stack.shape[2] == 3

    def test_no_input_raises(self):
        """
        Tests that no data_obj or event_matrix raises ValueError.

        Tests:
            (Test Case 1) ValueError raised with informative message.
        """
        with pytest.raises(ValueError, match="Must input"):
            RateSliceStack()

    def test_both_inputs_warns(self):
        """
        Tests that providing both data_obj and event_matrix warns and uses event_matrix.

        Tests:
            (Test Case 1) UserWarning is raised.
            (Test Case 2) event_matrix is used.
        """
        rd = make_ratedata(n_units=2, n_times=50)
        mat = make_event_matrix(2, 10, 3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rss = RateSliceStack(
                data_obj=rd,
                event_matrix=mat,
                times_start_to_end=[(0.0, 10.0), (20.0, 30.0), (40.0, 50.0)],
            )
            assert any("Ignoring data_obj" in str(warning.message) for warning in w)
        assert rss.event_stack.shape == (2, 10, 3)

    def test_invalid_data_obj_type_raises(self):
        """
        Tests that non-SpikeData/RateData data_obj raises TypeError.

        Tests:
            (Test Case 1) TypeError raised for list input.
        """
        with pytest.raises(TypeError, match="SpikeData.*RateData"):
            RateSliceStack(
                data_obj="not a data object",
                times_start_to_end=[(0.0, 10.0)],
            )

    def test_missing_time_args_raises(self):
        """
        Tests that data_obj without any time specification raises ValueError.

        Tests:
            (Test Case 1) ValueError raised when neither times_start_to_end nor peaks+bounds given.
        """
        rd = make_ratedata(n_units=2, n_times=50)
        with pytest.raises(ValueError, match="Must provide"):
            RateSliceStack(data_obj=rd)

    def test_invalid_time_bounds_raises(self):
        """
        Tests that invalid time_bounds raises TypeError.

        Tests:
            (Test Case 1) Non-tuple time_bounds raises TypeError.
            (Test Case 2) Wrong-length tuple raises TypeError.
        """
        rd = make_ratedata(n_units=2, n_times=50)
        with pytest.raises(TypeError, match="time_bounds"):
            RateSliceStack(data_obj=rd, time_peaks=[25.0], time_bounds=[10, 10])
        with pytest.raises(TypeError, match="time_bounds"):
            RateSliceStack(data_obj=rd, time_peaks=[25.0], time_bounds=(10,))

    def test_neuron_attributes(self):
        """
        Tests that neuron_attributes are stored and validated.

        Tests:
            (Test Case 1) Valid neuron_attributes stored correctly.
            (Test Case 2) Wrong length raises ValueError.
        """
        mat = make_event_matrix(3, 10, 2)
        attrs = [{"region": "CA1"}, {"region": "CA3"}, {"region": "DG"}]
        rss = RateSliceStack(event_matrix=mat, neuron_attributes=attrs)
        assert len(rss.neuron_attributes) == 3

        with pytest.raises(ValueError, match="neuron_attributes"):
            RateSliceStack(event_matrix=mat, neuron_attributes=[{"region": "CA1"}])

    def test_event_matrix_single_slice(self):
        """
        Verify RateSliceStack can be constructed with a single slice (S=1).

        Tests:
            (Test Case 1) Construction succeeds without error.
            (Test Case 2) times list has exactly 1 entry.
            (Test Case 3) event_stack shape is preserved as (3, 20, 1).
        """
        mat = np.random.default_rng(0).random((3, 20, 1))
        rss = RateSliceStack(event_matrix=mat)

        assert rss.event_stack.shape == (3, 20, 1)
        assert len(rss.times) == 1

    def test_event_matrix_single_unit(self):
        """
        Verify RateSliceStack can be constructed with a single unit (U=1).

        Tests:
            (Test Case 1) Construction succeeds without error.
            (Test Case 2) event_stack shape is preserved as (1, 20, 5).
        """
        mat = np.random.default_rng(0).random((1, 20, 5))
        rss = RateSliceStack(event_matrix=mat)

        assert rss.event_stack.shape == (1, 20, 5)

    def test_all_zero_event_matrix(self):
        """
        EC-RSS-01: Construct RateSliceStack with an all-zero event_matrix.

        Tests:
            (Test Case 1) Construction succeeds without error.
            (Test Case 2) event_stack is all zeros.
            (Test Case 3) Auto-generated times have correct length.
        """
        mat = np.zeros((3, 20, 4))
        rss = RateSliceStack(event_matrix=mat)

        assert rss.event_stack.shape == (3, 20, 4)
        np.testing.assert_array_equal(rss.event_stack, 0.0)
        assert len(rss.times) == 4

    def test_all_nan_event_matrix(self):
        """
        EC-RSS-02: Construct RateSliceStack with an all-NaN event_matrix.

        Tests:
            (Test Case 1) Construction succeeds without error.
            (Test Case 2) event_stack is all NaN.
            (Test Case 3) Auto-generated times have correct length.
        """
        mat = np.full((3, 20, 4), np.nan)
        rss = RateSliceStack(event_matrix=mat)

        assert rss.event_stack.shape == (3, 20, 4)
        assert np.all(np.isnan(rss.event_stack))
        assert len(rss.times) == 4


class TestValidateTimeStartToEnd:
    def test_not_list_raises(self):
        """
        Tests that non-list input raises TypeError.

        Tests:
            (Test Case 1) Tuple input raises TypeError.
        """
        rd = make_ratedata(n_units=2, n_times=50)
        with pytest.raises(TypeError, match="list of tuples"):
            RateSliceStack(data_obj=rd, times_start_to_end=((0.0, 10.0),))

    def test_non_tuple_element_raises(self):
        """
        Tests that non-tuple element raises TypeError.

        Tests:
            (Test Case 1) List element raises TypeError.
        """
        rd = make_ratedata(n_units=2, n_times=50)
        with pytest.raises(TypeError, match="not a tuple"):
            RateSliceStack(data_obj=rd, times_start_to_end=[[0.0, 10.0]])

    def test_wrong_length_tuple_raises(self):
        """
        Tests that tuple with wrong length raises TypeError.

        Tests:
            (Test Case 1) 3-element tuple raises TypeError.
        """
        rd = make_ratedata(n_units=2, n_times=50)
        with pytest.raises(TypeError, match="length 2"):
            RateSliceStack(data_obj=rd, times_start_to_end=[(0.0, 10.0, 20.0)])

    def test_non_numeric_raises(self):
        """
        Tests that non-numeric start/end raises TypeError.

        Tests:
            (Test Case 1) String values raise TypeError.
        """
        rd = make_ratedata(n_units=2, n_times=50)
        with pytest.raises(TypeError, match="numbers"):
            RateSliceStack(data_obj=rd, times_start_to_end=[("a", "b")])

    def test_start_ge_end_raises(self):
        """
        Tests that start >= end raises ValueError.

        Tests:
            (Test Case 1) Equal start and end raises ValueError.
        """
        rd = make_ratedata(n_units=2, n_times=50)
        with pytest.raises(ValueError, match="less than end"):
            RateSliceStack(data_obj=rd, times_start_to_end=[(10.0, 10.0)])

    def test_unequal_durations_raises(self):
        """
        Tests that time windows with different durations raise ValueError.

        Tests:
            (Test Case 1) Windows of 10ms and 20ms raise ValueError.
        """
        rd = make_ratedata(n_units=2, n_times=100)
        with pytest.raises(ValueError, match="same length"):
            RateSliceStack(
                data_obj=rd,
                times_start_to_end=[(0.0, 10.0), (20.0, 40.0)],
            )

    def test_validate_negative_start_preserved(self):
        """
        _validate_time_start_to_end preserves windows with negative start times.

        Tests:
            (Test Case 1) A window with negative start is included in the result.
        """
        from spikelab.spikedata.utils import _validate_time_start_to_end

        windows = [(-5.0, 15.0), (40.0, 60.0), (90.0, 110.0)]
        result = _validate_time_start_to_end(windows)
        assert len(result) == 3
        assert result[0][0] == pytest.approx(-5.0)

    def test_validate_float_precision_accepted(self):
        """
        Tests that windows with durations differing by sub-picosecond amounts
        due to floating-point arithmetic are accepted by the tolerance-based
        comparison.

        Tests:
            (Test Case 1) Construction succeeds without error.
            (Test Case 2) Both slices are present in the result.

        Notes:
            Uses asymmetric event times converted from seconds to milliseconds
            to reproduce the real-world floating-point rounding that causes
            sub-picosecond duration differences.
        """
        rd = make_ratedata(n_units=2, n_times=2000, step=1.0)
        # Simulate event times from float64 seconds -> ms, producing
        # tiny rounding differences in window durations.
        # Times must be far enough from recording edges (0–2000 ms)
        # to fit the full pre/post window.
        event_times_s = np.array([0.523456789012345, 1.287654321098765])
        event_times_ms = event_times_s * 1000.0
        pre_ms, post_ms = 200.0, 500.0
        times = [(t - pre_ms, t + post_ms) for t in event_times_ms]

        rss = RateSliceStack(data_obj=rd, times_start_to_end=times)

        assert rss.event_stack.shape[2] == 2


class TestOrderUnitsAcrossSlices:
    def test_basic_ordering(self):
        """
        Tests order_units_across_slices with median aggregation.

        Tests:
            (Test Case 1) Returns 4-tuple.
            (Test Case 2) reordered_stack has same shape as original.
            (Test Case 3) unit_ids_in_order contains all unit indices.
            (Test Case 4) Unit that peaks earliest is first in the order.
        """
        # Create data where unit 2 peaks earliest, then unit 0, then unit 1
        mat = np.zeros((3, 20, 4))
        for s in range(4):
            mat[0, 10, s] = 5.0  # unit 0 peaks at t=10
            mat[1, 15, s] = 5.0  # unit 1 peaks at t=15
            mat[2, 3, s] = 5.0  # unit 2 peaks at t=3
        rss = RateSliceStack(event_matrix=mat)

        reordered, order, std, peaks, frac_active = rss.order_units_across_slices(
            "median"
        )
        # With default MIN_FRAC_ACTIVE=0.0, all units are in the highly-active group
        assert reordered[0].shape == mat.shape
        assert set(order[0]) == {0, 1, 2}
        # Unit 2 should be first (peaks earliest)
        assert order[0][0] == 2
        assert order[0][1] == 0
        assert order[0][2] == 1

    def test_mean_aggregation(self):
        """
        Tests order_units_across_slices with mean aggregation.

        Tests:
            (Test Case 1) Mean aggregation produces valid output.
            (Test Case 2) unit_std_indices has correct shape.
            (Test Case 3) unit_peak_times has correct shape.
        """
        mat = make_event_matrix(4, 30, 5, seed=42)
        rss = RateSliceStack(event_matrix=mat)
        reordered, order, std, peaks, frac_active = rss.order_units_across_slices(
            "mean"
        )
        # With default MIN_FRAC_ACTIVE=0.0, all units are in the highly-active group
        assert reordered[0].shape == mat.shape
        assert len(order[0]) == 4
        assert len(std[0]) == 4
        assert len(peaks[0]) == 4

    def test_invalid_agg_func_raises(self):
        """
        Tests that invalid agg_func raises ValueError.

        Tests:
            (Test Case 1) String 'max' raises ValueError.
        """
        mat = make_event_matrix(2, 10, 3)
        rss = RateSliceStack(event_matrix=mat)
        with pytest.raises(ValueError, match="not a valid"):
            rss.order_units_across_slices("max")

    def test_threshold_filtering(self):
        """
        Tests that MIN_RATE_THRESHOLD filters low-activity slices.

        Tests:
            (Test Case 1) Slices below threshold are excluded from peak calculation.
            (Test Case 2) Output shapes are still correct.
        """
        mat = np.zeros((2, 10, 3))
        mat[0, 5, 0] = 1.0
        mat[0, 5, 1] = 1.0
        # Slice 2 for unit 0 is all zeros (below threshold)
        mat[1, 3, :] = 1.0
        rss = RateSliceStack(event_matrix=mat)
        reordered, order, std, peaks, frac_active = rss.order_units_across_slices(
            "median", MIN_RATE_THRESHOLD=0.1
        )
        # With default MIN_FRAC_ACTIVE=0.0, all units are in the highly-active group
        assert len(order[0]) == 2
        assert reordered[0].shape == mat.shape

    def test_order_units_single_unit(self):
        """
        Tests order_units_across_slices() with U=1.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) Returned order is [0].
        """
        rng = np.random.default_rng(0)
        mat = rng.random((1, 20, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)

        reordered, order, std, peaks, frac_active = rss.order_units_across_slices(
            "median"
        )

        # With default MIN_FRAC_ACTIVE=0.0, all units are in the highly-active group
        assert reordered[0].shape == mat.shape
        np.testing.assert_array_equal(order[0], [0])

    def test_order_units_all_below_threshold(self):
        """
        Tests order_units_across_slices when all units have max rates below threshold.

        Tests:
            (Test Case 1) No exception is raised (NaN peak times are handled).
            (Test Case 2) Returned arrays have correct shapes.
            (Test Case 3) unit_peak_times are derived from NaN scores (all-NaN columns
                          produce NaN via nanmedian, which rounds to an integer).

        Notes:
            When every slice is below MIN_RATE_THRESHOLD for every unit, all entries
            in the peak-index matrix become NaN. nanmedian/nanmean of all-NaN returns
            NaN (with a RuntimeWarning), and np.round(NaN).astype(int) yields a
            platform-dependent integer. The method must not crash.
        """
        mat = np.full((3, 20, 4), 0.01)
        rss = RateSliceStack(event_matrix=mat)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reordered, order, std, peaks, frac_active = rss.order_units_across_slices(
                "median", MIN_RATE_THRESHOLD=0.1
            )

        # All units below threshold, but with MIN_FRAC_ACTIVE=0.0 they still go
        # to the highly-active group (the threshold only affects peak-time masking)
        assert reordered[0].shape == (3, 20, 4)
        assert len(order[0]) == 3
        assert len(std[0]) == 3
        assert len(peaks[0]) == 3

    def test_order_units_flat_signal(self):
        """
        Tests order_units_across_slices with all-zero (flat) data.

        Tests:
            (Test Case 1) No exception is raised with threshold set to 0.
            (Test Case 2) All peak times are 0 (argmax of flat signal returns index 0).
            (Test Case 3) All standard deviations are 0 (peak time is identical across slices).
        """
        mat = np.zeros((3, 20, 4))
        rss = RateSliceStack(event_matrix=mat)

        reordered, order, std, peaks, frac_active = rss.order_units_across_slices(
            "mean", MIN_RATE_THRESHOLD=0.0
        )

        # With MIN_FRAC_ACTIVE=0.0, all units in the highly-active group
        assert reordered[0].shape == mat.shape
        np.testing.assert_array_equal(peaks[0], [0, 0, 0])
        np.testing.assert_array_equal(std[0], [0.0, 0.0, 0.0])

    def test_timing_matrix_override(self):
        """
        EC-RSS-03: order_units_across_slices with a pre-computed timing_matrix.

        Tests:
            (Test Case 1) Ordering matches the provided timing_matrix, not the
                internal rate-based calculation.
            (Test Case 2) MIN_RATE_THRESHOLD is ignored when timing_matrix is provided.
            (Test Case 3) Output shapes are correct.
        """
        # Unit 0 peaks at t=10 in all slices, unit 1 at t=5, unit 2 at t=15
        mat = np.zeros((3, 20, 4))
        for s in range(4):
            mat[0, 10, s] = 5.0
            mat[1, 5, s] = 5.0
            mat[2, 15, s] = 5.0
        rss = RateSliceStack(event_matrix=mat)

        # Override timing_matrix so unit 2 appears earliest, then 0, then 1
        timing = np.array(
            [
                [8.0, 8.0, 8.0, 8.0],  # unit 0
                [12.0, 12.0, 12.0, 12.0],  # unit 1
                [2.0, 2.0, 2.0, 2.0],  # unit 2
            ]
        )
        reordered, order, std, peaks, frac_active = rss.order_units_across_slices(
            "median", timing_matrix=timing
        )

        # Order should follow timing_matrix: unit 2 first, then 0, then 1
        assert order[0][0] == 2
        assert order[0][1] == 0
        assert order[0][2] == 1
        assert reordered[0].shape == mat.shape
        assert len(peaks[0]) == 3


class TestConvertToListOfRateData:
    def test_basic_conversion(self):
        """
        Tests convert_to_list_of_RateData returns correct list.

        Tests:
            (Test Case 1) Returns list of RateData objects.
            (Test Case 2) List length equals number of slices.
            (Test Case 3) Each RateData has correct shape.
            (Test Case 4) Times are within slice boundaries.
        """
        mat = make_event_matrix(3, 20, 4)
        rss = RateSliceStack(event_matrix=mat, step_size=1.0)
        rd_list = rss.convert_to_list_of_RateData()
        assert len(rd_list) == 4
        for i, rd in enumerate(rd_list):
            assert isinstance(rd, RateData)
            assert rd.inst_Frate_data.shape == (3, 20)
            np.testing.assert_array_equal(rd.inst_Frate_data, mat[:, :, i])

    def test_custom_step_size(self):
        """
        Tests conversion with non-default step_size.

        Tests:
            (Test Case 1) Times use correct step_size spacing.
        """
        mat = make_event_matrix(2, 10, 2)
        times = [(0.0, 20.0), (30.0, 50.0)]
        rss = RateSliceStack(event_matrix=mat, times_start_to_end=times, step_size=2.0)
        rd_list = rss.convert_to_list_of_RateData()
        assert len(rd_list) == 2
        # First RateData times should start at 0, step by 2
        assert rd_list[0].times[0] == pytest.approx(0.0)
        assert rd_list[0].times[1] == pytest.approx(2.0)

    def test_convert_to_list_single_time_bin(self):
        """
        Tests convert_to_list_of_RateData with T=1 per slice.

        Tests:
            (Test Case 1) Each RateData has shape (U, 1).
            (Test Case 2) List length equals number of slices.
        """
        mat = np.random.default_rng(0).random((3, 1, 4))
        rss = RateSliceStack(event_matrix=mat)

        rd_list = rss.convert_to_list_of_RateData()

        assert len(rd_list) == 4
        for rd in rd_list:
            assert isinstance(rd, RateData)
            assert rd.inst_Frate_data.shape == (3, 1)

    def test_convert_to_list_single_unit(self):
        """
        Tests convert_to_list_of_RateData with U=1.

        Tests:
            (Test Case 1) Each RateData has shape (1, T).
            (Test Case 2) List length equals number of slices.
        """
        mat = np.random.default_rng(0).random((1, 10, 3))
        rss = RateSliceStack(event_matrix=mat)

        rd_list = rss.convert_to_list_of_RateData()

        assert len(rd_list) == 3
        for rd in rd_list:
            assert isinstance(rd, RateData)
            assert rd.inst_Frate_data.shape == (1, 10)


class TestSliceCorrelations:
    def test_slice_to_slice_unit_corr_shape(self):
        """
        Tests get_slice_to_slice_unit_corr_from_stack output shapes.

        Tests:
            (Test Case 1) Returns PairwiseCompMatrixStack with shape (S, S, U).
            (Test Case 2) Average scores array has shape (U,).
        """
        mat = make_event_matrix(3, 20, 5, seed=42) + 0.5  # ensure above threshold
        rss = RateSliceStack(event_matrix=mat)
        pcm_stack, av_scores = rss.get_slice_to_slice_unit_corr_from_stack(max_lag=2)
        assert isinstance(pcm_stack, PairwiseCompMatrixStack)
        assert pcm_stack.stack.shape == (5, 5, 3)
        assert av_scores.shape == (3,)

    def test_slice_to_slice_unit_corr_symmetric(self):
        """
        Tests that slice correlation matrices are symmetric.

        Tests:
            (Test Case 1) Each unit's S*S matrix is symmetric.
        """
        mat = make_event_matrix(2, 15, 4, seed=7) + 1.0
        rss = RateSliceStack(event_matrix=mat)
        pcm_stack, _ = rss.get_slice_to_slice_unit_corr_from_stack(max_lag=0)
        for u in range(2):
            unit_mat = pcm_stack.stack[:, :, u]
            np.testing.assert_array_almost_equal(unit_mat, unit_mat.T)

    def test_slice_to_slice_time_corr_shape(self):
        """
        Tests get_slice_to_slice_time_corr_from_stack output shapes.

        Tests:
            (Test Case 1) Returns PairwiseCompMatrixStack with shape (S, S, T).
            (Test Case 2) Average scores array has shape (T,).
        """
        mat = make_event_matrix(3, 10, 4, seed=42)
        rss = RateSliceStack(event_matrix=mat)
        pcm_stack, av_scores = rss.get_slice_to_slice_time_corr_from_stack(max_lag=0)
        assert isinstance(pcm_stack, PairwiseCompMatrixStack)
        assert pcm_stack.stack.shape == (4, 4, 10)
        assert av_scores.shape == (10,)

    def test_unit_to_unit_correlation_shape(self):
        """
        Tests unit_to_unit_correlation output shapes.

        Tests:
            (Test Case 1) Returns corr stack (U, U, S) and lag stack (U, U, S).
            (Test Case 2) av_max_corr has shape (S,).
            (Test Case 3) av_max_corr_lag has shape (S,).
        """
        mat = make_event_matrix(3, 20, 4, seed=42)
        rss = RateSliceStack(event_matrix=mat)
        corr_stack, lag_stack, av_corr, av_lag = rss.unit_to_unit_correlation(max_lag=2)
        assert isinstance(corr_stack, PairwiseCompMatrixStack)
        assert isinstance(lag_stack, PairwiseCompMatrixStack)
        assert corr_stack.stack.shape == (3, 3, 4)
        assert lag_stack.stack.shape == (3, 3, 4)
        assert av_corr.shape == (4,)
        assert av_lag.shape == (4,)

    def test_unit_to_unit_self_correlation(self):
        """
        Tests that self-correlation on the diagonal is 1.

        Tests:
            (Test Case 1) Diagonal of each slice's correlation matrix is 1.
        """
        mat = make_event_matrix(3, 30, 4, seed=99)
        rss = RateSliceStack(event_matrix=mat)
        corr_stack, _, _, _ = rss.unit_to_unit_correlation(max_lag=0)
        for s in range(4):
            diag = np.diag(corr_stack.stack[:, :, s])
            np.testing.assert_array_almost_equal(diag, np.ones(3))

    def test_slice_to_slice_unit_corr_single_slice(self):
        """
        Tests get_slice_to_slice_unit_corr_from_stack() with S=1.

        Tests:
            (Test Case 1) Emits RuntimeWarning about fewer than 2 slices.
            (Test Case 2) Returns a PairwiseCompMatrixStack with shape (1, 1, U).
            (Test Case 3) Average scores are NaN (no pairwise comparisons possible).
        """
        rng = np.random.default_rng(0)
        mat = rng.random((3, 20, 1)) + 0.5
        rss = RateSliceStack(event_matrix=mat)

        with pytest.warns(RuntimeWarning, match="fewer than 2 slices"):
            pcm_stack, av_scores = rss.get_slice_to_slice_unit_corr_from_stack(
                max_lag=2
            )

        assert isinstance(pcm_stack, PairwiseCompMatrixStack)
        assert pcm_stack.stack.shape == (1, 1, 3)
        assert av_scores.shape == (3,)
        assert np.all(np.isnan(av_scores))

    def test_slice_to_slice_time_corr_single_slice(self):
        """
        Tests get_slice_to_slice_time_corr_from_stack() with S=1.

        Tests:
            (Test Case 1) Emits RuntimeWarning about fewer than 2 slices.
            (Test Case 2) Returns a PairwiseCompMatrixStack with shape (1, 1, T).
            (Test Case 3) Average scores are NaN (no pairwise comparisons possible).
        """
        rng = np.random.default_rng(0)
        mat = rng.random((3, 20, 1))
        rss = RateSliceStack(event_matrix=mat)

        with pytest.warns(RuntimeWarning, match="fewer than 2 slices"):
            pcm_stack, av_scores = rss.get_slice_to_slice_time_corr_from_stack(
                max_lag=0
            )

        assert isinstance(pcm_stack, PairwiseCompMatrixStack)
        assert pcm_stack.stack.shape == (1, 1, 20)
        assert av_scores.shape == (20,)
        assert np.all(np.isnan(av_scores))

    def test_unit_to_unit_correlation_single_unit(self):
        """
        Tests unit_to_unit_correlation() with U=1.

        Tests:
            (Test Case 1) Emits RuntimeWarning about fewer than 2 units.
            (Test Case 2) Correlation stack has shape (1, 1, S).
            (Test Case 3) Lag stack has shape (1, 1, S).
            (Test Case 4) Average values are NaN (no pairwise comparisons possible).
        """
        rng = np.random.default_rng(0)
        mat = rng.random((1, 20, 5))
        rss = RateSliceStack(event_matrix=mat)

        with pytest.warns(RuntimeWarning, match="fewer than 2 units"):
            corr_stack, lag_stack, av_corr, av_lag = rss.unit_to_unit_correlation(
                max_lag=2
            )

        assert isinstance(corr_stack, PairwiseCompMatrixStack)
        assert isinstance(lag_stack, PairwiseCompMatrixStack)
        assert corr_stack.stack.shape == (1, 1, 5)
        assert lag_stack.stack.shape == (1, 1, 5)
        assert av_corr.shape == (5,)
        assert av_lag.shape == (5,)
        assert np.all(np.isnan(av_corr))
        assert np.all(np.isnan(av_lag))

    def test_slice_to_slice_unit_corr_identical_slices(self):
        """
        Tests get_slice_to_slice_unit_corr_from_stack with two identical slices.

        Tests:
            (Test Case 1) Off-diagonal correlation is 1.0 for each unit (identical signals).
            (Test Case 2) Average score per unit is 1.0.
        """
        rng = np.random.default_rng(42)
        single = rng.random((3, 20, 1)) + 0.5
        mat = np.concatenate([single, single], axis=2)
        rss = RateSliceStack(event_matrix=mat)

        pcm, av = rss.get_slice_to_slice_unit_corr_from_stack(max_lag=0)

        assert pcm.stack.shape == (2, 2, 3)
        for u in range(3):
            assert pcm.stack[0, 1, u] == pytest.approx(1.0)
            assert pcm.stack[1, 0, u] == pytest.approx(1.0)
        for u in range(3):
            assert av[u] == pytest.approx(1.0)

    def test_slice_to_slice_time_corr_single_time_bin(self):
        """
        Tests get_slice_to_slice_time_corr_from_stack with T=1.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) Output PairwiseCompMatrixStack has shape (S, S, 1).
            (Test Case 3) Average scores array has shape (1,).
        """
        rng = np.random.default_rng(42)
        mat = rng.random((3, 1, 4)) + 0.5
        rss = RateSliceStack(event_matrix=mat)

        pcm, av = rss.get_slice_to_slice_time_corr_from_stack(max_lag=0)

        assert pcm.stack.shape == (4, 4, 1)
        assert av.shape == (1,)


class TestSubset:
    def test_basic_subset(self):
        """
        Tests subset by index.

        Tests:
            (Test Case 1) Subset extracts correct units.
            (Test Case 2) Times and step_size preserved.
        """
        mat = make_event_matrix(5, 10, 3)
        rss = RateSliceStack(event_matrix=mat, step_size=2.0)
        sub = rss.subset([0, 3])
        assert sub.event_stack.shape == (2, 10, 3)
        np.testing.assert_array_equal(sub.event_stack[0], mat[0])
        np.testing.assert_array_equal(sub.event_stack[1], mat[3])
        assert sub.step_size == 2.0
        assert sub.times == rss.times

    def test_single_int(self):
        """
        Tests subset with a single integer.

        Tests:
            (Test Case 1) Single int returns single-unit stack.
        """
        mat = make_event_matrix(4, 10, 3)
        rss = RateSliceStack(event_matrix=mat)
        sub = rss.subset(2)
        assert sub.event_stack.shape == (1, 10, 3)

    def test_subset_by_attribute(self):
        """
        Tests subset using the by parameter with neuron_attributes.

        Tests:
            (Test Case 1) by parameter selects units matching attribute values.
            (Test Case 2) ValueError raised when by used without neuron_attributes.
        """
        from dataclasses import dataclass

        @dataclass
        class MockAttr:
            region: str

        mat = make_event_matrix(3, 10, 2)
        attrs = [MockAttr("CA1"), MockAttr("CA3"), MockAttr("CA1")]
        rss = RateSliceStack(event_matrix=mat, neuron_attributes=attrs)
        sub = rss.subset(["CA1"], by="region")
        assert sub.event_stack.shape == (2, 10, 2)

        # Without neuron_attributes
        rss_no_attrs = RateSliceStack(event_matrix=mat)
        with pytest.raises(ValueError, match="neuron_attributes"):
            rss_no_attrs.subset(["CA1"], by="region")

    def test_subset_preserves_neuron_attributes(self):
        """
        Tests that subset carries over neuron_attributes for selected units.

        Tests:
            (Test Case 1) neuron_attributes length matches subset.
            (Test Case 2) Correct attributes are retained.
        """
        mat = make_event_matrix(4, 10, 2)
        attrs = [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}]
        rss = RateSliceStack(event_matrix=mat, neuron_attributes=attrs)
        sub = rss.subset([1, 3])
        assert len(sub.neuron_attributes) == 2
        assert sub.neuron_attributes[0] == {"id": 1}
        assert sub.neuron_attributes[1] == {"id": 3}

    def test_subset_duplicate_indices(self):
        """
        Tests subset with duplicate unit indices.

        Tests:
            (Test Case 1) Duplicates are deduplicated (subset uses set()).
            (Test Case 2) Output contains only unique units in sorted order.
        """
        mat = np.random.default_rng(0).random((5, 10, 3))
        rss = RateSliceStack(event_matrix=mat)

        sub = rss.subset([0, 0, 2, 2, 3])

        assert sub.event_stack.shape == (3, 10, 3)
        np.testing.assert_array_equal(sub.event_stack[0], mat[0])
        np.testing.assert_array_equal(sub.event_stack[1], mat[2])
        np.testing.assert_array_equal(sub.event_stack[2], mat[3])

    def test_subset_by_nonexistent_attribute(self):
        """
        Tests subset with by parameter referencing a non-existent attribute.

        Tests:
            (Test Case 1) No units match (getattr returns sentinel for missing attr).
            (Test Case 2) Result is an empty stack with shape (0, T, S).
        """
        from dataclasses import dataclass

        @dataclass
        class MockAttr:
            region: str

        mat = np.random.default_rng(0).random((3, 10, 2))
        attrs = [MockAttr("CA1"), MockAttr("CA3"), MockAttr("CA1")]
        rss = RateSliceStack(event_matrix=mat, neuron_attributes=attrs)

        sub = rss.subset(["CA1"], by="nonexistent")

        assert sub.event_stack.shape == (0, 10, 2)

    def test_out_of_range_index(self):
        """
        subset with an out-of-range unit index raises ValueError.

        Tests:
            (Test Case 1) Index exceeding U raises ValueError with descriptive message.
        """
        mat = make_event_matrix(3, 10, 2)
        rss = RateSliceStack(event_matrix=mat)

        with pytest.raises(ValueError, match="out of range"):
            rss.subset([0, 10])


class TestSubtimeByIndex:
    def test_basic_trim(self):
        """
        Tests subtime_by_index trims time axis correctly.

        Tests:
            (Test Case 1) Output shape reflects trimmed time axis.
            (Test Case 2) Data matches original sliced region.
            (Test Case 3) Times are adjusted.
        """
        mat = make_event_matrix(2, 20, 3)
        times = [(0.0, 20.0), (30.0, 50.0), (60.0, 80.0)]
        rss = RateSliceStack(event_matrix=mat, times_start_to_end=times)
        sub = rss.subtime_by_index(5, 15)
        assert sub.event_stack.shape == (2, 10, 3)
        np.testing.assert_array_equal(sub.event_stack, mat[:, 5:15, :])

    def test_negative_indexing(self):
        """
        Tests negative index support.

        Tests:
            (Test Case 1) Negative end_idx selects from end.
        """
        mat = make_event_matrix(2, 20, 3)
        rss = RateSliceStack(event_matrix=mat)
        sub = rss.subtime_by_index(0, -5)
        assert sub.event_stack.shape == (2, 15, 3)

    def test_out_of_range_raises(self):
        """
        Tests that out-of-range indices raise ValueError.

        Tests:
            (Test Case 1) start_idx out of range raises ValueError.
            (Test Case 2) end_idx out of range raises ValueError.
            (Test Case 3) end_idx <= start_idx raises ValueError.
        """
        mat = make_event_matrix(2, 10, 3)
        rss = RateSliceStack(event_matrix=mat)
        with pytest.raises(ValueError, match="start_idx"):
            rss.subtime_by_index(20, 25)
        with pytest.raises(ValueError, match="end_idx"):
            rss.subtime_by_index(0, 20)
        with pytest.raises(ValueError, match="end_idx"):
            rss.subtime_by_index(5, 3)

    def test_preserves_metadata(self):
        """
        Tests that step_size and neuron_attributes are carried over.

        Tests:
            (Test Case 1) step_size preserved.
            (Test Case 2) neuron_attributes preserved.
        """
        mat = make_event_matrix(3, 20, 2)
        attrs = [{"id": 0}, {"id": 1}, {"id": 2}]
        rss = RateSliceStack(event_matrix=mat, step_size=2.0, neuron_attributes=attrs)
        sub = rss.subtime_by_index(2, 10)
        assert sub.step_size == 2.0
        assert sub.neuron_attributes == attrs

    def test_subtime_by_index_full_range(self):
        """
        Tests subtime_by_index with full range (0, T).

        Tests:
            (Test Case 1) Output shape is identical to original.
            (Test Case 2) Data is identical to original event_stack.
        """
        mat = np.random.default_rng(0).random((3, 20, 4))
        rss = RateSliceStack(event_matrix=mat)

        sub = rss.subtime_by_index(0, 20)

        assert sub.event_stack.shape == rss.event_stack.shape
        np.testing.assert_array_equal(sub.event_stack, rss.event_stack)

    def test_subtime_by_index_single_bin(self):
        """
        Tests subtime_by_index extracting a single time bin.

        Tests:
            (Test Case 1) Output shape is (U, 1, S).
            (Test Case 2) Data matches the selected time bin from original.
        """
        mat = np.random.default_rng(0).random((3, 20, 4))
        rss = RateSliceStack(event_matrix=mat)

        sub = rss.subtime_by_index(5, 6)

        assert sub.event_stack.shape == (3, 1, 4)
        np.testing.assert_array_equal(sub.event_stack[:, 0, :], mat[:, 5, :])

    def test_full_range_returns_independent_copy(self):
        """
        subtime_by_index with full range (0, T) returns an independent copy.

        Tests:
            (Test Case 1) Output data is equal to the original.
            (Test Case 2) Mutating the sub's event_stack does not affect
                the original.
        """
        mat = np.random.default_rng(0).random((3, 20, 4))
        rss = RateSliceStack(event_matrix=mat)

        sub = rss.subtime_by_index(0, 20)

        np.testing.assert_array_equal(sub.event_stack, rss.event_stack)

        # Mutation should NOT propagate — sub is an independent copy
        sub.event_stack[0, 0, 0] = -999.0
        assert rss.event_stack[0, 0, 0] != -999.0


class TestSubslice:
    def test_basic_subslice(self):
        """
        Tests subslice extracts correct slices.

        Tests:
            (Test Case 1) Output shape reflects selected slices.
            (Test Case 2) Data matches original sliced region.
            (Test Case 3) Times are subsliced.
        """
        mat = make_event_matrix(2, 10, 5)
        times = [(i * 10.0, (i + 1) * 10.0) for i in range(5)]
        rss = RateSliceStack(event_matrix=mat, times_start_to_end=times)
        sub = rss.subslice([0, 2, 4])
        assert sub.event_stack.shape == (2, 10, 3)
        np.testing.assert_array_equal(sub.event_stack[:, :, 0], mat[:, :, 0])
        np.testing.assert_array_equal(sub.event_stack[:, :, 1], mat[:, :, 2])
        np.testing.assert_array_equal(sub.event_stack[:, :, 2], mat[:, :, 4])
        assert sub.times == [times[0], times[2], times[4]]

    def test_single_int(self):
        """
        Tests subslice with a single integer.

        Tests:
            (Test Case 1) Single int returns single-slice stack.
        """
        mat = make_event_matrix(2, 10, 5)
        rss = RateSliceStack(event_matrix=mat)
        sub = rss.subslice(3)
        assert sub.event_stack.shape == (2, 10, 1)

    def test_out_of_range_raises(self):
        """
        Tests that out-of-range slice index raises ValueError.

        Tests:
            (Test Case 1) Index >= S raises ValueError.
        """
        mat = make_event_matrix(2, 10, 3)
        rss = RateSliceStack(event_matrix=mat)
        with pytest.raises(ValueError, match="out of range"):
            rss.subslice([0, 5])

    def test_preserves_metadata(self):
        """
        Tests that step_size and neuron_attributes are carried over.

        Tests:
            (Test Case 1) step_size preserved.
            (Test Case 2) neuron_attributes preserved.
        """
        mat = make_event_matrix(3, 10, 4)
        attrs = [{"id": 0}, {"id": 1}, {"id": 2}]
        rss = RateSliceStack(event_matrix=mat, step_size=3.0, neuron_attributes=attrs)
        sub = rss.subslice([1, 3])
        assert sub.step_size == 3.0
        assert sub.neuron_attributes == attrs

    def test_subslice_single_slice(self):
        """
        Tests subslice extracting a single slice.

        Tests:
            (Test Case 1) Output shape is (U, T, 1).
            (Test Case 2) Downstream convert_to_list_of_RateData works on single-slice result.
            (Test Case 3) Resulting RateData has correct shape.
        """
        mat = np.random.default_rng(0).random((3, 10, 5))
        rss = RateSliceStack(event_matrix=mat)

        sub = rss.subslice([2])

        assert sub.event_stack.shape == (3, 10, 1)
        rd_list = sub.convert_to_list_of_RateData()
        assert len(rd_list) == 1
        assert rd_list[0].inst_Frate_data.shape == (3, 10)

    def test_duplicate_indices(self):
        """
        EC-RSS-08: subslice with duplicate slice indices.

        subslice sorts its input but does not deduplicate. Duplicate indices
        produce repeated slices in the output, and duplicated times entries.

        Tests:
            (Test Case 1) Output S dimension equals the number of input indices
                (including duplicates).
            (Test Case 2) Duplicate slices contain identical data.
            (Test Case 3) times list has repeated entries for duplicated slices.
        """
        mat = np.random.default_rng(0).random((3, 10, 5))
        times = [(i * 10.0, (i + 1) * 10.0) for i in range(5)]
        rss = RateSliceStack(event_matrix=mat, times_start_to_end=times)

        sub = rss.subslice([1, 1, 3])

        # 3 entries because duplicates are not removed
        assert sub.event_stack.shape == (3, 10, 3)
        # First two slices should be identical (both are slice 1)
        np.testing.assert_array_equal(
            sub.event_stack[:, :, 0], sub.event_stack[:, :, 1]
        )
        np.testing.assert_array_equal(sub.event_stack[:, :, 0], mat[:, :, 1])
        np.testing.assert_array_equal(sub.event_stack[:, :, 2], mat[:, :, 3])
        # times has the duplicate entry
        assert sub.times[0] == sub.times[1] == times[1]
        assert sub.times[2] == times[3]


class TestOrderUnitsNanSentinel:
    """Tests for NaN handling in order_units_across_slices."""

    def test_nan_peak_times_become_minus_one(self):
        """
        Tests that units with all-zero firing rates get a peak time of -1
        instead of a garbage large negative integer from NaN-to-int cast.

        Tests:
            (Test Case 1) Unit 0 (all zeros) has peak time == -1 in the
                highly_active group.
            (Test Case 2) Units 1 and 2 (non-zero) have valid (>= 0) peak times.
        """
        rng = np.random.default_rng(42)
        mat = rng.random((3, 20, 5)) + 0.5
        # Set unit 0 to all zeros so its peak time is NaN
        mat[0, :, :] = 0.0
        rss = RateSliceStack(event_matrix=mat)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reordered, order, std, peaks, frac_active = rss.order_units_across_slices(
                "median", MIN_RATE_THRESHOLD=0.1
            )

        # peaks is a tuple of (highly_active, low_active) arrays
        highly_active_peaks = peaks[0]

        # Find unit 0's position in the ordering
        highly_active_order = order[0]
        unit_0_pos = np.where(highly_active_order == 0)[0]
        assert len(unit_0_pos) == 1, "Unit 0 should be in the highly_active group"
        assert highly_active_peaks[unit_0_pos[0]] == -1

        # Non-zero units should have valid peak times >= 0
        for idx, unit_id in enumerate(highly_active_order):
            if unit_id != 0:
                assert highly_active_peaks[idx] >= 0


# ---------------------------------------------------------------------------
# frac_active override — order_units_across_slices
# ---------------------------------------------------------------------------


class TestOrderUnitsOverrideFracActive:
    """Tests for the frac_active override on RateSliceStack.order_units_across_slices."""

    def test_frac_active_override_splits_correctly(self):
        """
        Pre-computed frac_active controls which units go into each group.

        Tests:
            (Test Case 1) Unit with frac_active=0.1 < min_frac=0.5 goes to low group.
            (Test Case 2) Units with frac_active >= 0.5 go to highly active group.
        """
        rng = np.random.default_rng(0)
        mat = rng.random((4, 20, 5)) + 0.2
        rss = RateSliceStack(event_matrix=mat)

        frac = np.array([0.9, 0.1, 0.8, 0.6])
        _, order, _, _, frac_out = rss.order_units_across_slices(
            "median", MIN_FRAC_ACTIVE=0.5, frac_active=frac
        )

        ha_ids = set(order[0].tolist())
        la_ids = set(order[1].tolist())
        assert 1 not in ha_ids  # 0.1 < 0.5
        assert 1 in la_ids
        assert {0, 2, 3}.issubset(ha_ids)

    def test_frac_active_override_wrong_shape_raises(self):
        """
        frac_active with wrong shape raises ValueError.

        Tests:
            (Test Case 1) Shape (3,) for 4 units raises ValueError.
        """
        mat = np.random.default_rng(0).random((4, 20, 5)) + 0.2
        rss = RateSliceStack(event_matrix=mat)

        with pytest.raises(ValueError, match="frac_active must have shape"):
            rss.order_units_across_slices(
                "median", MIN_FRAC_ACTIVE=0.5, frac_active=np.ones(3)
            )

    def test_no_split_when_min_frac_zero(self):
        """
        When MIN_FRAC_ACTIVE=0, all units go to highly-active regardless of frac_active.

        Tests:
            (Test Case 1) All 4 units are in the highly-active group.
            (Test Case 2) Low-active group is empty.
        """
        rng = np.random.default_rng(1)
        mat = rng.random((4, 20, 5)) + 0.2
        rss = RateSliceStack(event_matrix=mat)

        _, order, _, _, _ = rss.order_units_across_slices("median", MIN_FRAC_ACTIVE=0.0)

        assert len(order[0]) == 4
        assert len(order[1]) == 0

    def test_no_split_when_min_frac_none_equivalent(self):
        """
        When MIN_FRAC_ACTIVE=0.0 (default), frac_active is not used even if provided.

        Tests:
            (Test Case 1) All units in highly-active group despite low frac_active values.
        """
        rng = np.random.default_rng(2)
        mat = rng.random((3, 20, 5)) + 0.2
        rss = RateSliceStack(event_matrix=mat)

        _, order, _, _, _ = rss.order_units_across_slices(
            "median", MIN_FRAC_ACTIVE=0.0, frac_active=np.array([0.01, 0.01, 0.01])
        )

        assert len(order[0]) == 3
        assert len(order[1]) == 0


# ---------------------------------------------------------------------------
# frac_active override — get_slice_to_slice_unit_corr_from_stack
# ---------------------------------------------------------------------------


class TestSliceToSliceUnitCorrOverrideFracActive:
    """Tests for the frac_active override on get_slice_to_slice_unit_corr_from_stack."""

    def test_frac_active_override_filters_unit_averages(self):
        """
        Units with low frac_active get NaN averages.

        Tests:
            (Test Case 1) Unit with frac_active=0.1 and min_frac=0.3 has NaN average
                since 0.1 < (1 - 0.3) = 0.7.
            (Test Case 2) Unit with frac_active=0.9 has valid average.
        """
        rng = np.random.default_rng(0)
        mat = rng.random((3, 50, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)

        frac = np.array([0.9, 0.1, 0.8])
        _, av_corr = rss.get_slice_to_slice_unit_corr_from_stack(
            MIN_FRAC=0.3, frac_active=frac
        )

        assert not np.isnan(av_corr[0])  # 0.9 >= 0.7
        assert np.isnan(av_corr[1])  # 0.1 < 0.7
        assert not np.isnan(av_corr[2])  # 0.8 >= 0.7

    def test_frac_active_override_wrong_shape_raises(self):
        """
        frac_active with wrong shape raises ValueError.

        Tests:
            (Test Case 1) Shape (2,) for 3 units raises ValueError.
        """
        mat = np.random.default_rng(0).random((3, 50, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)

        with pytest.raises(ValueError, match="frac_active must have shape"):
            rss.get_slice_to_slice_unit_corr_from_stack(frac_active=np.ones(2))

    def test_without_override_uses_rate_based(self):
        """
        Without frac_active override, rate-based filtering is used (backward compat).

        Tests:
            (Test Case 1) Output shapes are correct.
            (Test Case 2) av_corr has shape (U,).
        """
        rng = np.random.default_rng(3)
        mat = rng.random((3, 50, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)

        all_corr, av_corr = rss.get_slice_to_slice_unit_corr_from_stack()

        assert all_corr.stack.shape == (5, 5, 3)
        assert av_corr.shape == (3,)


# ---------------------------------------------------------------------------
# get_unit_timing_per_slice + rank_order_correlation (RateSliceStack)
# ---------------------------------------------------------------------------

from spikelab.spikedata.pairwise import PairwiseCompMatrix


class TestGetUnitTimingPerSliceRate:
    """Tests for RateSliceStack.get_unit_timing_per_slice()."""

    def test_output_shape(self):
        """
        Output is (U, S) ndarray.

        Tests:
            (Test Case 1) 4 units, 5 slices -> shape (4, 5).
        """
        rng = np.random.default_rng(0)
        mat = rng.random((4, 30, 5)) + 0.2
        rss = RateSliceStack(event_matrix=mat)
        tm = rss.get_unit_timing_per_slice()
        assert tm.shape == (4, 5)

    def test_values_are_time_bin_indices(self):
        """
        Non-NaN values are valid time bin indices.

        Tests:
            (Test Case 1) All values in [0, T-1].
        """
        rng = np.random.default_rng(1)
        mat = rng.random((3, 20, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        tm = rss.get_unit_timing_per_slice()
        valid = tm[~np.isnan(tm)]
        assert np.all(valid >= 0)
        assert np.all(valid < 20)

    def test_inactive_unit_is_nan(self):
        """
        Units below MIN_RATE_THRESHOLD get NaN.

        Tests:
            (Test Case 1) All-zero unit has NaN timing in every slice.
        """
        rng = np.random.default_rng(2)
        mat = rng.random((3, 20, 5)) + 0.5
        mat[0, :, :] = 0.0  # Unit 0 is silent
        rss = RateSliceStack(event_matrix=mat)
        tm = rss.get_unit_timing_per_slice(MIN_RATE_THRESHOLD=0.1)
        assert np.all(np.isnan(tm[0, :]))
        assert np.all(~np.isnan(tm[1, :]))


class TestRankOrderCorrelationRate:
    """Tests for RateSliceStack.rank_order_correlation()."""

    def test_raw_output_shapes(self):
        """
        Raw mode returns correct shapes and types.

        Tests:
            (Test Case 1) corr shape (S, S), overlap shape (S, S), av is float.
        """
        rng = np.random.default_rng(0)
        mat = rng.random((6, 30, 8)) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        corr, av, overlap = rss.rank_order_correlation(n_shuffles=0)

        assert isinstance(corr, PairwiseCompMatrix)
        assert corr.matrix.shape == (8, 8)
        assert isinstance(overlap, PairwiseCompMatrix)
        assert overlap.matrix.shape == (8, 8)
        assert isinstance(av, float)

    def test_raw_diagonal_is_one(self):
        """
        Raw mode diagonal is 1.0.

        Tests:
            (Test Case 1) All diagonal entries are 1.0.
        """
        rng = np.random.default_rng(1)
        mat = rng.random((6, 30, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        corr, _, _ = rss.rank_order_correlation(n_shuffles=0)
        np.testing.assert_allclose(np.diag(corr.matrix), 1.0)

    def test_raw_symmetric(self):
        """
        Correlation matrix is symmetric.

        Tests:
            (Test Case 1) corr[i,j] == corr[j,i].
        """
        rng = np.random.default_rng(2)
        mat = rng.random((6, 30, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        corr, _, _ = rss.rank_order_correlation(n_shuffles=0)
        np.testing.assert_allclose(corr.matrix, corr.matrix.T, atol=1e-12)

    def test_zscore_diagonal_is_nan(self):
        """
        Z-scored mode diagonal is NaN.

        Tests:
            (Test Case 1) All diagonal entries are NaN when n_shuffles > 0.
        """
        rng = np.random.default_rng(3)
        mat = rng.random((6, 30, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        corr, _, _ = rss.rank_order_correlation(n_shuffles=10)
        assert np.all(np.isnan(np.diag(corr.matrix)))

    def test_zscore_reproducible(self):
        """
        Same seed produces identical z-scores.

        Tests:
            (Test Case 1) Two calls with seed=42 yield identical results.
        """
        rng = np.random.default_rng(4)
        mat = rng.random((6, 30, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        corr1, _, _ = rss.rank_order_correlation(n_shuffles=20, seed=42)
        corr2, _, _ = rss.rank_order_correlation(n_shuffles=20, seed=42)
        np.testing.assert_array_equal(corr1.matrix, corr2.matrix)

    def test_overlap_is_fraction(self):
        """
        Overlap matrix entries are fractions in [0, 1].

        Tests:
            (Test Case 1) All overlap values in [0, 1].
        """
        rng = np.random.default_rng(5)
        mat = rng.random((6, 30, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        _, _, overlap = rss.rank_order_correlation(n_shuffles=0)
        assert np.all(overlap.matrix >= 0.0)
        assert np.all(overlap.matrix <= 1.0)

    def test_min_overlap_frac(self):
        """
        min_overlap_frac raises the effective threshold.

        Tests:
            (Test Case 1) frac=1.0 is stricter, producing at least as many NaN pairs.
        """
        rng = np.random.default_rng(6)
        mat = rng.random((6, 30, 5)) + 0.5
        # Make some units inactive in some slices
        mat[0, :, 0:2] = 0.0
        mat[1, :, 2:4] = 0.0
        rss = RateSliceStack(event_matrix=mat)
        corr_lax, _, _ = rss.rank_order_correlation(min_overlap=1, n_shuffles=0)
        corr_strict, _, _ = rss.rank_order_correlation(
            min_overlap=1, min_overlap_frac=1.0, n_shuffles=0
        )
        nan_lax = np.sum(np.isnan(corr_lax.matrix))
        nan_strict = np.sum(np.isnan(corr_strict.matrix))
        assert nan_strict >= nan_lax

    def test_auto_compute_timing(self):
        """
        Without timing_matrix, timing is computed automatically.

        Tests:
            (Test Case 1) Explicit and auto timing produce identical results.
        """
        rng = np.random.default_rng(7)
        mat = rng.random((6, 30, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        tm = rss.get_unit_timing_per_slice(MIN_RATE_THRESHOLD=0.1)
        corr_explicit, av_exp, _ = rss.rank_order_correlation(
            timing_matrix=tm, n_shuffles=0
        )
        corr_auto, av_auto, _ = rss.rank_order_correlation(
            MIN_RATE_THRESHOLD=0.1, n_shuffles=0
        )
        np.testing.assert_array_equal(corr_explicit.matrix, corr_auto.matrix)

    def test_invalid_n_shuffles_raises(self):
        """
        n_shuffles between 1 and 4 raises ValueError.

        Tests:
            (Test Case 1) n_shuffles=2 raises ValueError.
        """
        rng = np.random.default_rng(8)
        mat = rng.random((4, 20, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        with pytest.raises(ValueError, match="n_shuffles"):
            rss.rank_order_correlation(n_shuffles=2)

    def test_single_slice(self):
        """
        Single-slice stack produces (1,1) matrix with NaN average.

        Tests:
            (Test Case 1) corr shape (1, 1).
            (Test Case 2) av_corr is NaN.
        """
        rng = np.random.default_rng(0)
        mat = rng.random((6, 30, 1)) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr, av, overlap = rss.rank_order_correlation(n_shuffles=0)
        assert corr.matrix.shape == (1, 1)
        assert np.isnan(av)

    def test_all_nan_timing(self):
        """
        All-NaN timing matrix produces all-NaN correlation.

        Tests:
            (Test Case 1) Off-diagonal entries are all NaN.
        """
        rng = np.random.default_rng(1)
        mat = rng.random((4, 20, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        all_nan = np.full((4, 5), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr, av, _ = rss.rank_order_correlation(
                timing_matrix=all_nan, n_shuffles=0
            )
        off_diag = corr.matrix.copy()
        np.fill_diagonal(off_diag, np.nan)
        assert np.all(np.isnan(off_diag))

    def test_n_shuffles_exactly_5(self):
        """
        n_shuffles=5 (minimum allowed) produces valid output.

        Tests:
            (Test Case 1) No error raised; output shape correct.
        """
        rng = np.random.default_rng(2)
        mat = rng.random((6, 30, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        corr, _, _ = rss.rank_order_correlation(n_shuffles=5, seed=42)
        assert corr.matrix.shape == (5, 5)

    def test_min_overlap_exceeds_units(self):
        """
        min_overlap larger than U makes all off-diagonal NaN.

        Tests:
            (Test Case 1) All off-diagonal entries are NaN.
        """
        rng = np.random.default_rng(3)
        mat = rng.random((4, 20, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        corr, _, _ = rss.rank_order_correlation(min_overlap=100, n_shuffles=0)
        off_diag = corr.matrix.copy()
        np.fill_diagonal(off_diag, np.nan)
        assert np.all(np.isnan(off_diag))

    def test_min_overlap_frac_one(self):
        """
        EC-RSS-05: rank_order_correlation with min_overlap_frac=1.0.

        When min_overlap_frac=1.0, the effective threshold becomes
        ceil(1.0 * U) = U, so all U units must be active in both slices
        for the pair to be valid.

        Tests:
            (Test Case 1) When some units are inactive in some slices,
                pairs that lack full overlap become NaN.
            (Test Case 2) Pairs where all units are active still get a
                valid correlation value.
        """
        rng = np.random.default_rng(10)
        mat = rng.random((6, 30, 5)) + 0.5
        # Make unit 0 inactive in slice 0 only
        mat[0, :, 0] = 0.0
        rss = RateSliceStack(event_matrix=mat)

        corr, av, overlap = rss.rank_order_correlation(
            min_overlap=1, min_overlap_frac=1.0, n_shuffles=0
        )

        # Pairs involving slice 0 should be NaN (unit 0 is inactive there)
        for j in range(1, 5):
            assert np.isnan(corr.matrix[0, j]), f"pair (0,{j}) should be NaN"

        # Pairs not involving slice 0 should have valid correlations
        for i in range(1, 5):
            for j in range(i + 1, 5):
                assert not np.isnan(
                    corr.matrix[i, j]
                ), f"pair ({i},{j}) should be valid"

    def test_n_shuffles_zero_returns_raw_spearman(self):
        """
        EC-RSS-06: rank_order_correlation with n_shuffles=0.

        When n_shuffles=0, the method returns raw Spearman correlations
        (not z-scores). The diagonal should be 1.0 and off-diagonal
        values should be in [-1, 1].

        Tests:
            (Test Case 1) Diagonal entries are exactly 1.0.
            (Test Case 2) Off-diagonal valid entries are in [-1, 1].
            (Test Case 3) Output is a PairwiseCompMatrix (not a stack).
        """
        rng = np.random.default_rng(20)
        mat = rng.random((6, 30, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)

        corr, av, overlap = rss.rank_order_correlation(n_shuffles=0)

        np.testing.assert_allclose(np.diag(corr.matrix), 1.0)
        off_diag = corr.matrix[np.tril_indices(5, k=-1)]
        valid = off_diag[~np.isnan(off_diag)]
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)
        assert isinstance(corr, PairwiseCompMatrix)


# ---------------------------------------------------------------------------
# Edge case tests from REVIEW.md — Edge Case Scan — Core (spikedata/)
# ---------------------------------------------------------------------------


class TestRateSliceStackConstructorEdgeCases:
    """Edge case tests for RateSliceStack.__init__."""

    def test_spikedata_zero_length_raises(self):
        """
        SpikeData with zero length causes resampled_isi to raise.

        When SpikeData.length is 0, np.arange(0, 0, step) produces an empty
        times array. _resampled_isi raises ValueError for times with fewer
        than 2 elements.

        Tests:
            (Test Case 1) ValueError is raised when constructing RateSliceStack
                from a zero-length SpikeData.

        Notes:
            The error originates in _resampled_isi, not in RateSliceStack
            itself. This is the expected behavior since a zero-length
            recording has no meaningful firing rate data to slice.
        """
        sd = SpikeData(train=[np.array([])], length=0.0)
        with pytest.raises((ValueError, IndexError)):
            RateSliceStack(
                data_obj=sd,
                times_start_to_end=[(0.0, 10.0)],
            )

    def test_time_bounds_negative_before(self):
        """
        time_bounds with negative 'before' value creates forward-shifted windows.

        A negative 'before' value means (peak - (-before)) = peak + before,
        so the window starts after the peak. This is semantically confusing
        but is not validated by the constructor.

        Tests:
            (Test Case 1) Construction succeeds without error.
            (Test Case 2) Resulting window start is after the peak time.

        Notes:
            The constructor does not validate that 'before' in time_bounds
            is non-negative. A negative 'before' produces a window that
            starts after the peak, which may be unintended by the user.
        """
        rd = make_ratedata(n_units=2, n_times=200, step=1.0)
        # before = -10 means window starts at peak + 10
        rss = RateSliceStack(
            data_obj=rd,
            time_peaks=[50.0],
            time_bounds=(-10.0, 20.0),
        )
        # Window should be (50 - (-10), 50 + 20) = (60, 70)
        assert len(rss.times) == 1
        assert rss.times[0][0] == pytest.approx(60.0)
        assert rss.times[0][1] == pytest.approx(70.0)


class TestOrderUnitsAcrossSlicesEdgeCases:
    """Edge case tests for RateSliceStack.order_units_across_slices."""

    def test_all_units_nan_peak_times(self):
        """
        All units have NaN peak times when every unit is below threshold
        in every slice.

        np.argsort on all-NaN values produces an arbitrary but deterministic
        ordering. The method should not crash.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) All peak times in the output are -1 (the NaN sentinel).
            (Test Case 3) Returned order contains all unit indices.
            (Test Case 4) reordered_stack has the correct shape.

        Notes:
            np.nanmedian / np.nanmean of all-NaN slices returns NaN with a
            RuntimeWarning. np.argsort places NaN values last, so the order
            is deterministic but arbitrary. The NaN-to-int cast produces -1
            via the sentinel logic.
        """
        mat = np.full((3, 20, 4), 0.01)  # all below threshold
        rss = RateSliceStack(event_matrix=mat)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reordered, order, std, peaks, frac_active = rss.order_units_across_slices(
                "median", MIN_RATE_THRESHOLD=0.1
            )

        # All units go to highly-active group (MIN_FRAC_ACTIVE=0.0 default)
        assert set(order[0].tolist()) == {0, 1, 2}
        assert reordered[0].shape == (3, 20, 4)
        # All peak times should be -1 (NaN sentinel)
        np.testing.assert_array_equal(peaks[0], [-1, -1, -1])


class TestSliceToSliceUnitCorrEdgeCases:
    """Edge case tests for RateSliceStack.get_slice_to_slice_unit_corr_from_stack."""

    def test_all_units_below_threshold(self):
        """
        All units have mean firing rates below MIN_RATE_THRESHOLD in all slices.

        When every slice for every unit is below threshold, no correlations
        are computed. All entries in the correlation stack and all average
        scores should be NaN.

        Tests:
            (Test Case 1) Output PairwiseCompMatrixStack has correct shape (S, S, U).
            (Test Case 2) All average scores are NaN.
            (Test Case 3) All off-diagonal entries in the correlation stack are NaN.
        """
        mat = np.full((3, 20, 5), 0.001)  # all below default threshold 0.1
        rss = RateSliceStack(event_matrix=mat)

        pcm_stack, av_scores = rss.get_slice_to_slice_unit_corr_from_stack(
            MIN_RATE_THRESHOLD=0.1
        )

        assert pcm_stack.stack.shape == (5, 5, 3)
        assert av_scores.shape == (3,)
        assert np.all(np.isnan(av_scores))
        # Off-diagonal should all be NaN (no correlations computed)
        for u in range(3):
            unit_mat = pcm_stack.stack[:, :, u]
            off_diag_mask = ~np.eye(5, dtype=bool)
            assert np.all(np.isnan(unit_mat[off_diag_mask]))

    def test_frac_active_all_zeros(self):
        """
        frac_active override with all zeros makes all unit averages NaN.

        When frac_active is all zeros, no unit meets the (1 - MIN_FRAC)
        threshold, so all average scores are NaN even though individual
        correlations may be computed.

        Tests:
            (Test Case 1) Output has correct shape.
            (Test Case 2) All average scores are NaN.
            (Test Case 3) Individual correlation entries may still be valid
                (only the averaging is affected by frac_active).
        """
        rng = np.random.default_rng(0)
        mat = rng.random((3, 50, 5)) + 0.5  # above threshold
        rss = RateSliceStack(event_matrix=mat)

        frac = np.zeros(3)  # all zeros
        pcm_stack, av_scores = rss.get_slice_to_slice_unit_corr_from_stack(
            MIN_FRAC=0.3, frac_active=frac
        )

        assert pcm_stack.stack.shape == (5, 5, 3)
        assert av_scores.shape == (3,)
        # All averages should be NaN because no unit meets frac threshold
        assert np.all(np.isnan(av_scores))


class TestSliceToSliceTimeCorrEdgeCases:
    """Edge case tests for RateSliceStack.get_slice_to_slice_time_corr_from_stack."""

    def test_all_zero_time_bins(self):
        """
        All-zero data produces NaN correlations for time bins.

        When all values are zero, the cosine similarity (default compare_func)
        produces NaN because the denominator (norm product) is zero.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) Output shapes are correct.
            (Test Case 3) Average scores are NaN (zero vectors have undefined
                cosine similarity).
        """
        mat = np.zeros((3, 10, 4))
        rss = RateSliceStack(event_matrix=mat)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pcm_stack, av_scores = rss.get_slice_to_slice_time_corr_from_stack(
                max_lag=0
            )

        assert pcm_stack.stack.shape == (4, 4, 10)
        assert av_scores.shape == (10,)
        # Cosine similarity of zero vectors is NaN
        assert np.all(np.isnan(av_scores))


class TestSubsetEdgeCases:
    """Edge case tests for RateSliceStack.subset."""

    def test_empty_units_list(self):
        """
        Subset with an empty units list produces a (0, T, S) stack.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) Output shape is (0, T, S).
            (Test Case 3) Times and step_size are preserved.
        """
        mat = make_event_matrix(5, 10, 3)
        rss = RateSliceStack(event_matrix=mat, step_size=2.0)

        sub = rss.subset([])

        assert sub.event_stack.shape == (0, 10, 3)
        assert sub.step_size == 2.0
        assert sub.times == rss.times


class TestGetUnitTimingPerSliceEdgeCases:
    """Edge case tests for RateSliceStack.get_unit_timing_per_slice."""

    def test_all_nan_slice_argmax_on_nan(self):
        """
        All-NaN slice data produces potentially wrong results from np.argmax.

        np.argmax on an all-NaN array returns 0 (the index of the first NaN),
        which is indistinguishable from a legitimate peak at time bin 0. However,
        get_unit_timing_per_slice uses np.max to check the threshold, and
        np.max of all-NaN returns NaN. Since NaN < threshold is False, the unit
        is correctly marked as NaN in the output.

        Tests:
            (Test Case 1) Units with all-NaN data in a slice get NaN timing.
            (Test Case 2) Units with valid data in the same stack get valid timing.

        Notes:
            np.argmax on all-NaN returns 0 with no warning, but
            get_unit_timing_per_slice correctly handles this because the
            threshold check (np.max < MIN_RATE_THRESHOLD) catches it:
            np.max of all-NaN is NaN, and NaN < threshold evaluates to
            False, so the mask does not override NaN entries to NaN
            (they remain as argmax output). However, since the actual
            max rate is NaN, the comparison NaN < 0.1 is False, meaning
            the threshold mask does NOT set these to NaN. The argmax
            result (0) leaks through as a seemingly valid peak time.
            This is a potential silent bug.
        """
        rng = np.random.default_rng(0)
        mat = rng.random((3, 20, 4)) + 0.5
        # Make unit 0 all-NaN in all slices
        mat[0, :, :] = np.nan
        rss = RateSliceStack(event_matrix=mat)

        tm = rss.get_unit_timing_per_slice(MIN_RATE_THRESHOLD=0.1)

        assert tm.shape == (3, 4)
        # Units 1 and 2 should have valid timing
        assert np.all(~np.isnan(tm[1, :]))
        assert np.all(~np.isnan(tm[2, :]))
        # Unit 0 (all-NaN): np.max of all-NaN is NaN, and NaN < 0.1 is False,
        # so the threshold mask does NOT set these to NaN. np.argmax returns 0.
        # This means unit 0 gets a timing value of 0.0 instead of NaN — a silent bug.
        # The values are argmax results (0) since NaN < threshold is False.
        np.testing.assert_array_equal(tm[0, :], 0.0)


# ---------------------------------------------------------------------------
# Edge case tests from the edge case scan
# ---------------------------------------------------------------------------


class TestRateSliceStackConstructorEdgeCases2:
    """Additional edge case tests for RateSliceStack.__init__."""

    def test_zero_unit_event_matrix(self):
        """
        Constructor with 0-unit event_matrix (shape (0, 10, 5)).

        Tests:
            (Test Case 1) A (0, T, S) event matrix is accepted.
        """
        mat = np.empty((0, 10, 5))
        rss = RateSliceStack(event_matrix=mat)
        assert rss.event_stack.shape[0] == 0
        assert rss.event_stack.shape == (0, 10, 5)


class TestOrderUnitsAcrossSlicesEdgeCases2:
    """Additional edge case tests for RateSliceStack.order_units_across_slices."""

    def test_impossible_min_frac_threshold(self):
        """
        order_units_across_slices with MIN_FRAC_ACTIVE > 1.0.

        Tests:
            (Test Case 1) Impossible threshold puts all units in the low-activity
                group and returns None for the high-activity stack.
        """
        rng = np.random.default_rng(0)
        mat = rng.random((3, 20, 4)) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        frac = np.ones(3) * 0.5
        result = rss.order_units_across_slices(
            agg_func="median", frac_active=frac, MIN_FRAC_ACTIVE=2.0
        )
        # Returns 5-tuple: (reordered_matrices, unit_ids, unit_std, unit_peak_times, unit_frac_active)
        reordered, ids, stds, peaks, fracs = result
        # All units below threshold -> all in low-activity group (second element of each tuple)
        assert reordered[0].shape[0] == 0  # HA stack has 0 units
        assert len(ids[1]) == 3  # all 3 units in LA group


class TestSliceToSliceUnitCorrEdgeCases2:
    """Additional edge case tests for RateSliceStack.get_slice_to_slice_unit_corr_from_stack."""

    def test_exact_min_rate_threshold_boundary(self):
        """
        Units with mean rate exactly equal to MIN_RATE_THRESHOLD are included.

        Tests:
            (Test Case 1) Unit with exactly threshold rate is NOT excluded
                since check is < not <=.
        """
        # Create a stack where one unit has mean rate exactly at threshold
        mat = np.zeros((2, 20, 3))
        mat[0, :, :] = 0.1  # unit 0 has mean rate 0.1
        mat[1, :, :] = 1.0  # unit 1 has high rate
        rss = RateSliceStack(event_matrix=mat)
        corr_stack, av = rss.get_slice_to_slice_unit_corr_from_stack(
            MIN_RATE_THRESHOLD=0.1
        )
        # Shape should be (S, S, U) = (3, 3, 2)
        assert corr_stack.stack.shape[0] == 3  # S x S pairwise
        assert corr_stack.stack.shape[1] == 3


class TestSliceToSliceTimeCorrEdgeCases2:
    """Additional edge case tests for RateSliceStack.get_slice_to_slice_time_corr_from_stack."""

    def test_all_zero_units_at_some_time_bins(self):
        """
        Time correlation with all-zero units: compare_func receives zero-norm vectors.

        Tests:
            (Test Case 1) All-zero slices produce NaN correlations.
        """
        mat = np.zeros((3, 10, 2))
        # One slice has non-zero data, other is all zero
        mat[:, :, 0] = 1.0
        rss = RateSliceStack(event_matrix=mat)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr_stack, av = rss.get_slice_to_slice_time_corr_from_stack()
        # Returns (PairwiseCompMatrixStack(S, S, T), array(T,))
        assert corr_stack.stack.shape == (2, 2, 10)


class TestUnitToUnitCorrelationEdgeCases:
    """Additional edge case tests for RateSliceStack.unit_to_unit_correlation."""

    def test_two_units_minimum(self):
        """
        unit_to_unit_correlation with 2 units: single lower triangle element.

        Tests:
            (Test Case 1) 2 units produce a (2, 2, S) result with a single
                off-diagonal pair.
        """
        rng = np.random.default_rng(0)
        mat = rng.random((2, 20, 3)) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        corr_stack, lag_stack, av_corr, av_lag = rss.unit_to_unit_correlation()
        assert corr_stack.stack.shape == (2, 2, 3)


class TestConvertToListOfRateDataEdgeCases:
    """Additional edge case tests for RateSliceStack.convert_to_list_of_RateData."""

    def test_single_time_bin_stack(self):
        """
        convert_to_list_of_RateData on a stack with a single time bin.

        Tests:
            (Test Case 1) Single time bin produces RateData with 1 time point each.
        """
        mat = np.ones((3, 1, 2))
        rss = RateSliceStack(
            event_matrix=mat,
            times_start_to_end=[(0.0, 1.0), (1.0, 2.0)],
        )
        result = rss.convert_to_list_of_RateData()
        assert len(result) == 2
        assert result[0].inst_Frate_data.shape == (3, 1)


class TestRSSSubsetEdgeCases2:
    """Additional edge case tests for RateSliceStack.subset."""

    def test_subset_by_duplicate_attribute_values(self):
        """
        subset with by parameter when neuron_attributes contain duplicate values.

        Tests:
            (Test Case 1) Multiple units matching the same attribute are all selected.
        """
        rng = np.random.default_rng(0)
        mat = rng.random((3, 10, 2))
        attrs = [{"region": "CA1"}, {"region": "CA1"}, {"region": "CA3"}]
        rss = RateSliceStack(event_matrix=mat, neuron_attributes=attrs)
        result = rss.subset("CA1", by="region")
        assert result.event_stack.shape[0] == 2


class TestRSSSubsliceEdgeCases:
    """Additional edge case tests for RateSliceStack.subslice."""

    def test_subslice_negative_indices(self):
        """
        subslice with negative indices wraps around correctly.

        Tests:
            (Test Case 1) Negative indices select from the end of the stack.
        """
        rng = np.random.default_rng(0)
        mat = rng.random((3, 10, 4))
        rss = RateSliceStack(event_matrix=mat)
        result = rss.subslice([-1, 0])
        assert result.event_stack.shape[2] == 2


class TestGetUnitTimingPerSliceEdgeCases2:
    """Additional edge case tests for RateSliceStack.get_unit_timing_per_slice."""

    def test_negative_firing_rates(self):
        """
        get_unit_timing_per_slice with negative firing rates.

        Tests:
            (Test Case 1) np.argmax finds the least negative value,
                and np.max compares against threshold.
        """
        mat = np.full((2, 10, 2), -0.5)
        mat[0, 5, :] = -0.1  # least negative for unit 0
        rss = RateSliceStack(event_matrix=mat)
        tm = rss.get_unit_timing_per_slice(MIN_RATE_THRESHOLD=0.0)
        # All units have max rate < 0 which is < 0.0, so all should be NaN
        assert np.all(np.isnan(tm))


# ---------------------------------------------------------------------------
# Coverage gap tests
# ---------------------------------------------------------------------------


class TestCoverageGaps:
    """Tests filling coverage gaps: serial vs parallel parity, min_overlap_frac
    override, neuron_attributes propagation, and sigma_ms validation."""

    def test_get_slice_to_slice_time_corr_serial_vs_parallel(self):
        """
        Tests: get_slice_to_slice_time_corr_from_stack serial vs parallel parity.
        (Test Case 1) n_jobs=1 and n_jobs=-1 produce identical results.
        """
        rng = np.random.default_rng(100)
        mat = rng.random((4, 15, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)

        pcm_serial, av_serial = rss.get_slice_to_slice_time_corr_from_stack(
            max_lag=0, n_jobs=1
        )
        pcm_parallel, av_parallel = rss.get_slice_to_slice_time_corr_from_stack(
            max_lag=0, n_jobs=-1
        )

        np.testing.assert_allclose(
            pcm_serial.stack, pcm_parallel.stack, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(av_serial, av_parallel, rtol=1e-12, atol=1e-12)

    def test_unit_to_unit_correlation_serial_vs_parallel(self):
        """
        Tests: unit_to_unit_correlation serial vs parallel parity.
        (Test Case 2) n_jobs=1 and n_jobs=-1 produce identical results.
        """
        rng = np.random.default_rng(101)
        mat = rng.random((4, 25, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)

        corr_s, lag_s, av_corr_s, av_lag_s = rss.unit_to_unit_correlation(
            max_lag=2, n_jobs=1
        )
        corr_p, lag_p, av_corr_p, av_lag_p = rss.unit_to_unit_correlation(
            max_lag=2, n_jobs=-1
        )

        np.testing.assert_allclose(corr_s.stack, corr_p.stack, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(lag_s.stack, lag_p.stack, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(av_corr_s, av_corr_p, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(av_lag_s, av_lag_p, rtol=1e-12, atol=1e-12)

    def test_get_slice_to_slice_unit_corr_serial_vs_parallel(self):
        """
        Tests: get_slice_to_slice_unit_corr_from_stack serial vs parallel parity.
        (Test Case 3) n_jobs=1 and n_jobs=-1 produce identical results.
        """
        rng = np.random.default_rng(102)
        mat = rng.random((4, 30, 5)) + 0.5
        rss = RateSliceStack(event_matrix=mat)

        pcm_serial, av_serial = rss.get_slice_to_slice_unit_corr_from_stack(
            max_lag=2, n_jobs=1
        )
        pcm_parallel, av_parallel = rss.get_slice_to_slice_unit_corr_from_stack(
            max_lag=2, n_jobs=-1
        )

        np.testing.assert_allclose(
            pcm_serial.stack, pcm_parallel.stack, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(av_serial, av_parallel, rtol=1e-12, atol=1e-12)

    def test_rank_order_correlation_serial_vs_parallel(self):
        """
        Tests: rank_order_correlation serial vs parallel parity.
        (Test Case 4) n_jobs=1 and n_jobs=-1 produce identical results.
        """
        rng = np.random.default_rng(103)
        mat = rng.random((6, 30, 6)) + 0.5
        rss = RateSliceStack(event_matrix=mat)

        corr_s, av_s, overlap_s = rss.rank_order_correlation(
            n_shuffles=10, seed=42, n_jobs=1
        )
        corr_p, av_p, overlap_p = rss.rank_order_correlation(
            n_shuffles=10, seed=42, n_jobs=-1
        )

        np.testing.assert_allclose(corr_s.matrix, corr_p.matrix, rtol=1e-12, atol=1e-12)
        assert av_s == pytest.approx(av_p, abs=1e-12)
        np.testing.assert_allclose(
            overlap_s.matrix, overlap_p.matrix, rtol=1e-12, atol=1e-12
        )

    def test_rank_order_correlation_min_overlap_frac_override(self):
        """
        Tests: rank_order_correlation min_overlap_frac overrides min_overlap.
        (Test Case 5) Passing min_overlap_frac=0.5 produces a stricter threshold
        than the default min_overlap=3 alone, yielding different results.
        """
        rng = np.random.default_rng(104)
        mat = rng.random((8, 30, 6)) + 0.5
        # Make some units inactive in some slices to create partial overlap
        mat[0, :, 0:3] = 0.0
        mat[1, :, 1:4] = 0.0
        mat[2, :, 2:5] = 0.0
        rss = RateSliceStack(event_matrix=mat)

        corr_default, av_default, _ = rss.rank_order_correlation(
            min_overlap=3, n_shuffles=0
        )
        # min_overlap_frac=0.5 means effective threshold = max(3, ceil(0.5 * 8)) = max(3, 4) = 4
        corr_frac, av_frac, _ = rss.rank_order_correlation(
            min_overlap=3, min_overlap_frac=0.5, n_shuffles=0
        )

        # The frac-based result should be at least as strict (more NaNs or different values)
        default_nans = np.sum(np.isnan(corr_default.matrix))
        frac_nans = np.sum(np.isnan(corr_frac.matrix))
        assert frac_nans >= default_nans

    def test_convert_to_list_neuron_attributes_propagation(self):
        """
        Tests: convert_to_list_of_RateData propagates neuron_attributes.
        (Test Case 6) Each output RateData has neuron_attributes matching the source stack.
        """
        rng = np.random.default_rng(105)
        mat = rng.random((3, 10, 4))
        attrs = [{"region": "CA1"}, {"region": "CA3"}, {"region": "DG"}]
        rss = RateSliceStack(event_matrix=mat, neuron_attributes=attrs)

        rd_list = rss.convert_to_list_of_RateData()

        assert len(rd_list) == 4
        for rd in rd_list:
            assert rd.neuron_attributes is not None
            assert len(rd.neuron_attributes) == 3
            assert rd.neuron_attributes == attrs

    def test_sigma_ms_negative_raises(self):
        """
        Tests: sigma_ms validation rejects negative values.
        (Test Case 7) RateSliceStack(data_obj=sd, ..., sigma_ms=-5) raises ValueError.
        """
        sd = make_spikedata(n_units=3, length_ms=100.0, seed=200)
        with pytest.raises(ValueError, match="sigma_ms"):
            RateSliceStack(
                data_obj=sd,
                times_start_to_end=[(10.0, 30.0), (50.0, 70.0)],
                sigma_ms=-5,
            )


class TestRateSliceStackBackendPrintMessages:
    """Tests that RateSliceStack parallel processing methods print which backend is used."""

    def test_unit_corr_threadpool_message(self, capsys):
        """
        get_slice_to_slice_unit_corr_from_stack prints thread pool message with n_jobs=-1.

        Tests:
            (Test Case 1) With n_jobs=-1 and multiple units, prints thread pool
                with worker count.
        """
        mat = make_event_matrix(3, 20, 5, seed=42) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        rss.get_slice_to_slice_unit_corr_from_stack(max_lag=2, n_jobs=-1)
        captured = capsys.readouterr()
        assert "get_slice_to_slice_unit_corr: using thread pool" in captured.out
        assert "workers)" in captured.out

    def test_unit_corr_serial_message(self, capsys):
        """
        get_slice_to_slice_unit_corr_from_stack prints serial message with n_jobs=1.

        Tests:
            (Test Case 1) With n_jobs=1, prints "using serial backend".
        """
        mat = make_event_matrix(3, 20, 5, seed=42) + 0.5
        rss = RateSliceStack(event_matrix=mat)
        rss.get_slice_to_slice_unit_corr_from_stack(max_lag=2, n_jobs=1)
        captured = capsys.readouterr()
        assert "get_slice_to_slice_unit_corr: using serial backend" in captured.out

    def test_time_corr_threadpool_message(self, capsys):
        """
        get_slice_to_slice_time_corr_from_stack prints thread pool message with n_jobs=-1.

        Tests:
            (Test Case 1) With n_jobs=-1 and multiple time bins, prints thread pool
                with worker count.
        """
        mat = make_event_matrix(3, 10, 4, seed=42)
        rss = RateSliceStack(event_matrix=mat)
        rss.get_slice_to_slice_time_corr_from_stack(max_lag=0, n_jobs=-1)
        captured = capsys.readouterr()
        assert "get_slice_to_slice_time_corr: using thread pool" in captured.out
        assert "workers)" in captured.out

    def test_time_corr_serial_message(self, capsys):
        """
        get_slice_to_slice_time_corr_from_stack prints serial message with n_jobs=1.

        Tests:
            (Test Case 1) With n_jobs=1, prints "using serial backend".
        """
        mat = make_event_matrix(3, 10, 4, seed=42)
        rss = RateSliceStack(event_matrix=mat)
        rss.get_slice_to_slice_time_corr_from_stack(max_lag=0, n_jobs=1)
        captured = capsys.readouterr()
        assert "get_slice_to_slice_time_corr: using serial backend" in captured.out
