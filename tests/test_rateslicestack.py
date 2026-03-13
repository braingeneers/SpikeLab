"""
Tests for the RateSliceStack class (spikedata/rateslicestack.py).

Covers: constructor (both modes), validation, order_units_across_slices,
get_slice_to_slice_unit_corr_from_stack, get_slice_to_slice_time_corr_from_stack,
unit_to_unit_correlation, convert_to_list_of_RateData, subset, subtime_by_index,
subslice.
"""

import pathlib
import sys
import warnings

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpikeLab.spikedata.ratedata import RateData
from SpikeLab.spikedata.rateslicestack import RateSliceStack
from SpikeLab.spikedata.spikedata import SpikeData
from SpikeLab.spikedata.pairwise import PairwiseCompMatrixStack


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
            (Test Case 2) Windows with negative start are filtered out.
        """
        rd = make_ratedata(n_units=2, n_times=200, step=1.0)
        # Peak at 5 with bounds (10,10) gives (-5, 15) which has negative start -> filtered
        # Peak at 50 gives (40, 60), peak at 100 gives (90, 110)
        rss = RateSliceStack(
            data_obj=rd,
            time_peaks=[5.0, 50.0, 100.0],
            time_bounds=(10.0, 10.0),
        )
        # Peak at 5 should be filtered (negative start)
        assert rss.event_stack.shape[2] == 2

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

        reordered, order, std, peaks = rss.order_units_across_slices("median")
        assert reordered.shape == mat.shape
        assert set(order) == {0, 1, 2}
        # Unit 2 should be first (peaks earliest)
        assert order[0] == 2
        assert order[1] == 0
        assert order[2] == 1

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
        reordered, order, std, peaks = rss.order_units_across_slices("mean")
        assert reordered.shape == mat.shape
        assert len(order) == 4
        assert len(std) == 4
        assert len(peaks) == 4

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
        reordered, order, std, peaks = rss.order_units_across_slices(
            "median", MIN_RATE_THRESHOLD=0.1
        )
        assert len(order) == 2
        assert reordered.shape == mat.shape


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
            (Test Case 1) Each unit's S×S matrix is symmetric.
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
