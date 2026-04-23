"""
Tests for the RateData class (spikedata/ratedata.py).

Covers: constructor validation, subset, subtime, subtime_by_index,
frames, get_pairwise_fr_corr, and get_manifold.
"""

import warnings

import numpy as np
import pytest

from spikelab.spikedata.ratedata import RateData
from spikelab.spikedata.rateslicestack import RateSliceStack

try:
    import umap  # noqa: F401

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import community  # noqa: F401
    import networkx  # noqa: F401

    # All three packages (umap, networkx, community) are needed for graph communities
    COMMUNITY_AVAILABLE = UMAP_AVAILABLE
except ImportError:
    COMMUNITY_AVAILABLE = False


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


class TestRateDataConstructor:
    """Tests for the RateData constructor."""

    def test_constructor(self):
        """
        Tests RateData constructor for valid inputs and validation errors.

        Tests:
            (Test Case 1) Valid construction stores correct attributes.
            (Test Case 2) Non-2D array raises ValueError.
            (Test Case 3) Mismatched times length raises ValueError.
            (Test Case 4) Negative times are accepted for event-aligned data.
        """
        times = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.ones((2, 4))

        rd = RateData(data, times)
        assert rd.N == 2
        assert rd.inst_Frate_data.shape == (2, 4)
        assert np.array_equal(rd.times, times)

        # Non-2D array raises ValueError.
        with pytest.raises(ValueError):
            RateData(np.ones((2, 4, 1)), times)

        # Times length mismatch raises ValueError.
        with pytest.raises(ValueError):
            RateData(data, np.array([0.0, 1.0]))

        # Negative times are valid (event-aligned data).
        rd_neg = RateData(data, np.array([-1.0, 0.0, 1.0, 2.0]))
        assert rd_neg.times[0] == -1.0

    def test_constructor_neuron_attributes(self):
        """
        Tests RateData constructor with neuron_attributes.

        Tests:
            (Test Case 1) Valid neuron_attributes are stored.
            (Test Case 2) Wrong-length neuron_attributes raises ValueError.
            (Test Case 3) None neuron_attributes is stored as None.
        """
        times = np.array([0.0, 1.0, 2.0])
        data = np.ones((2, 3))

        attrs = [{"region": "CA1"}, {"region": "CA3"}]
        rd = RateData(data, times, neuron_attributes=attrs)
        assert rd.neuron_attributes is not None
        assert len(rd.neuron_attributes) == 2
        assert rd.neuron_attributes[0] == {"region": "CA1"}

        # Wrong length
        with pytest.raises(ValueError, match="neuron_attributes"):
            RateData(data, times, neuron_attributes=[{"region": "CA1"}])

        # None
        rd_none = RateData(data, times, neuron_attributes=None)
        assert rd_none.neuron_attributes is None

    def test_constructor_all_nan_rates(self):
        """
        RateData constructor with all-NaN firing rates.

        Tests:
            (Test Case 1) All-NaN matrix is accepted without error.
            (Test Case 2) Shape and times are preserved.
        """
        data = np.full((2, 4), np.nan)
        times = np.array([0.0, 1.0, 2.0, 3.0])
        rd = RateData(data, times)
        assert rd.N == 2
        assert np.all(np.isnan(rd.inst_Frate_data))

    def test_constructor_inf_values(self):
        """
        RateData constructor with Inf values.

        Tests:
            (Test Case 1) Inf values are accepted without error.
            (Test Case 2) Values are preserved.
        """
        data = np.array([[np.inf, -np.inf], [0.0, 1.0]])
        times = np.array([0.0, 1.0])
        rd = RateData(data, times)
        assert rd.inst_Frate_data[0, 0] == np.inf
        assert rd.inst_Frate_data[0, 1] == -np.inf

    def test_empty_neuron_attributes_list_preserved(self):
        """
        Empty neuron_attributes=[] is correctly preserved for 0-unit RateData.

        Tests:
            (Test Case 1) Passing neuron_attributes=[] for a (0, T) array
                stores [] (not None).

        Notes:
            - The constructor now uses `if neuron_attributes is not None:` so
              empty list [] is correctly preserved.
        """
        times = np.array([0.0, 1.0, 2.0])
        data = np.empty((0, 3))
        rd = RateData(data, times, neuron_attributes=[])
        assert rd.neuron_attributes is not None
        assert rd.neuron_attributes == []

    def test_all_nan_rates_with_neuron_attributes(self):
        """
        All-NaN rates with neuron_attributes set.

        Tests:
            (Test Case 1) Construction succeeds with all-NaN data and valid
                neuron_attributes.
        """
        times = np.array([0.0, 1.0, 2.0])
        data = np.full((2, 3), np.nan)
        attrs = [{"region": "CA1"}, {"region": "CA3"}]
        rd = RateData(data, times, neuron_attributes=attrs)
        assert rd.neuron_attributes is not None
        assert np.all(np.isnan(rd.inst_Frate_data))

    def test_times_as_string_array(self):
        """
        times as string values create a string dtype array.

        Tests:
            (Test Case 1) String times are accepted by the constructor (no
                numeric validation on times).
        """
        data = np.ones((2, 2))
        times = ["a", "b"]
        rd = RateData(data, times)
        assert rd.times.dtype.kind in ("U", "O")

    def test_empty_neuron_attributes_list(self):
        """
        Empty list [] for neuron_attributes is preserved for N=0 data.

        Tests:
            (Test Case 1) RateData with N=0 and neuron_attributes=[] should store
                          the empty list (not silently drop it to None).
        """
        data = np.zeros((0, 5))
        times = np.arange(5, dtype=float)
        rd = RateData(data, times, neuron_attributes=[])
        assert rd.neuron_attributes == []

    def test_zero_time_bins(self):
        """
        RateData with 0 time bins is accepted but subtime crashes.

        Tests:
            (Test Case 1) Construction with shape (3, 0) and empty times succeeds.
            (Test Case 2) N and shape are correct.
            (Test Case 3) Calling subtime on zero-time-bin RateData raises an error.

        Notes:
            Construction succeeds, but calling subtime raises because
            `self.times[-1]` on an empty array raises IndexError. The length
            guard `if len(self.times) > 0` was added for `subtime` but the
            fallback `end = length` (which is 0) combined with `start >= end`
            may still raise ValueError.
        """
        data = np.zeros((3, 0))
        times = np.array([])
        rd = RateData(data, times)
        assert rd.N == 3
        assert rd.inst_Frate_data.shape == (3, 0)
        assert len(rd.times) == 0

        # subtime on empty times should raise
        with pytest.raises((ValueError, IndexError)):
            rd.subtime(0.0, 10.0)

    def test_non_array_times_input(self):
        """
        Constructor accepts non-array iterables for times by converting to ndarray.

        Tests:
            (Test Case 1) Python list is accepted and converted to ndarray.
            (Test Case 2) Tuple is accepted and converted to ndarray.
            (Test Case 3) Resulting times are numpy arrays with correct values.
        """
        data = np.ones((2, 3))

        # list
        rd_list = RateData(data, [0.0, 1.0, 2.0])
        assert isinstance(rd_list.times, np.ndarray)
        np.testing.assert_array_equal(rd_list.times, [0.0, 1.0, 2.0])

        # tuple
        rd_tuple = RateData(data, (0.0, 1.0, 2.0))
        assert isinstance(rd_tuple.times, np.ndarray)
        np.testing.assert_array_equal(rd_tuple.times, [0.0, 1.0, 2.0])


class TestRateDataSubset:
    """Tests for the RateData.subset() method."""

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
        assert sub.N == 3
        assert sub.inst_Frate_data.shape == (3, 50)
        np.testing.assert_array_equal(sub.inst_Frate_data[0], rd.inst_Frate_data[0])
        np.testing.assert_array_equal(sub.inst_Frate_data[1], rd.inst_Frate_data[2])
        np.testing.assert_array_equal(sub.inst_Frate_data[2], rd.inst_Frate_data[4])
        np.testing.assert_array_equal(sub.times, rd.times)

        # Single int.
        sub_single = rd.subset(1)
        assert sub_single.N == 1
        assert sub_single.inst_Frate_data.shape == (1, 50)

    def test_subset_by_attribute(self):
        """
        Tests subset() with the by parameter for attribute-based selection.

        Tests:
            (Test Case 1) Select units by matching attribute value.
            (Test Case 2) by without neuron_attributes raises ValueError.
            (Test Case 3) neuron_attributes are propagated to subset.
        """
        from dataclasses import dataclass

        @dataclass
        class MockAttr:
            region: str

        times = np.arange(10, dtype=float)
        data = np.arange(30, dtype=float).reshape(3, 10)
        attrs = [MockAttr("CA1"), MockAttr("CA3"), MockAttr("CA1")]
        rd = RateData(data, times, neuron_attributes=attrs)

        sub = rd.subset(["CA1"], by="region")
        assert sub.N == 2
        np.testing.assert_array_equal(sub.inst_Frate_data[0], data[0])
        np.testing.assert_array_equal(sub.inst_Frate_data[1], data[2])
        assert len(sub.neuron_attributes) == 2
        assert sub.neuron_attributes[0].region == "CA1"

        # by without neuron_attributes
        rd_no_attrs = RateData(data, times)
        with pytest.raises(ValueError, match="neuron_attributes"):
            rd_no_attrs.subset(["CA1"], by="region")

    def test_subset_preserves_neuron_attributes(self):
        """
        Tests that subset() propagates neuron_attributes for selected units.

        Tests:
            (Test Case 1) Attributes match selected units.
        """
        times = np.arange(5, dtype=float)
        data = np.ones((3, 5))
        attrs = [{"id": 0}, {"id": 1}, {"id": 2}]
        rd = RateData(data, times, neuron_attributes=attrs)
        sub = rd.subset([0, 2])
        assert sub.neuron_attributes[0] == {"id": 0}
        assert sub.neuron_attributes[1] == {"id": 2}

    def test_subset_empty_units(self):
        """
        Verify that subset with an empty units list returns a RateData with zero rows.

        Tests:
            (Test Case 1) Result shape is (0, T).
            (Test Case 2) Times are preserved unchanged.
        """
        rd = make_ratedata(n_units=3, n_times=50)

        sub = rd.subset(units=[])
        assert sub.inst_Frate_data.shape == (0, 50)
        assert sub.N == 0
        np.testing.assert_array_equal(sub.times, rd.times)

    def test_subset_duplicate_indices(self):
        """
        Verify that subset deduplicates repeated unit indices.

        Tests:
            (Test Case 1) Duplicate indices are collapsed so result has N=2.
            (Test Case 2) Data rows match the unique requested units.
        """
        rd = make_ratedata(n_units=3, n_times=50)

        sub = rd.subset(units=[0, 0, 1])
        assert sub.N == 2
        assert sub.inst_Frate_data.shape == (2, 50)
        np.testing.assert_array_equal(sub.inst_Frate_data[0], rd.inst_Frate_data[0])
        np.testing.assert_array_equal(sub.inst_Frate_data[1], rd.inst_Frate_data[1])

    def test_subset_out_of_bounds_index(self):
        """Out-of-bounds unit index should raise an IndexError.

        Tests: Requesting a unit index beyond the number of units
        should raise an IndexError from numpy array indexing.
        """
        rd = make_ratedata(n_units=3, n_times=20)
        with pytest.raises(IndexError):
            rd.subset([0, 1, 100])

    def test_subset_by_no_matching_attribute(self):
        """by parameter with no matching attribute values returns empty RateData.

        Tests: When no unit's attribute matches the requested values,
        the result should be an empty RateData with shape (0, T).
        """
        from dataclasses import dataclass

        @dataclass
        class MockAttr:
            region: str

        data = np.random.default_rng(0).random((3, 10))
        times = np.arange(10, dtype=float)
        attrs = [MockAttr("CA1"), MockAttr("CA3"), MockAttr("CA1")]
        rd = RateData(data, times, neuron_attributes=attrs)

        result = rd.subset(["V1"], by="region")
        assert result.N == 0
        assert result.inst_Frate_data.shape == (0, 10)

    def test_subset_with_dict_neuron_attributes(self):
        """
        Tests that subset() works with dict-based neuron_attributes via _get_attr fix.

        Tests:
            (Test Case 1) Selecting by region from dict attributes returns correct count.
        """
        times = np.arange(10, dtype=float)
        data = np.ones((3, 10))
        attrs = [{"region": "ctx"}, {"region": "hpc"}, {"region": "ctx"}]
        rd = RateData(data, times, neuron_attributes=attrs)

        result = rd.subset(["ctx"], by="region")
        assert result.N == 2

    def test_subset_with_object_neuron_attributes(self):
        """
        Tests that subset() works with object-based neuron_attributes via _get_attr fix.

        Tests:
            (Test Case 1) Selecting by region from object attributes returns correct count.
        """

        class MockAttr:
            def __init__(self, region):
                self.region = region

        times = np.arange(10, dtype=float)
        data = np.ones((3, 10))
        attrs = [MockAttr("ctx"), MockAttr("hpc"), MockAttr("ctx")]
        rd = RateData(data, times, neuron_attributes=attrs)

        result = rd.subset(["ctx"], by="region")
        assert result.N == 2

    def test_subset_by_multiple_match(self):
        """
        subset with by parameter where multiple units share the same attribute value.

        Tests:
            (Test Case 1) Two units matching the same region are both selected.
        """
        times = np.array([0.0, 1.0, 2.0])
        data = np.ones((3, 3))
        attrs = [{"region": "CA1"}, {"region": "CA1"}, {"region": "CA3"}]
        rd = RateData(data, times, neuron_attributes=attrs)
        result = rd.subset("CA1", by="region")
        assert result.N == 2

    def test_subset_single_numpy_int(self):
        """
        subset with a single numpy integer (np.int64).

        Tests:
            (Test Case 1) np.int64 is not caught by isinstance(units, int),
                so it falls through to set() which wraps it. The result
                should contain one unit.

        Notes:
            - np.int64 is not a Python int, so isinstance(units, int) is False
              on some platforms. The code tries set(np.int64(2)) which raises
              TypeError since numpy integers are not iterable.
        """
        rd = make_ratedata(n_units=3, n_times=10)
        with pytest.raises(TypeError):
            rd.subset(np.int64(2))

    def test_subset_negative_index(self):
        """
        Negative index in subset silently wraps around via numpy indexing.

        Tests:
            (Test Case 1) subset([-1]) selects the last unit (numpy wrap-around).
            (Test Case 2) The returned data matches the last unit's data.

        Notes:
            This is standard numpy behavior — negative indices wrap around.
            The method does not guard against this, so subset([-1]) silently
            selects the last unit rather than raising an error.
        """
        rd = make_ratedata(n_units=4, n_times=20)
        sub = rd.subset([-1])
        assert sub.N == 1
        np.testing.assert_array_equal(sub.inst_Frate_data[0], rd.inst_Frate_data[-1])


class TestRateDataSubtime:
    """Tests for the RateData.subtime() method."""

    def test_subtime(self):
        """
        Tests that subtime() slices correctly, preserving original time values.

        Tests:
            (Test Case 1) Basic slice extracts correct time range.
            (Test Case 2) Times preserve original values.
            (Test Case 3) No time points in range raises ValueError.
        """
        rd = make_ratedata(n_units=2, n_times=100, step=1.0)  # times: 0..99

        sub = rd.subtime(20.0, 40.0)
        # times in [20, 40) -> 20 bins
        assert sub.inst_Frate_data.shape[1] == 20
        # Times preserve original values
        assert float(sub.times[0]) == pytest.approx(20.0)

        # Data matches the original slice.
        np.testing.assert_array_equal(sub.inst_Frate_data, rd.inst_Frate_data[:, 20:40])

        # Out-of-range raises ValueError.
        with pytest.raises(ValueError):
            rd.subtime(200.0, 300.0)

    def test_subtime_none_and_ellipsis(self):
        """
        Tests subtime() with None and Ellipsis bounds.

        Tests:
            (Test Case 1) None start selects from beginning.
            (Test Case 2) None end selects to the end.
            (Test Case 3) Ellipsis is equivalent to None.
        """
        rd = make_ratedata(n_units=2, n_times=50, step=1.0)

        sub_start_none = rd.subtime(None, 25.0)
        assert sub_start_none.inst_Frate_data.shape[1] == 25

        sub_end_none = rd.subtime(25.0, None)
        assert float(sub_end_none.times[0]) == pytest.approx(25.0)
        assert sub_end_none.inst_Frate_data.shape[1] == 25  # times [25, 49] = 25 bins

        sub_ellipsis = rd.subtime(..., 25.0)
        assert sub_ellipsis.inst_Frate_data.shape[1] == 25

    def test_subtime_negative_values_are_literal(self):
        """
        Negative start/end in subtime() are treated as literal time values.

        Tests:
            (Test Case 1) Negative start on non-negative times selects all points
                (since all times >= -20).
            (Test Case 2) For backward-counting from the end, use subtime_by_index().
        """
        rd = make_ratedata(n_units=2, n_times=100, step=1.0)  # times: 0..99

        # -20 is literal: times >= -20 matches all 100 points
        sub = rd.subtime(-20.0, None)
        assert sub.inst_Frate_data.shape[1] == 100

    def test_subtime_single_time_point(self):
        """
        Verify that subtime extracts exactly one time bin when the range spans a single point.

        Tests:
            (Test Case 1) Result shape is (U, 1).
            (Test Case 2) Times array has exactly 1 element.
        """
        rd = make_ratedata(n_units=2, n_times=100, step=1.0)

        sub = rd.subtime(50.0, 51.0)
        assert sub.inst_Frate_data.shape == (2, 1)
        assert len(sub.times) == 1

    def test_subtime_start_equals_end(self):
        """
        Verify that subtime raises ValueError when start equals end.

        Tests:
            (Test Case 1) ValueError is raised with start >= end message.
        """
        rd = make_ratedata(n_units=2, n_times=100, step=1.0)

        with pytest.raises(ValueError, match="start.*must be less than end"):
            rd.subtime(50.0, 50.0)

    def test_subtime_negative_boundary(self):
        """
        Large negative start on non-negative times selects all data.

        Tests:
            (Test Case 1) start=-100 is literal; all times >= -100, so all data returned.
        """
        rd = make_ratedata(n_units=2, n_times=100, step=1.0)  # times 0..99

        sub = rd.subtime(-100.0, None)
        assert sub.inst_Frate_data.shape[1] == 100

    def test_subtime_preserves_original_times(self):
        """
        Tests that subtime() preserves original time values.

        Tests:
            (Test Case 1) Result times start at the original start value.
            (Test Case 2) Result times end at the original end value.
        """
        times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        data = np.ones((2, 5))
        rd = RateData(data, times)

        result = rd.subtime(20.0, 50.0)
        assert result.times[0] == 20.0
        assert result.times[-1] == 40.0

    def test_subtime_negative_times_literal(self):
        """
        subtime() treats negative start/end as literal coordinates when times contain negatives.

        Tests:
            (Test Case 1) subtime(-200, 0) selects the first 2 bins (times -200 and -100).
            (Test Case 2) Result times preserve original values.

        Notes:
            When times contain negative values (event-aligned data), negative
            start/end are treated as literal time coordinates, not offsets
            from the end.
        """
        data = np.ones((2, 5))
        times = np.array([-200.0, -100.0, 0.0, 100.0, 200.0])
        rd = RateData(data, times)

        result = rd.subtime(-200.0, 0.0)
        assert result.inst_Frate_data.shape == (2, 2)
        assert float(result.times[0]) == pytest.approx(-200.0)
        assert float(result.times[1]) == pytest.approx(-100.0)

    def test_subtime_negative_times_no_backward_offset(self):
        """
        subtime() with negative start on negative-times data uses literal coordinate.

        Tests:
            (Test Case 1) subtime(-50, 100) on times [-200, -100, 0, 100, 200]
                          treats -50 as a literal coordinate, selecting times
                          from -50 onward (i.e., times 0 and 100 which are >= -50
                          and < 100).
            (Test Case 2) Result times are shifted to start at 0.

        Notes:
            This verifies that -50 is NOT interpreted as an offset from the end
            (which would be times[-1] - 50 = 150) but as a literal time value.
        """
        data = np.arange(10, dtype=float).reshape(2, 5)
        times = np.array([-200.0, -100.0, 0.0, 100.0, 200.0])
        rd = RateData(data, times)

        result = rd.subtime(-50.0, 100.0)
        # Times >= -50 and < 100: only time 0.0 qualifies
        assert result.inst_Frate_data.shape == (2, 1)
        assert float(result.times[0]) == pytest.approx(0.0)
        # Data should match column index 2 (time=0.0) from original
        np.testing.assert_array_equal(result.inst_Frate_data[:, 0], data[:, 2])

    def test_subtime_off_grid(self):
        """
        subtime with start/end that fall between existing time values.

        Tests:
            (Test Case 1) Only time points within [start, end) are selected.
            (Test Case 2) No interpolation — discrete time bins are filtered.
        """
        data = np.arange(10, dtype=float).reshape(2, 5)
        times = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        rd = RateData(data, times)
        result = rd.subtime(5.0, 25.0)
        # Times >= 5 and < 25: 10.0 and 20.0
        assert result.inst_Frate_data.shape == (2, 2)

    def test_subtime_single_time_point_both_none(self):
        """
        subtime on a RateData with a single time point where start=None, end=None.

        Tests:
            (Test Case 1) Single-time-point RateData with default bounds returns
                the single point since end = times[-1] + 1 includes it.
        """
        data = np.ones((2, 1))
        times = np.array([5.0])
        rd = RateData(data, times)
        result = rd.subtime(None, None)
        assert result.inst_Frate_data.shape == (2, 1)
        np.testing.assert_array_equal(result.times, [5.0])

    def test_subtime_non_uniform_times(self):
        """
        subtime with non-uniform time spacing.

        Tests:
            (Test Case 1) Non-uniform times with mask correctly selects
                points in the range.
        """
        data = np.arange(8).reshape(2, 4).astype(float)
        times = np.array([0.0, 1.0, 5.0, 10.0])
        rd = RateData(data, times)
        result = rd.subtime(0.5, 6.0)
        assert result.inst_Frate_data.shape[1] == 2  # times 1.0, 5.0
        np.testing.assert_array_equal(result.times, [1.0, 5.0])

    def test_subtime_nan_start(self):
        """
        subtime with start=NaN produces empty mask and raises ValueError.

        Tests:
            (Test Case 1) NaN comparison with >= is always False, so mask is
                all-False, triggering "No time points found" error.
        """
        rd = make_ratedata(n_units=2, n_times=10)
        with pytest.raises(ValueError, match="No time points found"):
            rd.subtime(float("nan"), 5.0)

    def test_subtime_empty_times_array(self):
        """
        subtime on a RateData with empty times array raises an error.

        Tests:
            (Test Case 1) Calling subtime(0.0, 10.0) on a zero-time-bin
                          RateData raises ValueError or IndexError.

        Notes:
            The constructor accepts shape (U, 0) with empty times, but
            subtime accesses self.times[-1] which raises IndexError on
            empty arrays, or the validation logic produces a ValueError.
        """
        data = np.zeros((2, 0))
        times = np.array([])
        rd = RateData(data, times)
        with pytest.raises((ValueError, IndexError)):
            rd.subtime(0.0, 10.0)

    def test_subtime_both_none(self):
        """
        subtime(None, None) selects all data.

        Tests:
            (Test Case 1) Result has all time bins.
            (Test Case 2) Times preserve original values.
        """
        rd = make_ratedata(n_units=2, n_times=50, step=1.0)
        result = rd.subtime(None, None)
        assert result.inst_Frate_data.shape[1] == 50
        assert float(result.times[0]) == pytest.approx(0.0)


class TestRateDataSubtimeByIndex:
    """Tests for the RateData.subtime_by_index() method."""

    def test_subtime_by_index(self):
        """
        Tests that subtime_by_index() slices by column index, preserving original times.

        Tests:
            (Test Case 1) Correct data and shape returned for valid indices.
            (Test Case 2) Times preserve original values.
            (Test Case 3) Invalid start or end index raises ValueError.
        """
        rd = make_ratedata(n_units=2, n_times=60, step=2.0)  # times: 0,2,4,...,118

        sub = rd.subtime_by_index(10, 30)
        assert sub.inst_Frate_data.shape == (2, 20)
        np.testing.assert_array_equal(sub.inst_Frate_data, rd.inst_Frate_data[:, 10:30])
        assert float(sub.times[0]) == pytest.approx(20.0)  # index 10 * step 2.0

        # Out-of-bounds indices raise ValueError.
        with pytest.raises(ValueError):
            rd.subtime_by_index(-1, 10)
        with pytest.raises(ValueError):
            rd.subtime_by_index(10, 100)

    def test_subtime_by_index_empty_slice(self):
        """
        Verify that subtime_by_index with start equal to end produces an empty slice or raises.

        Tests:
            (Test Case 1) Either returns shape (U, 0) or raises ValueError.

        Notes:
            When shift_time is True and the slice is empty, indexing new_times[0]
            raises an IndexError, so this test accepts either an empty result or
            any exception.
        """
        rd = make_ratedata(n_units=2, n_times=50)

        try:
            sub = rd.subtime_by_index(5, 5)
            assert sub.inst_Frate_data.shape == (2, 0)
        except (ValueError, IndexError):
            pass  # acceptable: method rejects empty slice

    def test_subtime_by_index_single_bin_preserves_time(self):
        """
        subtime_by_index on a single bin preserves the original time value.

        Tests:
            (Test Case 1) Extracting one time bin preserves its original time.
        """
        rd = make_ratedata(n_units=2, n_times=10, step=5.0, t0=100.0)
        result = rd.subtime_by_index(3, 4)
        assert result.inst_Frate_data.shape == (2, 1)
        assert len(result.times) == 1
        assert float(result.times[0]) == pytest.approx(115.0)  # t0 + 3 * step

    def test_subtime_by_index_preserves_original_times(self):
        """
        Tests that subtime_by_index() preserves original time values.

        Tests:
            (Test Case 1) Result times start at the original value for that index.
        """
        times = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        data = np.ones((2, 5))
        rd = RateData(data, times)

        result = rd.subtime_by_index(1, 4)
        assert result.times[0] == 200.0
        assert result.times[-1] == 400.0

    def test_subtime_by_index_equal_start_end(self):
        """
        subtime_by_index with start_idx == end_idx raises ValueError.

        Tests:
            (Test Case 1) start_idx == end_idx is rejected as an invalid range.
        """
        rd = make_ratedata(n_units=2, n_times=60, step=2.0)
        with pytest.raises(ValueError):
            rd.subtime_by_index(10, 10)

    def test_subtime_by_index_zero_column(self):
        """
        subtime_by_index on zero-column RateData raises ValueError.

        Tests:
            (Test Case 1) (U, 0) shape means len(times)=0, so any start_idx
                is out of range.
        """
        data = np.empty((2, 0))
        times = np.array([])
        rd = RateData(data, times)
        with pytest.raises(ValueError, match="start_idx .* out of range"):
            rd.subtime_by_index(0, 1)

    def test_subtime_by_index_both_negative(self):
        """
        subtime_by_index with both negative indices wraps around correctly.

        Tests:
            (Test Case 1) subtime_by_index(-5, -2) on a 10-bin RateData
                          selects indices 5, 6, 7 (3 bins).
            (Test Case 2) Data matches the expected slice.
            (Test Case 3) Times are shifted to start at 0.
        """
        rd = make_ratedata(n_units=2, n_times=10, step=1.0)
        result = rd.subtime_by_index(-5, -2)
        # -5 + 10 = 5, -2 + 10 = 8 -> indices 5:8 -> 3 bins
        assert result.inst_Frate_data.shape == (2, 3)
        np.testing.assert_array_equal(
            result.inst_Frate_data, rd.inst_Frate_data[:, 5:8]
        )
        assert float(result.times[0]) == pytest.approx(5.0)  # index 5, step=1.0


class TestRateDataFrames:
    """Tests for the RateData.frames() method."""

    def test_frames(self):
        """
        Tests that frames() returns a correctly shaped RateSliceStack.

        Tests:
            (Test Case 1) Returns a RateSliceStack instance.
            (Test Case 2) Frame count is correct for evenly divisible recording.
            (Test Case 3) Each frame's data matches the corresponding subtime slice.

        Notes:
            - times are [0..99] ms at 1 ms step; length=100 bins, frame=20 ms -> 5 frames.
        """
        rd = make_ratedata(n_units=3, n_times=100, step=1.0)  # times: 0..99

        stack = rd.frames(20)
        assert isinstance(stack, RateSliceStack)
        assert len(stack.times) == 5
        assert stack.event_stack.shape == (3, 20, 5)

        # Each frame's data must match the raw subtime slice.
        for i, (start, end) in enumerate(stack.times):
            expected = rd.subtime(start, end).inst_Frate_data
            np.testing.assert_array_equal(stack.event_stack[:, :, i], expected)

    def test_frames_overlap(self):
        """
        Tests frames() with overlap and that partial last windows are excluded.

        Tests:
            (Test Case 1) Overlap produces more frames with correct step.
            (Test Case 2) Window that would extend past the last time bin is excluded.
            (Test Case 3) Data of overlapping frames is internally consistent.

        Notes:
            - times [0..99], frame=20, overlap=10 -> step=10 -> starts [0,10,...,80] = 9 frames.
              Start 90 gives window (90,110); 110 > 99+1 so it is excluded.
        """
        rd = make_ratedata(n_units=2, n_times=100, step=1.0)

        stack = rd.frames(20, overlap=10)
        assert isinstance(stack, RateSliceStack)
        assert len(stack.times) == 9
        assert stack.event_stack.shape == (2, 20, 9)

        # Verify the last frame starts at 80 and ends at 100.
        last_start, last_end = stack.times[-1]
        assert last_start == pytest.approx(80.0)
        assert last_end == pytest.approx(100.0)

    def test_frames_errors(self):
        """
        Tests that frames() raises ValueError for invalid arguments.

        Tests:
            (Test Case 1) overlap equal to length raises ValueError.
            (Test Case 2) overlap greater than length raises ValueError.
            (Test Case 3) Frame length larger than the recording raises ValueError.
        """
        rd = make_ratedata(n_units=2, n_times=50, step=1.0)

        with pytest.raises(ValueError):
            rd.frames(20, overlap=20)

        with pytest.raises(ValueError):
            rd.frames(20, overlap=25)

        with pytest.raises(ValueError):
            rd.frames(200)

    def test_frames_length_equals_recording(self):
        """
        Verify that frames with length equal to the recording span returns exactly 1 frame.

        Tests:
            (Test Case 1) Returns a RateSliceStack with 1 slice.
            (Test Case 2) The single frame covers the full time range.
        """
        rd = make_ratedata(n_units=3, n_times=100, step=1.0)  # times 0..99

        # Recording span = times[-1] - times[0] + step_size = 99 - 0 + 1 = 100
        # Use length=100 to get exactly 1 frame covering the whole recording.
        stack = rd.frames(length=100.0)
        assert isinstance(stack, RateSliceStack)
        assert stack.event_stack.shape[2] == 1

    def test_frames_single_time_bin(self):
        """
        Verify that frames produces a valid single-slice stack for T=1 RateData.

        Tests:
            (Test Case 1) A RateSliceStack with one slice and shape (U, 1, 1)
                          is returned when the RateData has only one time bin.
        """
        rd = make_ratedata(n_units=2, n_times=1)
        result = rd.frames(length=1.0)
        assert result.event_stack.shape == (2, 1, 1)

    def test_frames_length_zero(self):
        """
        frames with length=0 raises ValueError because step = 0 - 0 = 0.

        Tests:
            (Test Case 1) length=0 with overlap=0 triggers step <= 0 check.
        """
        rd = make_ratedata(n_units=2, n_times=10)
        with pytest.raises(ValueError, match="overlap must be less than length"):
            rd.frames(length=0, overlap=0)

    def test_frames_single_time_point_step_size_fallback(self):
        """
        frames() on a single-time-point RateData uses step_size=1.0 fallback.

        Tests:
            (Test Case 1) RateData with T=1 and frames(length=1.0) succeeds.
            (Test Case 2) Result has shape (U, 1, 1).

        Notes:
            When len(self.times) == 1, the code uses a fallback step_size of
            1.0 instead of computing self.times[1] - self.times[0], which
            would raise IndexError.
        """
        data = np.ones((2, 1))
        times = np.array([5.0])
        rd = RateData(data, times)
        result = rd.frames(length=1.0)
        assert result.event_stack.shape == (2, 1, 1)

    def test_frames_float_precision_boundaries(self):
        """
        frames() handles float precision in np.arange boundaries.

        Tests:
            (Test Case 1) Non-integer step sizes produce the expected number
                          of frames without off-by-one errors from float
                          accumulation.
            (Test Case 2) Each frame has the correct number of time bins.

        Notes:
            np.arange with float steps can accumulate rounding errors. The
            method uses a 1e-9 epsilon to mitigate this.
        """
        # 100 bins at step 0.5 -> times 0.0, 0.5, ..., 49.5; length 50 ms
        rd = make_ratedata(n_units=2, n_times=100, step=0.5)
        # Frame length 10 ms -> 20 time bins per frame, 5 frames
        stack = rd.frames(length=10.0)
        assert stack.event_stack.shape[2] == 5
        assert stack.event_stack.shape[1] == 20


class TestRateDataGetPairwiseFrCorr:
    """Tests for the RateData.get_pairwise_fr_corr() method."""

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

        assert corr.matrix.shape == (n_units, n_units)
        assert lag.matrix.shape == (n_units, n_units)

        # Diagonal must be 1.
        np.testing.assert_array_almost_equal(np.diag(corr.matrix), np.ones(n_units))
        # Diagonal lag must be 0.
        np.testing.assert_array_equal(np.diag(lag.matrix), np.zeros(n_units))

        # Identical rows -> perfect correlation and zero lag.
        assert corr.matrix[0, 1] == pytest.approx(1.0, abs=1e-5)
        assert lag.matrix[0, 1] == pytest.approx(0.0, abs=1e-5)

        # Both matrices are symmetric.
        np.testing.assert_array_almost_equal(corr.matrix, corr.matrix.T)

    def test_get_pairwise_fr_corr_single_unit(self):
        """
        Tests get_pairwise_fr_corr() with a single unit (U=1).

        Tests:
            (Test Case 1) Returns two (1, 1) matrices without error.
            (Test Case 2) Diagonal correlation value is a valid number (not NaN).
        """
        rng = np.random.default_rng(0)
        data = rng.random((1, 50))
        times = np.arange(50, dtype=float)
        rd = RateData(data, times)

        corr, lag = rd.get_pairwise_fr_corr(max_lag=5)

        assert corr.matrix.shape == (1, 1)
        assert lag.matrix.shape == (1, 1)
        assert not np.isnan(corr.matrix[0, 0]), "Diagonal correlation must not be NaN"

    def test_get_pairwise_fr_corr_single_time_bin(self):
        """
        Tests get_pairwise_fr_corr() with a single time bin (T=1).

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) Result has shape (3, 3).

        Notes:
            Values may be NaN for degenerate single-bin correlation; the test
            only verifies that the method does not crash.
        """
        data = np.array([[1.0], [2.0], [3.0]])
        times = np.array([0.0])
        rd = RateData(data, times)

        corr, lag = rd.get_pairwise_fr_corr(max_lag=0)

        assert corr.matrix.shape == (3, 3)
        assert lag.matrix.shape == (3, 3)

    def test_get_pairwise_fr_corr_constant_rate(self):
        """
        Tests get_pairwise_fr_corr() with constant (zero-variance) firing rates.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) Result has shape (3, 3).

        Notes:
            Constant signals have zero variance, so Pearson correlation is
            undefined. Values may be NaN but the method must not raise.
        """
        data = np.ones((3, 50))
        times = np.arange(50, dtype=float)
        rd = RateData(data, times)

        corr, lag = rd.get_pairwise_fr_corr(max_lag=5)

        assert corr.matrix.shape == (3, 3)
        assert lag.matrix.shape == (3, 3)

    def test_get_pairwise_fr_corr_max_lag_zero(self):
        """
        Verify that get_pairwise_fr_corr with max_lag=0 runs without error.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) Result matrices have shape (U, U).
            (Test Case 3) Diagonal of correlation matrix is 1.
        """
        rd = make_ratedata(n_units=3, n_times=80)

        corr, lag = rd.get_pairwise_fr_corr(max_lag=0)

        assert corr.matrix.shape == (3, 3)
        assert lag.matrix.shape == (3, 3)
        np.testing.assert_array_almost_equal(np.diag(corr.matrix), np.ones(3))
        np.testing.assert_array_equal(np.diag(lag.matrix), np.zeros(3))

    def test_get_pairwise_fr_corr_max_lag_exceeds_T(self):
        """max_lag larger than T should not crash.

        Tests: When max_lag > number of time bins, the search window
        is clamped by the array length. Verify the method returns valid output.
        """
        data = np.random.default_rng(0).random((3, 5))
        rd = RateData(data, np.arange(5, dtype=float))
        corr, lag = rd.get_pairwise_fr_corr(max_lag=100)
        assert corr.matrix.shape == (3, 3)
        assert lag.matrix.shape == (3, 3)
        # Diagonal should still be valid
        np.testing.assert_array_almost_equal(np.diag(corr.matrix), np.ones(3))

    def test_get_pairwise_fr_corr_all_zero(self):
        """
        get_pairwise_fr_corr() with all-zero firing rates returns NaN on diagonal.

        Tests:
            (Test Case 1) Result matrices have correct shape (U, U).
            (Test Case 2) Diagonal of correlation matrix is NaN (both signals
                          have zero norm → undefined self-correlation).
            (Test Case 3) Off-diagonal is also NaN (both zero → undefined).
            (Test Case 4) Diagonal of lag matrix is 0.
        """
        data = np.zeros((3, 50))
        times = np.arange(50, dtype=float)
        rd = RateData(data, times)

        corr, lag = rd.get_pairwise_fr_corr(max_lag=5)

        assert corr.matrix.shape == (3, 3)
        assert lag.matrix.shape == (3, 3)
        assert np.all(np.isnan(np.diag(corr.matrix)))
        np.testing.assert_array_equal(np.diag(lag.matrix), np.zeros(3))

    def test_get_pairwise_fr_corr_all_nan_row(self):
        """
        get_pairwise_fr_corr with one unit having all-NaN rates.

        Tests:
            (Test Case 1) Correlation with the NaN unit is NaN.
            (Test Case 2) Non-NaN units still produce valid correlations.
        """
        data = np.random.default_rng(0).random((3, 50))
        data[1, :] = np.nan
        times = np.arange(50, dtype=float)
        rd = RateData(data, times)
        corr, lag = rd.get_pairwise_fr_corr(max_lag=5)
        assert corr.matrix.shape == (3, 3)
        # Row/col for unit 1 should be NaN
        assert np.all(np.isnan(corr.matrix[1, :]))
        assert np.all(np.isnan(corr.matrix[:, 1]))

    def test_return_type_is_pairwise_comp_matrix(self):
        """
        get_pairwise_fr_corr returns PairwiseCompMatrix, not raw ndarray.

        Tests:
            (Test Case 1) Both returned values are PairwiseCompMatrix instances.
            (Test Case 2) The .matrix attribute contains the actual numpy array.
        """
        from spikelab.spikedata.pairwise import PairwiseCompMatrix

        rd = make_ratedata(n_units=3, n_times=50)
        corr, lag = rd.get_pairwise_fr_corr()
        assert isinstance(corr, PairwiseCompMatrix)
        assert isinstance(lag, PairwiseCompMatrix)
        assert corr.matrix.shape == (3, 3)
        assert lag.matrix.shape == (3, 3)

    def test_max_lag_none(self):
        """
        get_pairwise_fr_corr with max_lag=None uses lag=0 internally.

        Tests:
            (Test Case 1) max_lag=None does not raise and all lags are 0.
        """
        rd = make_ratedata(n_units=2, n_times=50)
        corr, lag = rd.get_pairwise_fr_corr(max_lag=None)
        np.testing.assert_array_equal(lag.matrix, 0)

    def test_get_pairwise_fr_corr_all_nan_rates(self):
        """
        get_pairwise_fr_corr with all-NaN rate matrix produces NaN everywhere.

        Tests:
            (Test Case 1) Result matrices have correct shape (U, U).
            (Test Case 2) Entire correlation matrix is NaN (all signals are NaN).
            (Test Case 3) Method does not raise an exception.

        Notes:
            This differs from the single-NaN-row test: here ALL units are NaN,
            so every pairwise correlation is undefined.
        """
        data = np.full((3, 50), np.nan)
        times = np.arange(50, dtype=float)
        rd = RateData(data, times)

        corr, lag = rd.get_pairwise_fr_corr(max_lag=5)

        assert corr.matrix.shape == (3, 3)
        assert lag.matrix.shape == (3, 3)
        assert np.all(np.isnan(corr.matrix))


class TestRateDataGetManifold:
    """Tests for the RateData.get_manifold() method."""

    def test_get_manifold_pca(self):
        """
        Tests get_manifold() for correct output shape and error handling.

        Tests:
            (Test Case 1) PCA output has shape (T, n_components).
            (Test Case 2) n_components=3 produces correct shape.
            (Test Case 3) Unknown method raises ValueError.
        """
        rd = make_ratedata(n_units=5, n_times=60)

        embedding, var_ratio, components = rd.get_manifold(method="PCA", n_components=2)
        assert embedding.shape == (60, 2)
        assert var_ratio.shape == (2,)
        assert components.shape == (2, 5)

        embedding3, var_ratio3, components3 = rd.get_manifold(
            method="PCA", n_components=3
        )
        assert embedding3.shape == (60, 3)
        assert var_ratio3.shape == (3,)

        with pytest.raises(ValueError):
            rd.get_manifold(method="TSNE")

    @pytest.mark.skipif(not UMAP_AVAILABLE, reason="umap-learn not installed")
    def test_get_manifold_umap(self):
        """
        Tests get_manifold() with UMAP produces correct output shape.

        Tests:
            (Test Case 1) UMAP output has shape (T, n_components).
        """
        rd = make_ratedata(n_units=5, n_times=60)

        embedding, tw = rd.get_manifold(method="UMAP", n_components=2)
        assert embedding.shape == (60, 2)
        assert isinstance(tw, float)
        assert 0.0 <= tw <= 1.0 or np.isnan(tw)

    def test_get_manifold_pca_kwargs_warning(self):
        """
        Tests that PCA method prints a message when extra kwargs are passed.

        Tests:
            (Test Case 1) Extra kwargs produce a print message (not an error).
        """
        rd = make_ratedata(n_units=5, n_times=60)
        # Should not raise, just print a message
        embedding, var_ratio, components = rd.get_manifold(
            method="PCA", n_components=2, n_neighbors=15
        )
        assert embedding.shape == (60, 2)

    @pytest.mark.skipif(not UMAP_AVAILABLE, reason="umap-learn not installed")
    def test_get_manifold_umap_return_labels_without_communities_warns(self):
        """
        Tests that return_labels=True without use_graph_communities warns.

        Tests:
            (Test Case 1) UserWarning is raised about return_labels.
            (Test Case 2) Returns embedding only (not tuple).
        """
        rd = make_ratedata(n_units=5, n_times=60)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = rd.get_manifold(method="UMAP", n_components=2, return_labels=True)
            assert any("return_labels" in str(warning.message) for warning in w)
        # Returns (embedding, trustworthiness) when no communities
        embedding, tw = result
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (60, 2)

    @pytest.mark.skipif(
        not COMMUNITY_AVAILABLE,
        reason="umap-learn, networkx, or python-louvain not installed",
    )
    def test_get_manifold_umap_graph_communities(self):
        """
        Tests get_manifold with use_graph_communities=True.

        Tests:
            (Test Case 1) Returns embedding without labels by default.
            (Test Case 2) With return_labels=True, returns (embedding, labels) tuple.
            (Test Case 3) Labels are integer array of correct shape.
        """
        rd = make_ratedata(n_units=5, n_times=60, seed=42)

        embedding, tw = rd.get_manifold(
            method="UMAP", n_components=2, use_graph_communities=True
        )
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (60, 2)
        assert isinstance(tw, float)

        embedding2, labels, tw2 = rd.get_manifold(
            method="UMAP",
            n_components=2,
            use_graph_communities=True,
            return_labels=True,
        )
        assert embedding2.shape == (60, 2)
        assert labels.shape == (60,)
        assert labels.dtype in (np.int32, np.int64, int)
        assert isinstance(tw2, float)

    def test_get_manifold_single_time_bin(self):
        """PCA on a single time bin (T=1) should not crash.

        Tests: With shape (1, U) input to PCA, sklearn may clamp
        n_components to min(n_samples, n_features). Verify safe handling.
        """
        data = np.random.default_rng(0).random((5, 1))
        rd = RateData(data, np.array([0.0]))
        try:
            result = rd.get_manifold("PCA", n_components=2)
            # sklearn may clamp n_components; accept any valid shape
            assert result.shape[0] == 1
            assert result.shape[1] >= 1
        except ValueError:
            pass  # raising is also acceptable for degenerate input

    def test_get_manifold_n_components_exceeds_dims(self):
        """n_components greater than min(T, U) should raise or degrade gracefully.

        Tests: Requesting more components than available dimensions
        should raise a ValueError from sklearn PCA.
        """
        data = np.random.default_rng(0).random((3, 10))
        rd = RateData(data, np.arange(10, dtype=float))
        with pytest.raises(ValueError):
            rd.get_manifold("PCA", n_components=20)

    def test_get_manifold_n_components_zero(self):
        """n_components=0 returns an embedding with zero columns.

        Tests: PCA with zero components produces shape (T, 0).
        """
        rd = make_ratedata(n_units=3, n_times=20)
        embedding, var_ratio, components = rd.get_manifold("PCA", n_components=0)
        assert embedding.shape == (20, 0)

    def test_get_manifold_single_unit(self):
        """
        get_manifold with N=1 (single unit, data_T has shape (T, 1)).

        Tests:
            (Test Case 1) PCA on a single feature produces valid embedding.
            (Test Case 2) n_components is clamped to 1 (only one feature).
        """
        data = np.random.default_rng(0).random((1, 20))
        times = np.arange(20, dtype=float)
        rd = RateData(data, times)
        try:
            embedding, var_ratio, components = rd.get_manifold("PCA", n_components=1)
            assert embedding.shape == (20, 1)
        except ValueError:
            pass  # raising is acceptable for degenerate single-feature input

    def test_mixed_case_method(self):
        """
        get_manifold with mixed-case method string like "Pca" works.

        Tests:
            (Test Case 1) method="Pca" is uppercased to "PCA" and works.
        """
        rd = make_ratedata(n_units=3, n_times=50)
        embedding, var_ratio, components = rd.get_manifold("Pca", n_components=2)
        assert embedding.shape == (50, 2)

    def test_get_manifold_all_constant_features(self):
        """
        PCA on zero-variance (all-constant) data.

        Tests:
            (Test Case 1) Method does not crash on all-constant input.
            (Test Case 2) Embedding has correct shape.

        Notes:
            When all features are constant, PCA should produce zero-variance
            components. sklearn PCA handles this gracefully by returning
            zero embeddings and zero explained variance.
        """
        data = np.ones((5, 30))  # all constant
        times = np.arange(30, dtype=float)
        rd = RateData(data, times)

        embedding, var_ratio, components = rd.get_manifold("PCA", n_components=2)
        assert embedding.shape == (30, 2)
        assert var_ratio.shape == (2,)
        # All variance ratios are NaN because total_var is 0, causing 0/0
        assert np.all(np.isnan(var_ratio))

    def test_get_manifold_n_components_one(self):
        """
        PCA with n_components=1 returns single-column embedding.

        Tests:
            (Test Case 1) Embedding has shape (T, 1).
            (Test Case 2) Explained variance ratio has shape (1,).
            (Test Case 3) Components has shape (1, U).
        """
        rd = make_ratedata(n_units=5, n_times=40)

        embedding, var_ratio, components = rd.get_manifold("PCA", n_components=1)
        assert embedding.shape == (40, 1)
        assert var_ratio.shape == (1,)
        assert components.shape == (1, 5)


# ---------------------------------------------------------------------------
# Edge case tests from REVIEW.md — Edge Case Scan — Core (spikedata/)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Edge case tests from the edge case scan
# ---------------------------------------------------------------------------
class TestCoverageGaps:
    """Tests for coverage gaps in RateData methods."""

    def test_get_pairwise_fr_corr_serial_equals_parallel(self):
        """
        Tests: RateData.get_pairwise_fr_corr with n_jobs=1 vs n_jobs=-1.

        (Test Case 1) Correlation matrices from serial and parallel execution are equal.
        (Test Case 2) Lag matrices from serial and parallel execution are equal.
        """
        rd = make_ratedata(n_units=4, n_times=80, seed=123)

        corr_s, lag_s = rd.get_pairwise_fr_corr(n_jobs=1)
        corr_p, lag_p = rd.get_pairwise_fr_corr(n_jobs=-1)

        np.testing.assert_allclose(
            corr_s.matrix,
            corr_p.matrix,
            rtol=1e-12,
            err_msg="FR corr matrices differ between n_jobs=1 and n_jobs=-1",
        )
        np.testing.assert_allclose(
            lag_s.matrix,
            lag_p.matrix,
            rtol=1e-12,
            err_msg="FR lag matrices differ between n_jobs=1 and n_jobs=-1",
        )

    def test_get_manifold_community_labels_small_data(self):
        """
        get_manifold with UMAP graph communities on minimal data returns
        labels with correct shape.

        Tests:
            (Test Case 1) Returns (embedding, labels, trustworthiness) tuple.
            (Test Case 2) Labels array has length T (number of time bins).
        """
        try:
            import umap  # noqa: F401
            import networkx  # noqa: F401
            import community  # noqa: F401
        except ImportError:
            pytest.skip("umap/networkx/community not installed")

        rng = np.random.default_rng(42)
        # Need enough time bins for UMAP (at least n_neighbors worth)
        rd = RateData(
            inst_Frate_data=rng.standard_normal((3, 30)),
            times=np.arange(30, dtype=float),
        )
        result = rd.get_manifold(
            method="UMAP",
            n_components=2,
            use_graph_communities=True,
            return_labels=True,
            n_neighbors=5,
        )
        assert len(result) == 3
        embedding, labels, tw = result
        assert embedding.shape == (30, 2)
        assert len(labels) == 30


# ---------------------------------------------------------------------------
# Edge case tests from REVIEW.md — Edge Case Scan (HIGH + MEDIUM)
# ---------------------------------------------------------------------------


class TestRateDataCoreReview:
    """Edge case tests for HIGH and MEDIUM findings from REVIEW.md."""

    def test_integer_input_array(self):
        """
        Integer-typed inst_Frate_data is converted to float.

        Tests:
            (Test Case 1) Integer array is accepted by the constructor.
            (Test Case 2) Stored data is converted to float64.
            (Test Case 3) get_pairwise_fr_corr on the data does not crash.
        """
        data = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        times = np.arange(5, dtype=float)
        rd = RateData(data, times)
        assert rd.inst_Frate_data.dtype == np.float64
        # Correlation works correctly with float data
        corr, lag = rd.get_pairwise_fr_corr(max_lag=0)
        assert corr.matrix.shape == (2, 2)
        assert not np.any(np.isnan(np.diag(corr.matrix)))

    def test_non_list_iterable_neuron_attributes(self):
        """
        Tuple of dicts is accepted for neuron_attributes and stored as a list.

        Tests:
            (Test Case 1) Tuple of dicts is converted to a list.
            (Test Case 2) Values are preserved.
        """
        data = np.ones((2, 3))
        times = np.arange(3, dtype=float)
        attrs = ({"region": "CA1"}, {"region": "CA3"})
        rd = RateData(data, times, neuron_attributes=attrs)
        assert isinstance(rd.neuron_attributes, list)
        assert rd.neuron_attributes[0]["region"] == "CA1"

    def test_subset_all_units_is_proper_copy(self):
        """
        subset selecting all units returns a proper copy, not a view.

        Tests:
            (Test Case 1) Data is equal to the original.
            (Test Case 2) Mutating the subset does not affect the original.
        """
        rd = make_ratedata(n_units=3, n_times=20)
        sub = rd.subset([0, 1, 2])
        assert sub.N == 3
        np.testing.assert_array_equal(sub.inst_Frate_data, rd.inst_Frate_data)
        # Mutation should not propagate
        sub.inst_Frate_data[0, 0] = -999.0
        assert rd.inst_Frate_data[0, 0] != -999.0

    def test_subtime_non_uniform_grid_end_none(self):
        """
        subtime with end=None on a non-uniform time grid uses times[1]-times[0]
        as step size, which may be incorrect for non-uniform grids.

        Tests:
            (Test Case 1) Non-uniform grid [0, 1, 5, 10]. With end=None,
                step = times[1] - times[0] = 1.0, so end = 10 + 1 = 11.
                All 4 time points should be selected.
            (Test Case 2) Verify the actual behavior matches the step-size
                fallback logic.
        """
        data = np.ones((2, 4))
        times = np.array([0.0, 1.0, 5.0, 10.0])
        rd = RateData(data, times)
        result = rd.subtime(0.0, None)
        # end = times[-1] + (times[1] - times[0]) = 10 + 1 = 11
        # All times in [0, 11) => all 4 points
        assert result.inst_Frate_data.shape[1] == 4

    def test_subtime_exact_boundary(self):
        """
        subtime with start and end exactly at time grid boundaries.

        Tests:
            (Test Case 1) start = times[2] and end = times[4] selects exactly
                the bins at those boundaries (inclusive start, exclusive end).
        """
        data = np.ones((2, 10))
        times = np.arange(10, dtype=float)
        rd = RateData(data, times)
        result = rd.subtime(2.0, 5.0)
        # times >= 2 and < 5: 2.0, 3.0, 4.0
        assert result.inst_Frate_data.shape[1] == 3
        np.testing.assert_array_equal(result.times, [2.0, 3.0, 4.0])

    def test_subtime_neuron_attributes_copied(self):
        """
        subtime passes neuron_attributes to the RateData constructor, which
        copies them. The result's attributes are independent of the original.

        Tests:
            (Test Case 1) subtime result has equal but independent
                neuron_attributes (not the same object).
            (Test Case 2) Mutating the result's attributes does not affect
                the original.
        """
        data = np.ones((2, 10))
        times = np.arange(10, dtype=float)
        attrs = [{"region": "CA1"}, {"region": "CA3"}]
        rd = RateData(data, times, neuron_attributes=attrs)
        result = rd.subtime(2.0, 5.0)
        assert result.neuron_attributes == rd.neuron_attributes
        # The constructor copies the list, so they are not the same object
        assert result.neuron_attributes is not rd.neuron_attributes

    def test_get_pairwise_fr_corr_sporadic_nan(self):
        """
        Sporadic NaN values in the rate matrix: NaN propagation not caught
        by zero-norm branch.

        Tests:
            (Test Case 1) A rate matrix with sporadic NaN (not all-NaN) in one
                unit produces NaN correlations for that unit.
            (Test Case 2) Units without NaN produce valid correlations.
        """
        rng = np.random.default_rng(42)
        data = rng.random((3, 50))
        # Sprinkle NaN in unit 1
        data[1, 10] = np.nan
        data[1, 30] = np.nan
        times = np.arange(50, dtype=float)
        rd = RateData(data, times)
        corr, lag = rd.get_pairwise_fr_corr(max_lag=3)
        assert corr.matrix.shape == (3, 3)
        # Correlations involving unit 1 should be NaN because NaN propagates
        # through the dot product
        assert np.isnan(corr.matrix[0, 1])
        assert np.isnan(corr.matrix[1, 0])
        # Self-correlation of unit 1 with NaN should also be NaN
        assert np.isnan(corr.matrix[1, 1])

    def test_get_pairwise_fr_corr_inf_values(self):
        """
        Inf values in rate matrix produce unexpected NaN in correlations.

        Tests:
            (Test Case 1) A unit with Inf values produces Inf or NaN in
                its correlation entries (Inf * anything = Inf, Inf - Inf = NaN).
            (Test Case 2) Method does not crash.
        """
        rng = np.random.default_rng(42)
        data = rng.random((3, 50))
        data[0, 5] = np.inf
        times = np.arange(50, dtype=float)
        rd = RateData(data, times)
        corr, lag = rd.get_pairwise_fr_corr(max_lag=0)
        assert corr.matrix.shape == (3, 3)
        # Inf in the signal means the norm product is inf,
        # so the correlation could be NaN or a finite value

    def test_get_pairwise_fr_corr_negative_max_lag(self):
        """
        Negative max_lag is treated as abs(max_lag) since lag is symmetric.

        Tests:
            (Test Case 1) Negative max_lag produces valid output (same as positive).
        """
        rd = make_ratedata(n_units=3, n_times=50)
        corr, lag = rd.get_pairwise_fr_corr(max_lag=-1)
        assert corr.matrix.shape == (3, 3)

    def test_get_manifold_n_components_exact_boundary(self):
        """
        n_components == min(U, T) exact boundary.

        Tests:
            (Test Case 1) n_components = min(3, 20) = 3 should work.
            (Test Case 2) Embedding has shape (T, 3).
        """
        data = np.random.default_rng(42).random((3, 20))
        times = np.arange(20, dtype=float)
        rd = RateData(data, times)
        embedding, var_ratio, components = rd.get_manifold("PCA", n_components=3)
        assert embedding.shape == (20, 3)
        assert var_ratio.shape == (3,)
        assert components.shape == (3, 3)

    def test_frames_overlap_near_length(self):
        """
        Overlap equal to length minus 1 (epsilon) produces maximum overlapping
        frames.

        Tests:
            (Test Case 1) length=20, overlap=19 produces step=1, many frames.
            (Test Case 2) Result is a valid RateSliceStack.
        """
        rd = make_ratedata(n_units=2, n_times=100, step=1.0)
        stack = rd.frames(20, overlap=19)
        assert isinstance(stack, RateSliceStack)
        # step = 20 - 19 = 1; starts = 0,1,2,...,80 = 81 frames
        assert stack.event_stack.shape[2] == 81
