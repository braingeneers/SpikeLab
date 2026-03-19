"""
Tests for the RateData class (spikedata/ratedata.py).

Covers: constructor validation, subset, subtime, subtime_by_index,
frames, get_pairwise_fr_corr, and get_manifold.
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


class TestRateData:
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
        assert rd.N == 2
        assert rd.inst_Frate_data.shape == (2, 4)
        assert np.array_equal(rd.times, times)

        # Non-2D array raises ValueError.
        with pytest.raises(ValueError):
            RateData(np.ones((2, 4, 1)), times)

        # Times length mismatch raises ValueError.
        with pytest.raises(ValueError):
            RateData(data, np.array([0.0, 1.0]))

        # Negative time raises ValueError.
        with pytest.raises(ValueError):
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

    def test_subtime(self):
        """
        Tests that subtime() slices correctly, always shifting times to start at 0.

        Tests:
            (Test Case 1) Basic slice extracts correct time range.
            (Test Case 2) Times always start from 0.
            (Test Case 3) No time points in range raises ValueError.
        """
        rd = make_ratedata(n_units=2, n_times=100, step=1.0)  # times: 0..99

        sub = rd.subtime(20.0, 40.0)
        # times in [20, 40) -> 20 bins
        assert sub.inst_Frate_data.shape[1] == 20
        # Times always start at 0
        assert float(sub.times[0]) == pytest.approx(0.0)

        # Data matches the original slice.
        np.testing.assert_array_equal(sub.inst_Frate_data, rd.inst_Frate_data[:, 20:40])

        # Out-of-range raises ValueError.
        with pytest.raises(ValueError):
            rd.subtime(200.0, 300.0)

    def test_subtime_by_index(self):
        """
        Tests that subtime_by_index() slices by column index, always shifting to 0.

        Tests:
            (Test Case 1) Correct data and shape returned for valid indices.
            (Test Case 2) Times always start from 0.
            (Test Case 3) Invalid start or end index raises ValueError.
        """
        rd = make_ratedata(n_units=2, n_times=60, step=2.0)  # times: 0,2,4,...,118

        sub = rd.subtime_by_index(10, 30)
        assert sub.inst_Frate_data.shape == (2, 20)
        np.testing.assert_array_equal(sub.inst_Frate_data, rd.inst_Frate_data[:, 10:30])
        assert float(sub.times[0]) == pytest.approx(0.0)

        # Out-of-bounds indices raise ValueError.
        with pytest.raises(ValueError):
            rd.subtime_by_index(-1, 10)
        with pytest.raises(ValueError):
            rd.subtime_by_index(10, 100)

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

        assert corr.shape == (n_units, n_units)
        assert lag.shape == (n_units, n_units)

        # Diagonal must be 1.
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(n_units))
        # Diagonal lag must be 0.
        np.testing.assert_array_equal(np.diag(lag), np.zeros(n_units))

        # Identical rows -> perfect correlation and zero lag.
        assert corr[0, 1] == pytest.approx(1.0, abs=1e-5)
        assert lag[0, 1] == pytest.approx(0.0, abs=1e-5)

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
        assert float(sub_end_none.times[0]) == pytest.approx(0.0)
        assert sub_end_none.inst_Frate_data.shape[1] == 24  # times [25, 49) = 24 bins

        sub_ellipsis = rd.subtime(..., 25.0)
        assert sub_ellipsis.inst_Frate_data.shape[1] == 25

    def test_subtime_negative_indices(self):
        """
        Tests subtime() with negative start/end values.

        Tests:
            (Test Case 1) Negative start counts from end.
            (Test Case 2) Negative end counts from end.
        """
        rd = make_ratedata(n_units=2, n_times=100, step=1.0)

        sub = rd.subtime(-20.0, None)
        # -20 from end (99) = 79; times always shifted to 0
        assert float(sub.times[0]) == pytest.approx(0.0)
        assert sub.inst_Frate_data.shape[1] == 20  # times [79, 99) = 20 bins

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


class TestRateDataEdgeCases:
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

        assert corr.shape == (1, 1)
        assert lag.shape == (1, 1)
        assert not np.isnan(corr[0, 0]), "Diagonal correlation must not be NaN"

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

        assert corr.shape == (3, 3)
        assert lag.shape == (3, 3)

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

        assert corr.shape == (3, 3)
        assert lag.shape == (3, 3)

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
        Verify that subtime with a large negative start resolves correctly or raises.

        Tests:
            (Test Case 1) Start of -100 on a recording with times[-1]=99 resolves
                          to -1 after adjustment, which raises ValueError because
                          the adjusted value is still negative.
        """
        rd = make_ratedata(n_units=2, n_times=100, step=1.0)  # times 0..99

        # length = times[-1] = 99; start = -100 + 99 = -1 < 0 -> ValueError
        with pytest.raises(ValueError):
            rd.subtime(-100.0, None)

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

        assert corr.shape == (3, 3)
        assert lag.shape == (3, 3)
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(3))
        np.testing.assert_array_equal(np.diag(lag), np.zeros(3))

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

    def test_subtime_by_index_shift_time_single_bin(self):
        """subtime_by_index with shift_time=True on a single bin yields times=[0.0].

        Tests: Extracting one time bin with shift_time=True should
        produce a times array containing only 0.0.
        """
        rd = make_ratedata(n_units=2, n_times=10, step=5.0, t0=100.0)
        result = rd.subtime_by_index(3, 4)
        assert result.inst_Frate_data.shape == (2, 1)
        assert len(result.times) == 1
        assert float(result.times[0]) == pytest.approx(0.0)

    def test_get_pairwise_fr_corr_max_lag_exceeds_T(self):
        """max_lag larger than T should not crash.

        Tests: When max_lag > number of time bins, the search window
        is clamped by the array length. Verify the method returns valid output.
        """
        data = np.random.default_rng(0).random((3, 5))
        rd = RateData(data, np.arange(5, dtype=float))
        corr, lag = rd.get_pairwise_fr_corr(max_lag=100)
        assert corr.shape == (3, 3)
        assert lag.shape == (3, 3)
        # Diagonal should still be valid
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(3))

    def test_get_manifold_n_components_zero(self):
        """n_components=0 returns an embedding with zero columns.

        Tests: PCA with zero components produces shape (T, 0).
        """
        rd = make_ratedata(n_units=3, n_times=20)
        embedding, var_ratio, components = rd.get_manifold("PCA", n_components=0)
        assert embedding.shape == (20, 0)

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


class TestRecentFixes:
    """Tests for fixes applied during the 2026-03-19 code review."""

    def test_subtime_always_shifts_to_zero(self):
        """
        Tests that subtime() shifts times to start at 0 even when the slice starts mid-recording.

        Tests:
            (Test Case 1) Result times start at 0.0.
            (Test Case 2) Result times end at 20.0 (shifted from 40).
        """
        times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        data = np.ones((2, 5))
        rd = RateData(data, times)

        result = rd.subtime(20.0, 50.0)
        assert result.times[0] == 0.0
        assert result.times[-1] == 20.0

    def test_subtime_by_index_always_shifts_to_zero(self):
        """
        Tests that subtime_by_index() shifts times to start at 0.

        Tests:
            (Test Case 1) Result times start at 0.0 after slicing indices 1 through 4.
        """
        times = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        data = np.ones((2, 5))
        rd = RateData(data, times)

        result = rd.subtime_by_index(1, 4)
        assert result.times[0] == 0.0

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
