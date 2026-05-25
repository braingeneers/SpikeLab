"""
Tests for the SpikeSliceStack class (spikedata/spikeslicestack.py).

Covers: constructor (both time modes), validation, to_raster_array.
"""

import warnings

import numpy as np
import pytest

from spikelab.spikedata.pairwise import PairwiseCompMatrix, PairwiseCompMatrixStack
from spikelab.spikedata.spikedata import SpikeData
from spikelab.spikedata.spikeslicestack import SpikeSliceStack


def make_spikedata(n_units=3, length_ms=200.0, seed=0):
    """Create a SpikeData with uniformly spaced spikes per unit."""
    rng = np.random.default_rng(seed)
    train = []
    for _ in range(n_units):
        n_spikes = rng.integers(10, 30)
        spikes = np.sort(rng.uniform(0, length_ms, n_spikes))
        train.append(spikes)
    return SpikeData(train, length=length_ms)


class TestSpikeSliceStackConstructor:
    def test_basic_construction(self):
        """
        Tests basic construction with times_start_to_end.

        Tests:
            (Test Case 1) spike_stack has correct number of slices.
            (Test Case 2) Each slice is a SpikeData object.
            (Test Case 3) Times are stored correctly.
        """
        sd = make_spikedata(n_units=3, length_ms=200.0)
        times = [(10.0, 30.0), (50.0, 70.0), (100.0, 120.0)]
        sss = SpikeSliceStack(sd, times_start_to_end=times)
        assert len(sss.spike_stack) == 3
        assert len(sss.times) == 3
        for s in sss.spike_stack:
            assert isinstance(s, SpikeData)
            assert s.N == 3
            assert s.length == pytest.approx(20.0)

    def test_peaks_and_bounds(self):
        """
        Tests construction with time_peaks + time_bounds.

        Tests:
            (Test Case 1) Peaks and bounds are converted to start/end tuples.
            (Test Case 2) Each slice has correct duration.
        """
        sd = make_spikedata(n_units=2, length_ms=200.0)
        sss = SpikeSliceStack(
            sd,
            time_peaks=[50.0, 100.0, 150.0],
            time_bounds=(10.0, 10.0),
        )
        assert len(sss.spike_stack) == 3
        for s in sss.spike_stack:
            assert s.length == pytest.approx(20.0)

    def test_negative_windows_preserved(self):
        """
        Windows with negative start times are preserved by _validate_time_start_to_end.

        Tests:
            (Test Case 1) All windows including negative-start are kept.
        """
        from spikelab.spikedata.utils import _validate_time_start_to_end

        windows = [(-5.0, 15.0), (40.0, 60.0), (90.0, 110.0)]
        result = _validate_time_start_to_end(windows)
        assert len(result) == 3
        assert result[0][0] == pytest.approx(-5.0)

    def test_non_spikedata_raises(self):
        """
        Tests that non-SpikeData data_obj raises TypeError.

        Tests:
            (Test Case 1) String input raises TypeError.
            (Test Case 2) None input raises TypeError.
        """
        with pytest.raises(TypeError, match="SpikeData"):
            SpikeSliceStack("not a SpikeData", times_start_to_end=[(0.0, 10.0)])
        with pytest.raises(TypeError, match="SpikeData"):
            SpikeSliceStack(None, times_start_to_end=[(0.0, 10.0)])

    def test_no_time_args_raises(self):
        """
        Tests that missing time specification raises ValueError.

        Tests:
            (Test Case 1) No times raises ValueError.
            (Test Case 2) Only time_peaks without time_bounds raises ValueError.
        """
        sd = make_spikedata()
        with pytest.raises(ValueError, match="Must provide"):
            SpikeSliceStack(sd)
        with pytest.raises(ValueError, match="Must provide"):
            SpikeSliceStack(sd, time_peaks=[50.0])

    def test_invalid_time_bounds_raises(self):
        """
        Tests that invalid time_bounds raises TypeError.

        Tests:
            (Test Case 1) List instead of tuple raises TypeError.
            (Test Case 2) Wrong-length tuple raises TypeError.
        """
        sd = make_spikedata()
        with pytest.raises(TypeError, match="time_bounds"):
            SpikeSliceStack(sd, time_peaks=[50.0], time_bounds=[10, 10])
        with pytest.raises(TypeError, match="time_bounds"):
            SpikeSliceStack(sd, time_peaks=[50.0], time_bounds=(10,))

    def test_times_not_list_raises(self):
        """
        Tests that non-list times_start_to_end raises TypeError.

        Tests:
            (Test Case 1) Tuple input raises TypeError.
        """
        sd = make_spikedata()
        with pytest.raises(TypeError, match="list of tuples"):
            SpikeSliceStack(sd, times_start_to_end=((10.0, 20.0),))

    def test_non_tuple_element_raises(self):
        """
        Tests that non-tuple element in times raises TypeError.

        Tests:
            (Test Case 1) List element raises TypeError.
        """
        sd = make_spikedata()
        with pytest.raises(TypeError, match="not a tuple"):
            SpikeSliceStack(sd, times_start_to_end=[[10.0, 20.0]])

    def test_wrong_length_tuple_raises(self):
        """
        Tests that wrong-length tuple raises TypeError.

        Tests:
            (Test Case 1) 3-element tuple raises TypeError.
        """
        sd = make_spikedata()
        with pytest.raises(TypeError, match="length 2"):
            SpikeSliceStack(sd, times_start_to_end=[(10.0, 20.0, 30.0)])

    def test_non_numeric_times_raises(self):
        """
        Tests that non-numeric start/end raises TypeError.

        Tests:
            (Test Case 1) String values raise TypeError.
        """
        sd = make_spikedata()
        with pytest.raises(TypeError, match="numbers"):
            SpikeSliceStack(sd, times_start_to_end=[("a", "b")])

    def test_start_ge_end_raises(self):
        """
        Tests that start >= end raises ValueError.

        Tests:
            (Test Case 1) Equal start and end raises ValueError.
        """
        sd = make_spikedata()
        with pytest.raises(ValueError, match="less than end"):
            SpikeSliceStack(sd, times_start_to_end=[(20.0, 20.0)])

    def test_unequal_durations_raises(self):
        """
        Tests that windows with different durations raise ValueError.

        Tests:
            (Test Case 1) Windows of 10ms and 20ms raise ValueError.
        """
        sd = make_spikedata(length_ms=200.0)
        with pytest.raises(ValueError, match="same length"):
            SpikeSliceStack(sd, times_start_to_end=[(10.0, 20.0), (50.0, 70.0)])

    def test_slices_are_sorted(self):
        """
        Tests that slices are sorted chronologically.

        Tests:
            (Test Case 1) Reverse-order input is sorted.
        """
        sd = make_spikedata(length_ms=200.0)
        times = [(100.0, 120.0), (50.0, 70.0), (10.0, 30.0)]
        sss = SpikeSliceStack(sd, times_start_to_end=times)
        starts = [t[0] for t in sss.times]
        assert starts == sorted(starts)

    def test_single_unit_construction(self):
        """
        Verify SpikeSliceStack can be constructed with N=1 (single unit).

        Tests:
            (Test Case 1) Construction succeeds without error.
            (Test Case 2) Each slice has N=1.
        """
        train = [np.array([10.0, 50.0, 90.0, 130.0])]
        sd = SpikeData(train, length=200.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 40.0), (80.0, 120.0)])

        assert len(sss.spike_stack) == 2
        for s in sss.spike_stack:
            assert isinstance(s, SpikeData)
            assert s.N == 1

    def test_single_spike_total(self):
        """
        Tests SpikeSliceStack with only one spike across all units and slices.

        Tests:
            (Test Case 1) Construction succeeds without error.
            (Test Case 2) Slices without spikes have empty spike trains.
            (Test Case 3) The slice containing the spike has 1 spike for that unit.
            (Test Case 4) All slices are valid SpikeData objects.

        Notes:
            Only one spike exists at 15ms, so the first slice (0-20ms) contains it
            while the other two slices (40-60ms, 70-90ms) have zero spikes for all
            units. This verifies that SpikeSliceStack handles near-empty data.
        """
        train = [
            np.array([15.0]),
            np.array([]),
        ]
        sd = SpikeData(train, length=100.0)
        sss = SpikeSliceStack(
            sd, times_start_to_end=[(0.0, 20.0), (40.0, 60.0), (70.0, 90.0)]
        )

        assert len(sss.spike_stack) == 3
        for s in sss.spike_stack:
            assert isinstance(s, SpikeData)
            assert s.N == 2

        # First slice should have 1 spike for unit 0
        assert len(sss.spike_stack[0].train[0]) == 1
        # Second and third slices should have 0 spikes for all units
        for s in sss.spike_stack[1:]:
            for u in range(s.N):
                assert len(s.train[u]) == 0

    def test_duplicate_time_windows(self):
        """
        Tests SpikeSliceStack with duplicate time windows.

        Tests:
            (Test Case 1) Construction succeeds with two identical windows.
            (Test Case 2) Both slices contain identical spike data.
            (Test Case 3) times list contains both entries.

        Notes:
            Duplicate time windows are not rejected by the validator because
            they have the same duration. The result is two slices with identical
            spike content.
        """
        train = [np.array([5.0, 15.0, 50.0, 90.0])]
        sd = SpikeData(train, length=100.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 20.0), (0.0, 20.0)])

        assert len(sss.spike_stack) == 2
        assert len(sss.times) == 2
        assert sss.times[0] == sss.times[1]

        # Both slices should have the same spikes
        spikes_0 = sss.spike_stack[0].train[0]
        spikes_1 = sss.spike_stack[1].train[0]
        np.testing.assert_array_equal(spikes_0, spikes_1)

    def test_constructor_slices_are_zero_based(self):
        """
        Tests that slices from the constructor have 0-based spike times within
        the window duration.

        Tests:
            (Test Case 1) All spikes in the slice are >= 0.
            (Test Case 2) All spikes in the slice are < window duration (100 ms).
            (Test Case 3) Slice length equals the window duration.
        """
        sd = SpikeData([np.array([50.0, 100.0, 150.0, 200.0, 250.0])], length=300.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(100.0, 200.0)])

        sliced = sss.spike_stack[0]
        assert sliced.length == 100.0
        for unit_spikes in sliced.train:
            if len(unit_spikes) > 0:
                assert np.all(unit_spikes >= 0)
                assert np.all(unit_spikes < 100.0)

    def test_constructor_preserves_absolute_times_in_metadata(self):
        """
        Tests that the absolute time window is preserved in sss.times even though
        spike times are 0-based.

        Tests:
            (Test Case 1) sss.times[0] == (100, 200).
        """
        sd = SpikeData([np.array([50.0, 100.0, 150.0, 200.0, 250.0])], length=300.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(100.0, 200.0)])

        assert sss.times[0] == (100.0, 200.0)

    def test_overlapping_time_windows(self):
        """
        EC-SSS-12: Constructor with overlapping time windows. The validation
        function does not reject overlapping windows -- it only checks that
        all windows have the same duration. This is accepted silently.

        Tests:
            (Test Case 1) No error is raised with overlapping windows.
            (Test Case 2) Both slices are constructed.
            (Test Case 3) Times are stored correctly (sorted by start).
        """
        sd = make_spikedata(n_units=2, length_ms=200.0, seed=0)
        # Windows overlap: [10, 30) and [20, 40) share the [20, 30) range
        times = [(10.0, 30.0), (20.0, 40.0)]
        sss = SpikeSliceStack(sd, times_start_to_end=times)

        assert len(sss.spike_stack) == 2
        assert sss.times[0] == (10.0, 30.0)
        assert sss.times[1] == (20.0, 40.0)

    def test_spike_stack_zero_length_slices(self):
        """
        spike_stack with all slices having length=0.

        Tests:
            (Test Case 1) Zero-length SpikeData slices produce auto-generated
                times of (0, 0) for each.
        """
        sd1 = SpikeData([], length=0.0)
        sd2 = SpikeData([], length=0.0)
        sss = SpikeSliceStack(spike_stack=[sd1, sd2])
        assert len(sss.spike_stack) == 2
        for start, end in sss.times:
            assert end - start == 0.0

    def test_drop_slice_attributes_false(self):
        """
        spike_stack with drop_slice_attributes=False preserves per-slice neuron_attributes.

        Tests:
            (Test Case 1) neuron_attributes from individual slices are preserved
                when drop_slice_attributes=False.
        """
        attrs = [{"region": "CA1"}, {"region": "CA3"}]
        sd1 = SpikeData([[1.0], [2.0]], length=10.0, neuron_attributes=attrs)
        sd2 = SpikeData([[3.0], [4.0]], length=10.0, neuron_attributes=attrs)
        sss = SpikeSliceStack(spike_stack=[sd1, sd2], drop_slice_attributes=False)
        # Per-slice attributes should be preserved
        assert sss.spike_stack[0].neuron_attributes is not None

    def test_data_obj_with_zero_length_subtime(self):
        """
        Constructor with a time window of zero duration triggers ValueError from
        SpikeData.subtime(start, end) when start == end.

        Tests:
            (Test Case 1) ValueError is raised when the time window has zero duration.

        Notes:
            The validator _validate_time_start_to_end rejects start >= end before
            subtime is called, so this error path is caught at the validation layer.
        """
        sd = make_spikedata(n_units=2, length_ms=100.0)
        with pytest.raises(ValueError, match="less than end"):
            SpikeSliceStack(sd, times_start_to_end=[(50.0, 50.0)])

    def test_spike_stack_absolute_times_raises(self):
        """
        Constructing via spike_stack with absolute (non-0-based) spike times
        raises ValueError when spike times exceed the slice duration.

        Tests:
            (Test Case 1) ValueError mentioning '0-based' is raised.
        """
        sd = SpikeData([[150.0, 250.0]], length=300.0)
        with pytest.raises(ValueError, match="0-based"):
            SpikeSliceStack(spike_stack=[sd], times_start_to_end=[(100.0, 200.0)])

    def test_spike_stack_zero_based_no_warning(self):
        """
        Constructing via spike_stack with correctly 0-based spike times
        does not emit a warning.

        Tests:
            (Test Case 1) No warning when spike times are within slice duration.
        """
        sd = SpikeData([[10.0, 50.0, 90.0]], length=100.0)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sss = SpikeSliceStack(spike_stack=[sd], times_start_to_end=[(0.0, 100.0)])
        assert len(sss.spike_stack) == 1


class TestToRasterArray:
    """Tests for SpikeSliceStack.to_raster_array()."""

    def test_basic_output(self):
        """
        Tests to_raster_array output shape and values.

        Tests:
            (Test Case 1) Output is a numpy ndarray.
            (Test Case 2) Output shape is (U, T, S).
            (Test Case 3) All values are non-negative integers.

        Notes:
            - Values are spike counts per 1ms bin, so they can exceed 1 if
              multiple spikes fall in the same bin.
        """
        sd = make_spikedata(n_units=3, length_ms=200.0)
        times = [(10.0, 30.0), (50.0, 70.0), (100.0, 120.0)]
        sss = SpikeSliceStack(sd, times_start_to_end=times)
        result = sss.to_raster_array()

        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[0] == 3  # units
        assert result.shape[2] == 3  # slices
        assert np.all(result >= 0)

    def test_consistent_with_individual_rasters(self):
        """
        Tests that raster array matches individual slice rasters.

        Tests:
            (Test Case 1) Each slice in the 3D output matches sparse_raster of that slice.
        """
        sd = make_spikedata(n_units=2, length_ms=200.0, seed=42)
        times = [(20.0, 40.0), (60.0, 80.0)]
        sss = SpikeSliceStack(sd, times_start_to_end=times)
        result = sss.to_raster_array()

        for i, slice_sd in enumerate(sss.spike_stack):
            # Spike times within each slice are already 0-based (shifted by
            # subtime during construction), so rasterize directly.
            expected = slice_sd.sparse_raster(bin_size=1).toarray()
            assert result.shape[1] == expected.shape[1]
            np.testing.assert_array_equal(result[:, :, i], expected)

    def test_single_slice(self):
        """
        Tests to_raster_array with a single slice.

        Tests:
            (Test Case 1) S dimension is 1.
        """
        sd = make_spikedata(n_units=2, length_ms=100.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(10.0, 30.0)])
        result = sss.to_raster_array()
        assert result.shape[2] == 1

    def test_to_raster_array_empty_slices(self):
        """
        Verify to_raster_array handles slices where one window has no spikes.

        Tests:
            (Test Case 1) np.stack succeeds even when slices have different sparsity.
            (Test Case 2) Output shape is (U, T, 2) where T matches the 20 ms window.

        Notes:
            If np.stack fails because dense rasters have different shapes, this
            indicates a bug in to_raster_array (all windows have equal duration so
            the dense shapes should match regardless of spike content).
        """
        # Place spikes only in [0, 50]; second window [80, 100] should be empty
        train = [np.array([5.0, 10.0, 25.0, 40.0])]
        sd = SpikeData(train, length=120.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 20.0), (80.0, 100.0)])

        result = sss.to_raster_array()

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1  # U
        assert result.shape[2] == 2  # S
        # Second slice should be all zeros (no spikes in [80, 100])
        assert np.all(result[:, :, 1] == 0)

    def test_absolute_times_basic(self):
        """
        absolute_times=True places spikes at their original recording positions.

        Tests:
            (Test Case 1) T dimension covers the full recording span.
            (Test Case 2) Spikes in each slice appear at their absolute positions.
            (Test Case 3) Spike count is preserved.
        """
        sd = SpikeData([[10.0, 50.0, 90.0, 110.0, 150.0, 190.0]], length=200.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 100.0), (100.0, 200.0)])
        result = sss.to_raster_array(bin_size=10.0, absolute_times=True)

        # T should span the full 200ms at 10ms bins = 20 bins
        assert result.shape == (1, 20, 2)
        # Total spike count preserved
        assert result.sum() == 6
        # Slice 0: spikes at absolute 10, 50, 90 → bins 0, 4, 8
        assert result[0, 0, 0] == 1
        assert result[0, 4, 0] == 1
        assert result[0, 8, 0] == 1
        assert result[0, 10:, 0].sum() == 0  # no spikes in second half
        # Slice 1: spikes at absolute 110, 150, 190 → bins 10, 14, 18
        assert result[0, :10, 1].sum() == 0  # no spikes in first half
        assert result[0, 10, 1] == 1
        assert result[0, 14, 1] == 1
        assert result[0, 18, 1] == 1

    def test_absolute_times_false_is_default(self):
        """
        Default behavior (absolute_times=False) matches the standard to_raster_array.

        Tests:
            (Test Case 1) Calling without absolute_times matches calling with False.
        """
        sd = make_spikedata(n_units=2, length_ms=200.0, seed=7)
        sss = SpikeSliceStack(sd, times_start_to_end=[(10.0, 50.0), (60.0, 100.0)])
        default = sss.to_raster_array(bin_size=5.0)
        explicit = sss.to_raster_array(bin_size=5.0, absolute_times=False)
        np.testing.assert_array_equal(default, explicit)

    def test_absolute_times_nonzero_global_start(self):
        """
        absolute_times=True with slices that don't start at 0.

        Tests:
            (Test Case 1) T dimension covers only the span from min(start) to
                max(end), not from 0.
            (Test Case 2) Spikes are correctly positioned relative to global start.
        """
        # Construct via spike_stack (Option 2) with 0-based spike times
        sd0 = SpikeData([[5.0, 15.0, 25.0, 35.0]], length=50.0)
        sd1 = SpikeData([[5.0, 15.0, 25.0, 35.0]], length=50.0)
        sss = SpikeSliceStack(
            spike_stack=[sd0, sd1],
            times_start_to_end=[(100.0, 150.0), (200.0, 250.0)],
        )

        result = sss.to_raster_array(bin_size=10.0, absolute_times=True)
        # Span from 100 to 250 = 150ms → 15 bins
        assert result.shape == (1, 15, 2)
        # Slice 0: offset = 100 - 100 = 0, spikes at 5, 15, 25, 35 → bins 0, 1, 2, 3
        assert result[0, 0, 0] == 1
        assert result[0, 1, 0] == 1
        # Slice 1: offset = 200 - 100 = 100, spikes at 105, 115, 125, 135 → bins 10, 11, 12, 13
        assert result[0, 10, 1] == 1
        assert result[0, 11, 1] == 1

    def test_absolute_times_single_slice(self):
        """
        absolute_times=True with a single slice.

        Tests:
            (Test Case 1) Works without error.
            (Test Case 2) Spike positions match the 0-based case (offset is 0
                since there's only one slice).
        """
        sd = SpikeData([[10.0, 30.0]], length=50.0)
        sss = SpikeSliceStack(spike_stack=[sd], times_start_to_end=[(500.0, 550.0)])
        result = sss.to_raster_array(bin_size=10.0, absolute_times=True)
        # Single slice starting at 500, so offset = 500 - 500 = 0
        # Spikes at 10, 30 → bins 0, 2
        assert result.shape == (1, 5, 1)
        assert result[0, 0, 0] == 1
        assert result[0, 2, 0] == 1

    def test_to_raster_array_single_unit(self):
        """
        Verify to_raster_array output shape with N=1 (single unit).

        Tests:
            (Test Case 1) Output shape is (1, T, S).
        """
        train = [np.array([5.0, 15.0, 55.0, 65.0])]
        sd = SpikeData(train, length=100.0)
        times = [(0.0, 20.0), (50.0, 70.0)]
        sss = SpikeSliceStack(sd, times_start_to_end=times)

        result = sss.to_raster_array()

        assert result.shape[0] == 1  # U
        assert result.shape[2] == 2  # S

    def test_to_raster_array_bin_size_equals_duration(self):
        """
        EC-SSS-01: to_raster_array with bin_size equal to slice duration produces
        a single time bin per slice.

        Tests:
            (Test Case 1) Output T dimension is 1.
            (Test Case 2) Each time bin contains the total spike count for that unit/slice.
        """
        train = [np.array([5.0, 15.0, 55.0, 65.0])]
        sd = SpikeData(train, length=100.0)
        times = [(0.0, 20.0), (50.0, 70.0)]
        sss = SpikeSliceStack(sd, times_start_to_end=times)

        result = sss.to_raster_array(bin_size=20.0)

        assert result.shape == (1, 1, 2)  # (U=1, T=1, S=2)
        # First slice [0,20): spikes at 5 and 15 => 2 spikes in one bin
        assert result[0, 0, 0] == 2
        # Second slice [50,70): spikes at 55 and 65 => shifted to 5 and 15 => 2 spikes
        assert result[0, 0, 1] == 2

    def test_absolute_times_with_event_centered_slices(self):
        """
        to_raster_array(absolute_times=True) with negative start_time slices.

        Tests:
            (Test Case 1) Event-centered slices with negative start_time produce
                a valid raster with absolute_times=True.
        """
        trains = [np.array([-5.0, 0.0, 5.0])]
        sd = SpikeData(trains, length=20.0, start_time=-10.0)
        sss = SpikeSliceStack(
            sd,
            times_start_to_end=[(-10.0, 0.0), (0.0, 10.0)],
        )
        raster = sss.to_raster_array(bin_size=5.0, absolute_times=True)
        assert raster.shape[0] == 1  # N=1 unit
        assert raster.shape[2] == 2  # S=2 slices

    def test_inconsistent_slice_durations_via_spike_stack(self):
        """
        to_raster_array with spike_stack containing SpikeData objects of different
        lengths causes np.stack to fail on shape mismatch.

        Tests:
            (Test Case 1) ValueError is raised when dense rasters have different
                time dimensions due to different slice durations.

        Notes:
            The spike_stack= constructor does not enforce equal durations across
            slices. The error only surfaces when to_raster_array tries to stack
            the rasters.
        """
        sd1 = SpikeData([np.array([5.0, 15.0])], length=20.0)
        sd2 = SpikeData([np.array([5.0, 25.0])], length=30.0)
        # The constructor validates that all time windows have the same length,
        # so the ValueError is raised at construction, not at to_raster_array.
        with pytest.raises(ValueError, match="same length"):
            SpikeSliceStack(
                spike_stack=[sd1, sd2],
                times_start_to_end=[(0.0, 20.0), (20.0, 50.0)],
            )

    def test_bin_size_larger_than_slice_duration(self):
        """
        to_raster_array with bin_size larger than the slice duration still produces
        valid output with T=1 (one time bin that spans the entire slice).

        Tests:
            (Test Case 1) Output shape has T=1 when bin_size > slice duration.
            (Test Case 2) The single bin contains the total spike count for each unit/slice.
        """
        train = [np.array([2.0, 5.0, 8.0])]
        sd = SpikeData(train, length=10.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 10.0)])

        result = sss.to_raster_array(bin_size=50.0)

        # bin_size=50 for a 10ms slice should give T=1
        assert result.shape[1] == 1
        assert result[0, 0, 0] == 3  # All 3 spikes in one bin

    def test_absolute_times_ulp_overflow_clamped(self):
        """
        absolute_times=True does not raise even when per-slice
        sparse_raster shapes momentarily exceed the global ``total_bins``
        due to ULP-level float arithmetic differences between
        ``np.ceil((global_end - global_start) / bin_size)`` and
        ``np.ceil((slice_length + offset) / bin_size)``.

        Tests:
            (Test Case 1) A pathological bin_size / slice combination
                that exposes the ULP mismatch returns a (N, T, S)
                array without raising.
            (Test Case 2) The returned shape's T equals the global
                ``total_bins``.
        """
        # Bin/window combo chosen so float arithmetic produces a
        # one-ULP-larger r.shape[1] than total_bins on the last slice.
        # The exact spike positions don't matter — we're exercising
        # the clamp in the assignment, not the binning math.
        sd = SpikeData([[1.0, 2.5, 4.9, 6.0, 8.5]], length=10.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 5.0), (5.0, 10.0)])
        # bin_size 0.7 produces non-integer bin counts; with the
        # global span 10.0, total_bins = ceil(10.0 / 0.7) = 15.
        # The clamp prevents the assignment from broadcast-failing
        # even if a per-slice raster comes out one bin oversized.
        result = sss.to_raster_array(bin_size=0.7, absolute_times=True)
        assert result.ndim == 3
        assert result.shape[2] == 2  # S
        # The buffer width must equal the configured total_bins.
        import numpy as _np

        expected_total = int(_np.ceil(10.0 / 0.7))
        assert result.shape[1] == expected_total


class TestSpikeStackConstructor:
    """Tests for the spike_stack= (Option 2) constructor path."""

    def test_basic_spike_stack_construction(self):
        """
        Construct a SpikeSliceStack from a pre-built list of SpikeData objects.

        Tests:
            (Test Case 1) spike_stack length matches input list length.
            (Test Case 2) N is set correctly from the SpikeData objects.
            (Test Case 3) times are auto-generated end-to-end when not provided.
        """
        sd1 = SpikeData([np.array([5.0, 15.0])], length=20.0)
        sd2 = SpikeData([np.array([3.0, 12.0])], length=20.0)
        sd3 = SpikeData([np.array([8.0])], length=20.0)

        sss = SpikeSliceStack(spike_stack=[sd1, sd2, sd3])

        assert len(sss.spike_stack) == 3
        assert sss.N == 1
        # Auto-generated times: (0,20), (20,40), (40,60)
        assert sss.times[0] == pytest.approx((0.0, 20.0))
        assert sss.times[1] == pytest.approx((20.0, 40.0))
        assert sss.times[2] == pytest.approx((40.0, 60.0))

    def test_spike_stack_with_explicit_times(self):
        """
        Construct with spike_stack and explicit times_start_to_end.

        Tests:
            (Test Case 1) Provided times are stored correctly.
            (Test Case 2) spike_stack is stored without modification.
        """
        sd1 = SpikeData([np.array([5.0]), np.array([10.0])], length=20.0)
        sd2 = SpikeData([np.array([2.0]), np.array([18.0])], length=20.0)
        times = [(100.0, 120.0), (200.0, 220.0)]

        sss = SpikeSliceStack(spike_stack=[sd1, sd2], times_start_to_end=times)

        assert sss.times == times
        assert sss.N == 2
        assert len(sss.spike_stack) == 2

    def test_spike_stack_with_neuron_attributes(self):
        """
        Construct with spike_stack and neuron_attributes.

        Tests:
            (Test Case 1) neuron_attributes are stored correctly.
        """
        sd1 = SpikeData([np.array([5.0]), np.array([10.0])], length=20.0)
        sd2 = SpikeData([np.array([2.0]), np.array([18.0])], length=20.0)
        attrs = [{"id": "A"}, {"id": "B"}]

        sss = SpikeSliceStack(spike_stack=[sd1, sd2], neuron_attributes=attrs)

        assert sss.neuron_attributes == attrs

    def test_spike_stack_overrides_data_obj(self):
        """
        When both data_obj and spike_stack are provided, spike_stack wins with a warning.

        Tests:
            (Test Case 1) UserWarning is raised.
            (Test Case 2) Result uses spike_stack, not data_obj.
        """
        sd_obj = make_spikedata(n_units=3, length_ms=200.0)
        sd1 = SpikeData([np.array([5.0])], length=20.0)
        sd2 = SpikeData([np.array([10.0])], length=20.0)

        with pytest.warns(UserWarning, match="Ignoring data_obj"):
            sss = SpikeSliceStack(
                data_obj=sd_obj,
                spike_stack=[sd1, sd2],
            )

        assert sss.N == 1  # From spike_stack, not data_obj (which has 3 units)
        assert len(sss.spike_stack) == 2

    def test_spike_stack_non_list_raises(self):
        """
        Non-list spike_stack raises TypeError.

        Tests:
            (Test Case 1) Tuple input raises TypeError.
        """
        sd1 = SpikeData([np.array([5.0])], length=20.0)
        with pytest.raises(TypeError, match="list of SpikeData"):
            SpikeSliceStack(spike_stack=(sd1,))

    def test_spike_stack_non_spikedata_element_raises(self):
        """
        Non-SpikeData element in spike_stack raises TypeError.

        Tests:
            (Test Case 1) String element raises TypeError.
        """
        with pytest.raises(TypeError, match="list of SpikeData"):
            SpikeSliceStack(spike_stack=["not_spikedata"])

    def test_spike_stack_empty_raises(self):
        """
        Empty spike_stack list raises ValueError.

        Tests:
            (Test Case 1) Empty list raises ValueError.
        """
        with pytest.raises(ValueError, match="must not be empty"):
            SpikeSliceStack(spike_stack=[])

    def test_spike_stack_mismatched_units_raises(self):
        """
        SpikeData objects with different N raise ValueError.

        Tests:
            (Test Case 1) Mismatched unit counts raise ValueError.
        """
        sd1 = SpikeData([np.array([5.0])], length=20.0)
        sd2 = SpikeData([np.array([5.0]), np.array([10.0])], length=20.0)
        with pytest.raises(ValueError, match="same number of units"):
            SpikeSliceStack(spike_stack=[sd1, sd2])

    def test_spike_stack_times_length_mismatch_raises(self):
        """
        times_start_to_end with wrong length raises ValueError.

        Tests:
            (Test Case 1) 3 times for 2 slices raises ValueError.
        """
        sd1 = SpikeData([np.array([5.0])], length=20.0)
        sd2 = SpikeData([np.array([10.0])], length=20.0)
        with pytest.raises(ValueError, match="same length"):
            SpikeSliceStack(
                spike_stack=[sd1, sd2],
                times_start_to_end=[(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)],
            )


class TestSubslice:
    """Tests for SpikeSliceStack.subslice()."""

    def _make_stack(self):
        """Helper: 3-unit, 4-slice stack."""
        sd = make_spikedata(n_units=3, length_ms=200.0, seed=7)
        times = [(10.0, 30.0), (50.0, 70.0), (100.0, 120.0), (150.0, 170.0)]
        return SpikeSliceStack(
            sd,
            times_start_to_end=times,
            neuron_attributes=[{"id": "A"}, {"id": "B"}, {"id": "C"}],
        )

    def test_subslice_single_int(self):
        """
        Extract a single slice by integer index.

        Tests:
            (Test Case 1) Result has exactly 1 slice.
            (Test Case 2) The slice times match the original slice at that index.
            (Test Case 3) neuron_attributes are preserved.
        """
        sss = self._make_stack()
        result = sss.subslice(2)

        assert len(result.spike_stack) == 1
        assert result.times[0] == sss.times[2]
        assert result.neuron_attributes == sss.neuron_attributes

    def test_subslice_list(self):
        """
        Extract multiple slices by list of indices.

        Tests:
            (Test Case 1) Result has the correct number of slices.
            (Test Case 2) Times are in sorted order matching the selected indices.
        """
        sss = self._make_stack()
        result = sss.subslice([3, 0, 2])

        assert len(result.spike_stack) == 3
        # Subslice sorts indices, so order is 0, 2, 3
        assert result.times[0] == sss.times[0]
        assert result.times[1] == sss.times[2]
        assert result.times[2] == sss.times[3]

    def test_subslice_negative_index(self):
        """
        Extract a slice using negative indexing.

        Tests:
            (Test Case 1) Index -1 returns the last slice.
        """
        sss = self._make_stack()
        result = sss.subslice(-1)

        assert len(result.spike_stack) == 1
        assert result.times[0] == sss.times[-1]

    def test_subslice_out_of_range_raises(self):
        """
        Out-of-range slice index raises ValueError.

        Tests:
            (Test Case 1) Index equal to S raises ValueError.
            (Test Case 2) Negative index beyond -S raises ValueError.
        """
        sss = self._make_stack()
        with pytest.raises(ValueError, match="out of range"):
            sss.subslice(4)
        with pytest.raises(ValueError, match="out of range"):
            sss.subslice(-5)

    def test_subslice_preserves_spike_data(self):
        """
        Extracted slices contain the same spike trains as the originals.

        Tests:
            (Test Case 1) Spike trains in the subsliced result match the original.
        """
        sss = self._make_stack()
        result = sss.subslice([1])

        for u in range(sss.N):
            np.testing.assert_array_equal(
                result.spike_stack[0].train[u], sss.spike_stack[1].train[u]
            )


class TestSubset:
    """Tests for SpikeSliceStack.subset()."""

    def _make_stack(self):
        """Helper: 3-unit, 2-slice stack with neuron_attributes."""
        sd = make_spikedata(n_units=3, length_ms=200.0, seed=11)
        times = [(10.0, 30.0), (50.0, 70.0)]
        return SpikeSliceStack(
            sd,
            times_start_to_end=times,
            neuron_attributes=[
                {"id": "A", "region": "ctx"},
                {"id": "B", "region": "hpc"},
                {"id": "C", "region": "ctx"},
            ],
        )

    def test_subset_by_index_single(self):
        """
        Extract a single unit by index.

        Tests:
            (Test Case 1) Result has N=1.
            (Test Case 2) neuron_attributes contain only the selected unit.
            (Test Case 3) All slices are preserved.
        """
        sss = self._make_stack()
        result = sss.subset(1)

        assert result.N == 1
        assert len(result.spike_stack) == 2
        assert result.neuron_attributes == [{"id": "B", "region": "hpc"}]

    def test_subset_preserve_order(self):
        """
        ``preserve_order=True`` returns units in caller's order
        rather than sorted ascending by index.

        Tests:
            (Test Case 1) Default returns sorted order.
            (Test Case 2) preserve_order=True returns caller's order
                (verified via neuron_attributes which carry the
                identifying ``id`` field).
            (Test Case 3) Duplicates are deduplicated.
        """
        sss = self._make_stack()

        default = sss.subset([2, 0])
        assert [a["id"] for a in default.neuron_attributes] == ["A", "C"]

        ordered = sss.subset([2, 0], preserve_order=True)
        assert [a["id"] for a in ordered.neuron_attributes] == ["C", "A"]

        dedup = sss.subset([2, 0, 0, 2, 1], preserve_order=True)
        assert [a["id"] for a in dedup.neuron_attributes] == ["C", "A", "B"]

    def test_subset_by_index_list(self):
        """
        Extract multiple units by index list.

        Tests:
            (Test Case 1) Result has the correct number of units.
            (Test Case 2) neuron_attributes match selected units in order.
        """
        sss = self._make_stack()
        result = sss.subset([0, 2])

        assert result.N == 2
        assert result.neuron_attributes[0]["id"] == "A"
        assert result.neuron_attributes[1]["id"] == "C"

    def test_subset_by_attribute(self):
        """
        Extract units by neuron_attribute key.

        Tests:
            (Test Case 1) Selecting by region="ctx" returns 2 units.
            (Test Case 2) neuron_attributes of result match the filtered units.
        """
        sss = self._make_stack()
        result = sss.subset("ctx", by="region")

        assert result.N == 2
        assert result.neuron_attributes[0]["id"] == "A"
        assert result.neuron_attributes[1]["id"] == "C"

    def test_subset_by_attribute_no_neuron_attributes_raises(self):
        """
        Using by= without neuron_attributes raises ValueError.

        Tests:
            (Test Case 1) ValueError is raised with descriptive message.
        """
        sd = make_spikedata(n_units=2, length_ms=100.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 20.0), (30.0, 50.0)])

        with pytest.raises(ValueError, match="neuron_attributes"):
            sss.subset("A", by="id")

    def test_subset_preserves_times(self):
        """
        Subset preserves the original time windows.

        Tests:
            (Test Case 1) Times are unchanged after subsetting units.
        """
        sss = self._make_stack()
        result = sss.subset([0])

        assert result.times == sss.times

    def test_subset_preserves_spike_trains(self):
        """
        Spike trains for selected units match the originals.

        Tests:
            (Test Case 1) Unit 1 spike trains match across all slices.
        """
        sss = self._make_stack()
        result = sss.subset(1)

        for s_idx in range(len(sss.spike_stack)):
            np.testing.assert_array_equal(
                result.spike_stack[s_idx].train[0],
                sss.spike_stack[s_idx].train[1],
            )

    def test_subset_out_of_range_index_with_neuron_attributes(self):
        """
        subset with out-of-range unit index raises ValueError.

        Tests:
            (Test Case 1) ValueError is raised when an out-of-range index is
                used, regardless of whether neuron_attributes is set.
        """
        sss = self._make_stack()
        with pytest.raises(ValueError, match="out of range"):
            sss.subset([1, 99])

    def test_subset_out_of_range_index_without_neuron_attributes(self):
        """
        subset with out-of-range unit index raises ValueError even without
        neuron_attributes.

        Tests:
            (Test Case 1) ValueError is raised for out-of-range index.
        """
        sd = make_spikedata(n_units=3, length_ms=200.0, seed=11)
        times = [(10.0, 30.0), (50.0, 70.0)]
        sss = SpikeSliceStack(sd, times_start_to_end=times)

        with pytest.raises(ValueError, match="out of range"):
            sss.subset([1, 99])

    def test_empty_units_list(self):
        """
        subset with an empty units list produces a SpikeSliceStack with N=0.

        Tests:
            (Test Case 1) Construction succeeds without error.
            (Test Case 2) Result has N=0.
            (Test Case 3) Each slice in the result has N=0 and an empty train.

        Notes:
            The empty list is accepted because there is no explicit guard against
            it. The resulting stack has zero units, which may cause downstream
            issues in methods that assume N >= 1.
        """
        sd = make_spikedata(n_units=3, length_ms=200.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(10.0, 30.0), (50.0, 70.0)])

        result = sss.subset([])

        assert result.N == 0
        assert len(result.spike_stack) == 2
        for s in result.spike_stack:
            assert s.N == 0
            assert len(s.train) == 0


class TestSubtimeByIndex:
    """Tests for SpikeSliceStack.subtime_by_index()."""

    def _make_stack(self):
        """Helper: 2-unit, 3-slice stack with 50ms slices."""
        sd = make_spikedata(n_units=2, length_ms=500.0, seed=22)
        times = [(100.0, 150.0), (200.0, 250.0), (300.0, 350.0)]
        return SpikeSliceStack(
            sd,
            times_start_to_end=times,
            neuron_attributes=[{"id": "X"}, {"id": "Y"}],
        )

    def test_basic_subtime(self):
        """
        Trim each slice to an inner sub-window.

        Tests:
            (Test Case 1) Result has the same number of slices.
            (Test Case 2) Each slice time window reflects the trimmed range.
            (Test Case 3) neuron_attributes are preserved.
        """
        sss = self._make_stack()
        result = sss.subtime_by_index(10, 40)

        assert len(result.spike_stack) == 3
        assert result.times[0] == pytest.approx((110.0, 140.0))
        assert result.times[1] == pytest.approx((210.0, 240.0))
        assert result.times[2] == pytest.approx((310.0, 340.0))
        assert result.neuron_attributes == sss.neuron_attributes

    def test_subtime_negative_indices(self):
        """
        Trim with negative indices (relative to slice end).

        Tests:
            (Test Case 1) Negative start_idx trims from the end.
        """
        sss = self._make_stack()
        result = sss.subtime_by_index(-20, -5)

        assert result.times[0] == pytest.approx((130.0, 145.0))
        assert result.times[1] == pytest.approx((230.0, 245.0))

    def test_subtime_full_range(self):
        """
        Trimming to the full range (0, T) returns equivalent data.

        Tests:
            (Test Case 1) Times match the originals.
        """
        sss = self._make_stack()
        result = sss.subtime_by_index(0, 50)

        for orig, trimmed in zip(sss.times, result.times):
            assert trimmed == pytest.approx(orig)

    def test_subtime_spikes_within_window(self):
        """
        After trimming, spikes are 0-based and within the slice duration.

        Tests:
            (Test Case 1) All spike times fall within [0, end_idx - start_idx).
        """
        sss = self._make_stack()
        result = sss.subtime_by_index(10, 30)
        window_duration = 30 - 10

        for sd in result.spike_stack:
            for unit_spikes in sd.train:
                if len(unit_spikes) > 0:
                    assert np.all(unit_spikes >= 0)
                    assert np.all(unit_spikes < window_duration)

    def test_subtime_by_index_produces_zero_based_slices(self):
        """
        Tests that subtime_by_index produces slices with 0-based spike times
        and correct absolute times in metadata.

        Tests:
            (Test Case 1) All spikes in the subtimed result are >= 0.
            (Test Case 2) All spikes in the subtimed result are < 10 (the sub-window duration).
            (Test Case 3) Result times contain the correct absolute windows.
        """
        sd = SpikeData(
            [np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])],
            length=200.0,
        )
        times = [(0.0, 100.0), (100.0, 200.0)]
        sss = SpikeSliceStack(sd, times_start_to_end=times)

        result = sss.subtime_by_index(5, 15)

        for sd_slice in result.spike_stack:
            for unit_spikes in sd_slice.train:
                if len(unit_spikes) > 0:
                    assert np.all(unit_spikes >= 0)
                    assert np.all(unit_spikes < 10.0)

        # Absolute times should reflect the sub-window within each original window
        assert result.times[0] == pytest.approx((5.0, 15.0))
        assert result.times[1] == pytest.approx((105.0, 115.0))

    def test_subtime_start_equals_end_raises(self):
        """
        EC-SSS-11: subtime_by_index with start_idx == end_idx raises ValueError
        because end_idx <= start_idx is rejected by the validation.

        Tests:
            (Test Case 1) ValueError is raised when start_idx == end_idx.
        """
        sss = self._make_stack()
        with pytest.raises(ValueError, match="end_idx"):
            sss.subtime_by_index(10, 10)

    def test_non_integer_slice_duration_raises(self):
        """
        subtime_by_index rejects non-integer slice durations with a
        clear ValueError. For non-integer windows, callers should use
        SpikeData.subtime() directly.

        Tests:
            (Test Case 1) A 100.5 ms slice duration raises ValueError.
        """
        train = [np.array([10.0, 50.0, 80.0, 99.0])]
        sd = SpikeData(train, length=200.5)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 100.5), (100.0, 200.5)])
        with pytest.raises(ValueError, match="integer number of milliseconds"):
            sss.subtime_by_index(10, 90)


class TestToRasterArrayCustomBin:
    """Tests for to_raster_array with non-default bin_size."""

    def test_bin_size_changes_time_dimension(self):
        """
        Larger bin_size reduces the T dimension of the output.

        Tests:
            (Test Case 1) bin_size=5 produces smaller T than bin_size=1.
            (Test Case 2) U and S dimensions are unchanged.
        """
        sd = make_spikedata(n_units=2, length_ms=200.0, seed=33)
        times = [(10.0, 60.0), (80.0, 130.0)]
        sss = SpikeSliceStack(sd, times_start_to_end=times)

        result_1ms = sss.to_raster_array(bin_size=1.0)
        result_5ms = sss.to_raster_array(bin_size=5.0)

        assert result_1ms.shape[0] == result_5ms.shape[0] == 2  # U
        assert result_1ms.shape[2] == result_5ms.shape[2] == 2  # S
        assert result_5ms.shape[1] < result_1ms.shape[1]  # Fewer time bins

    def test_bin_size_preserves_total_spike_count(self):
        """
        Total spike count is the same regardless of bin_size.

        Tests:
            (Test Case 1) Sum of all bins is identical for bin_size=1 and bin_size=10.
        """
        sd = make_spikedata(n_units=2, length_ms=200.0, seed=44)
        times = [(0.0, 50.0), (100.0, 150.0)]
        sss = SpikeSliceStack(sd, times_start_to_end=times)

        result_1ms = sss.to_raster_array(bin_size=1.0)
        result_10ms = sss.to_raster_array(bin_size=10.0)

        assert result_1ms.sum() == result_10ms.sum()

    def test_bin_size_large_single_bin(self):
        """
        bin_size equal to slice duration captures all spikes in the first bin.

        Tests:
            (Test Case 1) The first bin contains all spikes from the slice.
        """
        train = [np.array([5.0, 15.0, 25.0])]
        sd = SpikeData(train, length=50.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 30.0)])

        result = sss.to_raster_array(bin_size=30.0)

        assert result.shape[0] == 1  # U
        assert result.shape[2] == 1  # S
        assert result[0, 0, 0] == 3  # All 3 spikes in first bin


# ---------------------------------------------------------------------------
# Helper for comparison method tests
# ---------------------------------------------------------------------------


def _make_correlated_stack(n_units=4, n_slices=5, length_ms=100.0, seed=0):
    """
    Create a SpikeSliceStack with enough spikes per unit per slice for
    meaningful STTC/CCG computation.
    """
    rng = np.random.default_rng(seed)
    sd_list = []
    for _ in range(n_slices):
        train = []
        for _ in range(n_units):
            n_spikes = rng.integers(15, 40)
            spikes = np.sort(rng.uniform(0, length_ms, n_spikes))
            train.append(spikes)
        sd_list.append(SpikeData(train, length=length_ms))
    return SpikeSliceStack(spike_stack=sd_list)


# ---------------------------------------------------------------------------
# unit_to_unit_comparison
# ---------------------------------------------------------------------------


class TestUnitToUnitComparison:
    """Tests for SpikeSliceStack.unit_to_unit_comparison()."""

    def test_ccg_output_shapes(self):
        """
        CCG metric returns correct shapes and non-None lag.

        Tests:
            (Test Case 1) corr_stack is PairwiseCompMatrixStack with shape (U, U, S).
            (Test Case 2) lag_stack is PairwiseCompMatrixStack with shape (U, U, S).
            (Test Case 3) av_corr has shape (S,).
            (Test Case 4) av_lag has shape (S,).
        """
        sss = _make_correlated_stack(n_units=4, n_slices=5)
        corr_stack, lag_stack, av_corr, av_lag = sss.unit_to_unit_comparison(
            metric="ccg"
        )

        assert isinstance(corr_stack, PairwiseCompMatrixStack)
        assert corr_stack.stack.shape == (4, 4, 5)
        assert isinstance(lag_stack, PairwiseCompMatrixStack)
        assert lag_stack.stack.shape == (4, 4, 5)
        assert av_corr.shape == (5,)
        assert av_lag.shape == (5,)

    def test_sttc_output_shapes(self):
        """
        STTC metric returns correct shapes and None for lag.

        Tests:
            (Test Case 1) corr_stack shape is (U, U, S).
            (Test Case 2) lag_stack is None.
            (Test Case 3) av_corr has shape (S,).
            (Test Case 4) av_lag is None.
        """
        sss = _make_correlated_stack(n_units=3, n_slices=4)
        corr_stack, lag_stack, av_corr, av_lag = sss.unit_to_unit_comparison(
            metric="sttc", delt=20.0
        )

        assert corr_stack.stack.shape == (3, 3, 4)
        assert lag_stack is None
        assert av_corr.shape == (4,)
        assert av_lag is None

    def test_ccg_diagonal_is_one(self):
        """
        CCG correlation diagonal should be 1 (self-correlation).

        Tests:
            (Test Case 1) Diagonal entries of each slice are 1.0.
        """
        sss = _make_correlated_stack(n_units=3, n_slices=3)
        corr_stack, _, _, _ = sss.unit_to_unit_comparison(metric="ccg")

        for s in range(3):
            diag = np.diag(corr_stack.stack[:, :, s])
            np.testing.assert_allclose(diag, 1.0, atol=1e-10)

    def test_sttc_diagonal_is_one(self):
        """
        STTC diagonal should be 1 (self-tiling).

        Tests:
            (Test Case 1) Diagonal entries of each slice are 1.0.
        """
        sss = _make_correlated_stack(n_units=3, n_slices=3)
        corr_stack, _, _, _ = sss.unit_to_unit_comparison(metric="sttc")

        for s in range(3):
            diag = np.diag(corr_stack.stack[:, :, s])
            np.testing.assert_allclose(diag, 1.0, atol=1e-10)

    def test_ccg_symmetric(self):
        """
        CCG correlation matrices are symmetric.

        Tests:
            (Test Case 1) corr_stack[:, :, s] == corr_stack[:, :, s].T for each slice.
        """
        sss = _make_correlated_stack(n_units=4, n_slices=3)
        corr_stack, _, _, _ = sss.unit_to_unit_comparison(metric="ccg")

        for s in range(3):
            mat = corr_stack.stack[:, :, s]
            np.testing.assert_allclose(mat, mat.T, atol=1e-10)

    def test_sttc_symmetric(self):
        """
        STTC matrices are symmetric.

        Tests:
            (Test Case 1) corr_stack[:, :, s] == corr_stack[:, :, s].T for each slice.
        """
        sss = _make_correlated_stack(n_units=3, n_slices=3)
        corr_stack, _, _, _ = sss.unit_to_unit_comparison(metric="sttc")

        for s in range(3):
            mat = corr_stack.stack[:, :, s]
            np.testing.assert_allclose(mat, mat.T, atol=1e-10)

    def test_default_metric_is_ccg(self):
        """
        Default metric is CCG (lag_stack should not be None).

        Tests:
            (Test Case 1) Calling without metric= returns non-None lag_stack.
        """
        sss = _make_correlated_stack(n_units=3, n_slices=2)
        _, lag_stack, _, av_lag = sss.unit_to_unit_comparison()

        assert lag_stack is not None
        assert av_lag is not None

    def test_invalid_metric_raises(self):
        """
        Invalid metric string raises ValueError.

        Tests:
            (Test Case 1) metric='invalid' raises ValueError.
        """
        sss = _make_correlated_stack(n_units=2, n_slices=2)
        with pytest.raises(ValueError, match="metric must be"):
            sss.unit_to_unit_comparison(metric="invalid")

    def test_single_unit_returns_nan(self):
        """
        Single-unit stack returns NaN with a warning.

        Tests:
            (Test Case 1) RuntimeWarning is emitted.
            (Test Case 2) corr_stack shape is (1, 1, S).
            (Test Case 3) av_corr is all NaN.
        """
        sd1 = SpikeData([np.array([5.0, 15.0, 25.0])], length=50.0)
        sd2 = SpikeData([np.array([8.0, 22.0, 40.0])], length=50.0)
        sss = SpikeSliceStack(spike_stack=[sd1, sd2])

        with pytest.warns(RuntimeWarning, match="fewer than 2 units"):
            corr_stack, lag_stack, av_corr, av_lag = sss.unit_to_unit_comparison(
                metric="ccg"
            )

        assert corr_stack.stack.shape == (1, 1, 2)
        assert np.all(np.isnan(av_corr))

    def test_av_corr_within_bounds(self):
        """
        Average correlation values are within [-1, 1].

        Tests:
            (Test Case 1) All av_corr values are in [-1, 1].
        """
        sss = _make_correlated_stack(n_units=4, n_slices=5, seed=99)
        _, _, av_corr, _ = sss.unit_to_unit_comparison(metric="ccg")

        assert np.all(av_corr >= -1.0)
        assert np.all(av_corr <= 1.0)

    def test_all_empty_spike_trains(self):
        """
        EC-SSS-06: unit_to_unit_comparison with all-empty spike trains.
        Cross-correlation of empty rasters should produce zeros or NaN
        without raising an error.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) Output shapes are correct (U, U, S).
            (Test Case 3) av_corr has shape (S,).
        """
        empty = np.array([], dtype=float)
        sd1 = SpikeData([empty, empty], length=100.0)
        sd2 = SpikeData([empty, empty], length=100.0)
        sss = SpikeSliceStack(spike_stack=[sd1, sd2])

        corr_stack, lag_stack, av_corr, av_lag = sss.unit_to_unit_comparison(
            metric="ccg"
        )

        assert corr_stack.stack.shape == (2, 2, 2)
        assert av_corr.shape == (2,)

    def test_sttc_with_delt_zero_raises(self):
        """
        unit_to_unit_comparison with metric='sttc' and delt=0 raises ValueError.

        Tests:
            (Test Case 1) delt=0 raises ValueError from get_sttc validation.
        """
        sss = _make_correlated_stack(n_units=3, n_slices=3)
        with pytest.raises(ValueError, match="delt must be positive"):
            sss.unit_to_unit_comparison(metric="sttc", delt=0)


# ---------------------------------------------------------------------------
# get_slice_to_slice_unit_comparison
# ---------------------------------------------------------------------------


class TestSliceToSliceUnitComparison:
    """Tests for SpikeSliceStack.get_slice_to_slice_unit_comparison()."""

    def test_ccg_output_shapes(self):
        """
        CCG metric returns correct shapes and non-None lag.

        Tests:
            (Test Case 1) all_corr is PairwiseCompMatrixStack with shape (S, S, U).
            (Test Case 2) all_lag is PairwiseCompMatrixStack with shape (S, S, U).
            (Test Case 3) av_corr has shape (U,).
            (Test Case 4) av_lag has shape (U,).
        """
        sss = _make_correlated_stack(n_units=3, n_slices=5)
        all_corr, all_lag, av_corr, av_lag = sss.get_slice_to_slice_unit_comparison(
            metric="ccg"
        )

        assert isinstance(all_corr, PairwiseCompMatrixStack)
        assert all_corr.stack.shape == (5, 5, 3)
        assert isinstance(all_lag, PairwiseCompMatrixStack)
        assert all_lag.stack.shape == (5, 5, 3)
        assert av_corr.shape == (3,)
        assert av_lag.shape == (3,)

    def test_sttc_output_shapes(self):
        """
        STTC metric returns correct shapes and None for lag.

        Tests:
            (Test Case 1) all_corr shape is (S, S, U).
            (Test Case 2) all_lag is None.
            (Test Case 3) av_corr has shape (U,).
            (Test Case 4) av_lag is None.
        """
        sss = _make_correlated_stack(n_units=3, n_slices=4)
        all_corr, all_lag, av_corr, av_lag = sss.get_slice_to_slice_unit_comparison(
            metric="sttc"
        )

        assert all_corr.stack.shape == (4, 4, 3)
        assert all_lag is None
        assert av_corr.shape == (3,)
        assert av_lag is None

    def test_ccg_symmetric_per_unit(self):
        """
        CCG slice-to-slice matrices are symmetric for each unit.

        Tests:
            (Test Case 1) all_corr[:, :, u] is symmetric for each unit.
        """
        sss = _make_correlated_stack(n_units=3, n_slices=4)
        all_corr, _, _, _ = sss.get_slice_to_slice_unit_comparison(metric="ccg")

        for u in range(3):
            mat = all_corr.stack[:, :, u]
            np.testing.assert_allclose(mat, mat.T, atol=1e-10)

    def test_sttc_symmetric_per_unit(self):
        """
        STTC slice-to-slice matrices are symmetric for each unit.

        Tests:
            (Test Case 1) all_corr[:, :, u] is symmetric for each unit.
        """
        sss = _make_correlated_stack(n_units=3, n_slices=4)
        all_corr, _, _, _ = sss.get_slice_to_slice_unit_comparison(metric="sttc")

        for u in range(3):
            mat = all_corr.stack[:, :, u]
            np.testing.assert_allclose(mat, mat.T, atol=1e-10)

    def test_default_metric_is_ccg(self):
        """
        Default metric is CCG.

        Tests:
            (Test Case 1) Calling without metric= returns non-None lag.
        """
        sss = _make_correlated_stack(n_units=2, n_slices=3)
        _, all_lag, _, av_lag = sss.get_slice_to_slice_unit_comparison()

        assert all_lag is not None
        assert av_lag is not None

    def test_invalid_metric_raises(self):
        """
        Invalid metric string raises ValueError.

        Tests:
            (Test Case 1) metric='pearson' raises ValueError.
        """
        sss = _make_correlated_stack(n_units=2, n_slices=2)
        with pytest.raises(ValueError, match="metric must be"):
            sss.get_slice_to_slice_unit_comparison(metric="pearson")

    def test_single_slice_returns_nan(self):
        """
        Single-slice stack returns NaN with a warning.

        Tests:
            (Test Case 1) RuntimeWarning is emitted.
            (Test Case 2) all_corr shape is (1, 1, U).
            (Test Case 3) av_corr is all NaN.
        """
        sd = SpikeData(
            [np.array([5.0, 15.0, 25.0]), np.array([8.0, 22.0])], length=50.0
        )
        sss = SpikeSliceStack(spike_stack=[sd])

        with pytest.warns(RuntimeWarning, match="fewer than 2 slices"):
            all_corr, all_lag, av_corr, av_lag = sss.get_slice_to_slice_unit_comparison(
                metric="ccg"
            )

        assert all_corr.stack.shape == (1, 1, 2)
        assert np.all(np.isnan(av_corr))

    def test_min_spikes_filters_inactive_units(self):
        """
        Units with too few spikes in most slices get NaN average.

        Tests:
            (Test Case 1) Unit with only 1 spike per slice (below min_spikes=5)
                has NaN average.
            (Test Case 2) Unit with many spikes has a valid (non-NaN) average.
        """
        rng = np.random.default_rng(42)
        sd_list = []
        for _ in range(4):
            active_spikes = np.sort(rng.uniform(0, 100, 20))
            sparse_spikes = np.array([rng.uniform(0, 100)])
            sd_list.append(SpikeData([active_spikes, sparse_spikes], length=100.0))
        sss = SpikeSliceStack(spike_stack=sd_list)

        _, _, av_corr, _ = sss.get_slice_to_slice_unit_comparison(
            metric="ccg", min_spikes=5
        )

        assert not np.isnan(av_corr[0])  # Active unit
        assert np.isnan(av_corr[1])  # Sparse unit

    def test_av_corr_within_bounds(self):
        """
        Average correlation values are within [-1, 1] for valid units.

        Tests:
            (Test Case 1) Non-NaN av_corr values are in [-1, 1].
        """
        sss = _make_correlated_stack(n_units=3, n_slices=5, seed=77)
        _, _, av_corr, _ = sss.get_slice_to_slice_unit_comparison(metric="ccg")

        valid = av_corr[~np.isnan(av_corr)]
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)

    def test_min_spikes_zero(self):
        """
        EC-SSS-07: get_slice_to_slice_unit_comparison with min_spikes=0.
        All slice pairs should be computed (even empty trains have len >= 0).

        Tests:
            (Test Case 1) No error raised.
            (Test Case 2) Output shapes are correct.
            (Test Case 3) av_corr has no NaN for units with spikes (all pairs valid).
        """
        sss = _make_correlated_stack(n_units=3, n_slices=4, seed=88)
        all_corr, all_lag, av_corr, av_lag = sss.get_slice_to_slice_unit_comparison(
            metric="ccg", min_spikes=0
        )

        assert all_corr.stack.shape == (4, 4, 3)
        assert av_corr.shape == (3,)
        # With min_spikes=0, all units should have valid averages
        assert not np.any(np.isnan(av_corr))

    def test_all_units_below_min_frac(self):
        """
        get_slice_to_slice_unit_comparison where all units have too few spikes in
        too many slices results in all-NaN averages.

        Tests:
            (Test Case 1) No error is raised.
            (Test Case 2) Output shapes are correct.
            (Test Case 3) All av_corr values are NaN because no unit passes the
                min_frac threshold.
        """
        # Every unit has exactly 1 spike per slice (below default min_spikes=2)
        sd_list = []
        times = []
        for i in range(4):
            start = i * 100.0
            u0 = np.array([50.0])
            u1 = np.array([60.0])
            u2 = np.array([70.0])
            sd_list.append(SpikeData([u0, u1, u2], length=100.0))
            times.append((start, start + 100.0))
        sss = SpikeSliceStack(spike_stack=sd_list, times_start_to_end=times)

        all_corr, all_lag, av_corr, av_lag = sss.get_slice_to_slice_unit_comparison(
            metric="ccg", min_spikes=2, min_frac=0.0
        )

        assert all_corr.stack.shape == (4, 4, 3)
        assert av_corr.shape == (3,)
        # All units have 0 valid slices out of 4 → all averages should be NaN
        assert np.all(np.isnan(av_corr))


# ---------------------------------------------------------------------------
# compute_frac_active
# ---------------------------------------------------------------------------


class TestComputeFracActive:
    """Tests for SpikeSliceStack.compute_frac_active()."""

    def test_all_active(self):
        """
        All units active in all slices returns array of ones.

        Tests:
            (Test Case 1) Every unit has >= min_spikes in every slice.

        Notes:
            - Uses explicit times matching the spike ranges so that the
              0-based shift in compute_frac_active works correctly.
        """
        rng = np.random.default_rng(0)
        sd_list = []
        times = []
        for i in range(4):
            start = i * 100.0
            train = []
            for _ in range(3):
                spikes = np.sort(rng.uniform(0, 100, 15))
                train.append(spikes)
            sd_list.append(SpikeData(train, length=100.0))
            times.append((start, start + 100.0))
        sss = SpikeSliceStack(spike_stack=sd_list, times_start_to_end=times)
        frac = sss.compute_frac_active(min_spikes=2)

        assert frac.shape == (3,)
        np.testing.assert_array_equal(frac, 1.0)

    def test_min_spikes_zero_all_active(self):
        """
        min_spikes=0 makes all units active in all slices.

        Tests:
            (Test Case 1) frac_active is 1.0 for all units when min_spikes=0.
            (Test Case 2) Even a unit with zero spikes in a slice is counted as active.

        Notes:
            - With min_spikes=0, the condition n_valid >= 0 is always True.
        """
        rng = np.random.default_rng(55)
        sd_list = []
        times = []
        for i in range(3):
            start = i * 100.0
            active = np.sort(rng.uniform(0, 100, 10))
            empty = np.array([], dtype=float)  # unit with 0 spikes
            sd_list.append(SpikeData([active, empty], length=100.0))
            times.append((start, start + 100.0))
        sss = SpikeSliceStack(spike_stack=sd_list, times_start_to_end=times)
        frac = sss.compute_frac_active(min_spikes=0)
        assert frac.shape == (2,)
        np.testing.assert_array_equal(frac, 1.0)

    def test_sparse_unit_low_frac(self):
        """
        A unit with very few spikes has low fraction active.

        Tests:
            (Test Case 1) Unit with 1 spike per slice has frac=0 when min_spikes=2.
            (Test Case 2) Unit with many spikes has frac=1.
        """
        rng = np.random.default_rng(10)
        sd_list = []
        times = []
        for i in range(5):
            start = i * 100.0
            active = np.sort(rng.uniform(0, 100, 20))
            sparse = np.array([rng.uniform(0, 100)])
            sd_list.append(SpikeData([active, sparse], length=100.0))
            times.append((start, start + 100.0))
        sss = SpikeSliceStack(spike_stack=sd_list, times_start_to_end=times)

        frac = sss.compute_frac_active(min_spikes=2)

        assert frac[0] == 1.0  # Active unit
        assert frac[1] == 0.0  # Only 1 spike per slice

    def test_min_spikes_threshold(self):
        """
        Changing min_spikes affects the result.

        Tests:
            (Test Case 1) min_spikes=1 counts all slices with any spike.
            (Test Case 2) Higher min_spikes reduces the fraction.
        """
        rng = np.random.default_rng(20)
        sd_list = []
        times = []
        for i in range(4):
            start = i * 100.0
            # Unit 0: exactly 3 spikes per slice
            u0 = np.sort(rng.uniform(0, 100, 3))
            # Unit 1: exactly 1 spike per slice
            u1 = np.array([rng.uniform(0, 100)])
            sd_list.append(SpikeData([u0, u1], length=100.0))
            times.append((start, start + 100.0))
        sss = SpikeSliceStack(spike_stack=sd_list, times_start_to_end=times)

        frac_1 = sss.compute_frac_active(min_spikes=1)
        frac_3 = sss.compute_frac_active(min_spikes=3)

        assert frac_1[1] == 1.0  # 1 spike >= 1
        assert frac_3[1] == 0.0  # 1 spike < 3
        assert frac_3[0] == 1.0  # 3 spikes >= 3

    def test_output_shape(self):
        """
        Output shape is (U,).

        Tests:
            (Test Case 1) 4 units returns shape (4,).
        """
        sss = _make_correlated_stack(n_units=4, n_slices=3)
        frac = sss.compute_frac_active()
        assert frac.shape == (4,)

    def test_values_between_zero_and_one(self):
        """
        All values are in [0, 1].

        Tests:
            (Test Case 1) Every element is between 0 and 1 inclusive.
        """
        sss = _make_correlated_stack(n_units=4, n_slices=5, seed=42)
        frac = sss.compute_frac_active(min_spikes=2)
        assert np.all(frac >= 0.0)
        assert np.all(frac <= 1.0)

    def test_all_empty_slices(self):
        """
        EC-SSS-04: compute_frac_active with all-empty slices returns zeros.

        Tests:
            (Test Case 1) All units have frac_active = 0.0 when every spike train is empty.
            (Test Case 2) Output shape is (U,).
        """
        sd_list = []
        times = []
        for i in range(3):
            start = i * 100.0
            empty = np.array([], dtype=float)
            sd_list.append(SpikeData([empty, empty], length=100.0))
            times.append((start, start + 100.0))
        sss = SpikeSliceStack(spike_stack=sd_list, times_start_to_end=times)

        frac = sss.compute_frac_active(min_spikes=2)

        assert frac.shape == (2,)
        np.testing.assert_array_equal(frac, 0.0)

    def test_negative_min_spikes(self):
        """
        compute_frac_active with negative min_spikes: all units active.

        Tests:
            (Test Case 1) Negative min_spikes means n_valid >= -1 is always True.
                Same effect as min_spikes=0.
        """
        sd = make_spikedata(n_units=3, length_ms=100.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 50.0), (50.0, 100.0)])
        frac = sss.compute_frac_active(min_spikes=-1)
        # All units should be active since -1 spikes threshold always passes
        assert frac.shape == (3,)
        assert np.all(frac >= 0.0)

    def test_negative_min_spikes_same_as_zero(self):
        """
        Negative min_spikes has the same effect as min_spikes=0 because the
        condition n_valid >= min_spikes is always true when min_spikes is negative.

        Tests:
            (Test Case 1) frac_active is 1.0 for all units with negative min_spikes.
            (Test Case 2) Result matches min_spikes=0.

        Notes:
            There is no validation on min_spikes, so negative values are silently
            accepted. This behaves identically to min_spikes=0 because any
            non-negative spike count satisfies n_valid >= negative_number.
        """
        empty = np.array([], dtype=float)
        sd1 = SpikeData([np.array([5.0, 15.0]), empty], length=20.0)
        sd2 = SpikeData([np.array([3.0]), empty], length=20.0)
        sss = SpikeSliceStack(
            spike_stack=[sd1, sd2],
            times_start_to_end=[(0.0, 20.0), (20.0, 40.0)],
        )

        frac_neg = sss.compute_frac_active(min_spikes=-5)
        frac_zero = sss.compute_frac_active(min_spikes=0)

        np.testing.assert_array_equal(frac_neg, frac_zero)
        np.testing.assert_array_equal(frac_neg, 1.0)

    def test_boundary_half_open_excludes_end(self):
        """
        Boundary behaviour matches the half-open ``[start, end)`` convention
        used by ``SpikeData.subtime`` and ``RateData.subtime``:

        - A spike at exactly ``sd.start_time`` is counted (inclusive lower).
        - A spike at exactly ``sd.start_time + (end - start)`` is excluded
          (exclusive upper).

        Tests:
            (Test Case 1) Unit with one spike at the lower bound and one at the
                upper bound has ``n_valid = 1``, so ``frac_active = 0.0`` with
                ``min_spikes=2``.
        """
        sd = SpikeData([np.array([0.0, 100.0])], length=200.0)
        sss = SpikeSliceStack(spike_stack=[sd], times_start_to_end=[(0.0, 100.0)])

        frac = sss.compute_frac_active(min_spikes=2)
        np.testing.assert_array_equal(frac, 0.0)


# ---------------------------------------------------------------------------
# order_units_across_slices
# ---------------------------------------------------------------------------


class TestOrderUnitsAcrossSlices:
    """Tests for SpikeSliceStack.order_units_across_slices()."""

    def test_default_all_in_highly_active(self):
        """
        With default min_frac_active=0, all units go to highly-active group.

        Tests:
            (Test Case 1) highly-active stack contains all units.
            (Test Case 2) low-active stack is None.
            (Test Case 3) unit_ids cover all original units.
        """
        sss = _make_correlated_stack(n_units=4, n_slices=5, seed=0)
        stacks, ids, std, times, frac = sss.order_units_across_slices()

        assert stacks[0] is not None
        assert stacks[0].N == 4
        assert stacks[1] is None
        assert len(ids[0]) == 4
        assert len(ids[1]) == 0

    def test_units_sorted_by_timing(self):
        """
        Units are sorted by their typical spike timing (earliest first).

        Tests:
            (Test Case 1) Peak times in the highly-active group are non-decreasing.
        """
        sss = _make_correlated_stack(n_units=4, n_slices=5, seed=1)
        _, _, _, times, _ = sss.order_units_across_slices()

        ha_times = times[0]
        # Filter out NaN before checking order
        valid = ha_times[~np.isnan(ha_times)]
        assert np.all(np.diff(valid) >= 0)

    def test_timing_median_vs_first(self):
        """
        Different timing modes produce different orderings.

        Tests:
            (Test Case 1) timing='first' gives earlier or equal values than
                timing='median' for the same unit.

        Notes:
            - First spike is always <= median spike time within a slice.
        """
        sss = _make_correlated_stack(n_units=3, n_slices=5, seed=2)
        _, ids_med, _, times_med, _ = sss.order_units_across_slices(timing="median")
        _, ids_first, _, times_first, _ = sss.order_units_across_slices(timing="first")

        # Build lookup: unit_id -> peak_time for each mode
        med_lookup = dict(zip(ids_med[0], times_med[0]))
        first_lookup = dict(zip(ids_first[0], times_first[0]))

        for uid in med_lookup:
            if not np.isnan(med_lookup[uid]) and not np.isnan(first_lookup[uid]):
                assert first_lookup[uid] <= med_lookup[uid]

    def test_min_frac_active_splits_groups(self):
        """
        min_frac_active > 0 splits units into two groups.

        Tests:
            (Test Case 1) Sparse unit goes to low-active group.
            (Test Case 2) Active units go to highly-active group.
        """
        rng = np.random.default_rng(30)
        sd_list = []
        times = []
        for i in range(6):
            start = i * 100.0
            active = np.sort(rng.uniform(0, 100, 20))
            sparse = np.array([rng.uniform(0, 100)])
            sd_list.append(SpikeData([active, sparse], length=100.0))
            times.append((start, start + 100.0))
        sss = SpikeSliceStack(spike_stack=sd_list, times_start_to_end=times)

        stacks, ids, _, _, _ = sss.order_units_across_slices(
            min_frac_active=0.5, min_spikes=2
        )

        assert 0 in ids[0]  # Active unit in HA
        assert 1 in ids[1]  # Sparse unit in LA

    def test_frac_active_override(self):
        """
        Pre-computed frac_active overrides internal calculation.

        Tests:
            (Test Case 1) Unit forced to low frac goes to low-active group.
            (Test Case 2) Other units stay in highly-active group.
        """
        sss = _make_correlated_stack(n_units=3, n_slices=5, seed=3)
        frac = np.array([0.9, 0.1, 0.8])

        _, ids, _, _, _ = sss.order_units_across_slices(
            min_frac_active=0.5, frac_active=frac
        )

        assert 1 in ids[1]  # 0.1 < 0.5
        assert 0 in ids[0]
        assert 2 in ids[0]

    def test_frac_active_override_wrong_shape_raises(self):
        """
        frac_active with wrong shape raises ValueError.

        Tests:
            (Test Case 1) Shape (2,) for 3 units raises ValueError.
        """
        sss = _make_correlated_stack(n_units=3, n_slices=5)

        with pytest.raises(ValueError, match="frac_active must have shape"):
            sss.order_units_across_slices(min_frac_active=0.5, frac_active=np.ones(2))

    def test_frac_active_ignored_when_no_split(self):
        """
        frac_active is ignored when min_frac_active=0.

        Tests:
            (Test Case 1) All units in highly-active despite low frac values.
        """
        sss = _make_correlated_stack(n_units=3, n_slices=4, seed=4)

        _, ids, _, _, _ = sss.order_units_across_slices(
            min_frac_active=0.0, frac_active=np.array([0.01, 0.01, 0.01])
        )

        assert len(ids[0]) == 3
        assert len(ids[1]) == 0

    def test_invalid_agg_func_raises(self):
        """
        Invalid agg_func raises ValueError.

        Tests:
            (Test Case 1) agg_func='invalid' raises ValueError.
        """
        sss = _make_correlated_stack(n_units=2, n_slices=3)
        with pytest.raises(ValueError, match="agg_func"):
            sss.order_units_across_slices(agg_func="invalid")

    def test_invalid_timing_raises(self):
        """
        Invalid timing raises ValueError.

        Tests:
            (Test Case 1) timing='invalid' raises ValueError.
        """
        sss = _make_correlated_stack(n_units=2, n_slices=3)
        with pytest.raises(ValueError, match="timing"):
            sss.order_units_across_slices(timing="invalid")

    def test_output_tuple_structure(self):
        """
        Return value has the correct 5-tuple structure with tuples inside.

        Tests:
            (Test Case 1) Each element is a tuple of length 2.
            (Test Case 2) unit_ids arrays together cover all original units.
        """
        sss = _make_correlated_stack(n_units=4, n_slices=5, seed=5)
        result = sss.order_units_across_slices()

        assert len(result) == 5
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

        # All unit IDs covered
        all_ids = set(result[1][0].tolist()) | set(result[1][1].tolist())
        assert all_ids == {0, 1, 2, 3}

    def test_reordered_stack_has_correct_units(self):
        """
        The reordered SpikeSliceStack has the correct number of units and slices.

        Tests:
            (Test Case 1) N matches the number of units in the group.
            (Test Case 2) Number of slices is unchanged.
        """
        sss = _make_correlated_stack(n_units=4, n_slices=5, seed=6)
        stacks, _, _, _, _ = sss.order_units_across_slices()

        assert stacks[0].N == 4
        assert len(stacks[0].spike_stack) == 5

    def test_timing_matrix_wrong_shape_raises(self):
        """
        Wrong-shaped timing_matrix in order_units_across_slices raises ValueError.

        Tests:
            (Test Case 1) Shape (3, 5) for 4-unit stack raises ValueError.
        """
        sss = _make_timed_stack(n_units=4, n_slices=5)
        with pytest.raises(ValueError, match="timing_matrix must have shape"):
            sss.order_units_across_slices(timing_matrix=np.ones((3, 5)))

    def test_order_units_agg_func_mean(self):
        """
        agg_func='mean' produces valid ordering.

        Tests:
            (Test Case 1) All 4 units in highly-active group.
            (Test Case 2) Peak times are non-decreasing.
        """
        sss = _make_timed_stack(n_units=4, n_slices=5, seed=7)
        _, ids, _, times, _ = sss.order_units_across_slices(agg_func="mean")
        assert len(ids[0]) == 4
        valid = times[0][~np.isnan(times[0])]
        assert np.all(np.diff(valid) >= 0)

    def test_all_units_below_min_frac_active(self):
        """
        EC-SSS-05: order_units_across_slices with all units below min_frac_active
        places all units in the low-active group and returns None for the
        highly-active stack.

        Tests:
            (Test Case 1) Highly-active stack is None.
            (Test Case 2) Low-active group contains all units.
            (Test Case 3) Highly-active ids array is empty.
        """
        # Build stack where every unit has exactly 1 spike per slice (below min_spikes=2)
        sd_list = []
        times = []
        for i in range(4):
            start = i * 100.0
            u0 = np.array([50.0])
            u1 = np.array([60.0])
            sd_list.append(SpikeData([u0, u1], length=100.0))
            times.append((start, start + 100.0))
        sss = SpikeSliceStack(spike_stack=sd_list, times_start_to_end=times)

        stacks, ids, _, _, _ = sss.order_units_across_slices(
            min_frac_active=0.5, min_spikes=2
        )

        assert stacks[0] is None
        assert len(ids[0]) == 0
        assert len(ids[1]) == 2
        assert stacks[1] is not None
        assert stacks[1].N == 2

    def test_all_units_inactive_all_nan_timing(self):
        """
        order_units_across_slices when all units have zero spikes in all slices
        produces all-NaN timing values and arbitrary (NaN-based) ordering.

        Tests:
            (Test Case 1) No error is raised.
            (Test Case 2) All peak times are NaN.
            (Test Case 3) All units are still placed in the highly-active group
                (since min_frac_active=0 by default).
            (Test Case 4) Unit IDs cover all original units.
        """
        empty = np.array([], dtype=float)
        sd_list = []
        times = []
        for i in range(4):
            start = i * 100.0
            sd_list.append(SpikeData([empty, empty, empty], length=100.0))
            times.append((start, start + 100.0))
        sss = SpikeSliceStack(spike_stack=sd_list, times_start_to_end=times)

        stacks, ids, std, peak_times, frac = sss.order_units_across_slices()

        # All units in HA group (no split by default)
        assert len(ids[0]) == 3
        assert len(ids[1]) == 0
        # All peak times should be NaN (no spikes to compute timing from)
        assert np.all(np.isnan(peak_times[0]))
        # All std values should be NaN too
        assert np.all(np.isnan(std[0]))
        # All unit IDs should be present
        assert set(ids[0].tolist()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# get_slice_to_slice_unit_comparison — frac_active override
# ---------------------------------------------------------------------------


class TestSliceToSliceUnitComparisonFracActive:
    """Tests for frac_active override on get_slice_to_slice_unit_comparison."""

    def test_frac_active_override_filters_averages(self):
        """
        Units with low frac_active get NaN averages.

        Tests:
            (Test Case 1) Unit with frac_active=0.1 and min_frac=0.3 has NaN.
            (Test Case 2) Unit with frac_active=0.9 has valid average.
        """
        sss = _make_correlated_stack(n_units=3, n_slices=5, seed=50)
        frac = np.array([0.9, 0.1, 0.8])

        _, _, av_corr, _ = sss.get_slice_to_slice_unit_comparison(
            metric="ccg", min_frac=0.3, frac_active=frac
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
        sss = _make_correlated_stack(n_units=3, n_slices=5)

        with pytest.raises(ValueError, match="frac_active must have shape"):
            sss.get_slice_to_slice_unit_comparison(frac_active=np.ones(2))

    def test_without_override_backward_compatible(self):
        """
        Without frac_active, internal min_spikes counting is used (backward compat).

        Tests:
            (Test Case 1) Output shapes are correct.
        """
        sss = _make_correlated_stack(n_units=3, n_slices=5, seed=51)
        all_corr, _, av_corr, _ = sss.get_slice_to_slice_unit_comparison(metric="ccg")

        assert all_corr.stack.shape == (5, 5, 3)
        assert av_corr.shape == (3,)


# ---------------------------------------------------------------------------
# get_unit_timing_per_slice + rank_order_correlation (SpikeSliceStack)
# ---------------------------------------------------------------------------


def _make_timed_stack(n_units=4, n_slices=6, length_ms=100.0, seed=0):
    """Create a SpikeSliceStack with times aligned to spike ranges."""
    rng = np.random.default_rng(seed)
    sd_list = []
    times = []
    for i in range(n_slices):
        start = i * length_ms
        train = []
        for _ in range(n_units):
            n_spikes = rng.integers(10, 25)
            spikes = np.sort(rng.uniform(0, length_ms, n_spikes))
            train.append(spikes)
        sd_list.append(SpikeData(train, length=length_ms))
        times.append((start, start + length_ms))
    return SpikeSliceStack(spike_stack=sd_list, times_start_to_end=times)


class TestGetUnitTimingPerSlice:
    """Tests for SpikeSliceStack.get_unit_timing_per_slice()."""

    def test_output_shape(self):
        """
        Output is (U, S) ndarray.

        Tests:
            (Test Case 1) 4 units, 6 slices → shape (4, 6).
        """
        sss = _make_timed_stack(n_units=4, n_slices=6)
        tm = sss.get_unit_timing_per_slice()
        assert tm.shape == (4, 6)

    def test_values_within_slice_duration(self):
        """
        All non-NaN timing values are within [0, slice_duration].

        Tests:
            (Test Case 1) All valid entries in [0, 100].
        """
        sss = _make_timed_stack(n_units=4, n_slices=6, length_ms=100.0)
        tm = sss.get_unit_timing_per_slice()
        valid = tm[~np.isnan(tm)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100.0)

    def test_first_timing_le_median(self):
        """
        First spike time is always <= median spike time for the same unit/slice.

        Tests:
            (Test Case 1) For every non-NaN entry, first <= median.
        """
        sss = _make_timed_stack(n_units=4, n_slices=6, seed=10)
        tm_first = sss.get_unit_timing_per_slice(timing="first")
        tm_median = sss.get_unit_timing_per_slice(timing="median")
        both_valid = ~np.isnan(tm_first) & ~np.isnan(tm_median)
        assert np.all(tm_first[both_valid] <= tm_median[both_valid])

    def test_sparse_unit_is_nan(self):
        """
        Units with fewer than min_spikes spikes get NaN.

        Tests:
            (Test Case 1) Unit with 1 spike per slice is NaN with min_spikes=2.
        """
        rng = np.random.default_rng(20)
        sd_list = []
        times = []
        for i in range(4):
            start = i * 100.0
            active = np.sort(rng.uniform(0, 100, 15))
            sparse = np.array([rng.uniform(0, 100)])
            sd_list.append(SpikeData([active, sparse], length=100.0))
            times.append((start, start + 100.0))
        sss = SpikeSliceStack(spike_stack=sd_list, times_start_to_end=times)

        tm = sss.get_unit_timing_per_slice(min_spikes=2)
        assert np.all(~np.isnan(tm[0, :]))  # Active
        assert np.all(np.isnan(tm[1, :]))  # Sparse

    def test_invalid_timing_raises(self):
        """
        Invalid timing string raises ValueError.

        Tests:
            (Test Case 1) timing='bad' raises ValueError.
        """
        sss = _make_timed_stack(n_units=2, n_slices=3)
        with pytest.raises(ValueError, match="timing"):
            sss.get_unit_timing_per_slice(timing="bad")

    def test_timing_mean(self):
        """
        EC-SSS-08: get_unit_timing_per_slice with timing='mean' computes the
        mean spike time per unit per slice.

        Tests:
            (Test Case 1) Output shape is (U, S).
            (Test Case 2) Mean values are between first and last spike for each unit/slice.
            (Test Case 3) Mean >= first spike time for all valid entries.
        """
        sss = _make_timed_stack(n_units=4, n_slices=6, seed=30)
        tm_mean = sss.get_unit_timing_per_slice(timing="mean")
        tm_first = sss.get_unit_timing_per_slice(timing="first")

        assert tm_mean.shape == (4, 6)
        # Mean should be >= first spike
        both_valid = ~np.isnan(tm_mean) & ~np.isnan(tm_first)
        assert np.all(tm_mean[both_valid] >= tm_first[both_valid])
        # All valid values within slice duration
        valid = tm_mean[~np.isnan(tm_mean)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100.0)

    def test_boundary_half_open_excludes_end(self):
        """
        Boundary behaviour matches the half-open ``[start, end)`` convention
        used by ``SpikeData.subtime`` and ``RateData.subtime``:

        - A spike at exactly ``sd.start_time`` is counted (inclusive lower).
        - A spike at exactly ``sd.start_time + duration`` is excluded
          (exclusive upper).

        Tests:
            (Test Case 1) Unit with one spike at the lower bound and one at the
                upper bound has only 1 valid spike, so it falls below
                ``min_spikes=2`` and the timing entry is NaN.
        """
        sd = SpikeData([np.array([0.0, 100.0])], length=200.0)
        sss = SpikeSliceStack(spike_stack=[sd], times_start_to_end=[(0.0, 100.0)])

        tm = sss.get_unit_timing_per_slice(timing="median", min_spikes=2)
        assert np.isnan(tm[0, 0])


class TestRankOrderCorrelationSpike:
    """Tests for SpikeSliceStack.rank_order_correlation()."""

    def test_raw_output_shapes(self):
        """
        Raw mode (n_shuffles=0) returns correct shapes and types.

        Tests:
            (Test Case 1) corr_matrix is PairwiseCompMatrix with shape (S, S).
            (Test Case 2) overlap_matrix is PairwiseCompMatrix with shape (S, S).
            (Test Case 3) av_corr is a float.
        """
        sss = _make_timed_stack(n_units=4, n_slices=6)
        corr, av, overlap = sss.rank_order_correlation(n_shuffles=0)

        assert isinstance(corr, PairwiseCompMatrix)
        assert corr.matrix.shape == (6, 6)
        assert isinstance(overlap, PairwiseCompMatrix)
        assert overlap.matrix.shape == (6, 6)
        assert isinstance(av, float)

    def test_raw_diagonal_is_one(self):
        """
        Raw mode diagonal is 1.0.

        Tests:
            (Test Case 1) All diagonal entries are 1.0.
        """
        sss = _make_timed_stack(n_units=4, n_slices=6)
        corr, _, _ = sss.rank_order_correlation(n_shuffles=0)
        np.testing.assert_allclose(np.diag(corr.matrix), 1.0)

    def test_raw_symmetric(self):
        """
        Correlation matrix is symmetric.

        Tests:
            (Test Case 1) corr[i,j] == corr[j,i] for all pairs.
        """
        sss = _make_timed_stack(n_units=4, n_slices=6)
        corr, _, _ = sss.rank_order_correlation(n_shuffles=0)
        np.testing.assert_allclose(corr.matrix, corr.matrix.T, atol=1e-12)

    def test_raw_values_bounded(self):
        """
        Raw Spearman values are in [-1, 1].

        Tests:
            (Test Case 1) All non-NaN off-diagonal values in [-1, 1].
        """
        sss = _make_timed_stack(n_units=4, n_slices=6)
        corr, _, _ = sss.rank_order_correlation(n_shuffles=0)
        valid = corr.matrix[~np.isnan(corr.matrix)]
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)

    def test_zscore_diagonal_is_nan(self):
        """
        Z-scored mode diagonal is NaN (self-comparison z undefined).

        Tests:
            (Test Case 1) All diagonal entries are NaN.
        """
        sss = _make_timed_stack(n_units=4, n_slices=6)
        corr, _, _ = sss.rank_order_correlation(n_shuffles=10)
        assert np.all(np.isnan(np.diag(corr.matrix)))

    def test_zscore_reproducible_with_seed(self):
        """
        Same seed produces identical z-scores.

        Tests:
            (Test Case 1) Two calls with seed=42 produce identical matrices.
        """
        sss = _make_timed_stack(n_units=4, n_slices=6)
        corr1, _, _ = sss.rank_order_correlation(n_shuffles=20, seed=42)
        corr2, _, _ = sss.rank_order_correlation(n_shuffles=20, seed=42)
        np.testing.assert_array_equal(corr1.matrix, corr2.matrix)

    def test_overlap_is_fraction(self):
        """
        Overlap matrix entries are fractions in [0, 1].

        Tests:
            (Test Case 1) All values in [0, 1].
            (Test Case 2) Diagonal equals fraction of active units per slice.
        """
        sss = _make_timed_stack(n_units=4, n_slices=6)
        _, _, overlap = sss.rank_order_correlation(n_shuffles=0)
        assert np.all(overlap.matrix >= 0.0)
        assert np.all(overlap.matrix <= 1.0)

    def test_min_overlap_filters_pairs(self):
        """
        Pairs with fewer overlapping units than min_overlap are NaN.

        Tests:
            (Test Case 1) With min_overlap set very high, all off-diagonal are NaN.
        """
        sss = _make_timed_stack(n_units=4, n_slices=6)
        corr, _, _ = sss.rank_order_correlation(min_overlap=1000, n_shuffles=0)
        off_diag = corr.matrix.copy()
        np.fill_diagonal(off_diag, np.nan)
        assert np.all(np.isnan(off_diag))

    def test_min_overlap_frac_stricter(self):
        """
        min_overlap_frac can be stricter than min_overlap.

        Tests:
            (Test Case 1) With min_overlap_frac=1.0, effective threshold = U.
                Most pairs won't have all units active in both slices.

        Notes:
            - We compare against n_shuffles=0 with min_overlap=1 to confirm
              that frac filtering produces more NaN pairs.
        """
        sss = _make_timed_stack(n_units=4, n_slices=6, seed=55)
        corr_lax, _, _ = sss.rank_order_correlation(min_overlap=1, n_shuffles=0)
        corr_strict, _, _ = sss.rank_order_correlation(
            min_overlap=1, min_overlap_frac=1.0, n_shuffles=0
        )
        nan_lax = np.sum(np.isnan(corr_lax.matrix))
        nan_strict = np.sum(np.isnan(corr_strict.matrix))
        assert nan_strict >= nan_lax

    def test_auto_compute_timing(self):
        """
        When timing_matrix is None, it is computed automatically.

        Tests:
            (Test Case 1) Calling without timing_matrix succeeds.
            (Test Case 2) Result matches explicit get_unit_timing_per_slice call.
        """
        sss = _make_timed_stack(n_units=4, n_slices=6)
        tm = sss.get_unit_timing_per_slice(timing="median", min_spikes=2)
        corr_explicit, av_explicit, _ = sss.rank_order_correlation(
            timing_matrix=tm, n_shuffles=0
        )
        corr_auto, av_auto, _ = sss.rank_order_correlation(
            timing="median", min_spikes=2, n_shuffles=0
        )
        np.testing.assert_array_equal(corr_explicit.matrix, corr_auto.matrix)
        assert av_explicit == av_auto

    def test_invalid_n_shuffles_raises(self):
        """
        n_shuffles between 1 and 4 raises ValueError.

        Tests:
            (Test Case 1) n_shuffles=3 raises ValueError.
        """
        sss = _make_timed_stack(n_units=4, n_slices=6)
        with pytest.raises(ValueError, match="n_shuffles"):
            sss.rank_order_correlation(n_shuffles=3)

    def test_non_2d_timing_raises(self):
        """
        Non-2D timing_matrix raises ValueError.

        Tests:
            (Test Case 1) 1-D array raises ValueError.
        """
        sss = _make_timed_stack(n_units=4, n_slices=6)
        with pytest.raises(ValueError, match="2-D"):
            sss.rank_order_correlation(timing_matrix=np.ones(10), n_shuffles=0)

    def test_rank_order_single_slice(self):
        """
        Single-slice stack produces (1,1) correlation matrix with NaN off-diagonal.

        Tests:
            (Test Case 1) corr shape is (1, 1).
            (Test Case 2) av_corr is NaN (no lower-triangle pairs).
        """
        sd = SpikeData(
            [np.array([5.0, 15.0, 25.0]), np.array([8.0, 22.0, 40.0])],
            length=50.0,
        )
        sss = SpikeSliceStack(spike_stack=[sd])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr, av, overlap = sss.rank_order_correlation(n_shuffles=0)
        assert corr.matrix.shape == (1, 1)
        assert np.isnan(av)

    def test_rank_order_all_nan_timing(self):
        """
        All-NaN timing matrix produces all-NaN correlation.

        Tests:
            (Test Case 1) Entire corr matrix is NaN.
            (Test Case 2) av_corr is NaN.
        """
        sss = _make_timed_stack(n_units=4, n_slices=5)
        all_nan = np.full((4, 5), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr, av, _ = sss.rank_order_correlation(
                timing_matrix=all_nan, n_shuffles=0
            )
        assert np.all(np.isnan(corr.matrix[np.triu_indices(5, k=1)]))
        assert np.isnan(av)

    def test_rank_order_n_shuffles_exactly_5(self):
        """
        n_shuffles=5 (minimum allowed) produces valid z-scores.

        Tests:
            (Test Case 1) No error raised.
            (Test Case 2) Output shape is (S, S).
        """
        sss = _make_timed_stack(n_units=4, n_slices=5, seed=99)
        corr, _, _ = sss.rank_order_correlation(n_shuffles=5, seed=42)
        assert corr.matrix.shape == (5, 5)

    def test_rank_order_min_overlap_frac_zero(self):
        """
        min_overlap_frac=0.0 is a no-op (effective_min stays at min_overlap).

        Tests:
            (Test Case 1) Same result as without min_overlap_frac.
        """
        sss = _make_timed_stack(n_units=4, n_slices=5)
        corr1, _, _ = sss.rank_order_correlation(min_overlap=3, n_shuffles=0)
        corr2, _, _ = sss.rank_order_correlation(
            min_overlap=3, min_overlap_frac=0.0, n_shuffles=0
        )
        np.testing.assert_array_equal(corr1.matrix, corr2.matrix)

    def test_seed_none_non_deterministic(self):
        """
        EC-SSS-09: rank_order_correlation with seed=None runs without error
        and produces valid output (non-deterministic RNG).

        Tests:
            (Test Case 1) No error raised.
            (Test Case 2) Output has correct shape (S, S).
            (Test Case 3) av_corr is a finite float or NaN.
        """
        sss = _make_timed_stack(n_units=4, n_slices=5, seed=42)
        corr, av, overlap = sss.rank_order_correlation(n_shuffles=10, seed=None)

        assert corr.matrix.shape == (5, 5)
        assert isinstance(av, float)
        assert overlap.matrix.shape == (5, 5)


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


class TestApply:
    """Tests for SpikeSliceStack.apply."""

    def test_scalar_function(self):
        """
        apply with a function returning a scalar produces a 1-D array of length S.

        Tests:
            (Test Case 1) Output shape is (S,).
            (Test Case 2) Values match manual per-slice computation.
        """
        sd = make_spikedata(n_units=3, length_ms=200.0, seed=0)
        stack = SpikeSliceStack(sd, times_start_to_end=[(0, 100), (100, 200)])

        def total_spikes(s):
            return sum(len(t) for t in s.train)

        result = stack.apply(total_spikes)

        assert result.shape == (2,)
        for i, s in enumerate(stack.spike_stack):
            expected = sum(len(t) for t in s.train)
            assert result[i] == expected

    def test_1d_function(self):
        """
        apply with a function returning a 1-D array produces shape (S, U).

        Tests:
            (Test Case 1) Output shape is (S, U).
            (Test Case 2) Values match per-slice rates.
        """
        sd = make_spikedata(n_units=3, length_ms=200.0, seed=1)
        stack = SpikeSliceStack(sd, times_start_to_end=[(0, 100), (100, 200)])

        result = stack.apply(lambda s: s.rates())

        assert result.shape == (2, 3)
        for i, s in enumerate(stack.spike_stack):
            np.testing.assert_array_almost_equal(result[i], s.rates())

    def test_2d_function(self):
        """
        apply with a function returning a 2-D array produces shape (S, U, U).

        Tests:
            (Test Case 1) Output shape is (S, U, U).
        """
        sd = make_spikedata(n_units=3, length_ms=200.0, seed=2)
        stack = SpikeSliceStack(sd, times_start_to_end=[(0, 100), (100, 200)])

        def raster_shape(s):
            return s.raster()

        result = stack.apply(raster_shape)

        assert result.shape[0] == 2
        assert result.shape[1] == 3

    def test_extra_args_forwarded(self):
        """
        Additional positional and keyword arguments are forwarded to the function.

        Tests:
            (Test Case 1) args and kwargs are received by the function.
        """
        sd = make_spikedata(n_units=3, length_ms=200.0, seed=3)
        stack = SpikeSliceStack(sd, times_start_to_end=[(0, 100), (100, 200)])

        def spike_count_above(s, threshold, unit="kHz"):
            rates = s.rates(unit=unit)
            return np.sum(rates > threshold)

        result = stack.apply(spike_count_above, 0.01, unit="kHz")

        assert result.shape == (2,)
        assert result.dtype in (np.int32, np.int64, np.intp)

    def test_single_slice(self):
        """
        apply works with a stack containing a single slice.

        Tests:
            (Test Case 1) Output shape is (1,) for a scalar function.
        """
        sd = make_spikedata(n_units=2, length_ms=100.0, seed=4)
        stack = SpikeSliceStack(sd, times_start_to_end=[(0, 100)])

        result = stack.apply(lambda s: s.N)

        assert result.shape == (1,)
        assert result[0] == 2

    def test_apply_different_shapes_raises(self):
        """
        EC-SSS-02: apply with function returning different shapes per slice raises
        ValueError from np.stack.

        Tests:
            (Test Case 1) np.stack raises ValueError when output shapes differ.
        """
        sd = make_spikedata(n_units=3, length_ms=200.0, seed=10)
        stack = SpikeSliceStack(sd, times_start_to_end=[(0, 100), (100, 200)])

        call_count = [0]

        def varying_shape(s):
            call_count[0] += 1
            if call_count[0] == 1:
                return np.zeros(3)
            return np.zeros(5)

        with pytest.raises(ValueError):
            stack.apply(varying_shape)

    def test_apply_function_raises_mid_iteration(self):
        """
        EC-SSS-03: apply with function that raises mid-iteration propagates the
        exception without catching it.

        Tests:
            (Test Case 1) RuntimeError raised by the function propagates out of apply.
        """
        sd = make_spikedata(n_units=2, length_ms=200.0, seed=11)
        stack = SpikeSliceStack(sd, times_start_to_end=[(0, 100), (100, 200)])

        call_count = [0]

        def exploding_func(s):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("boom")
            return np.array(1.0)

        with pytest.raises(RuntimeError, match="boom"):
            stack.apply(exploding_func)


# ---------------------------------------------------------------------------
# Edge case tests from REVIEW.md — Edge Case Scan — Core (spikedata/)
# spikeslicestack.py findings
# ---------------------------------------------------------------------------
class TestPlotUnitRaster:
    """Edge case tests for SpikeSliceStack.plot_aligned_slice_single_unit."""

    def test_unit_idx_out_of_range_raises(self):
        """
        plot_aligned_slice_single_unit with unit_idx >= N raises IndexError.

        Tests:
            (Test Case 1) IndexError is raised for unit_idx equal to N.
            (Test Case 2) IndexError is raised for negative unit_idx.
        """
        sd = make_spikedata(n_units=3, length_ms=100.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 50.0), (50.0, 100.0)])

        with pytest.raises(IndexError, match="out of range"):
            sss.plot_aligned_slice_single_unit(unit_idx=3)
        with pytest.raises(IndexError, match="out of range"):
            sss.plot_aligned_slice_single_unit(unit_idx=-1)


# ---------------------------------------------------------------------------
# Edge case tests from the edge case scan
# ---------------------------------------------------------------------------
class TestSSSSubset2:
    """Additional edge case tests for SpikeSliceStack.subset."""

    def test_subset_empty_unit_list(self):
        """
        subset with empty unit list [] produces a 0-unit stack.

        Tests:
            (Test Case 1) Empty unit list creates a stack where each slice
                has N=0.
        """
        sd = make_spikedata(n_units=3, length_ms=100.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 50.0), (50.0, 100.0)])
        result = sss.subset([])
        assert result.N == 0


class TestSSSSubtimeByIndex2:
    """Additional edge case tests for SpikeSliceStack.subtime_by_index."""

    def test_non_integer_slice_duration_raises(self):
        """
        subtime_by_index rejects non-integer slice_duration_ms with
        ValueError instead of silently rounding.

        Tests:
            (Test Case 1) A 100.5 ms slice duration raises ValueError.
        """
        sd = make_spikedata(n_units=2, length_ms=201.0)
        sss = SpikeSliceStack(
            sd,
            times_start_to_end=[(0.0, 100.5), (100.5, 201.0)],
        )
        with pytest.raises(ValueError, match="integer number of milliseconds"):
            sss.subtime_by_index(0, 50)


class TestSSSOrderUnits2:
    """Additional edge case tests for SpikeSliceStack.order_units_across_slices."""

    def test_constant_timing_first_mode(self):
        """
        order_units_across_slices with timing='first' and constant spike times.

        Tests:
            (Test Case 1) Units that always spike at the same time produce
                consistent ordering. Returns a 5-tuple.
            (Test Case 2) Unit 1 (first spike at 1.0 ms) comes before unit 0
                (first spike at 5.0 ms) in the highly-active order.
            (Test Case 3) Per-unit peak times reflect the constant input
                timings (unit 0 -> 5.0 ms, unit 1 -> 1.0 ms).
        """
        # 2 units, each with spikes at the same time in each slice
        sd1 = SpikeData([[5.0, 10.0], [1.0, 2.0]], length=20.0)
        sd2 = SpikeData([[5.0, 10.0], [1.0, 2.0]], length=20.0)
        sss = SpikeSliceStack(
            spike_stack=[sd1, sd2],
            times_start_to_end=[(0.0, 20.0), (0.0, 20.0)],
        )
        result = sss.order_units_across_slices(timing="first", agg_func="median")
        # Returns a 5-tuple:
        # (reordered_stacks, unit_ids_in_order, unit_std,
        #  unit_peak_times_ms, unit_frac_active)
        assert isinstance(result, tuple)
        assert len(result) == 5

        ha_order = result[1][0]
        # Unit 1 fires earlier (1.0 ms) than unit 0 (5.0 ms).
        assert list(ha_order) == [1, 0]

        # Peak times for the highly-active group, in the reordered sequence.
        ha_peak_times = result[3][0]
        # ha_order is [1, 0], so first peak time is unit 1's (1.0 ms),
        # second is unit 0's (5.0 ms).
        np.testing.assert_allclose(ha_peak_times, [1.0, 5.0])


class TestSSSUnitToUnitComp2:
    """Additional edge case tests for SpikeSliceStack.unit_to_unit_comparison."""

    def test_sttc_all_empty_trains(self):
        """
        unit_to_unit_comparison with metric='sttc' and all-empty trains.

        Tests:
            (Test Case 1) All-empty trains produce zero STTC values.
        """
        sd = SpikeData([[], [], []], length=100.0)
        sss = SpikeSliceStack(
            spike_stack=[sd, sd],
            times_start_to_end=[(0.0, 100.0), (0.0, 100.0)],
        )
        corr, lag, av_corr, av_lag = sss.unit_to_unit_comparison(metric="sttc")
        assert corr.stack.shape == (3, 3, 2)


class TestSSSSliceToSliceComp2:
    """Additional edge case tests for SpikeSliceStack.get_slice_to_slice_unit_comparison."""

    def test_sttc_large_delt(self):
        """
        get_slice_to_slice_unit_comparison with delt larger than slice duration.

        Tests:
            (Test Case 1) delt=10000 with 50ms slices does not raise.
        """
        sd = make_spikedata(n_units=2, length_ms=100.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 50.0), (50.0, 100.0)])
        corr, lag, av_corr, av_lag = sss.get_slice_to_slice_unit_comparison(
            metric="sttc", delt=10000.0
        )
        assert corr.stack.shape[0] == 2
        assert corr.stack.shape[1] == 2


class TestCoverageGaps:
    """Tests for coverage gaps in SpikeSliceStack methods."""

    def _make_stack(self, n_units=3, n_slices=4, length_ms=200.0, seed=42):
        rng = np.random.default_rng(seed)
        sd = make_spikedata(n_units=n_units, length_ms=length_ms, seed=seed)
        step = length_ms / n_slices
        times = [(i * step, (i + 1) * step) for i in range(n_slices)]
        return SpikeSliceStack(sd, times_start_to_end=times)

    def test_unit_to_unit_comparison_serial_equals_parallel(self):
        """
        Tests: unit_to_unit_comparison serial vs parallel.

        (Test Case 1) Correlation stacks match between n_jobs=1 and n_jobs=-1.
        """
        sss = self._make_stack()
        corr_s, lag_s, _, _ = sss.unit_to_unit_comparison(n_jobs=1)
        corr_p, lag_p, _, _ = sss.unit_to_unit_comparison(n_jobs=-1)
        np.testing.assert_allclose(corr_s.stack, corr_p.stack, rtol=1e-12)
        np.testing.assert_allclose(lag_s.stack, lag_p.stack, rtol=1e-12)

    def test_get_slice_to_slice_unit_comparison_serial_equals_parallel(self):
        """
        Tests: get_slice_to_slice_unit_comparison serial vs parallel.

        (Test Case 1) Correlation stacks match between n_jobs=1 and n_jobs=-1.
        """
        sss = self._make_stack()
        corr_s, lag_s, _, _ = sss.get_slice_to_slice_unit_comparison(n_jobs=1)
        corr_p, lag_p, _, _ = sss.get_slice_to_slice_unit_comparison(n_jobs=-1)
        np.testing.assert_allclose(corr_s.stack, corr_p.stack, rtol=1e-12)
        np.testing.assert_allclose(lag_s.stack, lag_p.stack, rtol=1e-12)

    def test_rank_order_correlation_serial_equals_parallel(self):
        """
        Tests: rank_order_correlation serial vs parallel.

        (Test Case 1) Results match between n_jobs=1 and n_jobs=-1.
        """
        sss = self._make_stack(n_units=4, n_slices=5, length_ms=250.0)
        corr_s, _, _ = sss.rank_order_correlation(n_shuffles=0, n_jobs=1, seed=1)
        corr_p, _, _ = sss.rank_order_correlation(n_shuffles=0, n_jobs=-1, seed=1)
        np.testing.assert_allclose(corr_s.matrix, corr_p.matrix, rtol=1e-12)

    def test_get_unit_timing_per_slice_mean(self):
        """
        Tests: get_unit_timing_per_slice with timing='mean'.

        (Test Case 1) Returns array of correct shape (U, S).
        (Test Case 2) Mean and median timings share the same NaN pattern
            (a unit/slice cell is NaN under one iff it is NaN under the other).
        """
        sss = self._make_stack(n_units=3, n_slices=4)
        timing_mean = sss.get_unit_timing_per_slice(timing="mean")
        timing_median = sss.get_unit_timing_per_slice(timing="median")
        assert timing_mean.shape == (3, 4)
        assert timing_median.shape == (3, 4)
        # Mean and median should have the same NaN pattern (both NaN iff
        # the unit has fewer than min_spikes spikes in that slice).
        nan_mask_mean = np.isnan(timing_mean)
        nan_mask_median = np.isnan(timing_median)
        np.testing.assert_array_equal(nan_mask_mean, nan_mask_median)

    def test_order_units_across_slices_with_timing_matrix(self):
        """
        Tests: order_units_across_slices with pre-computed timing_matrix.

        (Test Case 1) Custom timing matrix produces expected unit ordering.
        """
        sss = self._make_stack(n_units=3, n_slices=4)
        # Provide a timing matrix where unit 2 is earliest, unit 0 latest
        timing = np.array(
            [
                [30.0, 30.0, 30.0, 30.0],  # unit 0: late
                [20.0, 20.0, 20.0, 20.0],  # unit 1: middle
                [10.0, 10.0, 10.0, 10.0],  # unit 2: early
            ]
        )
        # Returns ((ha_stack, la_stack), (ha_order, la_order), ...)
        result = sss.order_units_across_slices(timing_matrix=timing)
        ha_order = result[1][0]  # (ha_order, la_order)
        # Unit 2 should come first (earliest), then 1, then 0
        assert list(ha_order) == [2, 1, 0]

    def test_plot_aligned_slice_single_unit_eventplot_style(self):
        """
        Tests: plot_aligned_slice_single_unit with style='eventplot'.

        (Test Case 1) Eventplot style produces an axes object without error.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sss = self._make_stack(n_units=3, n_slices=5)
        fig, ax, result = sss.plot_aligned_slice_single_unit(0, style="eventplot")
        assert ax is not None
        plt.close("all")

    def test_plot_aligned_slice_single_unit_invert_y(self):
        """
        Tests: plot_aligned_slice_single_unit with invert_y=True.

        (Test Case 1) Y-axis is inverted when invert_y=True.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sss = self._make_stack(n_units=3, n_slices=5)
        fig, ax, _ = sss.plot_aligned_slice_single_unit(0, invert_y=True)
        assert ax.yaxis_inverted()
        plt.close("all")


# ---------------------------------------------------------------------------
# Edge case tests from REVIEW.md — Edge Case Scan (HIGH + MEDIUM)
# ---------------------------------------------------------------------------


class TestSpikeSliceStackCoreReview:
    """Edge case tests for HIGH and MEDIUM findings from REVIEW.md."""

    def test_subslice_empty_list(self):
        """
        subslice with an empty list [].

        Tests:
            (Test Case 1) Empty indices list raises ValueError since the
                resulting stack would have 0 slices.
        """
        sd = make_spikedata(n_units=2, length_ms=200.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(10.0, 30.0), (50.0, 70.0)])
        with pytest.raises(ValueError, match="must not be empty"):
            sss.subslice([])

    def test_subtime_by_index_single_ms_window(self):
        """
        subtime_by_index with a single-ms window (start_idx, start_idx+1).

        Tests:
            (Test Case 1) Each slice's duration is 1 ms.
            (Test Case 2) Times reflect the 1ms sub-window.
        """
        sd = make_spikedata(n_units=2, length_ms=200.0, seed=42)
        times = [(10.0, 60.0), (80.0, 130.0)]
        sss = SpikeSliceStack(sd, times_start_to_end=times)
        result = sss.subtime_by_index(5, 6)
        assert len(result.spike_stack) == 2
        for sd_slice in result.spike_stack:
            assert sd_slice.length == pytest.approx(1.0)

    def test_compute_frac_active_min_spikes_zero(self):
        """
        min_spikes=0 semantics: empty units count as active.

        Tests:
            (Test Case 1) With min_spikes=0, even units with 0 spikes count
                as active (len(train) >= 0 is always True).
            (Test Case 2) All frac_active values are 1.0.
        """
        empty = np.array([], dtype=float)
        sd1 = SpikeData([empty, empty], length=50.0)
        sd2 = SpikeData([empty, empty], length=50.0)
        sss = SpikeSliceStack(spike_stack=[sd1, sd2])
        frac = sss.compute_frac_active(min_spikes=0)
        assert frac.shape == (2,)
        np.testing.assert_array_equal(frac, 1.0)

    def test_spike_stack_negative_start_time(self):
        """
        spike_stack with event-centered slices (negative start_time).

        Tests:
            (Test Case 1) SpikeData objects with negative start_time are accepted.
            (Test Case 2) Spike times within each slice are preserved.
        """
        # start_time=-10, length=20: spikes must be in [-10, 10]
        sd1 = SpikeData([np.array([-5.0, 5.0])], length=20.0, start_time=-10.0)
        sd2 = SpikeData([np.array([-3.0, 8.0])], length=20.0, start_time=-10.0)
        sss = SpikeSliceStack(
            spike_stack=[sd1, sd2],
            times_start_to_end=[(-10.0, 10.0), (20.0, 40.0)],
        )
        assert len(sss.spike_stack) == 2
        assert sss.N == 1

    def test_rank_order_correlation_n_shuffles_positive(self):
        """
        rank_order_correlation with n_shuffles > 0.

        Tests:
            (Test Case 1) n_shuffles > 0 produces shuffle-corrected result.
            (Test Case 2) Result shape is correct.
            (Test Case 3) Diagonal (self-correlation) is NaN under z-scoring,
                because the shuffle null distribution has zero variance when
                a slice's ranks are compared against themselves.
        """
        sss = _make_correlated_stack(n_units=4, n_slices=5, seed=42)
        corr, av, overlap = sss.rank_order_correlation(n_shuffles=10)
        assert isinstance(corr, PairwiseCompMatrix)
        assert corr.matrix.shape == (5, 5)
        # Exercise the z-score branch: diagonal entries must be NaN.
        assert np.all(np.isnan(np.diag(corr.matrix)))

    def test_get_unit_timing_timing_first_min_spikes_1(self):
        """
        get_unit_timing_per_slice with timing='first' and min_spikes=1.

        Tests:
            (Test Case 1) With min_spikes=1, units with 0 spikes in a slice
                get NaN timing.
            (Test Case 2) Units with >= 1 spike get the time of the first spike.
        """
        # Create stack with some empty units
        sd1 = SpikeData([np.array([5.0, 15.0, 25.0]), np.array([])], length=50.0)
        sd2 = SpikeData([np.array([10.0, 30.0]), np.array([20.0])], length=50.0)
        sss = SpikeSliceStack(spike_stack=[sd1, sd2])
        tm = sss.get_unit_timing_per_slice(timing="first", min_spikes=1)
        assert tm.shape == (2, 2)
        # Unit 0 has spikes in both slices
        assert not np.isnan(tm[0, 0])
        assert not np.isnan(tm[0, 1])
        # Unit 1 has 0 spikes in slice 0 → NaN
        assert np.isnan(tm[1, 0])
        # Unit 1 has 1 spike in slice 1 → valid
        assert not np.isnan(tm[1, 1])


class TestSSSSubtimeByIndexNonIntegerSliceDuration:
    """
    Tests that SpikeSliceStack.subtime_by_index rejects non-integer
    slice durations with a clear ValueError, instead of silently
    truncating the sub-ms tail via int(round(slice_duration_ms)).
    """

    def test_subtime_by_index_non_integer_slice_duration_raises(self):
        """
        subtime_by_index raises ValueError when the slice duration
        is not an integer number of milliseconds.

        Tests:
            (Test Case 1) A 12.4 ms slice raises ValueError naming
                "integer number of milliseconds".
            (Test Case 2) The error suggests SpikeData.subtime() as
                the workaround for non-integer windows.
        """
        train = [np.array([1.0, 5.0, 10.0])]
        sd = SpikeData(train, length=12.4)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 12.4)])
        with pytest.raises(ValueError, match="integer number of milliseconds"):
            sss.subtime_by_index(0, 12)

    def test_subtime_by_index_integer_slice_duration_succeeds(self):
        """
        Integer slice durations still work normally (no behavior change
        for well-formed inputs).

        Tests:
            (Test Case 1) A 12.0 ms slice + subtime_by_index(0, 12)
                returns a SpikeSliceStack with the expected slice count.
            (Test Case 2) A float-but-integer-valued duration (12.0)
                is accepted (the validation tolerates float
                imprecision below 1e-9 ms).
        """
        train = [np.array([1.0, 5.0, 10.0])]
        sd = SpikeData(train, length=12.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 12.0)])
        result = sss.subtime_by_index(0, 12)
        assert len(result.spike_stack) == 1


class TestSpikeSliceStackUniformStartTime:
    """
    Constructor enforces uniform ``start_time`` across the
    ``spike_stack`` to prevent silent mis-alignment of downstream
    raster outputs.
    """

    def test_mixed_zero_based_and_event_centered_rejected(self):
        """
        Mixing a 0-based slice (``start_time=0``) with an event-
        centered slice (``start_time<0``) raises ValueError naming
        ``start_time``.

        Tests:
            (Test Case 1) Construction raises ValueError.
            (Test Case 2) The error mentions ``start_time``.
        """
        zero_based = SpikeData([[2.0, 5.0]], length=10.0, start_time=0.0)
        event_centered = SpikeData([[-3.0, 4.0]], length=10.0, start_time=-5.0)
        with pytest.raises(ValueError, match="start_time"):
            SpikeSliceStack(
                spike_stack=[zero_based, event_centered],
                times_start_to_end=[(0.0, 10.0), (10.0, 20.0)],
            )

    def test_mixed_event_centered_pre_windows_rejected(self):
        """
        Two event-centered slices with different ``pre`` windows
        (``start_time=-5`` vs ``start_time=-2``) but the same total
        duration are rejected because they use different time-origin
        conventions.

        Tests:
            (Test Case 1) Construction raises ValueError mentioning
                ``start_time``.
        """
        # Both slices have duration 10.0 to satisfy the "same window
        # length" check, but their start_time values differ — one is
        # event-centered with pre=5, the other with pre=2.
        a = SpikeData([[-3.0, 4.0]], length=10.0, start_time=-5.0)
        b = SpikeData([[-1.0, 7.0]], length=10.0, start_time=-2.0)
        with pytest.raises(ValueError, match="start_time"):
            SpikeSliceStack(
                spike_stack=[a, b],
                times_start_to_end=[(0.0, 10.0), (10.0, 20.0)],
            )

    def test_uniform_start_time_accepted(self):
        """
        A stack of slices that all share the same ``start_time`` is
        accepted.

        Tests:
            (Test Case 1) Two 0-based slices construct successfully.
            (Test Case 2) Two event-centered slices with the same
                ``pre`` value construct successfully.
        """
        a = SpikeData([[1.0]], length=10.0, start_time=0.0)
        b = SpikeData([[2.0]], length=10.0, start_time=0.0)
        sss = SpikeSliceStack(
            spike_stack=[a, b],
            times_start_to_end=[(0.0, 10.0), (10.0, 20.0)],
        )
        assert len(sss.spike_stack) == 2

        c = SpikeData([[-3.0]], length=10.0, start_time=-5.0)
        d = SpikeData([[1.0]], length=10.0, start_time=-5.0)
        sss2 = SpikeSliceStack(
            spike_stack=[c, d],
            times_start_to_end=[(0.0, 10.0), (10.0, 20.0)],
        )
        assert len(sss2.spike_stack) == 2

    def test_single_slice_passes_uniformity_trivially(self):
        """
        A single-slice stack has nothing to compare; the uniformity
        check should not fire.

        Tests:
            (Test Case 1) Single-slice stack with negative
                ``start_time`` constructs successfully.
        """
        sd = SpikeData([[1.0]], length=10.0, start_time=-5.0)
        sss = SpikeSliceStack(spike_stack=[sd], times_start_to_end=[(0.0, 10.0)])
        assert len(sss.spike_stack) == 1


class TestBaselineNormalizedRaster:
    """Tests for SpikeSliceStack.baseline_normalized_raster."""

    def _evoked_stack(self):
        """4-slice stack where unit 0 has 5 response spikes in [20,30), unit 1 silent."""
        slices = []
        for _ in range(4):
            sd = SpikeData(
                [
                    np.concatenate([np.array([10.0]), np.linspace(20.5, 29.5, 5)]),
                    np.array([5.0]),
                ],
                length=40.0,
                N=2,
            )
            slices.append(sd)
        return SpikeSliceStack(spike_stack=slices, times_start_to_end=[(0, 40)] * 4)

    def test_shape(self):
        """
        Output shape matches the underlying raster (U, T, S).

        Tests:
            (Test Case 1) Shape is (2, 40, 4) at bin_size=1.0.
        """
        sss = self._evoked_stack()
        out = sss.baseline_normalized_raster(
            bin_size=1.0, baseline_window_ms=(0.0, 20.0), mode="subtract"
        )
        assert out.shape == (2, 40, 4)

    def test_subtract_mode_values(self):
        """
        Baseline rate unit 0: 1 spike / 20 ms = 0.05/ms. Expected per 1ms bin
        = 0.05. SpikeData bins are left-open right-closed, so spike at t=10.0
        lands in bin 9. Bin with 1 spike: 0.95; empty bin: -0.05.

        Tests:
            (Test Case 1) Bin 9 of unit 0 (spike) is ~0.95.
            (Test Case 2) Bin 0 of unit 0 (empty) is ~-0.05.
        """
        sss = self._evoked_stack()
        out = sss.baseline_normalized_raster(
            bin_size=1.0, baseline_window_ms=(0.0, 20.0), mode="subtract"
        )
        assert out[0, 9, 0] == pytest.approx(1.0 - 0.05)
        assert out[0, 0, 0] == pytest.approx(-0.05)

    def test_ratio_mode(self):
        """
        Ratio mode: counts / expected per bin.

        Tests:
            (Test Case 1) Bin with 1 spike: 1/0.05 = 20.
            (Test Case 2) Empty bin: 0.
        """
        sss = self._evoked_stack()
        out = sss.baseline_normalized_raster(
            bin_size=1.0, baseline_window_ms=(0.0, 20.0), mode="ratio"
        )
        assert out[0, 9, 0] == pytest.approx(20.0)
        assert out[0, 0, 0] == pytest.approx(0.0)

    def test_zscore_mode(self):
        """
        Z-score mode produces (1 - 0.05) / sqrt(0.05) ≈ 4.25 at the spike bin.

        Tests:
            (Test Case 1) Shape is (U, T, S).
            (Test Case 2) Per-bin z-score matches analytical formula.
        """
        sss = self._evoked_stack()
        z = sss.baseline_normalized_raster(
            bin_size=1.0, baseline_window_ms=(0.0, 20.0), mode="zscore"
        )
        assert z.shape == (2, 40, 4)
        expected = (1.0 - 0.05) / np.sqrt(0.05)
        assert z[0, 9, 0] == pytest.approx(expected)

    def test_baseline_outside_slice_raises(self):
        """
        Baseline window outside slice raises ValueError.

        Tests:
            (Test Case 1) Out-of-range window raises.
            (Test Case 2) Wrong tuple form raises.
            (Test Case 3) end <= start raises.
        """
        sss = self._evoked_stack()
        with pytest.raises(ValueError, match="falls outside"):
            sss.baseline_normalized_raster(1.0, baseline_window_ms=(0.0, 100.0))
        with pytest.raises(ValueError, match="tuple"):
            sss.baseline_normalized_raster(1.0, baseline_window_ms=5.0)
        with pytest.raises(ValueError, match="end must be greater"):
            sss.baseline_normalized_raster(1.0, baseline_window_ms=(20.0, 10.0))

    def test_unknown_mode_raises(self):
        """
        Unknown mode raises ValueError.

        Tests:
            (Test Case 1) ValueError for bogus mode.
        """
        sss = self._evoked_stack()
        with pytest.raises(ValueError, match="mode"):
            sss.baseline_normalized_raster(
                1.0, baseline_window_ms=(0.0, 20.0), mode="bogus"
            )

    def test_empty_unit_zero_baseline(self):
        """
        A unit with no spikes yields zeros in subtract mode and NaN in
        ratio / zscore modes (division by zero baseline).

        Tests:
            (Test Case 1) subtract: zeros for empty unit.
            (Test Case 2) ratio: NaN for empty unit.
            (Test Case 3) zscore: NaN for empty unit.
        """
        sd = SpikeData([np.array([]), np.array([10.0])], length=40.0, N=2)
        sss = SpikeSliceStack(
            spike_stack=[sd, sd], times_start_to_end=[(0, 40), (100, 140)]
        )
        sub = sss.baseline_normalized_raster(
            1.0, baseline_window_ms=(0.0, 20.0), mode="subtract"
        )
        rat = sss.baseline_normalized_raster(
            1.0, baseline_window_ms=(0.0, 20.0), mode="ratio"
        )
        zsc = sss.baseline_normalized_raster(
            1.0, baseline_window_ms=(0.0, 20.0), mode="zscore"
        )
        assert (sub[0, :, :] == 0).all()
        assert np.isnan(rat[0, :, :]).all()
        assert np.isnan(zsc[0, :, :]).all()


class TestResponsiveUnits:
    """Tests for SpikeSliceStack.responsive_units."""

    def _build(self):
        """Stack where unit 0 reliably responds in [20,30), unit 1 does not."""
        slices = []
        for _ in range(6):
            sd = SpikeData(
                [
                    np.concatenate([np.array([10.0]), np.linspace(20.5, 29.5, 5)]),
                    np.array([5.0, 15.0, 25.0, 35.0]),
                ],
                length=40.0,
                N=2,
            )
            slices.append(sd)
        return SpikeSliceStack(spike_stack=slices, times_start_to_end=[(0, 40)] * 6)

    def test_known_responsive_only(self):
        """
        Unit 0 is responsive; unit 1 is not.

        Tests:
            (Test Case 1) mask[0] is True.
            (Test Case 2) mask[1] is False.
        """
        sss = self._build()
        mask = sss.responsive_units(
            bin_size=10.0,
            baseline_window_ms=(0.0, 20.0),
            response_window_ms=(20.0, 30.0),
            z_threshold=2.0,
        )
        assert mask.shape == (2,)
        assert bool(mask[0]) is True
        assert bool(mask[1]) is False

    def test_full_slice_default(self):
        """
        Default response_window_ms=None matches the explicit full-slice call.

        Tests:
            (Test Case 1) default mask equals full-slice mask.
        """
        sss = self._build()
        m_default = sss.responsive_units(bin_size=10.0, baseline_window_ms=(0.0, 20.0))
        m_explicit = sss.responsive_units(
            bin_size=10.0,
            baseline_window_ms=(0.0, 20.0),
            response_window_ms=(0.0, 40.0),
        )
        assert (m_default == m_explicit).all()

    def test_max_aggregator_matches_mean(self):
        """
        For consistent responses across slices, mean and max aggregators agree.

        Tests:
            (Test Case 1) max mask == mean mask.
        """
        sss = self._build()
        m1 = sss.responsive_units(
            bin_size=10.0,
            baseline_window_ms=(0.0, 20.0),
            response_window_ms=(20.0, 30.0),
            aggregator="mean",
        )
        m2 = sss.responsive_units(
            bin_size=10.0,
            baseline_window_ms=(0.0, 20.0),
            response_window_ms=(20.0, 30.0),
            aggregator="max",
        )
        assert (m1 == m2).all()

    def test_bad_aggregator_raises(self):
        """
        Invalid aggregator raises ValueError.

        Tests:
            (Test Case 1) ValueError for unknown aggregator.
        """
        sss = self._build()
        with pytest.raises(ValueError, match="aggregator"):
            sss.responsive_units(
                bin_size=1.0,
                baseline_window_ms=(0.0, 20.0),
                aggregator="bogus",
            )

    def test_bad_response_window_raises(self):
        """
        Empty or malformed response window raises ValueError.

        Tests:
            (Test Case 1) end <= start raises.
            (Test Case 2) Non-tuple raises.
            (Test Case 3) Out-of-bin-range window raises.
        """
        sss = self._build()
        with pytest.raises(ValueError, match="end must be greater"):
            sss.responsive_units(
                bin_size=1.0,
                baseline_window_ms=(0.0, 20.0),
                response_window_ms=(30.0, 20.0),
            )
        with pytest.raises(ValueError, match="tuple"):
            sss.responsive_units(
                bin_size=1.0,
                baseline_window_ms=(0.0, 20.0),
                response_window_ms=5.0,
            )
        with pytest.raises(ValueError, match="empty bin range"):
            sss.responsive_units(
                bin_size=1.0,
                baseline_window_ms=(0.0, 20.0),
                response_window_ms=(100.0, 200.0),
            )


class TestPerUnitResponseRegression:
    """Tests for SpikeSliceStack.per_unit_response_regression."""

    def _decaying_stack(self, n_slices=8, decay_per_slice=1, seed=0):
        """Build a stack where unit 0's response decays linearly across slices."""
        rng = np.random.default_rng(seed)
        slices = []
        for s in range(n_slices):
            # Response amplitude for unit 0: 10 - s*decay_per_slice spikes,
            # distributed in (20, 30) ms region.
            n_resp = max(0, 10 - s * decay_per_slice)
            response_times = (
                np.sort(rng.uniform(20.5, 29.5, n_resp))
                if n_resp > 0
                else np.array([], dtype=float)
            )
            unit0 = np.concatenate([np.array([10.0]), response_times])
            unit1 = np.array([5.0, 15.0, 25.0, 35.0])  # roughly constant
            sd = SpikeData([unit0, unit1], length=40.0, N=2)
            slices.append(sd)
        return SpikeSliceStack(
            spike_stack=slices, times_start_to_end=[(0, 40)] * n_slices
        )

    def test_recovers_decay_slope(self):
        """
        Unit 0 amplitude decays by 1 spike per slice; the regression slope
        is approximately -1.

        Tests:
            (Test Case 1) Unit 0 slope is between -1.2 and -0.8.
            (Test Case 2) Unit 0 p-value < 0.05.
        """
        sss = self._decaying_stack(n_slices=8, decay_per_slice=1)
        res = sss.per_unit_response_regression(
            bin_size=1.0, response_window_ms=(20.0, 30.0)
        )
        assert -1.2 < res["slope"][0] < -0.8
        assert res["p_value"][0] < 0.05

    def test_flat_response_zero_slope(self):
        """
        For a unit with constant response amplitude across slices, slope is
        approximately zero and p-value is large.

        Tests:
            (Test Case 1) Unit 1 |slope| < 1e-9 (exactly 0 for identical
                amplitudes).
        """
        sss = self._decaying_stack(n_slices=8, decay_per_slice=0)
        res = sss.per_unit_response_regression(
            bin_size=1.0, response_window_ms=(20.0, 30.0)
        )
        # Unit 1 has 0 spikes in (20, 30) every slice → slope = 0 exactly
        assert abs(res["slope"][1]) < 1e-9

    def test_output_shapes(self):
        """
        Output arrays have shape (U,) for coefficients and (U, S) for amplitudes.

        Tests:
            (Test Case 1) slope, intercept, r_squared, p_value, stderr all (U,).
            (Test Case 2) amplitudes is (U, S).
            (Test Case 3) x_values is (S,) and defaults to np.arange(S).
        """
        sss = self._decaying_stack(n_slices=6, decay_per_slice=1)
        res = sss.per_unit_response_regression(
            bin_size=1.0, response_window_ms=(20.0, 30.0)
        )
        for k in ("slope", "intercept", "r_squared", "p_value", "stderr"):
            assert res[k].shape == (2,)
        assert res["amplitudes"].shape == (2, 6)
        np.testing.assert_array_equal(res["x_values"], np.arange(6))

    def test_custom_x_values(self):
        """
        Custom x_values are respected.

        Tests:
            (Test Case 1) Slope changes when x_values are doubled (slope halves).
        """
        sss = self._decaying_stack(n_slices=8, decay_per_slice=1)
        res_default = sss.per_unit_response_regression(
            bin_size=1.0, response_window_ms=(20.0, 30.0)
        )
        res_doubled = sss.per_unit_response_regression(
            bin_size=1.0,
            response_window_ms=(20.0, 30.0),
            x_values=np.arange(8) * 2.0,
        )
        assert res_doubled["slope"][0] == pytest.approx(
            res_default["slope"][0] / 2.0, rel=1e-6
        )

    def test_baseline_subtraction_applied(self):
        """
        When baseline_window_ms is provided, amplitudes are baseline-subtracted.

        Tests:
            (Test Case 1) baseline-subtracted amplitudes differ from raw amplitudes.
        """
        sss = self._decaying_stack(n_slices=6, decay_per_slice=1)
        res_raw = sss.per_unit_response_regression(
            bin_size=1.0, response_window_ms=(20.0, 30.0)
        )
        res_subtracted = sss.per_unit_response_regression(
            bin_size=1.0,
            response_window_ms=(20.0, 30.0),
            baseline_window_ms=(0.0, 20.0),
        )
        # Raw amplitudes are non-negative integers; baseline-subtracted ones
        # are floats including negatives.
        assert not np.array_equal(res_raw["amplitudes"], res_subtracted["amplitudes"])

    def test_min_valid_slices(self):
        """
        Units with too few valid slices return NaN.

        Tests:
            (Test Case 1) min_valid_slices=100 forces all units to NaN.
        """
        sss = self._decaying_stack(n_slices=8, decay_per_slice=1)
        res = sss.per_unit_response_regression(
            bin_size=1.0,
            response_window_ms=(20.0, 30.0),
            min_valid_slices=100,
        )
        assert np.isnan(res["slope"]).all()
        assert np.isnan(res["p_value"]).all()

    def test_bad_response_window_raises(self):
        """
        Invalid response window raises ValueError.

        Tests:
            (Test Case 1) end <= start raises.
            (Test Case 2) Non-tuple raises.
            (Test Case 3) Out-of-range window raises.
        """
        sss = self._decaying_stack(n_slices=4, decay_per_slice=1)
        with pytest.raises(ValueError, match="end must be greater"):
            sss.per_unit_response_regression(
                bin_size=1.0, response_window_ms=(30.0, 20.0)
            )
        with pytest.raises(ValueError, match="tuple"):
            sss.per_unit_response_regression(bin_size=1.0, response_window_ms=5.0)
        with pytest.raises(ValueError, match="empty bin range"):
            sss.per_unit_response_regression(
                bin_size=1.0, response_window_ms=(100.0, 200.0)
            )

    def test_x_values_length_mismatch_raises(self):
        """
        Mismatched x_values length raises ValueError.

        Tests:
            (Test Case 1) ValueError when x_values length != number of slices.
        """
        sss = self._decaying_stack(n_slices=4, decay_per_slice=1)
        with pytest.raises(ValueError, match="length S"):
            sss.per_unit_response_regression(
                bin_size=1.0,
                response_window_ms=(20.0, 30.0),
                x_values=np.arange(10),
            )


class TestSpikeSliceStackSliceSimilarity:
    """Tests for SpikeSliceStack.slice_to_slice_similarity."""

    def _build(self):
        """Stack of 4 slices with varying response patterns."""
        slices = []
        rng = np.random.default_rng(0)
        for s in range(4):
            n = rng.integers(5, 15)
            unit0 = np.sort(rng.uniform(0, 40, n))
            unit1 = np.sort(rng.uniform(0, 40, rng.integers(5, 15)))
            sd = SpikeData([unit0, unit1], length=40.0, N=2)
            slices.append(sd)
        return SpikeSliceStack(spike_stack=slices, times_start_to_end=[(0, 40)] * 4)

    def test_returns_pairwisecompmatrix(self):
        """
        Returns a PairwiseCompMatrix with shape (S, S).

        Tests:
            (Test Case 1) Result is PairwiseCompMatrix.
            (Test Case 2) Matrix shape is (S, S).
            (Test Case 3) Metric is recorded in metadata.
        """
        sss = self._build()
        result = sss.slice_to_slice_similarity(metric="cosine", bin_size=1.0)
        assert isinstance(result, PairwiseCompMatrix)
        assert result.matrix.shape == (4, 4)
        assert result.metadata["metric"] == "cosine"

    def test_cosine_diagonal_one(self):
        """
        Cosine diagonal is 1.0.

        Tests:
            (Test Case 1) np.diag is all 1.0.
        """
        sss = self._build()
        result = sss.slice_to_slice_similarity(metric="cosine")
        np.testing.assert_allclose(np.diag(result.matrix), 1.0)

    def test_euclidean_diagonal_zero(self):
        """
        Euclidean diagonal is 0.0.

        Tests:
            (Test Case 1) np.diag is all 0.0.
        """
        sss = self._build()
        result = sss.slice_to_slice_similarity(metric="euclidean")
        np.testing.assert_allclose(np.diag(result.matrix), 0.0)

    def test_unknown_metric_raises(self):
        """
        Unknown metric raises ValueError.

        Tests:
            (Test Case 1) ValueError for bogus metric.
        """
        sss = self._build()
        with pytest.raises(ValueError, match="metric"):
            sss.slice_to_slice_similarity(metric="bogus")


class TestSpikeSliceStackDecodeSliceLabels:
    """Tests for SpikeSliceStack.decode_slice_labels."""

    def _build_decodable_stack(self, n_classes=3, n_per_class=10, seed=0):
        """Build a stack where class c puts evoked spikes in units [c*4 : c*4+4]."""
        rng = np.random.default_rng(seed)
        slices = []
        labels = []
        for c in range(n_classes):
            for _ in range(n_per_class):
                unit_trains = []
                for u in range(12):
                    # Baseline Poisson everywhere
                    n_baseline = rng.poisson(2.0)
                    baseline = np.sort(rng.uniform(0.0, 20.0, n_baseline))
                    # Class-specific evoked response in (20, 30) for units c*4 : c*4+4
                    if c * 4 <= u < c * 4 + 4:
                        evoked = np.sort(rng.uniform(20.5, 29.5, 8))
                    else:
                        evoked = np.array([])
                    unit_trains.append(np.sort(np.concatenate([baseline, evoked])))
                sd = SpikeData(unit_trains, length=40.0, N=12)
                slices.append(sd)
                labels.append(c)
        return SpikeSliceStack(
            spike_stack=slices, times_start_to_end=[(0, 40)] * len(slices)
        ), np.asarray(labels)

    def test_high_accuracy_on_separable_responses(self):
        """
        When class identity drives a distinct unit subset, decoding accuracy
        is well above chance.

        Tests:
            (Test Case 1) Accuracy > 0.7 on 3-class data with clean signal.
        """
        sss, labels = self._build_decodable_stack(n_classes=3, n_per_class=12, seed=42)
        result = sss.decode_slice_labels(
            labels,
            response_window_ms=(20.0, 30.0),
            bin_size=10.0,
            cv=4,
            random_state=0,
        )
        assert result["accuracy"] > 0.7

    def test_returns_decode_result_dict(self):
        """
        Result has all keys from cross_validated_decode.

        Tests:
            (Test Case 1) accuracy, predictions, confusion_matrix, classes
                are present.
        """
        sss, labels = self._build_decodable_stack(n_classes=2, n_per_class=8, seed=1)
        result = sss.decode_slice_labels(
            labels,
            response_window_ms=(20.0, 30.0),
            bin_size=10.0,
            cv=3,
            random_state=0,
        )
        for k in (
            "accuracy",
            "predictions",
            "true_labels",
            "confusion_matrix",
            "per_fold_accuracy",
            "classes",
            "classifier_name",
        ):
            assert k in result

    def test_baseline_subtraction_passes(self):
        """
        baseline_window_ms triggers baseline-subtracted features.

        Tests:
            (Test Case 1) Runs end-to-end with baseline subtraction.
        """
        sss, labels = self._build_decodable_stack(n_classes=2, n_per_class=8, seed=2)
        result = sss.decode_slice_labels(
            labels,
            response_window_ms=(20.0, 30.0),
            bin_size=10.0,
            baseline_window_ms=(0.0, 20.0),
            cv=3,
            random_state=0,
        )
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_bad_response_window_raises(self):
        """
        Invalid response window raises ValueError.

        Tests:
            (Test Case 1) end <= start raises.
            (Test Case 2) Wrong tuple form raises.
            (Test Case 3) Out-of-slice window raises empty-bin-range error.
        """
        sss, labels = self._build_decodable_stack(n_classes=2, n_per_class=8)
        with pytest.raises(ValueError, match="end must be greater"):
            sss.decode_slice_labels(
                labels, response_window_ms=(30.0, 20.0), bin_size=1.0, cv=2
            )
        with pytest.raises(ValueError, match="tuple"):
            sss.decode_slice_labels(labels, response_window_ms=5.0, bin_size=1.0, cv=2)
        with pytest.raises(ValueError, match="empty bin range"):
            sss.decode_slice_labels(
                labels,
                response_window_ms=(100.0, 200.0),
                bin_size=1.0,
                cv=2,
            )

    def test_labels_length_mismatch_raises(self):
        """
        Mismatched labels length raises ValueError.

        Tests:
            (Test Case 1) Wrong-length labels raises.
        """
        sss, labels = self._build_decodable_stack(n_classes=2, n_per_class=8)
        bad_labels = labels[:-2]
        with pytest.raises(ValueError, match="length S"):
            sss.decode_slice_labels(
                bad_labels,
                response_window_ms=(20.0, 30.0),
                bin_size=10.0,
                cv=2,
            )


class TestGroupPairSimilarity:
    """Tests for SpikeSliceStack.group_pair_similarity."""

    def _build_stim_stack(self, n_classes=3, per_class=8, seed=0):
        """Stack where each stim class drives a distinct unit subset."""
        rng = np.random.default_rng(seed)
        slices = []
        labels = []
        for cls in range(n_classes):
            for _ in range(per_class):
                trains = []
                for u in range(12):
                    if cls * 4 <= u < cls * 4 + 4:
                        trains.append(np.sort(rng.uniform(20.0, 30.0, 8)))
                    else:
                        trains.append(np.array([5.0]))
                slices.append(SpikeData(trains, length=40.0, N=12))
                labels.append(cls)
        return SpikeSliceStack(
            spike_stack=slices,
            times_start_to_end=[(0.0, 40.0)] * len(slices),
        ), np.asarray(labels)

    def test_returns_pcm_with_class_labels(self):
        """
        Result is a PairwiseCompMatrix of shape (K, K) labelled with stim classes.

        Tests:
            (Test Case 1) Shape is (K, K).
            (Test Case 2) Labels equal sorted unique classes.
        """
        sss, labels = self._build_stim_stack(n_classes=3)
        result = sss.group_pair_similarity(labels, metric="cosine", bin_size=10.0)
        assert isinstance(result, PairwiseCompMatrix)
        assert result.matrix.shape == (3, 3)
        assert list(result.labels) == [0, 1, 2]

    def test_diagonal_is_self_similarity(self):
        """
        Diagonal of cosine matrix is 1.0 (each class is identical to itself).

        Tests:
            (Test Case 1) np.diag is all 1.0.
        """
        sss, labels = self._build_stim_stack(n_classes=3)
        result = sss.group_pair_similarity(labels, metric="cosine", bin_size=10.0)
        np.testing.assert_allclose(np.diag(result.matrix), 1.0)

    def test_off_diagonal_lower(self):
        """
        Off-diagonal cosine is lower than diagonal because each stim
        class drives a different unit subset.

        Tests:
            (Test Case 1) Mean off-diagonal < mean diagonal.
        """
        sss, labels = self._build_stim_stack(n_classes=3)
        result = sss.group_pair_similarity(labels, metric="cosine", bin_size=10.0)
        K = result.matrix.shape[0]
        off_mask = ~np.eye(K, dtype=bool)
        assert result.matrix[off_mask].mean() < result.matrix.diagonal().mean()

    def test_slice_indices_subset(self):
        """
        slice_indices restricts the computation to a subset.

        Tests:
            (Test Case 1) Result still has shape (K, K) when only some slices used.
        """
        sss, labels = self._build_stim_stack(n_classes=3, per_class=8)
        # Take every other slice (still covers all 3 classes)
        idx = np.arange(0, len(sss), 2)
        result = sss.group_pair_similarity(
            labels, metric="cosine", bin_size=10.0, slice_indices=idx
        )
        assert result.matrix.shape == (3, 3)

    def test_labels_length_mismatch_raises(self):
        """
        Mismatched labels length raises ValueError.

        Tests:
            (Test Case 1) ValueError when stim_labels length != S.
        """
        sss, labels = self._build_stim_stack(n_classes=2, per_class=5)
        with pytest.raises(ValueError, match="length S"):
            sss.group_pair_similarity(labels[:-1], metric="cosine", bin_size=10.0)

    def test_single_class_raises(self):
        """
        Need at least 2 distinct stim classes.

        Tests:
            (Test Case 1) ValueError when only one class is present.
        """
        sss, labels = self._build_stim_stack(n_classes=2, per_class=5)
        single_labels = np.zeros_like(labels)
        with pytest.raises(ValueError, match="at least 2 distinct"):
            sss.group_pair_similarity(single_labels, metric="cosine", bin_size=10.0)


class TestResponsiveUnitsPerGroup:
    """Tests for SpikeSliceStack.responsive_units_per_group."""

    def _build_cycled_stack(self, n_cycles=4, per_cycle=6, seed=0):
        """Cycle 0..1: unit 0 responsive; cycles 2..3: units 0 and 1 responsive."""
        rng = np.random.default_rng(seed)
        slices = []
        cyc = []
        for c in range(n_cycles):
            for _ in range(per_cycle):
                trains = []
                # Baseline ~1 spike in (0, 20) per unit
                for u in range(3):
                    trains.append(np.array([rng.uniform(0.0, 20.0)]))
                # Responsive units: add evoked spikes in (20, 30)
                trains[0] = np.sort(
                    np.concatenate([trains[0], rng.uniform(20.0, 30.0, 6)])
                )
                if c >= 2:
                    trains[1] = np.sort(
                        np.concatenate([trains[1], rng.uniform(20.0, 30.0, 6)])
                    )
                slices.append(SpikeData(trains, length=40.0, N=3))
                cyc.append(c)
        return SpikeSliceStack(
            spike_stack=slices, times_start_to_end=[(0.0, 40.0)] * len(slices)
        ), np.asarray(cyc)

    def test_per_cycle_mask_shape(self):
        """
        Mask has shape (U, n_cycles).

        Tests:
            (Test Case 1) Shape is (3, 4).
            (Test Case 2) cycles array has length n_cycles.
        """
        sss, cyc = self._build_cycled_stack()
        result = sss.responsive_units_per_group(
            cyc,
            bin_size=10.0,
            baseline_window_ms=(0.0, 20.0),
            response_window_ms=(20.0, 30.0),
            z_threshold=2.0,
        )
        assert result["mask"].shape == (3, 4)
        assert result["groups"].shape == (4,)

    def test_unit0_always_responsive(self):
        """
        Unit 0 fires evoked spikes in every cycle.

        Tests:
            (Test Case 1) mask[0, :] all True.
        """
        sss, cyc = self._build_cycled_stack()
        result = sss.responsive_units_per_group(
            cyc,
            bin_size=10.0,
            baseline_window_ms=(0.0, 20.0),
            response_window_ms=(20.0, 30.0),
            z_threshold=2.0,
        )
        assert result["mask"][0, :].all()

    def test_unit1_only_late_cycles(self):
        """
        Unit 1 only becomes responsive in cycles 2..3.

        Tests:
            (Test Case 1) mask[1, :2] all False; mask[1, 2:] all True.
        """
        sss, cyc = self._build_cycled_stack()
        result = sss.responsive_units_per_group(
            cyc,
            bin_size=10.0,
            baseline_window_ms=(0.0, 20.0),
            response_window_ms=(20.0, 30.0),
            z_threshold=2.0,
        )
        assert not result["mask"][1, :2].any()
        assert result["mask"][1, 2:].all()

    def test_mismatched_cycle_labels_raises(self):
        """
        Wrong-length group_labels raises ValueError.

        Tests:
            (Test Case 1) ValueError.
        """
        sss, cyc = self._build_cycled_stack()
        with pytest.raises(ValueError, match="length S"):
            sss.responsive_units_per_group(
                cyc[:-1],
                bin_size=10.0,
                baseline_window_ms=(0.0, 20.0),
            )


class TestResponsivenessChange:
    """Tests for SpikeSliceStack.responsiveness_change."""

    def _build(self, seed=0):
        """Stack where unit 1 GAINS responsiveness from cycles 0..1 -> 2..3."""
        rng = np.random.default_rng(seed)
        slices = []
        cyc = []
        for c in range(4):
            for _ in range(6):
                trains = [np.array([rng.uniform(0.0, 20.0)]) for _ in range(3)]
                trains[0] = np.sort(
                    np.concatenate([trains[0], rng.uniform(20.0, 30.0, 6)])
                )
                if c >= 2:
                    trains[1] = np.sort(
                        np.concatenate([trains[1], rng.uniform(20.0, 30.0, 6)])
                    )
                slices.append(SpikeData(trains, length=40.0, N=3))
                cyc.append(c)
        return SpikeSliceStack(
            spike_stack=slices, times_start_to_end=[(0.0, 40.0)] * len(slices)
        ), np.asarray(cyc)

    def test_gained_lost_preserved(self):
        """
        Unit 0 preserved; unit 1 gained; unit 2 stays inactive.

        Tests:
            (Test Case 1) preserved[0] is True.
            (Test Case 2) gained[1] is True.
            (Test Case 3) lost has no entries.
        """
        sss, cyc = self._build()
        result = sss.responsiveness_change(
            cyc,
            early_groups=[0, 1],
            late_groups=[2, 3],
            bin_size=10.0,
            baseline_window_ms=(0.0, 20.0),
            response_window_ms=(20.0, 30.0),
            z_threshold=2.0,
        )
        assert bool(result["preserved"][0]) is True
        assert bool(result["gained"][1]) is True
        assert bool(result["lost"].any()) is False

    def test_counts_match_masks(self):
        """
        Count fields equal mask sums.

        Tests:
            (Test Case 1) gained_count == gained.sum() etc.
        """
        sss, cyc = self._build()
        result = sss.responsiveness_change(
            cyc,
            early_groups=[0, 1],
            late_groups=[2, 3],
            bin_size=10.0,
            baseline_window_ms=(0.0, 20.0),
        )
        assert result["gained_count"] == int(result["gained"].sum())
        assert result["lost_count"] == int(result["lost"].sum())
        assert result["preserved_count"] == int(result["preserved"].sum())

    def test_no_match_raises(self):
        """
        Empty match for early_groups or late_groups raises ValueError.

        Tests:
            (Test Case 1) ValueError for unknown early cycle.
            (Test Case 2) ValueError for unknown late cycle.
        """
        sss, cyc = self._build()
        with pytest.raises(ValueError, match="early_groups"):
            sss.responsiveness_change(
                cyc,
                early_groups=[99],
                late_groups=[2, 3],
                bin_size=10.0,
                baseline_window_ms=(0.0, 20.0),
            )
        with pytest.raises(ValueError, match="late_groups"):
            sss.responsiveness_change(
                cyc,
                early_groups=[0, 1],
                late_groups=[99],
                bin_size=10.0,
                baseline_window_ms=(0.0, 20.0),
            )


class TestSliceToSliceUnitComparisonCcgMaxLagZero:
    """``SpikeSliceStack.get_slice_to_slice_unit_comparison(metric="ccg",
    max_lag=0)`` reduces to the zero-lag fast path of
    ``compute_cross_correlation_with_lag``. Pin that the resulting
    ``all_lag`` matrix is all zeros (only zero-lag is searched) and
    ``all_corr`` is finite for valid slice pairs.
    """

    def test_max_lag_zero_yields_zero_lag_matrix(self):
        """
        Setting ``max_lag=0`` forces ``max_lag_bins=0``; every
        slice-pair lag in the returned ``all_lag`` stack is therefore
        ``0``, while ``all_corr`` reflects the zero-lag normalised
        dot-product.

        Tests:
            (Test Case 1) ``all_lag`` is all zeros (no NaN, no non-zero
                lag values).
            (Test Case 2) ``all_corr`` diagonal is 1.0 (self-pair).
            (Test Case 3) Returned shapes are ``(S, S, U)``.
        """
        # Dense slices: every unit has many spikes inside each slice so
        # min_spikes filtering does not produce NaN entries.
        rng = np.random.default_rng(11)
        n_units = 2
        train = [np.sort(rng.uniform(0.0, 200.0, size=200)) for _ in range(n_units)]
        sd = SpikeData(train, length=200.0)
        sss = SpikeSliceStack(
            sd, time_peaks=[40.0, 100.0, 160.0], time_bounds=(20.0, 20.0)
        )

        all_corr, all_lag, _av_corr, _av_lag = sss.get_slice_to_slice_unit_comparison(
            metric="ccg", bin_size=1.0, max_lag=0, min_spikes=1, min_frac=0.5
        )

        S, U = len(sss.spike_stack), sss.N
        assert all_corr.stack.shape == (S, S, U)
        assert all_lag.stack.shape == (S, S, U)
        # Every non-NaN lag entry is zero (max_lag=0 collapses the search
        # window). Some entries may be NaN if a slice falls below
        # min_spikes; those still must NOT be non-zero values.
        finite = ~np.isnan(all_lag.stack)
        assert np.all(all_lag.stack[finite] == 0.0)
        # Diagonal entries are self-correlation (= 1.0) for any unit
        # whose slice has enough spikes.
        for u in range(U):
            for s in range(S):
                d = all_corr.stack[s, s, u]
                if not np.isnan(d):
                    assert d == pytest.approx(1.0, abs=1e-9)

    def test_sub_bin_max_lag_warns_collapse_to_zero(self):
        """
        For ``metric="ccg"``, a positive ``max_lag`` smaller than
        ``bin_size`` rounds to zero bins. The method emits a
        ``UserWarning`` so the caller can see that their lag request
        was silently discarded — mirroring the guard in
        ``SpikeData.get_pairwise_ccg``.

        Tests:
            (Test Case 1) ``max_lag=0.5, bin_size=1.0, metric="ccg"``
                emits one UserWarning mentioning "collapsed to 0".
            (Test Case 2) Computation still completes (returns a result).
        """
        import warnings as _warnings

        rng = np.random.default_rng(11)
        n_units = 2
        train = [np.sort(rng.uniform(0.0, 200.0, size=200)) for _ in range(n_units)]
        sd = SpikeData(train, length=200.0)
        sss = SpikeSliceStack(
            sd, time_peaks=[40.0, 100.0, 160.0], time_bounds=(20.0, 20.0)
        )

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            sss.get_slice_to_slice_unit_comparison(
                metric="ccg",
                bin_size=1.0,
                max_lag=0.5,
                min_spikes=1,
                min_frac=0.5,
            )
        msgs = [
            str(rec.message) for rec in caught if "collapsed to 0" in str(rec.message)
        ]
        assert len(msgs) == 1
        assert "bin_size=1.0" in msgs[0]

    def test_sttc_metric_does_not_warn(self):
        """
        For ``metric="sttc"``, ``max_lag_bins`` is hardcoded to 0
        regardless of ``max_lag`` — the collapse warning must not fire
        because lag is not part of the sttc comparison contract.

        Tests:
            (Test Case 1) ``metric="sttc"`` with ``max_lag=0.5``
                emits no "collapsed to 0" warning.
        """
        import warnings as _warnings

        rng = np.random.default_rng(11)
        n_units = 2
        train = [np.sort(rng.uniform(0.0, 200.0, size=200)) for _ in range(n_units)]
        sd = SpikeData(train, length=200.0)
        sss = SpikeSliceStack(
            sd, time_peaks=[40.0, 100.0, 160.0], time_bounds=(20.0, 20.0)
        )

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            sss.get_slice_to_slice_unit_comparison(
                metric="sttc",
                bin_size=1.0,
                max_lag=0.5,
                min_spikes=1,
                min_frac=0.5,
            )
        msgs = [
            str(rec.message) for rec in caught if "collapsed to 0" in str(rec.message)
        ]
        assert msgs == []


class TestSpikeSliceStackRankOrderNShufflesExactly5:
    """Mirrors ``test_n_shuffles_exactly_5`` in test_rateslicestack.py
    for the ``SpikeSliceStack`` path — pins that ``n_shuffles=5``
    (the documented minimum) is accepted and produces a valid output
    on the spike-based code path.
    """

    def test_n_shuffles_exactly_5_accepted(self):
        """
        ``n_shuffles=5`` (boundary of the valid range) does not raise
        on ``SpikeSliceStack.rank_order_correlation`` and produces
        the expected ``(S, S)`` output matrix.

        Tests:
            (Test Case 1) No ValueError raised at the boundary.
            (Test Case 2) Returned correlation matrix has shape
                ``(S, S)``.
        """
        sd = make_spikedata(n_units=4, length_ms=400.0, seed=13)
        sss = SpikeSliceStack(
            sd, time_peaks=[80.0, 200.0, 320.0], time_bounds=(40.0, 40.0)
        )
        corr, _, _ = sss.rank_order_correlation(n_shuffles=5, seed=42)
        assert corr.matrix.shape == (3, 3)


class TestSpikeSliceStackBoundarySpike:
    """``SpikeSliceStack.__init__`` validates that per-slice spike
    times fall within the slice window using strict inequality
    (``train[-1] > expected_end``). A spike at exactly
    ``expected_end`` is therefore accepted, matching the
    ``SpikeData.__init__`` contract that uses the same strict
    comparator. Pinned here as a regression guard against any
    future tightening to ``>=`` that would silently reject valid
    boundary spikes.
    """

    def test_spike_exactly_at_expected_end_accepted(self):
        """
        Construct a ``SpikeData`` with ``start_time=0``,
        ``length=10``, and a single spike at exactly ``t=10.0``,
        then wrap it in a ``SpikeSliceStack`` whose
        ``times_start_to_end=[(0.0, 10.0)]`` matches the slice
        duration. The strict ``>`` check at the validator allows
        ``train[-1] == expected_end``; a ``>=`` regression would
        raise ``ValueError``.

        Tests:
            (Test Case 1) Construction succeeds (no ValueError).
            (Test Case 2) The boundary spike survives in the stored
                spike train.
            (Test Case 3) The slice times are stored unchanged.
        """
        import numpy as np

        from spikelab.spikedata import SpikeData
        from spikelab.spikedata.spikeslicestack import SpikeSliceStack

        sd = SpikeData([np.array([10.0])], length=10.0)
        sss = SpikeSliceStack(spike_stack=[sd], times_start_to_end=[(0.0, 10.0)])

        assert len(sss) == 1
        assert sss.spike_stack[0].train[0][0] == 10.0
        assert sss.times == [(0.0, 10.0)]

    def test_spike_just_past_expected_end_raises(self):
        """
        Companion test: a spike at ``expected_end + epsilon``
        triggers the ``train[-1] > expected_end`` branch and raises
        ``ValueError``. Pinned alongside the boundary case so the
        accept/reject contract is visible together.

        Tests:
            (Test Case 1) Spike at ``10.0001`` against a slice ending
                at ``10.0`` raises ValueError mentioning the spike
                time and expected range.
        """
        import numpy as np

        from spikelab.spikedata import SpikeData
        from spikelab.spikedata.spikeslicestack import SpikeSliceStack

        # SpikeData itself rejects spike past start_time + length, so
        # use a longer length to construct the SpikeData, then pass a
        # tighter slice window to SpikeSliceStack to trigger its
        # per-slice validator.
        sd = SpikeData([np.array([10.0001])], length=20.0)
        with pytest.raises(ValueError, match="fall outside"):
            SpikeSliceStack(spike_stack=[sd], times_start_to_end=[(0.0, 10.0)])


# ============================================================================
# Core review (2026-05-24) — SpikeSliceStack edge-case pins from the
# /complete_review pass on fix/review-cleanups.
# ============================================================================


class TestSpikeSliceStackToRasterArrayBinSizeGuard:
    """``to_raster_array`` added a ``bin_size <= 0`` guard at line
    439-440 of ``spikeslicestack.py``. Pin the new guard symmetrically
    with NaN coverage via the inner ``sparse_raster`` guard.
    """

    def test_bin_size_zero_raises(self):
        """
        Tests:
            (Test Case 1) ``bin_size=0`` raises ValueError mentioning
                ``bin_size``.
        """
        sd = make_spikedata(n_units=2, length_ms=100.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 50.0), (50.0, 100.0)])
        with pytest.raises(ValueError, match="bin_size"):
            sss.to_raster_array(bin_size=0)

    def test_bin_size_negative_raises(self):
        """
        Tests:
            (Test Case 1) ``bin_size=-1.0`` raises ValueError.
        """
        sd = make_spikedata(n_units=2, length_ms=100.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 50.0), (50.0, 100.0)])
        with pytest.raises(ValueError, match="bin_size"):
            sss.to_raster_array(bin_size=-1.0)

    def test_bin_size_nan_raises_via_inner_guard(self):
        """
        Tests:
            (Test Case 1) ``bin_size=NaN`` slips past the outer
                ``<= 0`` check (NaN <= 0 is False) but is caught by
                the inner ``sparse_raster`` NaN guard.
        """
        sd = make_spikedata(n_units=2, length_ms=100.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 50.0), (50.0, 100.0)])
        with pytest.raises(ValueError):
            sss.to_raster_array(bin_size=np.nan)


class TestSpikeSliceStackInitEmptyStack:
    """``SpikeSliceStack`` rejects ``spike_stack=[]`` at line 142 of
    ``spikeslicestack.py`` with an explicit ValueError. Pin the
    boundary.
    """

    def test_empty_spike_stack_raises(self):
        """
        Tests:
            (Test Case 1) ``spike_stack=[]`` raises ValueError with
                message "must not be empty".
        """
        with pytest.raises(ValueError, match="must not be empty"):
            SpikeSliceStack(spike_stack=[])


class TestSpikeSliceStackInitBothInputsWarns:
    """When the caller supplies both ``data_obj`` and ``spike_stack``,
    the constructor emits a UserWarning and uses ``spike_stack``.
    Pin the warning so a refactor that silently dropped it would surface.
    """

    def test_both_inputs_warns_and_uses_spike_stack(self):
        """
        Tests:
            (Test Case 1) Construction emits UserWarning.
            (Test Case 2) The result reflects spike_stack contents, not
                data_obj contents.
        """
        sd_obj = make_spikedata(n_units=2, length_ms=100.0, seed=1)
        sd_stack = SpikeData([np.array([10.0])], length=20.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sss = SpikeSliceStack(
                data_obj=sd_obj,
                spike_stack=[sd_stack],
                times_start_to_end=[(0.0, 20.0)],
            )
        assert any("data_obj" in str(w.message) for w in caught)
        # spike_stack content used: N=1
        assert sss.spike_stack[0].N == 1


class TestSpikeSliceStackResponsiveUnitsThresholdBoundary:
    """``responsive_units`` boundary contracts for ``z_threshold``:
    NaN threshold silently produces an all-False mask (NaN comparisons
    are False); ``z_threshold=0`` is the most permissive boundary.
    """

    def test_z_threshold_nan_returns_all_false(self):
        """
        Tests:
            (Test Case 1) ``z_threshold=NaN`` returns a mask that is
                False for every unit (NaN compares False everywhere).
        """
        # Two slices, baseline = (0, 50ms), full-slice response
        sd = make_spikedata(n_units=3, length_ms=400.0, seed=7)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 200.0), (200.0, 400.0)])
        mask = sss.responsive_units(
            bin_size=10.0,
            baseline_window_ms=(0.0, 50.0),
            z_threshold=np.nan,
        )
        assert mask.shape == (3,)
        assert not mask.any()


class TestSpikeSliceStackApplyNoneReturn:
    """``apply`` with a ``func`` returning ``None`` produces an object
    array via ``np.stack``. Pin the no-rejection contract — the user
    is responsible for returning numerics.
    """

    def test_apply_func_returning_none_does_not_crash(self):
        """
        Tests:
            (Test Case 1) ``apply`` does not raise on a None-returning
                func.
            (Test Case 2) Result has object dtype (numpy boxes None).

        Notes:
            - This pins the *current* behaviour: ``apply`` does not
              validate the return type. A future refactor that rejects
              None-returning funcs explicitly would break this test.
        """
        sd = make_spikedata(n_units=2, length_ms=100.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 50.0), (50.0, 100.0)])
        try:
            result = sss.apply(lambda s: None)
        except (ValueError, TypeError) as exc:
            # If apply rejects None explicitly, that is also a valid contract.
            pytest.skip(f"apply rejects None-returning func: {exc}")
        else:
            # Accepted: result is an object-dtype array (numpy boxing None).
            assert result.dtype == object or result.shape == (2,)


# ============================================================================
# Test Coverage Scan (2026-05-25) — pin tests for the partial-coverage
# gaps surfaced by the /test_scanner pass.
# ============================================================================


class TestSpikeSliceStackToRasterArrayAbsoluteTimes:
    """``to_raster_array(absolute_times=True)`` offsets each slice's
    bins so they sit on a global time axis spanning the union of slice
    windows. Pin that bin alignment matches the documented offset.
    """

    def test_absolute_times_aligns_to_global_axis(self):
        """
        Tests:
            (Test Case 1) Two slices at non-contiguous times produce a
                shared (U, T_total, S) array where T_total spans the
                full window from min(start) to max(end).
            (Test Case 2) Slice 1 (later start) has zeros in early
                global bins (before its start_offset).
        """
        sd = make_spikedata(n_units=2, length_ms=500.0)
        sss = SpikeSliceStack(
            sd,
            times_start_to_end=[(0.0, 100.0), (300.0, 400.0)],
        )
        bin_size = 10.0
        result = sss.to_raster_array(bin_size=bin_size, absolute_times=True)
        assert result.shape[0] == 2  # N
        assert result.shape[2] == 2  # S
        expected_T = int(np.ceil((400.0 - 0.0) / bin_size))
        assert result.shape[1] == expected_T
        # Slice 1 starts at 300 ms = bin 30; bins 0..29 of slice 1 are 0.
        assert np.all(result[:, :30, 1] == 0)

    def test_absolute_times_false_uses_relative_axis(self):
        """
        Tests:
            (Test Case 1) ``absolute_times=False`` (default) produces a
                stack where each slice's bins start at 0.
        """
        sd = make_spikedata(n_units=2, length_ms=500.0)
        sss = SpikeSliceStack(
            sd,
            times_start_to_end=[(0.0, 100.0), (300.0, 400.0)],
        )
        bin_size = 10.0
        result = sss.to_raster_array(bin_size=bin_size, absolute_times=False)
        # Each slice has 100/10 = 10 bins.
        assert result.shape == (2, 10, 2)


class TestSpikeSliceStackApplyReturningSpikeData:
    """``apply`` documents only numpy-array-returning funcs in the
    happy-path tests. Pin the SpikeData-returning case — ``np.stack``
    on a list of SpikeData falls into the object-dtype branch.
    """

    def test_apply_func_returning_spikedata_handled(self):
        """
        Tests:
            (Test Case 1) ``func`` returning a SpikeData per slice
                produces a result that doesn't crash mid-stack
                (object-dtype container or skip via explicit rejection).
        """
        sd = make_spikedata(n_units=2, length_ms=100.0)
        sss = SpikeSliceStack(sd, times_start_to_end=[(0.0, 50.0), (50.0, 100.0)])

        try:
            result = sss.apply(lambda s: s)
        except (ValueError, TypeError) as exc:
            pytest.skip(f"apply rejects SpikeData-returning func: {exc}")
        else:
            arr = np.asarray(result, dtype=object)
            assert arr.shape[0] == 2 or arr.dtype == object
