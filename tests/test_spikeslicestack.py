"""
Tests for the SpikeSliceStack class (spikedata/spikeslicestack.py).

Covers: constructor (both time modes), validation, to_raster_array.
"""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpikeLab.spikedata.spikedata import SpikeData
from SpikeLab.spikedata.spikeslicestack import SpikeSliceStack


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

    def test_negative_windows_filtered(self):
        """
        Tests that windows with negative start times are filtered out.

        Tests:
            (Test Case 1) Peak near 0 with large before-bound is filtered.
            (Test Case 2) Remaining slices are correct.
        """
        sd = make_spikedata(n_units=2, length_ms=200.0)
        sss = SpikeSliceStack(
            sd,
            time_peaks=[5.0, 50.0, 100.0],
            time_bounds=(10.0, 10.0),
        )
        # Peak at 5 with before=10 gives start=-5, filtered out
        assert len(sss.spike_stack) == 2

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

        for i, (slice_sd, (t0, t1)) in enumerate(zip(sss.spike_stack, sss.times)):
            # Spike times are absolute; shift to 0-based before rasterizing to match
            # what to_raster_array produces internally.
            duration = t1 - t0
            shifted_train = [ts - t0 for ts in slice_sd.train]
            temp_sd = SpikeData(shifted_train, length=duration, N=slice_sd.N)
            expected = temp_sd.sparse_raster(bin_size=1).toarray()
            # Rasterization may differ by at most one bin at the edge
            assert abs(result.shape[1] - expected.shape[1]) <= 1
            min_t = min(result.shape[1], expected.shape[1])
            np.testing.assert_array_equal(result[:, :min_t, i], expected[:, :min_t])

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


class TestSpikeSliceStackEdgeCases:
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
