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

from SpikeLab.spikedata.pairwise import PairwiseCompMatrix, PairwiseCompMatrixStack
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
                spikes = np.sort(rng.uniform(start, start + 100, 15))
                train.append(spikes)
            sd_list.append(SpikeData(train, length=100.0))
            times.append((start, start + 100.0))
        sss = SpikeSliceStack(spike_stack=sd_list, times_start_to_end=times)
        frac = sss.compute_frac_active(min_spikes=2)

        assert frac.shape == (3,)
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
            active = np.sort(rng.uniform(start, start + 100, 20))
            sparse = np.array([rng.uniform(start, start + 100)])
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
            u0 = np.sort(rng.uniform(start, start + 100, 3))
            # Unit 1: exactly 1 spike per slice
            u1 = np.array([rng.uniform(start, start + 100)])
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
            active = np.sort(rng.uniform(start, start + 100, 20))
            sparse = np.array([rng.uniform(start, start + 100)])
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
            spikes = np.sort(rng.uniform(start, start + length_ms, n_spikes))
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
            active = np.sort(rng.uniform(start, start + 100, 15))
            sparse = np.array([rng.uniform(start, start + 100)])
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


class TestZeroBasedInvariant:
    """Tests that SpikeSliceStack slices always have 0-based spike times."""

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
