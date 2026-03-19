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

    @pytest.mark.xfail(
        reason="Source bug: SpikeSliceStack.subset(by=) passes by= to individual "
        "SpikeData.subset() but the slice SpikeData objects may not have "
        "neuron_attributes. Should pass resolved indices instead.",
        strict=True,
    )
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


@pytest.mark.xfail(
    reason="Source bug: subtime_by_index passes absolute times to SpikeData "
    "with relative length (from shift_time=False subtime). Needs source fix.",
    strict=True,
)
class TestSubtimeByIndex:
    """Tests for SpikeSliceStack.subtime_by_index().

    NOTE: These tests are currently xfail because subtime_by_index has a source
    bug — it passes absolute time coordinates to sd.subtime() but the slice
    SpikeData objects have length = window_duration (not absolute end time),
    causing subtime to clip the start/end to the duration and raise ValueError.
    """

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
        After trimming, spikes outside the new window are excluded.

        Tests:
            (Test Case 1) All spike times fall within the new absolute window.
        """
        sss = self._make_stack()
        result = sss.subtime_by_index(10, 30)

        for sd, (t_start, t_end) in zip(result.spike_stack, result.times):
            for unit_spikes in sd.train:
                if len(unit_spikes) > 0:
                    assert np.all(unit_spikes >= t_start)
                    assert np.all(unit_spikes < t_end)


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
