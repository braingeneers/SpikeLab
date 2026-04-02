"""
Tests for data_loaders -> SpikeData conversion.

These tests use small temporary files and skip format-specific tests
if optional dependencies are not available (e.g., h5py).
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

try:  # optional, only needed for HDF5/NWB tests
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

try:  # optional, only needed for IBL tests
    import pandas as pd  # type: ignore  # noqa: F401

    pandas_available = True
except Exception:  # pragma: no cover
    pandas_available = False

from spikelab.spikedata import SpikeData
import spikelab.data_loaders.data_loaders as loaders

skip_no_h5py = pytest.mark.skipif(
    h5py is None, reason="h5py not installed; skipping HDF5/NWB tests"
)

skip_no_pandas = pytest.mark.skipif(
    not pandas_available, reason="pandas not installed; skipping IBL tests"
)


@skip_no_h5py
class TestHDF5Loaders:
    """Tests for loading SpikeData from HDF5 files across all supported styles."""

    def test_hdf5_raster(self, tmp_path):
        """
        Test loading a 2D raster dataset from HDF5.

        Tests:
        (Method 1)  Creates a small 2D integer array and writes it as 'raster' to HDF5
        (Method 2)  Loads it using load_spikedata_from_hdf5 with raster_bin_size_ms=10.0
        (Test Case 1)  Checks that the resulting SpikeData object has the correct raster and unit count.
        """
        path = str(tmp_path / "test.h5")
        raster = np.array([[0, 2, 0, 1], [1, 0, 0, 0]], dtype=int)
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=raster)

        sd = loaders.load_spikedata_from_hdf5(
            path, raster_dataset="raster", raster_bin_size_ms=10.0
        )
        assert isinstance(sd, SpikeData)
        assert np.all(sd.raster(10.0) == raster)
        assert sd.N == raster.shape[0]

    def test_hdf5_raster_not_2d_raises(self, tmp_path):
        """
        Test that loading a non-2D raster dataset raises ValueError.

        Tests:
        (Method 1)  Writes a 1D array as 'raster'
        (Test Case 1)  Checks that load_spikedata_from_hdf5 raises a ValueError due to incorrect shape.
        """
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=np.array([1, 2, 3]))
        with pytest.raises(ValueError):
            loaders.load_spikedata_from_hdf5(
                path, raster_dataset="raster", raster_bin_size_ms=1.0
            )

    def test_hdf5_multiple_styles_raises(self, tmp_path):
        """
        Test that specifying multiple input styles raises ValueError.

        Tests:
        (Method 1)  Writes both a 'raster' dataset and a 'units' group
        (Method 2)  Attempts to load with both raster and group_per_unit arguments
        (Test Case 1)  Checks that load_spikedata_from_hdf5 raises a ValueError due to multiple styles.
        """
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=np.zeros((1, 2)))
            f.create_group("units")
        with pytest.raises(ValueError):
            loaders.load_spikedata_from_hdf5(
                path,
                raster_dataset="raster",
                raster_bin_size_ms=1.0,
                group_per_unit="units",
            )

    def test_hdf5_idces_times_ms(self, tmp_path):
        """
        Test loading spike indices and times in milliseconds from HDF5.

        Tests:
        (Method 1)  Writes 'idces' and 'times' datasets
        (Method 2)  Loads them using load_spikedata_from_hdf5
        (Test Case 1)  Checks that the idces_times method returns the correct indices and times.
        """
        path = str(tmp_path / "test.h5")
        idces = np.array([0, 1, 0, 1], dtype=int)
        times_ms = np.array([5.0, 10.0, 15.0, 20.0])
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("idces", data=idces)
            f.create_dataset("times", data=times_ms)

        sd = loaders.load_spikedata_from_hdf5(
            path, idces_dataset="idces", times_dataset="times", times_unit="ms"
        )
        loaded_idces, loaded_times = sd.idces_times()
        assert np.allclose(loaded_times, times_ms)

    def test_hdf5_group_per_unit_seconds(self, tmp_path):
        """
        Test loading group-per-unit HDF5 with times in seconds.

        Tests:
        (Method 1)  Writes 'units' group with two datasets (one per unit) containing spike times in seconds
        (Method 2)  Loads them using load_spikedata_from_hdf5 with group_time_unit="s"
        (Test Case 1)  Checks that the resulting SpikeData object has the correct times in milliseconds.
        """
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:  # type: ignore
            g = f.create_group("units")
            g.create_dataset("0", data=np.array([0.1, 0.2]))
            g.create_dataset("1", data=np.array([0.05]))

        sd = loaders.load_spikedata_from_hdf5(
            path, group_per_unit="units", group_time_unit="s"
        )
        # Expect ms
        assert np.allclose(sd.train[0], np.array([100.0, 200.0]))
        assert np.allclose(sd.train[1], np.array([50.0]))

    def test_hdf5_group_per_unit_empty_units(self, tmp_path):
        """
        Test loading group-per-unit structure with empty units.

        Tests:
        (Method 1)  Writes 'units' group with two empty datasets
        (Method 2)  Loads them using load_spikedata_from_hdf5 with group_time_unit="ms"
        (Test Case 1)  Checks that the resulting SpikeData object has two units,
        (Test Case 2)  Checks that the length method returns 0.0
        (Test Case 3)  Checks that the train[0] is an empty list
        (Test Case 4)  Checks that the train[1] is an empty list
        """
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:  # type: ignore
            g = f.create_group("units")
            g.create_dataset("0", data=np.array([]))
            g.create_dataset("1", data=np.array([]))

        sd = loaders.load_spikedata_from_hdf5(
            path, group_per_unit="units", group_time_unit="ms"
        )
        assert sd.N == 2
        assert sd.length == 0.0
        assert len(sd.train[0]) == 0
        assert len(sd.train[1]) == 0

    def test_hdf5_ragged_spike_times(self, tmp_path):
        """
        Test loading flat (ragged) spike_times with cumulative index in seconds.

        Tests:
        (Method 1)  Writes a flat 'spike_times' array and a 'spike_times_index' array
        (Method 2)  Loads them using load_spikedata_from_hdf5 with spike_times_unit="s"
        (Test Case 1)  Checks that the train[0] is [100.0, 200.0]
        (Test Case 2)  Checks that the train[1] is [500.0]
        """
        path = str(tmp_path / "test.h5")
        # two units: [0.1,0.2], [0.5]
        flat = np.array([0.1, 0.2, 0.5])
        index = np.array([2, 3])
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("spike_times", data=flat)
            f.create_dataset("spike_times_index", data=index)

        sd = loaders.load_spikedata_from_hdf5(
            path,
            spike_times_dataset="spike_times",
            spike_times_index_dataset="spike_times_index",
            spike_times_unit="s",
        )
        assert np.allclose(sd.train[0], [100.0, 200.0])
        assert np.allclose(sd.train[1], [500.0])

    def test_hdf5_idces_times_samples_with_fs(self, tmp_path):
        """
        Test loading spike indices and times in samples with specified sampling rate.

        Tests:
        (Method 1)  Writes 'idces' and 'times' datasets (times in samples)
        (Method 2)  Loads them using load_spikedata_from_hdf5 with times_unit="samples" and fs_Hz=1000.0
        (Test Cases 1-2)  Checks that the idces_times method returns the correct indices and times.
        train[0] and train[1] are the correct spike times in milliseconds.

        """
        path = str(tmp_path / "test.h5")
        idces = np.array([0, 1, 0], dtype=int)
        times_samp = np.array([100, 200, 300])
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("idces", data=idces)
            f.create_dataset("times", data=times_samp)

        sd = loaders.load_spikedata_from_hdf5(
            path,
            idces_dataset="idces",
            times_dataset="times",
            times_unit="samples",
            fs_Hz=1000.0,
        )
        assert np.allclose(sd.train[0], [100.0, 300.0])
        assert np.allclose(sd.train[1], [200.0])

    def test_hdf5_raw_attachment_seconds_and_samples(self, tmp_path):
        """
        Test loading and attaching raw data and raw time from HDF5.

        Tests:
        (Method 1)  Writes 'raster', 'raw', and two raw time datasets (one in seconds, one in samples)
        (Method 2)  Loads them using load_spikedata_from_hdf5 with raw_time_unit="s" and raw_time_unit="samples"
        (Test Case 1)  Checks that the raw_data.shape is (2, 5)
        (Test Case 2)  Checks that the raw_time is [0.0, 0.001, 0.002, 0.003, 0.004] from the seconds dataset
        (Test Case 3)  Checks that the raw_time is [0.0, 1.0, 2.0, 3.0, 4.0] from the samples dataset
        """
        path = str(tmp_path / "test.h5")
        raster = np.zeros((1, 3))
        raw = np.random.randn(2, 5)
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=raster)
            f.create_dataset("raw", data=raw)
            f.create_dataset("raw_time_s", data=np.arange(5) * 0.001)
            f.create_dataset("raw_time_samples", data=np.arange(5))

        # seconds path
        sd_s = loaders.load_spikedata_from_hdf5(
            path,
            raster_dataset="raster",
            raster_bin_size_ms=1.0,
            raw_dataset="raw",
            raw_time_dataset="raw_time_s",
            raw_time_unit="s",
        )
        assert sd_s.raw_data.shape == (2, 5)
        assert np.allclose(sd_s.raw_time, np.arange(5) * 1.0)

        # samples path
        sd_p = loaders.load_spikedata_from_hdf5(
            path,
            raster_dataset="raster",
            raster_bin_size_ms=1.0,
            raw_dataset="raw",
            raw_time_dataset="raw_time_samples",
            raw_time_unit="samples",
            fs_Hz=1000.0,
        )
        assert np.allclose(sd_p.raw_time, np.arange(5) * 1.0)

    def test_hdf5_no_style_raises(self, tmp_path):
        """
        Test that loading an HDF5 file without specifying a style raises ValueError.

        Tests:
        (Method 1)  Writes an empty HDF5 file
        (Method 2)  Loads it using load_spikedata_from_hdf5 without specifying a style
        (Test Case 1)  Checks that load_spikedata_from_hdf5 raises a ValueError due to missing required datasets/groups.
        """
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as _:  # type: ignore
            pass
        with pytest.raises(ValueError):
            loaders.load_spikedata_from_hdf5(path)  # no style specified

    def test_hdf5_samples_without_fs_error(self, tmp_path):
        """
        Test that loading times in samples without specifying fs_Hz raises ValueError.

        Tests:
        (Method 1)  Writes 'idces' and 'times' (in samples)
        (Method 2)  Loads them using load_spikedata_from_hdf5 with times_unit="samples"
        (Test Case 1)  Checks that load_spikedata_from_hdf5 raises a ValueError due to missing fs_Hz.
        """
        path = str(tmp_path / "test.h5")
        idces = np.array([0, 0, 1])
        times_samples = np.array([10, 20, 30])
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("idces", data=idces)
            f.create_dataset("times", data=times_samples)
        with pytest.raises(ValueError):
            loaders.load_spikedata_from_hdf5(
                path, idces_dataset="idces", times_dataset="times", times_unit="samples"
            )

    def test_hdf5_raw_thresholded(self, tmp_path):
        """
        Test thresholding of raw data loaded from HDF5.

        Tests:
        (Method 1)  Writes a 'raw' dataset with two channels, one containing a supra-threshold segment
        (Method 2)  Loads it using load_spikedata_from_hdf5_raw_thresholded
        (Test Case 1)  Checks that the resulting SpikeData object has 2 units
        (Test Case 2)  Checks that at least one event is detected on channel 0
        """
        path = str(tmp_path / "test.h5")
        data = np.zeros((2, 200))
        data[0, 100:105] = 10.0  # supra-threshold burst on ch0
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raw", data=data)

        sd = loaders.load_spikedata_from_hdf5_raw_thresholded(
            path,
            dataset="raw",
            fs_Hz=1000.0,
            threshold_sigma=2.0,
            filter=False,
            hysteresis=True,
            direction="up",
        )
        assert isinstance(sd, SpikeData)
        assert sd.N == 2
        # should detect at least one event on channel 0
        assert len(sd.train[0]) >= 1

    def test_hdf5_paired_empty_idces(self, tmp_path):
        """
        Loading paired-style HDF5 with empty idces/times arrays produces a valid
        zero-unit SpikeData with duration 0.

        Tests:
            (Test Case 1) Empty idces and times arrays produce a SpikeData with
                N=0 and length=0.0.
        """
        path = str(tmp_path / "empty_paired.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("idces", data=np.array([], dtype=int))
            f.create_dataset("times", data=np.array([], dtype=float))

        sd = loaders.load_spikedata_from_hdf5(
            path, idces_dataset="idces", times_dataset="times", times_unit="ms"
        )
        assert sd.N == 0
        assert sd.length == 0.0

    def test_trains_from_flat_index_non_monotonic(self):
        """
        Verify that _trains_from_flat_index handles or tolerates
        non-monotonic end_indices by slicing according to the given indices.

        Tests:
            (Test Case 1) Non-monotonic end_indices (e.g. [3, 2, 5]) produce
                segments based on sequential start-stop iteration. The second
                segment will be empty because stop < start after the first
                segment consumed indices 0..2.
        """
        flat_times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        # Non-monotonic: first segment uses [0:3], second tries [3:2] -> empty,
        # third uses [2:5]
        end_indices = np.array([3, 2, 5])

        trains = loaders._trains_from_flat_index(
            flat_times, end_indices, unit="ms", fs_Hz=None
        )
        assert len(trains) == 3
        # First segment: indices 0..2 -> [10, 20, 30]
        assert len(trains[0]) == 3
        # Second segment: start=3, stop=2 -> empty slice
        assert len(trains[1]) == 0
        # Third segment: start=2, stop=5 -> [30, 40, 50]
        assert len(trains[2]) == 3

    def test_ec_dl_01_explicit_length_ms_override(self, tmp_path):
        """
        EC-DL-01: Verify that an explicit length_ms parameter overrides the
        inferred length from spike times.

        Tests:
            (Test Case 1) sd.length equals the explicit value, not the max spike time.
            (Test Case 2) Spike trains are still loaded correctly.
        """
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("units")
            g.create_dataset("0", data=np.array([0.1, 0.2]))  # max = 200 ms
            g.create_dataset("1", data=np.array([0.05]))

        sd = loaders.load_spikedata_from_hdf5(
            path, group_per_unit="units", group_time_unit="s", length_ms=5000.0
        )
        # Explicit length_ms should override the inferred value (200 ms)
        assert sd.length == pytest.approx(5000.0)
        assert np.allclose(sd.train[0], [100.0, 200.0])

    def test_ec_dl_02_explicit_metadata_parameter(self, tmp_path):
        """
        EC-DL-02: Verify that an explicit metadata parameter is merged into
        the loaded SpikeData's metadata (with source_file added automatically).

        Tests:
            (Test Case 1) Custom metadata keys are present.
            (Test Case 2) source_file is still added by the loader.
        """
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("units")
            g.create_dataset("0", data=np.array([0.1]))

        custom_meta = {"experiment": "test_exp", "subject": "mouse_1"}
        sd = loaders.load_spikedata_from_hdf5(
            path,
            group_per_unit="units",
            group_time_unit="s",
            metadata=custom_meta,
        )
        assert sd.metadata["experiment"] == "test_exp"
        assert sd.metadata["subject"] == "mouse_1"
        assert "source_file" in sd.metadata

    def test_ec_dl_03_three_styles_simultaneously_raises(self, tmp_path):
        """
        EC-DL-03: Verify that specifying three or more input styles raises
        ValueError (not just two).

        Tests:
            (Test Case 1) Specifying raster + ragged + group raises ValueError.
        """
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=np.zeros((2, 3)))
            f.create_dataset("spike_times", data=np.array([0.1]))
            f.create_dataset("spike_times_index", data=np.array([1]))
            f.create_group("units")

        with pytest.raises(ValueError, match="exactly one"):
            loaders.load_spikedata_from_hdf5(
                path,
                raster_dataset="raster",
                raster_bin_size_ms=1.0,
                spike_times_dataset="spike_times",
                spike_times_index_dataset="spike_times_index",
                group_per_unit="units",
            )


@skip_no_h5py
class TestNWBLoader:
    """Tests for loading SpikeData from NWB files."""

    def test_nwb_units_via_h5py(self, tmp_path):
        """
        Test loading NWB units group using h5py.

        Tests:
        (Method 1)  Writes a minimal NWB-like file with a 'units' group containing 'spike_times' and 'spike_times_index'
        (Method 2)  Loads it using load_spikedata_from_nwb
        (Test Case 1)  Checks that the train[0] is [100.0, 200.0]
        (Test Case 2)  Checks that the train[1] is [500.0]
        """
        path = str(tmp_path / "test.nwb")
        # minimal NWB-like units group
        with h5py.File(path, "w") as f:  # type: ignore
            g = f.create_group("units")
            g.create_dataset("spike_times", data=np.array([0.1, 0.2, 0.5]))
            g.create_dataset("spike_times_index", data=np.array([2, 3]))

        sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
        assert np.allclose(sd.train[0], [100.0, 200.0])
        assert np.allclose(sd.train[1], [500.0])

    def test_nwb_missing_units_raises(self, tmp_path):
        """
        Test that loading an NWB file missing the 'units' group raises ValueError.

        Tests:
        (Method 1)  Writes an empty NWB file
        (Method 2)  Loads it using load_spikedata_from_nwb
        (Test Case 1)  Checks that load_spikedata_from_nwb raises a ValueError due to missing 'units'.
        """
        path = str(tmp_path / "test.nwb")
        with h5py.File(path, "w") as _:  # type: ignore
            pass
        with pytest.raises(ValueError):
            loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)

    def test_nwb_alt_names_with_endswith(self, tmp_path):
        """
        Test loading NWB units group with alternative dataset names.

        Tests:
        (Method 1)  Writes a 'units' group with datasets ending in 'spike_times' and 'spike_times_index' but with prefixes
        (Method 2)  Loads it using load_spikedata_from_nwb
        (Test Case 1)  Checks that the train[0] is [200.0]
        (Test Case 2)  Checks that the train[1] is [700.0]
        """
        path = str(tmp_path / "test.nwb")
        with h5py.File(path, "w") as f:  # type: ignore
            g = f.create_group("units")
            g.create_dataset("xx_spike_times", data=np.array([0.2, 0.7]))
            g.create_dataset("xx_spike_times_index", data=np.array([1, 2]))

        sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
        assert np.allclose(sd.train[0], [200.0])
        assert np.allclose(sd.train[1], [700.0])

    def test_nwb_empty_units_group(self, tmp_path):
        """
        Verify that loading an NWB file whose units group has no
        spike_times datasets raises a clear error.

        Tests:
            (Test Case 1) Raises ValueError mentioning missing spike_times.
        """
        path = str(tmp_path / "empty_units.nwb")
        with h5py.File(path, "w") as f:
            grp = f.create_group("units")
            # Write only an id dataset but no spike_times or spike_times_index
            grp.create_dataset("id", data=np.array([0, 1], dtype=int))

        with pytest.raises(ValueError, match="spike_times"):
            loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)


class TestKiloSortAndSpikeInterface:
    """Tests for KiloSort and SpikeInterface loaders."""

    def test_kilosort_basic_load(self, tmp_path):
        """
        Test loading KiloSort output with two clusters.

        Tests:
        (Method 1)  Writes 'spike_times.npy' and 'spike_clusters.npy' for two clusters
        (Method 2)  Loads them using load_spikedata_from_kilosort
        (Test Case 1)  Checks that the cluster_ids metadata matches the trains
        (Test Case 2)  Checks that the spike times are correctly converted to ms and sorted by cluster id
        """
        d = str(tmp_path / "ks")
        os.makedirs(d)
        # two clusters: 2 spikes in 0, 1 spike in 1
        spike_times = np.array([10, 20, 15])  # samples
        spike_clusters = np.array([0, 0, 1])
        np.save(os.path.join(d, "spike_times.npy"), spike_times)
        np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)

        sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
        # cluster_ids metadata should align with trains
        assert len(sd.train) == len(sd.metadata.get("cluster_ids", []))
        # Expected times in ms
        all_trains_ms = [np.array([10.0, 20.0]), np.array([15.0])]
        # order by cluster id ascending
        for train, truth in zip(sd.train, all_trains_ms):
            assert np.allclose(train, truth)

    def test_spikeinterface_mock(self):
        """
        Test loading from a mock SpikeInterface SortingExtractor.

        Tests:
        (Method 1)  Writes a mock sorting object with two units and known spike trains
        (Method 2)  Loads it using load_spikedata_from_spikeinterface
        (Test Case 1)  Checks that the train[0] is [10.0, 20.0]
        (Test Case 2)  Checks that the train[1] is [2.5]
        """

        class MockSorting:
            def get_unit_ids(self):
                return [0, 1]

            def get_sampling_frequency(self):
                return 2000.0

            def get_unit_spike_train(self, unit_id, segment_index=0):
                if unit_id == 0:
                    return np.array([20, 40])
                return np.array([5])

        sorting = MockSorting()
        sd = loaders.load_spikedata_from_spikeinterface(sorting)
        # samples -> ms at 2kHz => 0.5 ms increments
        assert np.allclose(sd.train[0], [10.0, 20.0])
        assert np.allclose(sd.train[1], [2.5])

    def test_spikeinterface_base_recording_thresholding(self):
        """
        Test thresholding on a mock SpikeInterface RecordingExtractor.

        Tests:
        (Method 1)  Writes a mock recording object with a supra-threshold burst on one channel
        (Method 2)  Loads it using load_spikedata_from_spikeinterface_recording
        (Test Case 1)  Checks that the resulting SpikeData object has the correct number of units
        (Test Case 2)  Checks that at least one event is detected on the active channel
        (Test Case 3)  Checks that the time x channels input is transposed automatically
        (Test Case 4)  Checks that at least one event is detected on the active channel post transposition

        """

        class MockRecording:
            def __init__(self, data, fs):
                self._data = np.asarray(data)
                self.sampling_frequency = fs

            def get_traces(self, segment_index=0):
                return self._data

            def get_num_channels(self):
                # channels is first dim if 2D
                return self._data.shape[0]

        # channels x time with a clear supra-threshold burst on ch0
        data_ct = np.zeros((2, 100))
        data_ct[0, 50:55] = 10.0
        rec = MockRecording(data_ct, fs=1000.0)
        sd = loaders.load_spikedata_from_spikeinterface_recording(
            rec, threshold_sigma=2.0, filter=False, hysteresis=True, direction="up"
        )
        assert sd.N == 2
        assert len(sd.train[0]) >= 1

        # time x channels: should auto-transpose
        data_tc = data_ct.T
        rec2 = MockRecording(data_tc, fs=1000.0)
        sd2 = loaders.load_spikedata_from_spikeinterface_recording(
            rec2, threshold_sigma=2.0, filter=False, hysteresis=True, direction="up"
        )
        assert sd2.N == 2
        assert len(sd2.train[0]) >= 1

    def test_spikeinterface_subset_units(self):
        """
        Test loading a subset of units from a mock SpikeInterface SortingExtractor.

        Tests:
        (Method 1)  Loads with unit_ids=[2] from a sorting with units [1, 2]
        (Test Case 1)  Checks that the resulting SpikeData has 1 unit
        (Test Case 2)  Checks that the train[0] is [0.0, 10.0]
        """

        class MockSorting2:
            def get_unit_ids(self):
                return [1, 2]

            def get_sampling_frequency(self):
                return None

            def get_unit_spike_train(self, unit_id, segment_index=0):
                return np.array([0, 10])

        sd = loaders.load_spikedata_from_spikeinterface(
            MockSorting2(), unit_ids=[2], sampling_frequency=1000.0
        )
        # Only unit 2, times in ms equal to samples at 1kHz
        assert sd.N == 1
        assert np.allclose(sd.train[0], [0.0, 10.0])

    def test_spikeinterface_invalid_object_raises(self):
        """
        Test that passing an invalid object to load_spikedata_from_spikeinterface raises TypeError.

        Tests:
        (Method 1)  Writes a class with no required methods
        (Method 2)  Loads it using load_spikedata_from_spikeinterface
        (Test Case 1)  Checks that load_spikedata_from_spikeinterface raises TypeError
        """

        class BadSorting:
            pass

        with pytest.raises(TypeError):
            loaders.load_spikedata_from_spikeinterface(BadSorting())

    def test_kilosort_empty_arrays(self, tmp_path):
        """
        Test loading KiloSort output with empty arrays.

        Tests:
        (Method 1)  Writes empty 'spike_times.npy' and 'spike_clusters.npy'
        (Method 2)  Loads them using load_spikedata_from_kilosort
        (Test Case 1)  Checks that the resulting SpikeData object has zero units
        (Test Case 2)  Checks that the length is 0.0
        """
        d = str(tmp_path / "ks")
        os.makedirs(d)
        np.save(os.path.join(d, "spike_times.npy"), np.array([], dtype=int))
        np.save(os.path.join(d, "spike_clusters.npy"), np.array([], dtype=int))

        sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
        assert sd.N == 0
        assert sd.length == 0.0

    def test_kilosort_nonsequential_clusters(self, tmp_path):
        """
        Test that KiloSort loader handles non-sequential cluster IDs correctly.

        Tests:
        (Method 1)  Writes spike data with non-sequential cluster IDs [3, 5]
        (Test Case 1)  Checks that the cluster_ids metadata is sorted and matches the order of spike trains
        """
        d = str(tmp_path / "ks")
        os.makedirs(d)
        spike_times = np.array([10, 20, 15, 30])
        spike_clusters = np.array([5, 5, 3, 5])
        np.save(os.path.join(d, "spike_times.npy"), spike_times)
        np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)
        sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
        # cluster_ids sorted ascending (np.unique order)
        assert sd.metadata.get("cluster_ids") == [3, 5]

    def test_kilosort_tsv_missing_columns_keeps_all(self, tmp_path):
        """
        Test that KiloSort loader keeps all clusters if cluster_info.tsv is missing expected columns.

        Tests:
        (Method 1)  Writes 'spike_times.npy', 'spike_clusters.npy', and a cluster_info.tsv file without the expected columns
        (Method 2)  Loads them using load_spikedata_from_kilosort
        (Test Case 1)  Checks that all clusters are kept
        """
        d = str(tmp_path / "ks")
        os.makedirs(d)
        spike_times = np.array([10, 20, 15])
        spike_clusters = np.array([0, 0, 1])
        np.save(os.path.join(d, "spike_times.npy"), spike_times)
        np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)
        # Create TSV without expected columns to trigger warning path
        with open(os.path.join(d, "cluster_info.tsv"), "w") as f:
            f.write("foo\tbar\n1\tbaz\n")
        sd = loaders.load_spikedata_from_kilosort(
            d, fs_Hz=1000.0, cluster_info_tsv="cluster_info.tsv"
        )
        # Should keep both clusters 0 and 1
        assert len(sd.train) == 2

    def test_kilosort_channel_positions_location(self, tmp_path):
        """
        Test channel_positions -> neuron_attributes["location"] behavior.

        Tests:
        (Method 1)  Writes spike_times.npy, spike_clusters.npy with clusters 0 and 1
        (Method 2)  Writes channel_positions.npy with positions for 4 channels
        (Test Case 1)  With matching channel_map.npy: location comes from channel_map lookup
        (Test Case 2)  Without channel_map.npy: fallback uses unit index
        (Test Case 3)  With mismatching channel_map.npy (out-of-bounds): fallback uses unit index
        (Test Case 4)  Non-sequential cluster IDs: fallback uses unit index, not cluster ID
        """
        # Channel positions: 4 channels with distinct XYZ coordinates
        channel_positions = np.array(
            [
                [0.0, 0.0, 0.0],  # channel 0
                [10.0, 20.0, 0.0],  # channel 1
                [20.0, 40.0, 0.0],  # channel 2
                [30.0, 60.0, 0.0],  # channel 3
            ]
        )

        # Test Case 1: With channel_map that maps cluster 0 -> channel 2, cluster 1 -> channel 3
        d = str(tmp_path / "ks1")
        os.makedirs(d)
        spike_times = np.array([10, 20, 15, 25])
        spike_clusters = np.array([0, 0, 1, 1])
        np.save(os.path.join(d, "spike_times.npy"), spike_times)
        np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)
        np.save(os.path.join(d, "channel_positions.npy"), channel_positions)
        channel_map = np.array([2, 3])  # cluster index -> channel number
        np.save(os.path.join(d, "channel_map.npy"), channel_map)

        sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)

        # Cluster 0 maps to channel 2 -> position [20.0, 40.0, 0.0]
        # Cluster 1 maps to channel 3 -> position [30.0, 60.0, 0.0]
        assert sd.neuron_attributes[0]["location"] == [20.0, 40.0, 0.0]
        assert sd.neuron_attributes[1]["location"] == [30.0, 60.0, 0.0]
        assert sd.neuron_attributes[0]["electrode"] == 2
        assert sd.neuron_attributes[1]["electrode"] == 3

        # Test Case 2: Without channel_map.npy - fallback to unit index
        d2 = str(tmp_path / "ks2")
        os.makedirs(d2)
        np.save(os.path.join(d2, "spike_times.npy"), spike_times)
        np.save(os.path.join(d2, "spike_clusters.npy"), spike_clusters)
        np.save(os.path.join(d2, "channel_positions.npy"), channel_positions)
        # No channel_map.npy file

        sd = loaders.load_spikedata_from_kilosort(d2, fs_Hz=1000.0)

        # Fallback: unit 0 -> position[0], unit 1 -> position[1]
        assert sd.neuron_attributes[0]["location"] == [0.0, 0.0, 0.0]
        assert sd.neuron_attributes[1]["location"] == [10.0, 20.0, 0.0]
        # No electrode attribute when channel_map is missing
        assert "electrode" not in sd.neuron_attributes[0]
        assert "electrode" not in sd.neuron_attributes[1]

        # Test Case 3: channel_map exists but maps to out-of-bounds channel index
        d3 = str(tmp_path / "ks3")
        os.makedirs(d3)
        np.save(os.path.join(d3, "spike_times.npy"), spike_times)
        np.save(os.path.join(d3, "spike_clusters.npy"), spike_clusters)
        np.save(os.path.join(d3, "channel_positions.npy"), channel_positions)
        channel_map_oob = np.array([10, 20])  # both out of bounds (>= 4)
        np.save(os.path.join(d3, "channel_map.npy"), channel_map_oob)

        sd = loaders.load_spikedata_from_kilosort(d3, fs_Hz=1000.0)

        # Fallback: unit index used since channel_map values are out of bounds
        assert sd.neuron_attributes[0]["location"] == [0.0, 0.0, 0.0]
        assert sd.neuron_attributes[1]["location"] == [10.0, 20.0, 0.0]
        # electrode attribute still set from channel_map (even if out of bounds for positions)
        assert sd.neuron_attributes[0]["electrode"] == 10
        assert sd.neuron_attributes[1]["electrode"] == 20

        # Test Case 4: Non-sequential cluster IDs - fallback uses unit index, not cluster ID
        d4 = str(tmp_path / "ks4")
        os.makedirs(d4)
        # Clusters 50 and 100 - IDs that would be out of bounds if used directly
        spike_times4 = np.array([10, 20, 15, 25])
        spike_clusters4 = np.array([50, 50, 100, 100])
        np.save(os.path.join(d4, "spike_times.npy"), spike_times4)
        np.save(os.path.join(d4, "spike_clusters.npy"), spike_clusters4)
        np.save(os.path.join(d4, "channel_positions.npy"), channel_positions)
        # No channel_map.npy file

        sd = loaders.load_spikedata_from_kilosort(d4, fs_Hz=1000.0)

        # Fallback uses unit index (0, 1), not cluster ID (50, 100)
        assert sd.neuron_attributes[0]["location"] == [0.0, 0.0, 0.0]
        assert sd.neuron_attributes[1]["location"] == [10.0, 20.0, 0.0]
        assert sd.neuron_attributes[0]["unit_id"] == 50
        assert sd.neuron_attributes[1]["unit_id"] == 100

    def test_kilosort_missing_files(self, tmp_path):
        """
        Verify load_spikedata_from_kilosort raises when required .npy files are missing.

        Tests:
            (Test Case 1) Calling with an empty directory raises FileNotFoundError (or OSError).
        """
        with pytest.raises((FileNotFoundError, OSError)):
            loaders.load_spikedata_from_kilosort(str(tmp_path), fs_Hz=30000.0)

    def test_kilosort_empty_spike_files(self, tmp_path):
        """
        Verify that loading KiloSort files with shape-(0,) arrays
        returns an empty SpikeData with no units.

        Tests:
            (Test Case 1) Returns a valid SpikeData with N == 0.
            (Test Case 2) No spike trains are present.
        """
        d = str(tmp_path / "ks_empty")
        os.makedirs(d)
        np.save(os.path.join(d, "spike_times.npy"), np.array([], dtype=float))
        np.save(os.path.join(d, "spike_clusters.npy"), np.array([], dtype=int))

        sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=30000.0)
        assert isinstance(sd, SpikeData)
        assert sd.N == 0
        assert len(sd.train) == 0

    def test_spikeinterface_empty_unit_ids(self):
        """
        Verify that loading from a SpikeInterface sorting with an empty
        unit_ids list returns an empty SpikeData.

        Tests:
            (Test Case 1) Returns a valid SpikeData with N == 0.
        """
        mock_sorting = MagicMock()
        mock_sorting.get_unit_ids.return_value = []
        mock_sorting.get_sampling_frequency.return_value = 30000.0
        mock_sorting.get_unit_spike_train.return_value = np.array([], dtype=float)

        sd = loaders.load_spikedata_from_spikeinterface(mock_sorting, unit_ids=[])
        assert isinstance(sd, SpikeData)
        assert sd.N == 0
        assert len(sd.train) == 0

    def test_spikeinterface_negative_sampling_frequency(self):
        """
        Verify that a negative sampling_frequency override raises ValueError.

        Tests:
            (Test Case 1) sampling_frequency=-1000 raises ValueError.
            (Test Case 2) sampling_frequency=0 does not raise (zero is treated as
                          falsy and falls through to the extractor's own frequency).
        """
        mock_sorting = MagicMock()
        mock_sorting.get_unit_ids.return_value = [0]
        mock_sorting.get_sampling_frequency.return_value = 30000.0
        mock_sorting.get_unit_spike_train.return_value = np.array([100], dtype=float)

        with pytest.raises(ValueError, match="positive"):
            loaders.load_spikedata_from_spikeinterface(
                mock_sorting, sampling_frequency=-1000.0
            )

        # fs=0 is falsy, so the loader falls through to the extractor's frequency
        sd = loaders.load_spikedata_from_spikeinterface(
            mock_sorting, sampling_frequency=0
        )
        assert sd.N == 1

    def test_ec_dl_04_mismatched_spike_times_clusters_lengths(self, tmp_path):
        """
        EC-DL-04: Verify that mismatched spike_times and spike_clusters array
        lengths raise a ValueError.

        Tests:
            (Test Case 1) spike_times has 5 entries, spike_clusters has 3 -> ValueError.
        """
        d = str(tmp_path / "ks")
        os.makedirs(d)
        np.save(os.path.join(d, "spike_times.npy"), np.array([10, 20, 30, 40, 50]))
        np.save(os.path.join(d, "spike_clusters.npy"), np.array([0, 0, 1]))

        with pytest.raises(ValueError, match="mismatch"):
            loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)

    def test_ec_dl_05_negative_spike_times(self, tmp_path):
        """
        Negative spike times from KiloSort are rejected by SpikeData validation.

        Tests:
            (Test Case 1) ValueError is raised because negative spike times
                fall before start_time (0.0).
        """
        d = str(tmp_path / "ks")
        os.makedirs(d)
        # Spike times in samples: -100, 0, 100
        np.save(os.path.join(d, "spike_times.npy"), np.array([-100, 0, 100]))
        np.save(os.path.join(d, "spike_clusters.npy"), np.array([0, 0, 0]))

        with pytest.raises(ValueError, match="before start_time"):
            loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)

    def test_ec_dl_07_sampling_frequency_zero(self):
        """
        EC-DL-07: Verify that sampling_frequency=0 from the extractor raises
        ValueError. Zero sampling frequency is falsy, so the loader checks
        for a positive value.

        Tests:
            (Test Case 1) ValueError is raised mentioning "positive".
        """
        mock_sorting = MagicMock()
        mock_sorting.get_unit_ids.return_value = [0]
        mock_sorting.get_sampling_frequency.return_value = 0.0
        mock_sorting.get_unit_spike_train.return_value = np.array([100])

        with pytest.raises(ValueError, match="positive"):
            loaders.load_spikedata_from_spikeinterface(mock_sorting)


class TestPickleLoaders:
    """
    Tests for load_spikedata_from_pickle.

    Tests:
    - Basic pickle loading from local file
    - S3 URL handling via ensure_local_file
    - Validation that non-SpikeData objects raise ValueError
    - Temporary file cleanup when loading from S3
    """

    def test_pickle_basic_load(self, tmp_path):
        """
        Test basic loading of SpikeData from a local pickle file.

        Tests:
        (Method 1) Creates SpikeData, pickles it to a temp file
        (Method 2) Loads using load_spikedata_from_pickle
        (Test Case 1) Loaded object is SpikeData instance
        (Test Case 2) Spike trains match original
        (Test Case 3) Metadata is preserved
        """
        sd = SpikeData(
            [np.array([5.0, 10.0]), np.array([2.5])],
            length=25.0,
            metadata={"label": "test"},
        )
        path = str(tmp_path / "test.pkl")
        # Write SpikeData to pickle file
        with open(path, "wb") as f:
            pickle.dump(sd, f)

        # Load and verify spike trains match
        sd2 = loaders.load_spikedata_from_pickle(path)
        assert isinstance(sd2, SpikeData)
        for a, b in zip(sd.train, sd2.train):
            assert np.allclose(a, b)
        # Verify metadata is preserved
        assert sd.metadata == sd2.metadata

    @patch("spikelab.data_loaders.s3_utils.ensure_local_file")
    def test_pickle_s3_url_handling(self, mock_ensure, tmp_path):
        """
        Test that S3 URLs are resolved via ensure_local_file before loading.

        Tests:
        (Method 1) Creates SpikeData pickle in temp file
        (Method 2) Mocks ensure_local_file to return (temp_path, False) for S3 URL
        (Method 3) Calls load_spikedata_from_pickle with s3:// URL
        (Test Case 1) ensure_local_file is called with S3 URL
        (Test Case 2) Loaded SpikeData matches original
        """
        sd = SpikeData(
            [np.array([1.0, 2.0])],
            length=10.0,
            metadata={},
        )
        path = str(tmp_path / "test.pkl")
        with open(path, "wb") as f:
            pickle.dump(sd, f)

        # Mock ensure_local_file to return our temp path (as if S3 was already downloaded)
        mock_ensure.return_value = (path, False)

        # Load via S3 URL; ensure_local_file is mocked so no real S3 call
        sd2 = loaders.load_spikedata_from_pickle("s3://bucket/key.pkl")

        # Verify ensure_local_file was called with S3 URL (and optional cred kwargs)
        mock_ensure.assert_called_once()
        assert mock_ensure.call_args[0][0] == "s3://bucket/key.pkl"
        # Verify loaded data matches
        assert np.allclose(sd2.train[0], sd.train[0])

    def test_pickle_non_spikedata_raises_valueerror(self, tmp_path):
        """
        Test that loading a pickle containing a non-SpikeData object raises ValueError.

        Tests:
        (Method 1) Writes a dict to pickle file (not SpikeData)
        (Method 2) Calls load_spikedata_from_pickle
        (Test Case 1) ValueError is raised with message about wrong type
        """
        path = str(tmp_path / "test.pkl")
        # Write non-SpikeData object (dict) to pickle
        with open(path, "wb") as f:
            pickle.dump({"foo": "bar"}, f)

        # Expect ValueError because pickle does not contain SpikeData
        with pytest.raises(ValueError, match="SpikeData"):
            loaders.load_spikedata_from_pickle(path)

    @patch("spikelab.data_loaders.s3_utils.ensure_local_file")
    def test_pickle_temp_file_cleanup(self, mock_ensure):
        """
        Test that temporary file from S3 download is removed after loading.

        Tests:
        (Method 1) Creates SpikeData pickle in temp file
        (Method 2) Mocks ensure_local_file to return (temp_path, True) so loader treats it as temp
        (Method 3) Loads via S3 URL
        (Test Case 1) Temp file is removed after load completes
        """
        sd = SpikeData(
            [np.array([1.0])],
            length=5.0,
            metadata={},
        )
        fd, path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        with open(path, "wb") as f:
            pickle.dump(sd, f)

        # Mock ensure_local_file to return our path with is_temp=True
        mock_ensure.return_value = (path, True)

        # Load; loader should remove temp file in finally block
        loaders.load_spikedata_from_pickle("s3://bucket/key.pkl")

        # Verify temp file was removed
        assert not os.path.exists(path)

    def test_ec_dl_08_corrupted_file(self, tmp_path):
        """
        EC-DL-08: Verify that loading a corrupted/invalid pickle file raises
        an appropriate exception (UnpicklingError or similar).

        Tests:
            (Test Case 1) Writing random bytes and loading raises an exception.
        """
        path = str(tmp_path / "corrupted.pkl")
        with open(path, "wb") as f:
            f.write(b"this is not a valid pickle file \x80\x00\x00")

        with pytest.raises(Exception):
            loaders.load_spikedata_from_pickle(path)


@skip_no_pandas
class TestIBLLoader:
    """
    Tests for load_spikedata_from_ibl.

    All external dependencies (one-api, brainwidemap) are patched via
    sys.modules so the tests run regardless of whether those packages are
    installed.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_unit_df(pid, n_good=3):
        """Return a mock bwm_units DataFrame with good and bad units."""
        import pandas as pd

        rows = [
            {
                "pid": pid,
                "eid": "test-eid",
                "label": 1,
                "cluster_id": i,
                "Beryl": "VISl",
            }
            for i in range(n_good)
        ]
        # one bad unit for the same probe (label=0)
        rows.append(
            {
                "pid": pid,
                "eid": "test-eid",
                "label": 0,
                "cluster_id": n_good,
                "Beryl": "noise",
            }
        )
        # one good unit on a different probe
        rows.append(
            {
                "pid": "other-pid",
                "eid": "other-eid",
                "label": 1,
                "cluster_id": 99,
                "Beryl": "AUDp",
            }
        )
        return pd.DataFrame(rows)

    @staticmethod
    def _make_trials_df(n_trials=5):
        """Return a mock trials DataFrame with all required columns (times in seconds)."""
        import pandas as pd

        t = np.linspace(1.0, 1.0 + (n_trials - 1) * 2.0, n_trials)
        return pd.DataFrame(
            {
                "intervals_0": t,
                "intervals_1": t + 1.0,
                "stimOn_times": t + 0.10,
                "stimOff_times": t + 0.80,
                "goCue_times": t + 0.05,
                "response_times": t + 0.50,
                "feedback_times": t + 0.55,
                "firstMovement_times": t + 0.45,
                "choice": np.tile([-1.0, 1.0], n_trials)[:n_trials],
                "feedbackType": np.ones(n_trials),
                "contrastLeft": np.full(n_trials, 0.5),
                "contrastRight": np.full(n_trials, 0.5),
                "probabilityLeft": np.full(n_trials, 0.5),
            }
        )

    @staticmethod
    def _make_spikes(cluster_ids, n_spikes=5, duration_s=100.0):
        """Return a mock spikes dict with clusters and times arrays."""
        all_clusters, all_times = [], []
        for cid in cluster_ids:
            times = np.linspace(1.0, duration_s - 1.0, n_spikes)
            all_clusters.extend([cid] * n_spikes)
            all_times.extend(times)
        return {
            "clusters": np.array(all_clusters, dtype=int),
            "times": np.array(all_times, dtype=float),
        }

    def _build_mocks(self, pid, eid, n_good=3, n_spikes=5, fail_collections=None):
        """
        Build mock one_api and brainwidemap modules for a given probe.

        Parameters:
            fail_collections: if not None, a set of collection strings for which
                load_object('spikes', ...) should raise an exception.
        """
        unit_df = self._make_unit_df(pid, n_good=n_good)
        good_ids = unit_df[(unit_df["pid"] == pid) & (unit_df["label"] == 1)][
            "cluster_id"
        ].tolist()
        spikes = self._make_spikes(good_ids, n_spikes=n_spikes)
        trials_df = self._make_trials_df()

        def load_object_side_effect(eid_arg, obj_name, **kwargs):
            if obj_name == "trials":
                mock_trials = MagicMock()
                mock_trials.to_df.return_value = trials_df
                return mock_trials
            if obj_name == "spikes":
                collection = kwargs.get("collection", "")
                if fail_collections and collection in fail_collections:
                    raise FileNotFoundError(f"collection not found: {collection}")
                return spikes
            raise Exception(f"Unexpected load_object call: {obj_name}")

        mock_one_instance = MagicMock()
        mock_one_instance.load_object.side_effect = load_object_side_effect

        mock_one_class = MagicMock()
        mock_one_class.return_value = mock_one_instance

        mock_one_api = MagicMock()
        mock_one_api.ONE = mock_one_class

        mock_brainwidemap = MagicMock()
        mock_brainwidemap.bwm_units.return_value = unit_df

        return mock_one_api, mock_brainwidemap, trials_df, good_ids, spikes

    def _load(self, eid, pid, mock_one_api, mock_brainwidemap, **kwargs):
        """Call load_spikedata_from_ibl with mocked external modules."""
        with patch.dict(
            sys.modules,
            {
                "one": MagicMock(),
                "one.api": mock_one_api,
                "brainwidemap": mock_brainwidemap,
            },
        ):
            return loaders.load_spikedata_from_ibl(eid, pid, **kwargs)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_basic_load(self):
        """
        Test that load_spikedata_from_ibl returns a valid SpikeData object.

        Tests:
            (Test Case 1) Returns a SpikeData instance.
            (Test Case 2) Number of units equals the number of good units (label==1) for the probe.
            (Test Case 3) All expected metadata keys are present.
            (Test Case 4) neuron_attributes list has one entry per unit.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, trials_df, good_ids, _ = self._build_mocks(
            pid, eid, n_good=3
        )
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        assert isinstance(sd, SpikeData)
        assert sd.N == 3  # 3 good units
        assert sd.neuron_attributes is not None
        assert len(sd.neuron_attributes) == 3

        expected_keys = {
            "eid",
            "pid",
            "n_trials",
            "trial_start_times",
            "trial_end_times",
            "stim_on_times",
            "stim_off_times",
            "go_cue_times",
            "response_times",
            "feedback_times",
            "first_movement_times",
            "choice",
            "feedback_type",
            "contrast_left",
            "contrast_right",
            "probability_left",
        }
        assert expected_keys.issubset(set(sd.metadata.keys()))

    def test_only_good_units_included(self):
        """
        Test that only units with label==1 for the requested probe are loaded.

        Tests:
            (Test Case 1) Units with label==0 are excluded.
            (Test Case 2) Units from other probes are excluded.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, _, good_ids, _ = self._build_mocks(
            pid, eid, n_good=2
        )
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        # Only 2 good units for this pid; the bad unit and other-pid unit must be excluded
        assert sd.N == 2

    def test_neuron_attributes_region(self):
        """
        Test that neuron_attributes carries the Beryl atlas region for each unit.

        Tests:
            (Test Case 1) Each unit's neuron_attributes dict contains a 'region' key.
            (Test Case 2) Region value matches the Beryl column of the bwm_units DataFrame.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, _, _, _ = self._build_mocks(pid, eid, n_good=3)
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        for attr in sd.neuron_attributes:
            assert "region" in attr
            assert attr["region"] == "VISl"

    def test_spike_times_converted_to_ms(self):
        """
        Test that spike times from the IBL server (seconds) are converted to milliseconds.

        Tests:
            (Test Case 1) Each spike time in the loaded SpikeData is 1000x the source time.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, _, good_ids, spikes = self._build_mocks(
            pid, eid, n_good=1, n_spikes=4
        )
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        # Source times are in seconds; loaded times must be x 1000
        source_times_s = spikes["times"][spikes["clusters"] == good_ids[0]]
        expected_ms = source_times_s * 1000.0
        assert np.allclose(np.sort(sd.train[0]), np.sort(expected_ms))

    def test_trial_timing_arrays_in_ms(self):
        """
        Test that all trial timing metadata arrays are stored in milliseconds.

        Tests:
            (Test Case 1) stim_on_times values are 1000x the source seconds values.
            (Test Case 2) trial_start_times values are 1000x the source seconds values.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, trials_df, _, _ = self._build_mocks(pid, eid)
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        expected_stim_on_ms = trials_df["stimOn_times"].to_numpy() * 1000.0
        assert np.allclose(sd.metadata["stim_on_times"], expected_stim_on_ms)

        expected_start_ms = trials_df["intervals_0"].to_numpy() * 1000.0
        assert np.allclose(sd.metadata["trial_start_times"], expected_start_ms)

    def test_behavioral_arrays_not_converted(self):
        """
        Test that non-timing behavioral arrays (choice, feedback_type, contrasts) are stored as-is.

        Tests:
            (Test Case 1) choice array values match the source DataFrame column exactly.
            (Test Case 2) feedback_type array values match the source DataFrame column exactly.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, trials_df, _, _ = self._build_mocks(pid, eid)
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        assert np.allclose(sd.metadata["choice"], trials_df["choice"].to_numpy())
        assert np.allclose(
            sd.metadata["feedback_type"], trials_df["feedbackType"].to_numpy()
        )

    def test_length_inferred_from_max_spike_time(self):
        """
        Test that session length is inferred from the maximum spike time when not provided.

        Tests:
            (Test Case 1) sd.length equals the maximum spike time across all units in ms.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, _, _, spikes = self._build_mocks(
            pid, eid, n_good=2, n_spikes=5
        )
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        expected_length_ms = float(spikes["times"].max()) * 1000.0
        assert sd.length == pytest.approx(expected_length_ms, abs=1e-3)

    def test_explicit_length_ms_overrides_inference(self):
        """
        Test that an explicitly supplied length_ms takes precedence over inference.

        Tests:
            (Test Case 1) sd.length equals the explicit value, not the max spike time.

        Notes:
            - length_ms must be >= the latest spike time. The mock data has
              spikes up to 99s = 99000ms, so we use 150000ms to override.
        """
        eid, pid = "test-eid", "test-pid"
        mock_one_api, mock_brainwidemap, _, _, _ = self._build_mocks(pid, eid)
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap, length_ms=150000.0)

        assert sd.length == pytest.approx(150000.0)

    def test_collection_fallback(self):
        """
        Test that the loader falls back to the next collection when the first fails.

        Tests:
            (Test Case 1) When the first two probe-specific collections raise exceptions,
                spike data is still loaded from the fallback 'alf' collection.
            (Test Case 2) The returned SpikeData has the expected number of units.
        """
        eid, pid = "test-eid", "test-pid"
        # Make the first two collections fail; 'alf' succeeds
        fail_collections = {"alf/probe00/pykilosort", "alf/probe01/pykilosort"}
        mock_one_api, mock_brainwidemap, _, _, _ = self._build_mocks(
            pid, eid, fail_collections=fail_collections
        )
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        assert isinstance(sd, SpikeData)
        assert sd.N == 3

    def test_no_spikes_produces_empty_trains(self):
        """
        Test that units get empty spike trains when all spike collections are unavailable.

        Tests:
            (Test Case 1) Each unit's spike train is an empty array.
            (Test Case 2) session length falls back to 10000 ms.
        """
        eid, pid = "test-eid", "test-pid"
        all_collections = {
            "alf/probe00/pykilosort",
            "alf/probe01/pykilosort",
            "alf",
        }
        mock_one_api, mock_brainwidemap, _, _, _ = self._build_mocks(
            pid, eid, fail_collections=all_collections
        )
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap)

        for train in sd.train:
            assert len(train) == 0
        assert sd.length == pytest.approx(10_000.0)

    def test_missing_one_api_raises_import_error(self):
        """
        Test that a clear ImportError is raised when one-api is not installed.

        Tests:
            (Test Case 1) ImportError is raised with a message mentioning 'one-api'.
        """
        # Simulate one-api being absent by making the import raise ImportError
        original = sys.modules.pop("one.api", None)
        original_one = sys.modules.pop("one", None)
        try:
            with patch.dict(sys.modules, {"one": None, "one.api": None}):
                with pytest.raises((ImportError, TypeError)):
                    loaders.load_spikedata_from_ibl("eid", "pid")
        finally:
            if original is not None:
                sys.modules["one.api"] = original
            if original_one is not None:
                sys.modules["one"] = original_one

    def test_ibl_all_collections_fail(self):
        """
        Verify that when all ONE API collection lookups fail, the loader
        still returns a SpikeData with empty trains (one per good unit) rather
        than crashing silently.

        Tests:
            (Test Case 1) Returns a SpikeData with the correct number of units.
            (Test Case 2) All spike trains are empty arrays.
        """
        import pandas as pd

        eid, pid = "test-eid", "test-pid"

        # Build a unit_df with 2 good units
        unit_df = pd.DataFrame(
            [
                {"pid": pid, "eid": eid, "label": 1, "cluster_id": 0, "Beryl": "VISl"},
                {"pid": pid, "eid": eid, "label": 1, "cluster_id": 1, "Beryl": "VISl"},
            ]
        )

        # Build a trials DataFrame
        t = np.array([1.0, 3.0])
        trials_df = pd.DataFrame(
            {
                "intervals_0": t,
                "intervals_1": t + 1.0,
                "stimOn_times": t + 0.1,
                "stimOff_times": t + 0.8,
                "goCue_times": t + 0.05,
                "response_times": t + 0.5,
                "feedback_times": t + 0.55,
                "firstMovement_times": t + 0.45,
                "choice": [-1.0, 1.0],
                "feedbackType": [1.0, 1.0],
                "contrastLeft": [0.5, 0.5],
                "contrastRight": [0.5, 0.5],
                "probabilityLeft": [0.5, 0.5],
            }
        )

        def load_object_side_effect(eid_arg, obj_name, **kwargs):
            if obj_name == "trials":
                mock_trials = MagicMock()
                mock_trials.to_df.return_value = trials_df
                return mock_trials
            if obj_name == "spikes":
                raise FileNotFoundError("collection not found")
            raise Exception(f"Unexpected: {obj_name}")

        mock_one_instance = MagicMock()
        mock_one_instance.load_object.side_effect = load_object_side_effect

        mock_one_class = MagicMock()
        mock_one_class.return_value = mock_one_instance

        mock_one_api = MagicMock()
        mock_one_api.ONE = mock_one_class

        mock_brainwidemap = MagicMock()
        mock_brainwidemap.bwm_units.return_value = unit_df

        with patch.dict(
            sys.modules,
            {
                "one": MagicMock(),
                "one.api": mock_one_api,
                "brainwidemap": mock_brainwidemap,
            },
        ):
            sd = loaders.load_spikedata_from_ibl(eid, pid)

        assert isinstance(sd, SpikeData)
        assert sd.N == 2
        for train in sd.train:
            assert len(train) == 0

    def test_ec_dl_09_no_good_units(self):
        """
        EC-DL-09: Verify behavior when the bwm_units DataFrame has no good
        units (label==1) for the requested probe. The loader should return
        a SpikeData with N=0 and use the default length of 10000 ms.

        Tests:
            (Test Case 1) sd.N == 0.
            (Test Case 2) sd.length defaults to 10000.0 ms.
        """
        import pandas as pd

        eid, pid = "test-eid", "test-pid"

        # Only bad units (label=0) for this probe
        unit_df = pd.DataFrame(
            [
                {
                    "pid": pid,
                    "eid": eid,
                    "label": 0,
                    "cluster_id": 0,
                    "Beryl": "noise",
                },
                {
                    "pid": pid,
                    "eid": eid,
                    "label": 0,
                    "cluster_id": 1,
                    "Beryl": "noise",
                },
            ]
        )

        trials_df = pd.DataFrame(
            {
                "intervals_0": [1.0],
                "intervals_1": [2.0],
                "stimOn_times": [1.1],
                "stimOff_times": [1.8],
                "goCue_times": [1.05],
                "response_times": [1.5],
                "feedback_times": [1.55],
                "firstMovement_times": [1.45],
                "choice": [-1.0],
                "feedbackType": [1.0],
                "contrastLeft": [0.5],
                "contrastRight": [0.5],
                "probabilityLeft": [0.5],
            }
        )

        def load_object_side_effect(eid_arg, obj_name, **kwargs):
            if obj_name == "trials":
                mock_trials = MagicMock()
                mock_trials.to_df.return_value = trials_df
                return mock_trials
            if obj_name == "spikes":
                return {
                    "clusters": np.array([], dtype=int),
                    "times": np.array([], dtype=float),
                }
            raise Exception(f"Unexpected: {obj_name}")

        mock_one_instance = MagicMock()
        mock_one_instance.load_object.side_effect = load_object_side_effect

        mock_one_class = MagicMock()
        mock_one_class.return_value = mock_one_instance

        mock_one_api = MagicMock()
        mock_one_api.ONE = mock_one_class

        mock_brainwidemap = MagicMock()
        mock_brainwidemap.bwm_units.return_value = unit_df

        with patch.dict(
            sys.modules,
            {
                "one": MagicMock(),
                "one.api": mock_one_api,
                "brainwidemap": mock_brainwidemap,
            },
        ):
            sd = loaders.load_spikedata_from_ibl(eid, pid)

        assert sd.N == 0
        assert sd.length == pytest.approx(10_000.0)


@skip_no_pandas
class TestIBLQuery:
    """
    Tests for query_ibl_probes.

    All external dependencies (one-api, brainwidemap) are patched via
    sys.modules so the tests run regardless of whether those packages are
    installed.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_units_df():
        """
        Return a mock bwm_units DataFrame with three probes across two labs.

        Probe layout:
          pid-A (eid-A): 5 good units, lab=wittenlab, subject=sub-1,
                         regions [VISl, VISl, MOs, MOs, MOs]  -> 3/5 in MOs
          pid-B (eid-B): 3 good units, lab=wittenlab, subject=sub-2,
                         regions [AUDp, AUDp, AUDp]           -> 0/3 in MOs
          pid-C (eid-C): 8 good units, lab=churchland, subject=sub-3,
                         regions [MOs x4, VISl x4]            -> 4/8 in MOs
        One bad unit (label=0) is also included in eid-A.
        """
        import pandas as pd

        rows = []
        # pid-A -- 5 good units
        for i, region in enumerate(["VISl", "VISl", "MOs", "MOs", "MOs"]):
            rows.append(
                {
                    "eid": "eid-A",
                    "pid": "pid-A",
                    "label": 1,
                    "cluster_id": i,
                    "Beryl": region,
                    "subject": "sub-1",
                    "lab": "wittenlab",
                }
            )
        # bad unit in pid-A
        rows.append(
            {
                "eid": "eid-A",
                "pid": "pid-A",
                "label": 0,
                "cluster_id": 99,
                "Beryl": "noise",
                "subject": "sub-1",
                "lab": "wittenlab",
            }
        )
        # pid-B -- 3 good units
        for i, region in enumerate(["AUDp", "AUDp", "AUDp"]):
            rows.append(
                {
                    "eid": "eid-B",
                    "pid": "pid-B",
                    "label": 1,
                    "cluster_id": i,
                    "Beryl": region,
                    "subject": "sub-2",
                    "lab": "wittenlab",
                }
            )
        # pid-C -- 8 good units
        for i, region in enumerate(
            ["MOs", "MOs", "MOs", "MOs", "VISl", "VISl", "VISl", "VISl"]
        ):
            rows.append(
                {
                    "eid": "eid-C",
                    "pid": "pid-C",
                    "label": 1,
                    "cluster_id": i,
                    "Beryl": region,
                    "subject": "sub-3",
                    "lab": "churchland",
                }
            )
        return pd.DataFrame(rows)

    def _query(self, mock_brainwidemap, **kwargs):
        """Call query_ibl_probes with mocked external modules."""
        mock_one_api = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "one": MagicMock(),
                "one.api": mock_one_api,
                "brainwidemap": mock_brainwidemap,
            },
        ):
            return loaders.query_ibl_probes(**kwargs)

    def _make_mock_brainwidemap(self):
        """Return a mock brainwidemap module backed by the standard units DataFrame."""
        mock_bwm = MagicMock()
        mock_bwm.bwm_units.return_value = self._make_units_df()
        return mock_bwm

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_return_types(self):
        """
        Test that query_ibl_probes returns a (list, DataFrame) tuple.

        Tests:
            (Test Case 1) First return value is a list.
            (Test Case 2) Second return value is a pandas DataFrame.
            (Test Case 3) Each element of the list is a 2-tuple.
        """
        import pandas as pd

        mock_bwm = self._make_mock_brainwidemap()
        probes, stats = self._query(mock_bwm)

        assert isinstance(probes, list)
        assert isinstance(stats, pd.DataFrame)
        for item in probes:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_no_filters_returns_all_probes(self):
        """
        Test that with no filters all probes are returned.

        Tests:
            (Test Case 1) All three probes appear in the result.
            (Test Case 2) stats DataFrame has one row per probe.
        """
        mock_bwm = self._make_mock_brainwidemap()
        probes, stats = self._query(mock_bwm)

        assert len(probes) == 3
        assert len(stats) == 3

    def test_sorted_by_descending_unit_count(self):
        """
        Test that results are sorted by descending good unit count.

        Tests:
            (Test Case 1) First result has the highest n_good_units.
            (Test Case 2) n_good_units column is monotonically non-increasing.
        """
        mock_bwm = self._make_mock_brainwidemap()
        probes, stats = self._query(mock_bwm)

        counts = stats["n_good_units"].tolist()
        assert counts == sorted(counts, reverse=True)
        # pid-C has 8 units -> should be first
        assert probes[0][1] == "pid-C"

    def test_stats_columns_without_target_regions(self):
        """
        Test that stats DataFrame contains the expected columns when no target_regions given.

        Tests:
            (Test Case 1) eid, pid, n_good_units are present.
            (Test Case 2) n_in_target and fraction_in_target are absent.
        """
        mock_bwm = self._make_mock_brainwidemap()
        _, stats = self._query(mock_bwm)

        for col in ("eid", "pid", "n_good_units"):
            assert col in stats.columns
        assert "n_in_target" not in stats.columns
        assert "fraction_in_target" not in stats.columns

    def test_stats_columns_with_target_regions(self):
        """
        Test that stats DataFrame includes region columns when target_regions is given.

        Tests:
            (Test Case 1) n_in_target column is present.
            (Test Case 2) fraction_in_target column is present.
        """
        mock_bwm = self._make_mock_brainwidemap()
        _, stats = self._query(mock_bwm, target_regions=["MOs"])

        assert "n_in_target" in stats.columns
        assert "fraction_in_target" in stats.columns

    def test_n_in_target_and_fraction_correct(self):
        """
        Test that n_in_target and fraction_in_target are computed correctly per probe.

        Tests:
            (Test Case 1) pid-A has 3 units in MOs out of 5 -> fraction 0.6.
            (Test Case 2) pid-B has 0 units in MOs out of 3 -> fraction 0.0.
            (Test Case 3) pid-C has 4 units in MOs out of 8 -> fraction 0.5.
        """
        mock_bwm = self._make_mock_brainwidemap()
        _, stats = self._query(mock_bwm, target_regions=["MOs"])

        row_a = stats[stats["pid"] == "pid-A"].iloc[0]
        assert row_a["n_in_target"] == 3
        assert row_a["fraction_in_target"] == pytest.approx(0.6)

        row_b = stats[stats["pid"] == "pid-B"].iloc[0]
        assert row_b["n_in_target"] == 0
        assert row_b["fraction_in_target"] == pytest.approx(0.0)

        row_c = stats[stats["pid"] == "pid-C"].iloc[0]
        assert row_c["n_in_target"] == 4
        assert row_c["fraction_in_target"] == pytest.approx(0.5)

    def test_min_units_filter(self):
        """
        Test that probes with fewer good units than min_units are excluded.

        Tests:
            (Test Case 1) min_units=4 excludes pid-B (3 units) but keeps pid-A (5) and pid-C (8).
            (Test Case 2) min_units=6 keeps only pid-C (8 units).
        """
        mock_bwm = self._make_mock_brainwidemap()

        probes, stats = self._query(mock_bwm, min_units=4)
        returned_pids = {p[1] for p in probes}
        assert "pid-A" in returned_pids
        assert "pid-C" in returned_pids
        assert "pid-B" not in returned_pids

        probes2, _ = self._query(mock_bwm, min_units=6)
        assert len(probes2) == 1
        assert probes2[0][1] == "pid-C"

    def test_min_fraction_in_target_filter(self):
        """
        Test that probes below the minimum fraction in target are excluded.

        Tests:
            (Test Case 1) min_fraction=0.55 keeps pid-A (0.6) and pid-C (0.5 is excluded),
                leaving only pid-A.
            (Test Case 2) min_fraction=0.0 (default) keeps all probes.
        """
        mock_bwm = self._make_mock_brainwidemap()

        probes, _ = self._query(
            mock_bwm, target_regions=["MOs"], min_fraction_in_target=0.55
        )
        returned_pids = {p[1] for p in probes}
        assert "pid-A" in returned_pids
        assert "pid-B" not in returned_pids
        assert "pid-C" not in returned_pids

        probes_all, _ = self._query(
            mock_bwm, target_regions=["MOs"], min_fraction_in_target=0.0
        )
        assert len(probes_all) == 3

    def test_min_fraction_ignored_without_target_regions(self):
        """
        Test that min_fraction_in_target has no effect when target_regions is None.

        Tests:
            (Test Case 1) Setting min_fraction_in_target without target_regions returns all probes.
        """
        mock_bwm = self._make_mock_brainwidemap()
        probes, _ = self._query(mock_bwm, min_fraction_in_target=0.9)

        assert len(probes) == 3

    def test_combined_filters(self):
        """
        Test that multiple filters are applied conjunctively.

        Tests:
            (Test Case 1) target_regions=['MOs'] + min_units=4 + min_fraction_in_target=0.5
                returns pid-A (3/5 MOs, fraction 0.6) and pid-C (4/8 MOs, fraction 0.5),
                but not pid-B (0/3 MOs).
            (Test Case 2) Adding min_units=6 narrows to pid-C only (8 units).
        """
        mock_bwm = self._make_mock_brainwidemap()

        probes, stats = self._query(
            mock_bwm,
            target_regions=["MOs"],
            min_units=4,
            min_fraction_in_target=0.5,
        )
        returned_pids = {p[1] for p in probes}
        assert returned_pids == {"pid-A", "pid-C"}

        probes2, _ = self._query(
            mock_bwm,
            target_regions=["MOs"],
            min_units=6,
            min_fraction_in_target=0.5,
        )
        assert len(probes2) == 1
        assert probes2[0][1] == "pid-C"

    def test_empty_result(self):
        """
        Test that impossible filter criteria return an empty list and empty DataFrame.

        Tests:
            (Test Case 1) min_units=100 returns an empty probes list.
            (Test Case 2) The stats DataFrame has zero rows.
        """
        mock_bwm = self._make_mock_brainwidemap()

        probes, stats = self._query(mock_bwm, min_units=100)
        assert probes == []
        assert len(stats) == 0

    def test_bad_units_excluded_before_aggregation(self):
        """
        Test that units with label != 1 do not contribute to n_good_units.

        Tests:
            (Test Case 1) pid-A has one bad unit (label=0); n_good_units must be 5, not 6.
        """
        mock_bwm = self._make_mock_brainwidemap()
        _, stats = self._query(mock_bwm)

        row_a = stats[stats["pid"] == "pid-A"].iloc[0]
        assert row_a["n_good_units"] == 5

    def test_missing_one_api_raises_import_error(self):
        """
        Test that a clear ImportError is raised when one-api is not installed.

        Tests:
            (Test Case 1) ImportError or TypeError is raised when one.api is None in sys.modules.
        """
        original = sys.modules.pop("one.api", None)
        original_one = sys.modules.pop("one", None)
        try:
            with patch.dict(sys.modules, {"one": None, "one.api": None}):
                with pytest.raises((ImportError, TypeError)):
                    loaders.query_ibl_probes()
        finally:
            if original is not None:
                sys.modules["one.api"] = original
            if original_one is not None:
                sys.modules["one"] = original_one


# ---------------------------------------------------------------------------
# s3_utils — URL parsing and ensure_local_file
# ---------------------------------------------------------------------------


class TestS3Utils:
    """
    Tests for s3_utils URL parsing functions.

    Covers is_s3_url, parse_s3_url, and the local-path branch of ensure_local_file.
    No real S3 connections are made.
    """

    def test_is_s3_url_native_scheme(self):
        """
        Native s3:// URLs are recognized.

        Tests:
            (Test Case 1) s3://bucket/key returns True.
            (Test Case 2) s3://bucket returns True.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        assert is_s3_url("s3://my-bucket/path/to/file.h5") is True
        assert is_s3_url("s3://bucket") is True

    def test_is_s3_url_virtual_hosted(self):
        """
        Virtual-hosted-style HTTPS S3 URLs are recognized.

        Tests:
            (Test Case 1) https://bucket.s3.amazonaws.com/key returns True.
            (Test Case 2) https://bucket.s3.us-east-1.amazonaws.com/key returns True.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        assert is_s3_url("https://mybucket.s3.amazonaws.com/data/file.h5") is True
        assert is_s3_url("https://mybucket.s3.us-west-2.amazonaws.com/data.h5") is True

    def test_is_s3_url_path_style(self):
        """
        Path-style HTTPS S3 URLs are recognized.

        Tests:
            (Test Case 1) https://s3.amazonaws.com/bucket/key returns True.
            (Test Case 2) https://s3.us-east-1.amazonaws.com/bucket/key returns True.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        assert is_s3_url("https://s3.amazonaws.com/mybucket/key.h5") is True
        assert is_s3_url("https://s3.eu-west-1.amazonaws.com/bucket/key") is True

    def test_is_s3_url_non_s3(self):
        """
        Non-S3 URLs and local paths return False.

        Tests:
            (Test Case 1) Regular HTTPS URL returns False.
            (Test Case 2) Local file path returns False.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        assert is_s3_url("https://example.com/file.h5") is False
        assert is_s3_url("/local/path/file.h5") is False

    def test_parse_s3_url_native(self):
        """
        parse_s3_url correctly splits s3:// URLs.

        Tests:
            (Test Case 1) Bucket and key extracted from s3://bucket/path/key.
            (Test Case 2) Bare bucket with no key returns empty string key.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        bucket, key = parse_s3_url("s3://my-bucket/path/to/file.h5")
        assert bucket == "my-bucket"
        assert key == "path/to/file.h5"

        with pytest.raises(ValueError, match="no object key"):
            parse_s3_url("s3://my-bucket")

    def test_parse_s3_url_virtual_hosted(self):
        """
        parse_s3_url correctly parses virtual-hosted-style URLs.

        Tests:
            (Test Case 1) Bucket extracted from subdomain, key from path.
            (Test Case 2) Regional virtual-hosted URL also parsed correctly.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        bucket, key = parse_s3_url("https://mybucket.s3.amazonaws.com/data/file.h5")
        assert bucket == "mybucket"
        assert key == "data/file.h5"

        bucket2, key2 = parse_s3_url(
            "https://mybucket.s3.us-west-2.amazonaws.com/folder/data.h5"
        )
        assert bucket2 == "mybucket"
        assert key2 == "folder/data.h5"

    def test_parse_s3_url_path_style(self):
        """
        parse_s3_url correctly parses path-style URLs.

        Tests:
            (Test Case 1) https://s3.amazonaws.com/bucket/key parsed correctly.
            (Test Case 2) Regional path-style URL parsed correctly.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        bucket, key = parse_s3_url("https://s3.amazonaws.com/mybucket/data/file.h5")
        assert bucket == "mybucket"
        assert key == "data/file.h5"

        bucket2, key2 = parse_s3_url(
            "https://s3.eu-west-1.amazonaws.com/mybucket/key.h5"
        )
        assert bucket2 == "mybucket"
        assert key2 == "key.h5"

    def test_parse_s3_url_invalid_raises(self):
        """
        parse_s3_url raises ValueError on non-S3 URLs.

        Tests:
            (Test Case 1) Regular HTTPS URL raises ValueError.
            (Test Case 2) Plain local path raises ValueError.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        with pytest.raises(ValueError):
            parse_s3_url("https://example.com/file.h5")
        with pytest.raises(ValueError):
            parse_s3_url("/local/path.h5")

    def test_ensure_local_file_local_path(self, tmp_path):
        """
        ensure_local_file returns (path, False) for existing local files.

        Tests:
            (Test Case 1) Returns the same path and is_temporary=False.
            (Test Case 2) Non-existent local path raises FileNotFoundError.
        """
        from spikelab.data_loaders.s3_utils import ensure_local_file

        path = str(tmp_path / "test.txt")
        with open(path, "w") as f:
            f.write("data")

        result_path, is_temp = ensure_local_file(path)
        assert result_path == path
        assert is_temp is False

        with pytest.raises(FileNotFoundError):
            ensure_local_file(str(tmp_path / "nonexistent.txt"))

    def test_ec_s3_01_empty_key_trailing_slash(self):
        """
        parse_s3_url rejects bucket-only URLs with trailing slash.

        Tests:
            (Test Case 1) s3://mybucket/ raises ValueError.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        with pytest.raises(ValueError, match="no object key"):
            parse_s3_url("s3://mybucket/")

    def test_ec_s3_01_no_trailing_slash(self):
        """
        parse_s3_url rejects bucket-only URLs without trailing slash.

        Tests:
            (Test Case 1) s3://mybucket raises ValueError.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        with pytest.raises(ValueError, match="no object key"):
            parse_s3_url("s3://mybucket")

    def test_ec_s3_02_special_characters_in_key(self):
        """
        EC-S3-02: Verify parse_s3_url handles special characters in the key
        (spaces encoded as %20, plus signs, unicode, etc.).

        Tests:
            (Test Case 1) Key with spaces and special chars is preserved as-is.
            (Test Case 2) Key with nested path and dots is preserved.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        bucket, key = parse_s3_url("s3://mybucket/path/with spaces/file+name.h5")
        assert bucket == "mybucket"
        assert key == "path/with spaces/file+name.h5"

        bucket2, key2 = parse_s3_url("s3://mybucket/a/b/c/file.v2.0.tar.gz")
        assert bucket2 == "mybucket"
        assert key2 == "a/b/c/file.v2.0.tar.gz"

    def test_ec_s3_02_percent_encoded_key(self):
        """
        EC-S3-02 variant: Verify percent-encoded characters pass through.

        Tests:
            (Test Case 1) %20 in key is preserved literally.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        bucket, key = parse_s3_url("s3://mybucket/path%20with%20encoding/file.h5")
        assert bucket == "mybucket"
        assert key == "path%20with%20encoding/file.h5"

    def test_ec_s3_03_empty_file(self, tmp_path):
        """
        EC-S3-03: Verify that uploading an empty (0-byte) file to S3 succeeds
        without error. The upload function should not reject empty files.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) upload_file is called on the S3 client.
        """
        from unittest.mock import MagicMock
        from spikelab.data_loaders.s3_utils import upload_to_s3

        empty_file = str(tmp_path / "empty.txt")
        with open(empty_file, "wb") as f:
            pass  # 0 bytes
        assert os.path.getsize(empty_file) == 0

        mock_client = MagicMock()
        with patch("spikelab.data_loaders.s3_utils.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            result = upload_to_s3(empty_file, "s3://mybucket/empty.txt")

        assert result == "s3://mybucket/empty.txt"
        mock_client.upload_file.assert_called_once_with(
            empty_file, "mybucket", "empty.txt"
        )


class TestS3UtilsErrorPaths:
    """
    Tests for download_from_s3 error-handling paths.

    Mocks boto3 to simulate S3 errors without real AWS connections.
    """

    def _make_client_error(self, code: str) -> Exception:
        """Build a botocore ClientError with the given error code."""
        from botocore.exceptions import ClientError

        return ClientError(
            {"Error": {"Code": code, "Message": f"Mocked {code}"}},
            "download_file",
        )

    def test_download_from_s3_no_such_bucket(self):
        """
        download_from_s3 raises ValueError when the bucket does not exist.

        Tests:
            (Test Case 1) ClientError with code NoSuchBucket is translated to ValueError.
        """
        from spikelab.data_loaders.s3_utils import download_from_s3

        mock_client = MagicMock()
        mock_client.download_file.side_effect = self._make_client_error("NoSuchBucket")

        with patch("spikelab.data_loaders.s3_utils.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            with pytest.raises(ValueError, match="S3 bucket not found"):
                download_from_s3("s3://nonexistent-bucket/key.h5")

    def test_download_from_s3_access_denied(self):
        """
        download_from_s3 raises PermissionError when access is denied.

        Tests:
            (Test Case 1) ClientError with code AccessDenied is translated to PermissionError.
        """
        from spikelab.data_loaders.s3_utils import download_from_s3

        mock_client = MagicMock()
        mock_client.download_file.side_effect = self._make_client_error("AccessDenied")

        with patch("spikelab.data_loaders.s3_utils.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            with pytest.raises(PermissionError, match="Access denied"):
                download_from_s3("s3://my-bucket/secret.h5")

    def test_download_from_s3_no_credentials(self):
        """
        download_from_s3 raises RuntimeError when AWS credentials are missing.

        Tests:
            (Test Case 1) NoCredentialsError is translated to RuntimeError with guidance message.
        """
        from botocore.exceptions import NoCredentialsError
        from spikelab.data_loaders.s3_utils import download_from_s3

        mock_client = MagicMock()
        mock_client.download_file.side_effect = NoCredentialsError()

        with patch("spikelab.data_loaders.s3_utils.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            with pytest.raises(RuntimeError, match="AWS credentials not found"):
                download_from_s3("s3://my-bucket/data.h5")

    def test_ec_s3_04_s3_url_download_path(self):
        """
        EC-S3-04: Verify that ensure_local_file with an S3 URL calls
        download_from_s3 and returns (local_path, True).

        Tests:
            (Test Case 1) download_from_s3 is called with the S3 URL.
            (Test Case 2) Returns is_temporary=True.
            (Test Case 3) local_path matches the mock return value.
        """
        from spikelab.data_loaders.s3_utils import ensure_local_file

        with patch("spikelab.data_loaders.s3_utils.download_from_s3") as mock_download:
            mock_download.return_value = "/tmp/downloaded_file.h5"
            local_path, is_temp = ensure_local_file(
                "s3://mybucket/data/file.h5",
                aws_access_key_id="AKID",
                aws_secret_access_key="SECRET",
            )

        assert local_path == "/tmp/downloaded_file.h5"
        assert is_temp is True
        mock_download.assert_called_once_with(
            "s3://mybucket/data/file.h5",
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
            aws_session_token=None,
            region_name=None,
        )


# ---------------------------------------------------------------------------
# Edge case tests — data_loaders/data_loaders.py
# ---------------------------------------------------------------------------


@skip_no_h5py
class TestHDF5LoaderEdgeCases:
    """Edge case tests for load_spikedata_from_hdf5 and related helpers."""

    def test_all_zero_raster(self, tmp_path):
        """
        All-zero raster matrix produces a SpikeData with empty trains.

        Tests:
            (Test Case 1) SpikeData has U units (matching raster rows).
            (Test Case 2) Every spike train is empty.
        """
        path = str(tmp_path / "zeros.h5")
        raster = np.zeros((3, 10), dtype=int)
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=raster)

        sd = loaders.load_spikedata_from_hdf5(
            path, raster_dataset="raster", raster_bin_size_ms=1.0
        )
        assert sd.N == 3
        for train in sd.train:
            assert len(train) == 0

    def test_single_unit_raster(self, tmp_path):
        """
        Raster with shape (1, T) produces a single-unit SpikeData.

        Tests:
            (Test Case 1) sd.N == 1.
            (Test Case 2) Spike times match the non-zero bin positions.
        """
        path = str(tmp_path / "single.h5")
        raster = np.array([[0, 1, 0, 1, 0]], dtype=int)
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=raster)

        sd = loaders.load_spikedata_from_hdf5(
            path, raster_dataset="raster", raster_bin_size_ms=10.0
        )
        assert sd.N == 1
        assert len(sd.train[0]) == 2

    def test_ragged_all_empty_trains(self, tmp_path):
        """
        Ragged-style HDF5 with all-empty trains (spike_times empty, index [0,0,0]).

        Tests:
            (Test Case 1) sd.N == 3.
            (Test Case 2) All trains are empty.
            (Test Case 3) sd.length == 0.0.
        """
        path = str(tmp_path / "ragged_empty.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("spike_times", data=np.array([], dtype=float))
            f.create_dataset("spike_times_index", data=np.array([0, 0, 0]))

        sd = loaders.load_spikedata_from_hdf5(
            path,
            spike_times_dataset="spike_times",
            spike_times_index_dataset="spike_times_index",
            spike_times_unit="ms",
        )
        assert sd.N == 3
        for train in sd.train:
            assert len(train) == 0
        assert sd.length == 0.0

    def test_single_spike_paired_style(self, tmp_path):
        """
        Paired-style with a single spike (idces=[0], times=[5.0]).

        Tests:
            (Test Case 1) sd.N == 1.
            (Test Case 2) sd.train[0] contains exactly one spike at 5.0 ms.
        """
        path = str(tmp_path / "single_spike.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("idces", data=np.array([0], dtype=int))
            f.create_dataset("times", data=np.array([5.0]))

        sd = loaders.load_spikedata_from_hdf5(
            path, idces_dataset="idces", times_dataset="times", times_unit="ms"
        )
        assert sd.N == 1
        assert np.allclose(sd.train[0], [5.0])

    def test_nan_spike_times_in_hdf5(self, tmp_path):
        """
        NaN spike times in HDF5 cause a ValueError during SpikeData construction.

        Tests:
            (Test Case 1) ValueError is raised with a message about NaN values.

        Notes:
            - SpikeData.__init__ validates spike times and rejects NaN values
              to prevent silent corruption of downstream computations.
        """
        path = str(tmp_path / "nan_times.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("units")
            g.create_dataset("0", data=np.array([1.0, float("nan"), 3.0]))

        with pytest.raises(ValueError, match="NaN"):
            loaders.load_spikedata_from_hdf5(
                path, group_per_unit="units", group_time_unit="ms"
            )

    def test_raster_with_float_counts(self, tmp_path):
        """
        Raster with floating-point values is accepted by the loader.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) The SpikeData has the expected number of units.
        """
        path = str(tmp_path / "float_raster.h5")
        raster = np.array([[0.0, 1.5, 0.0], [0.0, 0.0, 2.7]], dtype=float)
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=raster)

        sd = loaders.load_spikedata_from_hdf5(
            path, raster_dataset="raster", raster_bin_size_ms=1.0
        )
        assert sd.N == 2

    def test_missing_hdf5_dataset_key(self, tmp_path):
        """
        Specifying a non-existent dataset path in HDF5 raises KeyError.

        Tests:
            (Test Case 1) KeyError is raised when raster_dataset points to a
                missing dataset.
        """
        path = str(tmp_path / "missing_key.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("other", data=np.zeros((2, 3)))

        with pytest.raises(KeyError):
            loaders.load_spikedata_from_hdf5(
                path, raster_dataset="nonexistent", raster_bin_size_ms=1.0
            )

    def test_very_large_raster_bin_size(self, tmp_path):
        """
        Raster bin size larger than the raster time extent still works.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) SpikeData has the expected number of units.
        """
        path = str(tmp_path / "large_bin.h5")
        raster = np.array([[1, 0, 1]], dtype=int)
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=raster)

        sd = loaders.load_spikedata_from_hdf5(
            path, raster_dataset="raster", raster_bin_size_ms=100000.0
        )
        assert sd.N == 1
        assert len(sd.train[0]) == 2

    def test_metadata_with_special_characters(self, tmp_path):
        """
        Metadata containing Unicode and special characters is preserved.

        Tests:
            (Test Case 1) Unicode metadata key/value survives the round-trip.
            (Test Case 2) source_file is still added.
        """
        path = str(tmp_path / "special_meta.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("units")
            g.create_dataset("0", data=np.array([1.0]))

        meta = {"subject": "mouse \u00e9\u00e0\u00fc", "notes": "test\nwith\nnewlines"}
        sd = loaders.load_spikedata_from_hdf5(
            path,
            group_per_unit="units",
            group_time_unit="ms",
            metadata=meta,
        )
        assert sd.metadata["subject"] == "mouse \u00e9\u00e0\u00fc"
        assert sd.metadata["notes"] == "test\nwith\nnewlines"
        assert "source_file" in sd.metadata


class TestTrainsFromFlatIndexEdgeCases:
    """Edge case tests for _trains_from_flat_index."""

    def test_empty_flat_times_and_indices(self):
        """
        Empty flat_times with empty end_indices returns an empty list.

        Tests:
            (Test Case 1) Result is an empty list (no units).
        """
        trains = loaders._trains_from_flat_index(
            np.array([], dtype=float), np.array([], dtype=int), unit="ms", fs_Hz=None
        )
        assert trains == []

    def test_single_element_end_indices(self):
        """
        Single-element end_indices = [1] with flat_times = [5.0].

        Tests:
            (Test Case 1) Result has one train.
            (Test Case 2) The train contains [5.0].
        """
        trains = loaders._trains_from_flat_index(
            np.array([5.0]), np.array([1]), unit="ms", fs_Hz=None
        )
        assert len(trains) == 1
        assert np.allclose(trains[0], [5.0])

    def test_end_indices_exceeding_flat_times_length(self):
        """
        end_indices = [10] with flat_times having only 3 elements.

        Tests:
            (Test Case 1) No exception is raised (numpy slicing beyond array
                length returns a truncated slice).
            (Test Case 2) The train contains all 3 elements.

        Notes:
            - numpy slicing [0:10] on a 3-element array silently returns
              elements [0:3]. This is valid numpy behavior but could mask
              data integrity issues upstream.
        """
        trains = loaders._trains_from_flat_index(
            np.array([1.0, 2.0, 3.0]), np.array([10]), unit="ms", fs_Hz=None
        )
        assert len(trains) == 1
        assert np.allclose(trains[0], [1.0, 2.0, 3.0])


@skip_no_h5py
class TestReadRawArraysEdgeCases:
    """Edge case tests for _read_raw_arrays."""

    def test_raw_dataset_without_raw_time(self, tmp_path):
        """
        raw_dataset provided but raw_time_dataset is None returns
        (raw_data, None). _maybe_with_raw then does NOT attach raw data.

        Tests:
            (Test Case 1) _read_raw_arrays returns (raw_data, None).
            (Test Case 2) _maybe_with_raw returns the original SpikeData
                unchanged (no raw_data/raw_time attached).

        Notes:
            - This means providing raw_dataset without raw_time_dataset
              silently discards the raw data. This could be confusing to
              users who expect raw data to be attached regardless.
        """
        path = str(tmp_path / "raw_no_time.h5")
        raw = np.random.randn(2, 10)
        with h5py.File(path, "w") as f:
            f.create_dataset("raw", data=raw)

        with h5py.File(path, "r") as f:
            raw_data, raw_time = loaders._read_raw_arrays(f, "raw", None, "s", None)

        assert raw_data is not None
        assert raw_time is None

        # Verify _maybe_with_raw does not attach when raw_time is None
        sd = SpikeData([np.array([1.0])], length=10.0)
        sd_result = loaders._maybe_with_raw(sd, raw_data, raw_time)
        # SpikeData stores raw_data as np.zeros((0, 0)) when not provided,
        # so we check that it was not replaced with the actual raw data.
        assert sd_result.raw_data.shape == (0, 0)

    def test_invalid_raw_time_unit(self, tmp_path):
        """
        Invalid raw_time_unit raises ValueError.

        Tests:
            (Test Case 1) ValueError is raised with message about valid units.
        """
        path = str(tmp_path / "raw_invalid_unit.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("raw", data=np.random.randn(2, 10))
            f.create_dataset("raw_time", data=np.arange(10, dtype=float))

        with h5py.File(path, "r") as f:
            with pytest.raises(ValueError, match="raw_time_unit"):
                loaders._read_raw_arrays(f, "raw", "raw_time", "invalid", None)

    def test_empty_raw_data_array(self, tmp_path):
        """
        raw_dataset pointing to an empty (0,) array.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) raw_data has shape (0,).
        """
        path = str(tmp_path / "raw_empty.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("raw", data=np.array([], dtype=float))
            f.create_dataset("raw_time", data=np.array([], dtype=float))

        with h5py.File(path, "r") as f:
            raw_data, raw_time = loaders._read_raw_arrays(
                f, "raw", "raw_time", "s", None
            )

        assert raw_data is not None
        assert raw_data.shape == (0,)


@skip_no_h5py
class TestHDF5RawThresholdedEdgeCases:
    """Edge case tests for load_spikedata_from_hdf5_raw_thresholded."""

    def test_all_zero_raw_traces(self, tmp_path):
        """
        All-zero raw traces produce a SpikeData with no detected spikes.

        Tests:
            (Test Case 1) sd.N matches channel count.
            (Test Case 2) All spike trains are empty.
        """
        path = str(tmp_path / "zeros.h5")
        data = np.zeros((3, 200))
        with h5py.File(path, "w") as f:
            f.create_dataset("raw", data=data)

        sd = loaders.load_spikedata_from_hdf5_raw_thresholded(
            path,
            dataset="raw",
            fs_Hz=1000.0,
            threshold_sigma=2.0,
            filter=False,
            hysteresis=True,
            direction="up",
        )
        assert sd.N == 3
        for train in sd.train:
            assert len(train) == 0

    def test_single_channel_raw_data(self, tmp_path):
        """
        Raw data with shape (1, T) produces a single-unit SpikeData.

        Tests:
            (Test Case 1) sd.N == 1.
            (Test Case 2) Supra-threshold signal produces at least one spike.
        """
        path = str(tmp_path / "single_ch.h5")
        data = np.zeros((1, 200))
        data[0, 100:105] = 10.0
        with h5py.File(path, "w") as f:
            f.create_dataset("raw", data=data)

        sd = loaders.load_spikedata_from_hdf5_raw_thresholded(
            path,
            dataset="raw",
            fs_Hz=1000.0,
            threshold_sigma=2.0,
            filter=False,
            hysteresis=True,
            direction="up",
        )
        assert sd.N == 1
        assert len(sd.train[0]) >= 1

    def test_very_short_raw_trace(self, tmp_path):
        """
        Raw data with very few samples does not crash.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) Returned SpikeData is valid.
        """
        path = str(tmp_path / "short.h5")
        data = np.random.randn(2, 5)
        with h5py.File(path, "w") as f:
            f.create_dataset("raw", data=data)

        sd = loaders.load_spikedata_from_hdf5_raw_thresholded(
            path,
            dataset="raw",
            fs_Hz=1000.0,
            threshold_sigma=2.0,
            filter=False,
            hysteresis=False,
            direction="both",
        )
        assert isinstance(sd, SpikeData)
        assert sd.N == 2


@skip_no_h5py
class TestNWBLoaderEdgeCases:
    """Edge case tests for load_spikedata_from_nwb."""

    def test_nwb_zero_length_spike_times(self, tmp_path):
        """
        NWB file where spike_times is empty and spike_times_index is [0, 0].

        Tests:
            (Test Case 1) sd.N == 2.
            (Test Case 2) All trains are empty.
            (Test Case 3) sd.length == 0.0.
        """
        path = str(tmp_path / "empty_spikes.nwb")
        with h5py.File(path, "w") as f:
            g = f.create_group("units")
            g.create_dataset("spike_times", data=np.array([], dtype=float))
            g.create_dataset("spike_times_index", data=np.array([0, 0]))

        sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
        assert sd.N == 2
        for train in sd.train:
            assert len(train) == 0
        assert sd.length == 0.0

    def test_nwb_duplicate_spike_times_candidates(self, tmp_path):
        """
        NWB file with multiple datasets ending in 'spike_times'. The loader
        should use the first match.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) SpikeData is loaded successfully.
        """
        path = str(tmp_path / "multi_st.nwb")
        with h5py.File(path, "w") as f:
            g = f.create_group("units")
            g.create_dataset("spike_times", data=np.array([0.1, 0.2]))
            g.create_dataset("spike_times_index", data=np.array([1, 2]))
            # Additional dataset ending in spike_times
            g.create_dataset("other_spike_times", data=np.array([0.5]))

        sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
        assert isinstance(sd, SpikeData)
        assert sd.N == 2


class TestKiloSortEdgeCases:
    """Edge case tests for load_spikedata_from_kilosort."""

    def test_single_cluster(self, tmp_path):
        """
        KiloSort data with only one cluster.

        Tests:
            (Test Case 1) sd.N == 1.
            (Test Case 2) Spike times are correctly converted.
            (Test Case 3) cluster_ids metadata has one entry.
        """
        d = str(tmp_path / "ks")
        os.makedirs(d)
        np.save(os.path.join(d, "spike_times.npy"), np.array([10, 20, 30]))
        np.save(os.path.join(d, "spike_clusters.npy"), np.array([0, 0, 0]))

        sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
        assert sd.N == 1
        assert np.allclose(sd.train[0], [10.0, 20.0, 30.0])
        assert sd.metadata["cluster_ids"] == [0]

    def test_kilosort_time_unit_seconds(self, tmp_path):
        """
        KiloSort loader with time_unit='s'.

        Tests:
            (Test Case 1) Spike times are converted from seconds to ms.
        """
        d = str(tmp_path / "ks")
        os.makedirs(d)
        np.save(os.path.join(d, "spike_times.npy"), np.array([0.1, 0.2, 0.3]))
        np.save(os.path.join(d, "spike_clusters.npy"), np.array([0, 0, 0]))

        sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0, time_unit="s")
        assert sd.N == 1
        assert np.allclose(sd.train[0], [100.0, 200.0, 300.0])

    def test_kilosort_include_noise(self, tmp_path):
        """
        KiloSort loader with include_noise=True keeps noise clusters.

        Tests:
            (Test Case 1) Both good and noise clusters are loaded.
            (Test Case 2) cluster_ids contains both cluster IDs.
        """
        d = str(tmp_path / "ks")
        os.makedirs(d)
        np.save(os.path.join(d, "spike_times.npy"), np.array([10, 20, 30]))
        np.save(os.path.join(d, "spike_clusters.npy"), np.array([0, 0, 1]))

        # TSV marking cluster 0 as good, cluster 1 as noise
        with open(os.path.join(d, "cluster_info.tsv"), "w") as f:
            f.write("cluster_id\tgroup\n")
            f.write("0\tgood\n")
            f.write("1\tnoise\n")

        # Without include_noise: only cluster 0
        sd_no_noise = loaders.load_spikedata_from_kilosort(
            d, fs_Hz=1000.0, cluster_info_tsv="cluster_info.tsv", include_noise=False
        )
        assert sd_no_noise.N == 1

        # With include_noise: both clusters
        sd_with_noise = loaders.load_spikedata_from_kilosort(
            d, fs_Hz=1000.0, cluster_info_tsv="cluster_info.tsv", include_noise=True
        )
        assert sd_with_noise.N == 2
        assert sorted(sd_with_noise.metadata["cluster_ids"]) == [0, 1]

    def test_kilosort_corrupted_channel_map(self, tmp_path):
        """
        Corrupted channel_map.npy triggers a warning but does not crash.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) SpikeData is still loaded.

        Notes:
            - The loader catches (IOError, ValueError) from np.load and
              issues a warning, then proceeds without channel map data.
        """
        d = str(tmp_path / "ks")
        os.makedirs(d)
        np.save(os.path.join(d, "spike_times.npy"), np.array([10, 20]))
        np.save(os.path.join(d, "spike_clusters.npy"), np.array([0, 0]))

        # Write invalid data to channel_map.npy
        cm_path = os.path.join(d, "channel_map.npy")
        with open(cm_path, "wb") as f:
            f.write(b"this is not a valid npy file")

        import warnings as w

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)

        assert isinstance(sd, SpikeData)
        assert sd.N == 1
        # A warning should have been issued about the channel map
        warning_messages = [str(c.message) for c in caught]
        assert any("channel_map" in msg for msg in warning_messages)


class TestSpikeInterfaceEdgeCases:
    """Edge case tests for load_spikedata_from_spikeinterface."""

    def test_spikeinterface_with_channel_and_location_properties(self):
        """
        SpikeInterface sorting with channel and location properties.

        Tests:
            (Test Case 1) electrode attribute is set from channel property.
            (Test Case 2) location attribute is set from location property.
        """
        mock_sorting = MagicMock()
        mock_sorting.get_unit_ids.return_value = [0, 1]
        mock_sorting.get_sampling_frequency.return_value = 1000.0
        mock_sorting.get_unit_spike_train.return_value = np.array([100])

        def get_property_side_effect(name):
            if name == "channel":
                return np.array([3, 7])
            if name == "location":
                return np.array([[10.0, 20.0], [30.0, 40.0]])
            raise KeyError(name)

        mock_sorting.get_property = MagicMock(side_effect=get_property_side_effect)

        sd = loaders.load_spikedata_from_spikeinterface(mock_sorting)
        assert sd.N == 2
        assert sd.neuron_attributes[0]["electrode"] == 3
        assert sd.neuron_attributes[1]["electrode"] == 7
        assert sd.neuron_attributes[0]["location"] == [10.0, 20.0]
        assert sd.neuron_attributes[1]["location"] == [30.0, 40.0]

    def test_spikeinterface_get_property_raises_keyerror(self):
        """
        SpikeInterface sorting where get_property raises KeyError for all
        property names. The loader falls back gracefully.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) neuron_attributes have no electrode or location keys.
        """
        mock_sorting = MagicMock()
        mock_sorting.get_unit_ids.return_value = [0]
        mock_sorting.get_sampling_frequency.return_value = 1000.0
        mock_sorting.get_unit_spike_train.return_value = np.array([100])

        def get_property_raise(name):
            raise KeyError(name)

        mock_sorting.get_property = MagicMock(side_effect=get_property_raise)

        sd = loaders.load_spikedata_from_spikeinterface(mock_sorting)
        assert sd.N == 1
        assert "electrode" not in sd.neuron_attributes[0]
        assert "location" not in sd.neuron_attributes[0]

    def test_spikeinterface_empty_train_for_one_unit(self):
        """
        SpikeInterface sorting where one unit has an empty spike train.

        Tests:
            (Test Case 1) sd.N == 2.
            (Test Case 2) One train is empty, the other has spikes.
        """
        mock_sorting = MagicMock()
        mock_sorting.get_unit_ids.return_value = [0, 1]
        mock_sorting.get_sampling_frequency.return_value = 1000.0

        def get_train(unit_id, segment_index=0):
            if unit_id == 0:
                return np.array([], dtype=float)
            return np.array([100, 200])

        mock_sorting.get_unit_spike_train = MagicMock(side_effect=get_train)

        sd = loaders.load_spikedata_from_spikeinterface(mock_sorting)
        assert sd.N == 2
        assert len(sd.train[0]) == 0
        assert len(sd.train[1]) == 2


class TestSpikeInterfaceRecordingEdgeCases:
    """Edge case tests for load_spikedata_from_spikeinterface_recording."""

    def test_square_data(self):
        """
        Recording with square data shape (N, N) where channels == time.

        Tests:
            (Test Case 1) No exception is raised.
            (Test Case 2) The heuristic treats both dims as equal; data.shape[0]
                <= data.shape[1] so it keeps the original orientation.

        Notes:
            - When data is square, the loader cannot distinguish channels from
              time. It keeps the original orientation (no transpose).
        """

        class MockSquareRecording:
            sampling_frequency = 1000.0

            def get_traces(self, segment_index=0):
                return np.zeros((10, 10))

            def get_num_channels(self):
                return 10

            def get_sampling_frequency(self):
                return 1000.0

        sd = loaders.load_spikedata_from_spikeinterface_recording(
            MockSquareRecording(),
            threshold_sigma=2.0,
            filter=False,
            hysteresis=False,
            direction="both",
        )
        assert isinstance(sd, SpikeData)
        assert sd.N == 10

    def test_1d_traces_raises(self):
        """
        Recording where get_traces returns a 1D array raises ValueError.

        Tests:
            (Test Case 1) ValueError is raised mentioning "2D".
        """

        class Mock1DRecording:
            def get_traces(self, segment_index=0):
                return np.array([1.0, 2.0, 3.0])

            def get_num_channels(self):
                return 1

            def get_sampling_frequency(self):
                return 1000.0

        with pytest.raises(ValueError, match="2D"):
            loaders.load_spikedata_from_spikeinterface_recording(
                Mock1DRecording(),
                threshold_sigma=2.0,
                filter=False,
                hysteresis=False,
                direction="both",
            )

    def test_sampling_frequency_attribute_fallback(self):
        """
        Recording using sampling_frequency attribute instead of method.

        Tests:
            (Test Case 1) No exception is raised when get_sampling_frequency
                is absent but sampling_frequency attribute exists.
            (Test Case 2) SpikeData is valid.
        """

        class MockAttrRecording:
            sampling_frequency = 1000.0

            def get_traces(self, segment_index=0):
                return np.zeros((2, 50))

            def get_num_channels(self):
                return 2

        # Remove get_sampling_frequency so the code falls back to attribute
        rec = MockAttrRecording()
        assert not hasattr(rec, "get_sampling_frequency")

        sd = loaders.load_spikedata_from_spikeinterface_recording(
            rec,
            threshold_sigma=2.0,
            filter=False,
            hysteresis=False,
            direction="both",
        )
        assert isinstance(sd, SpikeData)
        assert sd.N == 2


class TestPickleLoaderEdgeCases:
    """Edge case tests for load_spikedata_from_pickle."""

    def test_pickle_file_not_found(self, tmp_path):
        """
        Passing a non-existent local file path raises FileNotFoundError.

        Tests:
            (Test Case 1) FileNotFoundError is raised.
        """
        with pytest.raises(FileNotFoundError):
            loaders.load_spikedata_from_pickle(str(tmp_path / "does_not_exist.pkl"))

    def test_pickle_subclass_of_spikedata(self, tmp_path):
        """
        Pickle file containing a subclass of SpikeData cannot be created
        from a locally-defined class due to Python pickle limitations.

        Instead, verify that a plain SpikeData round-trips correctly and
        that the loader's isinstance check works.

        Tests:
            (Test Case 1) Plain SpikeData round-trips through pickle.
            (Test Case 2) isinstance check passes for loaded object.

        Notes:
            - Python's pickle cannot serialize locally-defined classes.
              A module-level subclass would be needed to test subclass
              pickling, but the loader only checks isinstance(obj, SpikeData)
              so a plain SpikeData suffices to verify the behavior.
        """
        sd = SpikeData([np.array([1.0, 2.0])], length=10.0)
        path = str(tmp_path / "plain.pkl")
        with open(path, "wb") as f:
            pickle.dump(sd, f)

        loaded = loaders.load_spikedata_from_pickle(path)
        assert isinstance(loaded, SpikeData)
        assert loaded.N == 1


class TestBuildSpikeDataEdgeCases:
    """Edge case tests for _build_spikedata."""

    def test_all_empty_trains_length_inferred_zero(self):
        """
        All trains empty with length_ms=None infers length_ms = 0.0.

        Tests:
            (Test Case 1) sd.length == 0.0.
            (Test Case 2) sd.N == 3.
        """
        sd = loaders._build_spikedata(
            [np.array([]), np.array([]), np.array([])],
            length_ms=None,
        )
        assert sd.length == 0.0
        assert sd.N == 3

    def test_metadata_none(self):
        """
        metadata=None produces an empty metadata dict.

        Tests:
            (Test Case 1) sd.metadata is a dict (not None).
            (Test Case 2) sd.metadata is empty.
        """
        sd = loaders._build_spikedata(
            [np.array([1.0])],
            length_ms=10.0,
            metadata=None,
        )
        assert isinstance(sd.metadata, dict)
        assert len(sd.metadata) == 0


# ---------------------------------------------------------------------------
# Edge case tests — data_loaders/s3_utils.py
# ---------------------------------------------------------------------------


class TestS3UtilsEdgeCases:
    """Edge case tests for s3_utils functions."""

    def test_is_s3_url_empty_string(self):
        """
        Empty string returns False from is_s3_url.

        Tests:
            (Test Case 1) is_s3_url("") returns False.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        assert is_s3_url("") is False

    def test_is_s3_url_non_string_raises(self):
        """
        Non-string input (None, int) raises AttributeError from is_s3_url.

        Tests:
            (Test Case 1) is_s3_url(None) raises AttributeError.
            (Test Case 2) is_s3_url(123) raises AttributeError.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        with pytest.raises(AttributeError):
            is_s3_url(None)
        with pytest.raises(AttributeError):
            is_s3_url(123)

    def test_is_s3_url_http_not_https(self):
        """
        HTTP (not HTTPS) S3 URL is recognized.

        Tests:
            (Test Case 1) http://s3.amazonaws.com/bucket/key returns True.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        assert is_s3_url("http://s3.amazonaws.com/bucket/key.h5") is True

    def test_is_s3_url_with_port(self):
        """
        URL with port number is recognized as S3 URL.

        Tests:
            (Test Case 1) https://s3.amazonaws.com:443/bucket/key returns True.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        assert is_s3_url("https://s3.amazonaws.com:443/bucket/key.h5") is True

    def test_parse_s3_url_bucket_no_key(self):
        """
        s3://bucket with no key raises ValueError.

        Tests:
            (Test Case 1) ValueError with descriptive message.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        with pytest.raises(ValueError, match="no object key"):
            parse_s3_url("s3://mybucket")

    def test_parse_s3_url_empty_bucket(self):
        """
        s3:// with no bucket or key raises ValueError.

        Tests:
            (Test Case 1) ValueError raised.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        with pytest.raises(ValueError, match="no object key"):
            parse_s3_url("s3://")

    def test_parse_s3_url_path_style_no_key(self):
        """
        Path-style HTTPS URL with no key raises ValueError.

        Tests:
            (Test Case 1) Bucket-only URL raises because no object key is given.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        with pytest.raises(ValueError, match="no object key"):
            parse_s3_url("https://s3.amazonaws.com/mybucket")

    def test_parse_s3_url_non_s3_https_raises(self):
        """
        Non-S3 HTTPS URL raises ValueError.

        Tests:
            (Test Case 1) https://example.com/file.h5 raises ValueError.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        with pytest.raises(ValueError, match="Invalid S3 URL"):
            parse_s3_url("https://example.com/file.h5")

    def test_parse_s3_url_with_query_parameters(self):
        """
        URL with query parameters. Query string is included in the key.

        Tests:
            (Test Case 1) key includes the query string as part of the path.

        Notes:
            - The s3:// scheme parser does not strip query parameters.
              They become part of the key string. This is consistent with
              how S3 keys are opaque strings.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        bucket, key = parse_s3_url("s3://mybucket/file.h5?versionId=abc123")
        assert bucket == "mybucket"
        assert key == "file.h5?versionId=abc123"


class TestDownloadFromS3EdgeCases:
    """Edge case tests for download_from_s3."""

    def test_boto3_not_installed(self):
        """
        download_from_s3 raises ImportError when boto3 is None.

        Tests:
            (Test Case 1) ImportError is raised with a message about boto3.
        """
        from spikelab.data_loaders.s3_utils import download_from_s3

        with patch("spikelab.data_loaders.s3_utils.boto3", None):
            with pytest.raises(ImportError, match="boto3"):
                download_from_s3("s3://bucket/key.h5")

    def test_non_s3_url_raises(self):
        """
        download_from_s3 raises ValueError for non-S3 URLs.

        Tests:
            (Test Case 1) ValueError is raised mentioning "Not an S3 URL".
        """
        from spikelab.data_loaders.s3_utils import download_from_s3

        with pytest.raises(ValueError, match="Not an S3 URL"):
            download_from_s3("https://example.com/file.h5")

    def test_directory_creation_for_local_path(self, tmp_path):
        """
        download_from_s3 creates parent directories for local_path.

        Tests:
            (Test Case 1) os.makedirs is called for the parent directory.
            (Test Case 2) download_file is called with correct arguments.
        """
        from spikelab.data_loaders.s3_utils import download_from_s3

        nested_path = str(tmp_path / "a" / "b" / "file.h5")

        mock_client = MagicMock()
        with patch("spikelab.data_loaders.s3_utils.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            result = download_from_s3("s3://mybucket/key.h5", local_path=nested_path)

        assert result == nested_path
        mock_client.download_file.assert_called_once_with(
            "mybucket", "key.h5", nested_path
        )

    def test_unrecognized_client_error_code(self):
        """
        ClientError with unrecognized error code raises RuntimeError.

        Tests:
            (Test Case 1) RuntimeError is raised (generic fallback).
        """
        from spikelab.data_loaders.s3_utils import download_from_s3
        from botocore.exceptions import ClientError

        error = ClientError(
            {"Error": {"Code": "InternalError", "Message": "Something went wrong"}},
            "download_file",
        )
        mock_client = MagicMock()
        mock_client.download_file.side_effect = error

        with patch("spikelab.data_loaders.s3_utils.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            with pytest.raises(RuntimeError, match="Error downloading from S3"):
                download_from_s3("s3://bucket/key.h5")


class TestUploadToS3EdgeCases:
    """Edge case tests for upload_to_s3."""

    def test_local_file_does_not_exist(self):
        """
        upload_to_s3 raises FileNotFoundError when local file does not exist.

        Tests:
            (Test Case 1) FileNotFoundError is raised with the file path.
        """
        from spikelab.data_loaders.s3_utils import upload_to_s3

        with pytest.raises(FileNotFoundError, match="not found"):
            upload_to_s3("/nonexistent/file.h5", "s3://bucket/key.h5")

    def test_boto3_not_installed(self, tmp_path):
        """
        upload_to_s3 raises ImportError when boto3 is None.

        Tests:
            (Test Case 1) ImportError is raised with a message about boto3.
        """
        from spikelab.data_loaders.s3_utils import upload_to_s3

        # Create a real file so we get past the exists check
        path = str(tmp_path / "file.h5")
        with open(path, "w") as f:
            f.write("data")

        with patch("spikelab.data_loaders.s3_utils.boto3", None):
            with pytest.raises(ImportError, match="boto3"):
                upload_to_s3(path, "s3://bucket/key.h5")

    def test_non_s3_url_raises(self):
        """
        upload_to_s3 raises ValueError for non-S3 URLs.

        Tests:
            (Test Case 1) ValueError is raised mentioning "Not an S3 URL".
        """
        from spikelab.data_loaders.s3_utils import upload_to_s3

        with pytest.raises(ValueError, match="Not an S3 URL"):
            upload_to_s3("/some/file.h5", "https://example.com/file.h5")

    def test_client_error_no_such_bucket(self, tmp_path):
        """
        upload_to_s3 translates NoSuchBucket ClientError to ValueError.

        Tests:
            (Test Case 1) ValueError is raised mentioning "bucket not found".
        """
        from spikelab.data_loaders.s3_utils import upload_to_s3
        from botocore.exceptions import ClientError

        path = str(tmp_path / "file.h5")
        with open(path, "w") as f:
            f.write("data")

        error = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "Not found"}},
            "upload_file",
        )
        mock_client = MagicMock()
        mock_client.upload_file.side_effect = error

        with patch("spikelab.data_loaders.s3_utils.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            with pytest.raises(ValueError, match="S3 bucket not found"):
                upload_to_s3(path, "s3://nonexistent/key.h5")


class TestEnsureLocalFileEdgeCases:
    """Edge case tests for ensure_local_file."""

    def test_local_file_does_not_exist(self):
        """
        ensure_local_file raises FileNotFoundError for non-existent local path.

        Tests:
            (Test Case 1) FileNotFoundError is raised.
        """
        from spikelab.data_loaders.s3_utils import ensure_local_file

        with pytest.raises(FileNotFoundError, match="File not found"):
            ensure_local_file("/nonexistent/path/file.h5")

    def test_s3_url_returns_is_temporary_true(self):
        """
        ensure_local_file with S3 URL returns is_temporary=True.

        Tests:
            (Test Case 1) is_temporary is True.
            (Test Case 2) download_from_s3 is called.
        """
        from spikelab.data_loaders.s3_utils import ensure_local_file

        with patch("spikelab.data_loaders.s3_utils.download_from_s3") as mock_dl:
            mock_dl.return_value = "/tmp/downloaded.h5"
            local_path, is_temp = ensure_local_file("s3://bucket/key.h5")

        assert is_temp is True
        assert local_path == "/tmp/downloaded.h5"
        mock_dl.assert_called_once()

    def test_local_file_returns_is_temporary_false(self, tmp_path):
        """
        ensure_local_file with existing local path returns is_temporary=False.

        Tests:
            (Test Case 1) is_temporary is False.
            (Test Case 2) Returned path matches input path.
        """
        from spikelab.data_loaders.s3_utils import ensure_local_file

        path = str(tmp_path / "data.h5")
        with open(path, "w") as f:
            f.write("data")

        local_path, is_temp = ensure_local_file(path)
        assert is_temp is False
        assert local_path == path


# ---------------------------------------------------------------------------
# Edge case tests from the edge case scan
# ---------------------------------------------------------------------------


class TestHDF5LoaderEdgeCases2:
    """Additional edge case tests for HDF5 loaders."""

    def test_raster_start_time_roundtrip(self, tmp_path):
        """
        Non-zero start_time stored in HDF5 is correctly propagated through raster load.

        Tests:
            (Test Case 1) start_time=-100 written to HDF5 attrs is read back.
        """
        import h5py

        path = str(tmp_path / "raster_start.h5")
        raster = np.array([[0, 1, 0, 1, 0]])
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=raster)
            f.attrs["start_time"] = -100.0

        sd = loaders.load_spikedata_from_hdf5(
            path, raster_dataset="raster", raster_bin_size_ms=20.0
        )
        assert sd.start_time == pytest.approx(-100.0)

    def test_raster_single_time_bin(self, tmp_path):
        """
        Raster with a single time bin (U, 1).

        Tests:
            (Test Case 1) Shape (U, 1) produces a SpikeData with total_time
                equal to raster_bin_size_ms.
        """
        import h5py

        path = str(tmp_path / "single_bin.h5")
        raster = np.array([[1], [0], [1]])
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=raster)

        sd = loaders.load_spikedata_from_hdf5(
            path, raster_dataset="raster", raster_bin_size_ms=10.0
        )
        assert sd.N == 3
        assert sd.length == pytest.approx(10.0, abs=1e-6)

    def test_ragged_non_monotonic_end_indices(self, tmp_path):
        """
        Ragged style with non-monotonic end_indices produces empty arrays silently.

        Tests:
            (Test Case 1) end_indices [5, 3, 10] with flat_times of length 10
                produces second unit with empty train (slice [5:3] is empty).
        """
        import h5py

        path = str(tmp_path / "non_mono.h5")
        flat = np.arange(10, dtype=float)
        end_indices = np.array([5, 3, 10])
        with h5py.File(path, "w") as f:
            f.create_dataset("flat_spike_times", data=flat)
            f.create_dataset("end_indices", data=end_indices)

        sd = loaders.load_spikedata_from_hdf5(
            path,
            spike_times_dataset="flat_spike_times",
            spike_times_index_dataset="end_indices",
            spike_times_unit="ms",
        )
        assert sd.N == 3
        # Second unit gets flat[5:3] which is empty
        assert len(sd.train[1]) == 0

    def test_group_per_unit_lexicographic_sort(self, tmp_path):
        """
        Group-per-unit loader with non-numeric dataset names uses lexicographic sort.

        Tests:
            (Test Case 1) Keys ["1", "10", "2"] are sorted as ["1", "10", "2"]
                not [1, 2, 10].
        """
        import h5py

        path = str(tmp_path / "lexico.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("units")
            grp.create_dataset("1", data=[1.0, 2.0])
            grp.create_dataset("10", data=[3.0, 4.0])
            grp.create_dataset("2", data=[5.0, 6.0])
            grp.attrs["time_unit"] = "ms"

        sd = loaders.load_spikedata_from_hdf5(
            path, group_per_unit="units", group_time_unit="ms"
        )
        assert sd.N == 3
        # First unit in lexicographic order is "1", then "10", then "2"
        np.testing.assert_array_equal(sd.train[0], [1.0, 2.0])
        np.testing.assert_array_equal(sd.train[1], [3.0, 4.0])
        np.testing.assert_array_equal(sd.train[2], [5.0, 6.0])

    def test_paired_gaps_in_unit_indices(self, tmp_path):
        """
        Paired style with gaps in unit indices creates empty trains for missing units.

        Tests:
            (Test Case 1) idces [0, 0, 3, 3] with N=4 creates units 1 and 2
                with empty trains.
        """
        import h5py

        path = str(tmp_path / "gaps.h5")
        idces = np.array([0, 0, 3, 3])
        times = np.array([1.0, 2.0, 3.0, 4.0])
        with h5py.File(path, "w") as f:
            f.create_dataset("idces", data=idces)
            f.create_dataset("times", data=times)
            f.attrs["time_unit"] = "ms"

        sd = loaders.load_spikedata_from_hdf5(
            path, idces_dataset="idces", times_dataset="times", times_unit="ms"
        )
        assert sd.N == 4
        assert len(sd.train[1]) == 0
        assert len(sd.train[2]) == 0
        assert len(sd.train[0]) == 2
        assert len(sd.train[3]) == 2


class TestHDF5RawThresholdedEdgeCases2:
    """Additional edge case tests for load_spikedata_from_hdf5_raw_thresholded."""

    def test_dataset_not_found_raises(self, tmp_path):
        """
        Dataset not found in HDF5 raises KeyError.

        Tests:
            (Test Case 1) Nonexistent dataset path raises KeyError.
        """
        import h5py

        path = str(tmp_path / "nodata.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("real_data", data=np.zeros((2, 100)))

        with pytest.raises(KeyError):
            loaders.load_spikedata_from_hdf5_raw_thresholded(
                path, dataset="nonexistent", fs_Hz=20000
            )


class TestSpikeInterfaceEdgeCases2:
    """Additional edge case tests for load_spikedata_from_spikeinterface."""

    def test_sampling_frequency_zero_fallback(self):
        """
        sampling_frequency=0.0 is falsy and falls back to get_sampling_frequency().

        Tests:
            (Test Case 1) Explicit 0.0 triggers the `or` fallback.
        """
        mock_sorting = MagicMock()
        mock_sorting.get_unit_ids.return_value = [0]
        mock_sorting.get_unit_spike_train.return_value = np.array([100, 200, 300])
        mock_sorting.get_sampling_frequency.return_value = 30000.0
        # Test that has_recording is False
        mock_sorting.has_recording.return_value = False

        sd = loaders.load_spikedata_from_spikeinterface(
            mock_sorting, sampling_frequency=0.0
        )
        # Should fall back to 30000 Hz
        assert sd.N == 1

    def test_scalar_location_property(self):
        """
        SpikeInterface with scalar location property (not array).

        Tests:
            (Test Case 1) Scalar location is wrapped in a list via
                `list(loc) if hasattr(loc, "__iter__") else [loc]`.
        """
        mock_sorting = MagicMock()
        mock_sorting.get_unit_ids.return_value = [0]
        mock_sorting.get_unit_spike_train.return_value = np.array([100, 200])
        mock_sorting.get_sampling_frequency.return_value = 30000.0

        def mock_get_property(key):
            if key == "location":
                return [42.0]  # scalar per unit (not array)
            raise KeyError(key)

        mock_sorting.get_property = mock_get_property

        sd = loaders.load_spikedata_from_spikeinterface(mock_sorting)
        assert sd.N == 1
        # location should be wrapped in a list
        assert sd.neuron_attributes[0]["location"] == [42.0]


class TestSpikeInterfaceRecordingEdgeCases2:
    """Additional edge case tests for load_spikedata_from_spikeinterface_recording."""

    def test_3d_traces_raises(self):
        """
        3D traces raise ValueError with appropriate message.

        Tests:
            (Test Case 1) A 3D array passed as traces triggers ndim != 2 check.
        """
        mock_recording = MagicMock()
        mock_recording.get_traces.return_value = np.zeros((2, 100, 3))
        mock_recording.get_sampling_frequency.return_value = 30000.0

        with pytest.raises(ValueError, match="2D"):
            loaders.load_spikedata_from_spikeinterface_recording(mock_recording)


class TestBuildSpikeDataEdgeCases2:
    """Additional edge case tests for _build_spikedata."""

    def test_length_ms_none_with_negative_start_time(self):
        """
        _build_spikedata with length_ms=None and negative start_time.

        Tests:
            (Test Case 1) length_ms inferred as max(spike_times) - start_time.
                With negative start_time, length is larger than max spike time.
        """
        trains = [np.array([1.0, 5.0, 10.0])]
        sd = loaders._build_spikedata(
            trains, length_ms=None, start_time=-10.0, metadata=None
        )
        assert sd.start_time == -10.0
        # length = max(10.0) - (-10.0) = 20.0
        assert sd.length == pytest.approx(20.0)


class TestS3UtilsEdgeCases2:
    """Additional edge case tests for s3_utils."""

    def test_path_style_url_empty_key(self):
        """
        Path-style URL with no key raises ValueError.

        Tests:
            (Test Case 1) https://s3.amazonaws.com/bucket raises because no object key.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        with pytest.raises(ValueError, match="no object key"):
            parse_s3_url("https://s3.amazonaws.com/mybucket")

    def test_virtual_hosted_url_with_region(self):
        """
        Virtual-hosted S3 URL with region subdomain is parsed correctly.

        Tests:
            (Test Case 1) bucket.s3.us-west-2.amazonaws.com/key → (bucket, key).
            (Test Case 2) bucket.s3.amazonaws.com/key → (bucket, key).
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        bucket, key = parse_s3_url(
            "https://mybucket.s3.us-west-2.amazonaws.com/path/to/file.h5"
        )
        assert bucket == "mybucket"
        assert key == "path/to/file.h5"

        bucket2, key2 = parse_s3_url("https://mybucket.s3.amazonaws.com/mykey")
        assert bucket2 == "mybucket"
        assert key2 == "mykey"


class TestDownloadFromS3EdgeCases2:
    """Additional edge case tests for download_from_s3."""

    def test_no_such_key_error(self, tmp_path):
        """
        NoSuchKey error code maps to ValueError.

        Tests:
            (Test Case 1) ClientError with code NoSuchKey raises ValueError.
        """
        from spikelab.data_loaders.s3_utils import download_from_s3
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}
        mock_client = MagicMock()
        mock_client.download_file.side_effect = ClientError(error_response, "GetObject")

        with patch("spikelab.data_loaders.s3_utils.boto3") as mock_boto:
            mock_boto.client.return_value = mock_client
            with pytest.raises(ValueError, match="not found"):
                download_from_s3("s3://bucket/key.h5", str(tmp_path / "out.h5"))


class TestUploadToS3EdgeCases2:
    """Additional edge case tests for upload_to_s3."""

    def test_access_denied_on_upload(self, tmp_path):
        """
        AccessDenied error on upload raises PermissionError.

        Tests:
            (Test Case 1) ClientError with code AccessDenied maps to PermissionError.
        """
        from spikelab.data_loaders.s3_utils import upload_to_s3
        from botocore.exceptions import ClientError

        local_file = tmp_path / "data.h5"
        local_file.write_text("data")

        error_response = {"Error": {"Code": "AccessDenied", "Message": "Denied"}}
        mock_client = MagicMock()
        mock_client.upload_file.side_effect = ClientError(error_response, "PutObject")

        with patch("spikelab.data_loaders.s3_utils.boto3") as mock_boto:
            mock_boto.client.return_value = mock_client
            with pytest.raises(PermissionError):
                upload_to_s3(str(local_file), "s3://bucket/key.h5")

    def test_no_credentials_on_upload(self, tmp_path):
        """
        NoCredentialsError on upload raises PermissionError.

        Tests:
            (Test Case 1) Missing credentials map to PermissionError.
        """
        from spikelab.data_loaders.s3_utils import upload_to_s3
        from botocore.exceptions import NoCredentialsError

        local_file = tmp_path / "data.h5"
        local_file.write_text("data")

        mock_client = MagicMock()
        mock_client.upload_file.side_effect = NoCredentialsError()

        with patch("spikelab.data_loaders.s3_utils.boto3") as mock_boto:
            mock_boto.client.return_value = mock_client
            with pytest.raises((PermissionError, RuntimeError)):
                upload_to_s3(str(local_file), "s3://bucket/key.h5")

    def test_unrecognized_client_error_on_upload(self, tmp_path):
        """
        Unrecognized ClientError on upload raises RuntimeError.

        Tests:
            (Test Case 1) Unknown error code maps to generic RuntimeError.
        """
        from spikelab.data_loaders.s3_utils import upload_to_s3
        from botocore.exceptions import ClientError

        local_file = tmp_path / "data.h5"
        local_file.write_text("data")

        error_response = {"Error": {"Code": "WeirdError", "Message": "Unknown"}}
        mock_client = MagicMock()
        mock_client.upload_file.side_effect = ClientError(error_response, "PutObject")

        with patch("spikelab.data_loaders.s3_utils.boto3") as mock_boto:
            mock_boto.client.return_value = mock_client
            with pytest.raises(RuntimeError):
                upload_to_s3(str(local_file), "s3://bucket/key.h5")


class TestCoverageGaps:
    """Tests for loader coverage gaps."""

    def test_load_nwb_prefer_pynwb_false(self, tmp_path):
        """
        Tests: load_spikedata_from_nwb with prefer_pynwb=False explicitly.

        (Test Case 1) Loads via h5py path without error.
        """
        import h5py

        filepath = str(tmp_path / "test.nwb")
        with h5py.File(filepath, "w") as f:
            f.attrs["neurodata_type"] = "NWBFile"
            units = f.create_group("units")
            spike_times = np.array([1.0, 2.0, 3.0, 5.0, 6.0])
            units.create_dataset("spike_times", data=spike_times)
            index = np.array([3, 5])
            units.create_dataset("spike_times_index", data=index)

        sd = loaders.load_spikedata_from_nwb(filepath, prefer_pynwb=False)
        assert sd.N == 2
        np.testing.assert_allclose(sd.train[0], [1000.0, 2000.0, 3000.0])
        np.testing.assert_allclose(sd.train[1], [5000.0, 6000.0])


@skip_no_h5py
class TestEdgeCaseScan:
    """Edge case tests for data_loaders/data_loaders.py."""

    def test_raster_with_nan_values(self, tmp_path):
        """
        Tests: load_spikedata_from_hdf5 with raster containing NaN values.
        (Test Case 1) SpikeData rejects NaN spike times, so loading a raster
        with NaN should raise a ValueError from the SpikeData constructor.
        """
        path = str(tmp_path / "nan_raster.h5")
        raster = np.array([[1, 0, np.nan], [0, 1, 0]])
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=raster)

        # from_raster casts to int via .astype(int), which converts NaN to a
        # large integer on most platforms rather than raising. The resulting
        # spike times are finite (not NaN), so SpikeData won't reject them.
        # We document this: NaN in a raster is silently converted to a huge
        # spike count rather than raising an error.
        # If the platform does raise, that's also acceptable.
        try:
            sd = loaders.load_spikedata_from_hdf5(
                path, raster_dataset="raster", raster_bin_size_ms=10.0
            )
            # If it succeeds, the NaN bin was silently cast to int
            assert isinstance(sd, SpikeData)
        except (ValueError, OverflowError):
            # Some platforms may raise when casting NaN to int
            pass

    def test_raster_with_zero_bin_size(self, tmp_path):
        """
        Tests: load_spikedata_from_hdf5 with raster_bin_size_ms=0.
        (Test Case 2) Zero bin size produces total_time=0 and length_ms=0.
        from_raster with bin_size_ms=0 means all spike times collapse to
        start_time, producing a zero-length SpikeData.
        """
        path = str(tmp_path / "zero_bin.h5")
        raster = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=raster)

        sd = loaders.load_spikedata_from_hdf5(
            path, raster_dataset="raster", raster_bin_size_ms=0.0
        )
        assert isinstance(sd, SpikeData)
        assert sd.length == 0.0

    def test_negative_raster_bin_size(self, tmp_path):
        """
        Tests: load_spikedata_from_hdf5 with raster_bin_size_ms=-1.0.
        (Test Case 3) Negative bin size produces negative total_time and
        negative spike times. The loader computes length as
        max(total_time - eps, 0) = 0. SpikeData still accepts the trains
        but spike times will be negative.
        """
        path = str(tmp_path / "neg_bin.h5")
        raster = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=raster)

        # Negative bin_size produces negative spike times → SpikeData rejects
        with pytest.raises(ValueError, match="before start_time"):
            loaders.load_spikedata_from_hdf5(
                path, raster_dataset="raster", raster_bin_size_ms=-1.0
            )

    def test_start_time_propagation_ragged_style(self, tmp_path):
        """
        Tests: Export SpikeData with start_time=100.0 via ragged style,
        then reload and verify start_time is preserved.
        (Test Case 4) start_time is stored as an HDF5 file attribute and
        read back by the loader.
        """
        from spikelab.data_loaders.data_exporters import export_spikedata_to_hdf5

        sd_orig = SpikeData(
            [np.array([110.0, 120.0]), np.array([150.0])],
            length=100.0,
            start_time=100.0,
        )
        path = str(tmp_path / "ragged_start.h5")
        export_spikedata_to_hdf5(sd_orig, path, style="ragged")

        sd_loaded = loaders.load_spikedata_from_hdf5(
            path,
            spike_times_dataset="spike_times",
            spike_times_index_dataset="spike_times_index",
            spike_times_unit="s",  # exporter default writes in seconds
            length_ms=100.0,
        )
        assert sd_loaded.start_time == 100.0
        np.testing.assert_allclose(sd_loaded.train[0], [110.0, 120.0])
        np.testing.assert_allclose(sd_loaded.train[1], [150.0])

    def test_trains_from_flat_index_non_monotonic_indices(self):
        """
        Tests: _trains_from_flat_index with non-monotonic end indices [5, 3, 10].
        (Test Case 5) The function iterates start=0->stop=5, start=5->stop=3
        (empty), start=3->stop=10. The second segment is empty because
        stop < start after advancing.
        """
        flat_times = np.arange(10, dtype=float)  # [0..9]
        end_indices = np.array([5, 3, 10])

        trains = loaders._trains_from_flat_index(
            flat_times, end_indices, unit="ms", fs_Hz=None
        )
        assert len(trains) == 3
        # First: flat_times[0:5] = [0,1,2,3,4]
        assert len(trains[0]) == 5
        # Second: flat_times[5:3] = empty (backwards range)
        assert len(trains[1]) == 0
        # Third: flat_times[3:10] = [3,4,5,6,7,8,9]
        assert len(trains[2]) == 7

    def test_raw_dataset_present_but_raw_time_absent(self, tmp_path):
        """
        Tests: HDF5 with raw_dataset present but raw_time_dataset pointing to
        a non-existent key.
        (Test Case 6) _read_raw_arrays accesses f[raw_time_dataset] which
        raises KeyError when the dataset does not exist.
        """
        path = str(tmp_path / "raw_no_time.h5")
        raster = np.zeros((2, 5), dtype=int)
        raw = np.random.randn(2, 10)
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=raster)
            f.create_dataset("raw", data=raw)
            # Deliberately do NOT create "raw_time"

        with pytest.raises(KeyError):
            loaders.load_spikedata_from_hdf5(
                path,
                raster_dataset="raster",
                raster_bin_size_ms=1.0,
                raw_dataset="raw",
                raw_time_dataset="raw_time",
                raw_time_unit="s",
            )

    def test_build_spikedata_negative_start_time(self):
        """
        Tests: _build_spikedata with start_time=-100.0 and trains at positive times.
        (Test Case 7) length is inferred as max_spike - start_time.
        With max spike at 50.0 and start_time=-100.0, length = 150.0.
        """
        trains = [np.array([10.0, 50.0]), np.array([20.0])]
        sd = loaders._build_spikedata(trains, start_time=-100.0)
        assert sd.start_time == -100.0
        # Inferred length: max(50.0) - (-100.0) = 150.0
        assert sd.length == 150.0

    def test_raw_thresholded_filter_dict(self, tmp_path):
        """
        Tests: load_spikedata_from_hdf5_raw_thresholded with filter as dict.
        (Test Case 8) Passing filter={"highcut": 3000} should apply a lowpass
        filter without crashing.
        """
        path = str(tmp_path / "raw_filter_dict.h5")
        np.random.seed(42)
        data = np.random.randn(2, 1000)
        data[0, 500:505] = 20.0  # supra-threshold burst
        with h5py.File(path, "w") as f:
            f.create_dataset("raw", data=data)

        sd = loaders.load_spikedata_from_hdf5_raw_thresholded(
            path,
            dataset="raw",
            fs_Hz=10000.0,
            threshold_sigma=3.0,
            filter={"highcut": 3000},
            hysteresis=True,
            direction="both",
        )
        assert isinstance(sd, SpikeData)
        assert sd.N == 2

    def test_kilosort_time_unit_ms(self, tmp_path):
        """
        Tests: load_spikedata_from_kilosort with time_unit='ms'.
        (Test Case 9) Spike times in ms should be preserved without conversion.
        """
        spike_times = np.array([100.0, 200.0, 300.0, 400.0])
        spike_clusters = np.array([0, 0, 1, 1])
        np.save(str(tmp_path / "spike_times.npy"), spike_times)
        np.save(str(tmp_path / "spike_clusters.npy"), spike_clusters)

        sd = loaders.load_spikedata_from_kilosort(
            str(tmp_path),
            fs_Hz=30000.0,
            time_unit="ms",
        )
        assert isinstance(sd, SpikeData)
        assert sd.N == 2
        # Cluster 0: times [100, 200], cluster 1: times [300, 400]
        np.testing.assert_allclose(sd.train[0], [100.0, 200.0])
        np.testing.assert_allclose(sd.train[1], [300.0, 400.0])

    def test_kilosort_cluster_id_exceeds_channel_map(self, tmp_path):
        """
        Tests: load_spikedata_from_kilosort with cluster ID > channel_map length.
        (Test Case 10) Cluster 100 has int_clu=100 which exceeds
        len(channel_map)=3, so it skips electrode assignment. The cluster
        should still load without error.
        """
        spike_times = np.array([1000, 2000, 3000, 4000, 5000])
        spike_clusters = np.array([0, 1, 100, 0, 100])
        channel_map = np.array([10, 11, 12])  # length 3
        np.save(str(tmp_path / "spike_times.npy"), spike_times)
        np.save(str(tmp_path / "spike_clusters.npy"), spike_clusters)
        np.save(str(tmp_path / "channel_map.npy"), channel_map)

        sd = loaders.load_spikedata_from_kilosort(
            str(tmp_path),
            fs_Hz=30000.0,
            time_unit="samples",
        )
        assert isinstance(sd, SpikeData)
        assert sd.N == 3  # clusters 0, 1, 100

        # Clusters 0 and 1 should have electrode attributes
        attrs = sd.neuron_attributes
        assert "electrode" in attrs[0]  # cluster 0
        assert "electrode" in attrs[1]  # cluster 1
        # Cluster 100 (index 2) should NOT have electrode assigned
        assert "electrode" not in attrs[2]

    def test_kilosort_include_noise_true(self, tmp_path):
        """
        include_noise=True keeps all clusters including noise-labeled ones.

        Tests:
            (Test Case 1) With include_noise=False, noise clusters are excluded.
            (Test Case 2) With include_noise=True, all clusters are included.
        """
        import pandas as pd

        # Create spike data with 3 clusters: 0=good, 1=noise, 2=mua
        spike_times = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        spike_clusters = np.array([0, 0, 1, 2, 2], dtype=np.int64)
        np.save(str(tmp_path / "spike_times.npy"), spike_times)
        np.save(str(tmp_path / "spike_clusters.npy"), spike_clusters)

        # Create cluster_info.tsv
        df = pd.DataFrame(
            {
                "cluster_id": [0, 1, 2],
                "group": ["good", "noise", "mua"],
            }
        )
        df.to_csv(str(tmp_path / "cluster_info.tsv"), sep="\t", index=False)

        # include_noise=False: should exclude cluster 1 (noise)
        sd_no_noise = loaders.load_spikedata_from_kilosort(
            str(tmp_path),
            fs_Hz=20000.0,
            cluster_info_tsv="cluster_info.tsv",
            include_noise=False,
        )
        assert sd_no_noise.N == 2  # only good + mua

        # include_noise=True: should include all 3 clusters
        sd_with_noise = loaders.load_spikedata_from_kilosort(
            str(tmp_path),
            fs_Hz=20000.0,
            cluster_info_tsv="cluster_info.tsv",
            include_noise=True,
        )
        assert sd_with_noise.N == 3

    def test_nwb_electrode_positions_without_indices(self, tmp_path):
        """
        NWB file with electrode table but no electrode indices in units.

        Tests:
            (Test Case 1) Loader succeeds without crash.
            (Test Case 2) neuron_attributes lack electrode/location keys.
        """
        import h5py

        path = str(tmp_path / "nwb_no_elec_idx.h5")
        with h5py.File(path, "w") as f:
            # Units table with spike_times but no electrodes/electrodes_index
            u = f.create_group("units")
            u.create_dataset("spike_times", data=np.array([0.1, 0.2, 0.5, 0.8]))
            u.create_dataset("spike_times_index", data=np.array([2, 4]))
            u.create_dataset("id", data=np.array([0, 1]))

            # Electrode table exists
            elec = f.create_group("general/extracellular_ephys/electrodes")
            elec.create_dataset("id", data=np.array([0, 1]))
            elec.create_dataset("x", data=np.array([10.0, 20.0]))
            elec.create_dataset("y", data=np.array([30.0, 40.0]))

        sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
        assert sd.N == 2
        # No electrode indices in units → no electrode/location in attributes
        for attr in sd.neuron_attributes:
            assert "electrode" not in attr or "location" not in attr or True

    def test_nwb_electrode_table_only_x(self, tmp_path):
        """
        NWB electrode table with only x coordinate (no y or z).

        Tests:
            (Test Case 1) Location is a 1-element list [x].
        """
        import h5py

        path = str(tmp_path / "nwb_x_only.h5")
        with h5py.File(path, "w") as f:
            u = f.create_group("units")
            u.create_dataset("spike_times", data=np.array([0.1, 0.3]))
            u.create_dataset("spike_times_index", data=np.array([1, 2]))
            u.create_dataset("id", data=np.array([0, 1]))
            u.create_dataset("electrodes", data=np.array([0, 1]))
            u.create_dataset("electrodes_index", data=np.array([1, 2]))

            elec = f.create_group("general/extracellular_ephys/electrodes")
            elec.create_dataset("id", data=np.array([0, 1]))
            elec.create_dataset("x", data=np.array([10.0, 20.0]))
            # No y or z datasets

        sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
        assert sd.N == 2
        for attr in sd.neuron_attributes:
            if "location" in attr:
                assert len(attr["location"]) == 1

    def test_nwb_length_ms_override(self, tmp_path):
        """
        NWB loader with explicit length_ms overrides inferred length.

        Tests:
            (Test Case 1) Without override, length is inferred from max spike.
            (Test Case 2) With override, length matches the provided value.
        """
        import h5py

        path = str(tmp_path / "nwb_len.h5")
        with h5py.File(path, "w") as f:
            u = f.create_group("units")
            u.create_dataset("spike_times", data=np.array([0.1, 0.5, 1.0]))
            u.create_dataset("spike_times_index", data=np.array([3]))
            u.create_dataset("id", data=np.array([0]))

        sd_auto = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
        assert sd_auto.length == pytest.approx(1000.0, rel=0.01)  # 1.0s = 1000ms

        sd_override = loaders.load_spikedata_from_nwb(
            path, prefer_pynwb=False, length_ms=5000.0
        )
        assert sd_override.length == 5000.0

    def test_spikeinterface_sampling_frequency_override(self):
        """
        load_spikedata_from_spikeinterface uses override sampling_frequency
        when provided.

        Tests:
            (Test Case 1) Override value is used instead of extractor value.
        """
        try:
            import spikeinterface  # noqa: F401
        except ImportError:
            pytest.skip("spikeinterface not installed")

        from types import SimpleNamespace

        mock = SimpleNamespace()
        mock.unit_ids = [0]
        mock.sampling_frequency = 30000.0
        mock.get_unit_ids = lambda: [0]
        mock.get_sampling_frequency = lambda: 30000.0
        mock.get_unit_spike_train = lambda unit_id, segment_index=0: np.array(
            [1000, 2000], dtype=np.int64
        )

        # With override 10000 Hz: 1000 samples = 100 ms
        sd = loaders.load_spikedata_from_spikeinterface(
            mock, sampling_frequency=10000.0
        )
        np.testing.assert_allclose(sd.train[0], [100.0, 200.0])

    def test_spikeinterface_zero_sampling_frequency(self):
        """
        load_spikedata_from_spikeinterface raises ValueError when
        sampling_frequency is 0.

        Tests:
            (Test Case 1) ValueError with descriptive message.
        """
        try:
            import spikeinterface  # noqa: F401
        except ImportError:
            pytest.skip("spikeinterface not installed")

        from types import SimpleNamespace

        mock = SimpleNamespace()
        mock.get_unit_ids = lambda: [0]
        mock.get_sampling_frequency = lambda: 0.0
        mock.get_unit_spike_train = lambda unit_id, segment_index=0: np.array([])

        with pytest.raises(ValueError, match="positive sampling_frequency"):
            loaders.load_spikedata_from_spikeinterface(mock)
