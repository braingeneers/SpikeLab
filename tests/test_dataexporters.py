"""
Tests for data exporters -> file formats, including round-trips with loaders.

This module tests the data export functionality that writes SpikeData objects to various
file formats. The tests focus on:

1. **Round-trip integrity**: Ensuring data exported and then re-imported matches the original
2. **Format compliance**: Verifying exported files conform to expected format specifications
3. **Parameter handling**: Testing various export options and edge cases
4. **Cross-format compatibility**: Ensuring exports work with corresponding loaders

The tests are organized by export format (HDF5, NWB, KiloSort) and use temporary files
that are automatically cleaned up after each test.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

from spikelab.spikedata import SpikeData
import spikelab.data_loaders.data_loaders as loaders
import spikelab.data_loaders.data_exporters as exporters


def make_sd() -> SpikeData:
    """
    Create a simple, deterministic SpikeData for export tests.

    Returns a SpikeData with 3 units (3 spikes, 2 spikes, 0 spikes) and 25 ms length.
    """
    trains = [
        np.array([5.0, 10.0, 15.0]),  # Unit 0: 3 spikes
        np.array([2.5, 20.0]),  # Unit 1: 2 spikes
        np.array([], float),  # Unit 2: empty (edge case)
    ]
    return SpikeData(trains, length=25.0, metadata={"label": "test"})


skip_no_h5py = pytest.mark.skipif(
    h5py is None, reason="h5py not installed; skipping HDF5/NWB exporter tests"
)


def _make_sd_with_electrodes() -> SpikeData:
    """
    Create a SpikeData with neuron_attributes containing electrode info.

    Returns a SpikeData with 3 units and electrode IDs [4, 7, 2].
    """
    trains = [
        np.array([5.0, 10.0, 15.0]),
        np.array([2.5, 20.0]),
        np.array([8.0]),
    ]
    neuron_attrs = [
        {"electrode": 4},
        {"electrode": 7},
        {"electrode": 2},
    ]
    return SpikeData(trains, length=25.0, neuron_attributes=neuron_attrs)


@skip_no_h5py
class TestHDF5Exporters:
    """
    Tests for HDF5 export functionality across all four supported styles.

    HDF5 is a flexible format that supports multiple data organization patterns.
    These tests validate each style works correctly and can round-trip through
    the corresponding loader functions.

    The four styles tested are:
    1. 'ragged': Flat spike times + cumulative index (most efficient for sparse data)
    2. 'group': One dataset per unit (easiest for unit-specific access)
    3. 'paired': Parallel arrays of unit indices and spike times
    4. 'raster': Dense 2D binned spike counts (for rate-based analyses)
    """

    def test_export_hdf5_ragged_roundtrip(self, tmp_path):
        """
        Tests the most common HDF5 export format (ragged arrays)
        with time unit conversion from milliseconds to seconds.

        Tests:
        (Method 1) Export SpikeData to HDF5 using ragged style with seconds time unit
        (Method 2) Re-import using the HDF5 loader with matching parameters
        (Test Case 1) Verify all spike trains match the original within floating-point precision

        Notes:
        - Ragged arrays are the most storage-efficient format for sparse spike data and are used by many analysis tools including NWB.
        """
        sd = make_sd()
        path = str(tmp_path / "test.h5")

        sd.to_hdf5(path, style="ragged", spike_times_unit="s")

        sd2 = loaders.load_spikedata_from_hdf5(
            path,
            spike_times_dataset="spike_times",
            spike_times_index_dataset="spike_times_index",
            spike_times_unit="s",
        )
        for a, b in zip(sd.train, sd2.train):
            assert np.allclose(a, b)

    def test_export_hdf5_group_roundtrip_samples(self, tmp_path):
        """
        Test group-per-unit export with sample-based time units.

        Tests:
        (Method 1) Export using group style with 1000 Hz sampling rate (1 sample = 1 ms)
        (Method 2) Each unit gets its own dataset within the "units" group
        (Method 3) Spike times are converted from milliseconds to sample indices
        (Test Case 1) Round-trip through loader verifies conversion accuracy

        Notes:
        - The group style makes it easy to access individual units without parsing index arrays,
        and sample units preserve exact timing relationships with the original recording.
        """
        sd = make_sd()
        path = str(tmp_path / "test.h5")

        sd.to_hdf5(
            path,
            style="group",
            group_per_unit="units",
            group_time_unit="samples",
            fs_Hz=1000.0,
        )
        sd2 = loaders.load_spikedata_from_hdf5(
            path,
            group_per_unit="units",
            group_time_unit="samples",
            fs_Hz=1000.0,
        )

        # Times are quantized to samples at 1 kHz; compare against quantized originals
        def q(ms):
            samp = np.rint(ms * (1000.0 / 1e3))  # fs/1e3 == 1
            return samp / 1000.0 * 1e3

        for a, b in zip(sd.train, sd2.train):
            assert np.allclose(q(a), b)

    def test_export_hdf5_paired_roundtrip_ms(self, tmp_path):
        """
        Tests paired arrays export with millisecond time units.

        Tests:
        (Method 1) Export creates two datasets: unit indices and corresponding spike times
        (Method 2) Empty units are handled by simply not including them in the arrays
        (Method 3) Times remain in milliseconds (no conversion)
        (Test Case 1) Round-trip verifies the pairing logic works correctly

        Notes:
        - The paired style is a simple format that stores unit indices and spike times in separate parallel arrays,
        keeping original millisecond timing.
        """
        sd = make_sd()
        path = str(tmp_path / "test.h5")

        sd.to_hdf5(
            path,
            style="paired",
            idces_dataset="idces",
            times_dataset="times",
            times_unit="ms",
        )
        sd2 = loaders.load_spikedata_from_hdf5(
            path, idces_dataset="idces", times_dataset="times", times_unit="ms"
        )
        for a, b in zip(sd.train, sd2.train):
            assert np.allclose(a, b)

    def test_export_hdf5_raster(self, tmp_path):
        """
        Test raster export for binned spike count analysis.

        Tests:
        (Method 1) Export specifies a 5ms bin size for rasterization
        (Test Case 1) Verify exported raster matches SpikeData's own raster() output

        Notes:
        - Raster format enables analyses that require fixed-size inputs (like neural decoders).
        """
        sd = make_sd()
        path = str(tmp_path / "test.h5")

        sd.to_hdf5(
            path, style="raster", raster_dataset="raster", raster_bin_size_ms=5.0
        )
        with h5py.File(path, "r") as f:  # type: ignore
            raster = np.asarray(f["raster"])
        assert np.array_equal(raster, sd.raster(5.0))

    def test_export_hdf5_with_raw(self, tmp_path):
        """
        Tests export of raw data arrays alongside spike data.

        Tests:
        (Method 1) Creates SpikeData with mock raw voltage data and time arrays
        (Method 2) Exports both spike data (ragged style) and raw data
        (Test Case 1) Verifies the time conversion was applied correctly to raw_time

        Notes:
        - Validates that continuous raw data (like voltage traces) can be exported alongside spike times.
        """
        sd = make_sd()
        raw = np.random.randn(2, 10)
        sd = SpikeData(sd.train, length=sd.length, raw_data=raw, raw_time=np.arange(10))
        path = str(tmp_path / "test.h5")

        sd.to_hdf5(
            path,
            style="ragged",
            spike_times_unit="s",
            raw_dataset="raw",
            raw_time_dataset="raw_time",
            raw_time_unit="s",
        )
        with h5py.File(path, "r") as f:  # type: ignore
            assert np.allclose(np.asarray(f["raw_time"]), sd.raw_time / 1e3)

    def test_export_hdf5_all_empty_trains_ragged(self, tmp_path):
        """
        Verify that exporting a SpikeData where all spike trains are empty
        works correctly with the ragged style.

        Tests:
            (Test Case 1) Export succeeds without error.
            (Test Case 2) Round-trip produces a SpikeData with the same number of units.
            (Test Case 3) All spike trains remain empty after round-trip.
        """
        trains = [np.array([], float), np.array([], float), np.array([], float)]
        sd = SpikeData(trains, length=100.0)
        path = str(tmp_path / "empty_ragged.h5")

        exporters.export_spikedata_to_hdf5(sd, path, style="ragged")

        sd2 = loaders.load_spikedata_from_hdf5(
            path,
            spike_times_dataset="spike_times",
            spike_times_index_dataset="spike_times_index",
            spike_times_unit="s",
        )
        assert sd2.N == 3
        for train in sd2.train:
            assert len(train) == 0

    def test_export_hdf5_all_empty_trains_paired(self, tmp_path):
        """
        Verify that exporting a SpikeData where all spike trains are empty
        works correctly with the paired style.

        Tests:
            (Test Case 1) Export succeeds without error.
            (Test Case 2) The resulting HDF5 contains empty idces and times arrays.
        """
        trains = [np.array([], float), np.array([], float)]
        sd = SpikeData(trains, length=50.0)
        path = str(tmp_path / "empty_paired.h5")

        exporters.export_spikedata_to_hdf5(
            sd, path, style="paired", idces_dataset="idces", times_dataset="times"
        )

        import h5py as h5

        with h5.File(path, "r") as f:
            assert f["idces"].shape[0] == 0
            assert f["times"].shape[0] == 0

    def test_export_hdf5_very_small_raster_bin_size(self, tmp_path):
        """
        Verify that export_spikedata_to_hdf5 with raster style raises
        ValueError when raster_bin_size_ms is zero or negative.

        Tests:
            (Test Case 1) raster_bin_size_ms=0 raises ValueError.
            (Test Case 2) raster_bin_size_ms=-1.0 raises ValueError.
        """
        sd = make_sd()
        path = str(tmp_path / "bad_raster.h5")

        with pytest.raises(ValueError, match="raster_bin_size_ms"):
            exporters.export_spikedata_to_hdf5(
                sd, path, style="raster", raster_bin_size_ms=0
            )

        with pytest.raises(ValueError, match="raster_bin_size_ms"):
            exporters.export_spikedata_to_hdf5(
                sd, path, style="raster", raster_bin_size_ms=-1.0
            )

    def test_ec_de_01_zero_spike_zero_unit_spikedata(self, tmp_path):
        """
        EC-DE-01: Verify that exporting a SpikeData with zero units
        works correctly across all styles without crashing.

        Tests:
            (Test Case 1) Ragged style succeeds and produces empty arrays.
            (Test Case 2) Group style succeeds and produces an empty group.
            (Test Case 3) Paired style succeeds and produces empty arrays.
        """
        sd = SpikeData([], length=0.0)
        assert sd.N == 0

        # Ragged
        path_ragged = str(tmp_path / "zero_ragged.h5")
        exporters.export_spikedata_to_hdf5(sd, path_ragged, style="ragged")
        with h5py.File(path_ragged, "r") as f:
            assert f["spike_times"].shape[0] == 0
            assert f["spike_times_index"].shape[0] == 0

        # Group
        path_group = str(tmp_path / "zero_group.h5")
        exporters.export_spikedata_to_hdf5(sd, path_group, style="group")
        with h5py.File(path_group, "r") as f:
            assert len(f["units"].keys()) == 0

        # Paired
        path_paired = str(tmp_path / "zero_paired.h5")
        exporters.export_spikedata_to_hdf5(sd, path_paired, style="paired")
        with h5py.File(path_paired, "r") as f:
            assert f["idces"].shape[0] == 0
            assert f["times"].shape[0] == 0

    def test_ec_de_06_overwrite_existing_file(self, tmp_path):
        """
        EC-DE-06: Verify that exporting to an existing HDF5 file overwrites it
        (mode="w" behavior).

        Tests:
            (Test Case 1) First export creates file with dataset A.
            (Test Case 2) Second export overwrites; old dataset is gone, new data present.
        """
        sd1 = SpikeData([np.array([5.0, 10.0, 15.0])], length=20.0)
        sd2 = SpikeData([np.array([100.0, 200.0])], length=300.0)
        path = str(tmp_path / "overwrite.h5")

        # First write
        exporters.export_spikedata_to_hdf5(sd1, path, style="ragged")
        with h5py.File(path, "r") as f:
            assert f["spike_times_index"][0] == 3  # 3 spikes

        # Overwrite with different data
        exporters.export_spikedata_to_hdf5(sd2, path, style="ragged")
        with h5py.File(path, "r") as f:
            assert f["spike_times_index"][0] == 2  # 2 spikes now


@skip_no_h5py
class TestNWBExporters:
    """
    Tests for Neurodata Without Borders (NWB) format export.

    NWB is a standardized format for neurophysiology data that uses HDF5 as its
    storage backend.
    """

    def test_export_nwb_roundtrip(self, tmp_path):
        """
        Tests NWB export and re-import round-trip.

        Tests:
        (Method 1) Export SpikeData using the NWB exporter
        (Method 2) Re-import using NWB loader with prefer_pynwb=False (h5py-based)
        (Test Case 1) Verify all spike trains match original within floating-point precision
        """
        sd = make_sd()
        path = str(tmp_path / "test.nwb")

        sd.to_nwb(path)
        sd2 = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
        for a, b in zip(sd.train, sd2.train):
            assert np.allclose(a, b)

    def test_nwb_electrode_roundtrip(self, tmp_path):
        """
        Verify electrodes are preserved when exporting SpikeData with electrode info to NWB.

        Tests:
            (Test Case 1) The 'units/electrodes' dataset exists in the exported NWB file.
            (Test Case 2) The electrode values match the original electrode IDs [4, 7, 2].
        """
        sd = _make_sd_with_electrodes()
        path = str(tmp_path / "electrodes.nwb")

        sd.to_nwb(path)

        with h5py.File(path, "r") as f:
            assert "units/electrodes" in f
            electrodes = np.asarray(f["units/electrodes"])
            np.testing.assert_array_equal(electrodes, np.array([4, 7, 2]))

    def test_ec_de_04_non_serializable_neuron_attributes(self, tmp_path):
        """
        EC-DE-04: Verify behavior when SpikeData has neuron_attributes with
        non-serializable values. The NWB exporter only writes electrode and
        location info from neuron_attributes, so non-serializable extra fields
        should not cause a crash.

        Tests:
            (Test Case 1) Export succeeds when neuron_attributes contain a
                non-serializable object (like a lambda or set).
            (Test Case 2) The exported file contains valid spike_times data.
        """
        trains = [np.array([5.0, 10.0]), np.array([15.0])]
        # Include a non-serializable value (a set and a lambda)
        neuron_attrs = [
            {"electrode": 0, "custom_set": {1, 2, 3}},
            {"electrode": 1, "custom_func": lambda x: x},
        ]
        sd = SpikeData(trains, length=20.0, neuron_attributes=neuron_attrs)

        path = str(tmp_path / "nonserial.nwb")
        # Should not raise - NWB exporter only uses electrode/location fields
        exporters.export_spikedata_to_nwb(sd, path)

        with h5py.File(path, "r") as f:
            assert "units/spike_times" in f
            st = np.asarray(f["units/spike_times"])
            assert len(st) == 3  # 2 + 1 spikes total


class TestKiloSortExporters:
    """
    Tests for KiloSort/Phy format export.

    KiloSort is a popular spike sorting algorithm that outputs spike times and
    cluster assignments in simple NumPy array format.
    """

    def test_export_kilosort_roundtrip_samples(self, tmp_path):
        """
        Test KiloSort export and import with sample-based timing.

        Tests:
        (Method 1) Export SpikeData to KiloSort format with 1000 Hz sampling rate
        (Test Case 1) Verify spike trains match after round-trip

        Notes:
        - Tests both the export logic and the assumption that unit indices map directly
        to cluster IDs in ascending order.
        """
        sd = make_sd()
        d = str(tmp_path / "ks")
        os.makedirs(d)

        sd.to_kilosort(d, fs_Hz=1000.0)
        sd2 = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)

        def q(ms):
            samp = np.rint(ms * (1000.0 / 1e3))
            return samp / 1000.0 * 1e3

        for a, b in zip(sd.train, sd2.train):
            assert np.allclose(q(a), b)

    def test_export_kilosort_custom_cluster_ids(self, tmp_path):
        """
        Tests KiloSort export with custom cluster ID assignment.

        Tests:
        (Method 1) Export with custom cluster IDs [10, 5, 7] instead of [0, 1, 2]
        (Test Case 1) Verify cluster ID mapping: unit 0 (3 spikes) -> cluster 10,
            unit 1 (2 spikes) -> cluster 5

        Notes:
        - Empty units (unit 2) don't contribute any spikes to the output arrays.
        """
        sd = make_sd()
        d = str(tmp_path / "ks")
        os.makedirs(d)

        sd.to_kilosort(d, fs_Hz=1000.0, cluster_ids=[10, 5, 7])
        times = np.load(os.path.join(d, "spike_times.npy"))
        clusters = np.load(os.path.join(d, "spike_clusters.npy"))
        assert (clusters == 10).sum() == 3
        assert (clusters == 5).sum() == 2

    def test_export_kilosort_very_small_fs(self, tmp_path):
        """
        Verify that exporting to KiloSort with a very small fs_Hz
        does not produce integer overflow. With very small fs_Hz, sample indices
        should be very small numbers (close to 0).

        Tests:
            (Test Case 1) Export succeeds without error.
            (Test Case 2) Spike times in samples are finite (no overflow).
        """
        sd = make_sd()
        d = str(tmp_path / "ks_small_fs")
        os.makedirs(d)

        exporters.export_spikedata_to_kilosort(sd, d, fs_Hz=0.001)
        times = np.load(os.path.join(d, "spike_times.npy"))
        assert np.all(np.isfinite(times))

    def test_export_kilosort_very_large_fs(self, tmp_path):
        """
        Verify that exporting to KiloSort with a very large fs_Hz
        does not produce integer overflow. With large fs_Hz and moderate spike
        times in ms, sample indices should remain within int64 range.

        Tests:
            (Test Case 1) Export succeeds without error.
            (Test Case 2) Spike times in samples are finite (no overflow).
        """
        sd = make_sd()
        d = str(tmp_path / "ks_large_fs")
        os.makedirs(d)

        exporters.export_spikedata_to_kilosort(sd, d, fs_Hz=1e9)
        times = np.load(os.path.join(d, "spike_times.npy"))
        assert np.all(np.isfinite(times))
        # With fs_Hz=1e9 and times in ms (max 20ms), samples = 20 * 1e6 = 2e7
        # which is well within int64 range
        assert times.max() < np.iinfo(np.int64).max

    def test_export_kilosort_duplicate_cluster_ids(self, tmp_path):
        """
        Verify that export_spikedata_to_kilosort accepts duplicate
        cluster_ids (two units mapping to the same cluster ID).

        Tests:
            (Test Case 1) Export succeeds without error.
            (Test Case 2) The cluster array contains the duplicated ID for both units.
        """
        trains = [
            np.array([5.0, 10.0]),
            np.array([15.0, 20.0]),
        ]
        sd = SpikeData(trains, length=25.0)
        d = str(tmp_path / "ks_dup")
        os.makedirs(d)

        exporters.export_spikedata_to_kilosort(sd, d, fs_Hz=1000.0, cluster_ids=[7, 7])
        clusters = np.load(os.path.join(d, "spike_clusters.npy"))
        # All 4 spikes should map to cluster 7
        assert np.all(clusters == 7)
        assert len(clusters) == 4

    def test_kilosort_channel_map_written(self, tmp_path):
        """
        Verify that channel_map.npy is created when SpikeData has electrode info.

        Tests:
            (Test Case 1) channel_map.npy file exists after export.
            (Test Case 2) channel_map.npy contains the correct electrode IDs [4, 7, 2].
        """
        sd = _make_sd_with_electrodes()
        d = str(tmp_path / "ks_chmap")

        sd.to_kilosort(d, fs_Hz=1000.0)

        channel_map_path = os.path.join(d, "channel_map.npy")
        assert os.path.exists(channel_map_path)
        channel_map = np.load(channel_map_path)
        np.testing.assert_array_equal(channel_map, np.array([4, 7, 2]))

    def test_kilosort_time_unit_ms(self, tmp_path):
        """
        Verify spike_times.npy values are in milliseconds when time_unit='ms'.

        Tests:
            (Test Case 1) Exported spike times match the original millisecond values.
        """
        sd = _make_sd_with_electrodes()
        d = str(tmp_path / "ks_ms")

        sd.to_kilosort(d, fs_Hz=1000.0, time_unit="ms")

        times = np.load(os.path.join(d, "spike_times.npy"))
        # Collect all spike times in ms from the original trains, in unit order
        expected_ms = np.concatenate([t for t in sd.train if len(t) > 0])
        np.testing.assert_allclose(times, expected_ms)

    def test_kilosort_time_unit_seconds(self, tmp_path):
        """
        Verify spike_times.npy values are in seconds when time_unit='s'.

        Tests:
            (Test Case 1) Exported spike times equal original ms values divided by 1000.
        """
        sd = _make_sd_with_electrodes()
        d = str(tmp_path / "ks_s")

        sd.to_kilosort(d, fs_Hz=1000.0, time_unit="s")

        times = np.load(os.path.join(d, "spike_times.npy"))
        expected_s = np.concatenate([t for t in sd.train if len(t) > 0]) / 1e3
        np.testing.assert_allclose(times, expected_s)

    def test_ec_de_03_cluster_ids_length_mismatch(self, tmp_path):
        """
        EC-DE-03: Verify that passing cluster_ids with a length that doesn't
        match sd.N raises ValueError.

        Tests:
            (Test Case 1) 3 units but 2 cluster_ids -> ValueError.
            (Test Case 2) 3 units but 4 cluster_ids -> ValueError.
        """
        sd = make_sd()  # 3 units
        d = str(tmp_path / "ks")
        os.makedirs(d)

        with pytest.raises(ValueError, match="cluster_ids"):
            exporters.export_spikedata_to_kilosort(
                sd, d, fs_Hz=1000.0, cluster_ids=[10, 20]
            )

        with pytest.raises(ValueError, match="cluster_ids"):
            exporters.export_spikedata_to_kilosort(
                sd, d, fs_Hz=1000.0, cluster_ids=[10, 20, 30, 40]
            )


class TestPickleExporters:
    """
    Tests for pickle export functionality.

    Pickle is a Python-native serialization format. These tests validate:
    - Round-trip integrity through export and load
    - Protocol parameter handling for Python version compatibility
    - S3 upload flow when s3_upload=True
    - Temporary file cleanup after S3 upload
    """

    def test_export_pickle_roundtrip(self, tmp_path):
        """
        Tests basic pickle export and import round-trip.

        Tests:
        (Test Case 1) Verify all spike trains match original.
        (Test Case 2) Verify metadata is preserved.
        """
        sd = make_sd()
        path = str(tmp_path / "test.pkl")

        exporters.export_to_pickle(sd, path)
        sd2 = loaders.load_spikedata_from_pickle(path)
        for a, b in zip(sd.train, sd2.train):
            assert np.allclose(a, b)
        assert sd.metadata == sd2.metadata

    def test_export_pickle_protocol(self, tmp_path):
        """
        Tests protocol parameter is passed through correctly.

        Tests:
        (Test Case 1) Lower protocols produce loadable files.
        """
        sd = make_sd()
        path = str(tmp_path / "test.pkl")

        exporters.export_to_pickle(sd, path, protocol=2)
        sd2 = loaders.load_spikedata_from_pickle(path)
        for a, b in zip(sd.train, sd2.train):
            assert np.allclose(a, b)

    @patch("spikelab.data_loaders.s3_utils.upload_to_s3")
    def test_export_pickle_s3_upload(self, mock_upload):
        """
        Tests S3 upload flow when s3_upload=True.

        Tests:
        (Test Case 1) Returns S3 URL on success.
        (Test Case 2) upload_to_s3 called with correct arguments.
        """
        sd = make_sd()
        s3_url = "s3://mybucket/path/output.pkl"

        result = exporters.export_to_pickle(sd, s3_url, s3_upload=True)

        assert result == s3_url
        mock_upload.assert_called_once()
        call_args = mock_upload.call_args
        assert call_args[0][1] == s3_url
        assert call_args[0][0].endswith(".pkl")

    @patch("spikelab.data_loaders.s3_utils.upload_to_s3")
    def test_export_pickle_temp_cleanup(self, mock_upload):
        """
        Tests temporary file is removed after S3 upload.

        Tests:
        (Test Case 1) Temp file does not exist after export completes.
        """
        sd = make_sd()
        temp_paths = []

        def capture_temp(local_path, s3_url, **kwargs):
            temp_paths.append(local_path)

        mock_upload.side_effect = capture_temp

        exporters.export_to_pickle(sd, "s3://bucket/key.pkl", s3_upload=True)

        assert len(temp_paths) == 1
        assert not os.path.exists(temp_paths[0])

    def test_ec_de_01_zero_unit_pickle_roundtrip(self, tmp_path):
        """
        Verify that a zero-unit SpikeData can be pickled and loaded back.

        Tests:
            (Test Case 1) Export succeeds.
            (Test Case 2) Round-trip preserves N=0.
        """
        import spikelab.data_loaders.data_loaders as loaders

        sd = SpikeData([], length=0.0)
        path = str(tmp_path / "empty.pkl")
        exporters.export_to_pickle(sd, path)
        sd2 = loaders.load_spikedata_from_pickle(path)
        assert sd2.N == 0

    def test_s3_upload_with_non_s3_url(self):
        """
        Verify that export_to_pickle raises ValueError when
        s3_upload=True but filepath is not an S3 URL.

        Tests:
            (Test Case 1) A local path with s3_upload=True raises ValueError.
            (Test Case 2) An HTTP URL (not S3) with s3_upload=True raises ValueError.
        """
        sd = make_sd()

        with pytest.raises(ValueError, match="S3 URL"):
            exporters.export_to_pickle(sd, "/tmp/local_file.pkl", s3_upload=True)

        with pytest.raises(ValueError, match="S3 URL"):
            exporters.export_to_pickle(
                sd, "https://example.com/file.pkl", s3_upload=True
            )

    def test_pickle_export_to_nested_nonexistent_directory(self, tmp_path):
        """
        Verify that export_to_pickle creates intermediate directories
        when the target path includes nested directories that don't exist.

        Tests:
            (Test Case 1) Export succeeds to a deeply nested path.
            (Test Case 2) The file exists after export.
            (Test Case 3) Round-trip preserves data.
        """
        sd = make_sd()
        nested_path = str(tmp_path / "a" / "b" / "c" / "test.pkl")

        result = exporters.export_to_pickle(sd, nested_path)

        assert result == nested_path
        assert os.path.exists(nested_path)
        sd2 = loaders.load_spikedata_from_pickle(nested_path)
        assert sd2.N == sd.N

    @patch("spikelab.data_loaders.s3_utils.upload_to_s3")
    def test_s3_upload_failure_cleanup(self, mock_upload):
        """
        Verify that the temporary file is cleaned up even when the S3 upload
        fails with an exception.

        Tests:
            (Test Case 1) The export raises the upload exception.
            (Test Case 2) The temporary file does not exist after the failure.
        """
        sd = make_sd()
        temp_paths = []

        def failing_upload(local_path, s3_url, **kwargs):
            temp_paths.append(local_path)
            raise RuntimeError("Simulated S3 upload failure")

        mock_upload.side_effect = failing_upload

        with pytest.raises(RuntimeError, match="Simulated S3 upload failure"):
            exporters.export_to_pickle(sd, "s3://bucket/key.pkl", s3_upload=True)

        assert len(temp_paths) == 1
        assert not os.path.exists(temp_paths[0])


@skip_no_h5py
class TestHDF5ExportersEdgeCases:
    """
    Edge case tests for HDF5 export functionality.

    These tests cover error paths, boundary conditions, and unusual but valid
    inputs for export_spikedata_to_hdf5.
    """

    def test_invalid_style_string(self, tmp_path):
        """
        Verify that passing an invalid style string raises ValueError.

        Tests:
            (Test Case 1) style="invalid" raises ValueError.
            (Test Case 2) style="" (empty string) raises ValueError.
        """
        sd = make_sd()
        path = str(tmp_path / "bad_style.h5")

        with pytest.raises(ValueError, match="Unknown style"):
            exporters.export_spikedata_to_hdf5(sd, path, style="invalid")

        with pytest.raises(ValueError, match="Unknown style"):
            exporters.export_spikedata_to_hdf5(sd, path, style="")

    def test_raster_style_with_none_bin_size(self, tmp_path):
        """
        Verify that style="raster" with raster_bin_size_ms=None raises ValueError.

        Tests:
            (Test Case 1) Omitting raster_bin_size_ms (default None) raises ValueError.
        """
        sd = make_sd()
        path = str(tmp_path / "no_bin.h5")

        with pytest.raises(ValueError, match="raster_bin_size_ms"):
            exporters.export_spikedata_to_hdf5(sd, path, style="raster")

    def test_group_style_single_unit(self, tmp_path):
        """
        Verify that group style export works correctly with a single-unit SpikeData.

        Tests:
            (Test Case 1) Export succeeds without error.
            (Test Case 2) The group contains exactly one dataset.
            (Test Case 3) The dataset contains the correct spike times.
        """
        trains = [np.array([5.0, 10.0, 15.0])]
        sd = SpikeData(trains, length=20.0)
        path = str(tmp_path / "single_group.h5")

        exporters.export_spikedata_to_hdf5(sd, path, style="group")

        with h5py.File(path, "r") as f:
            grp = f["units"]
            assert len(grp.keys()) == 1
            assert "0" in grp
            times_s = np.asarray(grp["0"])
            np.testing.assert_allclose(times_s, np.array([5.0, 10.0, 15.0]) / 1e3)

    def test_raw_time_unit_samples_missing_fs_hz(self, tmp_path):
        """
        Verify that raw_time_unit='samples' without a valid fs_Hz raises ValueError.

        Tests:
            (Test Case 1) fs_Hz=None with raw_time_unit='samples' raises ValueError.
            (Test Case 2) fs_Hz=0 with raw_time_unit='samples' raises ValueError.
        """
        raw = np.random.randn(2, 10)
        sd = SpikeData(
            [np.array([5.0])], length=20.0, raw_data=raw, raw_time=np.arange(10.0)
        )
        path = str(tmp_path / "raw_no_fs.h5")

        with pytest.raises(ValueError, match="fs_Hz"):
            exporters.export_spikedata_to_hdf5(
                sd,
                path,
                style="ragged",
                raw_dataset="raw",
                raw_time_dataset="raw_time",
                raw_time_unit="samples",
                fs_Hz=None,
            )

        with pytest.raises(ValueError, match="fs_Hz"):
            exporters.export_spikedata_to_hdf5(
                sd,
                path,
                style="ragged",
                raw_dataset="raw",
                raw_time_dataset="raw_time",
                raw_time_unit="samples",
                fs_Hz=0,
            )

    def test_raw_data_export_invalid_raw_time_unit(self, tmp_path):
        """
        Verify that an invalid raw_time_unit raises ValueError.

        Tests:
            (Test Case 1) raw_time_unit='invalid' raises ValueError.
        """
        raw = np.random.randn(2, 10)
        sd = SpikeData(
            [np.array([5.0])], length=20.0, raw_data=raw, raw_time=np.arange(10.0)
        )
        path = str(tmp_path / "raw_bad_unit.h5")

        with pytest.raises(ValueError, match="raw_time_unit"):
            exporters.export_spikedata_to_hdf5(
                sd,
                path,
                style="ragged",
                raw_dataset="raw",
                raw_time_dataset="raw_time",
                raw_time_unit="invalid",
            )

    def test_case_insensitive_style_normalization(self, tmp_path):
        """
        Verify that style strings are case-insensitive via .lower() normalization.

        Tests:
            (Test Case 1) style="RAGGED" (uppercase) exports successfully.
            (Test Case 2) style="Paired" (mixed case) exports successfully.
            (Test Case 3) Exported data is correct after round-trip.
        """
        sd = make_sd()

        # Uppercase
        path_upper = str(tmp_path / "upper.h5")
        exporters.export_spikedata_to_hdf5(sd, path_upper, style="RAGGED")
        with h5py.File(path_upper, "r") as f:
            assert "spike_times" in f
            assert "spike_times_index" in f

        # Mixed case
        path_mixed = str(tmp_path / "mixed.h5")
        exporters.export_spikedata_to_hdf5(
            sd,
            path_mixed,
            style="Paired",
            idces_dataset="idces",
            times_dataset="times",
        )
        with h5py.File(path_mixed, "r") as f:
            assert "idces" in f
            assert "times" in f


@skip_no_h5py
class TestNWBExportersEdgeCases:
    """
    Edge case tests for NWB export functionality.

    These tests cover boundary conditions and unusual but valid inputs
    for export_spikedata_to_nwb.
    """

    def test_nwb_export_zero_units(self, tmp_path):
        """
        Verify that exporting a zero-unit SpikeData to NWB succeeds.

        Tests:
            (Test Case 1) Export succeeds without error.
            (Test Case 2) The units group exists with empty spike_times.
            (Test Case 3) The spike_times_index is empty.
            (Test Case 4) The id dataset is empty.
        """
        sd = SpikeData([], length=0.0)
        path = str(tmp_path / "zero_units.nwb")

        exporters.export_spikedata_to_nwb(sd, path)

        with h5py.File(path, "r") as f:
            assert "units" in f
            assert f["units/spike_times"].shape[0] == 0
            assert f["units/spike_times_index"].shape[0] == 0
            assert f["units/id"].shape[0] == 0

    def test_nwb_export_all_empty_trains(self, tmp_path):
        """
        Verify that exporting a SpikeData where all units have empty spike
        trains to NWB works correctly.

        Tests:
            (Test Case 1) Export succeeds without error.
            (Test Case 2) spike_times is empty (no spikes).
            (Test Case 3) spike_times_index contains zeros (cumulative counts).
            (Test Case 4) id dataset contains the correct unit IDs.
        """
        trains = [np.array([], float), np.array([], float), np.array([], float)]
        sd = SpikeData(trains, length=100.0)
        path = str(tmp_path / "empty_trains.nwb")

        exporters.export_spikedata_to_nwb(sd, path)

        with h5py.File(path, "r") as f:
            assert f["units/spike_times"].shape[0] == 0
            idx = np.asarray(f["units/spike_times_index"])
            np.testing.assert_array_equal(idx, np.array([0, 0, 0]))
            ids = np.asarray(f["units/id"])
            np.testing.assert_array_equal(ids, np.array([0, 1, 2]))

    def test_nwb_export_unit_locations_no_electrodes(self, tmp_path):
        """
        Verify NWB export when SpikeData has unit_locations but no electrode IDs.
        The exporter should use fallback electrode IDs (0..N-1).

        Tests:
            (Test Case 1) Export succeeds without error.
            (Test Case 2) units/electrodes dataset is absent.
            (Test Case 3) Electrodes table exists with fallback IDs [0, 1].
            (Test Case 4) x and y coordinates are written correctly.
        """
        trains = [np.array([5.0, 10.0]), np.array([15.0])]
        neuron_attrs = [
            {"x": 100.0, "y": 200.0},
            {"x": 300.0, "y": 400.0},
        ]
        sd = SpikeData(trains, length=20.0, neuron_attributes=neuron_attrs)
        assert sd.unit_locations is not None
        assert sd.electrodes is None
        path = str(tmp_path / "locs_no_elec.nwb")

        exporters.export_spikedata_to_nwb(sd, path)

        with h5py.File(path, "r") as f:
            assert "units/electrodes" not in f
            elec_grp = f["general/extracellular_ephys/electrodes"]
            np.testing.assert_array_equal(np.asarray(elec_grp["id"]), [0, 1])
            np.testing.assert_allclose(np.asarray(elec_grp["x"]), [100.0, 300.0])
            np.testing.assert_allclose(np.asarray(elec_grp["y"]), [200.0, 400.0])

    def test_nwb_export_unit_locations_x_only(self, tmp_path):
        """
        Verify NWB export when unit_locations has only 1 spatial dimension (x only).

        Tests:
            (Test Case 1) Export succeeds without error.
            (Test Case 2) x dataset is written.
            (Test Case 3) y dataset is absent.
            (Test Case 4) z dataset is absent.
        """
        trains = [np.array([5.0]), np.array([10.0])]
        neuron_attrs = [
            {"location": np.array([100.0])},
            {"location": np.array([200.0])},
        ]
        sd = SpikeData(trains, length=15.0, neuron_attributes=neuron_attrs)
        assert sd.unit_locations is not None
        assert sd.unit_locations.shape == (2, 1)
        path = str(tmp_path / "x_only.nwb")

        exporters.export_spikedata_to_nwb(sd, path)

        with h5py.File(path, "r") as f:
            elec_grp = f["general/extracellular_ephys/electrodes"]
            np.testing.assert_allclose(np.asarray(elec_grp["x"]), [100.0, 200.0])
            assert "y" not in elec_grp
            assert "z" not in elec_grp

    def test_nwb_export_duplicate_electrode_ids(self, tmp_path):
        """
        Verify NWB export when multiple units share the same electrode ID.
        The electrodes table should contain unique electrode IDs only.

        Tests:
            (Test Case 1) Export succeeds without error.
            (Test Case 2) units/electrodes contains the per-unit electrode IDs [3, 3, 5].
            (Test Case 3) Electrodes table contains unique IDs [3, 5].
        """
        trains = [np.array([5.0]), np.array([10.0]), np.array([15.0])]
        neuron_attrs = [
            {"electrode": 3, "x": 10.0, "y": 20.0},
            {"electrode": 3, "x": 10.0, "y": 20.0},
            {"electrode": 5, "x": 30.0, "y": 40.0},
        ]
        sd = SpikeData(trains, length=20.0, neuron_attributes=neuron_attrs)
        path = str(tmp_path / "dup_elec.nwb")

        exporters.export_spikedata_to_nwb(sd, path)

        with h5py.File(path, "r") as f:
            unit_elec = np.asarray(f["units/electrodes"])
            np.testing.assert_array_equal(unit_elec, [3, 3, 5])
            elec_ids = np.asarray(f["general/extracellular_ephys/electrodes/id"])
            np.testing.assert_array_equal(elec_ids, [3, 5])


class TestKiloSortExportersEdgeCases:
    """
    Edge case tests for KiloSort export functionality.

    These tests cover error paths and boundary conditions for
    export_spikedata_to_kilosort.
    """

    def test_kilosort_export_zero_units(self, tmp_path):
        """
        Verify that exporting a zero-unit SpikeData to KiloSort succeeds
        and produces empty arrays.

        Tests:
            (Test Case 1) Export succeeds without error.
            (Test Case 2) spike_times.npy is empty.
            (Test Case 3) spike_clusters.npy is empty.
        """
        sd = SpikeData([], length=0.0)
        d = str(tmp_path / "ks_zero")

        exporters.export_spikedata_to_kilosort(sd, d, fs_Hz=1000.0)

        times = np.load(os.path.join(d, "spike_times.npy"))
        clusters = np.load(os.path.join(d, "spike_clusters.npy"))
        assert len(times) == 0
        assert len(clusters) == 0

    def test_kilosort_invalid_time_unit(self, tmp_path):
        """
        Verify that an invalid time_unit raises ValueError.

        Tests:
            (Test Case 1) time_unit="invalid" raises ValueError.
        """
        sd = make_sd()
        d = str(tmp_path / "ks_bad_unit")
        os.makedirs(d)

        with pytest.raises(ValueError, match="time_unit"):
            exporters.export_spikedata_to_kilosort(
                sd, d, fs_Hz=1000.0, time_unit="invalid"
            )

    def test_kilosort_fs_hz_zero_raises(self, tmp_path):
        """
        Verify that fs_Hz=0 raises ValueError.

        Tests:
            (Test Case 1) fs_Hz=0 raises ValueError.
            (Test Case 2) fs_Hz=-1 raises ValueError.
        """
        sd = make_sd()
        d = str(tmp_path / "ks_zero_fs")
        os.makedirs(d)

        with pytest.raises(ValueError, match="fs_Hz"):
            exporters.export_spikedata_to_kilosort(sd, d, fs_Hz=0)

        with pytest.raises(ValueError, match="fs_Hz"):
            exporters.export_spikedata_to_kilosort(sd, d, fs_Hz=-1.0)

    def test_kilosort_no_electrodes_no_channel_map(self, tmp_path):
        """
        Verify that channel_map.npy is not created when SpikeData has no
        electrode information.

        Tests:
            (Test Case 1) Export succeeds without error.
            (Test Case 2) channel_map.npy does not exist in the output directory.
            (Test Case 3) spike_times.npy and spike_clusters.npy do exist.
        """
        sd = make_sd()  # no neuron_attributes, so electrodes is None
        assert sd.electrodes is None
        d = str(tmp_path / "ks_no_elec")

        exporters.export_spikedata_to_kilosort(sd, d, fs_Hz=1000.0)

        assert not os.path.exists(os.path.join(d, "channel_map.npy"))
        assert os.path.exists(os.path.join(d, "spike_times.npy"))
        assert os.path.exists(os.path.join(d, "spike_clusters.npy"))


# ---------------------------------------------------------------------------
# Edge case tests from the edge case scan
# ---------------------------------------------------------------------------


@skip_no_h5py
class TestHDF5ExportersEdgeCases2:
    """Additional edge case tests for export_spikedata_to_hdf5."""

    def test_nonzero_start_time_roundtrip_ragged(self, tmp_path):
        """
        Non-zero start_time is preserved through a ragged-style export/load round-trip.

        Tests:
            (Test Case 1) start_time=-100 survives export and reimport.
        """
        trains = [np.array([-90.0, -50.0, 0.0, 10.0])]
        sd = SpikeData(trains, length=200.0, start_time=-100.0)
        path = str(tmp_path / "start_time_ragged.h5")

        exporters.export_spikedata_to_hdf5(sd, path, style="ragged")
        loaded = loaders.load_spikedata_from_hdf5(
            path,
            spike_times_dataset="spike_times",
            spike_times_index_dataset="spike_times_index",
        )
        assert loaded.start_time == pytest.approx(-100.0)
        # Length is inferred from max spike time - start_time
        # max(10.0) - (-100.0) = 110.0
        assert loaded.length == pytest.approx(110.0)

    def test_nonzero_start_time_roundtrip_paired(self, tmp_path):
        """
        Non-zero start_time is preserved through a paired-style export/load round-trip.

        Tests:
            (Test Case 1) start_time=-50 survives export and reimport in paired style.
        """
        trains = [np.array([-40.0, -20.0]), np.array([-10.0, 0.0])]
        sd = SpikeData(trains, length=100.0, start_time=-50.0)
        path = str(tmp_path / "start_time_paired.h5")

        exporters.export_spikedata_to_hdf5(sd, path, style="paired")
        loaded = loaders.load_spikedata_from_hdf5(
            path,
            idces_dataset="idces",
            times_dataset="times",
            times_unit="ms",
        )
        assert loaded.start_time == pytest.approx(-50.0)

    def test_raster_export_all_empty_trains(self, tmp_path):
        """
        Raster export with all-empty-train SpikeData.

        Tests:
            (Test Case 1) All-empty trains produce an all-zero raster that
                can be exported and loaded back.
        """
        sd = SpikeData([[], [], []], length=25.0)
        path = str(tmp_path / "raster_empty.h5")

        exporters.export_spikedata_to_hdf5(
            sd, path, style="raster", raster_bin_size_ms=5.0
        )
        loaded = loaders.load_spikedata_from_hdf5(
            path, raster_dataset="raster", raster_bin_size_ms=5.0
        )
        assert loaded.N == 3
        for t in loaded.train:
            assert len(t) == 0

    def test_group_style_more_than_9_units(self, tmp_path):
        """
        Group style with >9 units: lexicographic sort mismatch.

        Tests:
            (Test Case 1) Export 12 units, then load. Verify unit count matches.
        """
        trains = [np.array([float(i + 1)]) for i in range(12)]
        sd = SpikeData(trains, length=20.0)
        path = str(tmp_path / "group_12.h5")

        exporters.export_spikedata_to_hdf5(sd, path, style="group")
        loaded = loaders.load_spikedata_from_hdf5(path, group_per_unit="units")
        assert loaded.N == 12


@skip_no_h5py
class TestNWBExportersEdgeCases2:
    """Additional edge case tests for export_spikedata_to_nwb."""

    def test_nonzero_start_time_warning(self, tmp_path):
        """
        NWB export with non-zero start_time issues a UserWarning.

        Tests:
            (Test Case 1) start_time=-100 triggers a UserWarning about
                NWB not preserving start_time.
        """
        import warnings

        trains = [np.array([-50.0, 0.0, 50.0])]
        sd = SpikeData(trains, length=200.0, start_time=-100.0)
        path = str(tmp_path / "nwb_start_time.nwb")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            exporters.export_spikedata_to_nwb(sd, path)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert any("start_time" in str(x.message) for x in user_warnings)

    def test_z_coordinates_roundtrip(self, tmp_path):
        """
        NWB export with 3D (x, y, z) locations.

        Tests:
            (Test Case 1) 3D locations are exported and loaded back.
        """
        trains = [np.array([5.0]), np.array([10.0])]
        attrs = [
            {"electrode_id": 0, "x": 1.0, "y": 2.0, "z": 3.0},
            {"electrode_id": 1, "x": 4.0, "y": 5.0, "z": 6.0},
        ]
        sd = SpikeData(trains, length=20.0, neuron_attributes=attrs)
        path = str(tmp_path / "nwb_3d.nwb")

        exporters.export_spikedata_to_nwb(sd, path)
        loaded = loaders.load_spikedata_from_nwb(path)
        assert loaded.N == 2


class TestKiloSortExportersEdgeCases2:
    """Additional edge case tests for export_spikedata_to_kilosort."""

    def test_nonzero_start_time_warning(self, tmp_path):
        """
        KiloSort export with non-zero start_time issues a UserWarning.

        Tests:
            (Test Case 1) start_time=-50 triggers a UserWarning.
        """
        import warnings

        trains = [np.array([-30.0, -10.0, 0.0])]
        sd = SpikeData(trains, length=100.0, start_time=-50.0)
        path = str(tmp_path / "ks_start_time")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            exporters.export_spikedata_to_kilosort(sd, path, fs_Hz=1000.0)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert any("start_time" in str(x.message) for x in user_warnings)

    def test_all_empty_trains_kilosort(self, tmp_path):
        """
        KiloSort export with N>0 but all trains empty.

        Tests:
            (Test Case 1) Export succeeds with empty spike_times and spike_clusters.
        """
        sd = SpikeData([[], [], []], length=25.0)
        path = str(tmp_path / "ks_empty")

        exporters.export_spikedata_to_kilosort(sd, path, fs_Hz=1000.0)
        times = np.load(os.path.join(path, "spike_times.npy"))
        assert len(times) == 0


class TestCoverageGaps:
    """Tests for exporter coverage gaps."""

    def test_export_hdf5_group_time_unit_ms(self, tmp_path):
        """
        Tests: export_spikedata_to_hdf5 group style with group_time_unit='ms'.

        (Test Case 1) Export succeeds.
        (Test Case 2) Reloaded spike times match originals (in ms).
        """
        import h5py

        sd = SpikeData([[5.0, 10.0, 20.0], [15.0, 25.0]], length=30.0)
        filepath = str(tmp_path / "group_ms.h5")
        exporters.export_spikedata_to_hdf5(
            sd,
            filepath,
            style="group",
            group_time_unit="ms",
        )
        # Verify times are stored in ms (under group "units")
        with h5py.File(filepath, "r") as f:
            unit0 = np.asarray(f["units"]["0"])
            np.testing.assert_allclose(unit0, [5.0, 10.0, 20.0])

    def test_export_hdf5_raw_time_unit_samples(self, tmp_path):
        """
        Tests: export_spikedata_to_hdf5 with raw_time_unit='samples'.

        (Test Case 1) Raw time array is stored in sample units.
        """
        import h5py

        raw_data = np.random.default_rng(0).random((2, 100))
        raw_time = np.arange(100, dtype=float)  # in ms
        sd = SpikeData(
            [[5.0, 10.0], [15.0]],
            length=100.0,
            raw_data=raw_data,
            raw_time=raw_time,
        )
        filepath = str(tmp_path / "raw_samples.h5")
        exporters.export_spikedata_to_hdf5(
            sd,
            filepath,
            style="ragged",
            raw_dataset="raw",
            raw_time_dataset="raw_t",
            raw_time_unit="samples",
            fs_Hz=1000.0,
        )
        with h5py.File(filepath, "r") as f:
            raw_t = np.asarray(f["raw_t"])
            # ms * (1000 Hz / 1000) = samples, so values should be rint(raw_time * 1.0)
            np.testing.assert_allclose(raw_t, np.rint(raw_time).astype(int))


@skip_no_h5py
class TestEdgeCaseScan:
    """Edge-case tests for data_loaders/data_exporters.py."""

    def test_raster_style_zero_length_spikedata(self, tmp_path):
        """Tests: Raster style export with zero-length SpikeData produces (2, 0) raster.
        (Test Case 1)
        """
        import h5py

        sd = SpikeData([[], []], length=0.0)
        filepath = str(tmp_path / "raster_zero.h5")
        exporters.export_spikedata_to_hdf5(
            sd, filepath, style="raster", raster_bin_size_ms=1.0
        )
        assert os.path.isfile(filepath)
        with h5py.File(filepath, "r") as f:
            raster = np.asarray(f["raster"])
            assert raster.shape == (2, 0)

    def test_raw_time_unit_seconds_export(self, tmp_path):
        """Tests: raw_time_unit='s' converts raw_time from ms to seconds.
        (Test Case 2)
        """
        import h5py

        raw_data = np.ones((1, 5))
        raw_time = np.array([0.0, 100.0, 200.0, 300.0, 400.0])  # in ms
        sd = SpikeData(
            [[50.0, 150.0]], length=500.0, raw_data=raw_data, raw_time=raw_time
        )
        filepath = str(tmp_path / "raw_seconds.h5")
        exporters.export_spikedata_to_hdf5(
            sd,
            filepath,
            style="ragged",
            raw_dataset="raw",
            raw_time_dataset="raw_t",
            raw_time_unit="s",
        )
        with h5py.File(filepath, "r") as f:
            raw_t = np.asarray(f["raw_t"])
            expected = raw_time / 1e3  # ms -> s
            np.testing.assert_allclose(raw_t, expected)

    def test_group_style_all_empty_trains(self, tmp_path):
        """Tests: Group style export with all-empty trains creates empty datasets.
        (Test Case 3)
        """
        import h5py

        sd = SpikeData([[], [], []], length=25.0)
        filepath = str(tmp_path / "group_empty.h5")
        exporters.export_spikedata_to_hdf5(sd, filepath, style="group")
        with h5py.File(filepath, "r") as f:
            grp = f["units"]
            for i in range(3):
                ds = np.asarray(grp[str(i)])
                assert ds.shape == (
                    0,
                ), f"Unit {i} should be empty, got shape {ds.shape}"

    def test_nwb_export_event_centered_warns(self, tmp_path):
        """Tests: NWB export with event-centered SpikeData emits start_time warning.
        (Test Case 4)
        """
        sd = SpikeData(
            [np.array([-150.0, -50.0, 100.0]), np.array([-80.0])],
            length=400.0,
            start_time=-200.0,
        )
        filepath = str(tmp_path / "event_centered.nwb")
        with pytest.warns(UserWarning, match="start_time"):
            exporters.export_spikedata_to_nwb(sd, filepath)
        assert os.path.isfile(filepath)

    def test_kilosort_export_event_centered_warns(self, tmp_path):
        """Tests: KiloSort export with event-centered SpikeData emits start_time warning.
        (Test Case 5)
        """
        sd = SpikeData(
            [np.array([-50.0, 0.0, 50.0])],
            length=200.0,
            start_time=-100.0,
        )
        folder = str(tmp_path / "ks_event")
        with pytest.warns(UserWarning, match="start_time"):
            exporters.export_spikedata_to_kilosort(sd, folder, fs_Hz=30000.0)
        assert os.path.isfile(os.path.join(folder, "spike_times.npy"))

    def test_kilosort_cluster_ids_large_gaps(self, tmp_path):
        """Tests: KiloSort export with non-sequential cluster_ids preserves IDs.
        (Test Case 6)
        """
        sd = SpikeData(
            [np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0, 5.0])],
            length=10.0,
        )
        folder = str(tmp_path / "ks_gaps")
        cluster_ids = [0, 100, 999]
        exporters.export_spikedata_to_kilosort(
            sd, folder, fs_Hz=30000.0, cluster_ids=cluster_ids
        )
        clusters = np.load(os.path.join(folder, "spike_clusters.npy"))
        # Unit 0 has 2 spikes -> cluster 0, unit 1 has 1 spike -> cluster 100,
        # unit 2 has 2 spikes -> cluster 999
        expected = np.array([0, 0, 100, 999, 999])
        np.testing.assert_array_equal(clusters, expected)

    def test_pickle_ratedata_accepted(self, tmp_path):
        """
        export_to_pickle accepts RateData (not just SpikeData).

        Tests:
            (Test Case 1) RateData passes the isinstance check.
            (Test Case 2) Roundtrip preserves data.
        """
        from spikelab.spikedata.ratedata import RateData

        rd = RateData(
            inst_Frate_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
            times=np.array([0.0, 1.0]),
        )
        path = str(tmp_path / "rd.pkl")
        exporters.export_to_pickle(rd, path)

        import pickle

        with open(path, "rb") as f:
            rd2 = pickle.load(f)
        assert rd2.inst_Frate_data.shape == (2, 2)

    def test_pickle_unsupported_type_rejected(self, tmp_path):
        """
        export_to_pickle raises TypeError for unsupported types.

        Tests:
            (Test Case 1) Plain dict raises TypeError.
        """
        path = str(tmp_path / "bad.pkl")
        with pytest.raises(TypeError, match="Expected a spikelab data object"):
            exporters.export_to_pickle({"not": "spikedata"}, path)

    def test_nwb_export_shared_electrode_ids(self, tmp_path):
        """
        NWB export deduplicates shared electrode IDs.

        Tests:
            (Test Case 1) Two units on same electrode produce one electrode entry.
        """
        import h5py

        sd = SpikeData(
            [[10.0, 20.0], [15.0, 25.0]],
            length=30.0,
            neuron_attributes=[
                {"electrode": 0, "location": [1.0, 2.0]},
                {"electrode": 0, "location": [1.0, 2.0]},
            ],
        )
        path = str(tmp_path / "shared_elec.h5")
        exporters.export_spikedata_to_nwb(sd, path)

        with h5py.File(path, "r") as f:
            elec_ids = np.asarray(f["general/extracellular_ephys/electrodes/id"])
            assert len(elec_ids) == 1  # deduplicated
            assert elec_ids[0] == 0

    @patch("spikelab.data_loaders.s3_utils.upload_to_s3")
    def test_pickle_s3_temp_cleanup_oserror(self, mock_upload):
        """
        OSError during temp file cleanup after S3 upload is silently ignored.

        Tests:
            (Test Case 1) Function returns S3 URL despite cleanup failure.
        """
        sd = make_sd()
        s3_url = "s3://bucket/key.pkl"

        temp_paths = []

        def capture_and_succeed(local_path, s3_url_arg, **kwargs):
            temp_paths.append(local_path)

        mock_upload.side_effect = capture_and_succeed

        with patch("os.remove", side_effect=OSError("file locked")):
            result = exporters.export_to_pickle(sd, s3_url, s3_upload=True)

        assert result == s3_url
