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
import pathlib
import sys
from unittest.mock import patch

import numpy as np
import pytest

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

# Ensure project root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpikeLab.spikedata import SpikeData
import SpikeLab.data_loaders.data_loaders as loaders
import SpikeLab.data_loaders.data_exporters as exporters


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


class TestPickleExporters:
    """
    Tests for pickle export functionality.

    Pickle is a Python-native serialization format. These tests validate:
    - Round-trip integrity through export and load
    - Protocol parameter handling for Python version compatibility
    - S3 upload flow when upload_to_s3=True
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

        exporters.export_spikedata_to_pickle(sd, path)
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

        exporters.export_spikedata_to_pickle(sd, path, protocol=2)
        sd2 = loaders.load_spikedata_from_pickle(path)
        for a, b in zip(sd.train, sd2.train):
            assert np.allclose(a, b)

    @patch("SpikeLab.data_loaders.s3_utils.upload_to_s3")
    def test_export_pickle_s3_upload(self, mock_upload):
        """
        Tests S3 upload flow when upload_to_s3=True.

        Tests:
        (Test Case 1) Returns S3 URL on success.
        (Test Case 2) upload_to_s3 called with correct arguments.
        """
        sd = make_sd()
        s3_url = "s3://mybucket/path/output.pkl"

        result = exporters.export_spikedata_to_pickle(sd, s3_url, upload_to_s3=True)

        assert result == s3_url
        mock_upload.assert_called_once()
        call_args = mock_upload.call_args
        assert call_args[0][1] == s3_url
        assert call_args[0][0].endswith(".pkl")

    @patch("SpikeLab.data_loaders.s3_utils.upload_to_s3")
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

        exporters.export_spikedata_to_pickle(
            sd, "s3://bucket/key.pkl", upload_to_s3=True
        )

        assert len(temp_paths) == 1
        assert not os.path.exists(temp_paths[0])


class TestDataExportersEdgeCases:
    """Edge case tests for data export functions."""

    @skip_no_h5py
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

    @skip_no_h5py
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

    @skip_no_h5py
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
