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
import unittest
import pathlib
import sys
from unittest.mock import patch

import numpy as np

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


class BaseExportTest(unittest.TestCase):
    """Base class providing common test data for export tests.

    Creates a simple, deterministic SpikeData object with:
    - 3 units: one with 3 spikes, one with 2 spikes, one empty
    - 25ms total length
    Predictable spike times for easy validation

    This standardized test data ensures consistent behavior across all export formats.
    """

    def make_sd(self) -> SpikeData:
        # Simple deterministic SpikeData
        trains = [
            np.array([5.0, 10.0, 15.0]),  # Unit 0: 3 spikes
            np.array([2.5, 20.0]),  # Unit 1: 2 spikes
            np.array([], float),  # Unit 2: empty (edge case)
        ]
        return SpikeData(trains, length=25.0, metadata={"label": "test"})


@unittest.skipIf(h5py is None, "h5py not installed; skipping HDF5/NWB exporter tests")
class TestHDF5Exporters(BaseExportTest):
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

    def _tmp_h5(self) -> str:
        """Creates a temporary HDF5 file path for testing."""
        fd, path = tempfile.mkstemp(suffix=(".h5"))
        os.close(fd)
        return path

    def tearDown(self) -> None:
        """Clean up temporary files created during tests."""
        for attr in ("_last",):
            p = getattr(self, attr, None)
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def test_export_hdf5_ragged_roundtrip(self):
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
        sd = self.make_sd()
        path = self._tmp_h5()
        self._last = path

        sd.to_hdf5(
            path,
            style="ragged",
            spike_times_unit="s",
        )

        sd2 = loaders.load_spikedata_from_hdf5(
            path,
            spike_times_dataset="spike_times",
            spike_times_index_dataset="spike_times_index",
            spike_times_unit="s",
        )
        # Round-trip equality on trains
        for a, b in zip(sd.train, sd2.train):
            self.assertTrue(np.allclose(a, b))

    def test_export_hdf5_group_roundtrip_samples(self):
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
        sd = self.make_sd()
        path = self._tmp_h5()
        self._last = path

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
            self.assertTrue(np.allclose(q(a), b))

    def test_export_hdf5_paired_roundtrip_ms(self):
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
        - This format is intuitive and matches how many analysis pipelines represent spike data internally.
        - Keeping millisecond units avoids precision loss from time conversions.
        """
        sd = self.make_sd()
        path = self._tmp_h5()
        self._last = path

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
            self.assertTrue(np.allclose(a, b))

    def test_export_hdf5_raster(self):
        """
        Test raster export for binned spike count analysis.

        Tests:
        (Method 1) Export specifies a 5ms bin size for rasterization
        (Method 2) SpikeData.raster() method is used internally to create the count matrix
        (Method 3) Result is a 2D array: units × time bins
        (Test Case 1) Verify exported raster matches SpikeData's own raster() output

        Notes:
        - Raster format enables analyses that require fixed-size inputs (like neural decoders) and
        is the standard format for population dynamics studies.
        """
        sd = self.make_sd()
        path = self._tmp_h5()
        self._last = path

        sd.to_hdf5(
            path, style="raster", raster_dataset="raster", raster_bin_size_ms=5.0
        )
        with h5py.File(path, "r") as f:  # type: ignore
            raster = np.asarray(f["raster"])
        self.assertTrue(np.array_equal(raster, sd.raster(5.0)))

    def test_export_hdf5_with_raw(self):
        """
        Tests export of raw data arrays alongside spike data.

        Tests:
        (Method 1) Creates SpikeData with mock raw voltage data and time arrays
        (Method 2) Exports both spike data (ragged style) and raw data
        (Method 3) Raw time is converted from milliseconds to seconds
        (Test Case 1) Verifies the time conversion was applied correctly to raw_time

        Notes:
        - Validates that continuous raw data (like voltage traces) can be exported alongside spike times with proper time unit conversion.
        - Many analyses require both spike times and the underlying continuous data.
        - This ensures both can be stored together with consistent time bases.

        """
        # Attach raw arrays and export raw dataset/time in seconds
        sd = self.make_sd()
        raw = np.random.randn(2, 10)
        sd = SpikeData(sd.train, length=sd.length, raw_data=raw, raw_time=np.arange(10))
        path = self._tmp_h5()
        self._last = path

        sd.to_hdf5(
            path,
            style="ragged",
            spike_times_unit="s",
            raw_dataset="raw",
            raw_time_dataset="raw_time",
            raw_time_unit="s",
        )
        with h5py.File(path, "r") as f:  # type: ignore
            self.assertTrue(np.allclose(np.asarray(f["raw_time"]), sd.raw_time / 1e3))


@unittest.skipIf(h5py is None, "h5py not installed; skipping NWB exporter tests")
class TestNWBExporters(BaseExportTest):
    """
    Tests for Neurodata Without Borders (NWB) format export.

    NWB is a standardized format for neurophysiology data that uses HDF5 as its
    storage backend. The exporter creates minimal NWB-compatible files that can
    be read by both our custom loader and standard NWB tools.

    These tests focus on the NWB-specific conventions like time units (seconds)
    and dataset organization within the /units group.
    """

    def _tmp_nwb(self) -> str:
        """Creates a temporary NWB file path for testing."""
        fd, path = tempfile.mkstemp(suffix=(".nwb"))
        os.close(fd)
        return path

    def tearDown(self) -> None:
        """Clean up temporary files created during tests."""
        for attr in ("_last",):
            p = getattr(self, attr, None)
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def test_export_nwb_roundtrip(self):
        """
        Tests that SpikeData can be exported to NWB format and
        successfully re-imported with round-trip compatibility / identical spike timing data.

        Tests:
        (Method 1) Export SpikeData using the NWB exporter (uses ragged array format internally)
        (Method 2) Times are automatically converted to seconds (NWB standard)
        (Method 3) Data is organized in /units group with standard dataset names
        (Method 4) Re-import using NWB loader with prefer_pynwb=False (h5py-based)
        (Test Case 1) Verify all spike trains match original within floating-point precision

        Notes:
        - This test ensures NWB format compliance and round-trip data integrity.
        - The use of prefer_pynwb=False tests our h5py-based NWB reader rather than
        the full pynwb library, ensuring compatibility with our minimal NWB export.
        """
        sd = self.make_sd()
        path = self._tmp_nwb()
        self._last = path

        sd.to_nwb(path)
        sd2 = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
        for a, b in zip(sd.train, sd2.train):
            self.assertTrue(np.allclose(a, b))


class TestKiloSortExporters(BaseExportTest):
    """
    Tests for KiloSort/Phy format export.

    KiloSort is a popular spike sorting algorithm that outputs spike times and
    cluster assignments in simple NumPy array format. This format is also used
    by Phy (manual curation GUI) and other spike sorting tools.

    The format consists of:
    spike_times.npy: All spike times (usually in samples)
    spike_clusters.npy: Cluster ID for each spike

    These tests validate the export creates properly formatted files and handles
    cluster ID assignment correctly.
    """

    def test_export_kilosort_roundtrip_samples(self):
        """
         Test KiloSort export and import with sample-based timing.

         Test:
         (Method 1) Export SpikeData to KiloSort format with 1000 Hz sampling rate
         (Method 2) Each unit index becomes a cluster ID (0, 1, 2, ...)
         (Method 3) Spike times are converted from milliseconds to sample indices
         (Method 4) Creates spike_times.npy and spike_clusters.npy files
         (Method 5) Round-trip through KiloSort loader
         (Test Case 1) Verify spike trains match (loader sorts by cluster ID, which matches our order)

        Notes:
         - Tests both the export logic and the assumption that unit indices map directly
         to cluster IDs in ascending order.
        """
        sd = self.make_sd()
        with tempfile.TemporaryDirectory() as d:
            sd.to_kilosort(d, fs_Hz=1000.0)
            # Round-trip through loader
            sd2 = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)

            # trains may be in different order: loader sorts by cluster id ascending
            # Our export uses unit index as cluster id, so order should match sd.train
            # Quantization to 1 kHz samples; compare to quantized originals
            def q(ms):
                samp = np.rint(ms * (1000.0 / 1e3))
                return samp / 1000.0 * 1e3

            for a, b in zip(sd.train, sd2.train):
                self.assertTrue(np.allclose(q(a), b))

    def test_export_kilosort_custom_cluster_ids(self):
        """
        Tests KiloSort export with custom cluster ID assignment.

        Tests:
        (Method 1) Export with custom cluster IDs [10, 5, 7] instead of [0, 1, 2]
        (Method 2) Load the raw NumPy files directly (not through loader)
        (Method 3) Verify that cluster IDs 10 and 5 appear with correct spike counts
        (Method 4) Unit 0 (3 spikes) → cluster 10, Unit 1 (2 spikes) → cluster 5
        (Method 5) Unit 2 (empty) → cluster 7 with 0 spikes (not present in arrays)
        (Test Case 1) Verify the cluster ID mapping works correctly with clusters
        10 and 5, and counts match events per unit (3 and 2)


        Notes:
        - This test ensures the cluster ID mapping works correctly and that
        empty units are handled properly (they don't contribute any spikes
        to the output arrays).
        """
        sd = self.make_sd()
        # Swap cluster ids to ensure mapping is honored
        with tempfile.TemporaryDirectory() as d:
            sd.to_kilosort(d, fs_Hz=1000.0, cluster_ids=[10, 5, 7])
            times = np.load(os.path.join(d, "spike_times.npy"))
            clusters = np.load(os.path.join(d, "spike_clusters.npy"))
            # Check that clusters contain 10 and 5, and counts match events per unit (3 and 2)
            self.assertEqual((clusters == 10).sum(), 3)
            self.assertEqual((clusters == 5).sum(), 2)


class TestPickleExporters(BaseExportTest):
    """
    Tests for pickle export functionality.

    Pickle is a Python-native serialization format. These tests validate:
    - Round-trip integrity through export and load
    - Protocol parameter handling for Python version compatibility
    - S3 upload flow when upload_to_s3=True
    - Temporary file cleanup after S3 upload
    """

    def _tmp_pkl(self) -> str:
        """Creates a temporary pickle file path for testing."""
        fd, path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        return path

    def tearDown(self) -> None:
        """Clean up temporary files created during tests."""
        for attr in ("_last",):
            p = getattr(self, attr, None)
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def test_export_pickle_roundtrip(self):
        """
        Tests basic pickle export and import round-trip.

        Tests:
        (Method 1) Export SpikeData to pickle using export_spikedata_to_pickle
        (Method 2) Re-import using load_spikedata_from_pickle
        (Test Case 1) Verify all spike trains match original
        (Test Case 2) Verify metadata is preserved
        """
        sd = self.make_sd()
        path = self._tmp_pkl()
        self._last = path

        # Export SpikeData to pickle file
        exporters.export_spikedata_to_pickle(sd, path)
        # Re-import and verify spike trains match
        sd2 = loaders.load_spikedata_from_pickle(path)
        for a, b in zip(sd.train, sd2.train):
            self.assertTrue(np.allclose(a, b))
        # Verify metadata is preserved
        self.assertEqual(sd.metadata, sd2.metadata)

    def test_export_pickle_protocol(self):
        """
        Tests protocol parameter is passed through correctly.

        Tests:
        (Method 1) Export with protocol=2 (Python 2.3+ compatible)
        (Method 2) Re-import and verify round-trip works
        (Test Case 1) Lower protocols produce loadable files
        """
        sd = self.make_sd()
        path = self._tmp_pkl()
        self._last = path

        # Export with protocol=2 for backward compatibility
        exporters.export_spikedata_to_pickle(sd, path, protocol=2)
        # Verify lower protocol files are loadable and round-trip correctly
        sd2 = loaders.load_spikedata_from_pickle(path)
        for a, b in zip(sd.train, sd2.train):
            self.assertTrue(np.allclose(a, b))

    @patch("SpikeLab.data_loaders.s3_utils.upload_to_s3")
    def test_export_pickle_s3_upload(self, mock_upload):
        """
        Tests S3 upload flow when upload_to_s3=True.

        Tests:
        (Method 1) Export with upload_to_s3=True and S3 URL
        (Method 2) Verify _upload_to_s3 is called with temp path and S3 URL
        (Test Case 1) Returns S3 URL on success
        """
        sd = self.make_sd()
        s3_url = "s3://mybucket/path/output.pkl"

        # Export with S3 upload; upload_to_s3 is mocked so no real AWS call
        result = exporters.export_spikedata_to_pickle(sd, s3_url, upload_to_s3=True)

        # Verify return value is the S3 URL
        self.assertEqual(result, s3_url)
        # Verify upload was called exactly once
        mock_upload.assert_called_once()
        call_args = mock_upload.call_args
        # Verify second arg (s3_url) matches
        self.assertEqual(call_args[0][1], s3_url)
        # Verify first arg (temp path) ends with .pkl
        self.assertTrue(call_args[0][0].endswith(".pkl"))

    @patch("SpikeLab.data_loaders.s3_utils.upload_to_s3")
    def test_export_pickle_temp_cleanup(self, mock_upload):
        """
        Tests temporary file is removed after S3 upload.

        Tests:
        (Method 1) Export with upload_to_s3=True
        (Method 2) Capture temp path passed to _upload_to_s3
        (Test Case 1) Temp file does not exist after export completes
        """
        sd = self.make_sd()
        temp_paths = []

        # Side effect captures the temp path passed to upload_to_s3
        def capture_temp(local_path, s3_url, **kwargs):
            temp_paths.append(local_path)

        mock_upload.side_effect = capture_temp

        # Export triggers temp file creation, upload, then cleanup in finally block
        exporters.export_spikedata_to_pickle(
            sd, "s3://bucket/key.pkl", upload_to_s3=True
        )

        # Verify exactly one temp file was created
        self.assertEqual(len(temp_paths), 1)
        # Verify temp file was removed (cleanup in finally block)
        self.assertFalse(os.path.exists(temp_paths[0]))


if __name__ == "__main__":
    unittest.main()
