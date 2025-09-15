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

import numpy as np

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

# Ensure project root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spikedata import SpikeData
import data_loaders.data_loaders as loaders
import data_loaders.data_exporters as exporters


class BaseExportTest(unittest.TestCase):
    """Base class providing common test data for export tests.

    Creates a simple, deterministic SpikeData object with:
    - 3 units: one with 3 spikes, one with 2 spikes, one empty
    - 25ms total length
    - Predictable spike times for easy validation

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
    """Tests for HDF5 export functionality across all four supported styles.

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
        """Create a temporary HDF5 file path for testing."""
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
        """Test ragged array export/import with time unit conversion.

        Purpose: Validates the most common HDF5 export format (ragged arrays)
        with time unit conversion from milliseconds to seconds.

        Why useful: Ragged arrays are the most storage-efficient format for
        sparse spike data and are used by many analysis tools including NWB.

        How it works:
        1. Export SpikeData to HDF5 using ragged style with seconds time unit
        2. Re-import using the HDF5 loader with matching parameters
        3. Verify all spike trains match the original within floating-point precision

        The test ensures both the export logic and time unit conversion work correctly.
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
        """Test group-per-unit export with sample-based time units.

        Purpose: Validates the group export style with conversion to sample indices,
        which is useful when working with electrophysiology data at native sampling rates.

        Why useful: The group style makes it easy to access individual units without
        parsing index arrays, and sample units preserve exact timing relationships
        with the original recording.

        How it works:
        1. Export using group style with 1000 Hz sampling rate (1 sample = 1 ms)
        2. Each unit gets its own dataset within the "units" group
        3. Spike times are converted from milliseconds to sample indices
        4. Round-trip through loader verifies conversion accuracy

        Tests both the group organization and sample-based time conversion.
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
        """Test paired arrays export with millisecond time units.

        Purpose: Validates the paired arrays format where unit indices and spike times
        are stored in separate parallel arrays, keeping original millisecond timing.

        Why useful: This format is intuitive and matches how many analysis pipelines
        represent spike data internally. Keeping millisecond units avoids precision
        loss from time conversions.

        How it works:
        1. Export creates two datasets: unit indices and corresponding spike times
        2. Empty units are handled by simply not including them in the arrays
        3. Times remain in milliseconds (no conversion)
        4. Round-trip verifies the pairing logic works correctly

        Tests the paired array generation and empty unit handling.
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
        """Test raster export for binned spike count analysis.

        Purpose: Validates conversion of spike trains to binned count matrices,
        which are essential for rate-based analyses and many machine learning applications.

        Why useful: Raster format enables analyses that require fixed-size inputs
        (like neural decoders) and is the standard format for population dynamics studies.

        How it works:
        1. Export specifies a 5ms bin size for rasterization
        2. SpikeData.raster() method is used internally to create the count matrix
        3. Result is a 2D array: units × time bins
        4. Test verifies exported raster matches SpikeData's own raster() output

        This ensures consistency between the export function and SpikeData's built-in
        rasterization method.
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
        """Test export of raw data arrays alongside spike data.

        Purpose: Validates that continuous raw data (like voltage traces) can be
        exported alongside spike times with proper time unit conversion.

        Why useful: Many analyses require both spike times and the underlying
        continuous data. This ensures both can be stored together with consistent
        time bases.

        How it works:
        1. Creates SpikeData with mock raw voltage data and time arrays
        2. Exports both spike data (ragged style) and raw data
        3. Raw time is converted from milliseconds to seconds
        4. Verifies the time conversion was applied correctly to raw_time

        Tests the optional raw data export functionality and time unit conversion
        for continuous data.
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
    """Tests for Neurodata Without Borders (NWB) format export.

    NWB is a standardized format for neurophysiology data that uses HDF5 as its
    storage backend. The exporter creates minimal NWB-compatible files that can
    be read by both our custom loader and standard NWB tools.

    These tests focus on the NWB-specific conventions like time units (seconds)
    and dataset organization within the /units group.
    """

    def _tmp_nwb(self) -> str:
        """Create a temporary NWB file path for testing."""
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
        """Test NWB export and import round-trip compatibility.

        Purpose: Validates that SpikeData can be exported to NWB format and
        successfully re-imported with identical spike timing data.

        Why useful: NWB is becoming the standard for sharing neurophysiology data.
        This ensures our export creates valid NWB files that maintain data integrity
        and can be used with other NWB-compatible tools.

        How it works:
        1. Export SpikeData using the NWB exporter (uses ragged array format internally)
        2. Times are automatically converted to seconds (NWB standard)
        3. Data is organized in /units group with standard dataset names
        4. Re-import using NWB loader with prefer_pynwb=False (h5py-based)
        5. Verify all spike trains match original within floating-point precision

        This test ensures NWB format compliance and round-trip data integrity.
        The use of prefer_pynwb=False tests our h5py-based NWB reader rather than
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
    """Tests for KiloSort/Phy format export.

    KiloSort is a popular spike sorting algorithm that outputs spike times and
    cluster assignments in simple NumPy array format. This format is also used
    by Phy (manual curation GUI) and other spike sorting tools.

    The format consists of:
    - spike_times.npy: All spike times (usually in samples)
    - spike_clusters.npy: Cluster ID for each spike

    These tests validate the export creates properly formatted files and handles
    cluster ID assignment correctly.
    """

    def test_export_kilosort_roundtrip_samples(self):
        """Test KiloSort export and import with sample-based timing.

        Purpose: Validates that SpikeData can be exported to KiloSort format
        and re-imported with correct spike timing and unit assignment.

        Why useful: KiloSort format is widely used in the spike sorting community.
        This ensures compatibility with KiloSort, Phy, and other tools that use
        this simple but effective format.

        How it works:
        1. Export SpikeData to KiloSort format with 1000 Hz sampling rate
        2. Each unit index becomes a cluster ID (0, 1, 2, ...)
        3. Spike times are converted from milliseconds to sample indices
        4. Creates spike_times.npy and spike_clusters.npy files
        5. Round-trip through KiloSort loader
        6. Verify spike trains match (loader sorts by cluster ID, which matches our order)

        Tests both the export logic and the assumption that unit indices map directly
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
        """Test KiloSort export with custom cluster ID assignment.

        Purpose: Validates that custom cluster IDs can be assigned to units
        instead of using the default unit index mapping.

        Why useful: Sometimes you want to preserve original cluster IDs from
        a spike sorting result, or use a specific numbering scheme that doesn't
        start from 0. This flexibility is important for data provenance.

        How it works:
        1. Export with custom cluster IDs [10, 5, 7] instead of [0, 1, 2]
        2. Load the raw NumPy files directly (not through loader)
        3. Verify that cluster IDs 10 and 5 appear with correct spike counts
        4. Unit 0 (3 spikes) → cluster 10, Unit 1 (2 spikes) → cluster 5
        5. Unit 2 (empty) → cluster 7 with 0 spikes (not present in arrays)

        This test ensures the cluster ID mapping works correctly and that
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


if __name__ == "__main__":
    unittest.main()
