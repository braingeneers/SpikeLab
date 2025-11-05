"""
Tests for data_loaders -> SpikeData conversion.

These tests use small temporary files and skip format-specific tests
if optional dependencies are not available (e.g., h5py).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Optional

import numpy as np

try:  # optional, only needed for HDF5/NWB tests
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

import pathlib
import sys

# Ensure project root is on sys.path, mirroring other tests
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spikedata import SpikeData
import data_loaders.data_loaders as loaders
import data_loaders.data_exporters as exporters
import pickle


@unittest.skipIf(h5py is None, "h5py not installed; skipping HDF5/NWB tests")
class TestHDF5Loaders(unittest.TestCase):
    def _tmp_h5(self) -> str:
        """Create a temporary HDF5 file and return its path."""
        fd, path = tempfile.mkstemp(suffix=".h5")
        os.close(fd)
        return path

    def tearDown(self) -> None:
        """Remove any temporary HDF5 files created during the test."""
        for attr in ("_last_h5_path",):
            path: Optional[str] = getattr(self, attr, None)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def test_hdf5_raster(self):
        """
        Test loading a 2D raster dataset from HDF5.

        Creates a small 2D integer array, writes it as 'raster' to HDF5,
        loads it using load_spikedata_from_hdf5, and checks that the
        resulting SpikeData object has the correct raster and unit count.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        raster = np.array([[0, 2, 0, 1], [1, 0, 0, 0]], dtype=int)
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=raster)

        sd = loaders.load_spikedata_from_hdf5(
            path, raster_dataset="raster", raster_bin_size_ms=10.0
        )
        self.assertIsInstance(sd, SpikeData)
        # Note: raster may have an extra bin due to edge case handling when
        # length is an exact multiple of bin_size, so we check the first N bins
        loaded_raster = sd.raster(10.0)
        self.assertTrue(
            np.all(loaded_raster[:, : raster.shape[1]] == raster),
            "Raster content doesn't match",
        )
        self.assertEqual(sd.N, raster.shape[0])

    def test_hdf5_raster_not_2d_raises(self):
        """
        Test that loading a non-2D raster dataset raises ValueError.

        Writes a 1D array as 'raster' and checks that load_spikedata_from_hdf5
        raises a ValueError due to incorrect shape.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=np.array([0, 1, 2]))
        with self.assertRaises(ValueError):
            loaders.load_spikedata_from_hdf5(
                path, raster_dataset="raster", raster_bin_size_ms=10.0
            )

    def test_hdf5_multiple_styles_raises(self):
        """
        Test that specifying multiple input styles raises ValueError.

        Writes both a 'raster' dataset and a 'units' group, then attempts to
        load with both raster and group_per_unit arguments, expecting a ValueError.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=np.zeros((1, 2)))
            f.create_group("units")
        with self.assertRaises(ValueError):
            loaders.load_spikedata_from_hdf5(
                path,
                raster_dataset="raster",
                raster_bin_size_ms=1.0,
                group_per_unit="units",
            )

    def test_hdf5_idces_times_ms(self):
        """
        Test loading spike indices and times in milliseconds from HDF5.

        Writes 'idces' and 'times' datasets, loads them, and checks that
        the SpikeData object returns the same indices and times.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        idces = np.array([0, 1, 0, 1], dtype=int)
        times_ms = np.array([5.0, 10.0, 15.0, 20.0])
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("idces", data=idces)
            f.create_dataset("times", data=times_ms)

        sd = loaders.load_spikedata_from_hdf5(
            path, idces_dataset="idces", times_dataset="times", times_unit="ms"
        )
        id2, t2 = sd.idces_times()
        self.assertTrue(np.all(id2 == idces))
        self.assertTrue(np.allclose(t2, times_ms))

    def test_hdf5_group_per_unit_seconds(self):
        """
        Test loading spike times from a group-per-unit structure in seconds.

        Writes a 'units' group with two datasets (one per unit) containing
        spike times in seconds, loads them, and checks that the times are
        correctly converted to milliseconds.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        with h5py.File(path, "w") as f:  # type: ignore
            g = f.create_group("units")
            g.create_dataset("0", data=np.array([0.1, 0.2]))
            g.create_dataset("1", data=np.array([0.05]))

        sd = loaders.load_spikedata_from_hdf5(
            path, group_per_unit="units", group_time_unit="s"
        )
        # Expect ms
        self.assertTrue(np.allclose(sd.train[0], np.array([100.0, 200.0])))
        self.assertTrue(np.allclose(sd.train[1], np.array([50.0])))

    def test_hdf5_group_per_unit_empty_units(self):
        """
        Test loading group-per-unit structure with empty units.

        Writes a 'units' group with two empty datasets, loads them, and checks
        that the resulting SpikeData object has two units, zero length, and
        empty spike trains.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        with h5py.File(path, "w") as f:  # type: ignore
            g = f.create_group("units")
            g.create_dataset("0", data=np.array([]))
            g.create_dataset("1", data=np.array([]))
        sd = loaders.load_spikedata_from_hdf5(
            path, group_per_unit="units", group_time_unit="ms"
        )
        self.assertEqual(sd.N, 2)
        self.assertEqual(sd.length, 0.0)
        self.assertEqual(len(sd.train[0]), 0)
        self.assertEqual(len(sd.train[1]), 0)

    def test_hdf5_flat_ragged_spike_times(self):
        """
        Test loading ragged spike times from flat arrays and index.

        Writes a flat 'spike_times' array and a 'spike_times_index' array
        indicating the end of each unit's spike times, loads them, and checks
        that the times are correctly split and converted to ms.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
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
        self.assertTrue(np.allclose(sd.train[0], [100.0, 200.0]))
        self.assertTrue(np.allclose(sd.train[1], [500.0]))

    def test_hdf5_idces_times_samples_with_fs(self):
        """
        Test loading spike indices and times in samples with specified sampling rate.

        Writes 'idces' and 'times' datasets (times in samples), loads them with
        fs_Hz=1000, and checks that times are correctly converted to ms.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
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
        # samples @1kHz => ms equal to samples
        self.assertTrue(np.allclose(sd.train[0], [100.0, 300.0]))
        self.assertTrue(np.allclose(sd.train[1], [200.0]))

    def test_hdf5_raw_attachment_seconds_and_samples(self):
        """
        Test loading and attaching raw data and raw time from HDF5.

        Writes a 'raster', 'raw', and two raw time datasets (one in seconds,
        one in samples). Loads both, checking that raw_time is correctly
        converted to ms in both cases.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
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
        self.assertEqual(sd_s.raw_data.shape, (2, 5))
        self.assertTrue(np.allclose(sd_s.raw_time, np.arange(5) * 1.0))

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
        self.assertTrue(np.allclose(sd_p.raw_time, np.arange(5) * 1.0))

    def test_hdf5_invalid_style_error(self):
        """
        Test that loading from an HDF5 file with no recognizable style raises ValueError.

        Creates an empty HDF5 file and checks that load_spikedata_from_hdf5
        raises a ValueError due to missing required datasets/groups.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        with h5py.File(path, "w") as _:  # type: ignore
            pass
        with self.assertRaises(ValueError):
            loaders.load_spikedata_from_hdf5(path)  # no style specified

    def test_hdf5_samples_without_fs_error(self):
        """
        Test that loading times in samples without specifying fs_Hz raises ValueError.

        Writes 'idces' and 'times' (in samples) but omits fs_Hz, expecting
        load_spikedata_from_hdf5 to raise a ValueError.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        idces = np.array([0, 0, 1])
        times_samples = np.array([10, 20, 30])
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("idces", data=idces)
            f.create_dataset("times", data=times_samples)
        with self.assertRaises(ValueError):
            loaders.load_spikedata_from_hdf5(
                path, idces_dataset="idces", times_dataset="times", times_unit="samples"
            )

    def test_hdf5_raw_thresholded(self):
        """
        Test thresholding of raw data loaded from HDF5.

        Writes a 'raw' dataset with two channels, one containing a supra-threshold
        segment. Loads and thresholds the data, checking that at least one event
        is detected on the active channel.
        """
        path = self._tmp_h5()
        self._last_h5_path = path
        # two channels, one has a supra-threshold segment
        data = np.zeros((2, 50), dtype=float)
        data[0, 10:13] = 10.0
        data[1, :] = 0.0
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
        self.assertIsInstance(sd, SpikeData)
        self.assertEqual(sd.N, 2)
        # should detect at least one event on channel 0
        self.assertTrue(len(sd.train[0]) >= 1)


@unittest.skipIf(h5py is None, "h5py not installed; skipping NWB tests")
class TestNWBLoader(unittest.TestCase):
    def test_nwb_units_via_h5py(self):
        """
        Test loading NWB units group using h5py.

        Creates a minimal NWB-like file with a 'units' group containing
        'spike_times' and 'spike_times_index', loads it, and checks that
        spike trains are correctly split and converted to ms.
        """
        fd, path = tempfile.mkstemp(suffix=".nwb")
        os.close(fd)
        try:
            # minimal NWB-like units group
            with h5py.File(path, "w") as f:  # type: ignore
                g = f.create_group("units")
                g.create_dataset("spike_times", data=np.array([0.1, 0.2, 0.5]))
                g.create_dataset("spike_times_index", data=np.array([2, 3]))

            sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
            self.assertTrue(np.allclose(sd.train[0], [100.0, 200.0]))
            self.assertTrue(np.allclose(sd.train[1], [500.0]))
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_nwb_missing_units_raises(self):
        """
        Test that loading an NWB file missing the 'units' group raises ValueError.

        Creates an empty NWB file and checks that load_spikedata_from_nwb
        raises a ValueError due to missing 'units'.
        """
        fd, path = tempfile.mkstemp(suffix=".nwb")
        os.close(fd)
        try:
            with h5py.File(path, "w") as _:  # type: ignore
                pass
            with self.assertRaises(ValueError):
                loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_nwb_alt_names_with_endswith(self):
        """
        Test loading NWB units group with alternative dataset names.

        Creates a 'units' group with datasets ending in 'spike_times' and
        'spike_times_index' but with prefixes, loads it, and checks that
        spike trains are correctly parsed and converted to ms.
        """
        fd, path = tempfile.mkstemp(suffix=".nwb")
        os.close(fd)
        try:
            with h5py.File(path, "w") as f:  # type: ignore
                g = f.create_group("units")
                g.create_dataset("xx_spike_times", data=np.array([0.2, 0.7]))
                g.create_dataset("xx_spike_times_index", data=np.array([1, 2]))

            sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
            self.assertTrue(np.allclose(sd.train[0], [200.0]))
            self.assertTrue(np.allclose(sd.train[1], [700.0]))
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


class TestKiloSortAndSpikeInterface(unittest.TestCase):
    def test_kilosort_basic(self):
        """
        Test loading KiloSort output with two clusters.

        Creates 'spike_times.npy' and 'spike_clusters.npy' for two clusters,
        loads them, and checks that the cluster_ids metadata matches the trains,
        and that spike times are correctly converted to ms and sorted by cluster id.
        """
        with tempfile.TemporaryDirectory() as d:
            # two clusters: 2 spikes in 0, 1 spike in 1
            spike_times = np.array([10, 20, 15])  # samples
            spike_clusters = np.array([0, 0, 1])
            np.save(os.path.join(d, "spike_times.npy"), spike_times)
            np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)

            sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
            # cluster_ids metadata should align with trains
            self.assertEqual(len(sd.train), len(sd.metadata.get("cluster_ids", [])))
            # Expected times in ms
            all_trains_ms = [np.array([10.0, 20.0]), np.array([15.0])]
            # order by cluster id ascending
            for train, truth in zip(sd.train, all_trains_ms):
                self.assertTrue(np.allclose(train, truth))

    def test_spikeinterface_mock(self):
        """
        Test loading from a mock SpikeInterface SortingExtractor.

        Defines a mock sorting object with two units and known spike trains,
        loads it, and checks that spike times are correctly converted to ms
        using the provided sampling frequency.
        """

        class MockSorting:
            def __init__(self):
                self._ids = [10, 20]

            def get_unit_ids(self):
                return self._ids

            def get_sampling_frequency(self):
                return 2000.0

            def get_unit_spike_train(self, unit_id, segment_index=0):
                if unit_id == 10:
                    return np.array([10, 20])
                return np.array([5])

        sorting = MockSorting()
        sd = loaders.load_spikedata_from_spikeinterface(sorting)
        # samples -> ms at 2kHz => 0.5 ms increments
        self.assertTrue(np.allclose(sd.train[0], [5.0, 10.0]))
        self.assertTrue(np.allclose(sd.train[1], [2.5]))

    def test_spikeinterface_base_recording_thresholding(self):
        """
        Test thresholding on a mock SpikeInterface RecordingExtractor.

        Defines a mock recording object with a supra-threshold burst on one channel,
        loads and thresholds it, and checks that at least one event is detected.
        Also tests that time x channels input is transposed automatically.
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
        self.assertEqual(sd.N, 2)
        self.assertTrue(len(sd.train[0]) >= 1)

        # time x channels input gets transposed automatically
        data_tc = data_ct.T
        rec2 = MockRecording(data_tc, fs=1000.0)
        sd2 = loaders.load_spikedata_from_spikeinterface_recording(
            rec2, threshold_sigma=2.0, filter=False, hysteresis=True, direction="up"
        )
        self.assertEqual(sd2.N, 2)
        self.assertTrue(len(sd2.train[0]) >= 1)

    def test_spikeinterface_subset_and_override_fs(self):
        """
        Test loading a subset of units and overriding sampling frequency.

        Defines a mock sorting object with two units and no sampling frequency,
        loads only one unit with a provided sampling frequency, and checks that
        the spike train is correct and in ms.
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
        self.assertEqual(sd.N, 1)
        self.assertTrue(np.allclose(sd.train[0], [0.0, 10.0]))

    def test_spikeinterface_invalid_object_raises(self):
        """
        Test that passing an invalid object to load_spikedata_from_spikeinterface raises TypeError.

        Defines a class with no required methods and checks that loading raises TypeError.
        """

        class BadSorting:
            pass

        with self.assertRaises(TypeError):
            loaders.load_spikedata_from_spikeinterface(BadSorting())

    def test_kilosort_empty_arrays(self):
        """
        Test loading KiloSort output with empty arrays.

        Writes empty 'spike_times.npy' and 'spike_clusters.npy', loads them,
        and checks that the resulting SpikeData object has zero units and zero length.
        """
        with tempfile.TemporaryDirectory() as d:
            np.save(os.path.join(d, "spike_times.npy"), np.array([], dtype=int))
            np.save(os.path.join(d, "spike_clusters.npy"), np.array([], dtype=int))
            sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
            self.assertEqual(sd.N, 0)
            self.assertEqual(sd.length, 0.0)

    def test_kilosort_metadata_cluster_ids_alignment(self):
        """
        Test that KiloSort cluster_ids metadata aligns with sorted trains.

        Writes 'spike_times.npy' and 'spike_clusters.npy' with two cluster ids,
        loads them, and checks that the cluster_ids metadata is sorted and matches
        the order of spike trains.
        """
        with tempfile.TemporaryDirectory() as d:
            spike_times = np.array([10, 20, 15, 30])
            spike_clusters = np.array([5, 5, 3, 5])
            np.save(os.path.join(d, "spike_times.npy"), spike_times)
            np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)
            sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
            # cluster_ids sorted ascending (np.unique order)
            self.assertEqual(sd.metadata.get("cluster_ids"), [3, 5])

    def test_kilosort_tsv_missing_columns_keeps_all(self):
        """
        Test that KiloSort loader keeps all clusters if cluster_info.tsv is missing expected columns.

        Writes 'spike_times.npy', 'spike_clusters.npy', and a cluster_info.tsv file
        without the expected columns, loads them, and checks that all clusters are kept.
        """
        with tempfile.TemporaryDirectory() as d:
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
            self.assertEqual(len(sd.train), 2)

    def test_kilosort_neuron_attributes_extraction(self):
        """
        Test comprehensive neuron attributes extraction from KiloSort/Phy files.

        Creates a mock KiloSort directory with templates, channel_positions,
        amplitudes, and spike data. Loads it with extract_attributes=True and
        verifies that all expected attributes are extracted correctly.
        """
        with tempfile.TemporaryDirectory() as d:
            # Create spike data: 2 clusters
            # Cluster 0: spikes at times [10, 20, 30] (samples)
            # Cluster 1: spikes at times [1, 2005] (samples) - second has ISI > 2ms, first violates
            spike_times = np.array([10, 20, 30, 1, 2005])
            spike_clusters = np.array([0, 0, 0, 1, 1])
            np.save(os.path.join(d, "spike_times.npy"), spike_times)
            np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)

            # Create templates: (n_templates=2, n_samples=82, n_channels=4)
            # Use more realistic waveform shapes based on actual KiloSort data
            templates = np.zeros((2, 82, 4))

            # Template 0: peak on channel 2 - realistic spike waveform
            wf0 = np.zeros(82)
            # Add baseline noise
            wf0[:16] = np.random.randn(16) * 0.5
            wf0[-16:] = np.random.randn(16) * 0.5
            # Add spike shape (inverted)
            peak_idx = 41
            for i, t in enumerate(range(-10, 20)):
                if 0 <= peak_idx + t < 82:
                    wf0[peak_idx + t] += -10 * np.exp(-(t**2) / 20.0)
            templates[0, :, 2] = wf0

            # Template 1: peak on channel 1 - realistic spike waveform
            wf1 = np.zeros(82)
            # Add baseline noise
            wf1[:16] = np.random.randn(16) * 0.5
            wf1[-16:] = np.random.randn(16) * 0.5
            # Add spike shape (inverted)
            for i, t in enumerate(range(-8, 18)):
                if 0 <= peak_idx + t < 82:
                    wf1[peak_idx + t] += -8 * np.exp(-(t**2) / 15.0)
            templates[1, :, 1] = wf1

            np.save(os.path.join(d, "templates.npy"), templates)

            # Create spike_templates: maps each spike to a template
            # Cluster 0 uses template 0, cluster 1 uses template 1
            spike_templates = np.array([0, 0, 0, 1, 1])
            np.save(os.path.join(d, "spike_templates.npy"), spike_templates)

            # Create channel positions: (n_channels=4, 2) for (x, y)
            channel_positions = np.array(
                [
                    [0.0, 100.0],  # channel 0
                    [100.0, 100.0],  # channel 1
                    [0.0, 200.0],  # channel 2
                    [100.0, 200.0],  # channel 3
                ]
            )
            np.save(os.path.join(d, "channel_positions.npy"), channel_positions)

            # Create amplitudes: amplitude for each spike
            amplitudes = np.array([1.0, 1.2, 1.1, 0.5, 0.6])
            np.save(os.path.join(d, "amplitudes.npy"), amplitudes)

            # Create cluster_info.tsv with additional metadata
            with open(os.path.join(d, "cluster_info.tsv"), "w") as f:
                f.write("cluster_id\tgroup\tquality_metric\n")
                f.write("0\tgood\t0.95\n")
                f.write("1\tgood\t0.87\n")

            # Load with attribute extraction
            sd = loaders.load_spikedata_from_kilosort(
                d,
                fs_Hz=1000.0,
                cluster_info_tsv="cluster_info.tsv",
                extract_attributes=True,
            )

            # Verify basic loading
            self.assertEqual(sd.N, 2)
            self.assertEqual(len(sd.train), 2)

            # Verify neuron_attributes exist
            self.assertIsNotNone(sd.neuron_attributes)
            attrs = sd.neuron_attributes.to_dataframe()

            # Verify shape
            self.assertEqual(len(attrs), 2)

            # Verify cluster_id
            self.assertEqual(list(attrs["cluster_id"]), [0, 1])

            # Verify channel extraction (peak channel)
            self.assertEqual(attrs.loc[0, "channel"], 2)  # template 0 peaks on ch 2
            self.assertEqual(attrs.loc[1, "channel"], 1)  # template 1 peaks on ch 1

            # Verify electrode (should equal channel)
            self.assertEqual(attrs.loc[0, "electrode"], 2)
            self.assertEqual(attrs.loc[1, "electrode"], 1)

            # Verify x, y coordinates
            self.assertEqual(attrs.loc[0, "x"], 0.0)  # channel 2: (0, 200)
            self.assertEqual(attrs.loc[0, "y"], 200.0)
            self.assertEqual(attrs.loc[1, "x"], 100.0)  # channel 1: (100, 100)
            self.assertEqual(attrs.loc[1, "y"], 100.0)

            # Verify average_waveform is extracted
            self.assertIn("average_waveform", attrs.columns)
            wf0 = attrs.loc[0, "average_waveform"]
            wf1 = attrs.loc[1, "average_waveform"]
            self.assertIsNotNone(wf0)
            self.assertIsNotNone(wf1)
            self.assertEqual(len(wf0), 82)  # waveform has 82 samples
            self.assertEqual(len(wf1), 82)

            # Verify SNR is calculated
            self.assertIn("snr", attrs.columns)
            self.assertFalse(np.isnan(attrs.loc[0, "snr"]))
            self.assertFalse(np.isnan(attrs.loc[1, "snr"]))
            self.assertTrue(attrs.loc[0, "snr"] > 0)
            self.assertTrue(attrs.loc[1, "snr"] > 0)

            # Verify amplitude
            self.assertIn("amplitude", attrs.columns)
            # Cluster 0: mean of [1.0, 1.2, 1.1] = 1.1
            self.assertAlmostEqual(attrs.loc[0, "amplitude"], 1.1, places=5)
            # Cluster 1: mean of [0.5, 0.6] = 0.55
            self.assertAlmostEqual(attrs.loc[1, "amplitude"], 0.55, places=5)

            # Verify ISI violations
            self.assertIn("isi_violations", attrs.columns)
            # Cluster 0: ISIs = [10ms, 10ms] -> no violations
            self.assertEqual(attrs.loc[0, "isi_violations"], 0.0)
            # Cluster 1: ISIs = [2004ms] -> no violations (>2ms)
            self.assertEqual(attrs.loc[1, "isi_violations"], 0.0)

            # Verify TSV data is included
            self.assertIn("group", attrs.columns)
            self.assertIn("quality_metric", attrs.columns)
            self.assertEqual(attrs.loc[0, "group"], "good")
            self.assertEqual(attrs.loc[1, "group"], "good")
            self.assertAlmostEqual(attrs.loc[0, "quality_metric"], 0.95, places=5)
            self.assertAlmostEqual(attrs.loc[1, "quality_metric"], 0.87, places=5)

    def test_kilosort_isi_violations_calculation(self):
        """
        Test ISI violations calculation with spikes that violate the 2ms refractory period.
        """
        with tempfile.TemporaryDirectory() as d:
            # Create spike data with ISI violations
            # Cluster 0: spikes at [0, 1, 1000] samples at 1000Hz = [0, 1, 1000]ms
            # ISIs: [1ms (violation), 999ms] -> 1/2 = 0.5 violation rate
            spike_times = np.array([0, 1, 1000])
            spike_clusters = np.array([0, 0, 0])
            np.save(os.path.join(d, "spike_times.npy"), spike_times)
            np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)

            # Load with attribute extraction
            sd = loaders.load_spikedata_from_kilosort(
                d, fs_Hz=1000.0, extract_attributes=True
            )

            # Verify ISI violation rate
            attrs = sd.neuron_attributes.to_dataframe()
            self.assertIn("isi_violations", attrs.columns)
            # ISIs: [1ms, 999ms] -> 1 violation out of 2 ISIs = 0.5
            self.assertAlmostEqual(attrs.loc[0, "isi_violations"], 0.5, places=5)

    def test_kilosort_extract_attributes_false(self):
        """
        Test that setting extract_attributes=False skips attribute extraction.
        """
        with tempfile.TemporaryDirectory() as d:
            spike_times = np.array([10, 20])
            spike_clusters = np.array([0, 0])
            np.save(os.path.join(d, "spike_times.npy"), spike_times)
            np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)

            sd = loaders.load_spikedata_from_kilosort(
                d, fs_Hz=1000.0, extract_attributes=False
            )

            # Verify that neuron_attributes is None
            self.assertIsNone(sd.neuron_attributes)


class TestS3AndACQMLoaders(unittest.TestCase):
    """Tests for S3 download functionality and ACQM file loading."""

    def test_download_s3_invalid_uri_raises(self):
        """
        Test that download_s3_to_local raises RuntimeError for non-S3 URIs.

        Verifies that the function rejects URIs that don't start with 's3://'.
        """
        with self.assertRaises(RuntimeError) as ctx:
            loaders.download_s3_to_local("http://example.com/file.h5", "/tmp/out.h5")
        self.assertIn("must start with s3://", str(ctx.exception))

    def test_acqm_basic_local_file(self):
        """
        Test loading a basic ACQM file with minimal required fields.

        Creates a temporary NPZ file with 'train' and 'fs' fields, loads it,
        and verifies that spike times are correctly converted from samples to ms.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal ACQM file (np.savez automatically adds .npz extension)
            acqm_base = os.path.join(tmpdir, "test")

            # Two units with spike times in samples
            train_dict = {
                0: np.array([10, 20, 30]),  # samples
                1: np.array([5, 15]),
            }
            fs_Hz = 1000.0  # 1 kHz -> samples equal ms

            np.savez(
                acqm_base,
                train=train_dict,
                fs=fs_Hz,
            )

            # np.savez creates test.npz
            sd = loaders.load_spikedata_from_acqm(acqm_base + ".npz")

            # Verify basic properties
            self.assertIsInstance(sd, SpikeData)
            self.assertEqual(sd.N, 2)

            # Verify spike times converted correctly (samples -> ms at 1kHz)
            self.assertTrue(np.allclose(sd.train[0], [10.0, 20.0, 30.0]))
            self.assertTrue(np.allclose(sd.train[1], [5.0, 15.0]))

    def test_acqm_with_neuron_attributes(self):
        """
        Test loading ACQM file with neuron_data attributes.

        Creates an ACQM file with neuron metadata including channel, electrode,
        waveforms, and amplitudes. Verifies that attributes are correctly extracted
        and processed (waveforms averaged, amplitudes averaged).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            acqm_base = os.path.join(tmpdir, "test_attrs")

            # Create train data
            train_dict = {
                0: np.array([100, 200]),
                1: np.array([150]),
            }

            # Create neuron_data with metadata
            neuron_data = {
                0: {
                    "channel": 5,
                    "electrode": 5,
                    "waveforms": np.array(
                        [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]
                    ),  # 2 waveforms
                    "amplitudes": np.array([10.0, 12.0]),  # 2 amplitudes
                    "position": np.array([100.0, 200.0]),
                },
                1: {
                    "channel": 3,
                    "electrode": 3,
                    "waveforms": np.array([5.0, 6.0, 7.0]),  # single waveform
                    "amplitudes": np.array([8.0, 9.0, 10.0]),
                },
            }

            np.savez(
                acqm_base,
                train=train_dict,
                fs=2000.0,  # 2 kHz
                neuron_data=neuron_data,
            )

            sd = loaders.load_spikedata_from_acqm(acqm_base + ".npz")

            # Verify neuron_attributes exist
            self.assertIsNotNone(sd.neuron_attributes)
            attrs = sd.neuron_attributes.to_dataframe()

            # Verify shape
            self.assertEqual(len(attrs), 2)

            # Verify channel/electrode
            self.assertEqual(attrs.loc[0, "channel"], 5)
            self.assertEqual(attrs.loc[0, "electrode"], 5)
            self.assertEqual(attrs.loc[1, "channel"], 3)
            self.assertEqual(attrs.loc[1, "electrode"], 3)

            # Verify average waveform
            self.assertIn("avg_waveform", attrs.columns)
            wf0 = attrs.loc[0, "avg_waveform"]
            self.assertTrue(np.allclose(wf0, [1.25, 2.25, 3.25]))  # mean of 2 waveforms

            # Verify average amplitude
            self.assertIn("avg_amplitude", attrs.columns)
            self.assertAlmostEqual(
                attrs.loc[0, "avg_amplitude"], 11.0, places=5
            )  # mean of [10, 12]
            self.assertAlmostEqual(
                attrs.loc[1, "avg_amplitude"], 9.0, places=5
            )  # mean of [8, 9, 10]

            # Verify position is stored as tuple
            self.assertIn("position", attrs.columns)
            self.assertEqual(attrs.loc[0, "position"], (100.0, 200.0))

    def test_acqm_empty_units(self):
        """
        Test loading ACQM file with empty spike trains.

        Creates an ACQM file with two units that have no spikes, verifies that
        the resulting SpikeData has correct unit count and zero length.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            acqm_base = os.path.join(tmpdir, "empty")

            train_dict = {
                0: np.array([]),
                1: np.array([]),
            }

            np.savez(acqm_base, train=train_dict, fs=1000.0)

            sd = loaders.load_spikedata_from_acqm(acqm_base + ".npz")

            self.assertEqual(sd.N, 2)
            self.assertEqual(sd.length, 0.0)
            self.assertEqual(len(sd.train[0]), 0)
            self.assertEqual(len(sd.train[1]), 0)

    def test_acqm_missing_train_raises(self):
        """
        Test that loading an ACQM file without 'train' field raises ValueError.

        Creates an NPZ file with only 'fs' field and verifies that loading
        raises a ValueError due to missing 'train'.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            acqm_base = os.path.join(tmpdir, "no_train")
            np.savez(acqm_base, fs=1000.0)

            with self.assertRaises(ValueError) as ctx:
                loaders.load_spikedata_from_acqm(acqm_base + ".npz")
            self.assertIn("missing required 'train' field", str(ctx.exception))

    def test_acqm_missing_fs_raises(self):
        """
        Test that loading an ACQM file without 'fs' field raises ValueError.

        Creates an NPZ file with only 'train' field and verifies that loading
        raises a ValueError due to missing 'fs' (sampling frequency).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            acqm_base = os.path.join(tmpdir, "no_fs")
            train_dict = {0: np.array([10, 20])}
            np.savez(acqm_base, train=train_dict)

            with self.assertRaises(ValueError) as ctx:
                loaders.load_spikedata_from_acqm(acqm_base + ".npz")
            self.assertIn("missing required 'fs'", str(ctx.exception))

    def test_acqm_invalid_fs_raises(self):
        """
        Test that loading an ACQM file with invalid sampling frequency raises ValueError.

        Creates an NPZ file with fs <= 0 and verifies that loading raises a ValueError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            acqm_base = os.path.join(tmpdir, "bad_fs")
            train_dict = {0: np.array([10, 20])}
            np.savez(acqm_base, train=train_dict, fs=0.0)

            with self.assertRaises(ValueError) as ctx:
                loaders.load_spikedata_from_acqm(acqm_base + ".npz")
            self.assertIn("Invalid sampling frequency", str(ctx.exception))

    def test_acqm_invalid_train_type_raises(self):
        """
        Test that loading an ACQM file with non-dict 'train' field raises ValueError.

        Creates an NPZ file where 'train' is an array instead of a dict and
        verifies that loading raises a ValueError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            acqm_base = os.path.join(tmpdir, "bad_train")
            # train should be a dict, not an array
            np.savez(acqm_base, train=np.array([1, 2, 3]), fs=1000.0)

            # Should raise ValueError (either "must be a dictionary" or from .item() call on non-scalar)
            with self.assertRaises(ValueError):
                loaders.load_spikedata_from_acqm(acqm_base + ".npz")

    def test_acqm_sampling_frequency_conversion(self):
        """
        Test that spike times are correctly converted from samples to ms at different fs.

        Creates an ACQM file with spike times in samples at 2kHz and verifies
        that they are correctly converted to milliseconds.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            acqm_base = os.path.join(tmpdir, "test_fs")

            # At 2000 Hz, 2000 samples = 1000 ms
            train_dict = {0: np.array([2000, 4000, 6000])}

            np.savez(acqm_base, train=train_dict, fs=2000.0)

            sd = loaders.load_spikedata_from_acqm(acqm_base + ".npz")

            # 2000 samples @ 2kHz = 1000 ms, etc.
            self.assertTrue(np.allclose(sd.train[0], [1000.0, 2000.0, 3000.0]))

    def test_acqm_with_length_ms_parameter(self):
        """
        Test loading ACQM file with explicit length_ms parameter.

        Creates an ACQM file and loads it with a specified length_ms,
        verifying that the length is set correctly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            acqm_base = os.path.join(tmpdir, "test_length")
            train_dict = {0: np.array([100, 200])}
            np.savez(acqm_base, train=train_dict, fs=1000.0)

            sd = loaders.load_spikedata_from_acqm(acqm_base + ".npz", length_ms=5000.0)

            # Verify the length is set as specified
            self.assertEqual(sd.length, 5000.0)

    def test_acqm_unit_id_sorting(self):
        """
        Test that units are sorted by ID for consistent ordering.

        Creates an ACQM file with unsorted unit IDs and verifies that
        spike trains are ordered by sorted unit IDs.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            acqm_base = os.path.join(tmpdir, "sorted")

            # Units in non-sorted order
            train_dict = {
                5: np.array([50.0]),
                1: np.array([10.0]),
                3: np.array([30.0]),
            }

            np.savez(acqm_base, train=train_dict, fs=1000.0)

            sd = loaders.load_spikedata_from_acqm(acqm_base + ".npz")

            # Should be ordered by sorted keys: 1, 3, 5
            self.assertEqual(sd.N, 3)
            self.assertTrue(np.allclose(sd.train[0], [10.0]))  # unit 1
            self.assertTrue(np.allclose(sd.train[1], [30.0]))  # unit 3
            self.assertTrue(np.allclose(sd.train[2], [50.0]))  # unit 5

    def test_acqm_spike_times_are_sorted(self):
        """
        Test that spike times within each train are sorted.

        Creates an ACQM file with unsorted spike times and verifies that
        they are sorted after loading.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            acqm_base = os.path.join(tmpdir, "unsorted_spikes")

            # Unsorted spike times
            train_dict = {0: np.array([300, 100, 200])}

            np.savez(acqm_base, train=train_dict, fs=1000.0)

            sd = loaders.load_spikedata_from_acqm(acqm_base + ".npz")

            # Should be sorted
            self.assertTrue(np.allclose(sd.train[0], [100.0, 200.0, 300.0]))


class TestPickleLoaders(unittest.TestCase):
    """Quick tests for pickle loader functionality."""

    def _tmp_pkl(self) -> str:
        """Create a temporary pickle file path."""
        fd, path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        return path

    def tearDown(self) -> None:
        """Clean up temporary files."""
        for attr in ("_last_pkl",):
            path = getattr(self, attr, None)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def test_pickle_loader_basic(self):
        """Test basic pickle loader round-trip."""
        # Create test data
        trains = [
            np.array([5.0, 10.0, 15.0]),
            np.array([2.5, 20.0]),
            np.array([], float),
        ]
        sd = SpikeData(trains, length=25.0)

        # Export using exporter
        path = self._tmp_pkl()
        self._last_pkl = path
        exporters.export_spikedata_to_pickle(sd, path)

        # Load using loader function
        sd2 = loaders.load_spikedata_from_pickle(path)

        # Verify
        self.assertEqual(sd.N, sd2.N)
        self.assertEqual(sd.length, sd2.length)
        for a, b in zip(sd.train, sd2.train):
            self.assertTrue(np.array_equal(a, b))

    def test_pickle_loader_with_metadata(self):
        """Test pickle loader preserves metadata."""
        trains = [np.array([5.0, 10.0])]
        metadata = {"experiment": "test", "value": 42}
        sd = SpikeData(trains, length=20.0, metadata=metadata)

        path = self._tmp_pkl()
        self._last_pkl = path
        sd.to_pickle(path)

        sd2 = loaders.load_spikedata_from_pickle(path)

        self.assertEqual(sd2.metadata, metadata)


if __name__ == "__main__":
    unittest.main()
