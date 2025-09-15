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


@unittest.skipIf(h5py is None, "h5py not installed; skipping HDF5/NWB tests")
class TestHDF5Loaders(unittest.TestCase):
    def _tmp_h5(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".h5")
        os.close(fd)
        return path

    def tearDown(self) -> None:
        # Clean up any temp files left around by tests in this class
        for attr in ("_last_h5_path",):
            path: Optional[str] = getattr(self, attr, None)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def test_hdf5_raster(self):
        path = self._tmp_h5()
        self._last_h5_path = path
        raster = np.array([[0, 2, 0, 1], [1, 0, 0, 0]], dtype=int)
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=raster)

        sd = loaders.load_spikedata_from_hdf5(
            path, raster_dataset="raster", raster_bin_size_ms=10.0
        )
        self.assertIsInstance(sd, SpikeData)
        self.assertTrue(np.all(sd.raster(10.0) == raster))
        self.assertEqual(sd.N, raster.shape[0])

    def test_hdf5_raster_not_2d_raises(self):
        path = self._tmp_h5()
        self._last_h5_path = path
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raster", data=np.array([0, 1, 2]))
        with self.assertRaises(ValueError):
            loaders.load_spikedata_from_hdf5(
                path, raster_dataset="raster", raster_bin_size_ms=10.0
            )

    def test_hdf5_multiple_styles_raises(self):
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
        path = self._tmp_h5()
        self._last_h5_path = path
        with h5py.File(path, "w") as _:  # type: ignore
            pass
        with self.assertRaises(ValueError):
            loaders.load_spikedata_from_hdf5(path)  # no style specified

    def test_hdf5_samples_without_fs_error(self):
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

    def test_spikeinterface_subset_and_override_fs(self):
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
        class BadSorting:
            pass

        with self.assertRaises(TypeError):
            loaders.load_spikedata_from_spikeinterface(BadSorting())

    def test_kilosort_empty_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            np.save(os.path.join(d, "spike_times.npy"), np.array([], dtype=int))
            np.save(os.path.join(d, "spike_clusters.npy"), np.array([], dtype=int))
            sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
            self.assertEqual(sd.N, 0)
            self.assertEqual(sd.length, 0.0)

    def test_kilosort_metadata_cluster_ids_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            spike_times = np.array([10, 20, 15, 30])
            spike_clusters = np.array([5, 5, 3, 5])
            np.save(os.path.join(d, "spike_times.npy"), spike_times)
            np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)
            sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
            # cluster_ids sorted ascending (np.unique order)
            self.assertEqual(sd.metadata.get("cluster_ids"), [3, 5])

    def test_kilosort_tsv_missing_columns_keeps_all(self):
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


if __name__ == "__main__":
    unittest.main()
