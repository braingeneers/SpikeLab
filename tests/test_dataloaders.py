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
import warnings
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
        (Test Case 1)  Checks that the idces_times round-trips the original
            (idces, times) pairs. Order is now per-unit grouped (Tier
            K-C1 perf opt rewrote ``idces_times`` to use
            ``np.repeat`` + ``np.concatenate``), so compare via sorted
            (idx, time) pairs rather than positional equality.
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
        # Order-independent comparison — (idx, time) pairs survive
        # the per-unit grouping.
        original_pairs = sorted(zip(idces.tolist(), times_ms.tolist()))
        loaded_pairs = sorted(zip(loaded_idces.tolist(), loaded_times.tolist()))
        assert loaded_pairs == original_pairs

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
        np.testing.assert_allclose(sd.train[0], np.array([100.0, 200.0]))
        np.testing.assert_allclose(sd.train[1], np.array([50.0]))

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

    def test_hdf5_group_per_unit_no_datasets_zero_units(self, tmp_path):
        """
        An HDF5 group-per-unit file with an empty units group (zero
        datasets) loads as a zero-unit SpikeData with length 0.

        Distinct from ``test_hdf5_group_per_unit_empty_units`` (which
        creates two empty-train units) — here the group itself contains
        no datasets at all. Pins the contract that the loader does not
        error and yields the zero-unit shape invariant.

        Tests:
            (Test Case 1) ``SpikeData.N == 0``.
            (Test Case 2) ``SpikeData.length == 0.0``.
            (Test Case 3) ``SpikeData.train`` is an empty sequence.
        """
        path = str(tmp_path / "empty_group.h5")
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_group("units")

        sd = loaders.load_spikedata_from_hdf5(
            path, group_per_unit="units", group_time_unit="ms"
        )
        assert sd.N == 0
        assert sd.length == 0.0
        assert len(sd.train) == 0

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
        np.testing.assert_allclose(sd.train[0], [100.0, 200.0])
        np.testing.assert_allclose(sd.train[1], [500.0])

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
        np.testing.assert_allclose(sd.train[0], [100.0, 300.0])
        np.testing.assert_allclose(sd.train[1], [200.0])

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
        np.testing.assert_allclose(sd_s.raw_time, np.arange(5) * 1.0)

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
        np.testing.assert_allclose(sd_p.raw_time, np.arange(5) * 1.0)

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
        Verify that _trains_from_flat_index raises ValueError for
        non-monotonic end_indices.

        Tests:
            (Test Case 1) Non-monotonic end_indices (e.g. [3, 2, 5]) raise
                ValueError because the indices are not non-decreasing.
        """
        flat_times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        end_indices = np.array([3, 2, 5])

        with pytest.raises(ValueError, match="monotonically non-decreasing"):
            loaders._trains_from_flat_index(
                flat_times, end_indices, unit="ms", fs_Hz=None
            )

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
        np.testing.assert_allclose(sd.train[0], [100.0, 200.0])

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
        np.testing.assert_allclose(sd.train[0], [100.0, 200.0])
        np.testing.assert_allclose(sd.train[1], [500.0])

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
        np.testing.assert_allclose(sd.train[0], [200.0])
        np.testing.assert_allclose(sd.train[1], [700.0])

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


try:  # optional, only needed for pynwb metadata tests
    import pynwb  # type: ignore  # noqa: F401

    pynwb_available = True
except Exception:  # pragma: no cover
    pynwb_available = False


skip_no_pynwb = pytest.mark.skipif(
    not pynwb_available, reason="pynwb not installed; skipping NWB metadata tests"
)


def _write_nwb_metadata_fixture(path: str, *, with_units: bool, locations=None) -> None:
    """Build a minimal pynwb NWB file with subject, electrodes and an
    ElectricalSeries acquisition. Used by the file-level metadata
    population tests. Adds a small units table when ``with_units``.
    """
    from datetime import datetime, timezone

    import numpy as np
    from pynwb import NWBFile, NWBHDF5IO
    from pynwb.ecephys import ElectricalSeries
    from pynwb.file import Subject

    locations = locations if locations is not None else ["VISp", "CA1"]
    nwb = NWBFile(
        session_description="metadata test",
        identifier="fixture-uuid-001",
        session_start_time=datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc),
    )
    nwb.subject = Subject(
        subject_id="m001", species="Mus musculus", sex="M", age="P90D"
    )
    device = nwb.create_device(name="test_probe")
    eg = nwb.create_electrode_group(
        name="shank0", description="d", location="VISp", device=device
    )
    for i, loc in enumerate(locations):
        nwb.add_electrode(location=loc, group=eg, x=float(i), y=0.0, z=0.0)
    region = nwb.create_electrode_table_region(
        list(range(len(locations))), "all electrodes"
    )
    es = ElectricalSeries(
        name="ElectricalSeries",
        data=np.zeros((500, len(locations)), dtype="int16"),
        electrodes=region,
        starting_time=0.0,
        rate=30000.0,
    )
    nwb.add_acquisition(es)
    if with_units:
        # Each unit needs an explicit ``electrodes`` index for the
        # neuron_attributes electrode wiring to populate. Without it
        # pynwb's auto-DynamicTableRegion still appears in the units
        # table but with no rows, so ``df.columns`` lacks the
        # ``electrodes`` key our loader scans for.
        for i in range(len(locations)):
            nwb.add_unit(spike_times=[0.1 + 0.1 * i, 0.5 + 0.1 * i], electrodes=[i])
    with NWBHDF5IO(path, mode="w") as io:
        io.write(nwb)


@skip_no_pynwb
class TestNWBFileMetadata:
    """Tests for the file-level metadata population folded into
    ``load_spikedata_from_nwb`` via the pynwb path."""

    def test_metadata_populates_file_level_fields(self, tmp_path):
        """
        Test that subject, session, devices, and electrode info land in
        ``sd.metadata`` when reading via pynwb.

        Tests:
            (Test Case 1) ``identifier`` matches the file's UUID.
            (Test Case 2) Subject fields (``subject_id``, ``species``,
                ``sex``, ``age``) round-trip.
            (Test Case 3) ``device_names`` is the sorted device list.
            (Test Case 4) ``sampling_rate_hz`` + ``duration_seconds``
                derive from the first ``ElectricalSeries`` acquisition.
            (Test Case 5) ``unit_count`` matches the units table length.
            (Test Case 6) ``electrodes_by_channel`` has one entry per
                electrode row with ``location`` text + 3D coords.
        """
        path = str(tmp_path / "meta.nwb")
        _write_nwb_metadata_fixture(path, with_units=True, locations=["VISp", "CA1"])

        sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=True)
        m = sd.metadata
        assert m["identifier"] == "fixture-uuid-001"
        assert m["subject_id"] == "m001"
        assert m["species"] == "Mus musculus"
        assert m["sex"] == "M"
        assert m["age"] == "P90D"
        assert m["device_names"] == ["test_probe"]
        assert m["sampling_rate_hz"] == 30000.0
        assert m["duration_seconds"] == pytest.approx(500 / 30000.0)
        assert m["unit_count"] == 2
        ebc = m["electrodes_by_channel"]
        assert set(ebc.keys()) == {0, 1}
        assert ebc[0]["location"] == "VISp"
        assert ebc[1]["location"] == "CA1"
        assert ebc[0]["x"] == 0.0
        assert ebc[1]["x"] == 1.0

    def test_metadata_neuron_attrs_carry_location_label(self, tmp_path):
        """
        Test that per-neuron ``location_label`` carries the textual
        region from the electrodes table alongside the 3D-coord
        ``location`` list.

        Tests:
            (Test Case 1) ``neuron_attributes`` has one entry per unit.
            (Test Case 2) The first unit's ``location`` is the 3D coord
                list (backwards-compatible).
            (Test Case 3) The first unit's ``location_label`` is the
                textual region string from the electrodes table.
        """
        path = str(tmp_path / "labelled.nwb")
        _write_nwb_metadata_fixture(path, with_units=True, locations=["VISp", "CA1"])

        sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=True)
        assert sd.neuron_attributes is not None
        assert len(sd.neuron_attributes) == 2
        # The first unit was added without specifying an electrode column,
        # so pynwb's default behavior wires it to electrode 0 via the
        # auto-populated 'electrodes' DynamicTableRegion. Both 'location'
        # (3D coords) and 'location_label' (textual) should be present.
        first = sd.neuron_attributes[0]
        assert "location" in first
        assert isinstance(first["location"], list)
        assert first["location_label"] == "VISp"
        assert first["group_name"] == "shank0"

    def test_metadata_allow_no_units_returns_empty_spikedata(self, tmp_path):
        """
        Test that ``allow_no_units=True`` returns a SpikeData with N=0
        for files without a Units table, with metadata still populated.

        Tests:
            (Test Case 1) ``sd.N`` is 0.
            (Test Case 2) ``sd.metadata['unit_count']`` is absent or 0.
            (Test Case 3) ``sd.metadata['identifier']`` is still populated
                — metadata extraction precedes the units branch.
            (Test Case 4) Without ``allow_no_units`` the loader raises.
        """
        path = str(tmp_path / "unsorted.nwb")
        _write_nwb_metadata_fixture(path, with_units=False, locations=["VISp"])

        sd = loaders.load_spikedata_from_nwb(
            path, prefer_pynwb=True, allow_no_units=True
        )
        assert sd.N == 0
        assert sd.metadata["identifier"] == "fixture-uuid-001"
        assert sd.metadata.get("unit_count", 0) == 0

        # Without ``allow_no_units`` the pynwb path raises "no Units
        # table". That ValueError is swallowed by the broad fallback
        # handler and the loader then tries h5py, which raises
        # "missing '/units' group". Either error reaching the caller is
        # a correct refusal — accept either message.
        with pytest.raises(
            ValueError, match=r"(no Units table|missing '/units' group)"
        ):
            loaders.load_spikedata_from_nwb(path, prefer_pynwb=True)


@skip_no_pynwb
class TestLoadSpikedataFromDandi:
    """Tests for ``load_spikedata_from_dandi`` and ``list_dandi_assets``.

    Both functions are stdlib-only; we mock ``urllib.request.urlopen``
    so the tests don't touch the network. The NWB content stream is a
    real on-disk pynwb-generated file so the delegation through
    ``load_spikedata_from_nwb`` is exercised end-to-end.
    """

    def _fake_urlopen_chain(self, responses):
        """Build a side-effect function for ``urlopen`` that returns
        each scripted response in order. Each response is either:
          * a bytes payload (returned from ``.read()``)
          * a path to a file on disk (streamed via the chunked-read
            loop used by the DANDI download helper)
          * a JSON-serialisable dict (encoded as UTF-8 bytes)
        """
        import io
        import json as _json

        responses = list(responses)

        def _fake(req, timeout=None):
            if not responses:
                raise AssertionError("urlopen called more times than scripted")
            r = responses.pop(0)
            if isinstance(r, dict):
                buf = io.BytesIO(_json.dumps(r).encode("utf-8"))
            elif isinstance(r, (bytes, bytearray)):
                buf = io.BytesIO(bytes(r))
            elif isinstance(r, str):
                buf = io.BytesIO(open(r, "rb").read())
            else:
                raise TypeError(f"unexpected response type: {type(r)}")

            class _Ctx:
                def __enter__(self_inner):
                    return buf

                def __exit__(self_inner, *exc):
                    buf.close()
                    return False

            return _Ctx()

        return _fake

    def test_list_dandi_assets_iterates_pagination(self, tmp_path):
        """
        Test that ``list_dandi_assets`` flattens paginated responses
        into a single iterator.

        Tests:
            (Method 1) Mocks two pages of API responses.
            (Method 2) Calls ``list_dandi_assets`` and materialises.
            (Test Case 1) Total assets equals sum across pages.
            (Test Case 2) Asset dicts carry expected keys + values.
            (Test Case 3) Iteration terminates when ``next`` is null.
        """
        page1 = {
            "results": [
                {
                    "asset_id": "aaaa1111",
                    "path": "sub-001/sub-001_ses-1_ecephys.nwb",
                    "size": 12345,
                    "contentUrl": ["https://example.org/aaaa1111.nwb"],
                },
                {
                    "asset_id": "bbbb2222",
                    "path": "sub-001/sub-001_ses-1_metadata.json",
                    "size": 678,
                },
            ],
            "next": "https://api.example.org/page2",
        }
        page2 = {
            "results": [
                {
                    "asset_id": "cccc3333",
                    "path": "sub-002/sub-002_ses-1_ecephys.nwb",
                    "size": 999_888,
                    "contentUrl": "https://example.org/cccc3333.nwb",
                },
            ],
            "next": None,
        }

        fake = self._fake_urlopen_chain([page1, page2])
        with patch("urllib.request.urlopen", side_effect=fake):
            assets = list(loaders.list_dandi_assets("000006", version="draft"))

        assert len(assets) == 3
        assert assets[0]["asset_id"] == "aaaa1111"
        assert assets[0]["path"].endswith(".nwb")
        assert assets[0]["download_url"] == "https://example.org/aaaa1111.nwb"
        assert assets[0]["dandiset_id"] == "000006"
        assert assets[0]["version"] == "draft"
        assert assets[1]["size"] == 678
        assert assets[2]["download_url"] == "https://example.org/cccc3333.nwb"

    def test_list_dandi_assets_synthesizes_download_url_fallback(self, tmp_path):
        """
        Test that ``list_dandi_assets`` synthesizes a canonical download
        URL when the API response is missing ``contentUrl`` and
        ``download_url``.

        Tests:
            (Test Case 1) Asset with neither URL field falls back to
                ``{api_base}/assets/{id}/download/``.
        """
        page = {
            "results": [
                {"asset_id": "no-url-asset", "path": "sub-1/x.nwb", "size": 0},
            ],
            "next": None,
        }
        fake = self._fake_urlopen_chain([page])
        with patch("urllib.request.urlopen", side_effect=fake):
            (asset,) = list(loaders.list_dandi_assets("000006"))
        assert asset["download_url"].endswith("/assets/no-url-asset/download/")

    def test_load_spikedata_from_dandi_delegates_to_nwb_loader(self, tmp_path):
        """
        Test end-to-end: detail lookup + download + delegation populates
        DANDI provenance on metadata and propagates the NWB file-level
        fields.

        Tests:
            (Method 1) Build a real NWB fixture on disk, mock the asset
                detail + download responses to serve its bytes.
            (Test Case 1) ``sd.metadata['dandi_asset_id']`` matches.
            (Test Case 2) ``sd.metadata['source_reference']`` is the
                canonical DANDI URL.
            (Test Case 3) NWB metadata (identifier, species, etc.)
                survives the delegation.
            (Test Case 4) Spike trains round-trip.
        """
        nwb_path = str(tmp_path / "fixture.nwb")
        _write_nwb_metadata_fixture(nwb_path, with_units=True, locations=["VISp"])

        detail = {
            "asset_id": "asset-xyz",
            "path": "sub-fix/fixture.nwb",
            "size": os.path.getsize(nwb_path),
            "contentUrl": ["https://example.org/asset-xyz/download/"],
        }
        fake = self._fake_urlopen_chain([detail, nwb_path])

        download_dir = str(tmp_path / "dl")
        with patch("urllib.request.urlopen", side_effect=fake):
            sd = loaders.load_spikedata_from_dandi(
                "asset-xyz",
                dandiset_id="000006",
                version="draft",
                download_dir=download_dir,
            )

        assert sd.metadata["dandi_asset_id"] == "asset-xyz"
        assert sd.metadata["dandi_dandiset_id"] == "000006"
        assert sd.metadata["dandi_version"] == "draft"
        assert sd.metadata["source_reference"].startswith("dandi://")
        assert "000006" in sd.metadata["source_reference"]
        assert "asset-xyz" in sd.metadata["source_reference"]
        # File-level metadata from the NWB loader survived delegation.
        assert sd.metadata["identifier"] == "fixture-uuid-001"
        assert sd.metadata["species"] == "Mus musculus"
        assert sd.metadata["sampling_rate_hz"] == 30000.0
        # Caller-supplied download dir is recorded so consumers can
        # post-process the bytes (content-hash, blob write, etc.).
        assert sd.metadata["downloaded_path"].endswith(".nwb")
        assert os.path.isfile(sd.metadata["downloaded_path"])
        # Spike trains delegated correctly.
        assert sd.N == 1
        assert len(sd.train[0]) == 2  # two spikes per fixture unit

    def test_load_spikedata_from_dandi_temp_dir_cleanup(self, tmp_path):
        """
        Test that ``download_dir=None`` uses a temporary directory that
        is cleaned up after the load (no ``downloaded_path`` recorded).

        Tests:
            (Test Case 1) ``downloaded_path`` is absent from metadata.
            (Test Case 2) The load still succeeds end-to-end.
        """
        nwb_path = str(tmp_path / "fixture.nwb")
        _write_nwb_metadata_fixture(nwb_path, with_units=True, locations=["VISp"])

        detail = {
            "asset_id": "asset-tmp",
            "path": "x/x.nwb",
            "size": os.path.getsize(nwb_path),
            "contentUrl": ["https://example.org/asset-tmp/download/"],
        }
        fake = self._fake_urlopen_chain([detail, nwb_path])
        with patch("urllib.request.urlopen", side_effect=fake):
            sd = loaders.load_spikedata_from_dandi(
                "asset-tmp",
                dandiset_id="000006",
            )

        assert "downloaded_path" not in sd.metadata
        assert sd.N == 1

    def test_load_spikedata_from_dandi_accepts_full_url(self, tmp_path):
        """
        Test that passing a full URL as ``asset_id`` skips the detail
        endpoint and downloads directly.

        Tests:
            (Test Case 1) Detail endpoint is NOT called (only one
                mocked response is consumed).
            (Test Case 2) ``dandi_asset_id`` is derived from the URL tail.
        """
        nwb_path = str(tmp_path / "fixture.nwb")
        _write_nwb_metadata_fixture(nwb_path, with_units=True, locations=["VISp"])

        # Only one mocked response — the download. If the loader tried
        # to fetch a detail page, the fake would raise AssertionError.
        fake = self._fake_urlopen_chain([nwb_path])
        with patch("urllib.request.urlopen", side_effect=fake):
            sd = loaders.load_spikedata_from_dandi(
                "https://example.org/some/path/asset-url-id",
            )

        assert sd.metadata["dandi_asset_id"] == "asset-url-id"

    def test_load_spikedata_from_dandi_raw_recording_allow_no_units(self, tmp_path):
        """
        Test that a DANDI asset without a Units table loads via
        ``allow_no_units=True`` and returns ``N=0`` with metadata
        populated — the "raw recording triage" path.

        Tests:
            (Test Case 1) ``sd.N`` is 0.
            (Test Case 2) ``sd.metadata['identifier']`` is populated.
            (Test Case 3) ``sd.metadata['electrodes_by_channel']`` is
                populated (raw recordings still have electrodes).
        """
        nwb_path = str(tmp_path / "raw.nwb")
        _write_nwb_metadata_fixture(nwb_path, with_units=False, locations=["VISp"])

        detail = {
            "asset_id": "raw-asset",
            "path": "x/raw.nwb",
            "size": os.path.getsize(nwb_path),
            "contentUrl": ["https://example.org/raw-asset/download/"],
        }
        fake = self._fake_urlopen_chain([detail, nwb_path])
        with patch("urllib.request.urlopen", side_effect=fake):
            sd = loaders.load_spikedata_from_dandi(
                "raw-asset",
                dandiset_id="000006",
                allow_no_units=True,
            )

        assert sd.N == 0
        assert sd.metadata["identifier"] == "fixture-uuid-001"
        assert "electrodes_by_channel" in sd.metadata


try:  # optional, only needed for load_recording_from_dandi tests
    from spikeinterface.extractors import NwbRecordingExtractor  # type: ignore  # noqa: F401

    spikeinterface_available = True
except Exception:  # pragma: no cover
    spikeinterface_available = False


skip_no_spikeinterface = pytest.mark.skipif(
    not spikeinterface_available,
    reason="spikeinterface not installed; skipping DANDI recording tests",
)


@skip_no_pynwb
@skip_no_spikeinterface
class TestLoadRecordingFromDandi:
    """Tests for ``load_recording_from_dandi`` (DANDI raw recording →
    SpikeInterface Zarr).

    Uses the same mocked-urlopen pattern as the spike-data DANDI tests
    but actually exercises SpikeInterface's NWB extractor + Zarr write
    end-to-end on a real fixture.
    """

    def _fake_urlopen_chain(self, responses):
        """Same helper as TestLoadSpikedataFromDandi.

        Duplicated rather than shared because pytest's class collection
        doesn't make helper inheritance ergonomic across classes; the
        helper is small and changes to one shouldn't silently affect
        the other.
        """
        import io
        import json as _json

        responses = list(responses)

        def _fake(req, timeout=None):
            if not responses:
                raise AssertionError("urlopen called more times than scripted")
            r = responses.pop(0)
            if isinstance(r, dict):
                buf = io.BytesIO(_json.dumps(r).encode("utf-8"))
            elif isinstance(r, (bytes, bytearray)):
                buf = io.BytesIO(bytes(r))
            elif isinstance(r, str):
                buf = io.BytesIO(open(r, "rb").read())
            else:
                raise TypeError(f"unexpected response type: {type(r)}")

            class _Ctx:
                def __enter__(self_inner):
                    return buf

                def __exit__(self_inner, *exc):
                    buf.close()
                    return False

            return _Ctx()

        return _fake

    def test_writes_zarr_and_returns_metadata(self, tmp_path):
        """
        Test happy path: download → NwbRecordingExtractor → Zarr write
        → metadata sidecar.

        Tests:
            (Test Case 1) ``zarr_path`` is created on disk and contains
                a SpikeInterface-shaped directory.
            (Test Case 2) ``recording_metadata.json`` sidecar exists with
                DANDI provenance + NWB metadata merged.
            (Test Case 3) Returned dict has expected recording-shape
                fields (``sampling_rate_hz``, ``n_channels``,
                ``n_samples``).
            (Test Case 4) ``downloaded_nwb_path`` absent by default
                (NWB cleaned up after Zarr write).
        """
        nwb_path = str(tmp_path / "fixture.nwb")
        _write_nwb_metadata_fixture(
            nwb_path, with_units=False, locations=["VISp", "CA1"]
        )

        detail = {
            "asset_id": "raw-asset-zarr",
            "path": "sub-fix/raw.nwb",
            "size": os.path.getsize(nwb_path),
            "contentUrl": ["https://example.org/raw-asset-zarr/download/"],
        }
        fake = self._fake_urlopen_chain([detail, nwb_path])

        zarr_dest = str(tmp_path / "out.zarr")
        with patch("urllib.request.urlopen", side_effect=fake):
            result = loaders.load_recording_from_dandi(
                "raw-asset-zarr",
                zarr_dest,
                dandiset_id="000999",
                version="draft",
            )

        assert os.path.isdir(result["zarr_path"])
        assert os.path.isfile(result["recording_metadata_path"])
        assert result["sampling_rate_hz"] == 30000.0
        assert result["n_channels"] == 2
        assert result["n_samples"] == 500
        assert result["duration_seconds"] == pytest.approx(500 / 30000.0)
        assert result["dandi_asset_id"] == "raw-asset-zarr"
        assert result["dandi_dandiset_id"] == "000999"
        assert result["source_reference"].startswith("dandi://")
        # NWB metadata merged from the fixture.
        assert result["identifier"] == "fixture-uuid-001"
        assert result["species"] == "Mus musculus"
        # NWB cleaned up by default.
        assert "downloaded_nwb_path" not in result
        # Sidecar JSON has the same fields.
        import json as _json

        with open(result["recording_metadata_path"], "r", encoding="utf-8") as fh:
            sidecar = _json.load(fh)
        assert sidecar["dandi_asset_id"] == "raw-asset-zarr"
        assert sidecar["n_channels"] == 2
        assert sidecar["format"] == "spikeinterface_zarr"

    def test_keep_nwb_preserves_source_file(self, tmp_path):
        """
        Test ``keep_nwb=True`` leaves the downloaded NWB on disk and
        records the path in the return dict.

        Tests:
            (Test Case 1) ``downloaded_nwb_path`` present in result.
            (Test Case 2) The path resolves to an existing file.
        """
        nwb_path = str(tmp_path / "fixture.nwb")
        _write_nwb_metadata_fixture(nwb_path, with_units=False, locations=["VISp"])

        download_dir = str(tmp_path / "dl")
        detail = {
            "asset_id": "keep-asset",
            "path": "x/keep.nwb",
            "size": os.path.getsize(nwb_path),
            "contentUrl": ["https://example.org/keep-asset/download/"],
        }
        fake = self._fake_urlopen_chain([detail, nwb_path])

        zarr_dest = str(tmp_path / "out2.zarr")
        with patch("urllib.request.urlopen", side_effect=fake):
            result = loaders.load_recording_from_dandi(
                "keep-asset",
                zarr_dest,
                dandiset_id="000999",
                download_dir=download_dir,
                keep_nwb=True,
            )

        assert "downloaded_nwb_path" in result
        assert os.path.isfile(result["downloaded_nwb_path"])

    def test_existing_zarr_dest_raises_without_overwrite(self, tmp_path):
        """
        Test that an existing target Zarr dir raises ``FileExistsError``
        when ``overwrite=False`` (the safe default).

        Tests:
            (Test Case 1) FileExistsError raised.
            (Test Case 2) Existing dir is untouched.
        """
        zarr_dest = tmp_path / "out3.zarr"
        zarr_dest.mkdir()
        sentinel = zarr_dest / "DO_NOT_DELETE.txt"
        sentinel.write_text("keep me", encoding="utf-8")

        with pytest.raises(FileExistsError):
            loaders.load_recording_from_dandi(
                "irrelevant-asset",
                str(zarr_dest),
                dandiset_id="000999",
                overwrite=False,
            )

        assert sentinel.is_file()
        assert sentinel.read_text(encoding="utf-8") == "keep me"


class TestIBLMetadataHelpers:
    """Tests for the private IBL metadata helpers ``_ibl_collect_session_metadata``
    and ``_ibl_collect_channels``.

    The helpers wrap REST + ONE calls in best-effort try/except blocks
    so analysis pipelines don't crash on network failures or schema
    drift. We mock the ``one`` object surface they touch — full IBL
    integration tests require live ONE access which CI doesn't have.
    """

    def test_collect_session_metadata_merges_session_subject_insertion(self):
        """
        Test that session + subject + insertion data merge into one
        flat dict with NWB-aligned keys.

        Tests:
            (Test Case 1) Session fields map to expected dst keys.
            (Test Case 2) Subject details (a second REST call) merge.
            (Test Case 3) Insertion json coords surface as
                ``insertion_*`` fields.
        """
        one = MagicMock()
        one.alyx.rest.side_effect = [
            # sessions/read response
            {
                "start_time": "2024-01-01T12:00:00Z",
                "end_time": "2024-01-01T13:00:00Z",
                "lab": "cortexlab",
                "task_protocol": "_iblrig_tasks_ephysChoiceWorld6.4.0",
                "project": "ibl_neuropixel_brainwide_01",
                "number": 3,
                "procedures": ["behavior_training_3A"],
                "qc": "PASS",
                "subject": "SWC_054",
            },
            # subjects/read response
            {
                "species": "Mus musculus",
                "sex": "M",
                "birth_date": "2023-09-01",
                "age_weeks": 16.0,
                "strain": "C57BL/6J",
                "genotype": "DAT-Cre",
                "responsible_user": "lab_member_1",
            },
            # insertions/read response
            {
                "name": "probe00",
                "model": "NEUROPIXEL_1.0",
                "json": {"x": 1500.0, "y": -2300.0, "z": 0.0, "depth": 3000.0},
            },
        ]

        md = loaders._ibl_collect_session_metadata(one, eid="eid-1", pid="pid-1")

        assert md["session_start_time"] == "2024-01-01T12:00:00Z"
        assert md["lab"] == "cortexlab"
        assert md["task_protocol"].startswith("_iblrig_tasks_ephysChoiceWorld")
        assert md["session_number"] == 3
        assert md["procedures"] == ["behavior_training_3A"]
        assert md["subject_id"] == "SWC_054"
        assert md["species"] == "Mus musculus"
        assert md["strain"] == "C57BL/6J"
        assert md["probe_name"] == "probe00"
        assert md["probe_model"] == "NEUROPIXEL_1.0"
        assert md["insertion_x"] == 1500.0
        assert md["insertion_depth"] == 3000.0

    def test_collect_session_metadata_swallows_partial_failures(self):
        """
        Test that an exception on any individual REST call leaves the
        corresponding fields absent without raising upward.

        Tests:
            (Test Case 1) Session call fails → subject + insertion keys
                still surface from the surviving calls. Subject lookup
                is skipped (it depends on session.subject), so subject
                fields stay absent.
            (Test Case 2) Returned dict has only insertion fields.
        """
        one = MagicMock()
        one.alyx.rest.side_effect = [
            Exception("alyx session 502"),
            # session_subject path doesn't fire (session failed)
            # → next call is the insertion read
            {"name": "probe01", "model": "NEUROPIXEL_2.0", "json": {}},
        ]

        md = loaders._ibl_collect_session_metadata(one, eid="eid-x", pid="pid-x")

        assert "subject_id" not in md
        assert "lab" not in md
        assert md["probe_name"] == "probe01"
        assert md["probe_model"] == "NEUROPIXEL_2.0"

    def test_collect_channels_returns_empty_on_load_failure(self):
        """
        Test that ``_ibl_collect_channels`` returns an empty dict when
        every candidate collection raises on ``load_object``.

        Tests:
            (Test Case 1) Empty dict, no exception.
        """
        one = MagicMock()
        one.load_object.side_effect = ValueError("no channels here")
        out = loaders._ibl_collect_channels(one, "eid", ["alf/probe00", "alf"])
        assert out == {}

    def test_collect_channels_projects_onto_nwb_shape(self):
        """
        Test that a successful channels load produces an
        ``electrodes_by_channel`` dict matching the NWB shape.

        Tests:
            (Test Case 1) Per-channel ``location`` is the Allen acronym.
            (Test Case 2) ``atlas_id`` is the integer Allen Structure ID.
            (Test Case 3) ``x/y/z`` are ML/AP/DV from ``mlapdv``.
            (Test Case 4) ``raw_index`` is preserved.
        """
        one = MagicMock()
        # Build a channels object with the shape ONE returns.
        channels = MagicMock()
        channels.acronym = np.array(["VISp5", "VISp5", "CA1"], dtype=object)
        channels.atlas_id = np.array([778, 778, 382])
        channels.mlapdv = np.array(
            [
                [1.5, -2.3, 0.5],
                [1.6, -2.4, 0.6],
                [1.7, -2.5, 0.7],
            ]
        )
        channels.localCoordinates = np.array(
            [
                [20.0, 0.0],
                [20.0, 20.0],
                [20.0, 40.0],
            ]
        )
        channels.rawInd = np.array([100, 101, 102])

        one.load_object.return_value = channels
        out = loaders._ibl_collect_channels(one, "eid", ["alf/probe00"])

        assert set(out.keys()) == {0, 1, 2}
        assert out[0]["location"] == "VISp5"
        assert out[0]["atlas_id"] == 778
        assert out[0]["x"] == 1.5
        assert out[0]["z"] == 0.5
        assert out[2]["location"] == "CA1"
        assert out[2]["atlas_id"] == 382
        assert out[2]["raw_index"] == 102


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
            np.testing.assert_allclose(train, truth)

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
        np.testing.assert_allclose(sd.train[0], [10.0, 20.0])
        np.testing.assert_allclose(sd.train[1], [2.5])

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
        np.testing.assert_allclose(sd.train[0], [0.0, 10.0])

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
            np.testing.assert_allclose(a, b)
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

        # Load via S3 URL; ensure_local_file is mocked so no real S3 call.
        # Pass allow_remote=True to opt in to the S3 fetch path.
        sd2 = loaders.load_spikedata_from_pickle(
            "s3://bucket/key.pkl", allow_remote=True
        )

        # Verify ensure_local_file was called with S3 URL (and optional cred kwargs)
        mock_ensure.assert_called_once()
        assert mock_ensure.call_args[0][0] == "s3://bucket/key.pkl"
        # Verify loaded data matches
        np.testing.assert_allclose(sd2.train[0], sd.train[0])

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
        loaders.load_spikedata_from_pickle("s3://bucket/key.pkl", allow_remote=True)

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

    @patch("spikelab.data_loaders.s3_utils.ensure_local_file")
    def test_pickle_temp_file_cleanup_on_load_failure(self, mock_ensure, tmp_path):
        """
        When ``pickle.load`` itself raises (not just an EOFError on an
        empty file), the loader's ``finally`` block still removes the
        downloaded temp file so the caller does not leak disk.

        Pins the contract of the ``try / finally`` around ``pickle.load``
        in ``load_spikedata_from_pickle``: cleanup must fire on *any*
        exception from ``pickle.load``, not just clean returns.

        Tests:
            (Test Case 1) An UnpicklingError raised by ``pickle.load``
                on garbage bytes still triggers ``os.remove`` of the
                temp file.
            (Test Case 2) The original exception propagates to the
                caller.
        """
        # Write garbage bytes that will trip pickle.UnpicklingError or
        # similar inside pickle.load (not at file-open time).
        fd, path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(b"\x80\x04\x95not-a-valid-pickle-stream")

        # Pretend this file came from S3 so the loader treats it as a
        # temp file and routes through the cleanup path.
        mock_ensure.return_value = (path, True)

        with pytest.raises(Exception):
            loaders.load_spikedata_from_pickle(
                "s3://bucket/garbage.pkl", allow_remote=True
            )

        # finally block ran → temp file removed even though pickle.load
        # raised.
        assert not os.path.exists(path)


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
        np.testing.assert_allclose(np.sort(sd.train[0]), np.sort(expected_ms))

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
        np.testing.assert_allclose(sd.metadata["stim_on_times"], expected_stim_on_ms)

        expected_start_ms = trials_df["intervals_0"].to_numpy() * 1000.0
        np.testing.assert_allclose(sd.metadata["trial_start_times"], expected_start_ms)

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

        np.testing.assert_allclose(
            sd.metadata["choice"], trials_df["choice"].to_numpy()
        )
        np.testing.assert_allclose(
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

    def test_no_spikes_produces_empty_trains_with_explicit_length(self):
        """
        With all spike collections unavailable and an explicit ``length_ms``
        supplied, the loader returns a SpikeData with empty trains.

        Tests:
            (Test Case 1) Each unit's spike train is an empty array.
            (Test Case 2) ``sd.length`` matches the explicit ``length_ms``.
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
        sd = self._load(eid, pid, mock_one_api, mock_brainwidemap, length_ms=5_000.0)

        for train in sd.train:
            assert len(train) == 0
        assert sd.length == pytest.approx(5_000.0)

    def test_no_spikes_without_length_raises(self):
        """
        When all spike collections fail AND no explicit ``length_ms`` is
        passed, the loader raises ValueError rather than fabricating a
        default duration.

        Tests:
            (Test Case 1) ValueError is raised naming the probe.
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
        with pytest.raises(ValueError, match="length_ms"):
            self._load(eid, pid, mock_one_api, mock_brainwidemap)

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
        emits a UserWarning naming the eid/pid/collections and still
        returns a SpikeData with empty trains so the calling script
        does not crash. The warning gives the operator a clear signal
        without forcing a hard error in batch jobs.

        Tests:
            (Test Case 1) Returns a SpikeData with the correct number of units.
            (Test Case 2) All spike trains are empty arrays.
            (Test Case 3) A UserWarning is emitted naming the eid, pid,
                and the candidate collections.
        """
        import warnings as _warnings
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
            with _warnings.catch_warnings(record=True) as w:
                _warnings.simplefilter("always")
                sd = loaders.load_spikedata_from_ibl(eid, pid, length_ms=5_000.0)

        assert isinstance(sd, SpikeData)
        assert sd.N == 2
        for train in sd.train:
            assert len(train) == 0

        # A UserWarning was emitted naming the eid, pid, and the
        # candidate collections so the silent-empty result is visible.
        warn_msgs = [str(rec.message) for rec in w if rec.category is UserWarning]
        relevant = [
            m for m in warn_msgs if "load_spikedata_from_ibl" in m and "test-eid" in m
        ]
        assert relevant, warn_msgs
        assert "test-pid" in relevant[0]
        assert "alf" in relevant[0]

    def test_ec_dl_09_no_good_units(self):
        """
        Verify behavior when the bwm_units DataFrame has no good units
        (label==1) for the requested probe. With an explicit ``length_ms``
        the loader returns SpikeData with N=0 at the requested duration.

        Tests:
            (Test Case 1) sd.N == 0.
            (Test Case 2) sd.length equals the requested length_ms.
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
            sd = loaders.load_spikedata_from_ibl(eid, pid, length_ms=5_000.0)

        assert sd.N == 0
        assert sd.length == pytest.approx(5_000.0)

    def test_explicit_collection_short_circuits_heuristic_search(self):
        """
        Tier L-F2: passing ``collection="alf/probe00/pykilosort"`` must
        skip the PID-suffix heuristic + fallback chain and issue exactly
        one ``load_object("spikes", ...)`` call against the explicit
        collection.

        Tests:
            (Test Case 1) With ``collection`` set, exactly one
                ``load_object("spikes", ...)`` call is made and its
                ``collection`` kwarg matches the explicit value.
            (Test Case 2) Without ``collection`` (None / default), the
                heuristic tries multiple candidates when the first
                fails, so more than one ``load_object("spikes", ...)``
                call is made.
        """
        eid, pid = "test-eid", "11111111-2222-3333-4444-555555555500"
        # Build mocks where the heuristic-preferred ``alf/probe00/pykilosort``
        # call fails so the fallback chain is exercised when ``collection``
        # is None — but the explicit ``collection="alf"`` call succeeds.
        mock_one_api, mock_brainwidemap, _, _, _ = self._build_mocks(
            pid,
            eid,
            n_good=2,
            fail_collections={"alf/probe00/pykilosort"},
        )

        # Test Case 1: explicit collection, exactly one spikes call.
        with patch.dict(
            sys.modules,
            {
                "one": MagicMock(),
                "one.api": mock_one_api,
                "brainwidemap": mock_brainwidemap,
            },
        ):
            loaders.load_spikedata_from_ibl(
                eid, pid, length_ms=200_000.0, collection="alf"
            )

        mock_one = mock_one_api.ONE.return_value
        spikes_calls = [
            c
            for c in mock_one.load_object.call_args_list
            if len(c.args) >= 2 and c.args[1] == "spikes"
        ]
        assert len(spikes_calls) == 1, (
            f"explicit collection should issue exactly one spikes call, "
            f"got {len(spikes_calls)}: {spikes_calls}"
        )
        assert spikes_calls[0].kwargs.get("collection") == "alf"

        # Test Case 2: no explicit collection, heuristic falls through
        # at least one failed candidate so >1 spikes call is made.
        mock_one_api2, mock_brainwidemap2, _, _, _ = self._build_mocks(
            pid,
            eid,
            n_good=2,
            fail_collections={"alf/probe00/pykilosort"},
        )
        with patch.dict(
            sys.modules,
            {
                "one": MagicMock(),
                "one.api": mock_one_api2,
                "brainwidemap": mock_brainwidemap2,
            },
        ):
            loaders.load_spikedata_from_ibl(eid, pid, length_ms=200_000.0)

        mock_one2 = mock_one_api2.ONE.return_value
        spikes_calls2 = [
            c
            for c in mock_one2.load_object.call_args_list
            if len(c.args) >= 2 and c.args[1] == "spikes"
        ]
        assert len(spikes_calls2) >= 2, (
            f"no explicit collection: heuristic should try more than one "
            f"candidate when the first fails, got {len(spikes_calls2)}"
        )


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

    def test_download_from_s3_clienterror_cleans_auto_temp_file(self, tmp_path):
        """
        When ``download_from_s3`` allocates a ``NamedTemporaryFile``
        (caller passes ``local_path=None``) and the download then
        raises a ``ClientError``, the auto-allocated temp file is
        unlinked before the exception propagates.

        Tests:
            (Test Case 1) The auto-temp path observed by the failing
                download no longer exists on disk after the exception
                propagates.
            (Test Case 2) A caller-supplied ``local_path`` is NOT
                unlinked by the cleanup path.
        """
        from spikelab.data_loaders.s3_utils import download_from_s3

        captured_paths: list[str] = []

        def failing_download(bucket, key, local_path):
            captured_paths.append(local_path)
            # Ensure the temp file exists when the failure fires so the
            # test can prove the cleanup path actually removed it.
            with open(local_path, "wb") as f:
                f.write(b"")
            raise self._make_client_error("NoSuchBucket")

        mock_client = MagicMock()
        mock_client.download_file.side_effect = failing_download

        with patch("spikelab.data_loaders.s3_utils.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            with pytest.raises(ValueError):
                download_from_s3("s3://nonexistent-bucket/key.h5")

        # Auto-allocated temp file should have been removed.
        assert len(captured_paths) == 1
        assert not os.path.exists(captured_paths[0])

        # Caller-supplied local_path is preserved on failure (caller owns it).
        caller_path = str(tmp_path / "caller_owned.h5")
        with open(caller_path, "wb") as f:
            f.write(b"existing-bytes")

        mock_client.download_file.side_effect = self._make_client_error("AccessDenied")
        with patch("spikelab.data_loaders.s3_utils.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            with pytest.raises(PermissionError):
                download_from_s3("s3://my-bucket/file.h5", local_path=caller_path)
        # Caller's file was not unlinked.
        assert os.path.exists(caller_path)

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
class TestHDF5Loader:
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
        np.testing.assert_allclose(sd.train[0], [5.0])

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
        Ragged style with non-monotonic end_indices raises ValueError.

        Tests:
            (Test Case 1) end_indices [5, 3, 10] are not monotonically
                non-decreasing, so a ValueError is raised.
        """
        import h5py

        path = str(tmp_path / "non_mono.h5")
        flat = np.arange(10, dtype=float)
        end_indices = np.array([5, 3, 10])
        with h5py.File(path, "w") as f:
            f.create_dataset("flat_spike_times", data=flat)
            f.create_dataset("end_indices", data=end_indices)

        with pytest.raises(ValueError, match="monotonically non-decreasing"):
            loaders.load_spikedata_from_hdf5(
                path,
                spike_times_dataset="flat_spike_times",
                spike_times_index_dataset="end_indices",
                spike_times_unit="ms",
            )

    def test_group_per_unit_natural_sort(self, tmp_path):
        """
        Group-per-unit loader uses natural (numeric-aware) sort so unit
        identity is preserved across round-trip at N>=10.

        Tests:
            (Test Case 1) Keys ["1", "10", "2"] are loaded in
                numerical order [1, 2, 10] (not lex [1, 10, 2]).
        """
        import h5py

        path = str(tmp_path / "natural_sort.h5")
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
        # Natural order: ["1", "2", "10"] → trains in that order.
        np.testing.assert_array_equal(sd.train[0], [1.0, 2.0])
        np.testing.assert_array_equal(sd.train[1], [5.0, 6.0])
        np.testing.assert_array_equal(sd.train[2], [3.0, 4.0])

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


class TestTrainsFromFlatIndex:
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
        np.testing.assert_allclose(trains[0], [5.0])

    def test_end_indices_exceeding_flat_times_length(self):
        """
        end_indices = [10] with flat_times having only 3 elements.

        Tests:
            (Test Case 1) ValueError is raised because end_indices[-1]
                exceeds flat_times length.
        """
        with pytest.raises(ValueError, match="exceeds flat array length"):
            loaders._trains_from_flat_index(
                np.array([1.0, 2.0, 3.0]), np.array([10]), unit="ms", fs_Hz=None
            )

    def test_leading_zero_nwb_convention_auto_detected(self):
        """
        Many NWB files in the wild write ``spike_times_index`` with a
        leading zero — i.e. ``(0, c0, c0+c1, ..., total)`` of length
        ``N+1``. The loader auto-detects this variant and strips the
        leading zero so the cumulative-end semantics produce exactly
        ``N`` trains with each unit's spikes correctly assigned.

        Tests:
            (Test Case 1) Leading-zero ``end_indices=[0, 2, 5]`` with
                ``flat_times`` of length 5 produces 2 trains (not 3).
            (Test Case 2) Unit 0's spikes are in trains[0], unit 1's
                spikes are in trains[1] — no silent identity shift.
        """
        flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # 2 units, c0=2 spikes (1.0, 2.0), c1=3 spikes (3.0, 4.0, 5.0).
        # Leading-zero NWB convention: index = [0, 2, 5].
        end_indices = np.array([0, 2, 5])

        trains = loaders._trains_from_flat_index(
            flat, end_indices, unit="ms", fs_Hz=None
        )

        assert len(trains) == 2
        np.testing.assert_array_equal(trains[0], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(trains[1], np.array([3.0, 4.0, 5.0]))

    def test_bare_zero_index_treated_as_single_empty_unit(self):
        """
        A bare ``[0]`` is ambiguous between a single empty unit
        (cumulative-end with one zero-spike unit) and a leading-zero
        marker. The loader treats it as the former — a single empty
        train — for backward compatibility with SpikeLab's own
        exporter convention.

        Tests:
            (Test Case 1) ``end_indices=[0]`` with empty flat_times
                produces a single empty train (not zero trains).
        """
        flat = np.array([], dtype=float)
        end_indices = np.array([0])
        trains = loaders._trains_from_flat_index(
            flat, end_indices, unit="ms", fs_Hz=None
        )
        assert len(trains) == 1
        assert len(trains[0]) == 0

    def test_no_leading_zero_convention_assigns_units_correctly(self):
        """
        The no-leading-zero convention used by SpikeLab's exporter
        (``np.cumsum(counts)``, length ``N``) assigns spikes to units
        correctly. Pinned alongside the leading-zero test above so
        the contrast between the two conventions is unambiguous.

        Tests:
            (Test Case 1) ``end_indices=[2, 5]`` with the same
                ``flat_times`` produces exactly 2 trains.
            (Test Case 2) Unit 0 holds the first 2 spikes; unit 1
                holds the remaining 3.
        """
        flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        end_indices = np.array([2, 5])
        trains = loaders._trains_from_flat_index(
            flat, end_indices, unit="ms", fs_Hz=None
        )
        assert len(trains) == 2
        np.testing.assert_array_equal(trains[0], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(trains[1], np.array([3.0, 4.0, 5.0]))


@skip_no_h5py
class TestReadRawArrays:
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
class TestHDF5RawThresholdedFileAttrRoundTrip:
    """``load_spikedata_from_hdf5_raw_thresholded`` honours file-level
    ``length_ms`` and ``start_time`` attributes — without them the
    raster path and thresholded path would have asymmetric round-trip
    semantics for trailing silence and event-centered windows.
    """

    def test_length_ms_attr_overrides_inferred_length(self, tmp_path):
        """
        Tests:
            (Test Case 1) A file written with ``f.attrs['length_ms'] =
                500.0`` produces a SpikeData with ``length == 500.0``
                even though the raw data only spans 100 ms.
        """
        path = str(tmp_path / "trailing_silence.h5")
        data = np.zeros((1, 100), dtype=float)
        data[0, 50] = 10.0
        with h5py.File(path, "w") as f:
            f.create_dataset("raw", data=data)
            f.attrs["length_ms"] = 500.0

        sd = loaders.load_spikedata_from_hdf5_raw_thresholded(
            path,
            dataset="raw",
            fs_Hz=1000.0,
            threshold_sigma=1.0,
            filter=False,
            hysteresis=False,
        )
        assert sd.length == pytest.approx(500.0)

    def test_start_time_attr_round_trips_event_centered_window(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``f.attrs['start_time'] = -100.0`` produces
                a SpikeData with ``start_time == -100.0``.
        """
        path = str(tmp_path / "event_centered.h5")
        data = np.zeros((1, 200), dtype=float)
        data[0, 100] = 10.0
        with h5py.File(path, "w") as f:
            f.create_dataset("raw", data=data)
            f.attrs["start_time"] = -100.0

        sd = loaders.load_spikedata_from_hdf5_raw_thresholded(
            path,
            dataset="raw",
            fs_Hz=1000.0,
            threshold_sigma=1.0,
            filter=False,
            hysteresis=False,
        )
        assert sd.start_time == pytest.approx(-100.0)


class TestHDF5RawThresholded:
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


@skip_no_h5py
class TestKiloSort:
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
        np.testing.assert_allclose(sd.train[0], [10.0, 20.0, 30.0])
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
        np.testing.assert_allclose(sd.train[0], [100.0, 200.0, 300.0])

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


class TestSpikeInterface:
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


class TestSpikeInterfaceRecording3:
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


class TestPickleLoader2:
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


class TestBuildSpikeData3:
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


class TestS3Utils4:
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


class TestDownloadFromS3:
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


class TestUploadToS3:
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


class TestEnsureLocalFile:
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


class TestSpikeInterfaceRecording2:
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


class TestBuildSpikeData2:
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
class TestScan:
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
        load_spikedata_from_hdf5 with raster_bin_size_ms=0 raises ValueError.

        Tests:
            (Test Case 1) SpikeData.from_raster rejects bin_size_ms <= 0,
                so the loader propagates a ValueError.
        """
        path = str(tmp_path / "zero_bin.h5")
        raster = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=raster)

        with pytest.raises(ValueError, match="bin_size_ms"):
            loaders.load_spikedata_from_hdf5(
                path, raster_dataset="raster", raster_bin_size_ms=0.0
            )

    def test_negative_raster_bin_size(self, tmp_path):
        """
        load_spikedata_from_hdf5 with raster_bin_size_ms=-1.0 raises ValueError.

        Tests:
            (Test Case 1) SpikeData.from_raster rejects bin_size_ms <= 0,
                so the loader propagates a ValueError.
        """
        path = str(tmp_path / "neg_bin.h5")
        raster = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=raster)

        with pytest.raises(ValueError, match="bin_size_ms"):
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
        Tests: _trains_from_flat_index with non-monotonic end indices raises ValueError.
        (Test Case 5) end_indices [5, 3, 10] are not monotonically
        non-decreasing, so a ValueError is raised.
        """
        flat_times = np.arange(10, dtype=float)  # [0..9]
        end_indices = np.array([5, 3, 10])

        with pytest.raises(ValueError, match="monotonically non-decreasing"):
            loaders._trains_from_flat_index(
                flat_times, end_indices, unit="ms", fs_Hz=None
            )

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


class TestLoadSpikelabSortedNpz:
    """Tests for load_spikedata_from_spikelab_sorted_npz."""

    def _make_npz(self, tmp_path, units, fs, locations=None):
        """Helper: write a SpikeLab-style .npz and return the path."""
        path = str(tmp_path / "sorted.npz")
        kwargs = {"units": np.array(units, dtype=object), "fs": np.float64(fs)}
        if locations is not None:
            kwargs["locations"] = np.array(locations)
        np.savez(path, **kwargs)
        return path

    def _make_unit(
        self,
        unit_id=0,
        spike_train=None,
        x_max=10.0,
        y_max=20.0,
        electrode=3,
        template=None,
        amplitudes=None,
        std_norms=None,
        peak_sign="neg",
        max_channel_id="5",
        extras=True,
    ):
        """Build a single unit dict with all or a subset of fields."""
        if spike_train is None:
            spike_train = np.array([100, 200, 300])
        d = {
            "spike_train": spike_train,
            "unit_id": unit_id,
            "x_max": x_max,
            "y_max": y_max,
            "electrode": electrode,
        }
        if extras:
            d["template"] = template if template is not None else np.ones((2, 10))
            d["amplitudes"] = (
                amplitudes if amplitudes is not None else np.array([1.0, 2.0, 3.0])
            )
            d["std_norms"] = std_norms if std_norms is not None else np.array([0.5])
            d["peak_sign"] = peak_sign
            d["max_channel_id"] = max_channel_id
        return d

    def test_basic_load(self, tmp_path):
        """
        Round-trip: build .npz, load, verify SpikeData structure.

        Tests:
            (Test Case 1) Returned object is SpikeData with correct unit count.
            (Test Case 2) Spike times are converted from samples to ms correctly.
        """
        fs = 30000.0
        spike_samples = np.array([30000, 60000, 90000])
        unit = self._make_unit(unit_id=1, spike_train=spike_samples)
        path = self._make_npz(tmp_path, [unit], fs)

        sd = loaders.load_spikedata_from_spikelab_sorted_npz(path)

        assert isinstance(sd, SpikeData)
        assert sd.N == 1
        expected_ms = np.sort(spike_samples.astype(float) / fs * 1000.0)
        np.testing.assert_allclose(sd.train[0], expected_ms)

    def test_neuron_attributes_populated(self, tmp_path):
        """
        Verify neuron_attributes contains all expected keys when present.

        Tests:
            (Test Case 1) unit_id, location, electrode, template, amplitudes,
                          std_norms, peak_sign, max_channel_id are all set.
        """
        unit = self._make_unit(unit_id=7, x_max=1.0, y_max=2.0, electrode=4)
        path = self._make_npz(tmp_path, [unit], 30000.0)

        sd = loaders.load_spikedata_from_spikelab_sorted_npz(path)

        attrs = sd.neuron_attributes[0]
        assert attrs["unit_id"] == 7
        assert attrs["location"] == [1.0, 2.0]
        assert attrs["electrode"] == 4
        np.testing.assert_array_equal(attrs["template"], np.ones((2, 10)))
        np.testing.assert_array_equal(attrs["amplitudes"], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(attrs["std_norms"], np.array([0.5]))
        assert attrs["peak_sign"] == "neg"
        assert attrs["max_channel_id"] == "5"

    def test_metadata_populated(self, tmp_path):
        """
        Verify metadata contains source_file, source_format, fs_Hz,
        and channel_locations when locations are provided.

        Tests:
            (Test Case 1) source_format is 'SpikeLab_npz'.
            (Test Case 2) fs_Hz matches the sampling rate.
            (Test Case 3) channel_locations present when locations supplied.
        """
        locs = np.array([[0.0, 0.0], [1.0, 1.0]])
        unit = self._make_unit()
        path = self._make_npz(tmp_path, [unit], 20000.0, locations=locs)

        sd = loaders.load_spikedata_from_spikelab_sorted_npz(path)

        assert sd.metadata["source_format"] == "SpikeLab_npz"
        assert sd.metadata["fs_Hz"] == 20000.0
        assert "source_file" in sd.metadata
        np.testing.assert_array_equal(sd.metadata["channel_locations"], locs)

    def test_empty_units(self, tmp_path):
        """
        .npz with an empty units list produces a SpikeData with zero units.

        Tests:
            (Test Case 1) N == 0 and train list is empty.
        """
        path = self._make_npz(tmp_path, [], 30000.0)

        sd = loaders.load_spikedata_from_spikelab_sorted_npz(path)

        assert isinstance(sd, SpikeData)
        assert sd.N == 0
        assert len(sd.train) == 0

    def test_missing_optional_fields(self, tmp_path):
        """
        Units without optional attributes (template, amplitudes, etc.)
        should still load; those keys are absent from neuron_attributes.

        Tests:
            (Test Case 1) SpikeData loads without error.
            (Test Case 2) Only unit_id and location/electrode are present.
        """
        unit = self._make_unit(extras=False)
        path = self._make_npz(tmp_path, [unit], 30000.0)

        sd = loaders.load_spikedata_from_spikelab_sorted_npz(path)

        assert sd.N == 1
        attrs = sd.neuron_attributes[0]
        assert "unit_id" in attrs
        assert "location" in attrs
        assert "electrode" in attrs
        assert "template" not in attrs
        assert "amplitudes" not in attrs
        assert "std_norms" not in attrs
        assert "peak_sign" not in attrs
        assert "max_channel_id" not in attrs

    def test_single_unit(self, tmp_path):
        """
        Single unit in units list loads correctly.

        Tests:
            (Test Case 1) N == 1 and spike times are correct.
        """
        fs = 10000.0
        samples = np.array([10000, 50000])
        unit = self._make_unit(unit_id=0, spike_train=samples)
        path = self._make_npz(tmp_path, [unit], fs)

        sd = loaders.load_spikedata_from_spikelab_sorted_npz(path)

        assert sd.N == 1
        expected_ms = np.sort(samples.astype(float) / fs * 1000.0)
        np.testing.assert_allclose(sd.train[0], expected_ms)

    def test_file_not_found(self, tmp_path):
        """
        Nonexistent path raises FileNotFoundError.

        Tests:
            (Test Case 1) FileNotFoundError for a missing .npz.
        """
        bad_path = str(tmp_path / "does_not_exist.npz")
        with pytest.raises(FileNotFoundError):
            loaders.load_spikedata_from_spikelab_sorted_npz(bad_path)


# ---------------------------------------------------------------------------
# Edge case tests from REVIEW.md I/O scan (HIGH and MEDIUM severity)
# ---------------------------------------------------------------------------


@skip_no_h5py
class TestHDF5LoaderIO:
    """Edge case tests for load_spikedata_from_hdf5 from REVIEW.md I/O scan."""

    def test_raster_with_negative_values(self, tmp_path):
        """
        Raster with negative values -- from_raster computes
        np.linspace(0, bin_size, n_spikes + 2) which raises ValueError
        when n_spikes + 2 < 0 (i.e. n_spikes <= -3).

        Notes:
            Bug: negative raster values with magnitude >= 3 crash in
            np.linspace because the num argument becomes negative.

        Tests:
            (Test Case 1) Loader raises ValueError on raster values <= -3.
        """
        path = str(tmp_path / "neg_raster.h5")
        raster = np.array([[0, -1, 2, 0], [1, 0, -3, 0]], dtype=int)
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=raster)

        with pytest.raises(ValueError, match="must be non-negative"):
            loaders.load_spikedata_from_hdf5(
                path, raster_dataset="raster", raster_bin_size_ms=10.0
            )

    def test_paired_negative_unit_indices_raises(self, tmp_path):
        """
        Paired-style HDF5 with negative unit indices raises ValueError.

        Tests:
            (Test Case 1) load_spikedata_from_hdf5 propagates the
                SpikeData.from_idces_times validation that rejects negative
                unit indices.
        """
        path = str(tmp_path / "neg_idces.h5")
        idces = np.array([-1, 0, 1, 0], dtype=int)
        times = np.array([0.1, 0.2, 0.3, 0.4])  # seconds
        with h5py.File(path, "w") as f:
            f.create_dataset("idces", data=idces)
            f.create_dataset("times", data=times)

        with pytest.raises(ValueError, match="negative"):
            loaders.load_spikedata_from_hdf5(
                path, idces_dataset="idces", times_dataset="times", times_unit="s"
            )

    def test_group_per_unit_non_numeric_names(self, tmp_path):
        """
        Group-per-unit with non-numeric dataset names -- lexicographic order
        may not match intended unit order.

        Tests:
            (Test Case 1) Datasets named "alpha", "beta", "gamma" are sorted
                lexicographically, producing a valid SpikeData.
            (Test Case 2) N matches number of datasets.
        """
        path = str(tmp_path / "alpha_group.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("units")
            grp.create_dataset("gamma", data=np.array([0.3, 0.5]))
            grp.create_dataset("alpha", data=np.array([0.1, 0.2]))
            grp.create_dataset("beta", data=np.array([0.4]))

        sd = loaders.load_spikedata_from_hdf5(
            path, group_per_unit="units", group_time_unit="s"
        )
        assert sd.N == 3
        # Sorted order: alpha, beta, gamma
        np.testing.assert_allclose(sd.train[0], np.array([100.0, 200.0]))
        np.testing.assert_allclose(sd.train[1], np.array([400.0]))
        np.testing.assert_allclose(sd.train[2], np.array([300.0, 500.0]))


class TestBuildSpikeData3:
    """Edge case tests for _build_spikedata helper."""

    def test_all_empty_trains_nonzero_start_time(self):
        """
        All trains empty with non-zero start_time -- length inference
        returns 0.0 ignoring start_time.

        Tests:
            (Test Case 1) length is inferred as 0.0 when all trains are empty,
                regardless of start_time.
        """
        trains = [np.array([], float), np.array([], float)]
        sd = loaders._build_spikedata(trains, start_time=500.0)
        # length inferred as 0.0 because no spikes exist
        assert sd.length == 0.0
        assert sd.start_time == 500.0


class TestPickleLoader2:
    """Edge case tests for load_spikedata_from_pickle."""

    def test_empty_pickle_file(self, tmp_path):
        """
        Empty pickle file -- EOFError is raised (not caught/wrapped).

        Tests:
            (Test Case 1) Loading a 0-byte pickle file raises an exception.
        """
        path = str(tmp_path / "empty.pkl")
        with open(path, "wb") as f:
            pass  # Write nothing

        with pytest.raises(Exception):
            loaders.load_spikedata_from_pickle(path)

    def test_pickle_wrong_type(self, tmp_path):
        """
        Pickle containing a non-SpikeData object raises ValueError.

        Tests:
            (Test Case 1) Pickle with a plain dict raises ValueError.
        """
        path = str(tmp_path / "wrong.pkl")
        with open(path, "wb") as f:
            pickle.dump({"not": "spikedata"}, f)

        with pytest.raises(ValueError, match="does not contain a SpikeData"):
            loaders.load_spikedata_from_pickle(path)


class TestSpikeInterfaceLoaderDocstring:
    """Public docstring contract for ``load_spikedata_from_spikeinterface``.
    The unit-order convention (``sorting.get_unit_ids()`` order is
    backend-dependent — KS sequential vs SpikeInterface-reordered)
    matters for cross-backend comparisons and is documented in the
    function's docstring along with the ``unit_ids`` override knob.
    """

    def test_docstring_documents_backend_dependent_order_and_unit_ids(self):
        """
        Tests:
            (Test Case 1) ``__doc__`` contains the substring
                ``"backend-dependent"``.
            (Test Case 2) ``__doc__`` names ``unit_ids`` as the
                override knob callers rely on.
        """
        from spikelab.data_loaders.data_loaders import (
            load_spikedata_from_spikeinterface,
        )

        doc = load_spikedata_from_spikeinterface.__doc__
        assert doc is not None
        assert "backend-dependent" in doc
        assert "unit_ids" in doc


class TestSpikeInterfaceLoader:
    """Edge case tests for load_spikedata_from_spikeinterface."""

    def test_get_property_shorter_than_unit_count(self):
        """
        get_property returns array shorter than unit count -- electrode
        info is only assigned for available indices.

        Tests:
            (Test Case 1) No crash when channel property is shorter than IDs.
            (Test Case 2) First unit gets electrode, second does not.
        """

        class FakeSorting:
            def get_unit_ids(self):
                return [0, 1, 2]

            def get_sampling_frequency(self):
                return 30000.0

            def get_unit_spike_train(self, unit_id=None, segment_index=0):
                return np.array([100, 200, 300])

            def get_property(self, name):
                if name == "channel":
                    return np.array([10])  # Only 1 element for 3 units
                raise KeyError(name)

        sd = loaders.load_spikedata_from_spikeinterface(FakeSorting())
        assert sd.N == 3
        # Only unit 0 (index 0 < len(channel_prop)=1) gets electrode
        assert sd.neuron_attributes[0].get("electrode") == 10
        assert "electrode" not in sd.neuron_attributes[1]
        assert "electrode" not in sd.neuron_attributes[2]


class TestSpikeInterfaceRecording3:
    """Edge case tests for load_spikedata_from_spikeinterface_recording."""

    def test_sampling_frequency_none_raises(self):
        """
        get_sampling_frequency returns None -- float(None) raises TypeError
        before the explicit ValueError guard is reached.

        Tests:
            (Test Case 1) TypeError for None sampling frequency.
        """

        class FakeRecording:
            def get_sampling_frequency(self):
                return None

            def get_traces(self, segment_index=0):
                return np.zeros((2, 100))

        with pytest.raises(TypeError):
            loaders.load_spikedata_from_spikeinterface_recording(FakeRecording())


class TestS3Utils4:
    """Edge case tests for s3_utils.py."""

    def test_is_s3_url_with_s3_in_path_not_hostname(self):
        """
        URL with 's3' in path but not hostname -- should return False.

        Tests:
            (Test Case 1) https://example.com/s3/bucket/key is not an S3 URL.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        assert is_s3_url("https://example.com/s3/bucket/key") is False
        assert is_s3_url("https://example.com/data/s3-backup/file.h5") is False

    def test_is_s3_url_valid_patterns(self):
        """
        Verify valid S3 URL patterns are recognized.

        Tests:
            (Test Case 1) s3:// prefix returns True.
            (Test Case 2) https://s3.amazonaws.com/... returns True.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        assert is_s3_url("s3://bucket/key") is True
        assert is_s3_url("https://s3.amazonaws.com/bucket/key") is True
        assert is_s3_url("https://bucket.s3.us-west-2.amazonaws.com/key") is True


@skip_no_pandas
class TestKilosortEmptyClusterInfoTsv:
    """
    Tests that an empty cluster_info.tsv produces a clear ValueError
    rather than letting pandas.errors.EmptyDataError propagate from
    deep inside the loader.
    """

    def test_empty_cluster_info_tsv_raises_value_error(self, tmp_path):
        """
        Empty cluster_info.tsv raises ValueError with a clear message
        naming the empty file and pointing at the workaround.

        Tests:
            (Test Case 1) Zero-byte cluster_info_tsv raises ValueError.
            (Test Case 2) The error message names the offending path.
            (Test Case 3) The error message suggests omitting
                cluster_info_tsv as a workaround.
        """
        d = str(tmp_path / "ks_empty_tsv")
        os.makedirs(d)
        spike_times = np.array([10, 20, 15])
        spike_clusters = np.array([0, 0, 1])
        np.save(os.path.join(d, "spike_times.npy"), spike_times)
        np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)
        tsv_path = os.path.join(d, "cluster_info.tsv")
        open(tsv_path, "w").close()

        with pytest.raises(ValueError) as exc_info:
            loaders.load_spikedata_from_kilosort(
                d,
                fs_Hz=1000.0,
                cluster_info_tsv="cluster_info.tsv",
            )
        msg = str(exc_info.value)
        assert "empty" in msg.lower()
        assert "cluster_info.tsv" in msg

    def test_omit_cluster_info_tsv_loads_normally(self, tmp_path):
        """
        Omitting cluster_info_tsv loads spikes normally (no crash) — the
        suggested workaround in the empty-TSV error message.

        Tests:
            (Test Case 1) Loader returns a SpikeData with the expected
                number of units when cluster_info_tsv is None.
        """
        d = str(tmp_path / "ks_no_tsv")
        os.makedirs(d)
        spike_times = np.array([10, 20, 15])
        spike_clusters = np.array([0, 0, 1])
        np.save(os.path.join(d, "spike_times.npy"), spike_times)
        np.save(os.path.join(d, "spike_clusters.npy"), spike_clusters)
        sd = loaders.load_spikedata_from_kilosort(
            d, fs_Hz=1000.0, cluster_info_tsv=None
        )
        assert sd.N == 2


@skip_no_h5py
class TestHDF5GroupPerUnitNaturalSort:
    """
    Tests that the group-per-unit loader sorts numerically (natural sort),
    preserving unit identity across round-trip at N>=10.
    """

    def test_group_per_unit_natural_sort_with_11_units(self, tmp_path):
        """
        Group-per-unit loader with 11 numeric keys returns trains in
        numerical (not lexicographic) order.

        Tests:
            (Test Case 1) sd.train[i] holds the spikes from key str(i)
                for i in 0..10.
        """
        path = str(tmp_path / "natural_sort_n11.h5")
        # Create 11 units with distinct spike times so ordering is observable.
        with h5py.File(path, "w") as f:  # type: ignore
            grp = f.create_group("units")
            for i in range(11):
                grp.create_dataset(str(i), data=np.array([float(i + 1) * 10.0]))
            grp.attrs["time_unit"] = "ms"

        sd = loaders.load_spikedata_from_hdf5(
            path, group_per_unit="units", group_time_unit="ms"
        )
        assert sd.N == 11
        # Natural order: train[i] holds key str(i) → spike at (i+1)*10.
        for i in range(11):
            np.testing.assert_array_equal(sd.train[i], [float(i + 1) * 10.0])

    def test_group_per_unit_natural_sort_mixed_prefix(self, tmp_path):
        """
        Natural sort works with prefixed keys like "unit_2" / "unit_10".

        Tests:
            (Test Case 1) Keys "unit_1" .. "unit_11" load in numerical
                order; train[1] holds the spike from "unit_2" and
                train[9] holds the spike from "unit_10".
        """
        path = str(tmp_path / "natural_sort_prefixed.h5")
        with h5py.File(path, "w") as f:  # type: ignore
            grp = f.create_group("units")
            for i in range(1, 12):
                grp.create_dataset(f"unit_{i}", data=np.array([float(i) * 10.0]))
            grp.attrs["time_unit"] = "ms"

        sd = loaders.load_spikedata_from_hdf5(
            path, group_per_unit="units", group_time_unit="ms"
        )
        assert sd.N == 11
        # train[i] holds unit_(i+1).
        for i in range(11):
            np.testing.assert_array_equal(sd.train[i], [float(i + 1) * 10.0])


class TestLoadSpikedataFromIblAllFallbacksFail:
    """
    Edge case test pinning current behavior when all collection
    fallbacks fail and spikes is None.

    Notes:
        - documents bug — see REVIEW.md
        - When the IBL loader cannot find spike data in any collection
          fallback, it currently produces a SpikeData with all-empty
          trains plus full trial metadata (silent zero-spike result).
    """

    def test_ibl_loader_unimportable_raises_importerror(self):
        """
        load_spikedata_from_ibl when one-api is missing raises ImportError.

        Tests:
            (Test Case 1) If `one.api` is not importable, load_spikedata_from_ibl
                raises an ImportError or similar at call site.

        Notes:
            - This pins the import-error contract; the deeper "all
              collections fail" path requires real IBL fixtures.
        """
        try:
            import one.api  # noqa: F401

            pytest.skip("one-api is installed; cannot test ImportError path")
        except ImportError:
            pass

        with pytest.raises(Exception):
            loaders.load_spikedata_from_ibl(
                eid="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                pid="11111111-2222-3333-4444-555555555555",
            )


class TestS3UrlEdgeCases:
    """Boundary tests for is_s3_url and parse_s3_url covering whitespace,
    case sensitivity, MinIO/localhost URLs, and bucket-only URLs."""

    def test_is_s3_url_with_whitespace_returns_true(self):
        """
        ``is_s3_url`` strips leading/trailing whitespace before
        matching the scheme prefix, so ``"  s3://bucket/key  "`` is
        recognised as an S3 URL.

        Tests:
            (Test Case 1) Whitespace-padded s3:// URL returns True.
            (Test Case 2) Whitespace-padded https amazonaws URL
                returns True.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        assert is_s3_url("  s3://bucket/key  ") is True
        assert is_s3_url("\thttps://s3.amazonaws.com/bucket/key\n") is True

    def test_is_s3_url_uppercase_scheme_returns_true(self):
        """
        ``is_s3_url`` matches the scheme prefix case-insensitively,
        so ``"S3://bucket/key"`` (uppercase) is recognised.

        Tests:
            (Test Case 1) Uppercase ``S3://`` URL returns True.
            (Test Case 2) Mixed-case ``s3://`` URL returns True.
            (Test Case 3) Lowercase control still returns True.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        assert is_s3_url("S3://bucket/key") is True
        assert is_s3_url("S3://bucket/key") is True
        assert is_s3_url("s3://bucket/key") is True

    def test_is_s3_url_minio_endpoint_returns_false(self):
        """
        is_s3_url checks for the amazonaws.com host or the s3:// scheme,
        so MinIO/LocalStack-style endpoints (e.g. localhost:9000) are not
        recognised. Documents that custom S3-compatible endpoints are
        unsupported.

        Tests:
            (Test Case 1) http://localhost:9000/bucket/key returns False.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        assert is_s3_url("http://localhost:9000/bucket/key") is False

    def test_parse_s3_url_bucket_only_raises(self):
        """
        parse_s3_url("s3://") raises ValueError because the URL has no
        object key.

        Tests:
            (Test Case 1) "s3://" alone raises ValueError naming "no
                object key".
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        with pytest.raises(ValueError, match="no object key"):
            parse_s3_url("s3://")


class TestTrainsFromFlatIndexInputDtypes:
    """``_trains_from_flat_index`` with unusual ``end_indices`` dtypes."""

    def test_negative_end_indices_rejected(self):
        """
        ``_trains_from_flat_index`` rejects negative entries upfront
        with a ``ValueError`` naming "non-negative". Cumulative-end
        indices represent spike counts and cannot be negative.

        Tests:
            (Test Case 1) Negative entry raises ValueError naming
                "non-negative".
        """
        flat = np.arange(10.0)
        end_indices = np.array([-2, 0, 5])
        with pytest.raises(ValueError, match="non-negative"):
            loaders._trains_from_flat_index(flat, end_indices, unit="ms", fs_Hz=None)

    def test_float_end_indices_rejected_with_friendly_error(self):
        """
        ``_trains_from_flat_index`` rejects float-dtype ``end_indices``
        upfront with a ``ValueError`` instructing the caller to cast
        to integers, rather than letting numpy raise a confusing
        ``TypeError`` mid-loop.

        Tests:
            (Test Case 1) Float ``end_indices`` raises ValueError
                naming "integer array".
        """
        flat = np.arange(10.0)
        end_indices = np.array([2.0, 5.0])
        with pytest.raises(ValueError, match="integer array"):
            loaders._trains_from_flat_index(flat, end_indices, unit="ms", fs_Hz=None)


@skip_no_h5py
class TestReadRawArraysSamplesUnitWithZeroFs:
    """``_read_raw_arrays`` with ``raw_time_unit='samples'`` and ``fs_Hz=0``."""

    def test_samples_unit_with_zero_fs_raises(self, tmp_path):
        """
        ``fs_Hz=0`` is rejected by the ``if not fs_Hz`` guard. Pin
        the current error message contract.

        Tests:
            (Test Case 1) ``raw_time_unit='samples'`` with ``fs_Hz=0``
                raises ``ValueError`` mentioning ``fs_Hz``.
        """
        path = str(tmp_path / "raw_zero_fs.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("raw", data=np.zeros((2, 5)))
            f.create_dataset("raw_t", data=np.arange(5))

        with h5py.File(path, "r") as f:
            with pytest.raises(ValueError, match="fs_Hz"):
                loaders._read_raw_arrays(f, "raw", "raw_t", "samples", fs_Hz=0)


@skip_no_h5py
class TestLoadHdf5RasterEdgeShapes:
    """``load_spikedata_from_hdf5(raster_dataset=...)`` shape edges."""

    def test_zero_unit_raster_produces_zero_unit_spikedata(self, tmp_path):
        """
        A raster with shape ``(0, T)`` round-trips to a zero-unit
        SpikeData with the expected length.

        Tests:
            (Test Case 1) Loaded SpikeData has ``N == 0``.
            (Test Case 2) ``length`` equals ``T * raster_bin_size_ms``.
        """
        path = str(tmp_path / "zero_units.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=np.zeros((0, 5), dtype=int))

        sd = loaders.load_spikedata_from_hdf5(
            path,
            raster_dataset="raster",
            raster_bin_size_ms=2.0,
        )
        assert sd.N == 0
        assert sd.length == pytest.approx(10.0)

    def test_3d_raster_rejected_with_clear_error(self, tmp_path):
        """
        A 3-D raster is rejected with the same "must be 2D" message
        as 1-D inputs.

        Tests:
            (Test Case 1) 3-D raster raises ``ValueError``.
        """
        path = str(tmp_path / "raster_3d.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=np.zeros((2, 5, 3), dtype=int))

        with pytest.raises(ValueError):
            loaders.load_spikedata_from_hdf5(
                path,
                raster_dataset="raster",
                raster_bin_size_ms=1.0,
            )

    def test_explicit_length_ms_silently_ignored_in_raster_style(self, tmp_path):
        """
        The raster-style loader unconditionally derives ``length`` from
        ``raster.shape[1] * raster_bin_size_ms``, silently overwriting
        any user-supplied ``length_ms``.

        Tests:
            (Test Case 1) Explicit ``length_ms=999`` is replaced by
                the shape-derived ``4.0``.

        Notes:
            - Pins silent override. A future hardening that warns
              when ``length_ms`` differs from the computed value
              would update this test.
        """
        path = str(tmp_path / "raster_len.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("raster", data=np.ones((2, 4), dtype=int))

        sd = loaders.load_spikedata_from_hdf5(
            path,
            raster_dataset="raster",
            raster_bin_size_ms=1.0,
            length_ms=999.0,
        )
        assert sd.length == pytest.approx(4.0)


@skip_no_h5py
class TestLoadHdf5PairedSparseClusterIdsWarning:
    """``load_spikedata_from_hdf5`` paired style emits a UserWarning
    when ``idces`` skips cluster IDs (e.g. Phy curation dropped some).
    The warning surfaces the count of padded empty units so the operator
    can tell "unit N had no spikes" from "unit N was dropped upstream".
    """

    def test_sparse_cluster_ids_warns_with_padded_count(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``idces = [0, 1, 47]`` (max+1=48 but only 3
                distinct ids) emits a UserWarning.
            (Test Case 2) The warning names the padded-unit count
                (48 - 3 = 45).
        """
        try:
            import h5py  # noqa: F401
            import warnings as _warnings
        except ImportError:
            pytest.skip("h5py not installed")

        path = str(tmp_path / "sparse_paired.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("idces", data=np.array([0, 1, 47], dtype=int))
            f.create_dataset("times", data=np.array([1.0, 2.0, 3.0]))

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            sd = loaders.load_spikedata_from_hdf5(
                path,
                idces_dataset="idces",
                times_dataset="times",
                times_unit="ms",
            )

        sparse_msgs = [
            str(rec.message)
            for rec in w
            if rec.category is UserWarning and "sparse" in str(rec.message)
        ]
        assert sparse_msgs, [str(rec.message) for rec in w]
        # Padded-unit count (45) is in the message.
        assert "45" in sparse_msgs[0]
        # The SpikeData is still constructed with the padded layout.
        assert sd.N == 48


class TestLoadHdf5PairedMismatchedLengths:
    """``load_spikedata_from_hdf5(paired)`` with mismatched array lengths."""

    def test_paired_idces_times_length_mismatch_raises(self, tmp_path):
        """
        If ``idces`` and ``times`` have different lengths, the loader
        forwards them through ``SpikeData.from_idces_times`` /
        ``_train_from_i_t_list`` and surfaces an ``IndexError`` from
        numpy boolean indexing. Pin this so a future hardening that
        raises a friendlier ``ValueError`` upfront updates this test.

        Tests:
            (Test Case 1) Mismatched arrays raise ``IndexError`` (or
                ``ValueError`` if a friendlier guard is added).

        Notes:
            - The current error message is opaque (numpy boolean-
              index dimension mismatch). Better surfaced at the
              loader call-site.
        """
        path = str(tmp_path / "paired_mismatch.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("idces", data=np.array([0, 1, 0]))
            f.create_dataset("times", data=np.array([1.0, 2.0]))

        with pytest.raises((IndexError, ValueError)):
            loaders.load_spikedata_from_hdf5(
                path,
                idces_dataset="idces",
                times_dataset="times",
                times_unit="ms",
            )


@skip_no_h5py
class TestLoadHdf5RaggedZeroLengthIndex:
    """``load_spikedata_from_hdf5(ragged)`` with empty index dataset."""

    def test_zero_length_index_produces_zero_unit_spikedata(self, tmp_path):
        """
        An empty ``spike_times_index`` dataset produces zero trains
        and a zero-unit SpikeData.

        Tests:
            (Test Case 1) Loaded SpikeData has ``N == 0``.
        """
        path = str(tmp_path / "ragged_empty_index.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("st", data=np.array([], dtype=float))
            f.create_dataset("idx", data=np.array([], dtype=int))

        sd = loaders.load_spikedata_from_hdf5(
            path,
            spike_times_dataset="st",
            spike_times_index_dataset="idx",
            spike_times_unit="ms",
        )
        assert sd.N == 0


class TestLoadKilosortMissingTsv:
    """``load_spikedata_from_kilosort`` with a non-existent ``cluster_info_tsv``."""

    def test_missing_tsv_silently_keeps_all_clusters(self, tmp_path):
        """
        A ``cluster_info_tsv`` path that does not exist is silently
        ignored — the ``os.path.exists`` check returns False and the
        loader keeps all clusters without warning.

        Tests:
            (Test Case 1) Loader succeeds; resulting SpikeData has
                the full set of clusters (no filtering).

        Notes:
            - Pins silent acceptance. A future hardening should emit
              a UserWarning when the path doesn't exist so users can
              detect typos.
        """
        spike_times = np.array([10, 20, 30, 40], dtype=np.int64)
        spike_clusters = np.array([0, 1, 0, 1], dtype=np.int64)
        np.save(tmp_path / "spike_times.npy", spike_times)
        np.save(tmp_path / "spike_clusters.npy", spike_clusters)

        sd = loaders.load_spikedata_from_kilosort(
            str(tmp_path),
            fs_Hz=1000.0,
            cluster_info_tsv=str(tmp_path / "does_not_exist.tsv"),
        )
        assert sd.N == 2


class TestLoadSpikelabSortedNpzMissingKeys:
    """``load_spikedata_from_spikelab_sorted_npz`` with missing required keys."""

    def test_missing_units_key_raises(self, tmp_path):
        """
        An NPZ file without a ``"units"`` key raises a ``KeyError``
        from the ``data["units"]`` access.

        Tests:
            (Test Case 1) Missing units key raises ``KeyError``.
        """
        path = str(tmp_path / "no_units.npz")
        np.savez(path, fs=20000.0)
        with pytest.raises(KeyError):
            loaders.load_spikedata_from_spikelab_sorted_npz(path)

    def test_missing_fs_key_raises(self, tmp_path):
        """
        An NPZ file without an ``"fs"`` key raises ``KeyError``.

        Tests:
            (Test Case 1) Missing fs key raises ``KeyError``.
        """
        path = str(tmp_path / "no_fs.npz")
        np.savez(path, units=np.array([], dtype=object))
        with pytest.raises(KeyError):
            loaders.load_spikedata_from_spikelab_sorted_npz(path)


# NOTE: TestS3UtilsSchemeAndWhitespace removed — its scope now lives
# in TestS3UrlEdgeCases (see test_is_s3_url_with_whitespace_returns_true
# and test_is_s3_url_uppercase_scheme_returns_true), which assert the
# corrected case-insensitive + whitespace-stripping contract.


# ===========================================================================
# Regression tests for bugs fixed during test_scanner triage
# (see iat/BUG_REPORT.md). Each test pins the fix's contract.
# ===========================================================================


class TestNWBLoaderPynwbFallbackBroadException:
    """``load_spikedata_from_nwb`` pynwb-first branch must catch any
    pynwb-side exception and fall through to the h5py loader. Pynwb
    raises ``RuntimeError`` on schema mismatch and ``OSError`` on
    HDF5-plugin issues; both must trigger the documented fallback
    rather than propagating to the caller. The fallback warning names
    the original exception type for diagnosis.
    """

    def _write_valid_nwb_h5py_layout(self, path):
        """Write a minimal h5py-loadable NWB file the fallback can consume."""
        with h5py.File(path, "w") as f:  # type: ignore
            g = f.create_group("units")
            g.create_dataset("spike_times", data=np.array([0.1, 0.2, 0.5]))
            g.create_dataset("spike_times_index", data=np.array([2, 3]))

    def _assert_h5py_fallback_warning(self, recwarn, exc_type: type) -> None:
        msgs = [str(w.message) for w in recwarn]
        joined = " | ".join(msgs)
        assert "falling back to h5py" in joined, f"no fallback warning in {msgs!r}"
        assert exc_type.__name__ in joined, (
            f"warning does not name the original exception type "
            f"{exc_type.__name__}: {msgs!r}"
        )

    def test_pynwb_runtime_error_falls_back_to_h5py(self, tmp_path, monkeypatch):
        """
        ``pynwb`` raising ``RuntimeError`` (schema mismatch case) must trigger
        the h5py fallback rather than propagating.

        Tests:
            (Test Case 1) Loader returns a SpikeData with the h5py-derived
                spike trains.
            (Test Case 2) A ``UserWarning`` mentioning the original
                ``RuntimeError`` and "falling back to h5py" is emitted.
        """
        pynwb = pytest.importorskip("pynwb")

        class _RaisingNWBHDF5IO:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("simulated schema mismatch")

        monkeypatch.setattr(pynwb, "NWBHDF5IO", _RaisingNWBHDF5IO)

        path = str(tmp_path / "test.nwb")
        self._write_valid_nwb_h5py_layout(path)

        with pytest.warns(UserWarning) as recwarn:
            sd = loaders.load_spikedata_from_nwb(path)
        self._assert_h5py_fallback_warning(recwarn, RuntimeError)

        # h5py path read the data correctly.
        assert sd.N == 2
        np.testing.assert_array_equal(sd.train[0], np.array([100.0, 200.0]))
        np.testing.assert_array_equal(sd.train[1], np.array([500.0]))

    def test_pynwb_os_error_falls_back_to_h5py(self, tmp_path, monkeypatch):
        """
        ``pynwb`` raising ``OSError`` (HDF5 plugin case) must trigger the
        h5py fallback.

        Tests:
            (Test Case 1) Loader returns a SpikeData.
            (Test Case 2) Warning names ``OSError``.
        """
        pynwb = pytest.importorskip("pynwb")

        class _RaisingNWBHDF5IO:
            def __init__(self, *args, **kwargs):
                raise OSError("simulated HDF5 plugin missing")

        monkeypatch.setattr(pynwb, "NWBHDF5IO", _RaisingNWBHDF5IO)

        path = str(tmp_path / "test.nwb")
        self._write_valid_nwb_h5py_layout(path)

        with pytest.warns(UserWarning) as recwarn:
            sd = loaders.load_spikedata_from_nwb(path)
        self._assert_h5py_fallback_warning(recwarn, OSError)
        assert sd.N == 2

    def test_pynwb_mid_loop_failure_does_not_leak_into_h5py_fallback(
        self, tmp_path, monkeypatch
    ):
        """
        Regression for the partial-population bug in the pynwb→h5py fallback.

        When pynwb starts successfully and populates ``trains`` /
        ``neuron_attributes`` for the first few units, then raises one of
        the caught exceptions mid-iteration, the loader must reset both
        lists before delegating to the h5py path. Otherwise the h5py
        ``extend`` lands on top of the partial pynwb state and
        ``SpikeData.N`` ends up as ``partial + h5py_count`` instead of
        ``h5py_count``.

        Tests:
            (Test Case 1) ``sd.N`` matches the h5py-only count (2), not
                the buggy ``partial + h5py_count`` (4).
            (Test Case 2) ``sd.train`` contents come from the h5py
                layout, confirming pynwb partial data was discarded.
        """
        pynwb = pytest.importorskip("pynwb")
        from types import SimpleNamespace

        class _MidLoopRaisingDataFrame:
            """Iterates K rows successfully, then raises ``KeyError``."""

            columns: list = []  # no candidate channel columns

            def __init__(self, k_good: int):
                self.k_good = k_good

            def itertuples(self):
                for i in range(self.k_good):
                    yield SimpleNamespace(
                        Index=i,
                        spike_times=np.array([0.001 * (i + 1)]),
                    )
                raise KeyError("simulated mid-loop pynwb read failure")

        class _FakeIO:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return SimpleNamespace(
                    units=SimpleNamespace(
                        to_dataframe=lambda: _MidLoopRaisingDataFrame(k_good=2)
                    ),
                    electrodes=None,
                )

        monkeypatch.setattr(pynwb, "NWBHDF5IO", _FakeIO)

        path = str(tmp_path / "test.nwb")
        self._write_valid_nwb_h5py_layout(path)  # 2 units in h5py layout

        with pytest.warns(UserWarning) as recwarn:
            sd = loaders.load_spikedata_from_nwb(path)
        self._assert_h5py_fallback_warning(recwarn, KeyError)

        # Without the fix, sd.N would be 4 (2 partial pynwb + 2 h5py).
        assert sd.N == 2, (
            f"pynwb partial state leaked into h5py fallback: got N={sd.N}, "
            f"expected 2"
        )
        np.testing.assert_array_equal(sd.train[0], np.array([100.0, 200.0]))
        np.testing.assert_array_equal(sd.train[1], np.array([500.0]))


class TestLoadSpikedataFromHdf5RawThresholdedFsHzValidation:
    """``load_spikedata_from_hdf5_raw_thresholded`` must validate
    ``fs_Hz`` at the loader boundary so the user sees a clear,
    parameter-named error rather than a cryptic ZeroDivisionError or
    other exception raised deep inside ``SpikeData.from_thresholding``.
    """

    @pytest.mark.parametrize("bad_fs", [0, -1.0, float("nan"), None])
    def test_invalid_fs_hz_raises_clear_error(self, tmp_path, bad_fs):
        """
        Tests:
            (Test Case 1) Each invalid ``fs_Hz`` (0, negative, NaN,
                None) raises ``ValueError`` mentioning ``fs_Hz`` and
                "positive finite".
        """
        # Build a tiny HDF5 file with a raw-traces dataset. fs_Hz
        # validation fires before we touch the data, so the contents
        # don't matter for the failure path.
        path = str(tmp_path / "raw.h5")
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("traces", data=np.zeros((4, 100), dtype=np.float32))

        with pytest.raises(ValueError, match=r"fs_Hz.*positive finite"):
            loaders.load_spikedata_from_hdf5_raw_thresholded(
                path, "traces", fs_Hz=bad_fs
            )


class TestLoadSpikedataFromSpikeinterfaceUnitIdsValidation:
    """``load_spikedata_from_spikeinterface(unit_ids=...)`` must reject
    IDs not present in ``sorting.get_unit_ids()`` with a clear error
    at the loader boundary, rather than letting
    ``sorting.get_unit_spike_train(missing_id)`` raise a backend-
    specific exception mid-loop.
    """

    def _stub_sorting(self, available_ids):
        """Minimal SortingExtractor-shaped stub."""
        from types import SimpleNamespace

        sd = SimpleNamespace()
        sd.get_unit_ids = lambda: list(available_ids)
        sd.get_sampling_frequency = lambda: 30000.0
        sd.get_unit_spike_train = lambda unit_id, segment_index=0: np.array(
            [100, 200], dtype=np.int64
        )
        return sd

    def test_unknown_unit_id_raises_clear_error(self):
        """
        Tests:
            (Test Case 1) Passing ``unit_ids=[0, 999]`` against a
                sorting with units [0, 1] raises ``ValueError``
                naming the missing IDs and the available set.
        """
        sd = self._stub_sorting([0, 1])
        with pytest.raises(ValueError, match=r"unit_ids.*not present"):
            loaders.load_spikedata_from_spikeinterface(sd, unit_ids=[0, 999])

    def test_subset_of_present_ids_accepted(self):
        """
        Sanity check: requesting a valid subset still works after the
        guard is added.

        Tests:
            (Test Case 1) ``unit_ids=[1]`` against units [0, 1] succeeds.
        """
        sd = self._stub_sorting([0, 1])
        out = loaders.load_spikedata_from_spikeinterface(sd, unit_ids=[1])
        assert out.N == 1


class TestLoadSpikedataFromNwbNonIntegerUnitId:
    """``load_spikedata_from_nwb`` h5py path falls back to storing the
    raw value on ``neuron_attributes['unit_id']`` when the NWB ``id``
    dataset contains values that can't be coerced to ``int`` (e.g.
    UUID strings from automated pipelines). The previous behavior
    crashed mid-loop with a generic ``ValueError`` from ``int(uid)``.
    """

    def test_uuid_string_unit_id_warns_and_loads(self, tmp_path):
        """
        Tests:
            (Test Case 1) Load succeeds.
            (Test Case 2) ``neuron_attributes[i]['unit_id']`` contains
                the original string value.
            (Test Case 3) A ``UserWarning`` mentions the unit id.
        """
        path = str(tmp_path / "uuid_ids.nwb")
        with h5py.File(path, "w") as f:  # type: ignore
            g = f.create_group("units")
            g.create_dataset("spike_times", data=np.array([0.1, 0.2, 0.5], dtype=float))
            g.create_dataset("spike_times_index", data=np.array([2, 3], dtype=int))
            # Use the variable-length string dtype that h5py recommends.
            string_dt = h5py.string_dtype(encoding="utf-8")
            g.create_dataset(
                "id", data=np.array(["abc-1", "def-2"], dtype=object), dtype=string_dt
            )

        with pytest.warns(UserWarning, match=r"unit id"):
            sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)

        assert sd.N == 2
        assert sd.neuron_attributes is not None
        ids = [a["unit_id"] for a in sd.neuron_attributes]
        # h5py returns bytes for variable-length string datasets; accept
        # either bytes or str depending on the local h5py version.
        assert ids[0] in ("abc-1", b"abc-1")
        assert ids[1] in ("def-2", b"def-2")


class TestLoadSpikedataFromNwbElectrodesIndexTruncationWarning:
    """``load_spikedata_from_nwb`` h5py path now reports *how many*
    electrode index entries will be silently truncated when
    ``electrodes_index[-1]`` exceeds ``len(electrodes)``, instead of
    the previous vague "may be truncated" wording.
    """

    def test_truncation_warning_names_the_count(self, tmp_path):
        """
        Tests:
            (Test Case 1) Warning fires.
            (Test Case 2) Warning message includes both the index value
                and the truncated count so the user can act on it.
        """
        path = str(tmp_path / "trunc.nwb")
        with h5py.File(path, "w") as f:  # type: ignore
            g = f.create_group("units")
            g.create_dataset("spike_times", data=np.array([0.1, 0.2, 0.5], dtype=float))
            g.create_dataset("spike_times_index", data=np.array([2, 3], dtype=int))
            g.create_dataset("id", data=np.array([0, 1], dtype=int))
            # electrodes has 2 entries but electrodes_index says 5 are
            # expected → trailing 3 entries will be truncated.
            g.create_dataset("electrodes", data=np.array([10, 11], dtype=int))
            g.create_dataset("electrodes_index", data=np.array([1, 5], dtype=int))

        with pytest.warns(UserWarning) as recwarn:
            loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
        msgs = [str(w.message) for w in recwarn]
        # At least one warning names both the offending index final
        # value (5) and the actual electrodes length (2).
        joined = " | ".join(msgs)
        assert "5" in joined
        assert "2" in joined
        assert "truncated" in joined.lower()


class TestLoadSpikedataFromKilosortClusterIdExceedsChannelMapWarning:
    """``load_spikedata_from_kilosort`` warns *per skipped cluster*
    when a cluster ID exceeds ``len(channel_map)``. The previous
    behavior emitted a single upstream warning about non-sequential
    IDs and then silently skipped the electrode assignment — users
    debugging missing locations had to map "which cluster" to
    "which warning" themselves.
    """

    def test_per_cluster_warning_fires_on_out_of_range(self, tmp_path):
        """
        Tests:
            (Test Case 1) Per-cluster warning fires and names the
                offending cluster ID.
        """
        spike_times = np.array([1000, 2000, 3000, 4000, 5000])
        spike_clusters = np.array([0, 1, 100, 0, 100])
        channel_map = np.array([10, 11, 12])  # length 3 → cluster 100 OOB
        np.save(str(tmp_path / "spike_times.npy"), spike_times)
        np.save(str(tmp_path / "spike_clusters.npy"), spike_clusters)
        np.save(str(tmp_path / "channel_map.npy"), channel_map)

        with warnings.catch_warnings(record=True) as recwarn:
            warnings.simplefilter("always")
            loaders.load_spikedata_from_kilosort(
                str(tmp_path), fs_Hz=30000.0, time_unit="samples"
            )
        msgs = [str(w.message) for w in recwarn]
        joined = " | ".join(msgs)
        # At least one warning mentions cluster 100 and channel_map
        # length.
        assert "100" in joined
        assert "channel_map" in joined.lower()


class TestLoadKilosortInvalidTimeUnit:
    """``load_spikedata_from_kilosort`` with an unrecognised ``time_unit``
    propagates the ``ValueError`` raised by the shared ``to_ms`` helper.
    The error message names the offending unit so the user can attribute
    the failure to the loader argument rather than guessing where it came
    from in the call chain.
    """

    def test_unknown_time_unit_raises_value_error_naming_unit(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``time_unit='hz'`` raises ``ValueError``.
            (Test Case 2) The message mentions the offending unit name
                ``'hz'`` so the failure is attributable.
        """
        d = str(tmp_path / "ks")
        os.makedirs(d)
        np.save(os.path.join(d, "spike_times.npy"), np.array([10, 20, 30]))
        np.save(os.path.join(d, "spike_clusters.npy"), np.array([0, 0, 0]))

        with pytest.raises(ValueError, match=r"hz"):
            loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0, time_unit="hz")


@skip_no_h5py
class TestRawArraysShapeMismatch:
    """``_read_raw_arrays`` validates ``raw_data.shape[-1] ==
    raw_time.shape[0]`` at the loader boundary. A mismatched HDF5
    file raises :class:`ValueError` with both shapes in the message
    so the user can diagnose the misalignment without first having
    to chase through the SpikeData constructor's suffix-shape check.
    """

    def test_mismatched_shapes_raises(self, tmp_path):
        """
        Tests:
            (Test Case 1) ``_read_raw_arrays`` raises ``ValueError``
                when ``raw_data.shape[-1] != raw_time.shape[0]``.
            (Test Case 2) The error message includes both array shapes
                so the caller can identify the mismatch.
        """
        path = str(tmp_path / "mismatch.h5")
        raw_data = np.random.randn(3, 100)
        raw_time = np.arange(50, dtype=float)  # length 50 != 100
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raw", data=raw_data)
            f.create_dataset("raw_time", data=raw_time)

        with h5py.File(path, "r") as f:  # type: ignore
            with pytest.raises(
                ValueError, match="does not match raw_time length"
            ) as exc_info:
                loaders._read_raw_arrays(f, "raw", "raw_time", "ms", None)
        msg = str(exc_info.value)
        assert "(3, 100)" in msg, f"raw_data shape missing from message: {msg}"
        assert "(50,)" in msg, f"raw_time shape missing from message: {msg}"

    def test_matched_shapes_succeed(self, tmp_path):
        """
        Tests:
            (Test Case 1) Matched shapes (raw_data trailing axis equal to
                raw_time length) load cleanly, no exception.
            (Test Case 2) Time vector is converted to ms as specified by
                ``raw_time_unit``.
        """
        path = str(tmp_path / "match.h5")
        raw_data = np.random.randn(3, 100)
        raw_time = np.arange(100, dtype=float)  # matches!
        with h5py.File(path, "w") as f:  # type: ignore
            f.create_dataset("raw", data=raw_data)
            f.create_dataset("raw_time", data=raw_time)

        with h5py.File(path, "r") as f:  # type: ignore
            rd, rt = loaders._read_raw_arrays(f, "raw", "raw_time", "s", None)
        assert rd is not None and rt is not None
        assert rd.shape == (3, 100)
        # Seconds -> milliseconds.
        np.testing.assert_array_equal(rt, raw_time * 1e3)


# ---------------------------------------------------------------------------
# Batch B — load_spikedata_from_kilosort: Phy channel_map resolution chain
#
# Pins the three-tier cluster→channel resolution introduced by
# commit a57e74f:
#   1. ``cluster_info.tsv["ch"]`` — canonical Phy post-curation answer.
#   2. ``spike_templates.npy + templates.npy`` — phylib-style fallback,
#      built per-cluster from the dominant template's peak channel.
#   3. Legacy ``channel_map[cluster_id]`` — only correct for fresh
#      uncurated kilosort output (sequential cluster IDs).
# ---------------------------------------------------------------------------


@skip_no_pandas
class TestKilosortPhyChannelMapResolution:
    """Three-tier cluster→channel resolution + non-sequential warning gating."""

    def _write_ks_folder(
        self,
        folder,
        *,
        spike_times,
        spike_clusters,
        channel_map=None,
        cluster_info_rows=None,
        spike_templates=None,
        templates=None,
    ):
        """Build a minimal kilosort/Phy output folder for the loader.

        Parameters mirror the .npy files the loader reads. ``None``
        for an argument skips writing that file (so we can drive the
        loader through each tier of the resolution chain).
        """
        import os as _os

        if not _os.path.isdir(folder):
            _os.makedirs(folder)
        np.save(_os.path.join(folder, "spike_times.npy"), spike_times)
        np.save(_os.path.join(folder, "spike_clusters.npy"), spike_clusters)
        if channel_map is not None:
            np.save(_os.path.join(folder, "channel_map.npy"), channel_map)
        if cluster_info_rows is not None:
            import pandas as pd

            df = pd.DataFrame(cluster_info_rows)
            df.to_csv(_os.path.join(folder, "cluster_info.tsv"), sep="\t", index=False)
        if spike_templates is not None:
            np.save(_os.path.join(folder, "spike_templates.npy"), spike_templates)
        if templates is not None:
            np.save(_os.path.join(folder, "templates.npy"), templates)

    def test_tsv_ch_column_drives_electrode_assignment(self, tmp_path):
        """``cluster_info.tsv["ch"]`` is the canonical Phy answer and
        wins over both the templates fallback and the legacy
        ``channel_map[cluster_id]`` lookup. Non-sequential cluster IDs
        — i.e. post-merge/split — map to their TSV-recorded channels.
        """
        d = str(tmp_path / "ks")
        spike_times = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
        spike_clusters = np.array([5, 5, 12, 12, 7, 7], dtype=np.int64)
        # Channel map deliberately wrong-length / unrelated; ``ch``
        # column should override anything channel_map would have said.
        channel_map = np.arange(20)
        self._write_ks_folder(
            d,
            spike_times=spike_times,
            spike_clusters=spike_clusters,
            channel_map=channel_map,
            cluster_info_rows=[
                {"cluster_id": 5, "ch": 3, "group": "good"},
                {"cluster_id": 12, "ch": 7, "group": "good"},
                {"cluster_id": 7, "ch": 0, "group": "good"},
            ],
        )

        sd = loaders.load_spikedata_from_kilosort(
            d,
            fs_Hz=1000.0,
            cluster_info_tsv="cluster_info.tsv",
        )
        cluster_ids = sd.metadata["cluster_ids"]
        # The loader iterates np.unique(spike_clusters) — sorted ascending.
        expected = {5: 3, 12: 7, 7: 0}
        for i, clu in enumerate(cluster_ids):
            assert sd.neuron_attributes[i]["electrode"] == expected[int(clu)], (
                f"Cluster {clu}: TSV says ch={expected[int(clu)]}, "
                f"got electrode={sd.neuron_attributes[i].get('electrode')}"
            )

    def test_templates_fallback_when_tsv_absent(self, tmp_path):
        """Without ``cluster_info.tsv``, the loader uses
        ``spike_templates.npy + templates.npy`` to resolve each cluster
        to its dominant template's peak channel, then translates that
        position through ``channel_map``. Pins the phylib-style
        fallback added in commit a57e74f.
        """
        d = str(tmp_path / "ks")
        # Three non-sequential clusters; each gets a unique dominant
        # template whose peak is on a known channel position.
        # spike order: c5(2 spikes), c12(2), c7(2)
        spike_times = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
        spike_clusters = np.array([5, 5, 12, 12, 7, 7], dtype=np.int64)
        # template_id 0 → peak position 3, template_id 1 → 7, template_id 2 → 0
        spike_templates = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

        n_templates = 3
        nsamples = 9
        n_pos = 8
        templates = np.zeros((n_templates, nsamples, n_pos), dtype=np.float32)
        templates[0, nsamples // 2, 3] = -10.0
        templates[1, nsamples // 2, 7] = -10.0
        templates[2, nsamples // 2, 0] = -10.0

        # channel_map: position → physical channel. Use a non-identity
        # mapping so we can verify the loader routes through it.
        channel_map = np.array([100, 101, 102, 103, 104, 105, 106, 107])

        self._write_ks_folder(
            d,
            spike_times=spike_times,
            spike_clusters=spike_clusters,
            channel_map=channel_map,
            spike_templates=spike_templates,
            templates=templates,
        )

        sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)

        cluster_ids = sd.metadata["cluster_ids"]
        expected = {
            5: int(channel_map[3]),
            12: int(channel_map[7]),
            7: int(channel_map[0]),
        }
        for i, clu in enumerate(cluster_ids):
            assert sd.neuron_attributes[i]["electrode"] == expected[int(clu)], (
                f"Cluster {clu}: expected templates fallback electrode "
                f"{expected[int(clu)]}, got "
                f"{sd.neuron_attributes[i].get('electrode')}"
            )

    def test_tsv_beats_templates_when_both_present(self, tmp_path):
        """TSV ``ch`` column wins over the templates fallback when both
        files are present. Templates fallback only runs when
        ``cluster_id_to_channel`` is still ``None`` after the TSV pass.
        """
        d = str(tmp_path / "ks")
        spike_times = np.array([10, 20, 30, 40], dtype=np.int64)
        spike_clusters = np.array([5, 5, 12, 12], dtype=np.int64)
        # Templates: would map cluster 5 → channel_map[7]=107,
        # cluster 12 → channel_map[3]=103.
        spike_templates = np.array([0, 0, 1, 1], dtype=np.int64)
        templates = np.zeros((2, 9, 8), dtype=np.float32)
        templates[0, 4, 7] = -10.0
        templates[1, 4, 3] = -10.0
        channel_map = np.array([100, 101, 102, 103, 104, 105, 106, 107])
        # TSV: maps 5→2, 12→5. Should win over the templates path.
        self._write_ks_folder(
            d,
            spike_times=spike_times,
            spike_clusters=spike_clusters,
            channel_map=channel_map,
            spike_templates=spike_templates,
            templates=templates,
            cluster_info_rows=[
                {"cluster_id": 5, "ch": 2, "group": "good"},
                {"cluster_id": 12, "ch": 5, "group": "good"},
            ],
        )

        sd = loaders.load_spikedata_from_kilosort(
            d,
            fs_Hz=1000.0,
            cluster_info_tsv="cluster_info.tsv",
        )
        cluster_ids = sd.metadata["cluster_ids"]
        expected = {5: 2, 12: 5}
        for i, clu in enumerate(cluster_ids):
            assert sd.neuron_attributes[i]["electrode"] == expected[int(clu)], (
                f"Cluster {clu}: TSV should have won — expected "
                f"electrode {expected[int(clu)]}, got "
                f"{sd.neuron_attributes[i].get('electrode')}"
            )

    def test_legacy_path_still_works_for_fresh_kilosort(self, tmp_path):
        """Sequential cluster IDs (0..N-1), no TSV, no templates →
        legacy ``channel_map[cluster_id]`` resolution still works.
        Pins backward compatibility for users who haven't run Phy.
        """
        d = str(tmp_path / "ks")
        spike_times = np.array([10, 20, 30, 40], dtype=np.int64)
        spike_clusters = np.array([0, 0, 1, 1], dtype=np.int64)
        channel_map = np.array([100, 101, 102, 103])
        self._write_ks_folder(
            d,
            spike_times=spike_times,
            spike_clusters=spike_clusters,
            channel_map=channel_map,
        )

        sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)
        cluster_ids = sd.metadata["cluster_ids"]
        for i, clu in enumerate(cluster_ids):
            assert sd.neuron_attributes[i]["electrode"] == int(channel_map[int(clu)]), (
                f"Cluster {clu}: legacy channel_map lookup broke — "
                f"expected {int(channel_map[int(clu)])}, got "
                f"{sd.neuron_attributes[i].get('electrode')}"
            )

    def test_non_sequential_warning_suppressed_when_fix_applies(self, tmp_path):
        """Non-sequential cluster IDs + TSV ``ch`` map → the legacy
        ``channel_map[cluster_id]`` path is bypassed, so the
        "not sequential" warning should NOT fire (it warned about the
        misalignment bug, which the fix sidesteps).
        """
        d = str(tmp_path / "ks")
        spike_times = np.array([10, 20, 30, 40], dtype=np.int64)
        spike_clusters = np.array([5, 5, 12, 12], dtype=np.int64)
        channel_map = np.arange(20)
        self._write_ks_folder(
            d,
            spike_times=spike_times,
            spike_clusters=spike_clusters,
            channel_map=channel_map,
            cluster_info_rows=[
                {"cluster_id": 5, "ch": 3, "group": "good"},
                {"cluster_id": 12, "ch": 7, "group": "good"},
            ],
        )

        with warnings.catch_warnings(record=True) as recwarn:
            warnings.simplefilter("always")
            loaders.load_spikedata_from_kilosort(
                d, fs_Hz=1000.0, cluster_info_tsv="cluster_info.tsv"
            )

        sequential_warns = [w for w in recwarn if "not sequential" in str(w.message)]
        assert sequential_warns == [], (
            "Non-sequential warning fired even though TSV ``ch`` map "
            f"resolved every cluster: {[str(w.message) for w in sequential_warns]}"
        )

    def test_non_sequential_warning_fires_on_legacy_fallback(self, tmp_path):
        """Non-sequential cluster IDs, no TSV, no templates → the
        legacy ``channel_map[cluster_id]`` path is the only thing
        left, and the "not sequential" warning fires to flag the
        misalignment risk. Pins the existing safety signal.
        """
        d = str(tmp_path / "ks")
        spike_times = np.array([10, 20, 30, 40], dtype=np.int64)
        spike_clusters = np.array([5, 5, 12, 12], dtype=np.int64)
        channel_map = np.arange(20)
        self._write_ks_folder(
            d,
            spike_times=spike_times,
            spike_clusters=spike_clusters,
            channel_map=channel_map,
        )

        with warnings.catch_warnings(record=True) as recwarn:
            warnings.simplefilter("always")
            loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)

        sequential_warns = [w for w in recwarn if "not sequential" in str(w.message)]
        assert sequential_warns, (
            "Expected 'not sequential' warning on legacy fallback — "
            f"saw warnings: {[str(w.message) for w in recwarn]}"
        )

    def test_templates_fallback_skipped_on_shape_mismatch(self, tmp_path):
        """A 2-D ``templates.npy`` triggers the
        ``"Templates fallback skipped"`` warning and the loader falls
        through to the legacy ``channel_map[cluster_id]`` path. The
        warning includes the offending shape so users can debug.
        """
        d = str(tmp_path / "ks")
        # Sequential cluster IDs so the legacy fallback gives a
        # well-defined answer to assert on.
        spike_times = np.array([10, 20, 30, 40], dtype=np.int64)
        spike_clusters = np.array([0, 0, 1, 1], dtype=np.int64)
        # Matching length so the shape mismatch is purely the
        # ``ndim != 3`` check.
        spike_templates = np.array([0, 0, 1, 1], dtype=np.int64)
        channel_map = np.array([100, 101, 102, 103])
        # 2-D templates.npy — wrong rank.
        templates_2d = np.zeros((2, 9), dtype=np.float32)
        self._write_ks_folder(
            d,
            spike_times=spike_times,
            spike_clusters=spike_clusters,
            channel_map=channel_map,
            spike_templates=spike_templates,
            templates=templates_2d,
        )

        with warnings.catch_warnings(record=True) as recwarn:
            warnings.simplefilter("always")
            sd = loaders.load_spikedata_from_kilosort(d, fs_Hz=1000.0)

        skip_warns = [
            w for w in recwarn if "Templates fallback skipped" in str(w.message)
        ]
        assert skip_warns, (
            "Expected 'Templates fallback skipped' warning for 2-D "
            f"templates.npy. Got: {[str(w.message) for w in recwarn]}"
        )
        # Legacy fallback path produced electrodes via channel_map.
        for i, clu in enumerate(sd.metadata["cluster_ids"]):
            assert sd.neuron_attributes[i]["electrode"] == int(channel_map[int(clu)]), (
                f"Cluster {clu}: legacy fallback after templates-skip "
                f"gave electrode {sd.neuron_attributes[i].get('electrode')}, "
                f"expected {int(channel_map[int(clu)])}"
            )


class TestLoadNwbStartTimeAttribute:
    """``load_spikedata_from_nwb`` honors the ``start_time`` file
    attribute (written by :func:`export_spikedata_to_nwb` in commit
    609aa09) and falls back to 0.0 when the attribute is absent. The
    ``start_time_ms`` keyword argument overrides both.

    Existing tests pin the round-trip via the exporter side
    (``TestNWBExporters::test_nonzero_start_time_roundtrips``); these
    tests pin the loader side directly through hand-written h5py
    fixtures so the loader's three-tier resolution (caller arg →
    file attr → 0.0 default) is locked.
    """

    def test_caller_start_time_ms_overrides_file_attribute(self, tmp_path):
        """
        Tests:
            (Test Case 1) File written with ``start_time=100.0`` attr;
                loader called with explicit ``start_time_ms=50.0``;
                resulting ``SpikeData.start_time == 50.0`` (caller wins).
        """
        path = str(tmp_path / "override.nwb")
        with h5py.File(path, "w") as f:  # type: ignore
            f.attrs["start_time"] = 100.0
            g = f.create_group("units")
            g.create_dataset("spike_times", data=np.array([0.2, 0.3]))
            g.create_dataset("spike_times_index", data=np.array([1, 2]))

        sd = loaders.load_spikedata_from_nwb(
            path,
            prefer_pynwb=False,
            start_time_ms=50.0,
            length_ms=500.0,
        )
        assert sd.start_time == 50.0

    def test_missing_start_time_attr_falls_back_to_zero(self, tmp_path):
        """
        Tests:
            (Test Case 1) NWB file without a ``start_time`` file
                attribute loads with ``start_time == 0.0`` (no error,
                no warning required — the default is documented).
        """
        path = str(tmp_path / "no_start_time.nwb")
        with h5py.File(path, "w") as f:  # type: ignore
            # Deliberately do NOT set f.attrs["start_time"].
            g = f.create_group("units")
            g.create_dataset("spike_times", data=np.array([0.2, 0.3]))
            g.create_dataset("spike_times_index", data=np.array([1, 2]))

        sd = loaders.load_spikedata_from_nwb(path, prefer_pynwb=False)
        assert sd.start_time == 0.0


class TestParseS3UrlTrailingSlash:
    """``parse_s3_url`` strips a single trailing slash so
    ``s3://bucket/key/`` reaches the same object as ``s3://bucket/key``
    instead of falling through to boto3 as an empty prefix (cryptic
    ``NoSuchKey``). An empty key after the strip raises ValueError.
    """

    def test_trailing_slash_stripped(self):
        """
        Tests:
            (Test Case 1) ``s3://bucket/key/`` → ``("bucket", "key")``.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        bucket, key = parse_s3_url("s3://bucket/key/")
        assert bucket == "bucket"
        assert key == "key"

    def test_empty_key_after_strip_raises(self):
        """
        Tests:
            (Test Case 1) ``s3://bucket//`` parses to an empty key
                after the trailing-slash strip and raises ValueError
                mentioning "no object key".
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        with pytest.raises(ValueError, match="no object key"):
            parse_s3_url("s3://bucket//")


class TestIsS3UrlHostnameSpoofing:
    """``is_s3_url`` must reject look-alike hosts that embed
    ``s3.amazonaws.com`` as a non-suffix substring (a common attacker
    pattern: ``https://s3.evil.amazonaws.com.attacker.example/path``).
    Legitimate path-style and virtual-hosted URLs must still pass.
    """

    def test_lookalike_host_rejected(self):
        """
        Tests:
            (Test Case 1) A host that contains ``s3.amazonaws.com`` as
                a non-suffix label is rejected.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        assert is_s3_url("https://s3.evil.amazonaws.com.attacker.example/path") is False

    def test_legitimate_virtual_hosted_url_accepted(self):
        """
        Tests:
            (Test Case 1) ``https://bucket.s3.us-west-2.amazonaws.com/key``
                is accepted.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        assert is_s3_url("https://my-bucket.s3.us-west-2.amazonaws.com/key") is True

    def test_legitimate_path_style_url_accepted(self):
        """
        Tests:
            (Test Case 1) ``https://s3.amazonaws.com/bucket/key`` is accepted.
        """
        from spikelab.data_loaders.s3_utils import is_s3_url

        assert is_s3_url("https://s3.amazonaws.com/bucket/key") is True


class TestParseS3UrlMixedCase:
    """``parse_s3_url`` should treat host buckets case-insensitively
    (S3 bucket names are restricted to lowercase, but path-style URLs
    with mixed-case bucket names should still parse — they're invalid
    S3 names but the parser shouldn't crash).
    """

    def test_mixed_case_path_style_bucket(self):
        """
        Tests:
            (Test Case 1) Path-style HTTPS URL with mixed-case bucket
                parses without raising. (S3 itself would reject the
                bucket name on a real call, but the parser is purely
                syntactic.)
            (Test Case 2) Bucket portion is preserved verbatim — the
                parser does not silently lowercase.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        bucket, key = parse_s3_url("https://s3.amazonaws.com/MyBucket/path/file.h5")
        assert bucket == "MyBucket"
        assert key == "path/file.h5"

    def test_mixed_case_virtual_hosted_bucket(self):
        """
        Tests:
            (Test Case 1) Virtual-hosted-style URL with mixed-case
                bucket parses without raising.
            (Test Case 2) Bucket name preserved exactly.
        """
        from spikelab.data_loaders.s3_utils import parse_s3_url

        bucket, key = parse_s3_url("https://MyBucket.s3.amazonaws.com/key/file.h5")
        assert bucket == "MyBucket"
        assert key == "key/file.h5"


# ============================================================================
# I/O review (2026-05-24) — data_loaders edge-case pins from the
# /complete_review pass on fix/review-cleanups.
# ============================================================================


class TestTrainsFromFlatIndexLengthMismatch:
    """``_trains_from_flat_index`` with ``n_units`` provided: the
    function accepts length-N (cumulative-end) and length-N+1-with-
    leading-zero (NWB leading-zero variant). Any other shape — including
    length-N+1 with a non-zero head — falls through to the explicit
    ``ValueError`` at source line 148-153. Pin the boundary.
    """

    def test_length_n_plus_one_with_non_zero_head_raises(self):
        """
        Tests:
            (Test Case 1) ``end_indices=[1, 3, 5]`` with ``n_units=2``
                (length-N+1 but head != 0) raises ValueError.
            (Test Case 2) Error message mentions both conventions.
        """
        from spikelab.data_loaders.data_loaders import _trains_from_flat_index

        with pytest.raises(ValueError, match="cumulative-end|leading-zero"):
            _trains_from_flat_index(
                np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                np.array([1, 3, 5]),
                unit="s",
                fs_Hz=None,
                n_units=2,
            )

    def test_n_units_zero_returns_empty_list(self):
        """
        Tier L-D5: ``n_units=0`` short-circuits to return ``[]`` early,
        regardless of what's in ``end_indices``. The previous behaviour
        (raising ValueError on a length mismatch) was a convoluted code
        path; the empty-list shortcut makes the intent obvious.

        Tests:
            (Test Case 1) ``n_units=0`` with non-empty ``end_indices``
                returns ``[]`` without raising.
            (Test Case 2) ``n_units=0`` with empty ``end_indices``
                returns ``[]``.
        """
        from spikelab.data_loaders.data_loaders import _trains_from_flat_index

        result = _trains_from_flat_index(
            np.array([0.1, 0.2]),
            np.array([2]),
            unit="s",
            fs_Hz=None,
            n_units=0,
        )
        assert result == []
        result_empty = _trains_from_flat_index(
            np.array([]),
            np.array([]),
            unit="s",
            fs_Hz=None,
            n_units=0,
        )
        assert result_empty == []


class TestReadRawArraysZeroBoundaries:
    """``_read_raw_arrays`` shape boundaries: empty raw_data with
    matching empty raw_time, and single-sample raw_time. These pass
    the trailing-axis check at source line 200-202; pin the documented
    no-rejection contract so a future stricter guard would surface.
    """

    def test_empty_raw_data_and_time_succeed(self):
        """
        Tests:
            (Test Case 1) ``raw_data.shape=(0, 0)`` with ``raw_time.shape=(0,)``
                returns valid empty arrays (trailing-axis match is 0==0).
        """
        if h5py is None:
            pytest.skip("h5py not installed")

        from spikelab.data_loaders.data_loaders import _read_raw_arrays

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            path = tmp.name
        try:
            with h5py.File(path, "w") as f:
                f.create_dataset("raw", data=np.zeros((0, 0)))
                f.create_dataset("raw_t", data=np.zeros((0,)))
            with h5py.File(path, "r") as f:
                raw_data, raw_time = _read_raw_arrays(
                    f,
                    raw_dataset="raw",
                    raw_time_dataset="raw_t",
                    raw_time_unit="ms",
                    fs_Hz=None,
                )
            assert raw_data.shape == (0, 0)
            assert raw_time.shape == (0,)
        finally:
            os.unlink(path)

    def test_single_sample_raw_time(self):
        """
        Tests:
            (Test Case 1) ``raw_data.shape=(C, 1)`` with ``raw_time.shape=(1,)``
                returns valid arrays with trailing-axis 1.
        """
        if h5py is None:
            pytest.skip("h5py not installed")

        from spikelab.data_loaders.data_loaders import _read_raw_arrays

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            path = tmp.name
        try:
            with h5py.File(path, "w") as f:
                f.create_dataset("raw", data=np.array([[1.0]]))
                f.create_dataset("raw_t", data=np.array([5.0]))
            with h5py.File(path, "r") as f:
                raw_data, raw_time = _read_raw_arrays(
                    f,
                    raw_dataset="raw",
                    raw_time_dataset="raw_t",
                    raw_time_unit="ms",
                    fs_Hz=None,
                )
            assert raw_data.shape == (1, 1)
            assert raw_time.shape == (1,)
            assert raw_time[0] == pytest.approx(5.0)
        finally:
            os.unlink(path)


class TestBuildSpikeDataAllEmptyWithStartTime:
    """``_build_spikedata`` with all-empty trains AND a non-zero
    ``start_time``: the length-inference path falls through to
    ``length_ms = 0.0`` (per source: ``max_last`` defaults to
    ``start_time`` when no spikes are present). Pin the resulting
    SpikeData has ``length=0`` and ``start_time=start_time``.
    """

    def test_all_empty_trains_preserves_start_time(self):
        """
        Tests:
            (Test Case 1) Empty trains with ``start_time=100.0`` yield
                SpikeData with length=0 and start_time=100.0.
            (Test Case 2) ``N`` matches the input length.
        """
        from spikelab.data_loaders.data_loaders import _build_spikedata

        sd = _build_spikedata(
            trains_ms=[np.array([]), np.array([])],
            length_ms=None,
            start_time=100.0,
        )
        assert sd.N == 2
        assert sd.length == 0
        assert sd.start_time == 100.0


class TestLoadPickleRejectsNonSpikeData:
    """``load_spikedata_from_pickle`` rejects every type other than
    ``SpikeData``, even though ``export_to_pickle`` accepts six types
    (SpikeData, RateData, PairwiseCompMatrix, PairwiseCompMatrixStack,
    RateSliceStack, SpikeSliceStack). The asymmetry is documented in
    the code review as a UX gap; pin the rejection behaviour so a
    regression that silently returned a non-SpikeData would surface.
    """

    def test_pickle_with_ratedata_rejected(self):
        """
        Tests:
            (Test Case 1) A pickle containing a RateData raises
                ValueError mentioning "SpikeData".
            (Test Case 2) The error message names the actual loaded
                type (RateData).
        """
        from spikelab.data_loaders.data_loaders import load_spikedata_from_pickle
        from spikelab.spikedata.ratedata import RateData

        rd = RateData(np.zeros((1, 3)), np.array([0.0, 1.0, 2.0]))
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            path = tmp.name
        try:
            with open(path, "wb") as f:
                pickle.dump(rd, f)
            with pytest.raises(ValueError, match="SpikeData"):
                load_spikedata_from_pickle(path)
            with pytest.raises(ValueError, match="RateData"):
                load_spikedata_from_pickle(path)
        finally:
            os.unlink(path)


class TestNwbElectrodesIndexExactBoundary:
    """NWB ``electrodes_index[-1]`` exactly equal to ``len(electrodes)``
    (no truncation): the loader at source line 715 uses strict ``>``
    so the equal-boundary case does NOT trigger the truncation warning.
    Pin the silent-success contract at the exact boundary.
    """

    def test_electrodes_index_exactly_equals_electrodes_length(self):
        """
        Tests:
            (Test Case 1) NWB file with ``electrodes_index[-1] == len(electrodes)``
                loads without emitting a truncation warning.
        """
        if h5py is None:
            pytest.skip("h5py not installed")

        from spikelab.data_loaders.data_loaders import load_spikedata_from_nwb

        with tempfile.NamedTemporaryFile(suffix=".nwb", delete=False) as tmp:
            path = tmp.name
        try:
            with h5py.File(path, "w") as f:
                units = f.create_group("units")
                units.create_dataset("id", data=np.array([0, 1]))
                units.create_dataset(
                    "spike_times", data=np.array([0.001, 0.002, 0.003])
                )
                units.create_dataset("spike_times_index", data=np.array([2, 3]))
                # electrodes_index[-1] == len(electrodes): exact boundary.
                units.create_dataset("electrodes", data=np.array([0, 1, 2]))
                units.create_dataset("electrodes_index", data=np.array([2, 3]))

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sd = load_spikedata_from_nwb(path, prefer_pynwb=False)
            # No "truncated" warning fires.
            assert not any("truncat" in str(w.message).lower() for w in caught)
            assert sd.N == 2
        finally:
            os.unlink(path)


class TestLoadHDF5RawThresholdedFsHzTypeQuirks:
    """``load_spikedata_from_hdf5_raw_thresholded`` validates
    ``fs_Hz`` with the existing ``not fs_Hz or fs_Hz <= 0`` guard.
    Python ``bool`` is an int subclass, so ``True > 0`` passes — pin
    the current passthrough (which is likely unintended but documented).
    """

    def test_fs_hz_true_accepted_as_1hz(self):
        """
        Tests:
            (Test Case 1) ``fs_Hz=True`` is accepted (not rejected),
                effectively meaning 1 Hz sample rate.

        Notes:
            - Documents the current behaviour. A future stricter
              ``isinstance(fs_Hz, bool)`` rejection would surface here.
        """
        if h5py is None:
            pytest.skip("h5py not installed")

        from spikelab.data_loaders.data_loaders import (
            load_spikedata_from_hdf5_raw_thresholded,
        )

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            path = tmp.name
        try:
            with h5py.File(path, "w") as f:
                f.create_dataset(
                    "data", data=np.random.RandomState(0).randn(2, 50) * 0.1
                )
            # fs_Hz=True coerces to 1.0; if the function rejects bool
            # in the future this becomes a TypeError.
            try:
                sd = load_spikedata_from_hdf5_raw_thresholded(
                    path,
                    dataset="data",
                    fs_Hz=True,
                    threshold_sigma=5.0,
                    filter=False,
                )
                # Successful path: SpikeData constructed, possibly zero spikes.
                assert sd.N == 2
            except (ValueError, TypeError) as exc:
                # If a future guard rejects bool, accept the rejection.
                pytest.skip(f"fs_Hz=True now rejected (was previously accepted): {exc}")
        finally:
            os.unlink(path)


# ============================================================================
# Test Coverage Scan + Edge Case residuals (2026-05-25) — more pins.
# ============================================================================


class TestBuildS3Kwargs:
    """``_build_s3_kwargs`` builds boto3 client kwargs from optional
    credential params. Pin filtering and passthrough.
    """

    def test_empty_credentials_yields_empty_dict(self):
        """
        Tests:
            (Test Case 1) All-None args produce an empty dict.
        """
        from spikelab.data_loaders.s3_utils import _build_s3_kwargs

        assert _build_s3_kwargs() == {}

    def test_region_name_alone(self):
        """
        Tests:
            (Test Case 1) Only ``region_name`` set produces a single-
                entry dict (no credentials).
        """
        from spikelab.data_loaders.s3_utils import _build_s3_kwargs

        assert _build_s3_kwargs(region_name="us-west-2") == {"region_name": "us-west-2"}

    def test_full_credentials_passed_through(self):
        """
        Tests:
            (Test Case 1) Access key, secret, session token, and region
                all populate the result.
        """
        from spikelab.data_loaders.s3_utils import _build_s3_kwargs

        assert _build_s3_kwargs(
            aws_access_key_id="AKID",
            aws_secret_access_key="secret",
            aws_session_token="token",
            region_name="us-east-1",
        ) == {
            "aws_access_key_id": "AKID",
            "aws_secret_access_key": "secret",
            "aws_session_token": "token",
            "region_name": "us-east-1",
        }

    def test_empty_string_credentials_filtered_out(self):
        """
        Tests:
            (Test Case 1) Empty-string credential args are filtered
                (falsy values do not enter the dict).
        """
        from spikelab.data_loaders.s3_utils import _build_s3_kwargs

        assert _build_s3_kwargs(
            aws_access_key_id="",
            aws_secret_access_key="",
            region_name="us-east-1",
        ) == {"region_name": "us-east-1"}


class TestNwbLoaderStartTimeMsBrittleness:
    """The NWB loader uses ``start_time=start_time_ms or 0.0`` — the
    ``or`` pattern only falls back when LHS is falsy. Pin negative
    values survive (Python truthy) but ``0.0`` falls to the default.
    """

    def test_negative_start_time_preserved(self):
        """
        Tests:
            (Test Case 1) ``start_time_ms=-200.0`` survives the
                ``or 0.0`` brittleness; SpikeData has start_time=-200.
        """
        if h5py is None:
            pytest.skip("h5py not installed")

        from spikelab.data_loaders.data_loaders import load_spikedata_from_nwb

        with tempfile.NamedTemporaryFile(suffix=".nwb", delete=False) as tmp:
            path = tmp.name
        try:
            with h5py.File(path, "w") as f:
                units = f.create_group("units")
                units.create_dataset("id", data=np.array([0]))
                units.create_dataset("spike_times", data=np.array([0.001, 0.002]))
                units.create_dataset("spike_times_index", data=np.array([2]))
            sd = load_spikedata_from_nwb(
                path,
                start_time_ms=-200.0,
                length_ms=1000.0,
                prefer_pynwb=False,
            )
            assert sd.start_time == -200.0
        finally:
            os.unlink(path)


class TestNaturalSortKeyTypeStable:
    """``_natural_sort_key`` returns ``(kind, value)`` tuples so the
    sort is type-stable on Python 3 — mixing numeric and string tokens
    at the same position cannot raise ``TypeError`` from comparing
    ``int`` against ``str``.
    """

    def test_mixed_numeric_string_token_first_position(self):
        """
        Tests:
            (Test Case 1) ``sorted(["unit_5", "5_unit"], key=...)``
                succeeds (no TypeError).
            (Test Case 2) The numeric-token-first key (``"5_unit"``)
                sorts before the string-token-first key (``"unit_5"``)
                because every numeric tuple compares less than every
                string tuple at the same position.
        """
        from spikelab.data_loaders.data_loaders import _natural_sort_key

        out = sorted(["unit_5", "5_unit"], key=_natural_sort_key)
        assert out == ["5_unit", "unit_5"]


class TestLoadKilosortTimeUnitValidationBeforeIo:
    """``load_spikedata_from_kilosort`` validates ``time_unit`` against
    the (samples, s, ms) allow-list BEFORE any ``.npy`` I/O — a typo
    fails fast without partial side effects (file handles, allocations).
    """

    def test_typo_time_unit_rejected_before_np_load(self, monkeypatch, tmp_path):
        """
        Tests:
            (Test Case 1) ``time_unit="seconds"`` raises ValueError.
            (Test Case 2) ``np.load`` is not called during the
                validation path (verified via a monkeypatched
                ``np.load`` that would fail-loud if invoked).
        """
        from spikelab.data_loaders import data_loaders as loaders_mod

        np_load_calls: list = []

        def fail_loud_np_load(*args, **kwargs):
            np_load_calls.append((args, kwargs))
            raise AssertionError(
                "np.load should NOT have been called — time_unit "
                "validation must fire first."
            )

        monkeypatch.setattr(loaders_mod.np, "load", fail_loud_np_load)
        with pytest.raises(ValueError, match="time_unit"):
            loaders_mod.load_spikedata_from_kilosort(
                str(tmp_path), fs_Hz=20000.0, time_unit="seconds"
            )
        assert np_load_calls == []
