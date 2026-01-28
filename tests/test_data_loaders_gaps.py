"""
Coverage gap tests for data_loaders/data_loaders.py.
"""

import pytest
import numpy as np
import h5py
import os
import tempfile
from unittest import mock
from data_loaders import data_loaders
from spikedata import SpikeData

def test_read_raw_arrays_logic():
    # Lines 77, 80, 85
    # Also 72->86 jump: raw_time_dataset is None
    rd, rt = data_loaders._read_raw_arrays({"raw": [1]}, "raw", None, "s", None)
    assert rd is not None
    assert rt is None

    # 77: raw_time_unit == "ms"
    mock_file = {}
    mock_file["time"] = np.array([1, 2, 3])
    raw_data, raw_time = data_loaders._read_raw_arrays(
        {"raw": [1, 2], "time": [1, 2, 3]}, 
        "raw", "time", "ms", None
    )
    np.testing.assert_array_equal(raw_time, [1, 2, 3])
    
    # 80: samples without fs_Hz
    with pytest.raises(ValueError, match="fs_Hz must be provided"):
        data_loaders._read_raw_arrays({"raw": [1], "time": [1]}, "raw", "time", "samples", None)
    
    # 85: invalid unit
    with pytest.raises(ValueError, match="raw_time_unit must be one of"):
        data_loaders._read_raw_arrays({"raw": [1], "time": [1]}, "raw", "time", "invalid", None)

def test_load_hdf5_errors(tmp_path):
    h5_path = str(tmp_path / "test.h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("raster", data=np.zeros((2, 10)))
    
    # 232: raster_bin_size_ms is None
    with pytest.raises(ValueError, match="raster_bin_size_ms is required"):
        data_loaders.load_spikedata_from_hdf5(h5_path, raster_dataset="raster")

    # 243: total_time <= 0
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("raster_empty", data=np.zeros((2, 0)))
    sd = data_loaders.load_spikedata_from_hdf5(h5_path, raster_dataset="raster_empty", raster_bin_size_ms=1.0)
    assert sd.length == 0.0

def test_load_nwb_pynwb_success(tmp_path):
    # 349-356: prefer_pynwb=True and pynwb works
    h5_path = str(tmp_path / "test_success.nwb")
    
    mock_pynwb = mock.MagicMock()
    mock_io = mock.MagicMock()
    mock_nwb = mock.MagicMock()
    
    class Row:
        def __init__(self, st):
            self.spike_times = st
    
    row = Row(np.array([1.0, 2.0]))
    mock_df = mock.MagicMock()
    mock_df.itertuples.return_value = [row]
    
    mock_units = mock.MagicMock()
    mock_units.to_dataframe.return_value = mock_df
    mock_nwb.units = mock_units
    
    mock_io.__enter__.return_value = mock_io
    mock_io.read.return_value = mock_nwb
    mock_pynwb.NWBHDF5IO.return_value = mock_io
    
    with mock.patch.dict("sys.modules", {"pynwb": mock_pynwb}):
        sd = data_loaders.load_spikedata_from_nwb(h5_path, prefer_pynwb=True)
        assert sd.N == 1
        assert len(sd.train[0]) == 2

def test_load_nwb_pynwb_missing(tmp_path):
    # 346-356: prefer_pynwb=True but pynwb fails/missing
    h5_path = str(tmp_path / "test.nwb")
    # Create a valid-ish NWB structure via h5py to hit the h5py fallback logic
    with h5py.File(h5_path, "w") as f:
        u = f.create_group("units")
        u.create_dataset("spike_times", data=np.array([1.0, 2.0]))
        u.create_dataset("spike_times_index", data=np.array([2]))

    # Mock pynwb as not installed
    with mock.patch.dict("sys.modules", {"pynwb": None}):
        with pytest.warns(UserWarning, match="Falling back to h5py"):
             sd = data_loaders.load_spikedata_from_nwb(h5_path, prefer_pynwb=True)
             assert sd.N == 1
             assert len(sd.train[0]) == 2

def test_load_nwb_h5py_errors(tmp_path):
    # 375: Could not find spike_times
    h5_path = str(tmp_path / "test.nwb")
    with h5py.File(h5_path, "w") as f:
        u = f.create_group("units")
        u.create_dataset("dummy", data=[1])
    
    with pytest.raises(ValueError, match="Could not find spike_times"):
        data_loaders.load_spikedata_from_nwb(h5_path, prefer_pynwb=False)

def test_load_spikeinterface_sorting_errors():
    # 420: fs <= 0
    mock_sorting = mock.MagicMock()
    mock_sorting.get_sampling_frequency.return_value = 0
    with pytest.raises(ValueError, match="positive sampling_frequency"):
        data_loaders.load_spikedata_from_spikeinterface(mock_sorting)

def test_load_kilosort_errors(tmp_path):
    os.makedirs(tmp_path / "ks")
    np.save(tmp_path / "ks/spike_times.npy", np.array([1, 2]))
    np.save(tmp_path / "ks/spike_clusters.npy", np.array([1])) # Mismatch
    
    # 474: length mismatch
    with pytest.raises(ValueError, match="mismatch"):
        data_loaders.load_spikedata_from_kilosort(str(tmp_path / "ks"), fs_Hz=30000)

def test_load_kilosort_tsv(tmp_path):
    # 479->514, 499-510, 518
    folder = str(tmp_path / "ks_tsv")
    os.makedirs(folder)
    np.save(os.path.join(folder, "spike_times.npy"), np.array([100, 200, 300]))
    np.save(os.path.join(folder, "spike_clusters.npy"), np.array([0, 1, 2]))
    
    # Create TSV
    tsv_content = "cluster_id\tgroup\n0\tgood\n1\tmua\n2\tnoise\n"
    with open(os.path.join(folder, "cluster_info.tsv"), "w") as f:
        f.write(tsv_content)
        
    # include_noise=False (default)
    sd = data_loaders.load_spikedata_from_kilosort(folder, fs_Hz=1000, cluster_info_tsv="cluster_info.tsv")
    assert sd.N == 2 # 0 and 1
    
    # include_noise=True
    sd2 = data_loaders.load_spikedata_from_kilosort(folder, fs_Hz=1000, cluster_info_tsv="cluster_info.tsv", include_noise=True)
    assert sd2.N == 3

def test_load_kilosort_tsv_edge_cases(tmp_path):
    # TSV missing (479->514)
    folder = str(tmp_path / "ks_no_tsv")
    os.makedirs(folder)
    np.save(os.path.join(folder, "spike_times.npy"), np.array([100]))
    np.save(os.path.join(folder, "spike_clusters.npy"), np.array([0]))
    
    sd = data_loaders.load_spikedata_from_kilosort(folder, fs_Hz=1000, cluster_info_tsv="missing.tsv")
    assert sd.N == 1

    # TSV parsing error (509-510)
    with open(os.path.join(folder, "bad.tsv"), "w") as f:
        f.write("id\tgroup\n0\tgood\n") # This is fine, but let's mock pd.read_csv to fail
    
    with mock.patch("pandas.read_csv", side_effect=Exception("parse error")):
        with pytest.warns(UserWarning, match="Failed parsing cluster info"):
            sd = data_loaders.load_spikedata_from_kilosort(folder, fs_Hz=1000, cluster_info_tsv="bad.tsv")
            assert sd.N == 1

def test_load_spikeinterface_recording_errors():
    # 562, 566, 575
    mock_rec = mock.MagicMock()
    # 562: use getattr instead of hasattr if we want
    del mock_rec.get_sampling_frequency
    mock_rec.sampling_frequency = 0
    with pytest.raises(ValueError, match="positive sampling_frequency"):
        data_loaders.load_spikedata_from_spikeinterface_recording(mock_rec)
        
    mock_rec.sampling_frequency = 1000
    mock_rec.get_traces.return_value = np.zeros((10, 10, 10)) # 3D
    with pytest.raises(ValueError, match="2D array"):
        data_loaders.load_spikedata_from_spikeinterface_recording(mock_rec)

def test_load_spikeinterface_recording_get_sf_fallback():
    # 562: hasattr get_sampling_frequency is False, use sampling_frequency attr
    mock_rec = mock.MagicMock()
    del mock_rec.get_sampling_frequency
    mock_rec.sampling_frequency = 30000.0
    mock_rec.get_traces.return_value = np.zeros((10, 100))
    
    sd = data_loaders.load_spikedata_from_spikeinterface_recording(mock_rec)
    assert sd.N == 10
