"""
Coverage gap tests for data_loaders/data_exporters.py.
"""

import pytest
import numpy as np
import os
import h5py
from unittest import mock
from data_loaders import data_exporters
from spikedata import SpikeData

def test_export_hdf5_errors(tmp_path):
    sd = SpikeData([np.array([1, 2])], length=10)
    h5_path = str(tmp_path / "test.h5")
    
    # 117: unknown style
    with pytest.raises(ValueError, match="Unknown style"):
        data_exporters.export_spikedata_to_hdf5(sd, h5_path, style="invalid") # type: ignore

    # 148: raster_bin_size_ms invalid
    with pytest.raises(ValueError, match="raster_bin_size_ms must be provided"):
        data_exporters.export_spikedata_to_hdf5(sd, h5_path, style="raster", raster_bin_size_ms=None)

def test_export_hdf5_raw_units(tmp_path):
    sd = SpikeData([np.array([1, 2])], length=10, 
                   raw_data=np.array([10, 20]), raw_time=np.array([0, 1]))
    h5_path = str(tmp_path / "test_raw.h5")
    
    # 133: raw_time_unit == "ms"
    data_exporters.export_spikedata_to_hdf5(sd, h5_path, style="ragged", 
                                            raw_dataset="rd", raw_time_dataset="rt", raw_time_unit="ms")
    with h5py.File(h5_path, "r") as f:
        assert f["rt"].shape == (2,)

    # 141: raw_time_unit == "samples"
    data_exporters.export_spikedata_to_hdf5(sd, h5_path, style="ragged", 
                                            raw_dataset="rd", raw_time_dataset="rt_s", 
                                            raw_time_unit="samples", fs_Hz=1000)
    with h5py.File(h5_path, "r") as f:
        assert f["rt_s"].dtype == int

    # 138: samples without fs_Hz
    with pytest.raises(ValueError, match="fs_Hz must be provided"):
         data_exporters.export_spikedata_to_hdf5(sd, h5_path, style="ragged", 
                                            raw_dataset="rd", raw_time_dataset="rt_e", 
                                            raw_time_unit="samples", fs_Hz=0)

    # 143: invalid raw_time_unit
    with pytest.raises(ValueError, match="raw_time_unit must be one of"):
         data_exporters.export_spikedata_to_hdf5(sd, h5_path, style="ragged", 
                                            raw_dataset="rd", raw_time_dataset="rt_e", 
                                            raw_time_unit="invalid") # type: ignore

def test_export_kilosort_errors(tmp_path):
    sd = SpikeData([np.array([1, 2])], length=10)
    folder = str(tmp_path / "ks_err")
    
    # 277: fs_Hz <= 0
    with pytest.raises(ValueError, match="positive fs_Hz"):
        data_exporters.export_spikedata_to_kilosort(sd, folder, fs_Hz=0)
        
    # 293: cluster_ids length mismatch
    with pytest.raises(ValueError, match="cluster_ids length must match"):
        data_exporters.export_spikedata_to_kilosort(sd, folder, fs_Hz=1000, cluster_ids=[1, 2])

def test_export_kilosort_units(tmp_path):
    sd = SpikeData([np.array([1, 2])], length=10)
    folder = str(tmp_path / "ks_units")
    
    # 299: time_unit == "ms"
    data_exporters.export_spikedata_to_kilosort(sd, folder, fs_Hz=1000, time_unit="ms")
    
    # 301: time_unit == "s"
    data_exporters.export_spikedata_to_kilosort(sd, folder, fs_Hz=1000, time_unit="s")
    
    # 304: invalid time_unit
    with pytest.raises(ValueError, match="time_unit must be one of"):
        data_exporters.export_spikedata_to_kilosort(sd, folder, fs_Hz=1000, time_unit="invalid") # type: ignore
