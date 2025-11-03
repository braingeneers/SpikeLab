"""
Convenient imports for the data_loaders package.

Allows:
    from data_loaders import load_spikedata_from_hdf5, load_spikedata_from_nwb, ...
"""

from .data_loaders import (
    load_spikedata_from_hdf5,
    load_spikedata_from_hdf5_raw_thresholded,
    load_spikedata_from_nwb,
    load_spikedata_from_kilosort,
    load_spikedata_from_spikeinterface,
    load_spikedata_from_spikeinterface_recording,
    load_spikedata_from_acqm,
    download_s3_to_local,
)

__all__ = [
    "load_spikedata_from_hdf5",
    "load_spikedata_from_hdf5_raw_thresholded",
    "load_spikedata_from_nwb",
    "load_spikedata_from_kilosort",
    "load_spikedata_from_spikeinterface",
    "load_spikedata_from_spikeinterface_recording",
    "load_spikedata_from_acqm",
    "download_s3_to_local",
]
