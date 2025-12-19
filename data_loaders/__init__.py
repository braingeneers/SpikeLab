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
)

from .s3_utils import download_from_s3, ensure_local_file, is_s3_url, parse_s3_url

__all__ = [
    "load_spikedata_from_hdf5",
    "load_spikedata_from_hdf5_raw_thresholded",
    "load_spikedata_from_nwb",
    "load_spikedata_from_kilosort",
    "load_spikedata_from_spikeinterface",
    "load_spikedata_from_spikeinterface_recording",
    "download_from_s3",
    "ensure_local_file",
    "is_s3_url",
    "parse_s3_url",
]
