"""Loader for Kilosort/Phy sorting output files (spike_times.npy, spike_clusters.npy, etc.)."""

from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from . import _globals
from .waveform_extractor import Utils


class KilosortSortingExtractor:
    """
    Represents data from Phy and Kilosort output folder as Python object

    Parameters
    ----------
    folder_path: str or Path
        Path to the output Phy folder (containing the params.py which stores data about the raw recording)
    exclude_cluster_groups: list or str (optional)
        Cluster groups to exclude (e.g. "noise" or ["noise", "mua"])
    """

    def __init__(self, folder_path, exclude_cluster_groups=None):
        # Folder containing the numpy results of Kilosort
        phy_folder = Path(folder_path)
        self.folder = phy_folder.absolute()

        self.spike_times = np.atleast_1d(
            np.load(str(phy_folder / "spike_times.npy")).astype(int).flatten()
        )
        self.spike_clusters = np.atleast_1d(
            np.load(str(phy_folder / "spike_clusters.npy")).flatten()
        )

        # The unit_ids with at least 1 spike
        unit_ids_with_spike = set(self.spike_clusters)

        params = Utils.read_python(str(phy_folder / "params.py"))
        self.sampling_frequency = params["sample_rate"]

        # Load properties from tsv/csv files
        all_property_files = [
            p for p in phy_folder.iterdir() if p.suffix in [".csv", ".tsv"]
        ]

        cluster_info = None
        for file in all_property_files:
            if file.suffix == ".tsv":
                delimeter = "\t"
            else:
                delimeter = ","
            new_property = pd.read_csv(file, delimiter=delimeter)
            if cluster_info is None:
                cluster_info = new_property
            else:
                if new_property.columns[-1] not in cluster_info.columns:
                    # cluster_KSLabel.tsv and cluster_group.tsv are identical and have the same columns
                    # This prevents the same column data being added twice
                    cluster_info = pd.merge(cluster_info, new_property, on="cluster_id")

        # In case no tsv/csv files are found populate cluster info with minimal info
        if cluster_info is None:
            unit_ids_with_spike_list = list(unit_ids_with_spike)
            cluster_info = pd.DataFrame({"cluster_id": unit_ids_with_spike_list})
            cluster_info["group"] = ["unsorted"] * len(unit_ids_with_spike_list)

        # If pandas column for the unit_ids uses different name
        if "cluster_id" not in cluster_info.columns:
            if "id" not in cluster_info.columns:
                raise ValueError(
                    "Couldn't find cluster IDs in the TSV file. Expected a "
                    f"'cluster_id' or 'id' column, found: {list(cluster_info.columns)}"
                )
            cluster_info["cluster_id"] = cluster_info["id"]
            del cluster_info["id"]

        if exclude_cluster_groups is not None:
            if isinstance(exclude_cluster_groups, str):
                cluster_info = cluster_info.query(
                    f"group != '{exclude_cluster_groups}'"
                )
            elif isinstance(exclude_cluster_groups, list):
                if len(exclude_cluster_groups) > 0:
                    for exclude_group in exclude_cluster_groups:
                        cluster_info = cluster_info.query(f"group != '{exclude_group}'")

        if (
            _globals.KILOSORT_PARAMS["keep_good_only"]
            and "KSLabel" in cluster_info.columns
        ):
            cluster_info = cluster_info.query("KSLabel == 'good'")

        all_unit_ids = cluster_info["cluster_id"].values
        self.unit_ids = []
        # Exclude units with 0 spikes
        for unit_id in all_unit_ids:
            if unit_id in unit_ids_with_spike:
                self.unit_ids.append(int(unit_id))

    @staticmethod
    def get_num_segments():
        # Sorting should always have 1 segment
        return 1

    def get_unit_spike_train(
        self,
        unit_id,
        segment_index: Union[int, None] = None,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
    ):
        spike_times = self.spike_times[self.spike_clusters == unit_id]
        if start_frame is not None:
            spike_times = spike_times[spike_times >= start_frame]
        if end_frame is not None:
            spike_times = spike_times[spike_times < end_frame]

        return np.atleast_1d(spike_times.copy().squeeze())

    def get_templates_all(self):
        # Returns Kilosort2's outputted templates as mmap np.array
        return np.load(str(self.folder / "templates.npy"), mmap_mode="r")

    def get_channel_map(self):
        # Returns Kilosort2's channel map as mmap np.array
        return np.load(str(self.folder / "channel_map.npy"), mmap_mode="r").squeeze()

    def get_chans_max(self):
        """
        Get the max channel of each unit based on Kilosort2's template
        and whether to use (min/argmin or max/argmax) for computing peak values

        Returns
        -------
        All are np.arrays that follow np.array[unit_id] = value
        In other words, the np.arrays contain data for ALL units (even units with 0 spikes)

        use_pos_peak
            0 = Use negative peak
            1 = Use positive peak
        chans_max_kilosort
            The channel with the highest amplitude for each unit based on kilosort's selected channels
            that were used during spike sorting (considered not "bad channels")
        chans_max
            The channel with the highest amplitude for each unit converted from kilosort's channels
            to channels in the recording (with all channels)
        """

        templates_all = self.get_templates_all()

        chans_neg_peaks_values = np.min(templates_all, axis=1)
        chans_neg_peaks_indices = chans_neg_peaks_values.argmin(axis=1)
        chans_neg_peaks_values = np.min(chans_neg_peaks_values, axis=1)

        chans_pos_peaks_values = np.max(templates_all, axis=1)
        chans_pos_peaks_indices = chans_pos_peaks_values.argmax(axis=1)
        chans_pos_peaks_values = np.max(chans_pos_peaks_values, axis=1)

        use_pos_peak = chans_pos_peaks_values >= _globals.POS_PEAK_THRESH * np.abs(
            chans_neg_peaks_values
        )
        chans_max_kilosort = np.where(
            use_pos_peak, chans_pos_peaks_indices, chans_neg_peaks_indices
        )
        chans_max_all = self.get_channel_map()[chans_max_kilosort]

        return use_pos_peak, chans_max_kilosort, chans_max_all

    def get_templates_half_windows_sizes(
        self, chans_max_kilosort, window_size_scale=0.75
    ):
        """
        Get the half window sizes that will be used to recenter the spike times on the peak

        Parameters
        ----------
        chans_max_kilosort: np.array
            np.array with shape (n_templates,) giving the max channel of each template using
            Kilosort's channel map
        window_size_scale: float
            Value to scale the window size for finding the peak
                Smaller = smaller window, less risk of picking wrong peak, higher risk of picking not the peak value of the peak

        Returns
        -------

        """
        # Get the half window sizes that will be used to recenter the spike times on the peak
        templates_all = self.get_templates_all()[
            np.arange(chans_max_kilosort.size), :, chans_max_kilosort
        ]
        n_templates, n_samples = templates_all.shape
        template_mid = n_samples // 2
        half_windows_sizes = []
        for i in range(n_templates):
            template = templates_all[i, :]
            # Find where the template amplitude drops below 1% of peak
            # before the midpoint.  Works for both KS2 (zero-padded) and
            # KS4 (dense, non-zero edges) templates.
            peak_amp = np.abs(template).max()
            threshold = peak_amp * 0.01
            small_indices = np.flatnonzero(np.abs(template[:template_mid]) < threshold)
            if small_indices.size > 0:
                size = template_mid - small_indices[-1]
            else:
                size = template_mid
            half_windows_sizes.append(int(size * window_size_scale))

        return half_windows_sizes

    def ms_to_samples(self, ms: float) -> int:
        return round(ms * self.sampling_frequency / 1000.0)
