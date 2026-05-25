"""Loader for Kilosort/Phy sorting output files (spike_times.npy, spike_clusters.npy, etc.)."""

from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np

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
    compact: bool (default False)
        If True, remap the surviving Phy cluster IDs to a dense
        ``0..N-1`` range. The original cluster IDs are preserved on
        ``self.original_unit_ids``. ``spike_clusters`` values are
        remapped accordingly, and spikes belonging to filtered-out
        clusters are dropped. Use this when Phy curation has left a
        sparse cluster_id space (e.g. ``[0, 1, 47, 50000]``) that
        would otherwise blow up downstream template caches sized by
        ``max(unit_ids) + 1``. When False (the default), Phy cluster
        IDs flow through unchanged.
    """

    def __init__(
        self,
        folder_path,
        exclude_cluster_groups=None,
        keep_good_only=False,
        pos_peak_thresh=2.0,
        compact: bool = False,
    ):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for KilosortSortingExtractor. "
                "Install with: pip install pandas"
            ) from e

        # Folder containing the numpy results of Kilosort
        phy_folder = Path(folder_path)
        self.folder = phy_folder.absolute()
        self.pos_peak_thresh = pos_peak_thresh

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

        # Coerce cluster_id to int explicitly. ``pd.read_csv`` infers
        # dtypes per column, so a TSV that writes IDs as ``1.0`` (float
        # literal) or ``"001"`` (string-padded) ends up as float or
        # object dtype — the ``int(unit_id)`` casts later break with
        # confusing errors. Coerce up-front and surface the actual
        # offending value cleanly when coercion fails.
        try:
            cluster_info["cluster_id"] = cluster_info["cluster_id"].astype(int)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"cluster_id column has non-integer values "
                f"(dtype={cluster_info['cluster_id'].dtype}): {exc}. "
                "Expected integer cluster IDs from Phy/kilosort output."
            ) from exc

        if exclude_cluster_groups is not None:
            # Boolean indexing instead of ``query`` so a group name
            # containing a single quote or other expression-special
            # character doesn't reach the ``pd.eval`` parser and
            # silently mismatch or raise a parser error.
            if isinstance(exclude_cluster_groups, str):
                cluster_info = cluster_info[
                    cluster_info["group"] != exclude_cluster_groups
                ]
            elif isinstance(exclude_cluster_groups, list):
                if len(exclude_cluster_groups) > 0:
                    cluster_info = cluster_info[
                        ~cluster_info["group"].isin(exclude_cluster_groups)
                    ]

        if keep_good_only and "KSLabel" in cluster_info.columns:
            cluster_info = cluster_info.query("KSLabel == 'good'")

        all_unit_ids = cluster_info["cluster_id"].values
        surviving_original_ids = [
            int(uid) for uid in all_unit_ids if uid in unit_ids_with_spike
        ]
        self._compacted = bool(compact)
        if self._compacted:
            self.original_unit_ids = list(surviving_original_ids)
            self.unit_ids = list(range(len(surviving_original_ids)))
            if surviving_original_ids:
                orig_to_dense = {
                    orig: i for i, orig in enumerate(surviving_original_ids)
                }
                keep_mask = np.isin(self.spike_clusters, surviving_original_ids)
                self.spike_times = self.spike_times[keep_mask]
                kept_clusters = self.spike_clusters[keep_mask]
                self.spike_clusters = np.fromiter(
                    (orig_to_dense[int(c)] for c in kept_clusters),
                    dtype=self.spike_clusters.dtype,
                    count=len(kept_clusters),
                )
            else:
                self.spike_times = self.spike_times[:0]
                self.spike_clusters = self.spike_clusters[:0]
        else:
            self.unit_ids = list(surviving_original_ids)
            self.original_unit_ids = list(surviving_original_ids)

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

        # ``ravel`` always returns a 1-D view regardless of input shape.
        # The previous ``np.atleast_1d(spike_times.copy().squeeze())``
        # idiom worked for the current 1-D ``spike_times`` storage but
        # was fragile: if ``self.spike_times`` ever became 2-D with
        # one column, ``squeeze`` would collapse it to 1-D but a
        # multi-column 2-D shape would be returned as-is and break
        # callers expecting 1-D. ``ravel`` is robust to either case.
        return np.asarray(spike_times.copy()).ravel()

    def get_templates_all(self):
        """Return templates aligned to ``self.unit_ids``.

        Uncompacted (default): returns the raw on-disk
        ``templates.npy`` as an mmap array indexed by Phy template/
        cluster id (callers do ``templates[unit_id]``).

        Compacted: fancy-indexes by ``self.original_unit_ids`` so that
        row ``i`` is the template for ``self.unit_ids[i]``. Callers
        keep doing ``templates[unit_id]`` and get the right row.
        """
        raw = np.load(str(self.folder / "templates.npy"), mmap_mode="r")
        if not self._compacted:
            return raw
        if not self.original_unit_ids:
            return raw[:0]
        return raw[self.original_unit_ids]

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

        use_pos_peak = chans_pos_peaks_values >= self.pos_peak_thresh * np.abs(
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
            if peak_amp == 0:
                half_windows_sizes.append(0)
                continue
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
