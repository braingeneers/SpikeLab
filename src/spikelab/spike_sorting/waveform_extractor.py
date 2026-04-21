"""Custom waveform extractor with per-spike peak centering, used by all Kilosort backends."""

import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

from spikeinterface.core import BaseRecording

from . import _globals
from .sorting_utils import Stopwatch, create_folder, print_stage


class WaveformExtractor:
    """Per-unit waveform storage, template computation, and curation helper.

    Extracts spike waveforms from a recording aligned to each unit's
    spike times, computes average and standard-deviation templates, and
    supports saving/loading curated subsets of units to disk.

    Parameters:
        recording (BaseRecording): SpikeInterface recording object.
        sorting (KilosortSortingExtractor): Sorting result containing
            unit IDs and spike trains.
        root_folder (Path): Root folder for all waveform data (contains
            ``extraction_parameters.json``).
        folder (Path): Sub-folder for this instance's unit ID list
            (e.g. initial, first-curation, second-curation).
    """

    # region Initialize
    def __init__(self, recording, sorting, root_folder, folder):
        with open(root_folder / "extraction_parameters.json", "r") as f:
            parameters = json.load(f)

        self.recording = recording
        self.sampling_frequency = parameters["sampling_frequency"]

        self.sorting = sorting
        self.root_folder = root_folder
        self.folder = Path(folder)
        create_folder(self.folder)

        # Cache in memory
        self._waveforms = {}
        self.template_cache = {}

        # Set Parameters
        self.nbefore = self.ms_to_samples(
            parameters["ms_before"]
        )  # Number of samples before waveform peak to include
        self.nafter = (
            self.ms_to_samples(parameters["ms_after"]) + 1
        )  # Number of samples after waveform peak to include (+1 since Python slicing is [inlusive, exclusive))
        self.nsamples = (
            self.nbefore + self.nafter
        )  # Total number of samples in waveform
        self.peak_ind = parameters["peak_ind"]

        # Extract waveforms as µV when the recording supports scaling
        if recording.has_scaleable_traces():
            self.return_scaled = True
            self.dtype = "float32"
        else:
            self.return_scaled = False
            self.dtype = parameters["dtype"]

        self.chans_max_folder = root_folder / "channels_max"
        self.use_pos_peak = None
        self.chans_max_kilosort = None
        self.chans_max_all = None

    @classmethod
    def create_initial(
        cls, recording_path, recording, sorting, root_folder, initial_folder
    ):
        # Create root waveform folder and data
        root_folder = Path(root_folder)
        create_folder(root_folder / "waveforms")

        # Use float32 when the recording supports µV scaling
        if recording.has_scaleable_traces():
            waveform_dtype = "float32"
        else:
            waveform_dtype = str(recording.get_dtype())

        parameters = {
            "recording_path": str(recording_path.absolute()),
            "sampling_frequency": recording.get_sampling_frequency(),
            "ms_before": _globals.WAVEFORMS_MS_BEFORE,
            "ms_after": _globals.WAVEFORMS_MS_AFTER,
            "peak_ind": int(
                _globals.WAVEFORMS_MS_BEFORE
                * recording.get_sampling_frequency()
                / 1000.0
            ),
            "pos_peak_thresh": _globals.POS_PEAK_THRESH,
            "max_waveforms_per_unit": _globals.MAX_WAVEFORMS_PER_UNIT,
            "dtype": waveform_dtype,
            "n_jobs": _globals.N_JOBS,
            "total_memory": _globals.TOTAL_MEMORY,
        }
        with open(root_folder / "extraction_parameters.json", "w") as f:
            json.dump(parameters, f)

        we = cls(recording, sorting, root_folder, initial_folder)

        # Get template window sizes for computing location of negative peak during waveform extraction
        (
            we.use_pos_peak,
            we.chans_max_kilosort,
            we.chans_max_all,
        ) = we.sorting.get_chans_max()
        create_folder(we.chans_max_folder)
        for save_file, save_data in zip(
            ("use_pos_peak.npy", "chans_max_kilosort.npy", "chans_max_all.npy"),
            (we.use_pos_peak, we.chans_max_kilosort, we.chans_max_all),
        ):
            np.save(we.chans_max_folder / save_file, save_data)

        # Save unit data
        np.save(str(initial_folder / "unit_ids.npy"), sorting.unit_ids)
        np.save(str(initial_folder / "spike_times.npy"), sorting.spike_times)
        np.save(str(initial_folder / "spike_clusters.npy"), sorting.spike_clusters)

        return we

    @classmethod
    def load_from_folder(
        cls,
        recording,
        sorting,
        root_folder,
        folder,
        use_pos_peak=None,
        chans_max_kilosort=None,
        chans_max_all=None,
    ):
        # Load waveform data from folder
        we = cls(recording, sorting, root_folder, folder)

        _possible_template_modes = ("average", "std", "median")
        for mode in _possible_template_modes:
            # Load cached templates
            template_file = we.root_folder / f"templates/templates_{mode}.npy"
            if template_file.is_file():
                we.template_cache[mode] = np.load(template_file, mmap_mode="r")

        if use_pos_peak is None:
            we.use_pos_peak = np.load(
                we.chans_max_folder / "use_pos_peak.npy", mmap_mode="r"
            )
            we.chans_max_kilosort = np.load(
                we.chans_max_folder / "chans_max_kilosort.npy", mmap_mode="r"
            )
            we.chans_max_all = np.load(
                we.chans_max_folder / "chans_max_all.npy", mmap_mode="r"
            )
        else:
            we.use_pos_peak = use_pos_peak
            we.chans_max_kilosort = chans_max_kilosort
            we.chans_max_all = chans_max_all

        we.load_units()
        return we

    def ms_to_samples(self, ms: float) -> int:
        return int(ms * self.sampling_frequency / 1000.0)

    # endregion

    # region Extract waveforms
    def run_extract_waveforms(self, **job_kwargs: Any) -> None:
        self.templates_half_windows_sizes = (
            self.sorting.get_templates_half_windows_sizes(self.chans_max_kilosort)
        )

        num_chans = self.recording.get_num_channels()
        job_kwargs["n_jobs"] = Utils.ensure_n_jobs(
            self.recording, job_kwargs.get("n_jobs", None)
        )

        selected_spikes = self.sample_spikes()

        # Get spike times
        selected_spike_times = {}
        for unit_id in self.sorting.unit_ids:
            selected_spike_times[unit_id] = []
            for segment_index in range(self.sorting.get_num_segments()):
                spike_times = self.sorting.get_unit_spike_train(
                    unit_id=unit_id, segment_index=segment_index
                )
                sel = selected_spikes[unit_id][segment_index]
                spike_times_sel = spike_times[sel]

                selected_spike_times[unit_id].append(spike_times_sel)

        # Prepare memmap for waveforms
        print("Preparing memory maps for waveforms")
        wfs_memmap = {}
        for unit_id in self.sorting.unit_ids:
            file_path = self.root_folder / "waveforms" / f"waveforms_{unit_id}.npy"
            n_spikes = np.sum([e.size for e in selected_spike_times[unit_id]])
            shape = (n_spikes, self.nsamples, num_chans)
            wfs = np.zeros(shape, self.dtype)
            np.save(str(file_path), wfs)
            wfs_memmap[unit_id] = file_path

        # Run extract waveforms
        func = WaveformExtractor._waveform_extractor_chunk
        init_func = WaveformExtractor._init_worker_waveform_extractor

        init_args = (
            self.recording,
            self.sorting,
            self,
            wfs_memmap,
            selected_spikes,
            selected_spike_times,
            self.nbefore,
            self.nafter,
            self.return_scaled,
        )
        processor = ChunkRecordingExecutor(
            self.recording,
            func,
            init_func,
            init_args,
            job_name="extract waveforms",
            handle_returns=True,
            **job_kwargs,
        )
        spike_times_centered_dicts = processor.run()

        # Copy original kilosort spike times
        shutil.copyfile(
            self.sorting.folder / "spike_times.npy",
            self.sorting.folder / "spike_times_kilosort.npy",
        )

        # Center spike times
        spike_times = self.sorting.spike_times
        spike_time_to_ind = {}
        for i, st in enumerate(spike_times):
            spike_time_to_ind[st] = i

        for st_dict in spike_times_centered_dicts:
            for st, st_cen in st_dict.items():
                spike_times[spike_time_to_ind[st]] = st_cen
        np.save(self.sorting.folder / "spike_times.npy", spike_times)

    def run_extract_waveforms_streaming(self) -> None:
        """Per-unit streaming waveform extraction and template computation.

        Processes one unit at a time.  For each unit:
          1. Read trace windows for only its selected spikes,
          2. Recenter each spike on its peak,
          3. Accumulate ``average`` and ``std`` templates into a shared
             ``template_cache`` array,
          4. Optionally persist the unit's waveforms to disk (gated by
             ``SAVE_WAVEFORM_FILES``),
          5. Drop the in-memory waveform buffer before moving on.

        Peak RAM usage scales with *one* unit's waveform buffer
        (``n_spikes × nsamples × num_channels``) rather than with the
        total unit count.  For a 1018-channel MaxOne recording and
        300 waveforms / 4 ms / float32, that is ~100 MB per unit —
        independent of how many units the sorter produced.

        Output compatibility: downstream code consumes templates via
        ``get_computed_template`` which reads from ``template_cache``,
        and per-epoch template code / waveform-path attributes both
        guard on file existence, so skipping waveform files is a
        supported degraded mode.
        """
        self.templates_half_windows_sizes = (
            self.sorting.get_templates_half_windows_sizes(self.chans_max_kilosort)
        )

        num_chans = self.recording.get_num_channels()
        unit_ids = list(self.sorting.unit_ids)

        selected_spikes = self.sample_spikes()

        # Persisted spike-times-per-segment
        selected_spike_times: Dict[Any, List[np.ndarray]] = {}
        for unit_id in unit_ids:
            per_seg: List[np.ndarray] = []
            for segment_index in range(self.sorting.get_num_segments()):
                st = self.sorting.get_unit_spike_train(
                    unit_id=unit_id, segment_index=segment_index
                )
                per_seg.append(st[selected_spikes[unit_id][segment_index]])
            selected_spike_times[unit_id] = per_seg

        # Pre-allocate template_cache so templates can be streamed in
        # per unit as we go (same layout as compute_templates()).
        templates_shape = (max(unit_ids) + 1, self.nsamples, num_chans)
        for mode in ("average", "std"):
            self.template_cache[mode] = np.zeros(templates_shape, dtype=self.dtype)

        save_waveforms = _globals.SAVE_WAVEFORM_FILES
        waveforms_dir = self.root_folder / "waveforms"
        nbefore = self.nbefore
        nafter = self.nafter

        spike_times_centered_all: Dict[int, int] = {}

        print(
            f"[streaming] Extracting waveforms + templates for {len(unit_ids)} "
            f"units (peak RAM per unit ~"
            f"{_globals.MAX_WAVEFORMS_PER_UNIT * self.nsamples * num_chans * 4 / 1024 / 1024:.0f} MB)"
        )

        for unit_id in tqdm(unit_ids, desc="units"):
            half_window_size = self.templates_half_windows_sizes[unit_id]
            before_buffer = max(nbefore, half_window_size)
            after_buffer = max(nafter, half_window_size)
            chan_max = self.chans_max_all[unit_id]
            use_pos_peak = bool(self.use_pos_peak[unit_id])

            # Collect spike-time buffers across segments (usually 1 segment)
            unit_spike_times: List[Tuple[int, int]] = []
            for segment_index in range(self.sorting.get_num_segments()):
                seg_size = self.recording.get_num_samples(
                    segment_index=segment_index
                )
                st_array = selected_spike_times[unit_id][segment_index]
                for st in st_array:
                    st = int(st)
                    if st - before_buffer < 0 or st + after_buffer > seg_size:
                        continue
                    unit_spike_times.append((segment_index, st))

            n_spikes = len(unit_spike_times)
            if n_spikes == 0:
                continue

            # Local buffer — the only multi-MB allocation for this unit.
            wfs_local = np.zeros((n_spikes, self.nsamples, num_chans), dtype=self.dtype)

            for i, (segment_index, st) in enumerate(unit_spike_times):
                start = int(st - before_buffer)
                end = int(st + after_buffer)
                traces = self.recording.get_traces(
                    start_frame=start,
                    end_frame=end,
                    segment_index=segment_index,
                    return_scaled=self.return_scaled,
                )
                st_trace = st - start

                peak_left = max(st_trace - half_window_size, 0)
                peak_right = min(st_trace + half_window_size + 1, traces.shape[0])
                peak_window = traces[peak_left:peak_right, chan_max]
                if peak_window.size == 0:
                    spike_times_centered_all[st] = st
                    wfs_local[i] = traces[st_trace - nbefore : st_trace + nafter, :]
                    continue

                peak_value = np.max(peak_window) if use_pos_peak else np.min(peak_window)
                peak_indices = np.flatnonzero(peak_window == peak_value)
                st_offset = (
                    peak_indices[peak_indices.size // 2]
                    - peak_window.size // 2
                )
                st_trace += st_offset
                spike_times_centered_all[st] = st + st_offset

                # Clamp if recentering pushed the window past the chunk
                lo = max(st_trace - nbefore, 0)
                hi = min(st_trace + nafter, traces.shape[0])
                wf_lo = nbefore - (st_trace - lo)
                wf_hi = wf_lo + (hi - lo)
                wfs_local[i, wf_lo:wf_hi, :] = traces[lo:hi, :]

            # Templates — write directly into the shared cache.
            self.template_cache["average"][unit_id, :, :] = np.average(wfs_local, axis=0)
            self.template_cache["std"][unit_id, :, :] = np.std(wfs_local, axis=0)

            if save_waveforms:
                file_path = waveforms_dir / f"waveforms_{unit_id}.npy"
                np.save(str(file_path), wfs_local)

            del wfs_local

        # Templates folder on disk (same layout as compute_templates())
        templates_folder = self.root_folder / "templates"
        create_folder(templates_folder)
        for mode in ("average", "std"):
            np.save(
                str(templates_folder / f"templates_{mode}.npy"),
                self.template_cache[mode],
            )

        # Recenter spike times in the sorting (same as parallel path)
        shutil.copyfile(
            self.sorting.folder / "spike_times.npy",
            self.sorting.folder / "spike_times_kilosort.npy",
        )
        spike_times = self.sorting.spike_times
        spike_time_to_ind = {st: i for i, st in enumerate(spike_times)}
        for st, st_cen in spike_times_centered_all.items():
            if st in spike_time_to_ind:
                spike_times[spike_time_to_ind[st]] = st_cen
        np.save(self.sorting.folder / "spike_times.npy", spike_times)

    def sample_spikes(self) -> dict:
        """
        Uniform random selection of spikes per unit and save to .npy

        self.samples_spikes just calls self.random_spikes_uniformly and saves data to .npy files

        Returns
        -------
        Dictionary of {unit_id, [selected_spike_times]}
        """

        print("Sampling spikes for each unit")
        selected_spikes = self.select_random_spikes_uniformly()

        # Store in 2 columns (spike_index, segment_index) in a .npy file
        # NOT NECESSARY BUT COULD BE USEFUL FOR DEBUGGING
        print("Saving sampled spikes in .npy format")
        for unit_id in self.sorting.unit_ids:
            n = np.sum([e.size for e in selected_spikes[unit_id]])
            sampled_index = np.zeros(
                n, dtype=[("spike_index", "int64"), ("segment_index", "int64")]
            )
            pos = 0
            for segment_index in range(self.sorting.get_num_segments()):
                inds = selected_spikes[unit_id][segment_index]
                sampled_index[pos : pos + inds.size]["spike_index"] = inds
                sampled_index[pos : pos + inds.size]["segment_index"] = segment_index
                pos += inds.size

            sampled_index_file = (
                self.root_folder / "waveforms" / f"sampled_index_{unit_id}.npy"
            )
            np.save(str(sampled_index_file), sampled_index)

        return selected_spikes

    def select_random_spikes_uniformly(self) -> dict:
        """
        Uniform random selection of spikes per unit.

        More complicated than necessary because it is designed to handle multi-segment data
        Must keep complications since ChunkRecordingExecutor expects multi-segment data

        :return:
        Dictionary of {unit_id, [selected_spike_times]}
        """
        sorting = self.sorting
        unit_ids = sorting.unit_ids
        num_seg = sorting.get_num_segments()

        selected_spikes = {}
        for unit_id in unit_ids:
            # spike per segment
            n_per_segment = [
                sorting.get_unit_spike_train(unit_id, segment_index=i).size
                for i in range(num_seg)
            ]
            cum_sum = [0] + np.cumsum(n_per_segment).tolist()
            total = np.sum(n_per_segment)
            if _globals.MAX_WAVEFORMS_PER_UNIT is not None:
                if total > _globals.MAX_WAVEFORMS_PER_UNIT:
                    global_inds = np.random.choice(
                        total, size=_globals.MAX_WAVEFORMS_PER_UNIT, replace=False
                    )
                    global_inds = np.sort(global_inds)
                else:
                    global_inds = np.arange(total)
            else:
                global_inds = np.arange(total)
            sel_spikes = []
            for segment_index in range(num_seg):
                in_segment = (global_inds >= cum_sum[segment_index]) & (
                    global_inds < cum_sum[segment_index + 1]
                )
                inds = global_inds[in_segment] - cum_sum[segment_index]

                if _globals.MAX_WAVEFORMS_PER_UNIT is not None:
                    # clean border when sub selection
                    if self.nafter is None:
                        raise RuntimeError(
                            "nafter is not set — waveform extraction parameters "
                            "were not initialized."
                        )
                    spike_times = sorting.get_unit_spike_train(
                        unit_id=unit_id, segment_index=segment_index
                    )
                    sampled_spike_times = spike_times[inds]
                    num_samples = self.recording.get_num_samples(
                        segment_index=segment_index
                    )
                    mask = (sampled_spike_times >= self.nbefore) & (
                        sampled_spike_times < (num_samples - self.nafter)
                    )
                    inds = inds[mask]

                sel_spikes.append(inds)
            selected_spikes[unit_id] = sel_spikes
        return selected_spikes

    @staticmethod
    def _waveform_extractor_chunk(segment_index, start_frame, end_frame, worker_ctx):
        # recover variables of the worker
        recording = worker_ctx["recording"]
        sorting = worker_ctx["sorting"]

        waveform_extractor = worker_ctx["waveform_extractor"]
        templates_half_windows_sizes = waveform_extractor.templates_half_windows_sizes
        use_pos_peak = waveform_extractor.use_pos_peak
        chans_max_all = waveform_extractor.chans_max_all

        wfs_memmap_files = worker_ctx["wfs_memmap_files"]
        selected_spikes = worker_ctx["selected_spikes"]
        selected_spike_times = worker_ctx["selected_spike_times"]
        nbefore = worker_ctx["nbefore"]
        nafter = worker_ctx["nafter"]
        return_scaled = worker_ctx["return_scaled"]
        unit_cum_sum = worker_ctx["unit_cum_sum"]

        seg_size = recording.get_num_samples(segment_index=segment_index)

        to_extract = {}
        for unit_id in sorting.unit_ids:
            spike_times = selected_spike_times[unit_id][segment_index]
            i0 = np.searchsorted(spike_times, start_frame)
            i1 = np.searchsorted(spike_times, end_frame)
            if i0 != i1:
                # protect from spikes on border :  spike_time<0 or spike_time>seg_size
                # useful only when max_spikes_per_unit is not None
                # waveform will not be extracted and a zeros will be left in the memmap file
                template_half_window_size = templates_half_windows_sizes[unit_id]
                before_buffer = max(nbefore, template_half_window_size)
                after_buffer = max(nafter, template_half_window_size)
                while (spike_times[i0] - before_buffer) < 0 and (i0 != i1):
                    i0 = i0 + 1
                while (spike_times[i1 - 1] + after_buffer) > seg_size and (i0 != i1):
                    i1 = i1 - 1

            if i0 != i1:
                to_extract[unit_id] = i0, i1, spike_times[i0:i1]

        spike_times_centered = {}
        if len(to_extract) > 0:
            start = min(
                st[0] - nbefore - templates_half_windows_sizes[uid]
                for uid, (_, _, st) in to_extract.items()
            )  # Get the minimum time frame from recording needed for extracting waveform from the minimum spike time - nbefore
            end = max(
                st[-1] + nafter + templates_half_windows_sizes[uid]
                for uid, (_, _, st) in to_extract.items()
            )
            start = int(max(0, start))
            end = int(min(end, recording.get_num_samples()))
            # load trace in memory
            traces = recording.get_traces(
                start_frame=start,
                end_frame=end,
                segment_index=segment_index,
                return_scaled=return_scaled,
            )
            max_trace_ind = traces.shape[0] - 1
            for unit_id, (i0, i1, local_spike_times) in to_extract.items():
                wfs = np.load(wfs_memmap_files[unit_id], mmap_mode="r+")
                half_window_size = templates_half_windows_sizes[unit_id]
                chan_max = chans_max_all[unit_id]
                for i in range(local_spike_times.size):
                    st = int(local_spike_times[i])  # spike time
                    st_trace = (
                        st - start
                    )  # Convert the spike time defined by all the samples in recording to only samples in "traces"

                    peak_window_left = max(st_trace - half_window_size, 0)
                    peak_window_right = min(
                        st_trace + half_window_size + 1, max_trace_ind + 1
                    )
                    traces_peak_window = traces[
                        peak_window_left:peak_window_right, chan_max
                    ]
                    if traces_peak_window.size == 0:
                        # Spike at chunk boundary — skip recentering
                        spike_times_centered[st] = st
                        continue
                    if use_pos_peak[unit_id]:
                        peak_value = np.max(traces_peak_window)
                    else:
                        peak_value = np.min(traces_peak_window)
                    peak_indices = np.flatnonzero(traces_peak_window == peak_value)
                    st_offset = (
                        peak_indices[peak_indices.size // 2]
                        - traces_peak_window.size // 2
                    )
                    st_trace += st_offset

                    spike_times_centered[st] = st + st_offset

                    pos = (
                        unit_cum_sum[unit_id][segment_index] + i0 + i
                    )  # Index for waveform along 0th axis in .npy waveforms file
                    wf = traces[
                        st_trace - nbefore : st_trace + nafter, :
                    ]  # Python slices with [start, end), so waveform is in format (nbefore + spike_location + nafter-1, n_channels)
                    wfs[pos, :, :] = wf
        return spike_times_centered

    @staticmethod
    def _init_worker_waveform_extractor(
        recording,
        sorting,
        waveform_extractor,
        wfs_memmap,
        selected_spikes,
        selected_spike_times,
        nbefore,
        nafter,
        return_scaled,
    ):
        # create a local dict per worker
        worker_ctx = {}
        worker_ctx["recording"] = recording
        worker_ctx["sorting"] = sorting
        worker_ctx["waveform_extractor"] = waveform_extractor

        worker_ctx["wfs_memmap_files"] = wfs_memmap
        worker_ctx["selected_spikes"] = selected_spikes
        worker_ctx["selected_spike_times"] = selected_spike_times
        worker_ctx["nbefore"] = nbefore
        worker_ctx["nafter"] = nafter
        worker_ctx["return_scaled"] = return_scaled

        num_seg = sorting.get_num_segments()
        unit_cum_sum = {}
        for unit_id in sorting.unit_ids:
            # spike per segment
            n_per_segment = [selected_spikes[unit_id][i].size for i in range(num_seg)]
            cum_sum = [0] + np.cumsum(n_per_segment).tolist()
            unit_cum_sum[unit_id] = cum_sum
        worker_ctx["unit_cum_sum"] = unit_cum_sum

        return worker_ctx

    # endregion

    # region Get waveforms and templates
    def get_waveforms(
        self,
        unit_id: int,
        with_index: bool = False,
        cache: bool = False,
        memmap: bool = True,
    ) -> Any:  # SpikeInterface has cache=True by default
        """
        Return waveforms for the specified unit id.

        Parameters
        ----------
        unit_id: int or str
            Unit id to retrieve waveforms for
        with_index: bool
            If True, spike indices of extracted waveforms are returned (default False)
        cache: bool
            If True, waveforms are cached to the self.waveforms dictionary (default False)
        memmap: bool
            If True, waveforms are loaded as memmap objects.
            If False, waveforms are loaded as np.array objects (default True)

        Returns
        -------
        wfs: np.array
            The returned waveform (num_spikes, num_samples, num_channels)
            num_samples = nbefore + 1 (for value at peak) + nafter
        indices: np.array
            If 'with_index' is True, the spike indices corresponding to the waveforms extracted
        """
        wfs = self._waveforms.get(unit_id, None)
        if wfs is None:
            waveform_file = self.root_folder / "waveforms" / f"waveforms_{unit_id}.npy"
            if not waveform_file.is_file():
                raise Exception(
                    "Waveforms not extracted yet: "
                    "please set 'REEXTRACT_WAVEFORMS' to True"
                )
            if memmap:
                wfs = np.load(waveform_file, mmap_mode="r")
            else:
                wfs = np.load(waveform_file)
            if cache:
                self._waveforms[unit_id] = wfs

        if with_index:
            sampled_index = self.get_sampled_indices(unit_id)
            return wfs, sampled_index
        else:
            return wfs

    def get_sampled_indices(self, unit_id: int) -> list:
        """
        Return sampled spike indices of extracted waveforms
        (which waveforms correspond to which spikes if "max_spikes_per_unit" is not None)

        Parameters
        ----------
        unit_id: int
            Unit id to retrieve indices for

        Returns
        -------
        sampled_indices: np.array
            The sampled indices with shape (n_waveforms,)
        """

        sampled_index_file = (
            self.root_folder / "waveforms" / f"sampled_index_{unit_id}.npy"
        )
        sampled_index = np.load(str(sampled_index_file))

        # When this function was written, the sampled_index .npy files also included segment index of spikes
        # This disregards segment index since there should only be 1 segment
        sampled_index_without_segment_index = []
        for index in sampled_index:
            sampled_index_without_segment_index.append(index[0])
        return sampled_index_without_segment_index

    def get_computed_template(self, unit_id: int, mode: str) -> np.ndarray:
        """
        Return template (average waveform).

        Parameters
        ----------
        unit_id: int
            Unit id to retrieve waveforms for
        mode: str
            'average' (default), 'median' , 'std'(standard deviation)
        Returns
        -------
        template: np.array
            The returned template (num_samples, num_channels)
        """

        _possible_template_modes = {"average", "std", "median"}
        if mode not in _possible_template_modes:
            raise ValueError(
                f"mode must be one of {_possible_template_modes}, got '{mode}'"
            )

        if mode in self.template_cache:
            # already in the global cache
            template = self.template_cache[mode][unit_id, :, :]
            return template

        # compute from waveforms
        wfs = self.get_waveforms(unit_id)
        if mode == "median":
            template = np.median(wfs, axis=0)
        elif mode == "average":
            template = np.average(wfs, axis=0)
        elif mode == "std":
            template = np.std(wfs, axis=0)
        return template

    def compute_templates(
        self, modes=("average", "std"), unit_ids=None, folder=None, n_jobs=1
    ):
        """
        Compute all template for different "modes":
          * average
          * std
          * median

        The results are cached in memory as 3d ndarray (nunits, nsamples, nchans)
        and also saved as npy file in the folder to avoid recomputation each time.

        Parameters
        ----------
        modes: tuple
            Template modes to compute (average, std, median)
        unit_ids: None or List
            Unit ids to compute templates for
            If None-> unit ids are taken from self.sorting.unit_ids
        folder: None or Path
            Folder to save templates to
            If None-> use self.folder
        n_jobs: int
            Number of threads for parallel template computation.
            Default 1 (sequential). Values > 1 use a thread pool
            which speeds up I/O-bound waveform loading from disk.
        """
        print_stage("COMPUTING TEMPLATES")
        print("Template modes: " + ", ".join(modes))
        stopwatch = Stopwatch()

        if unit_ids is None:
            unit_ids = self.sorting.unit_ids
        if folder is None:
            folder = self.root_folder / "templates"

        num_chans = self.recording.get_num_channels()

        for mode in modes:
            # With max(unit_ids)+1 instead of len(unit_ids), the template of unit_id can be retrieved by template[unit_id]
            # Instead of first converting unit_id to an index
            templates = np.zeros(
                (max(unit_ids) + 1, self.nsamples, num_chans), dtype=self.dtype
            )
            self.template_cache[mode] = templates

        def _compute_unit_template(unit_id):
            """Load waveforms and compute templates for a single unit."""
            wfs = self.get_waveforms(unit_id, cache=False)
            for mode in modes:
                if mode == "median":
                    arr = np.median(wfs, axis=0)
                elif mode == "average":
                    arr = np.average(wfs, axis=0)
                elif mode == "std":
                    arr = np.std(wfs, axis=0)
                else:
                    raise ValueError("mode must in median/average/std")
                self.template_cache[mode][unit_id, :, :] = arr

        n_units = len(unit_ids)
        n_workers = min(n_jobs, n_units) if n_jobs > 1 else 1
        print(f"Computing templates for {n_units} units (n_jobs={n_workers})")

        if n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(_compute_unit_template, uid): uid for uid in unit_ids
                }
                for _ in tqdm(as_completed(futures), total=n_units, desc="Templates"):
                    pass
        else:
            for unit_id in tqdm(unit_ids):
                _compute_unit_template(unit_id)

        create_folder(folder)
        print("Saving templates to .npy")
        for mode in modes:
            templates = self.template_cache[mode]
            template_file = folder / f"templates_{mode}.npy"
            np.save(str(template_file), templates)
        stopwatch.log_time("Done computing and saving templates.")

    def load_units(self) -> None:
        self.sorting.unit_ids = np.load(str(self.folder / "unit_ids.npy")).tolist()
        self.sorting.spike_times = np.load(str(self.folder / "spike_times.npy"))
        self.sorting.spike_clusters = np.load(str(self.folder / "spike_clusters.npy"))

    def get_curation_history(self) -> Optional[dict]:
        path = self.folder / "curation_history.json"
        if path.exists():
            with open(self.folder / "curation_history.json", "r") as f:
                return json.load(f)
        else:
            return None

    # endregion

    # region Format files
    def save_curated_units(
        self, unit_ids, waveforms_root_folder, curated_folder, curation_history
    ):
        """
        Filters units by storing curated unit ids in a new folder.

        Parameters
        ----------
        unit_ids: list
            Contains which unit ids are curated
        waveforms_root_folder: Path
            The root of all waveforms
        curated_folder: Path
            The new folder where curated unit ids are saved
        curation_history: dict
            Contains curation history to be saved

        Return
        ------
        we :  WaveformExtractor
            The newly create waveform extractor with the selected units
        """
        print_stage("SAVING CURATED UNITS")
        stopwatch = Stopwatch()
        print(f"Saving {len(unit_ids)} curated units to new folder")
        create_folder(curated_folder)

        # Save data about unit ids
        spike_times_og = self.sorting.spike_times
        spike_clusters_og = self.sorting.spike_clusters
        unit_ids_set = set(unit_ids)
        selected_indices = [
            i for i, c in enumerate(spike_clusters_og) if c in unit_ids_set
        ]
        spike_times = spike_times_og[selected_indices]
        spike_clusters = spike_clusters_og[selected_indices]

        np.save(str(curated_folder / "unit_ids.npy"), unit_ids)
        np.save(str(curated_folder / "spike_times.npy"), spike_times)
        np.save(str(curated_folder / "spike_clusters.npy"), spike_clusters)

        # Save curation history
        with open(curated_folder / "curation_history.json", "w") as f:
            json.dump(curation_history, f)

        we = WaveformExtractor.load_from_folder(
            self.recording,
            self.sorting,
            waveforms_root_folder,
            curated_folder,
            self.use_pos_peak,
            self.chans_max_kilosort,
            self.chans_max_all,
        )
        stopwatch.log_time("Done saving curated units.")

        return we

    # endregion


class ChunkRecordingExecutor:
    """
    Used to extract waveforms from recording

    Core class for parallel processing to run a "function" over chunks on a recording.

    It supports running a function:
        * in loop with chunk processing (low RAM usage)
        * at once if chunk_size is None (high RAM usage)
        * in parallel with ProcessPoolExecutor (higher speed)

    The initializer ('init_func') allows to set a global context to avoid heavy serialization
    (for examples, see implementation in `core.WaveformExtractor`).

    Parameters
    ----------
    recording: RecordingExtractor
        The recording to be processed
    func: function
        Function that runs on each chunk
    init_func: function
        Initializer function to set the global context (accessible by 'func')
    init_args: tuple
        Arguments for init_func
    verbose: bool
        If True, output is verbose
    progress_bar: bool
        If True, a progress bar is printed to monitor the progress of the process
    handle_returns: bool
        If True, the function can return values
    n_jobs: int
        Number of jobs to be used (default 1). Use -1 to use as many jobs as number of cores
    total_memory: str
        Total memory (RAM) to use (e.g. "1G", "500M")
    chunk_memory: str
        Memory per chunk (RAM) to use (e.g. "1G", "500M")
    chunk_size: int or None
        Size of each chunk in number of samples. If 'TOTAL_MEMORY' or 'CHUNK_MEMORY' are used, it is ignored.
    job_name: str
        Job name

    Returns
    -------
    res: list
        If 'handle_returns' is True, the results for each chunk process
    """

    def __init__(
        self,
        recording,
        func,
        init_func,
        init_args,
        verbose=True,
        progress_bar=False,
        handle_returns=False,
        n_jobs=1,
        total_memory=None,
        chunk_size=None,
        chunk_memory=None,
        job_name="",
    ):
        self.recording = recording
        self.func = func
        self.init_func = init_func
        self.init_args = init_args

        self.verbose = verbose
        self.progress_bar = progress_bar

        self.handle_returns = handle_returns

        self.n_jobs = Utils.ensure_n_jobs(recording, n_jobs=n_jobs)
        self.chunk_size = Utils.ensure_chunk_size(
            recording,
            total_memory=total_memory,
            chunk_size=chunk_size,
            chunk_memory=chunk_memory,
            n_jobs=self.n_jobs,
        )
        self.job_name = job_name

        if verbose:
            print(
                self.job_name,
                "with",
                "n_jobs",
                self.n_jobs,
                " chunk_size",
                self.chunk_size,
            )

    def run(self):
        """
        Runs the defined jobs.
        """
        all_chunks = ChunkRecordingExecutor.divide_recording_into_chunks(
            self.recording, self.chunk_size
        )

        if self.handle_returns:
            returns = []
        else:
            returns = None

        import sys

        if self.n_jobs != 1 and not (sys.version_info >= (3, 8)):
            self.n_jobs = 1

        if self.n_jobs == 1:
            if self.progress_bar:
                all_chunks = tqdm(all_chunks, ascii=True, desc=self.job_name)

            worker_ctx = self.init_func(*self.init_args)
            for segment_index, frame_start, frame_stop in all_chunks:
                res = self.func(segment_index, frame_start, frame_stop, worker_ctx)
                if self.handle_returns:
                    returns.append(res)
        else:
            n_jobs = min(self.n_jobs, len(all_chunks))

            ######## Do you want to limit the number of threads per process?
            ######## It has to be done to speed up numpy a lot if multicores
            ######## Otherwise, np.dot will be slow. How to do that, up to you
            ######## This is just a suggestion, but here it adds a dependency

            # parallel
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=ChunkRecordingExecutor.worker_initializer,
                initargs=(self.func, self.init_func, self.init_args),
            ) as executor:
                results = executor.map(
                    ChunkRecordingExecutor.function_wrapper, all_chunks
                )

                if self.progress_bar:
                    results = tqdm(results, desc=self.job_name, total=len(all_chunks))

                if self.handle_returns:  # Should be false
                    for res in results:
                        returns.append(res)
                else:
                    for res in results:
                        pass

        return returns

    @staticmethod
    def function_wrapper(args):
        segment_index, start_frame, end_frame = args
        global _func
        global _worker_ctx
        return _func(segment_index, start_frame, end_frame, _worker_ctx)

    @staticmethod
    def divide_recording_into_chunks(recording, chunk_size):
        all_chunks = []
        for segment_index in range(recording.get_num_segments()):
            num_frames = recording.get_num_samples(segment_index)
            chunks = ChunkRecordingExecutor.divide_segment_into_chunks(
                num_frames, chunk_size
            )
            all_chunks.extend(
                [
                    (segment_index, frame_start, frame_stop)
                    for frame_start, frame_stop in chunks
                ]
            )
        return all_chunks

    @staticmethod
    def divide_segment_into_chunks(num_frames, chunk_size):
        if chunk_size is None:
            chunks = [(0, num_frames)]
        else:
            n = num_frames // chunk_size

            frame_starts = np.arange(n) * chunk_size
            frame_stops = frame_starts + chunk_size

            frame_starts = frame_starts.tolist()
            frame_stops = frame_stops.tolist()

            if (num_frames % chunk_size) > 0:
                frame_starts.append(n * chunk_size)
                frame_stops.append(num_frames)

            chunks = list(zip(frame_starts, frame_stops))

        return chunks

    @staticmethod
    def worker_initializer(func, init_func, init_args):
        global _worker_ctx
        _worker_ctx = init_func(*init_args)
        global _func
        _func = func


global _worker_ctx
global _func
# ProcessPoolExecutor: using stdlib concurrent.futures instead of vendored copy
# (already imported at top of module)

# endregion


# region Utilities
class Utils:
    """Utility helpers adapted from SpikeInterface.

    Provides static methods for parsing Kilosort2 Python parameter
    files, clamping worker counts to OS limits, and computing chunk
    sizes for parallel waveform extraction.
    """

    @staticmethod
    def read_python(path):
        """Parses python scripts in a dictionary

        Parameters
        ----------
        path: str or Path
            Path to file to parse

        Returns
        -------
        metadata:
            dictionary containing parsed file

        """
        from six import exec_
        import re

        path = Path(path).absolute()
        if not path.is_file():
            raise FileNotFoundError(f"Kilosort2 parameter file not found: {path}")
        with path.open("r") as f:
            contents = f.read()
        contents = re.sub(r"range\(([\d,]*)\)", r"list(range(\1))", contents)
        metadata = {}
        exec_(contents, {}, metadata)
        metadata = {k.lower(): v for (k, v) in metadata.items()}
        return metadata

    @staticmethod
    def ensure_n_jobs(recording, n_jobs=1):
        # Ensures that the number of jobs specified is possible by the operating system

        import joblib

        if n_jobs == -1:
            n_jobs = joblib.cpu_count()
        elif n_jobs == 0:
            n_jobs = 1
        elif n_jobs is None:
            n_jobs = 1

        version = sys.version_info

        if (n_jobs != 1) and not (version.major >= 3 and version.minor >= 7):
            print(f"Python {sys.version} does not support parallel processing")
            n_jobs = 1

        return n_jobs

    @staticmethod
    def ensure_chunk_size(
        recording,
        total_memory=None,
        chunk_size=None,
        chunk_memory=None,
        n_jobs=1,
        **other_kwargs,
    ):
        """
        'chunk_size' is the traces.shape[0] for each worker.

        Flexible chunk_size setter with 3 ways:
            * "chunk_size": is the length in sample for each chunk independently of channel count and dtype.
            * "chunk_memory": total memory per chunk per worker
            * "total_memory": total memory over all workers.

        If chunk_size/chunk_memory/total_memory are all None then there is no chunk computing
        and the full trace is retrieved at once.

        Parameters
        ----------
        chunk_size: int or None
            size for one chunk per job
        chunk_memory: str or None
            must endswith 'k', 'M' or 'G'
        total_memory: str or None
            must endswith 'k', 'M' or 'G'
        """

        if chunk_size is not None:
            # manual setting
            chunk_size = int(chunk_size)
        elif chunk_memory is not None:
            if total_memory is not None:
                raise ValueError(
                    "Cannot specify both 'chunk_memory' and 'total_memory'. "
                    "Provide only one."
                )
            # set by memory per worker size
            chunk_memory = Utils._mem_to_int(chunk_memory)
            n_bytes = np.dtype(recording.get_dtype()).itemsize
            num_channels = recording.get_num_channels()
            chunk_size = int(chunk_memory / (num_channels * n_bytes))
        if total_memory is not None:
            # clip by total memory size
            n_jobs = Utils.ensure_n_jobs(recording, n_jobs=n_jobs)
            total_memory = Utils._mem_to_int(total_memory)
            n_bytes = np.dtype(recording.get_dtype()).itemsize
            num_channels = recording.get_num_channels()
            chunk_size = int(total_memory / (num_channels * n_bytes * n_jobs))
        else:
            if n_jobs == 1:
                # not chunk computing
                chunk_size = None
            else:
                raise ValueError(
                    "For N_JOBS >1 you must specify TOTAL_MEMORY or chunk_size or CHUNK_MEMORY"
                )

        return chunk_size

    @staticmethod
    def _mem_to_int(mem):
        # Converts specified memory (e.g. 4G) to integer number
        _exponents = {"k": 1e3, "M": 1e6, "G": 1e9}

        suffix = mem[-1]
        if suffix not in _exponents:
            raise ValueError(
                f"Invalid memory suffix '{suffix}' in '{mem}'. "
                f"Expected one of: {list(_exponents.keys())} (e.g. '4G', '500M')"
            )
        mem = int(float(mem[:-1]) * _exponents[suffix])
        return mem


# endregion
