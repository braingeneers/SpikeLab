"""RT-Sort core algorithm: sequence detection and offline spike sorting.

This module is adapted from the RT-Sort algorithm:

    van der Molen, T., Lim, M., Bartram, J., Cheng, Z., Robbins, A.,
    Parks, D. F., ... & Kosik, K. S. (2024). RT-Sort: An action potential
    propagation-based algorithm for real time spike detection and sorting
    with millisecond latencies. PLoS ONE, 19(12), e0312438.
    https://doi.org/10.1371/journal.pone.0312438

Original implementation by Tjitse van der Molen and Max Lim,
upstream at https://github.com/KosikLabUCSB/RT-Sort (MIT License).

Copyright (c) 2024 Tjitse van der Molen, Max Lim
Licensed under MIT (see LICENSE).

Changes from the braindance upstream version:
    - Removed hard top-level ``torch`` import (now guarded/deferred so
      importing this module does not require torch unless RT-Sort is
      actually invoked).
    - Removed hard top-level ``pynvml`` and ``threadpoolctl`` imports
      (now lazily imported inside the functions that use them).
    - Removed the ``rt_sort_maxwell_env_process`` closed-loop streaming
      entry point; SpikeLab integrates RT-Sort for offline sorting only.
    - Rewired the ``ModelSpikeSorter`` import to the local ``model``
      module rather than ``braindance.core.spikedetector.model``.
"""

from copy import deepcopy
from math import ceil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, Manager


def _thread_map(num_workers, fn, items):
    """Map ``fn`` over ``items`` using a thread pool, returning an iterator
    that yields results as they complete.

    Used in place of ``multiprocessing.Pool.imap_unordered`` for the
    per-channel trace-extraction step.  Threads share the parent's
    interpreter and recording object — no fork-COW amplification, no
    ``spawn`` re-import overhead, no Manager-pickled recording proxy.
    The work is I/O-bound (h5py reads, numpy ``.astype`` / ``.flatten``)
    and both libraries release the GIL during their hot loops, so
    threads achieve genuine parallelism on the parts that matter.

    For Python's ``multiprocessing.Pool``-replacement semantics:
        * Returns a generator over completed results (order not preserved).
        * Cleans up the pool when the generator is exhausted.
    """
    pool = ThreadPoolExecutor(max_workers=max(1, num_workers))
    futures = [pool.submit(fn, item) for item in items]
    try:
        from concurrent.futures import as_completed

        for fut in as_completed(futures):
            yield fut.result()
    finally:
        pool.shutdown(wait=True)


import os
from pathlib import Path
import pickle
import shutil
from time import perf_counter
import warnings

import numpy as np
import scipy
from sklearn.mixture import GaussianMixture

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from diptest import diptest
except ImportError:  # pragma: no cover
    diptest = None

from spikeinterface.core import BaseRecording
from spikeinterface.extractors import (
    MaxwellRecordingExtractor,
    NwbRecordingExtractor,
    NumpySorting,
)
from tqdm import tqdm

from .model import ModelSpikeSorter


def _threadpool_limits(limits=1):
    """Return a threadpoolctl.threadpool_limits context manager, or a no-op.

    ``threadpoolctl`` is used to constrain BLAS threading during sklearn
    GMM fits (to avoid oversubscription when the outer loop is already
    parallelised).  It is optional; if not installed, this returns a
    dummy context manager so the call sites work unchanged.
    """

    try:
        from threadpoolctl import threadpool_limits as _tl

        return _tl(limits=limits)
    except ImportError:

        class _Null:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Null()


neuropixels_params = {
    "stringent_thresh": 0.175,
    "loose_thresh": 0.075,
    "inference_scaling_numerator": 15.4,
    "min_amp_dist_p": 0.1,
    "max_latency_diff_spikes": 2.5,
    "max_amp_median_diff_spikes": 0.45,
    "max_latency_diff_sequences": 2.5,
    "max_amp_median_diff_sequences": 0.45,
    "max_root_amp_median_std_sequences": 2.5,
}


def detect_sequences(
    recording,
    inter_path,
    detection_model=None,
    # Recording params
    recording_window_ms=None,  # If None, use entire recording. If an integer, use last recording_window_ms minutes
    # Detection model params
    stringent_thresh=0.275,
    loose_thresh=0.1,
    inference_scaling_numerator=12.6,
    # General params
    ms_before=0.5,
    ms_after=0.5,
    pre_median_ms=50,
    inner_radius=50,
    outer_radius=100,
    min_elecs_for_array_noise_n=100,
    min_elecs_for_array_noise_f=0.1,
    min_elecs_for_seq_noise_n=50,
    min_elecs_for_seq_noise_f=0.05,
    # Root elec needs :min_activity_hz: codetections where a codetection is when there are at least :min_activity_root_cocs: detections in the propagation window
    min_activity_root_cocs=2,
    min_activity_hz=0.05,
    max_n_components_latency=4,
    min_coc_n=10,
    min_coc_p=10,
    min_extend_comp_p=50,
    elec_patience=6,
    split_coc_clusters_amps=True,
    min_amp_dist_p=0.1,
    max_n_components_amp=4,
    min_loose_elec_prob=0.03,
    min_inner_loose_detections=3,
    min_loose_detections_n=4,
    min_loose_detections_r_spikes=1 / 3,
    min_loose_detections_r_sequences=1 / 3,
    max_latency_diff_spikes=3.5,
    max_latency_diff_sequences=3.5,
    clip_latency_diff_factor=2,
    max_amp_median_diff_spikes=0.65,
    max_amp_median_diff_sequences=0.65,
    clip_amp_median_diff_factor=2,
    max_root_amp_median_std_spikes=2.5,
    max_root_amp_median_std_sequences=np.inf,
    repeated_detection_overlap_time=0.2,
    min_seq_spikes_n=10,
    min_seq_spikes_hz=0.05,
    relocate_root_min_amp=0.8,
    relocate_root_max_latency=-2,
    # Other params
    return_spikes=False,
    delete_inter=False,
    device="cuda",  # For PyTorch, "cuda" or "cpu"
    num_processes=None,
    ignore_warnings=True,
    verbose=True,
    debug=False,
):
    """
    Detects sequences of spikes from a given recording using RT-Sort, a real-time spike detection and sorting algorithm.

    This function supports detection using a neural network model and a set of configurable parameters
    to control the spike detection process. It also handles sequence formation, spike reassignment, and merging.

    Note: The RT-Sort paper uses the term "intervals". Here, "intervals" is replaced with "latencies".

    Args:
        recording (str or pathlib.Path or SpikeInterface Recording):
            The recording to process. Can be a SpikeInterface recording object or a path to a
            supported recording format (.h5 or .nwb for Maxwell MEA and Neurodata Without Borders, respectively).
        inter_path (str or pathlib.Path):
            Path to a folder (existing or non-existing) where RT-Sort's intermediate cached data is stored.
        detection_model (ModelSpikeSorter, str or pathlib.Path, optional):
            A `ModelSpikeSorter` object or a path to a folder containing a `ModelSpikeSorter`
            object's `init_dict.json` and `state_dict.pt`. Defaults to None, in which case a default model is loaded.
        recording_window_ms (tuple, optional):
            A tuple `(start_ms, end_ms)` defining the portion of the recording (in milliseconds) to process.
            If None, the entire recording is used. Defaults to None.
        stringent_thresh (float, optional):
            The stringent threshold for spike detection. Defaults to 0.275.
        loose_thresh (float, optional):
            The loose threshold for spike detection. Defaults to 0.1.
        inference_scaling_numerator (float, optional):
            Scaling factor for inference. Defaults to 12.6.
        ms_before (float, optional):
            Time (in milliseconds) to consider before each detected spike for sequence formation. Defaults to 0.5 ms.
        ms_after (float, optional):
            Time (in milliseconds) to consider after each detected spike for sequence formation. Defaults to 0.5 ms.
        pre_median_ms (float, optional):
            Duration (in milliseconds) to compute the median for normalization. Defaults to 50 ms.
        inner_radius (float, optional):
            Inner radius (in micrometers). Defaults to 50.
        outer_radius (float, optional):
            Outer radius (in micrometers). Defaults to 100.
        min_elecs_for_array_noise_n (int, optional):
            Minimum number of electrodes for array-wide noise filtering. Defaults to 100.
        min_elecs_for_array_noise_f (float, optional):
            Minimum fraction of electrodes for array-wide noise filtering. Defaults to 0.1.
        min_elecs_for_seq_noise_n (int, optional):
            Minimum number of electrodes for sequence-wide noise filtering. Defaults to 50.
        min_elecs_for_seq_noise_f (float, optional):
            Minimum fraction of electrodes for sequence-wide noise filtering. Defaults to 0.05.
        min_activity_root_cocs (int, optional):
            Minimum number of stringent spike detections on inner electrodes within
            the maximum propagation window that cause a stringent spike detection on a
            root electrode to be counted as a stringent codetection. Defaults to 2.
        min_activity_hz (float, optional):
            Minimum activity rate of root detections (in Hz) for an electrode to be used as a root electrode. Defaults to 0.05 Hz.
        max_n_components_latency (int, optional):
            Maximum number of latency components for Gaussian mixture model used for splitting latency distribution. Defaults to 4.
        min_coc_n (int, optional):
            After splitting a cluster of codetections, a cluster is discarded if it
            does not have at least min_coc_n codetections. Defaults to 10.
        min_coc_p (int, optional):
            After splitting a cluster of codetections, a cluster is discarded if it
            does not have at least (min_coc_p * the total number of codetections before splitting)
            codetections. Defaults to 10.
        min_extend_comp_p (int, optional):
            The required percentage of codetections before splitting that is
            preserved after the split in order for the inner electrodes of the
            current splitting electrode to be added to the total list of electrodes
            used to further split the cluster. Defaults to 50.
        elec_patience (int, optional):
            Number of electrodes considered for splitting that do not lead to a
            split before terminating the splitting process. Defaults to 6.
        split_coc_clusters_amps (bool, optional):
            Whether to split clusters based on amplitude. Defaults to True.
        min_amp_dist_p (float, optional):
            The minimum Hartigan's dip test p-value for a distribution to be
            considered unimodal. Defaults to 0.1.
        max_n_components_amp (int, optional):
            Maximum number of componenst for Gaussian mixture model used for splitting amplitude distribution. Defaults to 4.
        min_loose_elec_prob (float, optional):
            Minimum average detection score (smaller values are set to 0) in decimal form (ranging from 0 to 1). Defaults to 0.03.
        min_inner_loose_detections (int, optional):
            Minimum inner loose electrode detections for assigning spikes / overlaps for merging. Defaults to 3.
        min_loose_detections_n (int, optional):
            Minimum loose electrode detections for assiging spikes / overlaps for merging. Defaults to 4.
        min_loose_detections_r_spikes (float, optional):
            Minimum ratio of loose electrode detections for assigning spikes. Defaults to 1/3.
        min_loose_detections_r_sequences (float, optional):
            Minimum ratio of loose electrode detections overlaps for merging. Defaults to 1/3.
        max_latency_diff_spikes (float, optional):
            Maximum allowed weighted latency difference for spike assignment. Defaults to 3.5.
        max_latency_diff_sequences (float, optional):
            Maximum allowed weighted latency difference for sequence merging. Defaults to 3.5.
        clip_latency_diff_factor (float, optional):
            Latency clip = clip_latency_diff_factor * max_latency_diff. Defaults to 2.
        max_amp_median_diff_spikes (float, optional):
            Maximum allowed weighted percent amplitude difference for spike assignment. Defaults to 0.65.
        max_amp_median_diff_sequences (float, optional):
            Maximum allowed weighted percent amplitude difference for sequence merging. Defaults to 0.65.
        clip_amp_median_diff_factor (float, optional):
            Amplitude clip = clip_amp_median_diff_factor * max_amp_median_diff. Defaults to 2.
        max_root_amp_median_std_spikes (float, optional):
            Maximum allowed root amplitude standard deviation for spike assignment. Defaults to 2.5.
        max_root_amp_median_std_sequences (float, optional):
            Maximum allowed root amplitude standard deviation for sequence merging. Defaults to infinity (not used).
        repeated_detection_overlap_time (float, optional):
            Time window (in seconds) for overlapping repeated detections. Defaults to 0.2 s.
        min_seq_spikes_n (int, optional):
            Minimum number of spikes required for a valid sequence. Defaults to 10.
        min_seq_spikes_hz (float, optional):
            Minimum spike rate for a valid sequence. Defaults to 0.05 Hz.
        relocate_root_min_amp (float, optional):
            Minimum amplitude ratio for relocating a root electrode before first merging. Defaults to 0.8.
        relocate_root_max_latency (float, optional):
            Maximum latency for relocating a root electrode before first merging. Defaults to -2.
        return_spikes (bool, optional):
            Whether to return spike times instead of an RTSort object. Defaults to False.
        delete_inter (bool, optional):
            Whether to delete the intermediate folder after processing. Defaults to False.
        device (str, optional):
            The device for PyTorch operations ("cuda" or "cpu"). Defaults to "cuda".
        num_processes (int, optional):
            Number of processes to use for parallelization. Defaults to None, which auto-selects the value based on the number of logical processors.
        ignore_warnings (bool, optional):
            Whether to suppress warnings during execution. Defaults to True.
        verbose (bool, optional):
            Whether to print detailed execution information. Defaults to True.
        debug (bool, optional):
            Whether to enable debugging features such as saving intermediate steps. Defaults to False.

    Returns:
        RTSort or np.ndarray:
            If `return_spikes` is False, an RTSort object containing the detected sequences is returned.
            If `return_spikes` is True, a NumPy array of spike times is returned.
    """

    with warnings.catch_warnings():
        if ignore_warnings:
            warnings.filterwarnings("ignore")
        else:
            warnings.filterwarnings("default")

        recording = load_recording(recording)

        try:
            chan_ids = [int(i) for i in recording.get_channel_ids()]
            chan_ids = None if chan_ids == list(range(len(chan_ids))) else chan_ids
        except ValueError:
            chan_ids = None

        if detection_model is None:
            detection_model = ModelSpikeSorter.load_mea()
        elif not isinstance(
            detection_model, ModelSpikeSorter
        ):  # detection model is a path
            detection_model = ModelSpikeSorter.load(detection_model)

        # Set up paths
        inter_path = model_inter_path = Path(inter_path)
        # model_inter_path = inter_path / "dl_model"
        model_inter_path.mkdir(exist_ok=True, parents=True)
        model_traces_path = model_inter_path / "model_traces.npy"
        model_outputs_path = model_inter_path / "model_outputs.npy"

        # Set up which part of recording to detect sequences in
        if recording_window_ms is None:
            recording_window_ms = (0, recording.get_total_duration() * 1000)
        elif isinstance(recording_window_ms, int) or isinstance(
            recording_window_ms, float
        ):  # recording_window_ms is in minutes
            rec_duration = recording.get_total_duration() * 1000
            recording_window_ms = (
                rec_duration - recording_window_ms * 60 * 1000,
                rec_duration,
            )
        detection_window_duration_s = (
            recording_window_ms[1] - recording_window_ms[0]
        ) / 1000  # In seconds

        # Save traces for detection model
        if num_processes is None:
            num_processes = max(1, round(os.cpu_count() * 2 / 3))
            num_processes_save_traces = max(1, os.cpu_count() // 3)
        else:
            num_processes_save_traces = num_processes

        if debug and verbose and (inter_path / "scaled_traces.npy").exists():
            print(
                "Skipping saving scaled traces because file scaled_traces.npy already exists and debug=True"
            )
        else:
            save_traces(
                recording,
                inter_path,
                *recording_window_ms,
                num_processes=num_processes_save_traces,
                verbose=verbose,
                dtype="float16",
            )

        # Run detection model
        samp_freq = round(recording.get_sampling_frequency() / 1000)  # kHz
        pre_median_frames = round(pre_median_ms * samp_freq)

        if debug and verbose and (model_inter_path / "model_outputs.npy").exists():
            print(
                "Skipping running detection model because file model_outputs.npy already exists and debug=True"
            )
        else:
            run_detection_model(
                recording,
                detection_model,
                inter_path / "scaled_traces.npy",
                model_traces_path,
                model_outputs_path,
                inference_scaling_numerator,
                pre_median_frames,
                model_inter_path,
                device,
                verbose,
            )

        # Set up params for detecting sequences
        samp_freq = round(recording.get_sampling_frequency() / 1000)  # kHz
        num_elecs = recording.get_num_channels()
        params = {
            "samp_freq": samp_freq,
            "elec_locs": recording.get_channel_locations(),
            "chan_ids": chan_ids,
            "model_inter_path": model_inter_path,
            "stringent_thresh": stringent_thresh,
            "loose_thresh": loose_thresh,
            "inference_scaling_numerator": inference_scaling_numerator,
            "front_buffer": detection_model.buffer_front_sample,
            "n_before": round(ms_before * samp_freq),
            "n_after": round(ms_after * samp_freq),
            "pre_median_frames": pre_median_frames,
            "inner_radius": inner_radius,
            "outer_radius": outer_radius,
            "min_elecs_for_array_noise": max(
                min_elecs_for_array_noise_n,
                round(min_elecs_for_array_noise_f * num_elecs),
            ),
            "min_elecs_for_seq_noise": max(
                min_elecs_for_seq_noise_n, round(min_elecs_for_seq_noise_f * num_elecs)
            ),
            "min_activity_root_cocs": min_activity_root_cocs,
            "min_activity": min_activity_hz * detection_window_duration_s,
            "max_n_components_latency": max_n_components_latency,
            "min_coc_n": min_coc_n,
            "min_coc_p": min_coc_p,
            "min_extend_comp_p": min_extend_comp_p,
            "elec_patience": elec_patience,
            "split_coc_clusters_amps": split_coc_clusters_amps,
            "min_amp_dist_p": min_amp_dist_p,
            "max_n_components_amp": max_n_components_amp,
            "min_loose_elec_prob": min_loose_elec_prob,
            "min_inner_loose_detections": min_inner_loose_detections,
            "min_loose_detections_n": min_loose_detections_n,
            "min_loose_detections_r_spikes": min_loose_detections_r_spikes,
            "min_loose_detections_r_sequences": min_loose_detections_r_sequences,
            "max_latency_diff_spikes": max_latency_diff_spikes,
            "max_latency_diff_sequences": max_latency_diff_sequences,
            "clip_latency_diff_factor": clip_latency_diff_factor,
            "max_amp_median_diff_spikes": max_amp_median_diff_spikes,
            "max_amp_median_diff_sequences": max_amp_median_diff_sequences,
            "clip_amp_median_diff_factor": clip_amp_median_diff_factor,
            "max_root_amp_median_std_spikes": max_root_amp_median_std_spikes,
            "max_root_amp_median_std_sequences": max_root_amp_median_std_sequences,
            "repeated_detection_overlap_time": repeated_detection_overlap_time,
            "min_seq_spikes": max(
                min_seq_spikes_n, min_seq_spikes_hz * detection_window_duration_s
            ),
            "relocate_root_min_amp": relocate_root_min_amp,
            "relocate_root_max_latency": relocate_root_max_latency,
            "device": device,
            "num_processes": num_processes,
            "ignore_warnings": ignore_warnings,
            "verbose": verbose,
            "debug": debug,
        }
        if debug:
            pickle_dump(params, inter_path / "params.pickle")

        # Actually detect sequences now

        # Form prelim prop seqs
        if debug and (inter_path / "all_clusters.pickle").exists():
            if verbose:
                print(
                    "Skipping detecting preliminary propagation sequences because file all_clusters.pickle already exists and debug=True"
                )
            all_clusters = pickle_load(inter_path / "all_clusters.pickle")
        else:
            all_clusters = form_all_clusters(params)
            all_clusters = setup_coc_clusters_parallel(all_clusters, params)
            if debug:
                pickle_dump(all_clusters, inter_path / "all_clusters.pickle")
        if len(all_clusters) == 0:
            if return_spikes:
                return []
            else:
                return None

        # Reassign spikes
        if debug and (inter_path / "all_clusters_reassigned.pickle").exists():
            if verbose:
                print(
                    "Skipping reassigning spikes because file all_clusters_reassigned.pickle already exists and debug=True"
                )
            all_clusters_reassigned = pickle_load(
                inter_path / "all_clusters_reassigned.pickle"
            )
        else:
            all_clusters_reassigned = reassign_spikes_to_clusters(
                all_clusters, detection_model, params
            )
            if debug:
                pickle_dump(
                    all_clusters_reassigned,
                    inter_path / "all_clusters_reassigned.pickle",
                )
        if len(all_clusters_reassigned) == 0:
            if verbose:
                print(
                    "0 preliminary propagation sequences remain after reassigning spikes and filtering"
                )
            if return_spikes:
                return []
            else:
                return None

        # Merging (happens very quickly, so don't need to dump if debug=True)
        intra_merged_clusters = intra_merge(all_clusters_reassigned, params)
        if (
            len(intra_merged_clusters) == 0
        ):  # It should't be possible for this to be True, but just in case
            if verbose:
                print("0 sequences remain first merging")
            if return_spikes:
                return []
            else:
                return None

        inter_merged_clusters = inter_merge(intra_merged_clusters, params)
        if (
            len(inter_merged_clusters) == 0
        ):  # It should't be possible for this to be True, but just in case
            if verbose:
                print("0 sequences remain second merging")
            if return_spikes:
                return []
            else:
                return None
        if debug:
            pickle_dump(
                inter_merged_clusters, inter_path / "inter_merged_clusters.pickle"
            )

        # Create RTSort object
        rt_sort = RTSort(inter_merged_clusters, detection_model, params)

        if return_spikes:  # Reassign spikes
            if verbose:
                print("Final spike assignment for final spike times:")
            all_spike_trains = rt_sort.sort_offline(
                inter_path / "scaled_traces.npy", verbose=verbose
            )  # NOTE: These spikes may be a few frames offset from the actual peak on the root electrode when compared to model_traces.npy and model_outputs.npy because different windows are used --> will cause problems if doing any setup_cluster
            # return_value = np.array([
            #     (spike_train, root_elec) for spike_train, root_elec in zip(all_spike_trains, rt_sort.get_seq_root_elecs())
            # ], dtype=object)
            # returns a NumPy array of shape (num_sequences,) where each element represents a detected sequence’s data (a tuple of length 2). The first element in the tuple is a NumPy array containing the sequence’s assigned spikes (in milliseconds). The second element is the channel index of the sequence’s root electrode.
            return_value = all_spike_trains
        else:
            return_value = rt_sort

        if delete_inter:
            shutil.rmtree(inter_path)
        else:
            rt_sort.save(inter_path / "rt_sort.pickle")

        return return_value


class RTSort:
    def __init__(
        self,
        sequences,
        model,
        params: dict,
        buffer_size=100,
        device="cuda",
        dtype=torch.float16,
    ):
        self.samp_freq = samp_freq = params["samp_freq"]
        elec_locs = params["elec_locs"]
        self.chan_ids = params.get("chan_ids", None)  # None for backwards compatibility
        self.max_chan_id = max(self.chan_ids) if self.chan_ids is not None else None
        model_inter_path = params["model_inter_path"]
        stringent_thresh = params["stringent_thresh"]
        loose_thresh = params["loose_thresh"]
        inference_scaling_numerator = params["inference_scaling_numerator"]
        n_before = params["n_before"]
        n_after = params["n_after"]
        pre_median_frames = params["pre_median_frames"]
        inner_radius = params["inner_radius"]
        min_elecs_for_array_noise = params["min_elecs_for_array_noise"]
        min_inner_loose_detections = params["min_inner_loose_detections"]
        max_latency_diff_spikes = params["max_latency_diff_spikes"]
        clip_latency_diff_factor = params["clip_latency_diff_factor"]
        max_amp_median_diff_spikes = params["max_amp_median_diff_spikes"]
        clip_amp_median_diff_factor = params["clip_amp_median_diff_factor"]
        max_root_amp_median_std_spikes = params["max_root_amp_median_std_spikes"]
        repeated_detection_overlap_time = params["repeated_detection_overlap_time"]

        # self.sequences = sequences  # This requires the sequence class to be importable and picklable --> not ideal
        self._seq_root_elecs = [
            seq.root_elec for seq in sequences
        ]  # For posterity, used to retrieve sequence data, NOT used for sorting (like self.seq_root_elecs)
        self.seq_spike_trains = [seq.spike_train for seq in sequences]  # For posterity
        self.seq_comp_elecs = [
            seq.comp_elecs for seq in sequences
        ]  # For select_seqs method
        self.seq_locs = np.array([elec_locs[seq.root_elec] for seq in sequences])

        if buffer_size != 100:
            raise NotImplementedError(
                "If buffer_size < 100, then need to account for current forward pass detecting spikes that were detected in previous pass (for running_sort())"
            )
        assert (
            buffer_size <= 100
        ), "If buffer_size > 100, some spikes will be missed in between forward passes (for running_sort())"
        self.buffer_size = buffer_size  # Frames
        self.device = device
        self.dtype = dtype

        torch.backends.cudnn.benchmark = True

        if device == "cuda":
            if (model_inter_path / ModelSpikeSorter.compiled_name).exists():
                self.model = ModelSpikeSorter.load_compiled(model_inter_path)
            else:
                self.model = model.compile(len(elec_locs), None)
        else:
            self.model = model.model.conv.to(device)
        self.front_buffer = model.buffer_front_sample
        self.end_buffer = model.buffer_end_sample

        # region Setup tensors
        num_seqs = len(sequences)

        all_comp_elecs = set()
        seq_no_overlap_mask = torch.full(
            (num_seqs, num_seqs), 0, dtype=torch.bool, device=device
        )
        for a, seq in enumerate(sequences):
            all_comp_elecs.update(seq.comp_elecs)
            for b, seq_b in enumerate(sequences):
                if (
                    calc_dist(*elec_locs[seq.root_elec], *elec_locs[seq_b.root_elec])
                    > inner_radius
                ):
                    seq_no_overlap_mask[a, b] = 1
        self.seq_no_overlap_mask = seq_no_overlap_mask

        all_comp_elecs = list(all_comp_elecs)
        seq_n_before = n_before
        seq_n_after = n_after
        self.spike_arange = torch.arange(
            0, seq_n_before + seq_n_after + 1, device=device
        )

        seqs_root_elecs = set()
        seqs_root_elecs_rel_comp_elecs = []
        seqs_inner_loose_elecs = []
        seqs_min_loose_elecs = []
        seqs_loose_elecs = []
        seqs_latencies = []
        seqs_latency_weights = []
        seqs_amps = []
        seqs_root_amp_means = []
        seqs_root_amp_stds = []
        seqs_amp_weights = []
        for seq in sequences:
            seqs_root_elecs.add(seq.root_elec)
            seqs_root_elecs_rel_comp_elecs.append(all_comp_elecs.index(seq.root_elec))

            # Binary arrays (1 for comp_elec in inner_loose_elecs/loose_elecs, else 0)
            seqs_inner_loose_elecs.append(
                torch.tensor(
                    [
                        1 if elec in seq.inner_loose_elecs else 0
                        for elec in all_comp_elecs
                    ],
                    dtype=torch.bool,
                    device=device,
                )
            )
            seqs_loose_elecs.append(
                torch.tensor(
                    [1 if elec in seq.loose_elecs else 0 for elec in all_comp_elecs],
                    dtype=torch.bool,
                    device=device,
                )
            )

            seqs_min_loose_elecs.append(ceil(seq.min_loose_detections))

            seq_comp_elecs = set(seq.comp_elecs)
            seqs_latencies.append(
                torch.tensor(
                    [
                        (
                            seq.all_latencies[elec] + seq_n_before
                            if elec in seq_comp_elecs and elec != seq.root_elec
                            else 0
                        )
                        for elec in all_comp_elecs
                    ],
                    dtype=dtype,
                    device=device,
                )
            )
            # Needs to be 1 to prevent divide by zero
            seqs_amps.append(
                torch.tensor(
                    [
                        seq.all_amp_medians[elec] if elec in seq_comp_elecs else 1
                        for elec in all_comp_elecs
                    ],
                    dtype=dtype,
                    device=device,
                )
            )

            elec_probs = torch.tensor(
                [
                    (
                        seq.all_elec_probs[elec]
                        if elec in seq_comp_elecs and elec != seq.root_elec
                        else 0
                    )
                    for elec in all_comp_elecs
                ],
                dtype=dtype,
                device=device,
            )
            seqs_latency_weights.append(elec_probs / torch.sum(elec_probs))

            elec_probs = torch.tensor(
                [
                    seq.all_elec_probs[elec] if elec in seq_comp_elecs else 0
                    for elec in all_comp_elecs
                ],
                dtype=dtype,
                device=device,
            )
            seqs_amp_weights.append(elec_probs / torch.sum(elec_probs))

            seqs_root_amp_means.append(seq.all_amp_medians[seq.root_elec])
            seqs_root_amp_stds.append(seq.root_to_amp_median_std[seq.root_elec])

        self.comp_elecs = torch.tensor(all_comp_elecs, dtype=torch.long, device=device)[
            :, None
        ]
        self.comp_elecs_flattened = self.comp_elecs.flatten()

        self.seqs_root_elecs = list(seqs_root_elecs)

        self.seqs_inner_loose_elecs = torch.vstack(seqs_inner_loose_elecs)
        self.seqs_loose_elecs = torch.vstack(seqs_loose_elecs)
        self.seqs_min_loose_elecs = torch.tensor(
            seqs_min_loose_elecs, dtype=torch.int16, device=device
        )
        self.seqs_latencies = torch.vstack(seqs_latencies)
        self.seqs_amps = torch.vstack(seqs_amps)
        self.seqs_latency_weights = torch.vstack(seqs_latency_weights)
        self.seqs_amp_weights = torch.vstack(seqs_amp_weights)
        self.seqs_root_amp_means = torch.tensor(
            seqs_root_amp_means, dtype=dtype, device=device
        )
        self.seqs_root_amp_stds = torch.tensor(
            seqs_root_amp_stds, dtype=dtype, device=device
        )

        self.seqs_root_elecs_rel_comp_elecs = seqs_root_elecs_rel_comp_elecs

        self.max_pool = torch.nn.MaxPool1d(
            seq_n_before + seq_n_after + 1, return_indices=True
        )

        # step_size = OUTPUT_WINDOW_HALF_SIZE * 2 - seq_n_after - seq_n_before
        # endregion

        # These variables need to be cached when call self.sort() multiple times
        self.overlap = round(repeated_detection_overlap_time * samp_freq)
        self.last_pre_median_output_frame = -np.inf

        self.input_scale = model.input_scale
        # self.inference_scaling_numerator = INFERENCE_SCALING_NUMERATOR
        # self.inference_scaling = None
        # Calculating this at start of experiment causes up to first second data to be lagged/frames missing
        # if INFERENCE_SCALING_NUMERATOR is not None:
        #     iqrs = scipy.stats.iqr(self.pre_median_frames, axis=1)
        #     median = np.median(iqrs)
        #     self.inference_scaling = self.inference_scaling_numerator / median
        # else:
        #     self.inference_scaling = 1  # self.inference_scaling=1 could be in __init__ BUT then another variable would be needed to wait for the first PRE_MEDIAN_FRAMES frames. Setting input_scale like this means another variable does not need to be set
        if inference_scaling_numerator is not None:
            window = np.load(model_inter_path / "scaled_traces.npy", mmap_mode="r")[
                :, :pre_median_frames
            ]
            iqrs = scipy.stats.iqr(window, axis=1)
            median = np.median(iqrs)
            inference_scaling = inference_scaling_numerator / median
            self.input_scale *= inference_scaling

        self.last_detections = torch.full(
            (num_seqs,), -pre_median_frames, dtype=torch.int64, device=device
        )

        # self.input_chunk = np.full((NUM_ELECS, model.sample_size), np.nan, dtype="float16")
        self.input_size = model.sample_size
        self.model_num_output_locs = model.num_output_locs

        self.pre_median_frames = torch.full(
            (elec_locs.shape[0], pre_median_frames),
            torch.nan,
            dtype=dtype,
            device=device,
        )
        self.pre_medians = None  # Used to divide traces
        self.cur_num_pre_median_frames = 0
        self.total_num_pre_median_frames = pre_median_frames

        # self.pre_sub_medians = None # torch.full((NUM_ELECS, 1), 0, dtype=dtype, device=device)  # Used to subtract traces
        # self.cur_num_pre_sub_median_frames = 0
        # self.total_num_pre_sub_median_frames = model.sample_size  # Results are different (probably worse) if use different sized chunks

        self.num_seqs = len(sequences)
        self.num_elecs = elec_locs.shape[0]
        self.seq_n_before = seq_n_before
        self.seq_n_after = seq_n_after
        self.stringent_thresh_logit = sigmoid_inverse(stringent_thresh)
        self.loose_thresh_logit = sigmoid_inverse(loose_thresh)
        self.min_elecs_for_array_noise = min_elecs_for_array_noise
        self.min_inner_loose_detections = min_inner_loose_detections
        self.max_latency_diff_spikes = max_latency_diff_spikes
        self.clip_latency_diff = max_latency_diff_spikes * clip_latency_diff_factor
        self.max_root_amp_median_std_spikes = max_root_amp_median_std_spikes
        self.max_amp_median_diff_spikes = max_amp_median_diff_spikes
        self.clip_amp_median_diff = (
            max_amp_median_diff_spikes * clip_amp_median_diff_factor
        )

        self.latest_frame = 0  # Spike times returned from self.sort() will be relative to the first frame of the first chunk received by RT-Sort
        self.ignore_spikes_before_minuend = (
            self.input_size - self.end_buffer - self.seq_n_after
        )  # ignore_spikes_before = self.ignore_spikes_before_minuend - num_new_frames

        self.full_front_buffer = self.front_buffer + self.seq_n_before
        self.full_end_buffer = self.end_buffer + self.seq_n_after

    def reset(self):
        """
        Resets the internal state to prepare for a new experiment.

        This method should be called before starting a new sorting session,
        especially if the real-time data stream has paused or stopped for more than 50ms.

        Args:
            None

        Returns:
            None
        """

        self.latest_frame = 0
        self.cur_num_pre_median_frames = 0
        self.pre_medians = None
        self.pre_median_frames = torch.full_like(self.pre_median_frames, torch.nan)
        self.last_detections = torch.full_like(
            self.last_detections, -self.total_num_pre_median_frames
        )

    def running_sort(self, obs, model_chunk=None, latest_frame=None):
        """
        Sorts spikes in real time with incoming data while keeping track of past data for ongoing spike detection.

        Args:
            obs (numpy.ndarray):
                A NumPy array of shape (num_frames, num_electrodes) representing the most recent frames of data from the recording.
                - The first 50ms of data passed after the last `rt_sort.reset()` is needed to initialize the RTSort object, so no spikes will be sorted during this time.
                - `num_frames` must be at least 1, and can vary between calls to `running_sort()`. To minimize latency, `num_frames` should ideally match the time passed since the last call. For better spike removal, larger frame counts improve detection.
            model_chunk (numpy.ndarray, optional):
                A NumPy array of shape (num_electrodes, num_output_samples) containing model outputs if available. This is typically used for offline processing.
                - If None, `self.model` will compute the model outputs based on `obs`.
                - If provided, `obs` will not be median-subtracted because it is assumed to come from `model_traces.npy`.
            latest_frame (int, optional):
                The latest (most recent) frame number in the recording. If None, the frame count will be incremented internally based on `obs`.

        Returns:
            list:
                Each element is a tuple of length 2, containing the data for a sorted spike:
                - The 0th element is the ID number of the sequence the spike was assigned to.
                - The 1st element is the time the spike occurred (in milliseconds), based on the number of frames passed to `running_sort()` since the last call to `rt_sort.reset()`. The internal clock tracks the elapsed time based on the number of frames, assuming no breaks in data.
        """
        obs = np.asarray(obs)
        if self.chan_ids is not None and obs.shape[1] > self.max_chan_id:
            obs = obs[:, self.chan_ids]

        obs = torch.tensor(obs, device=self.device, dtype=self.dtype).T
        if model_chunk is not None:
            model_chunk = torch.tensor(
                model_chunk, device=self.device, dtype=self.dtype
            )
        else:
            obs -= torch.median(obs, dim=1, keepdim=True).values
        num_new_frames = obs.shape[1]

        # Update internal traces cache
        self.pre_median_frames[:, :-num_new_frames] = self.pre_median_frames[
            :, num_new_frames:
        ]

        self.pre_median_frames[:, -num_new_frames:] = obs

        if latest_frame is None:
            self.latest_frame += num_new_frames
        else:
            self.latest_frame = latest_frame

        self.cur_num_pre_median_frames += num_new_frames  # self.buffer_size
        if self.cur_num_pre_median_frames >= self.total_num_pre_median_frames:
            self.cur_num_pre_median_frames = 0
            self.pre_medians = self.calc_pre_medians(self.pre_median_frames)

        if self.pre_medians is None:
            return []

        traces_torch = self.pre_median_frames[:, -self.input_size :]
        return self.sort_chunk(
            traces_torch,
            torch_window=model_chunk,
            spike_times_frame_offset=self.latest_frame - self.input_size,
            ignore_spikes_before=self.ignore_spikes_before_minuend - num_new_frames,
        )

    def sort_offline(
        self,
        recording,
        inter_path=None,
        recording_window_ms=None,
        model_outputs=None,
        reset=True,
        return_spikeinterface_sorter=False,
        verbose=False,
    ):
        """
        Sorts spikes from a recording either from raw traces or precomputed model outputs.

        Args:
            recording (numpy.ndarray, str, pathlib.Path, or object):
                The input recording data, which can be:
                - A NumPy array of shape (n_channels, n_samples).
                - A path to a NumPy file (.npy) containing the scaled traces.
                - A SpikeInterface recording extractor object.
            inter_path (str or pathlib.Path, optional):
                Directory path to save `scaled_traces.npy` when the `recording` is a recording object.
                If None, the file will be saved to the current directory and deleted after sorting.
            recording_window_ms (tuple of (start_ms, end_ms), optional):
                A time window (in milliseconds) to sort part of the recording.
                If None, the entire recording is used.
            model_outputs (numpy.ndarray, str, or pathlib.Path, optional):
                Precomputed model outputs for the recording, either as:
                - A NumPy array of shape (n_channels, n_samples).
                - A path to a saved `.npy` file.
                If None, the model outputs will be computed using the provided `recording`.
            reset (bool, optional):
                Whether to reset the internal state of the sorter by calling `self.reset()`.
                Typically, this should be True (default).
            return_spikeinterface_sorter (bool, optional):
                If True, returns a spikeinterface.extractors.NumpySorting object containing the sorted results.
            verbose (bool, optional):
                If True, prints progress information during sorting. Default is False.

        Returns:
            numpy.ndarray or spikeinterface.extractors.NumpySorting:
                If `return_spikeinterface_sorter` is False, returns a NumPy array of shape (num_seqs,), where each element is a list containing the detected spike times (in milliseconds) for a given sequence.
                If `return_spikeinterface_sorter` is True, returns a spikeinterface.extractors.NumpySorting object containing the sorted results.
        """

        if reset:
            self.reset()

        # Set up recording
        og_recording = recording
        remove_traces = False
        if isinstance(recording, np.ndarray):
            scaled_traces = recording
        elif (isinstance(recording, str) or isinstance(recording, Path)) and (
            str(recording).endswith(".npy")
        ):
            scaled_traces = np.load(recording, mmap_mode="r")
        else:
            if inter_path is None:
                inter_path = Path.cwd()
                remove_traces = True

            if recording_window_ms is None:
                recording_window_ms = (0, None)
            recording = save_traces(
                recording, inter_path, *recording_window_ms, verbose=verbose
            )
            scaled_traces = np.load(recording, mmap_mode="r")
            recording_window_ms = None  # Don't need to slice again
        if recording_window_ms is not None:
            scaled_traces = scaled_traces[
                :,
                round(recording_window_ms[0] * self.samp_freq) : round(
                    recording_window_ms[1] * self.samp_freq
                ),
            ]

        # Optionally set up model outputs
        if model_outputs is not None and str(model_outputs).endswith(".npy"):
            model_outputs = np.load(model_outputs, mmap_mode="r")

        # Start sorting
        all_start_frames = range(0, scaled_traces.shape[1], self.buffer_size)

        if verbose:
            print("Sorting recording")
            all_start_frames = tqdm(all_start_frames)

        all_detections = [[] for _ in range(self.num_seqs)]
        model_chunk = None
        for start_frame in all_start_frames:
            if model_outputs is not None:
                model_chunk_start_frame = (
                    start_frame + self.buffer_size - self.input_size
                )
                if model_chunk_start_frame >= 0:
                    model_chunk = model_outputs[
                        :,
                        model_chunk_start_frame : model_chunk_start_frame
                        + self.model_num_output_locs,
                    ]

            detections = self.running_sort(
                scaled_traces[:, start_frame : start_frame + self.buffer_size].T,
                model_chunk=model_chunk,
            )
            for seq_idx, spike_time in detections:
                all_detections[seq_idx].append(spike_time)

        # Can't use list comprehension because then if no sequences detect spikes, the array will have shape (num_seqs, 0)
        array_detections = np.empty(self.num_seqs, dtype=object)
        for seq_idx, detections in enumerate(all_detections):
            array_detections[seq_idx] = np.sort(detections)

        if remove_traces:
            os.remove(recording)

        if return_spikeinterface_sorter:
            times_list = []
            labels_list = []
            for seq_idx, detections in enumerate(array_detections):
                if len(detections) > 0:
                    times_list.extend(
                        np.round(detections * self.samp_freq).astype("int64")
                    )
                    labels_list.extend([seq_idx] * len(detections))
            ind_order = np.argsort(times_list)
            times = np.array(times_list)[ind_order]
            labels = np.array(labels_list)[ind_order]
            np_sorting = NumpySorting.from_samples_and_labels(
                [times],
                [labels],
                self.samp_freq * 1000,
                unit_ids=list(range(self.num_seqs)),
            )
            if isinstance(og_recording, BaseRecording):
                np_sorting.register_recording(og_recording)
            return np_sorting
        else:
            return array_detections

    def sort_chunk(
        self,
        chunk,
        torch_window=None,
        spike_times_frame_offset=0,
        ignore_spikes_before=0,
    ):
        """
        Sort one chunk of a recording

        Params
            chunk: torch.Tensor or np.ndarray
                Shape (num_elecs, num_frames)
            torch_window:
                See method :running_sort:, torch_window == model_chunk
            spike_times_frame_offset:
                Spike times will be given in ms relative to [start of chunk + spike_times_frame_offset)
            ignore_spikes_before
                Ignore spikes that occur before frame ignore_spikes_before (relative to :chunk:)
        """
        # Setup
        pre_medians = self.pre_medians
        # if pre_median_frames is None:
        #     pre_medians = self.pre_medians
        # else:
        #     if isinstance(pre_median_frames, np.ndarray):
        #         pre_median_frames = torch.tensor(pre_median_frames, device=self.device, dtype=self.dtype)
        #     pre_medians = self.calc_pre_medians(pre_median_frames)
        # if isinstance(chunk, np.ndarray):
        #     chunk = self.to_tensor(chunk)

        # Now do sorting
        detections = []

        if torch_window is None:
            # traces_torch = self.pre_median_frames[:, -self.input_size:]  # This is now :chunk:
            torch_window = self.model(chunk[:, None, :] * self.input_scale)[
                :, 0, :
            ]  # TODO: It might be faster if part of chunk passed to model in previous iteration (for running_sort) is cached, so model does not have to recompute that part

        rec_window = chunk[:, self.front_buffer : -self.end_buffer]

        # region Find peaks
        # start = perf_counter()
        window = torch_window[
            self.seqs_root_elecs, self.seq_n_before - 1 : -self.seq_n_after + 1
        ]
        main = window[:, 1:-1]
        greater_than_left = main > window[:, :-2]
        greater_than_right = main > window[:, 2:]
        peaks = greater_than_left & greater_than_right
        crosses = main >= self.stringent_thresh_logit
        peak_ind_flat = torch.nonzero(peaks & crosses, as_tuple=True)[1]
        # peak_ind_flat = peak_ind_flat[torch.sum(TORCH_OUTPUTS[:, output_start_frame+peak_ind_flat+seq_n_before] >= LOOSE_THRESH_LOGIT, dim=0) <= MIN_ELECS_FOR_ARRAY_NOISE]  # may be slow
        peak_ind_flat = peak_ind_flat[
            torch.sum(
                torch_window[:, peak_ind_flat + self.seq_n_before]
                >= self.loose_thresh_logit,
                dim=0,
            )
            <= self.min_elecs_for_array_noise
        ]  # may be slow
        # if torch.numel(peak_ind_flat) == 0:  # This is very rare (only happend once for 1/16/24 spikeinterface simulated recording)
        #     # end = perf_counter()
        #     # delays_find_peak.append((end-start)*1000)
        #     # delays_total.append(delays_find_peak[-1])
        #     # delays_total_spike_detected.append(False)
        #     continue
        # end = perf_counter()
        # delays_find_peak.append((end-start)*1000)
        # endregion

        # # Plot peak waveform and detection footprints
        # peak=peak_ind_flat[0].item()
        # elec=seqs_root_elecs[0]
        # ##
        # frame = peak + seq_n_before + rec_start_frame
        # ms = frame / SAMP_FREQ
        # plot_spikes([ms], elec)
        # plt.show()

        # start = perf_counter()
        # Relative to output_start_frame+seq_n_before
        peak_ind = peak_ind_flat[:, None, None]
        spike_window = torch_window[self.comp_elecs, peak_ind + self.spike_arange]

        # Latencies are relative to peak-seq_n_before
        elec_probs, latencies = self.max_pool(spike_window)
        elec_crosses = (elec_probs >= self.loose_thresh_logit).transpose(1, 2)
        num_inner_loose = torch.sum(elec_crosses & self.seqs_inner_loose_elecs, dim=2)
        pass_inner_loose = num_inner_loose >= self.min_inner_loose_detections

        num_loose = torch.sum(elec_crosses & self.seqs_loose_elecs, dim=2)
        pass_loose = num_loose >= self.seqs_min_loose_elecs
        # end = perf_counter()
        # delays_elec_cross.append((end-start)*1000)

        # Slower to check this since so many DL detections
        # can_spike = pass_inner_loose & pass_loose
        # if not torch.any(can_spike):
        #     continue

        # start = perf_counter()
        latencies_float = latencies.transpose(1, 2).to(self.dtype)
        latency_diff = torch.abs(latencies_float - self.seqs_latencies)
        latency_diff = torch.clip(latency_diff, min=None, max=self.clip_latency_diff)
        latency_diff = torch.sum(latency_diff * self.seqs_latency_weights, axis=2)
        pass_latency = latency_diff <= self.max_latency_diff_spikes
        # end = perf_counter()
        # delays_latency.append((end-start)*1000)

        # # Getting rec_window was moved up in code to not include it in sorting computation time
        # # start = perf_counter()
        # # rec_start_frame = output_start_frame + FRONT_BUFFER
        # rec_start_frame = (output_frame - OUTPUT_WINDOW_HALF_SIZE) + FRONT_BUFFER
        # # amps = torch.abs(TORCH_TRACES[comp_elecs, rec_start_frame + peak_ind + latencies].transpose(1, 2)) / pre_medians  # peak_ind+seq_n_before = peak_ind_in_rec_window. latency-seq_n_before = latency_rel_peak_ind. --> (peak_ind+seq_n_before)+(latency-seq_n_before) = peak_ind+latency = spike index in rec_window
        # rec_window = torch.tensor(TRACES[:, rec_start_frame:rec_start_frame+OUTPUT_WINDOW_HALF_SIZE*2], dtype=dtype, device=device)
        # peak_ind+seq_n_before = peak_ind_in_rec_window. latency-seq_n_before = latency_rel_peak_ind. --> (peak_ind+seq_n_before)+(latency-seq_n_before) = peak_ind+latency = spike index in rec_window
        amps = (
            torch.abs(rec_window[self.comp_elecs, peak_ind + latencies].transpose(1, 2))
            / pre_medians
        )

        root_amp_z = (
            torch.abs(
                amps[:, 0, self.seqs_root_elecs_rel_comp_elecs]
                - self.seqs_root_amp_means
            )
            / self.seqs_root_amp_stds
        )
        pass_root_amp_z = root_amp_z <= self.max_root_amp_median_std_spikes
        # end = perf_counter()
        # delays_root_z.append((end-start)*1000)

        # start = perf_counter()
        amp_diff = torch.abs(amps - self.seqs_amps) / self.seqs_amps
        amp_diff = torch.clip(amp_diff, min=None, max=self.clip_amp_median_diff)
        amp_diff = torch.sum(amp_diff * self.seqs_amp_weights, axis=2)
        pass_amp_diff = amp_diff <= self.max_amp_median_diff_spikes
        # end = perf_counter()
        # delays_amp.append((end-start)*1000)

        # Since every seq is compared to every peak, this is needed (or a form of this) so (in extreme case) peak detected on one side of array is not assigned to seq on other side by coincidence
        strict_crosses_root = (
            spike_window[:, self.seqs_root_elecs_rel_comp_elecs, self.seq_n_before]
            >= self.stringent_thresh_logit
        )

        # start = perf_counter()
        can_spike = (
            strict_crosses_root
            & pass_inner_loose
            & pass_loose
            & pass_latency
            & pass_root_amp_z
            & pass_amp_diff
        )
        # end = perf_counter()
        # delays_can_spike.append((end-start)*1000)

        # Slighty faster than the following due to only slicing once: # spike_scores = latency_diff[can_spike] / MAX_LATENCY_DIFF_SPIKES + amp_diff[can_spike] / MAX_AMP_MEDIAN_DIFF_SPIKES  - num_loose[can_spike] / torch.sum(elec_crosses, dim=2)
        # start = perf_counter()
        spike_scores = (
            latency_diff / self.max_latency_diff_spikes
            + amp_diff / self.max_amp_median_diff_spikes
            - (num_loose / torch.sum(elec_crosses, dim=2) * 0.5)
        )
        spike_scores = 2.1 - spike_scores  # (additional 0.1 in case spike_scores=2)
        spike_scores *= can_spike

        # For debugging:
        # breakpoint: output_start_frame+seq_n_before <= rec_ms_to_output_frame(TIME) < output_frame+OUTPUT_WINDOW_HALF_SIZE-seq_n_after
        # time = (peak_ind.flatten()[2].item() + rec_start_frame + START_FRAME)/SAMP_FREQ
        # plot_spikes([time], all_sequences[20].root_elec)
        # plt.show()

        peak_ind_2d = peak_ind[:, 0]
        # next_can_spike = torch.full_like(can_spike, fill_value=1, dtype=torch.bool, device=device)
        # end = perf_counter()
        # cur_delay_split_spike = (end-start)*1000

        # cur_delay_assign_spike = 0
        # spike_detected = False  # TODO: This is only needed for speed testing
        while torch.any(spike_scores):
            # start = perf_counter()
            spike_seq_idx = torch.argmax(spike_scores).item()
            spike_idx = spike_seq_idx // self.num_seqs
            seq_idx = spike_seq_idx % self.num_seqs
            # TODO: If spike assignment is now slow, remove .item() (which gets the spike time in python value from torch tensor)
            offset_spike_time = peak_ind_flat[spike_idx].item()
            spike_time_in_chunk = (
                offset_spike_time + self.seq_n_before + self.front_buffer
            )

            # # Spike detected in previous window or sequence ISI violation (not real spike)
            # if spike_time_in_chunk < ignore_spikes_before or spike_time - self.last_detections[seq_idx] <= self.overlap:
            #     spike_scores[spike_idx, seq_idx] = 0
            #     continue
            # spike_time = (spike_time_in_chunk + spike_times_frame_offset) / self.samp_freq

            if spike_time_in_chunk >= ignore_spikes_before:
                spike_time = (
                    spike_time_in_chunk + spike_times_frame_offset
                ) / self.samp_freq
                detections.append((seq_idx, spike_time))
            # end = perf_counter()
            # cur_delay_assign_spike += ((end-start)*1000)

            # # start = perf_counter()
            # if spike_time - self.last_detections[seq_idx] <= self.overlap:
            #     spike_scores[spike_idx, seq_idx] = 0
            # # continue

            # Spike splitting for current window
            # set score to 0 if (seq is spatially close enough) and (peak is temporally close enough)
            # keep score if (seq is NOT spatially close enough) or (peak is temporally far enough)
            spike_scores *= self.seq_no_overlap_mask[seq_idx] | (
                torch.abs(peak_ind_2d - offset_spike_time) > self.overlap
            )
            # end = perf_counter()
            # cur_delay_split_spike += ((end-start)*1000)
            # spike_detected = True

            # Spike splitting for next window pseudocode:
            # FAST: just change seq_n_before for all seqs, but inaccurate
            # Slower: last_spike[~no_overlap_mask] = max(last_spike[~no_overlap_mask], spike_time)
            # self.last_detections[(~self.seq_no_overlap_mask) & (self.last_detections < spike_time)] = spike_time  # Slower version

            # delays_assign_spike.append(cur_delay_assign_spike)
            # delays_split_spike.append(cur_delay_split_spike)
            # delays_total.append(delays_find_peak[-1] + delays_elec_cross[-1] + delays_latency[-1] + delays_root_z[-1] + delays_amp[-1] + delays_can_spike[-1] + delays_assign_spike[-1] + delays_split_spike[-1])
            # delays_total_spike_detected.append(spike_detected)
            # torch.cuda.synchronize()  # Not needed because of while loop
            # end = perf_counter()
            # sorting_computation_times.append((end-start_sorting)*1000)
        return detections

    def select_seqs(self, seq_ind: list):
        """
        Selects sequences from the detected sequences to keep for online sorting

        Args:
            seq_ind (list):
                A list of sequence indices to keep.
                Note that sequences that are not selected may be included if they are within the inner radius of a selected sequence.
                This is needed for repeated detection removal.
                To account for this, the first len(seq_ind) sequences are the selected sequences, and the remaining sequences are
                the additional sequences needed for repeated detection removal.

        Returns:
            RTSort:
                An RTSort object with the selected sequences and additional sequences needed for repeated detection removal.
        """
        other = deepcopy(
            self
        )  # TODO: Rather than deepcopy (which can be slow), instantiate new RTSort object

        other.seq_ind_og = seq_ind[:]

        seq_ind_full = set(
            seq_ind
        )  # Sequences to keep and all sequences within inner radius of sequences to keep
        for seq_idx in seq_ind:
            overlap_seqs = (
                (self.seq_no_overlap_mask[seq_idx] == 0).nonzero().flatten().tolist()
            )
            seq_ind_full.update(overlap_seqs)
            # May need to recursively add more sequences if overlap_seqs have overlap_seqs so detected spikes perfectly match before select_seqs

        seq_ind_full = seq_ind + list(
            seq_ind_full.difference(seq_ind)
        )  # Reorder so that original seq ind are first
        seq_ind_full_2d_mask = torch.tensor(seq_ind_full)[:, None]

        other._seq_root_elecs = [
            self._seq_root_elecs[seq_idx] for seq_idx in seq_ind_full
        ]
        other.seq_spike_trains = [
            self.seq_spike_trains[seq_idx] for seq_idx in seq_ind_full
        ]
        other.seq_comp_elecs = [
            self.seq_comp_elecs[seq_idx] for seq_idx in seq_ind_full
        ]
        other.seq_locs = self.seq_locs[seq_ind_full]

        other.seq_no_overlap_mask = self.seq_no_overlap_mask[seq_ind_full, seq_ind_full]

        comp_elecs = set().union(*other.seq_comp_elecs)
        comp_elecs = torch.tensor(list(comp_elecs), device=self.device)
        comp_elecs_mask = torch.isin(self.comp_elecs_flattened, comp_elecs)

        other.comp_elecs = self.comp_elecs[comp_elecs_mask]
        other.comp_elecs_flattened = self.comp_elecs_flattened[comp_elecs_mask]

        other.seqs_root_elecs = list(set(other._seq_root_elecs))

        other.seqs_inner_loose_elecs = self.seqs_inner_loose_elecs[
            seq_ind_full_2d_mask, comp_elecs_mask
        ]
        other.seqs_loose_elecs = self.seqs_loose_elecs[
            seq_ind_full_2d_mask, comp_elecs_mask
        ]
        other.seqs_min_loose_elecs = self.seqs_min_loose_elecs[seq_ind_full]
        other.seqs_latencies = self.seqs_latencies[
            seq_ind_full_2d_mask, comp_elecs_mask
        ]
        other.seqs_amps = self.seqs_amps[seq_ind_full_2d_mask, comp_elecs_mask]
        other.seqs_latency_weights = self.seqs_latency_weights[
            seq_ind_full_2d_mask, comp_elecs_mask
        ]
        other.seqs_amp_weights = self.seqs_amp_weights[
            seq_ind_full_2d_mask, comp_elecs_mask
        ]
        other.seqs_root_amp_means = self.seqs_root_amp_means[seq_ind_full]
        other.seqs_root_amp_stds = self.seqs_root_amp_stds[seq_ind_full]

        other.seqs_root_elecs_rel_comp_elecs = [
            torch.where(other.comp_elecs_flattened == root_elec)[0].item()
            for root_elec in other._seq_root_elecs
        ]

        other.last_detections = self.last_detections[seq_ind_full]

        other.num_seqs = len(seq_ind_full)

        return other

    def calc_pre_medians(self, pre_median_frames: torch.Tensor):
        # if isinstance(pre_median_frames, np.ndarray):
        #     pre_median_frames = self.to_tensor(pre_median_frames)

        # Pytorch median different than numpy median: https://stackoverflow.com/a/54310996
        pre_medians = torch.median(
            torch.abs(pre_median_frames[self.comp_elecs_flattened]), dim=1
        ).values
        # a_min=0.5 to prevent dividing by zero when data is just 1s and 0s (median could equal 0)
        return torch.clip(pre_medians / 0.6745, min=0.5, max=None)

    def set_model(self, model, num_elecs=None, input_size=None):
        if not isinstance(model, ModelSpikeSorter):  # If detection_model is a path
            model = ModelSpikeSorter.load(model)
        if num_elecs is None:
            num_elecs = self.num_elecs
        else:
            raise NotImplementedError(
                "Can't change num_elecs without having to reinitialize RT-Sort object"
            )

        if input_size is None:
            input_size = self.input_size

        self.model = model.compile(num_elecs, input_size=input_size, device=self.device)
        self.input_size = input_size

    def to_tensor(self, data):
        return torch.tensor(data, dtype=self.dtype, device=self.device)

    def save(self, pickle_path):
        """
        Save the RTSort object to a pickle file.

        The compiled TensorRT/TorchScript model cannot be pickled, so it is
        excluded from the saved file. After loading the pickle in a new
        session, reattach the model before calling ``sort_offline``::

            import pickle
            from braindance.core.spikedetector.model import ModelSpikeSorter

            rt_sort = pickle.load(open(pickle_path, "rb"))
            rt_sort.model = ModelSpikeSorter.load_compiled(inter_path)
            rt_sort.sort_offline(recording)

        If the RTSort object was created via :func:`detect_sequences`, the
        compiled model is cached at ``inter_path / "compiled.ts"``.

        Args:
            pickle_path (str or pathlib.Path): Destination path for the pickle
                file. Parent directories are created if they do not exist.
        """
        self.reset()
        model = self.model
        self.model = None

        pickle_path = Path(pickle_path)
        pickle_path.parent.mkdir(exist_ok=True, parents=True)
        pickle_dump(self, pickle_path)
        self.model = model

    def save_seq_data(self, save_path):
        """
        As numpy array, save each sequence's (spike_train, root_elecrode)
        This is mainly for testing
        """

        save_data = [
            (spike_train, root_elec)
            for spike_train, root_elec in zip(
                self.seq_spike_trains, self.get_seq_root_elecs()
            )
        ]
        np.save(save_path, np.array(save_data, dtype=object))

    def get_seq_root_elecs(self):
        """
        self.seq_root_elecs is not in the correct order because this is only used for sorting
        """

        if hasattr(self, "_seq_root_elecs"):
            return self._seq_root_elecs
        else:  # For backwards compatibility
            all_comp_elecs = self.comp_elecs_flattened.cpu().numpy()
            root_elecs = [
                all_comp_elecs[elec] for elec in self.seqs_root_elecs_rel_comp_elecs
            ]
            return root_elecs

    def get_units(self):
        raise NotImplementedError
        from braindance.analysis.select_units import Unit

        return [
            Unit(root_elec, spike_train, idx)
            for idx, (spike_train, root_elec) in enumerate(
                zip(self.seq_spike_trains, self.get_seq_root_elecs())
            )
        ]

    def to_spikedata(self, all_seq_detections=None):
        """
        Params
            all_seq_detections
                If None, use self.seq_spike_trains (sequences' spikes after second merging but before final spike reassignment) to create SpikeData object
                If not None, should be the return value of method self.sort_offline
        """

        if all_seq_detections is None:
            all_seq_detections = self.seq_spike_trains

        return SpikeData(all_seq_detections, N=self.num_seqs)

    @staticmethod
    def load_from_file(pickle_path, model=None):
        """
        Need to specify model because cannot save model in .pickle

        Params
            model
                Can be a ModelSpikeSorter object, a path to one (str or Path), or None
                If None, self.model needs to be set later before the sorting functions work
        """
        rt_sort = pickle_load(pickle_path)
        if model is not None:
            rt_sort.set_model(model)
        else:
            rt_sort.model = None

        return rt_sort  # type: RTSort


# region Save traces and detection model outputs for fast access when detecting sequences
def save_traces(
    recording,
    inter_path,
    start_ms=0,
    end_ms=None,
    num_processes=None,
    dtype="float16",
    verbose=True,
):
    """
    For Maxwell recordings, iterating through traces for DL model can take about 3-4 times longer than recording duration
    because some channels contain meaningless data, so traces have to be sliced which makes loading data slow.
    This is resolved through parallel processing.
    """
    if verbose:
        print("Saving traces:")
    recording = load_recording(recording)

    if num_processes is None:
        # Conservative default — see _spawn_pool docstring for why
        # high process counts (the previous ``cpu_count() // 2``
        # default) were OOM-killing tight cgroup runs.
        num_processes = min(4, max(1, os.cpu_count() // 4))

    inter_path = Path(inter_path)
    inter_path.mkdir(exist_ok=True, parents=True)
    scaled_traces_path = inter_path / "scaled_traces.npy"
    if isinstance(recording, MaxwellRecordingExtractor):
        # Use h5py instead of spikeinterface to save Maxwell recording traces since h5py is much faster
        save_traces_mea(
            recording._kwargs["file_path"],
            scaled_traces_path,
            start_ms=start_ms,
            end_ms=end_ms,
            num_processes=num_processes,
            dtype=dtype,
            verbose=verbose,
        )
    else:
        save_traces_si(
            recording,
            scaled_traces_path,
            start_ms=start_ms,
            end_ms=end_ms,
            num_processes=num_processes,
            dtype=dtype,
            verbose=verbose,
        )
    return scaled_traces_path


def _is_in_memory_recording(recording) -> bool:
    """Heuristic: is this recording's data already materialised in RAM?

    True for a vanilla ``NumpyRecording`` (data lives in a numpy array
    held by the recording).  False for any disk-backed extractor
    (Maxwell, NWB, Binary, …) and for lazy preprocessing chains over
    them — those benefit from per-channel parallel disk reads.

    The check is by ``isinstance(NumpyRecording)`` because lazy chains
    over a NumpyRecording would re-run the chain per ``get_traces``
    call, which is no longer "free": let the parallel path handle them.
    """
    try:
        from spikeinterface.core import NumpyRecording

        return isinstance(recording, NumpyRecording)
    except Exception:
        return False


def save_traces_si(
    recording: BaseRecording,
    scaled_traces_path,
    start_ms=0,
    end_ms=None,
    num_processes=4,
    dtype="float16",
    verbose=True,
    chunk_seconds=10.0,
):
    # Save scaled traces (microvolts) for a spikeinterface recording

    samp_freq = recording.get_sampling_frequency() / 1000  # kHz
    num_elecs = recording.get_num_channels()

    start_frame = round(start_ms * samp_freq)

    if end_ms is None:
        end_frame = recording.get_total_samples()
    else:
        end_frame = round(end_ms * samp_freq)

    # Fast path for in-memory recordings: skip the multiprocessing pool
    # and the Manager.Namespace pickle round-trip entirely.  The pool
    # exists to parallelize per-channel disk reads on slow extractor
    # backends; for an in-memory NumpyRecording the data is already in
    # RAM, so multiprocessing only adds Manager pickling overhead — the
    # 5.6 GB Phase-2-style array would otherwise be serialized into a
    # separate proxy process and re-faulted across 16 forked workers,
    # producing the OOM-kill observed in the lowmem-rt-sort
    # investigation.  np.save here writes the array directly.
    if _is_in_memory_recording(recording):
        if verbose:
            print("In-memory recording detected — direct save (no pool)")
        traces = recording.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            return_scaled=recording.has_scaleable_traces(),
        )
        # NumpyRecording.get_traces returns (samples, channels); the
        # rest of save_traces consumers expect (channels, samples).
        traces = np.ascontiguousarray(traces.T.astype(dtype, copy=False))
        np.save(scaled_traces_path, traces)
        del traces
        return

    # Pre-allocate the output file on disk so the chunk loop writes
    # directly into mapped pages — no eager (num_elecs, n_samples)
    # RAM allocation.
    n_samples = end_frame - start_frame
    if verbose:
        print(
            f"Alllocating disk space for traces ({num_elecs} ch × "
            f"{n_samples} samples, {dtype})..."
        )
    out = np.lib.format.open_memmap(
        scaled_traces_path, mode="w+", dtype=dtype, shape=(num_elecs, n_samples)
    )

    # Time-chunked bulk extraction: read *all* channels for a single
    # time window per ``get_traces`` call.  For a filtered Maxwell
    # chain (or any lazy preprocessing chain), the filter evaluates
    # all channels in one vectorised pass per chunk — replacing 1018
    # full-duration filter passes (one per channel) with N_chunks
    # passes (e.g. 36 for a 180 s recording at 5 s / chunk).  Expected
    # speedup on filtered chains: 1 to 2 orders of magnitude.
    #
    # For unfiltered / raw extractors (Maxwell raw, Binary, NumpyRec)
    # this path is no slower than per-channel — ``get_traces`` is
    # cheap when there is no filter to rerun.
    fs_Hz = recording.get_sampling_frequency()
    chunk_samples = max(1, int(round(chunk_seconds * fs_Hz)))
    n_chunks = (n_samples + chunk_samples - 1) // chunk_samples
    if verbose:
        print(
            f"Extracting traces in time chunks ({chunk_seconds:.1f} s × "
            f"{n_chunks} chunks, all channels per chunk — one filter pass "
            f"per chunk)"
        )

    iterator = range(n_chunks)
    if verbose:
        iterator = tqdm(iterator, total=n_chunks)
    for i in iterator:
        chunk_out_start = i * chunk_samples
        chunk_out_end = min(chunk_out_start + chunk_samples, n_samples)
        chunk_rec_start = start_frame + chunk_out_start
        chunk_rec_end = start_frame + chunk_out_end
        traces_chunk = recording.get_traces(
            start_frame=chunk_rec_start,
            end_frame=chunk_rec_end,
            return_scaled=recording.has_scaleable_traces(),
        )  # (samples, channels)
        out[:, chunk_out_start:chunk_out_end] = traces_chunk.T.astype(dtype, copy=False)

    out.flush()
    del out


def _save_traces_si(task):
    config, start_frame, end_frame, channel_idx, save_path, dtype = task
    recording = config.recording
    traces = (
        recording.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=[recording.get_channel_ids()[channel_idx]],
            return_scaled=recording.has_scaleable_traces(),
        )
        .flatten()
        .astype(dtype)
    )
    saved_traces = np.load(save_path, mmap_mode="r+")
    saved_traces[channel_idx] = traces


def _save_traces_si_thread(task):
    """Thread-friendly version of _save_traces_si.

    The thread pool passes the recording directly (not via a
    Manager.Namespace proxy), so we unpack a slightly different tuple.
    Same I/O work; no Manager round-trip.
    """
    recording, start_frame, end_frame, channel_idx, save_path, dtype = task
    traces = (
        recording.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=[recording.get_channel_ids()[channel_idx]],
            return_scaled=recording.has_scaleable_traces(),
        )
        .flatten()
        .astype(dtype)
    )
    saved_traces = np.load(save_path, mmap_mode="r+")
    saved_traces[channel_idx] = traces


def save_traces_mea(
    rec_path,
    save_path,
    start_ms=0,
    end_ms=None,
    samp_freq=20,  # kHz
    default_gain=1,
    chunk_size=100000,
    num_processes=2,
    dtype="float16",
    verbose=True,
):
    """
    Can't save traces with spikeinterface get_traces() because it is really slow on MaxWell MEA recordings
    """

    rec_h5 = h5py.File(rec_path)
    rec_si = MaxwellRecordingExtractor(rec_path)

    start_frame = round(start_ms * samp_freq)

    if end_ms is None:
        end_frame = rec_si.get_total_samples()
    else:
        end_frame = round(end_ms * samp_freq)

    if "sig" in rec_h5:  # Old file format
        # chan_ind = []
        # for mapping in recording['mapping']:  # (chan_idx, elec_id, x_cord, y_cord)
        #     if mapping[1] != -1:
        #         chan_ind.append(mapping[0])
        # if 'lsb' in recording['settings']:
        #     gain = recording['settings']['lsb'][0] * 1e6
        # else:
        #     gain = default_gain
        #     if verbose:
        #         print(f"'lsb' not found in 'settings'. Setting gain to uV to {gain}")
        chan_ind = [
            int(chan_id) for chan_id in rec_si.get_channel_ids()
        ]  # This gives same result as recording['mapping] for-loop
        get_traces = _get_traces_mea_old
    else:
        # Check that h5py matches rec_si
        assert rec_h5["recordings"]["rec0000"]["well000"]["groups"]["routed"][
            "raw"
        ].shape == (
            rec_si.get_num_channels(),
            rec_si.get_total_samples(),
        ), "h5py file doesn't match what spikeinterface loads"
        chan_ind = list(range(rec_si.get_num_channels()))
        get_traces = _get_traces_mea_new
    if rec_si.has_scaleable_traces():
        gain = rec_si.get_channel_gains()
    else:
        gain = np.full_like(chan_ind, default_gain, dtype="float16")
        if verbose:
            print(f"Recording does not have channel gains. Setting gain to {gain}")
    gain = gain[:, None]

    # Allocate the .npy file directly on disk via open_memmap.  The
    # previous ``np.zeros + np.save + del`` pattern materialised a
    # full-sized eager array (~7-15 GB for a 1018-ch × 3-min recording)
    # before reassigning to disk — the same Maxwell-path version of the
    # fix Patch B applied to ``run_detection_model``.
    out_alloc = np.lib.format.open_memmap(
        save_path,
        mode="w+",
        dtype=dtype,
        shape=(len(chan_ind), end_frame - start_frame),
    )
    del out_alloc  # close the handle; workers reopen via mmap_mode="r+"

    # print("Extracting traces ...")
    tasks = [
        (
            rec_path,
            save_path,
            start_frame,
            chan_ind,
            chunk_start,
            chunk_size,
            gain,
            dtype,
            get_traces,
        )
        for chunk_start in range(start_frame, end_frame, chunk_size)
    ]

    # Threads — same rationale as save_traces_si.  h5py is the
    # workhorse here and releases the GIL during low-level reads, so
    # threads parallelize effectively without process overhead.
    imap = _thread_map(num_processes, _save_traces_mea, tasks)
    if verbose:
        imap = tqdm(imap, total=len(tasks))
    for _ in imap:
        pass


def _get_traces_mea_old(rec_path):
    # Used in save_traces_mea
    return h5py.File(rec_path, "r")["sig"]


def _get_traces_mea_new(rec_path):
    # Used in save_traces_mea
    return h5py.File(rec_path, "r")["recordings"]["rec0000"]["well000"]["groups"][
        "routed"
    ]["raw"]


def _save_traces_mea(task):
    (
        rec_path,
        save_path,
        start_frame,
        chan_ind,
        chunk_start,
        chunk_size,
        gain,
        dtype,
        get_traces,
    ) = task
    sig = get_traces(rec_path)
    traces = sig[chan_ind, chunk_start : chunk_start + chunk_size].astype(dtype) * gain
    saved_traces = np.load(save_path, mmap_mode="r+")
    saved_traces[
        :, chunk_start - start_frame : chunk_start - start_frame + traces.shape[1]
    ] = traces  # using traces.shape[1] in case chunk_start is within chunk_size of the end of the file (does not raise index error)


def run_detection_model(
    recording,
    model,
    scaled_traces_path,
    model_traces_path=None,
    model_outputs_path=None,
    inference_scaling_numerator=12.6,
    pre_median_frames=1000,
    model_inter_path=None,
    device="cuda",
    verbose=True,
):
    """
    WARNING: [Torch-TensorRT] - Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors
        - This is nothing unless using model on a different GPU than what created it (https://github.com/dusty-nv/jetson-inference/issues/883#issuecomment-754106437)

    Params
        model_traces_path
            If None, set to scaled_traces_path.parent / "model_traces.npy"
        model_outputs_path
            If None, set to scaled_traces_path.parent / "model_outputs.npy"
    """
    if verbose:
        print("Running detection model:")

    if model_traces_path is None:
        model_traces_path = Path(scaled_traces_path).parent / "model_traces.npy"
    if model_outputs_path is None:
        model_outputs_path = Path(scaled_traces_path).parent / "model_outputs.npy"

    torch.backends.cudnn.benchmark = True
    np_dtype = "float16"

    # region Load model
    # print("Loading DL model ...")
    # model = ModelSpikeSorter.load(model_path)
    sample_size = model.sample_size
    num_output_locs = model.num_output_locs
    input_scale = model.input_scale
    # Compile detection model
    if verbose:
        print(f"Compiling detection model for {recording.get_num_channels()} elecs ...")
    model_compiled = model.compile(
        recording.get_num_channels(), model_save_path=model_inter_path, device=device
    )
    # model_compiled = ModelSpikeSorter.load_compiled(model_inter_path)
    # endregion

    # region Prepare data
    scaled_traces = np.load(scaled_traces_path, mmap_mode="r")

    num_chans, rec_duration = scaled_traces.shape

    all_start_frames = list(
        range(0, rec_duration - sample_size + 1, num_output_locs)
    )  # Some frames at the end of the recording may not be included because they can not be a part of a window that 1) does not overlap with a previous window 2) is duration sample_size (10ms)

    if verbose:
        print("Allocating disk space to save model traces and outputs ...")
    # Allocate the .npy files directly on disk via open_memmap. The previous
    # np.zeros_like + np.save + np.load round-trip materialised two full-sized
    # arrays (traces and outputs) in RAM before reassigning to memmaps — for a
    # 1018-ch × 3-min × float16 recording that was ~14 GB of avoidable peak.
    traces_all = np.lib.format.open_memmap(
        model_traces_path, mode="w+", dtype=np_dtype, shape=scaled_traces.shape
    )
    outputs_all = np.lib.format.open_memmap(
        model_outputs_path,
        mode="w+",
        dtype=np_dtype,
        shape=(
            num_chans,
            rec_duration - model.buffer_end_sample - model.buffer_front_sample,
        ),
    )
    # endregion

    # region Calculating inference scaling
    if inference_scaling_numerator is not None:
        window = scaled_traces[:, :pre_median_frames]
        iqrs = scipy.stats.iqr(window, axis=1)
        median = np.median(iqrs)
        inference_scaling = inference_scaling_numerator / median
    else:
        inference_scaling = 1
    print(f"Inference scaling: {inference_scaling}")
    # endregion

    # region Run model
    print("Running model ...")
    with torch.no_grad():
        for start_frame in tqdm(all_start_frames):
            traces_torch = torch.tensor(
                scaled_traces[:, start_frame : start_frame + sample_size],
                device=device,
                dtype=torch.float16,
            )
            traces_torch -= torch.median(traces_torch, dim=1, keepdim=True).values
            outputs = model_compiled(
                traces_torch[:, None, :] * input_scale * inference_scaling
            ).cpu()

            traces_all[:, start_frame : start_frame + sample_size] = traces_torch.cpu()
            outputs_all[:, start_frame : start_frame + num_output_locs] = outputs[
                :, 0, :
            ]

    # Check if there is data remaining at end of recording that was not included in all_start_frames for-loop
    remaining_frames = rec_duration - (start_frame + sample_size)
    if remaining_frames > 0:
        traces_torch = torch.tensor(
            scaled_traces[:, -sample_size:], device=device, dtype=torch.float16
        )
        traces_torch -= torch.median(traces_torch, dim=1, keepdim=True).values
        with torch.no_grad():
            outputs = model(
                traces_torch[:, None, :] * input_scale * inference_scaling
            ).cpu()
        traces_all[:, -remaining_frames:] = traces_torch[:, -remaining_frames:].cpu()
        outputs_all[:, -remaining_frames:] = outputs[:, -remaining_frames:]

    # endregion

    # region Release model + memmap references for downstream phases
    # Drop the GPU model + Python references to the memmap views.  The
    # numpy memmap files on disk persist (downstream phases reopen them
    # via ``np.load(..., mmap_mode="r")``); releasing the in-process
    # references here lets the kernel reclaim the page-cache pages
    # held for THIS process between phases.  Combined with an explicit
    # ``gc.collect()`` and ``torch.cuda.empty_cache()`` this returns
    # several GB of host RAM + GPU VRAM before sequence detection
    # starts allocating its own per-sequence tensors.
    import gc as _gc

    del traces_all
    del outputs_all
    del scaled_traces
    del traces_torch
    del model_compiled
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _gc.collect()
    # endregion


# endregion

# region Form preliminary propagation sequences


class CocCluster:
    def __init__(self, root_elec, split_elecs, spike_train):
        # Form 1 initial cluster on root elec
        self.root_elec = root_elec
        self.root_elecs = [root_elec]
        self.split_elecs = split_elecs
        self._spike_train = spike_train
        # self.latencies = []

    def split(self, split_elec, spike_train):
        # Split cluster using split_elec
        return CocCluster(
            self.root_elec, self.split_elecs.union({split_elec}), spike_train
        )

    @property
    def spike_train(self):
        return np.sort(self._spike_train)


def form_coc_clusters(root_elec, params):
    samp_freq = params["samp_freq"]
    elec_locs = params["elec_locs"]
    model_inter_path = params["model_inter_path"]
    stringent_thresh = params["stringent_thresh"]
    loose_thresh = params["loose_thresh"]
    front_buffer = params["front_buffer"]
    n_before = params["n_before"]
    n_after = params["n_after"]
    pre_median_frames = params["pre_median_frames"]
    inner_radius = params["inner_radius"]
    outer_radius = params["outer_radius"]
    min_elecs_for_array_noise = params["min_elecs_for_array_noise"]
    min_activity_root_cocs = params["min_activity_root_cocs"]
    min_activity = params["min_activity"]
    min_coc_n = params["min_coc_n"]
    elec_patience = params["elec_patience"]
    split_coc_clusters_amps = params["split_coc_clusters_amps"]
    min_amp_dist_p = params["min_amp_dist_p"]
    max_n_components_amp = params["max_n_components_amp"]
    verbose = params["verbose"]

    # Setup
    root_elec_loc = elec_locs[root_elec]

    # This will give comp_elecs and max_amp_elecs in a different order than manuscript code (when multiple electrodes are equidistant) --> clusters are slightly different
    comp_elecs = {}
    max_amp_elecs = {}
    for elec, loc in enumerate(elec_locs):
        if elec == root_elec:
            continue
        dist = calc_dist(*root_elec_loc, *loc)
        if dist <= inner_radius:
            comp_elecs[elec] = dist
        if dist <= outer_radius:
            max_amp_elecs[elec] = dist
    comp_elecs = sorted(comp_elecs.keys(), key=lambda elec: comp_elecs[elec])
    max_amp_elecs = sorted(max_amp_elecs.keys(), key=lambda elec: max_amp_elecs[elec])
    if len(comp_elecs) == 0:
        return []

    min_coc_prob = sigmoid_inverse(stringent_thresh)
    loose_thresh_logit = sigmoid_inverse(loose_thresh)
    all_traces = np.load(model_inter_path / "model_traces.npy", mmap_mode="r")
    all_outputs = np.load(model_inter_path / "model_outputs.npy", mmap_mode="r")

    # Extract stringent threshold crossings
    # window = torch.tensor(all_outputs[root_elec, pre_median_frames-front_buffer:])  # Can't start processing until pre_median_frames-th index of traces
    window = all_outputs[
        root_elec, pre_median_frames - front_buffer : -n_after - 1
    ]  # Don't include spikes at the very end
    main = window[1:-1]
    greater_than_left = main > window[:-2]
    greater_than_right = main > window[2:]
    peaks = greater_than_left & greater_than_right
    crosses = main >= min_coc_prob
    # crossing_output_frames = torch.nonzero((peaks & crosses)).flatten().numpy() + pre_median_frames - front_buffer
    crossing_output_frames = (
        np.flatnonzero((peaks & crosses)) + pre_median_frames - front_buffer + 1
    )

    # Remove array-wide noise spikes
    all_noise_probs = all_outputs[:, crossing_output_frames]
    num_loose = np.sum(all_noise_probs >= loose_thresh_logit, axis=0)
    crossing_output_frames = crossing_output_frames[
        num_loose < min_elecs_for_array_noise
    ]

    # Remove spikes with too few cooccurrences
    output_windows = all_outputs[
        [[c] for c in comp_elecs],
        crossing_output_frames[:, None, None] + np.arange(-n_before, n_after + 1),
    ]  # (num_crossings, num_comp_elecs, num_frames)
    crossing_probs = np.max(output_windows, axis=2)
    valid_ind = np.any(crossing_probs >= min_coc_prob, axis=1)
    crossing_probs = crossing_probs[valid_ind]
    crossing_output_frames = crossing_output_frames[valid_ind]
    crossing_latencies = np.argmax(output_windows[valid_ind], axis=2) - n_before

    crossing_output_frames += front_buffer  # Convert to time in recording (so this is actually crossing_traces_frames, but not creating new variable to save memory)

    # Remove spikes that are not the largest amplitude
    traces_windows = all_traces[
        [[e] for e in [root_elec] + max_amp_elecs],
        crossing_output_frames[:, None, None]
        + np.arange(-pre_median_frames - n_before, n_after + 1),
    ]
    pre_medians = np.median(
        np.abs(traces_windows[:, :, :pre_median_frames]), axis=2
    )  # (num_crossings, num_max_amp_elecs)
    pre_medians = np.clip(pre_medians / 0.6745, a_min=0.5, a_max=None)
    root_amp_medians = (
        traces_windows[:, 0, pre_median_frames + n_before] / pre_medians[:, 0]
    )

    # Update min,max,abs usage (better accounts for postitive-peak spikes)
    # traces_windows = np.abs(traces_windows[:, 1:, pre_median_frames:])
    # max_amp_medians = np.max(traces_windows, axis=2)
    # max_amp_medians = np.max(max_amp_medians / pre_medians[:, 1:], axis=1)

    # Original min,max,abs usage
    traces_windows = traces_windows[:, 1:, pre_median_frames:]
    max_amp_medians = np.min(traces_windows, axis=2)
    max_amp_medians = np.max(np.abs(max_amp_medians) / pre_medians[:, 1:], axis=1)
    valid_ind = np.abs(root_amp_medians) >= max_amp_medians
    crossing_probs = crossing_probs[valid_ind]
    crossing_output_frames = crossing_output_frames[valid_ind]
    crossing_latencies = crossing_latencies[valid_ind]
    root_amp_medians = root_amp_medians[valid_ind]

    all_times = crossing_output_frames / samp_freq  # Convert to ms in recording

    if verbose:
        print(f"Starting with elec {root_elec}, loc: {elec_locs[root_elec]}")
        print("\nFinding coocurrences")
        # all_times = tqdm(all_times)

    coc_dict = {}  # root time to cocs [(elec, latency)]
    root_time_to_amp_median = {}
    num_activity_cocs = 0  # To see if elec contains activity

    for time, probs, latencies, root_amp_median in zip(
        all_times, crossing_probs, crossing_latencies, root_amp_medians
    ):
        # region Slower original version
        # for time in tqdm(all_times):
        #     rec_frame = round(time * samp_freq)

        #     # Check if time is largest amp/median NOTE: Amp here is defined as max value in traces. Amp in other areas is defined by location of DL prediction. (Probably doesn't make a difference since DL is pretty accurate. Also doing it differently here might be better since the max-amp threshold is slightly more stringent this way, which is better for forming sequence backbones)

        #     # Raw traces and mean and median of preceeding window
        #     start_frame = rec_frame - n_before
        #     pre_medians = calc_pre_median(start_frame, all_traces, pre_median_frames, [root_elec] + max_amp_elecs)

        #     root_pre_median = pre_medians[0]
        #     pre_medians = pre_medians[1:]
        #     # Use rec_frame here so its rec_frame-n_before:rec_frame+n_after+1
        #     traces = all_traces[max_amp_elecs, start_frame:rec_frame+n_after+1]
        #     amp_medians = np.abs(np.min(traces, axis=1)) / pre_medians

        #     root_amp_median = np.abs(all_traces[root_elec, rec_frame]) / root_pre_median
        #     if root_amp_median < np.max(amp_medians):
        #         continue

        #     # Check if not noise spike
        #     output_frame = rec_ms_to_output_frame(time, samp_freq, front_buffer)
        #     noise_probs = all_outputs[:, output_frame]
        #     if np.sum(noise_probs >= loose_thresh_logit) >= min_elecs_for_array_noise:
        #         continue

        #     output_frame = rec_ms_to_output_frame(time, samp_freq, front_buffer)
        #     # Check if time has enough a coac with comp_elecs
        #     cocs = []
        #     for elec in comp_elecs:
        #         # Check if elec coactivates
        #         output_window = all_outputs[elec, output_frame - n_before:output_frame+n_after+1]

        #         prob = np.max(output_window)
        #         if prob < min_coc_prob:
        #             continue

        #         # Add to coc_dict
        #         # (np.argmax(output_window) - n_before) / SAMP_FREQ  # root_elec detects spike at n_before
        #         latency = np.argmax(output_window) - n_before
        #         cocs.append((elec, time, latency))
        # endregion

        cocs = [
            (comp_elecs[elec], time, latencies[elec])
            for elec in np.flatnonzero(probs >= min_coc_prob)
        ]

        if len(cocs) > 0:
            coc_dict[time] = cocs
            root_time_to_amp_median[time] = root_amp_median
        if len(cocs) >= min_activity_root_cocs:
            num_activity_cocs += 1

    if len(root_time_to_amp_median) < min_coc_n or num_activity_cocs < min_activity:
        return []

    allowed_root_times = set(coc_dict.keys())
    if verbose:
        print(f"{len(allowed_root_times)} cocs total")

    all_coc_clusters = []
    root_cluster = CocCluster(root_elec, {root_elec}, [])
    # for allowed_root_times in amps_allowed_root_times:
    if verbose and len(allowed_root_times) > 1:
        print(f"-" * 50)
        print(f"Starting on amp/median group with {len(allowed_root_times)} cocs")
    # allowed_root_times = set(allowed_root_times)

    # patience_counter = 0
    # Compare root to each comp elec
    for c in range(len(comp_elecs)):
        if verbose:
            print(
                f"\nComparing to elec {comp_elecs[c]}, loc: {elec_locs[comp_elecs[c]]}"
            )

        # Grow tree on root-comp elec pair
        coc_clusters = branch_coc_cluster(
            root_cluster,
            comp_elecs[
                c:
            ],  # Elecs before c would have already been compared to root-comp elec pair
            coc_dict,
            allowed_root_times,
            Patience(root_elec, elec_patience, elec_locs),
            params,
        )
        for cluster in coc_clusters:
            allowed_root_times.difference_update(cluster._spike_train)
            all_coc_clusters.append(cluster)

        if verbose:
            print(f"Found {len(coc_clusters)} clusters")
            print(f"{len(allowed_root_times)} cocs remaining")

        if len(allowed_root_times) < min_coc_n:
            if verbose:
                print(f"\nEnding early because too few cocs remaining")
            break

    # region Split coc_clusters based on root amp medians
    if not split_coc_clusters_amps:
        if verbose:
            print(f"\nTotal: {len(all_coc_clusters)} clusters")

        return all_coc_clusters

    all_split_coc_clusters = []
    for i, cluster in enumerate(all_coc_clusters):
        if len(cluster._spike_train) < min_coc_n:
            continue

        root_amp_medians = np.array(
            [root_time_to_amp_median[time] for time in cluster._spike_train]
        )
        dip, pval = diptest(root_amp_medians)

        if pval >= min_amp_dist_p:  # root_amp_medians are unimodal
            all_split_coc_clusters.append(cluster)
            continue
        if verbose:
            print(f"\nCluster {i}: p-val={pval:.4f}")

        # Reshape to (n_samples, n_features)
        root_amp_medians = root_amp_medians.reshape(-1, 1)
        best_score = np.inf
        best_gmm = None
        for n_components in range(2, max_n_components_amp + 1):
            with _threadpool_limits(limits=1):
                gmm = GaussianMixture(
                    n_components=n_components,
                    random_state=1150,
                    n_init=1,
                    max_iter=1000,
                )
                gmm.fit(root_amp_medians)
                score = gmm.bic(root_amp_medians)
            if score < best_score:
                best_score = score
                best_gmm = gmm
            # print(n_components, score)

        # split() using cluster.root_elec so that no elec is added to splitting elecs
        split_coc_clusters = [
            cluster.split(cluster.root_elec, []) for _ in range(best_gmm.n_components)
        ]
        for label, time in zip(
            best_gmm.predict(root_amp_medians), cluster._spike_train
        ):
            split_coc_clusters[label]._spike_train.append(time)
        for cluster in split_coc_clusters:
            if len(cluster._spike_train) >= min_coc_n:
                all_split_coc_clusters.append(cluster)
        if verbose:
            print(f"Split cluster {i} into {len(split_coc_clusters)} clusters")
    # endregion

    if verbose:
        print(f"\nTotal: {len(all_split_coc_clusters)} clusters")

    return all_split_coc_clusters


def _form_all_clusters(task):
    root_elec, params = task
    params["verbose"] = False
    with warnings.catch_warnings():
        set_warning_level(params["ignore_warnings"])
        with _threadpool_limits(limits=1):
            coc_clusters = form_coc_clusters(root_elec, params)
            # setup_coc_clusters(coc_clusters, params)
            # coc_clusters = filter_clusters(coc_clusters, params)

    return coc_clusters


def form_all_clusters(params):
    """
    Form all preliminary propagation sequences (essentially, call form_coc_clusters() on all electrodes)
    """
    elec_locs = params["elec_locs"]
    num_processes = params["num_processes"]
    verbose = params["verbose"]

    if verbose:
        print("Detecting sequences")

    # np.random.seed(1150)
    all_clusters = []
    tasks = [(root_elec, params) for root_elec in range(elec_locs.shape[0])]
    with Pool(processes=num_processes) as pool:
        for clusters in tqdm(
            pool.imap_unordered(_form_all_clusters, tasks), total=len(tasks)
        ):
            all_clusters += clusters

    if verbose:
        print(f"Detected {len(all_clusters)} preliminary propagation sequences")
    return all_clusters


def branch_coc_cluster(
    root_cluster, comp_elecs, coc_dict, allowed_root_times, patience, params
):
    """
    Recursive function, first called in form_coc_clusters

    Look into gmm.fit warning if forming prelminary propagation sequences is too slow or causing problems:
    /home/mea/anaconda3/envs/brain_dance/lib/python3.11/site-packages/sklearn/base.py:1474: ConvergenceWarning: Number of distinct clusters (3) found smaller than n_clusters (4). Possibly due to duplicate points in X.
    return fit_method(estimator, *args, **kwargs)
    """
    elec_locs = params["elec_locs"]
    inner_radius = params["inner_radius"]
    max_n_components = params["max_n_components_latency"]
    min_coc_n = params["min_coc_n"]
    min_coc_p = params["min_coc_p"]
    min_extend_comp_p = params["min_extend_comp_p"]
    verbose = params["verbose"]

    comp_elec = comp_elecs[0]

    if verbose:
        print(f"Comparing to elec {comp_elec}, loc: {elec_locs[comp_elec]}")

    """
    Pseudocode:
        1. Find latencies on comparison electrode
        2. Split cluster based on latencise
            a. Fit GMM
            b. Form clusters based on GMM
            c. Add electrode to clusters's group of splitting elecs
        3. Pick next electrode
        4. Determine if group of upcoming comparison electrodes need to be extended
    """

    # 1.
    all_times = []
    all_latencies = []
    # all_amps = []
    for root_time in allowed_root_times:
        cocs = coc_dict[root_time]
        # Check each electrode that cooccurs
        for tar_elec, tar_time, tar_latency in cocs:
            if tar_elec == comp_elec:  # Comp elec found
                # For plotting all latency distribution
                all_latencies.append(tar_latency)
                all_times.append(root_time)
                break

    # 2.
    min_cocs = max(min_coc_n, len(allowed_root_times) * min_coc_p / 100)

    # Not enough codetections on comp_elec to split, so allowed_root_times stay together
    if len(all_times) <= min_cocs:
        coc_clusters = [root_cluster.split(comp_elec, list(allowed_root_times))]

        if patience.end(comp_elec):
            return coc_clusters
    else:
        all_latencies = np.array(all_latencies)
        # Reshape to (n_samples, n_features) for GMM
        gmm_latencies = all_latencies.reshape(-1, 1)
        best_score = np.inf
        best_gmm = None
        # Prevents warning "ConvergenceWarning: Number of distinct clusters (3) found smaller than n_clusters (4). Possibly due to duplicate points in X."
        max_n_components = min(max_n_components, len(set(all_latencies)))
        for n_components in range(1, max_n_components + 1):
            with _threadpool_limits(limits=1):
                gmm = GaussianMixture(
                    n_components=n_components,
                    random_state=1150,
                    n_init=1,
                    max_iter=1000,
                )
                try:
                    gmm.fit(gmm_latencies)
                    score = gmm.bic(gmm_latencies)
                except (
                    FloatingPointError
                ):  # FloatingPointError: underflow encountered in exp, 1/8/24: guess that this is caused by probably of fitting being so low that it causes underflow error
                    continue

            if score < 0:  # If negative, it fits data too well
                continue

            if score < best_score:
                best_score = score
                best_gmm = gmm

            if score < 0:
                continue

        if (
            best_gmm is None or best_gmm.n_components == 1
        ):  # If all GMM are negative, use n_components=1
            coc_clusters = [root_cluster.split(comp_elec, all_times)]
            patience.reset()  # Can just reset since "len(all_times) <= min_cocs" ensures enough coc if n_components=1
        else:
            predictions = best_gmm.predict(gmm_latencies)
            coc_clusters = [
                root_cluster.split(comp_elec, []) for _ in range(best_gmm.n_components)
            ]
            for cluster, time in zip(predictions, all_times):
                coc_clusters[cluster]._spike_train.append(time)

            coc_clusters = [c for c in coc_clusters if len(c._spike_train) >= min_cocs]

            if (
                len(coc_clusters) == 0
            ):  # If allowed_root_times were split into clusters with not enough spikes, allow original cluster to continue branching
                coc_clusters = [root_cluster.split(comp_elec, list(allowed_root_times))]
                if patience.end(comp_elec):
                    return coc_clusters
            else:
                patience.reset()

    if len(comp_elecs) == 1:  # If no more elecs to compare to/branch
        return coc_clusters

    # Recursion branching
    min_extend_comp = len(allowed_root_times) * min_extend_comp_p / 100
    comp_elecs_set = set(comp_elecs)
    new_coc_clusters = []
    for cluster in coc_clusters:
        # Check if enough cocs for further splitting
        if len(cluster._spike_train) <= min_coc_n:
            new_coc_clusters.append(cluster)
            continue

        # Check whether to add more electrodes to comp_elecs
        if len(cluster._spike_train) >= min_extend_comp:
            cluster_comp_elecs = comp_elecs[1:]
            to_be_added_cluster_comp_elecs = {}
            for elec, loc in enumerate(elec_locs):
                if elec == comp_elec:
                    continue

                dist = calc_dist(*elec_locs[comp_elec], *loc)
                # Prevent 1) double counting elecs 2) splitting on an elec that has already been used for splitting
                if (
                    dist <= inner_radius
                    and elec not in comp_elecs_set
                    and elec not in cluster.split_elecs
                ):
                    to_be_added_cluster_comp_elecs[elec] = dist
            cluster_comp_elecs += sorted(
                to_be_added_cluster_comp_elecs.keys(),
                key=lambda elec: to_be_added_cluster_comp_elecs[elec],
            )
        else:
            cluster_comp_elecs = comp_elecs[1:]

        # Actually do recursion
        branches = branch_coc_cluster(
            cluster,
            cluster_comp_elecs,
            coc_dict,
            set(cluster._spike_train),
            patience,
            params,
        )

        new_coc_clusters += branches

    return new_coc_clusters


class Patience:
    def __init__(self, root_elec, patience_end, elec_locs):
        self.counter = 0
        self.root_elec = root_elec

        self.last_dist = 0

        self.patience_end = patience_end
        self.elec_locs = elec_locs

    def reset(self):
        self.counter = 0

    def end(self, comp_elec) -> bool:
        """
        Increment counter and check if to end
        """

        dist = calc_dist(*self.elec_locs[self.root_elec], *self.elec_locs[comp_elec])
        # dist = calc_elec_dist(self.root_elec, comp_elec)
        if dist != self.last_dist:
            self.counter += 1
            if self.counter >= self.patience_end:
                return True

        self.last_dist = dist
        return False

    def verbose(self):
        print(f"Patience: {self.counter}/{self.patience_end}")


# endregion

# region Spike assignment


def reassign_spikes_to_clusters(all_clusters, model, params):
    """
    Reassign spikes to preliminary propagation sequences
    """
    elec_locs = params["elec_locs"]
    model_inter_path = params["model_inter_path"]
    inner_radius = params["inner_radius"]
    min_seq_spikes = params["min_seq_spikes"]
    device = params["device"]
    verbose = params["verbose"]
    # VERY rough estimate (and properly calculating the amount of memory needed may require handling the sequence matrices found in RTSort.__init__ and the intermediate matrices found in RTSort.sort)
    gb_per_sequence_per_elec = 35 / 2600 / 1000

    # all_clusters = all_clusters[:2415]
    if verbose:
        print("Reassigning spikes to preliminary propagation sequences")

    # Sequences will stay together like this unless using cuda and not enough free GPU memory
    seq_groups = [all_clusters]
    if device == "cuda":
        # Check if there is enough GPU memory to store reassign to all sequences at once
        # If not, need to divide sequences into groups, where sequences in the same group are close enough for repeated detection removal
        # TODO: Handle case where even after this split, there is not enough memory for a sequence group
        free_memory = get_gpu_free_memory()
        num_elecs = elec_locs.shape[0]
        memory_needed = gb_per_sequence_per_elec * len(all_clusters) * num_elecs
        if memory_needed > free_memory:
            if verbose:
                print(
                    f"To handle all {len(all_clusters)} sequences at once, the GPU needs approximately {memory_needed:.1f}GB, but only {free_memory:.1f}GB is free.\nSeparating sequences into groups ..."
                )

            seq_groups = [[all_clusters[0]]]
            for seq in all_clusters[1:]:
                seq_loc = elec_locs[seq.root_elec]
                group_add_idx = None  # Index of seq_groups to add seq to

                group_idx = 0
                while group_idx < len(seq_groups):
                    for other_seq in seq_groups[group_idx]:
                        # Seq is close enough to a seq in the current group for repeated detection removal
                        if (
                            calc_dist(*seq_loc, *elec_locs[other_seq.root_elec])
                            <= inner_radius
                        ):
                            if (
                                group_add_idx is None
                            ):  # If seq has not yet found group, make current group the sequences group
                                seq_groups[group_idx].append(seq)
                                group_add_idx = group_idx
                                group_idx += 1
                                break
                            else:  # If seq has found group, merge the current group with the sequence's group
                                seq_groups[group_add_idx] += seq_groups.pop(group_idx)
                                break
                    else:
                        group_idx += 1

                if (
                    group_add_idx is None
                ):  # seq is not close enough with any sequence in any group
                    seq_groups.append([seq])

            # A seq_group may be small. So group seq_groups to minimize the number of seq_groups while ensuring that the number of seqs in each seq_groups is below the memory limit
            max_num_seq = free_memory / (gb_per_sequence_per_elec * elec_locs.shape[0])
            seq_groups = sorted(seq_groups, key=lambda group: len(group))
            new_seq_groups = []
            while len(seq_groups) > 0:
                group = seq_groups.pop(-1)
                if len(group) >= max_num_seq or len(seq_groups) == 0:
                    new_seq_groups.append(group)
                else:
                    while (
                        len(seq_groups) > 0
                        and len(group) + len(seq_groups[-1]) <= max_num_seq
                    ):
                        group += seq_groups.pop(-1)
                    new_seq_groups.append(group)
            seq_groups = new_seq_groups

            if verbose:
                print(
                    f"Created {len(seq_groups)} groups of sizes: {[len(group) for group in seq_groups]}"
                )

    for group_idx, seqs in enumerate(seq_groups):
        if verbose:
            if len(seq_groups) > 1:
                print(f"Initializing group {group_idx} ...")
            else:
                print("Initializing ...")
        rt_sort = RTSort(seqs, model, params, device=device)
        all_detections = rt_sort.sort_offline(
            model_inter_path / "model_traces.npy",
            model_outputs=model_inter_path / "model_outputs.npy",
            verbose=verbose,
        )
        del rt_sort  # Save memory
        torch.cuda.empty_cache()
        for seq, spike_train in zip(seqs, all_detections):
            seq._spike_train = spike_train

    all_clusters = [
        cluster
        for cluster in all_clusters
        if len(cluster._spike_train) >= min_seq_spikes
    ]
    all_clusters = setup_coc_clusters_parallel(all_clusters, params)
    # all_clusters = filter_clusters(all_clusters, params)
    return all_clusters


# endregion

# region Set up sequences


def _setup_coc_clusters_parallel(task):
    # Job for setup_coc_clusters_parallel
    cluster, params = task
    with warnings.catch_warnings():
        set_warning_level(params["ignore_warnings"])
        filtered = setup_cluster(cluster, params)
        if filtered:
            return cluster
        else:
            return None
        # return cluster


def setup_coc_clusters_parallel(coc_clusters, params):
    """
    Run setup_cluster on coc_clusters with parallel processing
    """
    num_processes = params["num_processes"]
    verbose = params["verbose"]

    if verbose:
        print("Extracting sequences' detections, intervals, and amplitudes")

    new_coc_clusters = []
    tasks = [(cluster, params) for cluster in coc_clusters]
    with Pool(processes=num_processes) as pool:
        imap = pool.imap(_setup_coc_clusters_parallel, tasks)
        if verbose:
            imap = tqdm(imap, total=len(tasks))
        for cluster in imap:
            if cluster is not None:
                new_coc_clusters.append(cluster)
    if verbose:
        print(f"{len(new_coc_clusters)} clusters remain after filtering")
    return new_coc_clusters


def setup_coc_clusters(coc_clusters, params):
    # Set important data needed for merging and other analyses
    if params["verbose"]:
        coc_clusters = tqdm(coc_clusters)

    for cluster in coc_clusters:
        setup_cluster(cluster, params)


def setup_cluster(cluster, params, n_cocs=None):
    """
    Parameters:
    n_cocs:
        If not None, setup cluster using randomly selected n_cocs
    """
    # For extracting
    samp_freq = params["samp_freq"]
    elec_locs = params["elec_locs"]
    model_inter_path = params["model_inter_path"]
    front_buffer = params["front_buffer"]
    n_before = params["n_before"]
    n_after = params["n_after"]
    pre_median_frames = params["pre_median_frames"]
    min_loose_elec_prob = params["min_loose_elec_prob"]

    # For filtering
    loose_thresh = params["loose_thresh"]
    inner_radius = params["inner_radius"]
    min_elecs_for_seq_noise = params["min_elecs_for_seq_noise"]
    min_inner_loose_detections = params["min_inner_loose_detections"]
    min_loose_detections_n = params["min_loose_detections_n"]
    min_loose_detections_r_spikes = params["min_loose_detections_r_spikes"]

    root_elec = cluster.root_elec
    array_elecs = np.arange(elec_locs.shape[0])[:, None]  #  Used to indexing
    outputs = np.load(model_inter_path / "model_outputs.npy", mmap_mode="r")
    traces = np.load(model_inter_path / "model_traces.npy", mmap_mode="r")

    # Select random cocs
    spike_train = cluster.spike_train
    if n_cocs is not None and n_cocs < len(spike_train):
        raise ValueError(
            "n_cocs must be None because the merging of metrics assumes all spikes are extracted"
        )
        spike_train = np.random.choice(spike_train, n_cocs, replace=False)
    output_frames = np.round(spike_train * samp_freq - front_buffer).astype(int)
    output_windows = outputs[
        array_elecs, output_frames[:, None, None] + np.arange(-n_before, n_after + 1)
    ]  # (num_spikes, num_elecs, num_frames)

    # Elec probs
    all_elec_probs = np.max(output_windows, axis=2)
    all_elec_probs[:, root_elec] = output_windows[
        :, root_elec, n_before
    ]  # max value in window maynot be at the detected spike time
    all_elec_probs = sigmoid(all_elec_probs)
    cluster.every_elec_prob = all_elec_probs.T  # (n_elecs, n_cocs)

    all_elec_probs = np.median(all_elec_probs, axis=0)
    all_elec_probs[all_elec_probs < min_loose_elec_prob] = 0
    cluster.all_elec_probs = all_elec_probs  # (n_elecs)

    if np.sum(cluster.all_elec_probs >= loose_thresh) >= min_elecs_for_seq_noise:
        return False

    # Loose elecs
    cluster.loose_elecs = []
    for elec in np.flatnonzero(all_elec_probs >= loose_thresh):
        for split_elec in cluster.split_elecs:
            if calc_dist(*elec_locs[elec], *elec_locs[split_elec]) <= inner_radius:
                cluster.loose_elecs.append(elec)
                break
    if len(cluster.loose_elecs) < min_loose_detections_n:
        return False
    cluster.min_loose_detections = max(
        min_loose_detections_n, min_loose_detections_r_spikes * len(cluster.loose_elecs)
    )

    # Loose inner elecs
    cluster.inner_loose_elecs = []
    # set() to prevent an elec being added more than once
    comp_elecs = set(cluster.loose_elecs)
    # Find inner_loose_elecs and comp_elecs
    for loose_elec in cluster.loose_elecs:
        # Check if loose elec within INNER_RADIUS of any inner elec to be a inner_loose_elec
        for root_elec in cluster.root_elecs:
            if calc_dist(*elec_locs[root_elec], *elec_locs[loose_elec]) <= inner_radius:
                cluster.inner_loose_elecs.append(loose_elec)
                break  # Add loose_elec only once
        # Add elec's inner elecs to comp_elecs
        for elec, loc in enumerate(elec_locs):
            if (
                calc_dist(*elec_locs[loose_elec], *loc) <= inner_radius
                and all_elec_probs[elec] > 0
            ):
                comp_elecs.add(elec)
    if len(cluster.inner_loose_elecs) < min_inner_loose_detections:
        return False

    # Comp elecs
    root_elec = cluster.root_elecs[0]
    cluster.comp_elecs = [root_elec] + [
        elec for elec in comp_elecs if elec != root_elec
    ]

    # Latencies
    all_latencies = (
        np.argmax(output_windows, axis=2) - n_before
    )  # (num_spikes, num_elecs)
    all_latencies[:, root_elec] = 0
    cluster.every_latency = all_latencies.T  # (n_elecs, n_cocs)
    cluster.all_latencies = np.mean(all_latencies, axis=0)

    # Amplitudes
    output_frames += front_buffer  # This actually refers to frames in the recording (not the output frames, but not creating new variable to save time and memory)
    traces_windows = traces[
        array_elecs,
        output_frames[:, None, None]
        + np.arange(-pre_median_frames - n_before, n_after + 1),
    ]  # (num_spikes, num_elecs, num_frames)

    pre_medians = []  # (num_crossings, num_max_amp_elecs)
    num_cocs = 300  # Only handle num_cocs at a time (np.abs loads entire traces into memory since it is only loaded as mmap before)
    i = 0
    while i < len(spike_train):
        cur_traces_windows = traces_windows[i : i + 300, :, :pre_median_frames]
        cur_pre_medians = np.median(np.abs(cur_traces_windows), axis=2)
        pre_medians.append(np.clip(cur_pre_medians / 0.6745, a_min=0.5, a_max=None))
        i += num_cocs
    pre_medians = np.vstack(pre_medians)
    # # Calculating the pre_medians all at once requires too much memory and sometimes results in crashing
    # pre_medians = np.median(np.abs(traces_windows[:, :, :pre_median_frames]), axis=2)  # (num_crossings, num_max_amp_elecs)
    # pre_medians = np.clip(pre_medians/0.6745, a_min=0.5, a_max=None)

    all_amp_medians = (
        np.abs(
            traces_windows[
                np.arange(output_frames.size)[:, None],  # (num_spikes, 1)
                array_elecs.T,  # (1, num_elecs)
                all_latencies + pre_median_frames + n_before,  # (num_spikes, num_elecs)
            ]
        )
        / pre_medians
    )  # (num_spikes, num_elecs)
    cluster.every_amp_median = all_amp_medians.T  # (n_elecs, n_cocs)
    cluster.all_amp_medians = np.mean(all_amp_medians, axis=0)

    # region Slower original code
    # # Start extracting stats
    # all_elec_probs = []  # (n_cocs, n_elecs)
    # all_latencies = []  # (n_cocs, n_elecs)
    # all_amp_medians = []  # (n_cocs, n_elecs)

    # for time in spike_train:
    #     # Get elec probs
    #     output_frame = rec_ms_to_output_frame(time, samp_freq, front_buffer)
    #     this_n_before = n_before if output_frame - n_before >= 0 else output_frame  # Prevents indexing problems
    #     output_window = outputs[:, output_frame - this_n_before:output_frame+n_after+1]
    #     elec_probs = np.max(output_window, axis=1)
    #     # max value in window may not be at :time:
    #     elec_probs[root_elec] = output_window[root_elec, this_n_before]

    #     all_elec_probs.append(sigmoid(elec_probs))

    #     # Get latencies
    #     latencies = np.argmax(output_window, axis=1) - this_n_before
    #     latencies[root_elec] = 0
    #     all_latencies.append(latencies)

    #     # Get amp/medians
    #     rec_frame = round(time * samp_freq)
    #     pre_medians = calc_pre_median(
    #         max(0, rec_frame-n_before), traces, pre_median_frames)
    #     amps = np.abs(traces[array_elecs.flatten(), rec_frame + latencies])
    #     amp_medians = amps / pre_medians
    #     all_amp_medians.append(amp_medians)

    # # Set stats (all for all electroes in array, but store values for self.comp_elecs for fast comparison for assigning spikes)

    # # When elecs are only based on inner and outer radius
    # # elecs = [root_elec] + get_nearby_elecs(root_elec, max_elec_dist)  # Store sliced array for fast comparision with intraelec merging and assigning spikes
    # # cluster.elecs = elecs

    # # all_elec_probs = sum_elec_probs / len(spike_train)
    # cluster.every_elec_prob = np.array(all_elec_probs).T  # (n_elecs, n_cocs)
    # all_elec_probs = np.median(all_elec_probs, axis=0)
    # all_elec_probs[all_elec_probs < min_loose_elec_prob] = 0
    # cluster.all_elec_probs = all_elec_probs  # (n_elecs)

    # cluster.every_latency = np.array(all_latencies).T  # (n_elecs, n_cocs)
    # cluster.all_latencies = np.mean(all_latencies, axis=0)
    # # cluster.latencies = cluster.all_latencies[comp_elecs[1:]]  # Don't include root elec since always 0

    # cluster.every_amp_median = np.array(all_amp_medians).T  # (n_elecs, n_cocs)
    # cluster.all_amp_medians = np.mean(all_amp_medians, axis=0)
    # endregion

    if params["debug"]:
        test = (
            cluster.every_elec_prob[root_elec] - params["stringent_thresh"] < -0.01
        )  # Rounding error? Sometimes 0.001 off
        if np.any(test):
            save_path = params["model_inter_path"] / "cluster.pickle"
            pickle_dump(cluster, save_path)
            raise ValueError(
                f"There is a bug with spike times not being centered on threshold or subthreshold spikes being assigned.\nA cluster has {np.sum(test)} spikes with root elec prob below stringent thresh: {np.flatnonzero(test)}\nSaved to {save_path}"
            )

    # setup_elec_stats(cluster, params)
    cluster.root_to_amp_median_std = {
        root: np.std(cluster.every_amp_median[root, :], ddof=1)
        for root in cluster.root_elecs
    }
    cluster.root_amp_median = cluster.all_amp_medians[cluster.root_elec]
    return True


def setup_elec_stats(cluster, params):
    """
    Set cluster.loose_elecs, cluster.inner_loose_elecs
    and the root elec

    240813 speed update: NOT NEEDED OR IMPLEMENTED ANYMORE
    cluster.root_to_stats = {root_elec: [comp_elecs, elec_probs, latencies, amp_medians]}
        Stats are from all_stat[comp_elecs]. Each root_elec has same comp_elecs, but comp_elecs[0] = root_elec
    """

    elec_locs = params["elec_locs"]
    loose_thresh = params["loose_thresh"]
    inner_radius = params["inner_radius"]
    min_loose_detections_n = params["min_loose_detections_n"]
    min_loose_detections_r_spikes = params["min_loose_detections_r_spikes"]

    cluster.root_to_amp_median_std = {
        root: np.std(cluster.every_amp_median[root, :], ddof=1)
        for root in cluster.root_elecs
    }
    cluster.root_amp_median = cluster.all_amp_medians[cluster.root_elec]

    # Find elecs
    cluster.loose_elecs = []
    for elec in np.flatnonzero(cluster.all_elec_probs >= loose_thresh):
        for split_elec in cluster.split_elecs:
            if calc_dist(*elec_locs[elec], *elec_locs[split_elec]) <= inner_radius:
                cluster.loose_elecs.append(elec)
                break

    cluster.inner_loose_elecs = []
    # set() to prevent an elec being added more than once
    comp_elecs = set(cluster.loose_elecs)
    # Find inner_loose_elecs and comp_elecs
    for loose_elec in cluster.loose_elecs:
        # Check if loose elec within INNER_RADIUS of any inner elec to be a inner_loose_elec
        for root_elec in cluster.root_elecs:
            if calc_dist(*elec_locs[root_elec], *elec_locs[loose_elec]) <= inner_radius:
                cluster.inner_loose_elecs.append(loose_elec)
                break  # Add loose_elec only once
        # Add elec's inner elecs to comp_elecs
        for elec, loc in enumerate(elec_locs):
            if (
                calc_dist(*elec_locs[loose_elec], *loc) <= inner_radius
                and cluster.all_elec_probs[elec] > 0
            ):
                comp_elecs.add(elec)

    cluster.min_loose_detections = max(
        min_loose_detections_n, min_loose_detections_r_spikes * len(cluster.loose_elecs)
    )

    # For each root elec, make separate comp_elecs so that first elec is root_elec (needed for fast access to compare latencies since root elec should not be considered)
    # This is for fast indexing for assigning spikes
    cluster.root_to_stats = {}
    # Do it in reverse order so that comp_elecs will be set for root_elecs[0] for rest of function
    for root_elec in cluster.root_elecs[::-1]:
        comp_elecs = [root_elec] + [elec for elec in comp_elecs if elec != root_elec]
        latencies = cluster.all_latencies[comp_elecs]
        cluster.root_to_stats[root_elec] = (
            comp_elecs,
            cluster.all_elec_probs[comp_elecs],
            # For individual spikes due to variations in latency, offsetting latency like this may not be accurate. But averaged over hundreds of spikes, it should be fine
            latencies[1:] - latencies[0],
            cluster.all_amp_medians[comp_elecs],
            cluster.root_to_amp_median_std[root_elec],
        )

    root_elec = cluster.root_elecs[0]
    cluster.comp_elecs = [root_elec] + [
        elec for elec in comp_elecs if elec != root_elec
    ]


def relocate_root(cluster, new_root, params):
    """
    Relocate cluster's root electrode
    """
    cluster.root_elec = new_root
    cluster.root_elecs = [new_root]
    # Adjust latencies
    cluster.every_latency -= cluster.every_latency[new_root, :]
    cluster.all_latencies = np.mean(cluster.every_latency, axis=1)
    # Adjust elecs
    setup_elec_stats(cluster, params)


def relocate_root_latency(cluster, params):
    """
    Change root electrode to most negative-latency electrode with:
        Mean latency of -2 frames or less
        Median detection above stringent threhsold
        Mean amplitude that is 80% or more of current root
    If no electrode meets requirements, keep current root electrode
    """
    stringent_thresh = params["stringent_thresh"]
    relocate_root_min_amp = params["relocate_root_min_amp"]
    relocate_root_max_latency = params["relocate_root_max_latency"]

    possible_roots = np.flatnonzero(
        (
            cluster.all_amp_medians / cluster.all_amp_medians[cluster.root_elec]
            >= relocate_root_min_amp
        )
        & (cluster.all_elec_probs >= stringent_thresh)
    )
    latencies = cluster.all_latencies[possible_roots]
    min_latency = np.min(latencies)
    if min_latency <= relocate_root_max_latency:
        root_elec = possible_roots[np.argmin(latencies)]
        relocate_root(cluster, root_elec, params)


def relocate_root_prob(cluster, params):
    """
    Change root electrode to electrode with highest median detection score
    """
    new_root = np.argmax(cluster.all_elec_probs)
    # new_root = np.argmax(cluster.all_amp_medians)
    relocate_root(cluster, new_root, params)


def filter_clusters(coc_clusters, params):
    """
    Return coc_clusters that enough loose and inner loose electrodes
        But not too many to be considered a noise sequence
    """
    loose_thresh = params["loose_thresh"]
    min_elecs_for_seq_noise = params["min_elecs_for_seq_noise"]
    min_inner_loose_detections = params["min_inner_loose_detections"]
    min_loose_detections_n = params["min_loose_detections_n"]

    filtered_clusters = []
    for cluster in coc_clusters:
        if (
            len(cluster.inner_loose_elecs) >= min_inner_loose_detections
            and len(cluster.loose_elecs) >= min_loose_detections_n
            and np.sum(cluster.all_elec_probs >= loose_thresh) < min_elecs_for_seq_noise
        ):
            filtered_clusters.append(cluster)
    return filtered_clusters


# endregion

# region Merge clusters


class Merge:
    # Represent a CocCluster merge
    def __init__(self, cluster_i, cluster_j):
        self.cluster_i = cluster_i
        self.cluster_j = cluster_j

    def merge(self, params):
        # region Combine spike trains, but if both clusters detect same spike, only add once

        # Now handled in merge_coc_clusters
        # if self.cluster_i.root_amp_median >= self.cluster_j.root_amp_median:
        #     cluster_i = self.cluster_i
        #     cluster_j = self.cluster_j
        # else:
        #     cluster_i = self.cluster_j
        #     cluster_j = self.cluster_i
        #     self.cluster_i = cluster_i
        #     self.cluster_j = cluster_j
        overlap_time = params["repeated_detection_overlap_time"]
        min_loose_elec_prob = params["min_loose_elec_prob"]

        cluster_i = self.cluster_i
        cluster_j = self.cluster_j

        spike_train_i = cluster_i.spike_train
        spike_train_j = cluster_j.spike_train

        spike_train = [spike_train_i[0]]
        every_elec_prob = [cluster_i.every_elec_prob[:, 0]]
        every_latency = [cluster_i.every_latency[:, 0]]
        every_amp_median = [cluster_i.every_amp_median[:, 0]]

        i, j = 1, 0
        while i < len(spike_train_i) and j < len(spike_train_j):
            spike_i, spike_j = spike_train_i[i], spike_train_j[j]
            if spike_i < spike_j:  # i is next to be added
                # 1/SAMP_FREQ:  # Ensure not adding same spikes twice (clusters being merged often detect the same spikes) (account for rounding error)
                if spike_i - spike_train[-1] > overlap_time:
                    spike_train.append(spike_i)
                    every_elec_prob.append(cluster_i.every_elec_prob[:, i])
                    every_latency.append(cluster_i.every_latency[:, i])
                    every_amp_median.append(cluster_i.every_amp_median[:, i])
                i += 1
            else:  # j is next to be added
                # 1/SAMP_FREQ: # Ensure not adding same spikes twice (clusters being merged often detect the same spikes) (account for rounding error)
                if spike_j - spike_train[-1] > overlap_time:
                    spike_train.append(spike_j)
                    every_elec_prob.append(cluster_j.every_elec_prob[:, j])

                    latency = cluster_j.every_latency[:, j]
                    # Need to adjust cluster_j latencies to cluster_i
                    if cluster_i.root_elecs[0] != cluster_j.root_elecs[0]:
                        latency = latency - latency[cluster_i.root_elecs[0]]
                    every_latency.append(latency)

                    every_amp_median.append(cluster_j.every_amp_median[:, j])
                j += 1

        # Append remaning elements (only one cluster's spike train can be appended due to while loop)
        if i < len(spike_train_i):
            spike_train.extend(spike_train_i[i:])
            every_elec_prob.extend(cluster_i.every_elec_prob[:, i:].T)
            every_latency.extend(cluster_i.every_latency[:, i:].T)
            every_amp_median.extend(cluster_i.every_amp_median[:, i:].T)
        else:
            spike_train.extend(spike_train_j[j:])
            every_elec_prob.extend(cluster_j.every_elec_prob[:, j:].T)
            every_latency.extend(cluster_j.every_latency[:, j:].T)
            every_amp_median.extend(cluster_j.every_amp_median[:, j:].T)

        # Set new spike train
        # try:
        #     self.cluster_i._spike_train = spike_train
        # except AttributeError:
        #     self.cluster_i.spike_train = spike_train
        cluster_i._spike_train = spike_train
        # endregion

        # region Update stats
        # n = len(cluster_i._spike_train)
        # m = len(cluster_j._spike_train)

        # Update root elecs
        cluster_i_elecs = set(cluster_i.root_elecs)
        for elec in cluster_j.root_elecs:
            if elec not in cluster_i_elecs:
                cluster_i.root_elecs.append(elec)

        # Elec probs
        # all_elec_probs = combine_means(cluster_i.all_elec_probs, n, cluster_j.all_elec_probs, m)
        # cluster_i.every_elec_prob = np.concatenate((cluster_i.every_elec_prob, cluster_j.every_elec_prob), axis=1)
        cluster_i.every_elec_prob = np.vstack(every_elec_prob).T
        all_elec_probs = np.median(cluster_i.every_elec_prob, axis=1)
        all_elec_probs[all_elec_probs < min_loose_elec_prob] = 0
        cluster_i.all_elec_probs = all_elec_probs  # (n_elecs)

        # cluster_i.elec_probs = cluster_i.all_elec_probs[elecs]

        # Latencies
        # every_latency = cluster_j.every_latency
        # # all_latencies = cluster_j.all_latencies
        # if cluster_i.root_elecs[0] != cluster_j.root_elecs[0]:  #  Need to adjust cluster_j latencies to cluster_i)
        #     every_latency -= every_latency[cluster_i.root_elecs[0], :]
        #     # all_latencies = np.mean(every_latency, axis=1)
        # cluster_i.every_latency = np.concatenate((cluster_i.every_latency, every_latency), axis=1)
        # cluster_i.all_latencies = combine_means(cluster_i.all_latencies, n, all_latencies, m)
        cluster_i.every_latency = np.vstack(every_latency).T
        # cluster_i.all_latencies = np.median(cluster_i.every_latency, axis=1)
        cluster_i.all_latencies = np.mean(cluster_i.every_latency, axis=1)
        # cluster_i.latencies = cluster_i.all_latencies[elecs[1:]]

        # Amp/medians
        # cluster_i.every_amp_median = np.concatenate((cluster_i.every_amp_median, cluster_j.every_amp_median), axis=1)
        cluster_i.every_amp_median = np.vstack(every_amp_median).T
        cluster_i.all_amp_medians = np.mean(cluster_i.every_amp_median, axis=1)
        cluster_i.root_to_amp_median_std = {
            root: np.std(cluster_i.every_amp_median[root, :], ddof=1)
            for root in cluster_i.root_elecs
        }
        cluster_i.root_amp_median = cluster_i.all_amp_medians[cluster_i.root_elec]
        # cluster_i.all_amp_medians = combine_means(cluster_i.all_amp_medians, n, cluster_j.all_amp_medians, m)
        # luster_i.amp_medians = cluster_i.all_amp_medians[elecs]
        # endregion

        # try:
        # cluster_i._spike_train.extend(cluster_j._spike_train)
        # except AttributeError:
        #     self.cluster_i.spike_train.extend(self.cluster_j.spike_train)
        #     self.cluster_i.spike_train = np.sort(self.cluster_i.spike_train)  # If accessing spike train this way, keep it sorted

        setup_elec_stats(cluster_i, params)
        # setup_cluster(self.cluster_i)  # Update stats

        return cluster_j  # Return to update dead_clusters


def merge_verbose(merge, update_history=True):
    """
    Verbose for merge_coc_clusters

    Params:
    update_history:
        If True, history of clusters will be updated
        False is for when no merge is found, but still want verbose
    """
    raise NotImplementedError

    cluster_i, cluster_j = merge.cluster_i, merge.cluster_j

    if hasattr(cluster_i, "merge_history"):
        message = f"\nMerged {cluster_i.merge_history} with "
        if update_history:
            cluster_i.merge_history.append(cluster_j.idx)
    else:
        message = f"\nMerged {cluster_i.idx} with "
        if update_history:
            cluster_i.merge_history = [cluster_i.idx, cluster_j.idx]

    if hasattr(cluster_j, "merge_history"):
        message += str(cluster_j.merge_history)
        if update_history:
            cluster_i.merge_history += cluster_j.merge_history[1:]
    else:
        message += f"{cluster_j.idx}"
    print(message)
    print(
        f"Latency diff: {merge.latency_diff:.2f}. Amp median diff: {merge.amp_median_diff:.2f}"
    )
    print(f"Amp dist p-value {merge.dip_p:.4f}")

    print(f"#spikes:")
    num_overlaps = Comparison.count_matching_events(
        cluster_i.spike_train, cluster_j.spike_train, delta=OVERLAP_TIME
    )
    # num_overlaps = len(set(cluster_i.spike_train).intersection(cluster_j.spike_train))
    print(
        f"Merge base: {len(cluster_i.spike_train)}, Add: {len(cluster_j.spike_train)}, Overlaps: {num_overlaps}"
    )

    # Find ISI violations after merging
    # cat_spikes = np.sort(np.concatenate((cluster_i.spike_train, cluster_j.spike_train)))
    # diff = np.diff(cat_spikes)
    # num_viols = np.sum(diff <= 1.5)
    # print(f"ISI viols: {num_viols}")

    # Plot footprints
    # amp_kwargs, prob_kwargs = plot_elec_probs(cluster_i, idx=cluster_i.idx)
    # plt.show()
    # plot_elec_probs(cluster_j, amp_kwargs=amp_kwargs, prob_kwargs=prob_kwargs, idx=cluster_j.idx)
    # plt.show()

    # # Plot amp distribution
    # all_amps = get_amp_dist(cluster_i) + get_amp_dist(cluster_j)
    # plot_amp_dist(np.array(all_amps))
    # plt.show()


def merge_coc_clusters(coc_clusters, params):
    """
    Params
        stop_at: None or int: NOT IMPLEMENTED
            If remaining clusters that could be merged is equal to stop_at, stop merging even if more merging is possible

    """

    elec_locs = params["elec_locs"]
    inner_radius = params["inner_radius"]
    min_inner_loose_detections = params["min_inner_loose_detections"]
    min_loose_detections_n = params["min_loose_detections_n"]
    min_loose_detections_r = params["min_loose_detections_r_sequences"]
    max_latency_diff = params["max_latency_diff_sequences"]
    clip_latency_diff = max_latency_diff * params["clip_latency_diff_factor"]
    max_amp_median_diff = params["max_amp_median_diff_sequences"]
    clip_amp_median_diff = max_amp_median_diff * params["clip_amp_median_diff_factor"]
    max_root_amp_median_std_sequences = params["max_root_amp_median_std_sequences"]
    verbose = False  # params['verbose']

    for idx, cluster in enumerate(coc_clusters):
        cluster.idx = idx

    dead_clusters = set()
    while True:
        # Find best merge
        best_merge = None
        # best_unmerge = None  # Best merge that cannot merge (for final verbose)

        # if stop_at is not None and len(coc_clusters) - len(dead_clusters) == stop_at:
        #     break

        for i in range(len(coc_clusters)):
            # Load cluster i
            cluster_i = coc_clusters[i]
            if cluster_i in dead_clusters:
                continue

            for j in range(i + 1, len(coc_clusters)):
                # Load cluster j
                cluster_j = coc_clusters[j]
                if cluster_j in dead_clusters:
                    continue

                # Check if root elecs are close enough (find max dist between i_root_elecs and j_root_elecs)
                max_dist = 0
                for root_i in cluster_i.root_elecs:
                    for root_j in cluster_j.root_elecs:
                        max_dist = max(
                            max_dist, calc_dist(*elec_locs[root_i], *elec_locs[root_j])
                        )
                        if max_dist >= inner_radius:
                            break
                    else:
                        continue
                    break
                if max_dist >= inner_radius:
                    continue

                # Check if enough overlapping loose electrodes
                total_loose = len(
                    set(cluster_i.loose_elecs).union(cluster_j.loose_elecs)
                )
                num_loose_overlaps = len(
                    set(cluster_i.loose_elecs).intersection(cluster_j.loose_elecs)
                )
                # print(num_loose_overlaps, total_loose, num_loose_overlaps/total_loose)
                if (
                    num_loose_overlaps < min_loose_detections_n
                    or num_loose_overlaps / total_loose < min_loose_detections_r
                ):
                    continue
                num_inner_loose_overlaps = len(
                    set(cluster_i.inner_loose_elecs).intersection(
                        cluster_j.inner_loose_elecs
                    )
                )
                if num_inner_loose_overlaps < min_inner_loose_detections:
                    continue

                # if calc_elec_dist(cluster_i.root_elecs[0], cluster_j.root_elecs[0]) > max_root_elec_dist:
                #     continue

                # Get elecs for comparison (do it this way so comp_elecs[0] is root elec)
                # Find which cluster's root amp to use (use one with higher amplitude)
                if cluster_i.root_amp_median >= cluster_j.root_amp_median:
                    i_comp_elecs = cluster_i.comp_elecs
                    i_comp_elecs_set = set(i_comp_elecs)
                    comp_elecs = i_comp_elecs + [
                        elec
                        for elec in cluster_j.comp_elecs
                        if elec not in i_comp_elecs_set
                    ]
                else:
                    j_comp_elecs = cluster_j.comp_elecs
                    j_comp_elecs_set = set(j_comp_elecs)
                    comp_elecs = j_comp_elecs + [
                        elec
                        for elec in cluster_i.comp_elecs
                        if elec not in j_comp_elecs_set
                    ]

                # Get elec probs
                i_elec_probs = cluster_i.all_elec_probs[comp_elecs]
                j_elec_probs = cluster_j.all_elec_probs[comp_elecs]

                # Compare latencies
                i_latencies = cluster_i.all_latencies[comp_elecs][1:]
                # Relative to same electrode as cluster_i
                j_latencies = (
                    cluster_j.all_latencies[comp_elecs][1:]
                    - cluster_j.all_latencies[comp_elecs[0]]
                )
                elec_weights = (i_elec_probs[1:] + j_elec_probs[1:]) / 2
                elec_weights /= np.sum(elec_weights)
                latency_diff = np.abs(i_latencies - j_latencies)
                latency_diff = np.clip(
                    latency_diff, a_min=None, a_max=clip_latency_diff
                )
                latency_diff = np.sum(latency_diff * elec_weights)
                # latency_diff = np.sum(np.abs(i_latencies - j_latencies) * elec_weights)
                if latency_diff > max_latency_diff:
                    continue

                # Compare amp/medians
                i_amp_medians = cluster_i.all_amp_medians[comp_elecs]
                j_amp_medians = cluster_j.all_amp_medians[comp_elecs]
                elec_weights = (i_elec_probs + j_elec_probs) / 2
                elec_weights /= np.sum(elec_weights)
                amp_median_div = (i_amp_medians + j_amp_medians) / 2
                amp_median_diff = (
                    np.abs((i_amp_medians - j_amp_medians)) / amp_median_div
                )
                amp_median_diff = np.clip(
                    amp_median_diff, a_min=None, a_max=clip_amp_median_diff
                )
                amp_median_diff = np.sum(amp_median_diff * elec_weights)
                if amp_median_diff > max_amp_median_diff:
                    continue

                # Test if merge is bimodal
                are_unimodal = True
                pval = np.inf  # Called p-value but is double z-score test now
                for root_elec in set(cluster_i.root_elecs + cluster_j.root_elecs):
                    # root_amps_i = cluster_i.every_amp_median[root_elec, :] # get_amp_medians(cluster_i, root_elec=root_elec)
                    # root_amps_j = cluster_j.every_amp_median[root_elec, :] # get_amp_medians(cluster_j, root_elec=root_elec)
                    # Skip if distribution is not unimodal before merging
                    # if diptest(root_amps_i)[1] < min_amp_dist_p or diptest(root_amps_j)[1] < min_amp_dist_p:
                    #     pval = -1
                    #     continue

                    # Dip test
                    # dip, pval = diptest(np.concatenate([root_amps_i, root_amps_j]))
                    # if pval < min_amp_dist_p:
                    #     are_unimodal = False
                    #     break

                    mean_i = cluster_i.all_amp_medians[root_elec]
                    std_i = (
                        cluster_i.root_to_amp_median_std[root_elec]
                        if root_elec in cluster_i.root_elecs
                        else np.std(cluster_i.every_amp_median[root_elec], ddof=1)
                    )
                    mean_j = cluster_j.all_amp_medians[root_elec]
                    std_j = (
                        cluster_j.root_to_amp_median_std[root_elec]
                        if root_elec in cluster_j.root_elecs
                        else np.std(cluster_j.every_amp_median[root_elec], ddof=1)
                    )
                    mean_diff = np.abs(mean_i - mean_j)
                    two_z_score = pval = max(mean_diff / std_i, mean_diff / std_j)
                    if two_z_score > max_root_amp_median_std_sequences:
                        are_unimodal = False
                        break

                if not are_unimodal:
                    continue

                # Calculate quality of merge
                cur_merge = Merge(cluster_i, cluster_j)
                score = (
                    latency_diff / max_latency_diff
                    + amp_median_diff / max_amp_median_diff
                )
                if best_merge is None or score < best_merge.score:
                    best_merge = cur_merge
                    best_merge.score = score
                    best_merge.latency_diff = latency_diff
                    best_merge.amp_median_diff = amp_median_diff
                    best_merge.dip_p = pval

                # if not cur_merge.can_merge(max_latency_diff, max_rel_amp_diff, min_amp_dist_p):
                #     if verbose and (best_unmerge is None or cur_merge.is_better(best_unmerge)):
                #         best_unmerge = cur_merge
                #     continue
                # if best_merge is None or cur_merge.is_better(best_merge, max_latency_diff, max_rel_amp_diff):
                #     best_merge = cur_merge

        # If no merges are good enough
        if best_merge is None:
            # if not best_merge.can_merge(max_latency_diff, max_rel_amp_diff):
            # if verbose:
            #     print(f"\nNo merge found. Next best merge:")
            #     merge_verbose(best_unmerge, update_history=False)
            break

        # Merge best merge
        # Possibly switch cluster_i and cluster_j if cluster_j has larger root amp median
        if best_merge.cluster_j.root_amp_median > best_merge.cluster_i.root_amp_median:
            cluster_j = best_merge.cluster_j
            best_merge.cluster_j = best_merge.cluster_i
            best_merge.cluster_i = cluster_j

        if verbose:
            merge_verbose(best_merge)

        dead_clusters.add(best_merge.merge(params))
        if verbose:
            print(f"After merging: {len(best_merge.cluster_i.spike_train)}")

    merged_clusters = [
        cluster for cluster in coc_clusters if cluster not in dead_clusters
    ]

    if verbose:
        print(f"\nFormed {len(merged_clusters)} merged clusters:")
        for m, cluster in enumerate(merged_clusters):
            # message = f"cluster {m}: {cluster.idx}"
            # if hasattr(cluster, "merge_history"):
            #     message += f",{cluster.merge_history}"
            # print(message)

            # Without +[]
            if hasattr(cluster, "merge_history"):
                print(f"cluster {m}: {cluster.merge_history}")
            else:
                print(f"cluster {m}: {cluster.idx}")
        # print(f"Formed {len(merged_clusters)} merged clusters")  # Reprint this because jupyter notebook cuts of middle of long outputs
    return merged_clusters


def _intra_merge(task):
    clusters, params = task
    with warnings.catch_warnings():
        set_warning_level(params["ignore_warnings"])
        clusters = merge_coc_clusters(clusters, params)
        for cluster in clusters:
            relocate_root_latency(cluster, params)
        return clusters


def intra_merge(all_clusters, params):
    """
    intra = merge clusters with same root electrode
    """
    num_processes = params["num_processes"]
    verbose = params["verbose"]

    if verbose:
        print("Merging preliminary propagation sequences - first round")

    root_elec_to_clusters = {}
    for cluster in all_clusters:
        if cluster.root_elec not in root_elec_to_clusters:
            root_elec_to_clusters[cluster.root_elec] = [cluster]
        else:
            root_elec_to_clusters[cluster.root_elec].append(cluster)
    tasks = [(clusters, params) for clusters in root_elec_to_clusters.values()]

    intra_merged_clusters = []
    with Pool(processes=num_processes) as pool:
        imap = pool.imap_unordered(_intra_merge, tasks)
        if verbose:
            imap = tqdm(imap, total=len(tasks))
        for clusters in imap:
            intra_merged_clusters.extend(clusters)

    if verbose:
        print(f"{len(intra_merged_clusters)} sequences after first merging")

    return intra_merged_clusters


def inter_merge(intra_merged_clusters, params):
    """
    inter = merge clusters with different root electrodes
    """
    min_spikes = params["min_seq_spikes"]
    verbose = params["verbose"]
    if verbose:
        print("Merging preliminary propagation sequences - second round ...")

    merged_sequences = merge_coc_clusters(intra_merged_clusters, params)
    merged_sequences = [
        seq for seq in merged_sequences if len(seq.spike_train) >= min_spikes
    ]
    merged_sequences = filter_clusters(merged_sequences, params)
    for seq in merged_sequences:
        relocate_root_prob(seq, params)
    merged_sequences = order_sequences(merged_sequences)

    # Set formation spike train (spikes used to detect sequences)
    for seq in merged_sequences:
        seq.formation_spike_train = seq.spike_train

    if verbose:
        print(f"\nRT-Sort detected {len(merged_sequences)} sequences")
    return merged_sequences


# endregion


# region Utils
def pickle_dump(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def pickle_load(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def load_recording(recording):
    """
    If recording is a spikeinterface recording (some subclass of BaseRecording), then simply return it
    Else, recording should be a file path, and it will be loaded as MaxwellRecordingExtractor
    """

    if isinstance(recording, BaseRecording):
        return recording
    recording = str(recording)
    file_type = recording.split(".")[-1]
    if file_type == "h5":
        return MaxwellRecordingExtractor(recording)
    elif file_type == "nwb":
        return NwbRecordingExtractor(recording)
    else:
        raise NotImplementedError(
            f"Cannot load recording {recording} because recording file type {file_type} is not implemented"
        )


def calc_dist(x1, y1, x2, y2):
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))


def rec_ms_to_output_frame(ms, samp_freq, front_buffer):
    # Convert time (in ms) in recording to frame in model's outputs
    return round(ms * samp_freq) - front_buffer


def sigmoid(x):
    # return np.where(x>=0,
    #                 1 / (1 + np.exp(-x)),
    #                 np.exp(x) / (1+np.exp(x))
    #                 )
    x = np.clip(x, a_min=-9, a_max=10)  # Prevent overflow error
    # Positive overflow is not an issue because DL does not output large positive values (only large negative)
    return np.exp(x) / (1 + np.exp(x))


def sigmoid_inverse(y):
    return -np.log(1 / y - 1)


def calc_pre_median(frame, traces, pre_median_frames, elecs=slice(None)):
    """
    Get the preceeding mean and median of traces before :time:

    Says "mean" but is actually "median" (because it was changed to well after it was first implemented as mean)

    1/13/24 - median has already been extracted from window
    """
    # frame = round(time * SAMP_FREQ)
    # pre_frames = round(pre_ms * SAMP_FREQ)
    window = traces[elecs, max(0, frame - pre_median_frames) : frame]

    # means = np.median(window, axis=1) # np.mean(window, axis=1)

    # median = np.median(np.abs(window - means[:, None]), axis=1)
    median = np.median(np.abs(window), axis=1)

    # median = np.abs(np.median(np.abs(window), axis=1) - means)  # Faster to subtract means after finding median, but this is inaccurate due to taking abs val of window
    # return means, np.clip(median / 0.6745, a_min=0.5, a_max=None)
    return np.clip(median / 0.6745, a_min=0.5, a_max=None)


def order_sequences(sequences):
    # Order sequences based on root_elec idx
    ordered = sorted(sequences, key=lambda seq: seq.root_elec)
    for idx, seq in enumerate(ordered):
        seq.idx = idx
    return ordered


def get_gpu_free_memory():
    if torch is None or not torch.cuda.is_available():
        return 0
    try:
        import pynvml
    except ImportError:
        return 0
    pynvml.nvmlInit()
    try:
        device_index = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = info.free
    finally:
        pynvml.nvmlShutdown()
    return free_memory / (1024**3)  # GB


def set_warning_level(ignore_warnings=False):
    if ignore_warnings:
        warnings.filterwarnings("ignore")
    else:
        warnings.filterwarnings("default")


# endregion
