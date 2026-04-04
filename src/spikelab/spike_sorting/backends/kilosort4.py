"""Kilosort4 sorter backend.

Implements the ``SorterBackend`` interface using Kilosort4 (pure Python,
PyTorch-based) via SpikeInterface's ``run_sorter("kilosort4", ...)``.
Uses the same custom ``WaveformExtractor`` and per-spike centering as
the Kilosort2 backend.

Requirements:
    pip install kilosort
    # Plus PyTorch with CUDA — see https://pytorch.org/get-started/locally/
"""

from typing import Any

from ..config import SortingPipelineConfig
from .base import SorterBackend

DEFAULT_KILOSORT4_PARAMS = {
    "do_CAR": True,
    "invert_sign": False,
    "save_extra_vars": False,
    "save_preprocessed_copy": False,
    "torch_device": "auto",
    "bad_channels": None,
    "clear_cache": False,
    "do_correction": True,
    "skip_kilosort_preprocessing": False,
    "keep_good_only": False,
    "use_binary_file": True,
    "delete_recording_dat": True,
}
"""Default Kilosort4 parameters.  Originally tuned for Neuropixels probes.
Used by the backend and ``KILOSORT4_NEUROPIXELS`` preset config."""


class Kilosort4Backend(SorterBackend):
    """SorterBackend implementation for Kilosort4.

    Kilosort4 is a pure Python spike sorter (no MATLAB required).  It
    runs via SpikeInterface's ``run_sorter("kilosort4", ...)`` interface,
    which handles binary conversion, parameter passing, and result
    loading.

    Waveform extraction uses the same custom ``WaveformExtractor`` from
    ``kilosort2.py`` with per-spike centering, since the output format
    (``spike_times.npy``, ``spike_clusters.npy``, ``templates.npy``) is
    compatible.

    Parameters:
        config (SortingPipelineConfig): Full pipeline configuration.
    """

    def __init__(self, config: SortingPipelineConfig) -> None:
        super().__init__(config)
        self._sync_globals()

    def _sync_globals(self) -> None:
        """Set module-level globals in kilosort2.py from the config.

        The legacy WaveformExtractor, load_recording, and related
        functions still read globals.  This bridges the config-based
        architecture with those functions.
        """
        from .. import kilosort2 as ks2

        cfg = self.config
        rec = cfg.recording
        sor = cfg.sorter
        wf = cfg.waveform
        cur = cfg.curation
        comp = cfg.compilation
        exe = cfg.execution

        # Recording
        ks2.STREAM_ID = rec.stream_id
        ks2.FIRST_N_MINS = rec.first_n_mins
        ks2.MEA_Y_MAX = rec.mea_y_max
        ks2.GAIN_TO_UV = rec.gain_to_uv
        ks2.OFFSET_TO_UV = rec.offset_to_uv
        ks2.REC_CHUNKS = list(rec.rec_chunks)
        ks2._REC_CHUNK_NAMES = []
        ks2.FREQ_MIN = rec.freq_min
        ks2.FREQ_MAX = rec.freq_max

        # Sorter — KS4 params are passed directly to SI, but we store
        # them in the global for the KilosortSortingExtractor to read
        ks2.KILOSORT_PATH = sor.sorter_path
        ks2.KILOSORT_PARAMS = {
            **DEFAULT_KILOSORT4_PARAMS,
            **(sor.sorter_params or {}),
        }
        ks2.USE_DOCKER = sor.use_docker

        # Waveforms
        ks2.WAVEFORMS_MS_BEFORE = wf.ms_before
        ks2.WAVEFORMS_MS_AFTER = wf.ms_after
        ks2.POS_PEAK_THRESH = wf.pos_peak_thresh
        ks2.MAX_WAVEFORMS_PER_UNIT = wf.max_waveforms_per_unit
        ks2.COMPILED_WAVEFORMS_MS_BEFORE = wf.compiled_ms_before
        ks2.COMPILED_WAVEFORMS_MS_AFTER = wf.compiled_ms_after
        ks2.SCALE_COMPILED_WAVEFORMS = wf.scale_compiled_waveforms
        ks2.STD_AT_PEAK = wf.std_at_peak
        ks2.STD_OVER_WINDOW_MS_BEFORE = wf.std_over_window_ms_before
        ks2.STD_OVER_WINDOW_MS_AFTER = wf.std_over_window_ms_after

        # Curation
        ks2.CURATE_FIRST = cur.curate_first
        ks2.CURATE_SECOND = cur.curate_second
        ks2.CURATION_EPOCH = cur.curation_epoch
        ks2.FR_MIN = cur.fr_min
        ks2.ISI_VIOL_MAX = cur.isi_viol_max
        ks2.ISI_VIOLATION_METHOD = cur.isi_violation_method
        ks2.SNR_MIN = cur.snr_min
        ks2.SPIKES_MIN_FIRST = cur.spikes_min_first
        ks2.SPIKES_MIN_SECOND = cur.spikes_min_second
        ks2.STD_NORM_MAX = cur.std_norm_max

        # Compilation
        ks2.COMPILE_SINGLE_RECORDING = comp.compile_single_recording
        ks2.COMPILE_TO_MAT = comp.compile_to_mat
        ks2.COMPILE_TO_NPZ = comp.compile_to_npz
        ks2.COMPILE_WAVEFORMS = comp.compile_waveforms
        ks2.SAVE_ELECTRODES = comp.save_electrodes
        ks2.SAVE_SPIKE_TIMES = comp.save_spike_times
        ks2.SAVE_RAW_PKL = comp.save_raw_pkl
        ks2.SAVE_DL_DATA = comp.save_dl_data

        # Execution
        ks2.N_JOBS = exe.n_jobs
        ks2.TOTAL_MEMORY = exe.total_memory
        ks2.USE_PARALLEL_PROCESSING_FOR_RAW_CONVERSION = (
            exe.use_parallel_processing_for_raw_conversion
        )
        ks2.SAVE_SCRIPT = exe.save_script
        ks2.OUT_FILE = exe.out_file
        ks2.RECOMPUTE_RECORDING = exe.recompute_recording
        ks2.RECOMPUTE_SORTING = exe.recompute_sorting
        ks2.REEXTRACT_WAVEFORMS = exe.reextract_waveforms
        ks2.RECURATE_FIRST = exe.recurate_first
        ks2.RECURATE_SECOND = exe.recurate_second
        ks2.RECOMPILE_SINGLE_RECORDING = exe.recompile_single_recording

    def load_recording(self, rec_path: Any) -> Any:
        """Load and preprocess a recording via the shared loader.

        Uses the same Maxwell/NWB loader as the Kilosort2 backend.
        """
        from .. import kilosort2 as ks2

        recording = ks2.load_recording(rec_path)

        self.rec_chunk_names = getattr(ks2, "_REC_CHUNK_NAMES", []) or []
        self.config.recording.rec_chunks = list(getattr(ks2, "REC_CHUNKS", []) or [])

        return recording

    def sort(
        self,
        recording: Any,
        rec_path: Any,
        recording_dat_path: Any,
        output_folder: Any,
    ) -> Any:
        """Run Kilosort4 spike sorting via SpikeInterface.

        Uses ``spikeinterface.sorters.run_sorter("kilosort4", ...)``
        which handles binary conversion and parameter passing.  When
        ``use_docker=True``, runs in a Docker container.
        """
        import spikeinterface.sorters as ss
        from .. import kilosort2 as ks2
        from ..sorting_utils import Stopwatch, print_stage

        print_stage("SPIKE SORTING WITH KILOSORT4")
        stopwatch = Stopwatch()

        sorter_params = dict(ks2.KILOSORT_PARAMS)

        # Check if KS4 results already exist
        output_folder_path = output_folder
        if hasattr(output_folder, "__fspath__"):
            from pathlib import Path

            output_folder_path = Path(output_folder)

        if (
            not ks2.RECOMPUTE_SORTING
            and output_folder_path.exists()
            and (output_folder_path / "spike_times.npy").exists()
        ):
            print("Loading existing Kilosort4 results")
            sorting = ks2.KilosortSortingExtractor(folder_path=output_folder_path)
            stopwatch.log_time("Done loading existing results.")
            return sorting

        try:
            docker_kwargs = {}
            if ks2.USE_DOCKER:
                from ..docker_utils import get_docker_image

                docker_kwargs["docker_image"] = (
                    ks2.USE_DOCKER
                    if isinstance(ks2.USE_DOCKER, str)
                    else get_docker_image("kilosort4")
                )
                # Use "pypi" instead of "no-install" to work around an SI
                # 0.104 bug where extra_requirements triggers an undefined
                # 'cmd' variable when installation_mode="no-install".
                # SI will detect the pre-installed version and skip the install.
                docker_kwargs["installation_mode"] = "pypi"

            sorting = ss.run_sorter(
                "kilosort4",
                recording,
                folder=str(output_folder),
                remove_existing_folder=True,
                verbose=True,
                **docker_kwargs,
                **sorter_params,
            )
        except Exception as e:
            print(f"Kilosort4 sorting failed: {e}")
            stopwatch.log_time("Sorting failed.")
            return e

        # Load results using the shared KilosortSortingExtractor
        # (KS4 output format is compatible: spike_times.npy, spike_clusters.npy)
        sorter_output = output_folder_path
        if (output_folder_path / "sorter_output").exists():
            sorter_output = output_folder_path / "sorter_output"

        sorting = ks2.KilosortSortingExtractor(folder_path=sorter_output)

        stopwatch.log_time("Done sorting with Kilosort4.")
        return sorting

    def extract_waveforms(
        self,
        recording: Any,
        sorting: Any,
        waveforms_folder: Any,
        curation_folder: Any,
        rec_path: Any = None,
    ) -> Any:
        """Extract waveforms via the custom WaveformExtractor.

        Uses the same extraction pipeline and per-spike centering as
        the Kilosort2 backend.
        """
        from .. import kilosort2 as ks2

        return ks2.extract_waveforms(
            recording_path=rec_path,
            recording=recording,
            sorting=sorting,
            root_folder=waveforms_folder,
            initial_folder=curation_folder,
            n_jobs=self.config.execution.n_jobs,
            total_memory=self.config.execution.total_memory,
            progress_bar=True,
        )
