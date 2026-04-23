"""Configuration dataclass for the spike sorting pipeline.

Replaces the ~80 module-level globals in kilosort2.py with a single
typed, inspectable configuration object that is passed explicitly to
every pipeline function.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RecordingConfig:
    """Parameters for recording loading and preprocessing."""

    stream_id: Optional[str] = None
    hdf5_plugin_path: Optional[str] = None
    first_n_mins: Optional[float] = None
    mea_y_max: Optional[int] = None
    gain_to_uv: Optional[float] = None
    offset_to_uv: Optional[float] = None
    rec_chunks: List[Tuple[int, int]] = field(default_factory=list)
    rec_chunks_s: List[Tuple[float, float]] = field(default_factory=list)
    start_time_s: Optional[float] = None
    end_time_s: Optional[float] = None
    freq_min: int = 300
    freq_max: int = 6000


@dataclass
class SorterConfig:
    """Parameters for the spike sorter itself."""

    sorter_name: str = "kilosort2"
    sorter_path: Optional[str] = None
    sorter_params: Optional[Dict[str, Any]] = None
    use_docker: bool = False


@dataclass
class RTSortConfig:
    """Parameters for the RT-Sort detection and sorting backend.

    RT-Sort is an action-potential-propagation-based spike sorter using
    a deep learning detection model followed by codetection clustering
    and template matching.  See van der Molen, Lim et al. 2024
    (PLOS ONE, DOI: 10.1371/journal.pone.0312438) for algorithmic
    details.

    Parameters:
        model_path (str or None): Path to a folder containing
            ``init_dict.json`` and ``state_dict.pt`` for a pretrained
            ``ModelSpikeSorter``.  When None, the bundled model
            corresponding to ``probe`` is loaded.
        probe (str): Which bundled pretrained model to use when
            ``model_path`` is None.  ``"mea"`` or ``"neuropixels"``.
        device (str): PyTorch device for inference.  ``"cuda"`` or
            ``"cpu"``.
        num_processes (int or None): Number of worker processes for
            parallel detection/clustering stages.  None selects an
            automatic value based on CPU count.
        recording_window_ms (tuple or None): ``(start_ms, end_ms)``
            window of the recording to process.  None processes the
            entire recording.
        save_rt_sort_pickle (bool): If True, serialize the final
            ``RTSort`` object to the sorter output folder so the
            trained sequences can be re-used in Phase 2 stim-aware
            sorting.
        delete_inter (bool): If True, delete the intermediate cache
            directory after sorting completes.
        verbose (bool): Print progress messages during sorting.
        params (dict or None): Override dictionary merged into the
            RT-Sort parameter set.  Takes precedence over the preset
            defaults; useful for one-off tuning without editing a
            preset.  Keys must match ``detect_sequences`` parameter
            names.
        detection_window_s (float or None): If set, run sequence
            detection on only the first ``detection_window_s`` seconds
            of the recording (the heavy GPU + clustering phase), then
            apply the resulting sequences to the full recording during
            ``sort_offline``.  Decouples the detection-phase memory
            ceiling from total recording length.  ``None`` uses the
            full window for both phases (legacy behavior).
    """

    model_path: Optional[str] = None
    probe: str = "mea"
    device: str = "cuda"
    num_processes: Optional[int] = None
    recording_window_ms: Optional[Any] = None
    save_rt_sort_pickle: bool = True
    delete_inter: bool = False
    verbose: bool = True
    params: Optional[Dict[str, Any]] = None
    detection_window_s: Optional[float] = None


@dataclass
class WaveformConfig:
    """Parameters for waveform extraction and template computation.

    Memory-budget note: the default extractor pre-allocates one
    ``(n_spikes, nsamples, num_channels)`` ``.npy`` memmap per unit
    before extraction begins.  For high-unit-count sorters on
    high-density MEAs this grows to tens of GB (e.g. 400 units ×
    1018 channels = ~39 GB).  When that exceeds host RAM, set
    ``streaming=True`` to use a one-unit-at-a-time path that
    discards each unit's waveforms after templates and metrics are
    computed — peak RAM becomes one unit's buffer (~100 MB for
    MaxOne) regardless of total unit count.  Waveform files are
    only written when ``save_waveform_files=True``.
    """

    ms_before: float = 2.0
    ms_after: float = 2.0
    pos_peak_thresh: float = 2.0
    max_waveforms_per_unit: int = 300
    compiled_ms_before: float = 2.0
    compiled_ms_after: float = 2.0
    scale_compiled_waveforms: bool = True
    std_at_peak: bool = True
    std_over_window_ms_before: float = 0.5
    std_over_window_ms_after: float = 1.5
    streaming: bool = True
    save_waveform_files: bool = True


@dataclass
class CurationConfig:
    """Parameters for unit quality-control curation."""

    curate_first: bool = True
    curate_second: bool = True
    curation_epoch: Optional[int] = None
    fr_min: Optional[float] = 0.05
    isi_viol_max: Optional[float] = 1.0
    isi_violation_method: str = "percent"
    snr_min: Optional[float] = 5.0
    spikes_min_first: Optional[int] = 30
    spikes_min_second: Optional[int] = 50
    std_norm_max: Optional[float] = 1.0


@dataclass
class CompilationConfig:
    """Parameters for result compilation and export."""

    compile_single_recording: bool = True
    compile_to_mat: bool = False
    compile_to_npz: bool = True
    compile_waveforms: bool = False

    save_electrodes: bool = True
    save_spike_times: bool = True
    save_raw_pkl: bool = False
    save_dl_data: bool = False


@dataclass
class FigureConfig:
    """Parameters for QC figure generation."""

    create_figures: bool = False
    create_unit_figures: bool = False
    dpi: Optional[int] = None
    font_size: int = 12
    bar_x_label: str = "Recording"
    bar_y_label: str = "Number of Units"
    bar_label_rotation: int = 0
    bar_total_label: str = "First Curation"
    bar_selected_label: str = "Selected Curation"
    scatter_std_max_units_per_recording: Optional[int] = None
    scatter_recording_colors: List[str] = field(
        default_factory=lambda: [
            "#f74343",
            "#fccd56",
            "#74fc56",
            "#56fcf6",
            "#1e1efa",
            "#fa1ed2",
        ]
    )
    scatter_recording_alpha: float = 1.0
    scatter_x_label: str = "Number of Spikes"
    scatter_y_label: str = "avg. STD / amplitude"
    scatter_x_max_buffer: float = 300.0
    scatter_y_max_buffer: float = 0.2
    templates_color_curated: str = "#000000"
    templates_color_failed: str = "#FF0000"
    templates_per_column: int = 50
    templates_y_spacing: float = 50.0
    templates_y_lim_buffer: float = 10.0
    templates_window_ms_before: float = 5.0
    templates_window_ms_after: float = 5.0
    templates_line_ms_before: Optional[float] = 1.0
    templates_line_ms_after: Optional[float] = 4.0
    templates_x_label: str = "Time Rel. to Peak (ms)"



@dataclass
class ExecutionConfig:
    """Parameters for pipeline execution control."""

    n_jobs: int = 8
    total_memory: str = "16G"
    use_parallel_processing_for_raw_conversion: bool = True
    save_script: bool = False
    out_file: str = "sort_with_kilosort2.out"
    random_seed: int = 1
    recompute_recording: bool = False
    recompute_sorting: bool = False
    reextract_waveforms: bool = False
    recurate_first: bool = False
    recurate_second: bool = False
    recompile_single_recording: bool = False

    delete_inter: bool = True


@dataclass
class SortingPipelineConfig:
    """Complete configuration for a spike sorting pipeline run.

    Groups all parameters into typed sub-configs. Passed explicitly to
    every pipeline function, replacing module-level globals.

    Parameters:
        recording (RecordingConfig): Recording loading and preprocessing.
        sorter (SorterConfig): Spike sorter selection and parameters.
        rt_sort (RTSortConfig): RT-Sort specific parameters (only used
            when ``sorter.sorter_name == "rt_sort"``).
        waveform (WaveformConfig): Waveform extraction and templates.
        curation (CurationConfig): Unit quality-control filters.
        compilation (CompilationConfig): Result export options.
        figures (FigureConfig): QC figure generation.
        execution (ExecutionConfig): Pipeline control and parallelism.
    """

    recording: RecordingConfig = field(default_factory=RecordingConfig)
    sorter: SorterConfig = field(default_factory=SorterConfig)
    rt_sort: RTSortConfig = field(default_factory=RTSortConfig)
    waveform: WaveformConfig = field(default_factory=WaveformConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    compilation: CompilationConfig = field(default_factory=CompilationConfig)
    figures: FigureConfig = field(default_factory=FigureConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Build a config from flat keyword arguments.

        Maps the flat parameter names used by ``sort_with_kilosort2()``
        to the nested sub-config fields. Unknown keys raise ``TypeError``.

        Parameters:
            **kwargs: Flat keyword arguments matching
                ``sort_with_kilosort2()`` parameter names.

        Returns:
            config (SortingPipelineConfig): Populated configuration.
        """
        flat_map = cls._build_flat_map()

        sub_kwargs = {
            "recording": {},
            "sorter": {},
            "rt_sort": {},
            "waveform": {},
            "curation": {},
            "compilation": {},
            "figures": {},
            "execution": {},
        }

        for key, value in kwargs.items():
            if key in flat_map:
                group, field_name = flat_map[key]
                sub_kwargs[group][field_name] = value
            else:
                raise TypeError(
                    f"Unknown parameter '{key}'. Check spelling or see "
                    "SortingPipelineConfig for valid fields."
                )

        return cls(
            recording=RecordingConfig(**sub_kwargs["recording"]),
            sorter=SorterConfig(**sub_kwargs["sorter"]),
            rt_sort=RTSortConfig(**sub_kwargs["rt_sort"]),
            waveform=WaveformConfig(**sub_kwargs["waveform"]),
            curation=CurationConfig(**sub_kwargs["curation"]),
            compilation=CompilationConfig(**sub_kwargs["compilation"]),
            figures=FigureConfig(**sub_kwargs["figures"]),
            execution=ExecutionConfig(**sub_kwargs["execution"]),
        )

    def override(self, **kwargs):
        """Return a copy of this config with selected fields overridden.

        Accepts the same flat keyword arguments as ``from_kwargs()``.
        Unspecified fields retain their current values.

        Parameters:
            **kwargs: Flat keyword arguments to override.

        Returns:
            config (SortingPipelineConfig): New config with overrides.
        """
        from copy import deepcopy

        new = deepcopy(self)
        flat_map = self._build_flat_map()

        for key, value in kwargs.items():
            if key not in flat_map:
                raise TypeError(
                    f"Unknown parameter '{key}'. Check spelling or see "
                    "SortingPipelineConfig for valid fields."
                )
            group, field_name = flat_map[key]
            sub_config = getattr(new, group)
            setattr(sub_config, field_name, value)

        return new

    @staticmethod
    def _build_flat_map():
        """Return the flat kwarg → (group, field) mapping."""
        return {
            # RecordingConfig
            "stream_id": ("recording", "stream_id"),
            "hdf5_plugin_path": ("recording", "hdf5_plugin_path"),
            "first_n_mins": ("recording", "first_n_mins"),
            "mea_y_max": ("recording", "mea_y_max"),
            "gain_to_uv": ("recording", "gain_to_uv"),
            "offset_to_uv": ("recording", "offset_to_uv"),
            "rec_chunks": ("recording", "rec_chunks"),
            "rec_chunks_s": ("recording", "rec_chunks_s"),
            "start_time_s": ("recording", "start_time_s"),
            "end_time_s": ("recording", "end_time_s"),
            "freq_min": ("recording", "freq_min"),
            "freq_max": ("recording", "freq_max"),
            # SorterConfig
            "kilosort_path": ("sorter", "sorter_path"),
            "kilosort_params": ("sorter", "sorter_params"),
            "use_docker": ("sorter", "use_docker"),
            # RTSortConfig
            "rt_sort_model_path": ("rt_sort", "model_path"),
            "rt_sort_probe": ("rt_sort", "probe"),
            "rt_sort_device": ("rt_sort", "device"),
            "rt_sort_num_processes": ("rt_sort", "num_processes"),
            "rt_sort_recording_window_ms": ("rt_sort", "recording_window_ms"),
            "rt_sort_save_pickle": ("rt_sort", "save_rt_sort_pickle"),
            "rt_sort_delete_inter": ("rt_sort", "delete_inter"),
            "rt_sort_verbose": ("rt_sort", "verbose"),
            "rt_sort_params": ("rt_sort", "params"),
            "rt_sort_detection_window_s": ("rt_sort", "detection_window_s"),
            # WaveformConfig
            "waveforms_ms_before": ("waveform", "ms_before"),
            "waveforms_ms_after": ("waveform", "ms_after"),
            "pos_peak_thresh": ("waveform", "pos_peak_thresh"),
            "max_waveforms_per_unit": ("waveform", "max_waveforms_per_unit"),
            "compiled_waveforms_ms_before": ("waveform", "compiled_ms_before"),
            "compiled_waveforms_ms_after": ("waveform", "compiled_ms_after"),
            "scale_compiled_waveforms": ("waveform", "scale_compiled_waveforms"),
            "std_at_peak": ("waveform", "std_at_peak"),
            "std_over_window_ms_before": ("waveform", "std_over_window_ms_before"),
            "std_over_window_ms_after": ("waveform", "std_over_window_ms_after"),
            "streaming_waveforms": ("waveform", "streaming"),
            "save_waveform_files": ("waveform", "save_waveform_files"),
            # CurationConfig
            "curate_first": ("curation", "curate_first"),
            "curate_second": ("curation", "curate_second"),
            "curation_epoch": ("curation", "curation_epoch"),
            "fr_min": ("curation", "fr_min"),
            "isi_viol_max": ("curation", "isi_viol_max"),
            "isi_violation_method": ("curation", "isi_violation_method"),
            "snr_min": ("curation", "snr_min"),
            "spikes_min_first": ("curation", "spikes_min_first"),
            "spikes_min_second": ("curation", "spikes_min_second"),
            "std_norm_max": ("curation", "std_norm_max"),
            # CompilationConfig
            "compile_single_recording": ("compilation", "compile_single_recording"),
            "compile_to_mat": ("compilation", "compile_to_mat"),
            "compile_to_npz": ("compilation", "compile_to_npz"),
            "compile_waveforms": ("compilation", "compile_waveforms"),
            "save_electrodes": ("compilation", "save_electrodes"),
            "save_spike_times": ("compilation", "save_spike_times"),
            "save_raw_pkl": ("compilation", "save_raw_pkl"),
            "save_dl_data": ("compilation", "save_dl_data"),
            # FigureConfig
            "create_figures": ("figures", "create_figures"),
            "create_unit_figures": ("figures", "create_unit_figures"),
            "figures_dpi": ("figures", "dpi"),
            "figures_font_size": ("figures", "font_size"),
            "bar_x_label": ("figures", "bar_x_label"),
            "bar_y_label": ("figures", "bar_y_label"),
            "bar_label_rotation": ("figures", "bar_label_rotation"),
            "bar_total_label": ("figures", "bar_total_label"),
            "bar_selected_label": ("figures", "bar_selected_label"),
            "scatter_std_max_units_per_recording": (
                "figures",
                "scatter_std_max_units_per_recording",
            ),
            "scatter_recording_colors": ("figures", "scatter_recording_colors"),
            "scatter_recording_alpha": ("figures", "scatter_recording_alpha"),
            "scatter_x_label": ("figures", "scatter_x_label"),
            "scatter_y_label": ("figures", "scatter_y_label"),
            "scatter_x_max_buffer": ("figures", "scatter_x_max_buffer"),
            "scatter_y_max_buffer": ("figures", "scatter_y_max_buffer"),
            "all_templates_color_curated": ("figures", "templates_color_curated"),
            "all_templates_color_failed": ("figures", "templates_color_failed"),
            "all_templates_per_column": ("figures", "templates_per_column"),
            "all_templates_y_spacing": ("figures", "templates_y_spacing"),
            "all_templates_y_lim_buffer": ("figures", "templates_y_lim_buffer"),
            "all_templates_window_ms_before_peak": (
                "figures",
                "templates_window_ms_before",
            ),
            "all_templates_window_ms_after_peak": (
                "figures",
                "templates_window_ms_after",
            ),
            "all_templates_line_ms_before_peak": (
                "figures",
                "templates_line_ms_before",
            ),
            "all_templates_line_ms_after_peak": (
                "figures",
                "templates_line_ms_after",
            ),
            "all_templates_x_label": ("figures", "templates_x_label"),
            # ExecutionConfig
            "random_seed": ("execution", "random_seed"),
            "n_jobs": ("execution", "n_jobs"),
            "total_memory": ("execution", "total_memory"),
            "use_parallel_processing_for_raw_conversion": (
                "execution",
                "use_parallel_processing_for_raw_conversion",
            ),
            "save_script": ("execution", "save_script"),
            "out_file": ("execution", "out_file"),
            "recompute_recording": ("execution", "recompute_recording"),
            "recompute_sorting": ("execution", "recompute_sorting"),
            "reextract_waveforms": ("execution", "reextract_waveforms"),
            "recurate_first": ("execution", "recurate_first"),
            "recurate_second": ("execution", "recurate_second"),
            "recompile_single_recording": (
                "execution",
                "recompile_single_recording",
            ),
            "delete_inter": ("execution", "delete_inter"),
        }


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

#: Default configuration for Kilosort2.
#: Parameters are compatible with Maxwell MEA and other probe types.
#: Hardware-specific presets can be created by overriding parameters.
KILOSORT2 = SortingPipelineConfig()

#: Kilosort2 with Docker (no local MATLAB needed).
KILOSORT2_DOCKER = SortingPipelineConfig(
    sorter=SorterConfig(sorter_name="kilosort2", use_docker=True),
)

#: Default configuration for Kilosort4.
#: Kilosort4 is pure Python (PyTorch) — no MATLAB required.
#: Default parameters are tuned for Neuropixels probes but work for
#: other probe types.  Hardware-specific presets (e.g. for Maxwell
#: MEAs) can be created by overriding detection/filtering parameters.
KILOSORT4 = SortingPipelineConfig(
    sorter=SorterConfig(sorter_name="kilosort4"),
)

#: Kilosort4 with Docker.
KILOSORT4_DOCKER = SortingPipelineConfig(
    sorter=SorterConfig(sorter_name="kilosort4", use_docker=True),
)

#: RT-Sort with the bundled MEA detection model.
#: Uses the propagation-based RT-Sort algorithm (van der Molen, Lim et
#: al. 2024, PLOS ONE) with the pretrained model tuned for Maxwell
#: multi-electrode arrays.
RT_SORT_MEA = SortingPipelineConfig(
    sorter=SorterConfig(sorter_name="rt_sort"),
    rt_sort=RTSortConfig(probe="mea"),
)

#: RT-Sort with the bundled Neuropixels detection model.
#: Uses Neuropixels-tuned detection thresholds and merge parameters.
RT_SORT_NEUROPIXELS = SortingPipelineConfig(
    sorter=SorterConfig(sorter_name="rt_sort"),
    rt_sort=RTSortConfig(
        probe="neuropixels",
        params={
            "stringent_thresh": 0.175,
            "loose_thresh": 0.075,
            "inference_scaling_numerator": 15.4,
            "min_amp_dist_p": 0.1,
            "max_latency_diff_spikes": 2.5,
            "max_amp_median_diff_spikes": 0.45,
            "max_latency_diff_sequences": 2.5,
            "max_amp_median_diff_sequences": 0.45,
            "max_root_amp_median_std_sequences": 2.5,
        },
    ),
)
