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
class WaveformConfig:
    """Parameters for waveform extraction and template computation."""

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
    dpi: Optional[int] = None
    font_size: int = 12
    bar_x_label: str = "Recording"
    bar_y_label: str = "Number of Units"
    bar_label_rotation: int = 0
    bar_total_label: str = "First Curation"
    bar_selected_label: str = "Selected Curation"
    scatter_std_max_units_per_recording: Optional[int] = None
    scatter_recording_colors: Optional[List[str]] = None
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

    def __post_init__(self) -> None:
        if self.scatter_recording_colors is None:
            self.scatter_recording_colors = [
                "#f74343",
                "#fccd56",
                "#74fc56",
                "#56fcf6",
                "#1e1efa",
                "#fa1ed2",
            ]


@dataclass
class ExecutionConfig:
    """Parameters for pipeline execution control."""

    n_jobs: int = 8
    total_memory: str = "16G"
    use_parallel_processing_for_raw_conversion: bool = True
    save_script: bool = False
    out_file: str = "sort_with_kilosort2.out"
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
        waveform (WaveformConfig): Waveform extraction and templates.
        curation (CurationConfig): Unit quality-control filters.
        compilation (CompilationConfig): Result export options.
        figures (FigureConfig): QC figure generation.
        execution (ExecutionConfig): Pipeline control and parallelism.
    """

    recording: RecordingConfig = field(default_factory=RecordingConfig)
    sorter: SorterConfig = field(default_factory=SorterConfig)
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
            "freq_min": ("recording", "freq_min"),
            "freq_max": ("recording", "freq_max"),
            # SorterConfig
            "kilosort_path": ("sorter", "sorter_path"),
            "kilosort_params": ("sorter", "sorter_params"),
            "use_docker": ("sorter", "use_docker"),
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
