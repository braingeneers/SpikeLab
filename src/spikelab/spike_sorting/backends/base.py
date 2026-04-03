"""Abstract base class for spike sorter backends.

Each backend implements the three-step pipeline: load recording, run
sorter, extract waveforms.  The pipeline module (``pipeline.py``)
calls these methods and handles everything downstream (SpikeData
conversion, curation, compilation, figures).

To add a new sorter:

1. Create a new module in ``backends/`` (e.g. ``kilosort4.py``).
2. Subclass ``SorterBackend`` and implement all three methods.
3. Register the backend in ``backends/__init__.py``.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..config import SortingPipelineConfig


class SorterBackend(ABC):
    """Interface that each spike sorter backend must implement.

    Parameters:
        config (SortingPipelineConfig): Full pipeline configuration.
            Backends read their relevant sub-configs (``config.recording``,
            ``config.sorter``, ``config.waveform``, ``config.execution``).
    """

    def __init__(self, config: SortingPipelineConfig):
        self.config = config

    @abstractmethod
    def load_recording(self, rec_path: Any):
        """Load and preprocess a single recording.

        Handles format-specific loading (Maxwell ``.h5``, NWB, etc.),
        gain/offset scaling, and bandpass filtering.

        Parameters:
            rec_path: Path to a recording file, a directory of files
                to concatenate, or a pre-loaded BaseRecording object.

        Returns:
            recording: A SpikeInterface ``BaseRecording`` ready for
                sorting (scaled, filtered, single-segment).
        """

    @abstractmethod
    def sort(self, recording, rec_path, output_folder):
        """Run the spike sorter on a preprocessed recording.

        Parameters:
            recording: SpikeInterface ``BaseRecording`` from
                ``load_recording``.
            rec_path: Original recording file path (for binary
                conversion or metadata).
            output_folder (Path): Directory for sorter output files.

        Returns:
            sorting: A SpikeInterface ``BaseSorting`` with detected
                units and spike trains.
        """

    @abstractmethod
    def extract_waveforms(self, recording, sorting, waveforms_folder, curation_folder):
        """Extract per-unit waveforms and compute templates.

        Parameters:
            recording: SpikeInterface ``BaseRecording``.
            sorting: SpikeInterface ``BaseSorting`` from ``sort``.
            waveforms_folder (Path): Root directory for waveform
                storage.
            curation_folder (Path): Directory for initial unit list
                and metadata.

        Returns:
            waveform_extractor: An object providing at minimum:

                - ``sorting`` — the sorting object (possibly with
                  centered spike times)
                - ``recording`` — the recording object
                - ``sampling_frequency`` — float
                - ``peak_ind`` — int (peak sample index in template)
                - ``chans_max_all`` — dict or array mapping unit_id
                  to max-amplitude channel index
                - ``use_pos_peak`` — dict or array mapping unit_id
                  to bool (polarity)
                - ``get_computed_template(unit_id, mode)`` — returns
                  ``(n_samples, n_channels)`` template array
                - ``ms_to_samples(ms)`` — time conversion
                - ``root_folder`` — Path to waveform files

              This can be the custom ``WaveformExtractor`` (Kilosort2
              backend) or a wrapper around SpikeInterface's
              ``WaveformExtractor`` (future backends).
        """

    def write_recording(self, recording, dat_path):
        """Convert a recording to the binary format needed by the sorter.

        Not all sorters need this (some read recordings directly via
        SpikeInterface).  The default implementation is a no-op.

        Parameters:
            recording: SpikeInterface ``BaseRecording``.
            dat_path (Path): Output binary file path.
        """
        pass
