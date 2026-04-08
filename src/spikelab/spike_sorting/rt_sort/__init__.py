"""RT-Sort algorithm subpackage.

Vendored from the RT-Sort algorithm by van der Molen, Lim et al. 2024
(PLoS ONE, https://doi.org/10.1371/journal.pone.0312438), originally
released at https://github.com/KosikLabUCSB/RT-Sort under the MIT
license.  See ``NOTICE.md`` for the full attribution.

Public entry points are imported lazily so that importing
``spikelab.spike_sorting`` does not require PyTorch or other RT-Sort
runtime dependencies unless RT-Sort is actually invoked.
"""

from pathlib import Path

__all__ = [
    "detect_sequences",
    "RTSort",
    "load_detection_model",
    "DEFAULT_MEA_MODEL_PATH",
    "DEFAULT_NEUROPIXELS_MODEL_PATH",
    "NEUROPIXELS_PARAMS",
]

_MODELS_ROOT = Path(__file__).parent / "detection_models"
DEFAULT_MEA_MODEL_PATH = _MODELS_ROOT / "mea"
DEFAULT_NEUROPIXELS_MODEL_PATH = _MODELS_ROOT / "neuropixels"


def __getattr__(name):
    """Lazy-import public API symbols from the algorithm module."""
    if name in ("detect_sequences", "RTSort", "NEUROPIXELS_PARAMS"):
        from . import _algorithm

        if name == "NEUROPIXELS_PARAMS":
            return _algorithm.neuropixels_params
        return getattr(_algorithm, name)

    if name == "load_detection_model":
        from .model import ModelSpikeSorter

        def load_detection_model(path=None, *, probe="mea"):
            """Load a pretrained RT-Sort detection model.

            Parameters:
                path (str or Path or None): Explicit path to a folder
                    containing ``init_dict.json`` and ``state_dict.pt``.
                    If None, the pretrained model bundled with SpikeLab
                    is loaded based on ``probe``.
                probe (str): Which bundled model to load when ``path``
                    is None.  One of ``"mea"`` or ``"neuropixels"``.

            Returns:
                model (ModelSpikeSorter): The loaded spike detection
                    model.
            """
            if path is None:
                if probe == "mea":
                    path = DEFAULT_MEA_MODEL_PATH
                elif probe == "neuropixels":
                    path = DEFAULT_NEUROPIXELS_MODEL_PATH
                else:
                    raise ValueError(
                        f"Unknown probe {probe!r}; expected 'mea' or 'neuropixels'."
                    )
            return ModelSpikeSorter.load(Path(path))

        return load_detection_model

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
