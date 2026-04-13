"""Stimulation-aware spike sorting pipeline.

This subpackage provides tools for sorting spikes in stimulation
recordings using a pre-trained RT-Sort model.  The workflow is:

1. Recenter logged stim times to actual artifact peaks
   (``recenter_stim_times``).
2. Remove stimulation artifacts while preserving neural spikes
   (``remove_stim_artifacts``).
3. Sort the cleaned recording with the pre-trained RT-Sort sequences
   and align to stim events (``sort_stim_recording``).

Public entry points are imported lazily so that importing this
subpackage does not pull in heavy dependencies (torch, spikeinterface)
unless the functions are actually called.
"""

__all__ = [
    "sort_stim_recording",
    "remove_stim_artifacts",
    "recenter_stim_times",
]


def __getattr__(name):
    if name == "sort_stim_recording":
        from .pipeline import sort_stim_recording

        return sort_stim_recording

    if name == "remove_stim_artifacts":
        from .artifact_removal import remove_stim_artifacts

        return remove_stim_artifacts

    if name == "recenter_stim_times":
        from .recentering import recenter_stim_times

        return recenter_stim_times

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
