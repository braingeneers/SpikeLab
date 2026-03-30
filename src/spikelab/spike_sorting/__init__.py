__all__ = ["sort_with_kilosort2"]


def __getattr__(name):
    """
    Lazily provide spike-sorting helpers so that importing ``spikelab.spike_sorting``
    does not require all optional spike-sorting dependencies to be installed.
    """
    if name == "sort_with_kilosort2":
        try:
            from .kilosort2 import sort_with_kilosort2  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sort_with_kilosort2 requires optional spike-sorting dependencies "
                "(e.g. 'spikeinterface', 'h5py', and possibly 'braindance'). "
                "Install the spike-sorting extras to use this functionality."
            ) from exc
        # Cache the imported symbol for future attribute lookups.
        globals()["sort_with_kilosort2"] = sort_with_kilosort2
        return sort_with_kilosort2

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
