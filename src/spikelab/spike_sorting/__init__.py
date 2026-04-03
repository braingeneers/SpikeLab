__all__ = ["sort_with_kilosort2", "sort_maxtwo_multiwell"]

_LAZY_IMPORTS = {
    "sort_with_kilosort2": "sort_with_kilosort2",
    "sort_maxtwo_multiwell": "sort_maxtwo_multiwell",
}


def __getattr__(name):
    """
    Lazily provide spike-sorting helpers so that importing ``spikelab.spike_sorting``
    does not require all optional spike-sorting dependencies to be installed.
    """
    if name in _LAZY_IMPORTS:
        try:
            from . import kilosort2 as _mod  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                f"{name} requires optional spike-sorting dependencies "
                "(e.g. 'spikeinterface', 'h5py', and possibly 'braindance'). "
                "Install the spike-sorting extras to use this functionality."
            ) from exc
        attr = getattr(_mod, _LAZY_IMPORTS[name])
        globals()[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
