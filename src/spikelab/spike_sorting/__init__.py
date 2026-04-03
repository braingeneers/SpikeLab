__all__ = [
    "sort_recording",
    "sort_multistream",
    "sort_with_kilosort2",
    "sort_maxtwo_multiwell",
]

_LAZY_IMPORTS = {
    # New generic entry points (no sorter-specific dependencies)
    "sort_recording": ("pipeline", "sort_recording"),
    "sort_multistream": ("pipeline", "sort_multistream"),
    # Legacy entry points (Kilosort2-specific, kept for backward compat)
    "sort_with_kilosort2": ("kilosort2", "sort_with_kilosort2"),
    "sort_maxtwo_multiwell": ("kilosort2", "sort_maxtwo_multiwell"),
}


def __getattr__(name):
    """
    Lazily provide spike-sorting helpers so that importing ``spikelab.spike_sorting``
    does not require all optional spike-sorting dependencies to be installed.
    """
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        try:
            import importlib

            mod = importlib.import_module(f".{module_name}", package=__name__)
        except ImportError as exc:
            raise ImportError(
                f"{name} requires optional spike-sorting dependencies "
                "(e.g. 'spikeinterface', 'h5py', and possibly 'braindance'). "
                "Install the spike-sorting extras to use this functionality."
            ) from exc
        attr = getattr(mod, attr_name)
        globals()[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
