"""Spike sorter backend registry.

Maps sorter names to their backend classes. Backends are imported
lazily to avoid requiring all sorter dependencies at import time.
"""

# Registry: sorter name → (module_path, class_name)
_BACKEND_REGISTRY = {
    "kilosort2": (".kilosort2", "Kilosort2Backend"),
    "kilosort4": (".kilosort4", "Kilosort4Backend"),
    "rt_sort": (".rt_sort", "RTSortBackend"),
}


def get_backend_class(sorter_name):
    """Look up and import the backend class for a sorter name.

    Parameters:
        sorter_name (str): Registered sorter name (e.g. ``"kilosort2"``).

    Returns:
        cls: The ``SorterBackend`` subclass.

    Raises:
        ValueError: If the sorter name is not registered.
    """
    if sorter_name not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY.keys()))
        raise ValueError(
            f"Unknown sorter '{sorter_name}'. " f"Available sorters: {available}"
        )
    module_path, class_name = _BACKEND_REGISTRY[sorter_name]
    import importlib

    mod = importlib.import_module(module_path, package=__name__)
    return getattr(mod, class_name)


def list_sorters():
    """Return the list of registered sorter names.

    Returns:
        sorters (list of str): Available sorter names.
    """
    return sorted(_BACKEND_REGISTRY.keys())
