"""
Distribution plotting: violin or box plots for lists of 1D arrays (e.g. per-unit rates per condition).

Supports:
- plot_distributions(data_list, ...) when IntegratedAnalysisTools is on PYTHONPATH.
- plot_distributions_from_workspace(workspace_id, namespace, keys, save_path, ...) to load
  arrays from workspace keys and plot.
- Run from the IntegratedAnalysisTools directory: python viz/new_plot_distributions.py (or set PYTHONPATH to IAT).

Original TODOs from old_plots/original_plot_distributions.py (addressed in IAT spikedata/plot_utils.plot_distributions):
  (1) LET USER CHOOSE BETWEEN BOXPLOTS OR VIOLIN PLOTS -> style='violin'|'box'.
  (2) LET USER CUSTOMIZE AXIS LABELS, FIG SIZE, VIOLIN/BOXPLOT COLORING, ETC. -> xlabel, ylabel, figsize, colors, edge_color, yscale, ylim, yticks, yticklabels, show_means, show_medians, alpha.
  (3) MAKE FONT SIZE SPECIFIC TO FIGURE/AX -> font_size on ax only, no rcParams.
  (4) RETURN FIGURE OR AX; ALLOW NOT PRINT OR SAVE -> returns (fig, ax); save_path, show_fig.
"""

from typing import Any, List, Optional, Tuple, Union

try:
    from spikedata.plot_utils import plot_distributions as _plot_distributions
    _HAS_IAT = True
except ImportError:
    _HAS_IAT = False
    _plot_distributions = None  # type: ignore


def plot_distributions(
    data_list: List[Any],
    ax: Optional[Any] = None,
    style: str = "violin",
    x_tick_labs: Optional[List[str]] = None,
    **kwargs: Any,
) -> Tuple[Optional[Any], Any]:
    """
    Plot distributions as violin or box plots. Wraps spikedata.plot_utils.plot_distributions
    when IntegratedAnalysisTools is on PYTHONPATH.

    Parameters
    ----------
    data_list : list of 1D arrays
        One array per distribution.
    ax : Axes, optional
    style : "violin" or "box"
    x_tick_labs : list of str, optional
    **kwargs
        Passed to plot_utils.plot_distributions (xlabel, ylabel, font_size, figsize,
        colors, yscale, ylim, save_path, show_fig, etc.).

    Returns
    -------
    fig, ax
    """
    if not _HAS_IAT:
        raise ImportError(
            "plot_distributions requires IntegratedAnalysisTools (spikedata). "
            "Run from the IntegratedAnalysisTools directory or set PYTHONPATH to it."
        )
    return _plot_distributions(
        data_list, ax=ax, style=style, x_tick_labs=x_tick_labs, **kwargs
    )


def plot_distributions_from_workspace(
    workspace_id: str,
    namespace: str,
    keys: List[str],
    save_path: str,
    x_tick_labs: Optional[List[str]] = None,
    style: str = "violin",
    **kwargs: Any,
) -> dict:
    """
    Load 1D arrays from workspace (namespace, key) for each key in keys, then plot
    distributions and save to save_path. Requires workspace and spikedata on PYTHONPATH.
    """
    if not _HAS_IAT:
        raise ImportError(
            "plot_distributions_from_workspace requires IntegratedAnalysisTools. "
            "Run from the IntegratedAnalysisTools directory or set PYTHONPATH to it."
        )
    from workspace.workspace import get_workspace_manager

    wm = get_workspace_manager()
    ws = wm.get_workspace(workspace_id)
    if ws is None:
        raise ValueError(f"Workspace not found: {workspace_id}")

    import numpy as np
    data_list = []
    for key in keys:
        obj = ws.get(namespace, key)
        if obj is None:
            raise ValueError(f"No item at ({namespace!r}, {key!r}).")
        data_list.append(np.asarray(obj).ravel())

    _plot_distributions(
        data_list,
        style=style,
        x_tick_labs=x_tick_labs if x_tick_labs is not None else keys,
        save_path=save_path,
        show_fig=False,
        **kwargs,
    )
    return {"save_path": save_path, "keys": keys, "n_per_group": [len(d) for d in data_list]}


if __name__ == "__main__":
    if not _HAS_IAT:
        print("Run with IntegratedAnalysisTools on PYTHONPATH to use plot_distributions.")
    else:
        print("plot_distributions and plot_distributions_from_workspace are available.")
