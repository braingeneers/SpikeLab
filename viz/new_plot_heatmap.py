"""
Heatmap plotting for rate data (and generic 2D heatmaps).

Supports:
- Direct 2D array: plot_av_rate(data_mat, ...) or plot_heatmap(data_mat, ...) for generic use.
- Workspace (MCP flow): plot_av_rate_from_workspace(workspace_id, namespace, key, ...)
  Use after loading data and computing rates via IntegratedAnalysisTools MCP.
- Run from the IntegratedAnalysisTools directory: python viz/new_plot_heatmap.py (or set PYTHONPATH to IAT).

Original TODOs from old_plots/original_plot_heatmap.py (addressed here and in IAT):
  (1) MAKE THIS FUNCTION MORE GENERAL FOR ANY HEATMAP -> plot_heatmap() below.
  (2) MAKE THIS A METHOD OF RateData -> spikedata/ratedata.py RateData.plot_heatmap().
  (3) MAKE FONT SIZE SPECIFIC TO FIGURE/AX -> font_size on ax only in plot_av_rate (no rcParams).
  (4) NORMALIZE PER ROW / PER COLUMN / NONE -> norm='row'|'column'|False.
  (5) LET USER PASS vmin AND vmax (NONE BY DEFAULT) -> vmin, vmax; only set in imshow if not None.
  (6) LET USER CHOOSE AXIS LABELS -> xlabel, ylabel.
  (7) LET USER PASS X AND Y TICK LOCATIONS AND LABELS -> xticks, xticklabels, yticks, yticklabels.
  (8) ONE OR MULTIPLE X/Y LINES; COLOR, STYLE, WIDTH -> vlines, hlines (list or dicts).
  (9) ALLOW USER TO CHOOSE NOT TO PLOT COLORBAR -> show_colorbar.
  (10) RETURN FIGURE/AX; ALLOW NOT PRINT OR SAVE -> returns (fig, ax); save_path, show_fig.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt


def _norm_per_row(data: np.ndarray) -> np.ndarray:
    out = np.zeros_like(data, dtype=float)
    for i in range(data.shape[0]):
        rmin, rmax = data[i].min(), data[i].max()
        if rmax > rmin:
            out[i] = (data[i] - rmin) / (rmax - rmin)
        else:
            out[i] = data[i]
    return out


def _norm_per_column(data: np.ndarray) -> np.ndarray:
    out = np.zeros_like(data, dtype=float)
    for j in range(data.shape[1]):
        cmin, cmax = data[:, j].min(), data[:, j].max()
        if cmax > cmin:
            out[:, j] = (data[:, j] - cmin) / (cmax - cmin)
        else:
            out[:, j] = data[:, j]
    return out


def plot_av_rate(
    data_mat: np.ndarray,
    ax: Optional[Any] = None,
    norm: Union[bool, str] = False,
    colorbar_label: str = "Av. Rate (Hz)",
    temporal_offset: float = 0,
    vmax: Optional[float] = 40,
    vmin: Optional[float] = 0,
    font_size: int = 14,
    save_path: Optional[str] = None,
    show_colorbar: bool = True,
    show_fig: bool = True,
    xlabel: Optional[str] = "Relative time (ms)",
    ylabel: Optional[str] = "Unit",
    xticks: Optional[List[float]] = None,
    xticklabels: Optional[List[str]] = None,
    yticks: Optional[List[float]] = None,
    yticklabels: Optional[List[str]] = None,
    vlines: Optional[List[Union[float, Dict[str, Any]]]] = None,
    hlines: Optional[List[Union[float, Dict[str, Any]]]] = None,
    figsize: tuple = (8, 6),
) -> tuple:
    """
    Plot a heatmap of a rate matrix (units x time).

    Parameters
    ----------
    data_mat : np.ndarray, shape (n_units, n_time_bins)
        Rate matrix to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    norm : bool or str, default False
        Normalization: False (none), True or "row" (per row), "column" (per column).
    colorbar_label : str
        Label for the colorbar.
    temporal_offset : float
        Column index for reference line and x-tick label offset (when xticklabels not provided).
    vmax, vmin : float or None
        Color scale. If None, matplotlib auto-scales.
    font_size : int
        Font size for labels and ticks (set on this ax only, not globally).
    save_path : str, optional
        If set, save figure to this path and close.
    show_colorbar : bool, default True
        Whether to draw the colorbar.
    show_fig : bool, default True
        If True and ax was None, call plt.show() when save_path is None.
    xlabel, ylabel : str or None
        Axis labels. Defaults "Relative time (ms)" and "Unit". None = leave unset.
    xticks, xticklabels : optional
        If provided, set x tick positions and labels (overrides default temporal_offset labels).
    yticks, yticklabels : optional
        If provided, set y tick positions and labels.
    vlines : optional
        List of x-positions (numbers) or dicts with keys x, color, linestyle, linewidth
        (e.g. {"x": 0, "color": "green", "linestyle": "dotted", "linewidth": 2}).
        If None and temporal_offset is used, draws one line at temporal_offset.
    hlines : optional
        List of y-positions or dicts with keys y, color, linestyle, linewidth.
    figsize : tuple
        Figure size when ax is None.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure if one was created, else None.
    ax : matplotlib.axes.Axes
        The axes used.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        plot_fig = True
    else:
        fig = None
        plot_fig = False

    # Normalize
    if norm is True:
        norm = "row"
    if norm == "row":
        data_mat = _norm_per_row(data_mat)
        colorbar_label = "Norm. " + colorbar_label
        vmax_plot, vmin_plot = 1.0, 0.0
    elif norm == "column":
        data_mat = _norm_per_column(data_mat)
        colorbar_label = "Norm. " + colorbar_label
        vmax_plot, vmin_plot = 1.0, 0.0
    else:
        vmin_plot = vmin
        vmax_plot = vmax

    # imshow: only pass vmin/vmax if set (so None = auto)
    imshow_kw: Dict[str, Any] = {"cmap": "hot", "aspect": "auto"}
    if vmin_plot is not None:
        imshow_kw["vmin"] = vmin_plot
    if vmax_plot is not None:
        imshow_kw["vmax"] = vmax_plot
    im = ax.imshow(data_mat, **imshow_kw)

    # Axis labels (ax-specific, not global)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=font_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=font_size)
    ax.tick_params(axis="both", labelsize=font_size)

    # Y ticks
    if yticks is not None:
        ax.set_yticks(yticks)
    else:
        ax.set_yticks([0, data_mat.shape[0] - 1] if data_mat.shape[0] > 1 else [0])
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    # X ticks
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    else:
        # Default: label by (tick - temporal_offset)
        xt = ax.get_xticks()
        ax.set_xticklabels([str(int(t) - int(temporal_offset)) for t in xt])

    # Vertical lines
    if vlines is not None:
        for v in vlines:
            if isinstance(v, (int, float)):
                ax.axvline(x=v, color="green", linestyle="dotted", linewidth=2)
            else:
                opts = dict(v)
                x = opts.pop("x", 0)
                ax.axvline(x=x, **opts)
    elif temporal_offset is not None:
        ax.axvline(x=temporal_offset, color="green", linestyle="dotted", linewidth=2)

    # Horizontal lines
    if hlines is not None:
        for h in hlines:
            if isinstance(h, (int, float)):
                ax.axhline(y=h, color="gray", linestyle="dotted", linewidth=1)
            else:
                opts = dict(h)
                y = opts.pop("y", 0)
                ax.axhline(y=y, **opts)

    if show_colorbar:
        plt.colorbar(im, ax=ax, label=colorbar_label)

    if plot_fig:
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        elif show_fig:
            plt.show()

    return fig, ax


def plot_heatmap(
    data_mat: np.ndarray,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    colorbar_label: str = "",
    **kwargs: Any,
) -> tuple:
    """
    Generic 2D heatmap (no rate-specific defaults). Pass axis labels and options as needed.
    """
    return plot_av_rate(
        data_mat,
        xlabel=xlabel,
        ylabel=ylabel,
        colorbar_label=colorbar_label or "",
        temporal_offset=0,
        vlines=[],
        **kwargs,
    )


def plot_av_rate_from_workspace(
    workspace_id,
    namespace,
    key="ratedata",
    temporal_offset=None,
    **kwargs,
):
    """
    Load RateData from an IntegratedAnalysisTools workspace and plot its rate matrix.

    Use this when data has been loaded and rates computed via MCP (e.g. load_from_hdf5
    then compute_rates or compute_binned_meanrate). Requires the workspace package
    to be importable (e.g. run with IntegratedAnalysisTools on PYTHONPATH or from
    that repo).

    Parameters
    ----------
    workspace_id : str
        Workspace ID (from create_workspace or list_workspaces).
    namespace : str
        Recording namespace (e.g. from load_from_hdf5).
    key : str, default "ratedata"
        Workspace key where RateData is stored (e.g. output key from compute_rates).
    temporal_offset : float or int, optional
        Column index for the reference line and for x-axis label offset (see plot_av_rate).
        Default 0 (first column = 0 ms in labels).
    **kwargs
        Passed through to plot_av_rate (e.g. norm, vmax, save_path, font_size).

    Returns
    -------
    fig, ax
        As returned by plot_av_rate.
    """
    try:
        from workspace.workspace import get_workspace_manager
    except ImportError as e:
        raise ImportError(
            "plot_av_rate_from_workspace requires the IntegratedAnalysisTools "
            "workspace package. Run from that repo or set PYTHONPATH to include it."
        ) from e

    wm = get_workspace_manager()
    ws = wm.get_workspace(workspace_id)
    if ws is None:
        raise ValueError(f"Workspace not found: {workspace_id}")

    rd = ws.get(namespace, key)
    if rd is None:
        raise ValueError(
            f"No item at ({namespace!r}, {key!r}). Load and compute rates first."
        )

    try:
        data_mat = rd.inst_Frate_data
    except AttributeError:
        raise TypeError(
            f"Item at ({namespace!r}, {key!r}) is not RateData (no inst_Frate_data). "
            "Store RateData from compute_rates or compute_binned_meanrate."
        )

    if temporal_offset is None:
        temporal_offset = 0

    return plot_av_rate(
        data_mat,
        temporal_offset=temporal_offset,
        **kwargs,
    )
