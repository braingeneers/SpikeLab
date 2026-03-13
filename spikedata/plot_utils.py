"""
Plotting utilities for spikedata (heatmaps, etc.).

Kept in a separate module so RateData and other classes can call it
without pulling in heavy or optional deps at import time.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

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
    origin: Optional[str] = None,
) -> tuple:
    """
    Plot a heatmap of a rate matrix (units x time).

    Original heatmap TODOs addressed (from old_plots/original_plot_heatmap.py):
    - MAKE THIS FUNCTION MORE GENERAL: see plot_heatmap() for generic 2D heatmap.
    - MAKE THIS A METHOD OF RateData: see RateData.plot_heatmap() in ratedata.py.
    - MAKE FONT SIZE SPECIFIC TO FIGURE/AX: font_size applied on ax only below, no rcParams.
    - LET USER CHOOSE NORMALIZE PER ROW / PER COLUMN / NONE: norm='row'|'column'|False.
    - LET USER PASS vmin AND vmax (NONE BY DEFAULT): vmin_plot/vmax_plot only set if not None.
    - LET USER CHOOSE AXIS LABELS: xlabel, ylabel params.
    - LET USER PASS X AND Y TICK LOCATIONS AND LABELS: xticks, xticklabels, yticks, yticklabels.
    - ALLOW ONE OR MULTIPLE X/Y LINES WITH COLOR, STYLE, WIDTH: vlines, hlines (list or dicts).
    - ALLOW USER TO CHOOSE NOT TO PLOT COLORBAR: show_colorbar param.
    - RETURN FIGURE/AX; ALLOW USER TO NOT PRINT OR SAVE: returns (fig, ax); save_path, show_fig.

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
        List of x-positions (numbers) or dicts with keys x, color, linestyle, linewidth.
        If None and temporal_offset is used, draws one line at temporal_offset.
    hlines : optional
        List of y-positions or dicts with keys y, color, linestyle, linewidth.
    figsize : tuple
        Figure size when ax is None.
    origin : str, optional
        Passed to imshow (e.g. "lower" so row 0 is at bottom; use when stacking
        with a raster plot so unit order matches). None uses matplotlib default.

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

    # Original TODO: LET USER CHOOSE NORMALIZE PER ROW / PER COLUMN / NONE (DEFAULT)
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

    # Original TODO: LET USER PASS vmin AND vmax (NONE BY DEFAULT); DO NOT SET IF NOT PASSED
    imshow_kw: Dict[str, Any] = {"cmap": "hot", "aspect": "auto"}
    if vmin_plot is not None:
        imshow_kw["vmin"] = vmin_plot
    if vmax_plot is not None:
        imshow_kw["vmax"] = vmax_plot
    if origin is not None:
        imshow_kw["origin"] = origin
    im = ax.imshow(data_mat, **imshow_kw)

    # Original TODO: LET USER CHOOSE AXIS LABELS
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=font_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=font_size)
    # Original TODO: MAKE FONT SIZE SPECIFIC TO FIGURE/AX (not global rcParams)
    ax.tick_params(axis="both", labelsize=font_size)

    # Original TODO: LET USER PASS X AND Y TICK LOCATIONS AND LABELS OPTIONALLY
    if yticks is not None:
        ax.set_yticks(yticks)
    else:
        ax.set_yticks([0, data_mat.shape[0] - 1] if data_mat.shape[0] > 1 else [0])
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    else:
        xt = ax.get_xticks()
        ax.set_xticklabels([str(int(t) - int(temporal_offset)) for t in xt])

    # Original TODO: ALLOW ONE OR MULTIPLE X/Y LINES; MORE CONTROL OVER COLOR, STYLE, WIDTH
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

    if hlines is not None:
        for h in hlines:
            if isinstance(h, (int, float)):
                ax.axhline(y=h, color="gray", linestyle="dotted", linewidth=1)
            else:
                opts = dict(h)
                y = opts.pop("y", 0)
                ax.axhline(y=y, **opts)

    # Original TODO: ALLOW USER TO CHOOSE NOT TO PLOT COLORBAR
    if show_colorbar:
        plt.colorbar(im, ax=ax, label=colorbar_label)

    # Original TODO: RETURN FIGURE/AX; ALLOW USER TO NOT PRINT OR SAVE BUT ONLY HAVE IT RETURNED
    if plot_fig:
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        elif show_fig:
            plt.show()

    return fig, ax


def plot_raster(
    spike_matrix: np.ndarray,
    bin_size_ms: float = 1.0,
    time_range: Optional[Tuple[float, float]] = None,
    pop_rate: Optional[np.ndarray] = None,
    fr_rates: Optional[np.ndarray] = None,
    sort_indices: Optional[np.ndarray] = None,
    event_times_ms: Optional[List[float]] = None,
    event_periods: Optional[List[Tuple[float, float]]] = None,
    figsize: Tuple[float, float] = (12, 6),
    height_ratios: Optional[List[float]] = None,
    font_size: int = 14,
    xlabel: str = "Time (ms)",
    time_axis_absolute: bool = True,
    fr_vmin: Optional[float] = None,
    fr_vmax: Optional[float] = None,
    save_path: Optional[str] = None,
    show_fig: bool = True,
) -> Tuple[Optional[Any], List[Any]]:
    """
    Plot spike raster with optional population rate and per-unit rate heatmap.

    Original raster TODOs addressed (from old_plots/original_plot_raster.py):
    - MAKE THIS A SPIKEDATA METHOD: see SpikeData.plot_raster() in spikedata.py (script here in plot_utils).
    - MAKE FONT SIZE SPECIFIC TO FIGURE: font_size on ax only below, no rcParams.
    - ALLOW fr_rates COMPUTED FROM SPIKEDATA: SpikeData.plot_raster() supports fr_rate_bin_ms.
    - ENABLE PRECOMPUTED POP_RATE; USER CHOOSE PARAMS: pop_rate optional; pop_rate_square_width, pop_rate_gauss_sigma.
    - LET USER CHOOSE FIGURE SIZE AND SUBPLOT RATIOS: figsize, height_ratios.
    - SEPARATE EVENT PERIODS FROM EVENT TIMES: event_periods = list of (start_ms, end_ms), clipped to time_range.
    - USE HEATMAP FUNCTION FOR fr_rates: plot_av_rate() used for rate subplot below.
    - LET USER PASS vmin AND vmax (NONE BY DEFAULT) for rate heatmap: fr_vmin, fr_vmax.
    - SET X LIMITS FOR ALL AXES FROM time_range: ax.set_xlim(0, n_bins) for all axes.
    - X-TICK LABELS: time_axis_absolute = True -> time_range[0]..time_range[1]; False -> 0..duration.
    - RETURN FIGURE; ALLOW USER TO NOT PRINT OR SAVE: returns (fig, axes); save_path, show_fig.

    Parameters
    ----------
    spike_matrix : np.ndarray, shape (n_units, n_time_bins)
        Binary or count raster (units x time bins).
    bin_size_ms : float
        Bin width in ms (used for time axis labels and pop_rate Hz conversion).
    time_range : (float, float) or None
        (start_ms, end_ms) to display; None = full range.
    pop_rate : np.ndarray 1D or None
        Population rate in spikes per bin (same length as time bins). If None, no pop-rate subplot.
    fr_rates : np.ndarray, shape (n_units, n_time_bins) or None
        Per-unit rate matrix for bottom heatmap subplot. If None, no heatmap subplot.
    sort_indices : np.ndarray or None
        Reorder units (rows) for raster and fr_rates.
    event_times_ms : list of float or None
        Event times in ms; marked with scatter on population rate subplot.
    event_periods : list of (start_ms, end_ms) or None
        Periods to shade with axvspan. Clipped to time_range when set.
    figsize : (width, height) in inches
    height_ratios : list of float or None
        Subplot height ratios, e.g. [2, 1] or [2, 1, 1]. Default [2, 1] or [2, 1, 1] if fr_rates.
    font_size : int
        Font size for labels/ticks on axes only (not global rcParams).
    xlabel : str
        X-axis label.
    time_axis_absolute : bool
        If True and time_range set, x-tick labels are start_ms..end_ms; else 0..duration_ms.
    fr_vmin, fr_vmax : float or None
        Color scale for fr_rates heatmap; None = auto.
    save_path : str or None
        If set, save figure and close.
    show_fig : bool
        If True and save_path is None, call plt.show(); else only return figure.

    Returns
    -------
    fig : Figure or None
        The figure (if created).
    axes : list of Axes
        [raster_ax, pop_rate_ax] or [raster_ax, pop_rate_ax, fr_ax].
    """
    spk = np.asarray(spike_matrix)
    if spk.ndim != 2:
        raise ValueError("spike_matrix must be 2D (units x time_bins)")
    n_units, n_bins_full = spk.shape
    start_bin = 0
    end_bin = n_bins_full
    start_ms = 0.0
    end_ms = n_bins_full * bin_size_ms
    if time_range is not None:
        t0, t1 = time_range
        # Bin index = floor(t / bin_size_ms) (matches SpikeData.sparse_raster).
        # Include the bin that contains t1, so exclusive end = floor(t1/bin_size_ms) + 1.
        start_bin = max(0, int(t0 / bin_size_ms))
        end_bin = min(n_bins_full, int(np.floor(t1 / bin_size_ms)) + 1)
        start_ms = start_bin * bin_size_ms
        end_ms = end_bin * bin_size_ms
    spk = spk[:, start_bin:end_bin]
    n_bins = spk.shape[1]
    duration_ms = end_ms - start_ms
    if n_bins == 0:
        raise ValueError(
            "time_range yields no bins (empty slice). Check time_range and bin_size_ms."
        )

    if sort_indices is not None:
        spk = spk[sort_indices, :]
        if fr_rates is not None:
            fr_rates = np.asarray(fr_rates)[sort_indices, :]
    else:
        if fr_rates is not None:
            fr_rates = np.asarray(fr_rates).copy()

    if fr_rates is not None:
        n_fr = fr_rates.shape[1]
        if n_fr == n_bins:
            pass  # already aligned
        elif n_fr == n_bins_full:
            fr_rates = fr_rates[:, start_bin:end_bin]
        else:
            # Resample to raster grid (e.g. RateData from resampled ISI)
            from scipy.ndimage import zoom

            zoom_factors = (1, n_bins / n_fr)
            fr_rates = zoom(fr_rates, zoom_factors, order=1)

    if pop_rate is not None:
        pop_rate = np.asarray(pop_rate).ravel()[start_bin:end_bin]
        # Convert spikes per bin to Hz for display
        pop_rate_Hz = pop_rate / (bin_size_ms / 1000.0)
    else:
        pop_rate_Hz = None

    # Event times: filter to window, convert to bin index in [0, n_bins]
    if event_times_ms is not None:
        event_times_ms = np.unique(np.asarray(event_times_ms))
        event_bins = (event_times_ms - start_ms) / bin_size_ms
        in_window = (event_bins >= 0) & (event_bins < n_bins)
        event_bins = event_bins[in_window]
    else:
        event_bins = np.array([])

    # Original TODO: SEPARATE EVENT PERIODS FROM EVENT TIMES; LIST OF (START, END); CLIP TO time_range
    if event_periods is not None:
        period_bins = []
        for s, e in event_periods:
            s_clip = max(s, start_ms)
            e_clip = min(e, end_ms)
            if s_clip < e_clip:
                b0 = (s_clip - start_ms) / bin_size_ms
                b1 = (e_clip - start_ms) / bin_size_ms
                period_bins.append((b0, b1))
    else:
        period_bins = []

    has_pop = pop_rate_Hz is not None
    has_fr = fr_rates is not None
    n_subplots = 1 + (1 if has_pop else 0) + (1 if has_fr else 0)
    # Original TODO: LET USER CHOOSE FIGURE SIZE AND SUBPLOT HEIGHT RATIOS
    if height_ratios is None:
        if n_subplots == 1:
            height_ratios = [1]
        elif n_subplots == 2:
            height_ratios = [2, 1]
        else:
            height_ratios = [2, 1, 1]
    fig, axes = plt.subplots(
        n_subplots,
        1,
        sharex=True,
        figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = [axes] if n_subplots == 1 else list(axes)
    ax1 = axes[0]

    # Raster: eventplot uses bin indices on x (one marker per bin that has any spike)
    spike_times_list = [
        np.where(spk[i, :] > 0)[0].astype(float) for i in range(spk.shape[0])
    ]
    ax1.eventplot(spike_times_list, colors="black")
    ax1.set_ylim(0, spk.shape[0])
    ax1.set_ylabel("Unit", fontsize=font_size)
    ax1.set_xlabel(xlabel, fontsize=font_size)
    # Original TODO: MAKE FONT SIZE SPECIFIC TO FIGURE (ax only)
    ax1.tick_params(axis="both", labelsize=font_size)
    # Original TODO: SET X LIMITS FOR ALL AXES BASED ON time_range
    ax1.set_xlim(0, n_bins)

    idx = 1
    if has_pop:
        ax2 = axes[idx]
        idx += 1
        ax2.plot(pop_rate_Hz, color="blue")
        ax2.set_ylabel("Population rate (Hz)", fontsize=font_size)
        ax2.set_xlabel(xlabel, fontsize=font_size)
        ax2.tick_params(axis="both", labelsize=font_size)
        ax2.set_xlim(0, n_bins)
        for b in event_bins:
            i = int(round(b))
            if 0 <= i < len(pop_rate_Hz):
                ax2.scatter(b, pop_rate_Hz[i], c="k", zorder=9)
        for b0, b1 in period_bins:
            ax2.axvspan(b0, b1, color="b", alpha=0.2)

    if has_fr:
        ax3 = axes[idx]
        # Original TODO: USE THE FUNCTION FROM plot_heatmap (plot_av_rate) FOR fr_rates
        # Original TODO: LET USER PASS vmin AND vmax (NONE BY DEFAULT) for rate heatmap
        plot_av_rate(
            fr_rates,
            ax=ax3,
            vmin=fr_vmin,
            vmax=fr_vmax,
            font_size=font_size,
            show_colorbar=True,
            colorbar_label="Av. Rate (Hz)",
            xlabel=xlabel,
            ylabel="Unit",
            show_fig=False,
            origin="lower",
        )
        ax3.set_xlim(0, n_bins)

    # Original TODO: X-TICK LABELS: time_axis_absolute -> time_range[0]..time_range[1]; else 0..duration
    for ax in axes:
        ax.set_xlim(0, n_bins)
        xt = ax.get_xticks()
        xt = xt[(xt >= 0) & (xt <= n_bins)]
        if time_range is not None and time_axis_absolute:
            labels = [str(int(start_ms + t * bin_size_ms)) for t in xt]
        else:
            labels = [str(int(t * bin_size_ms)) for t in xt]
        ax.set_xticks(xt)
        ax.set_xticklabels(labels)

    plt.tight_layout()
    # Original TODO: RETURN FIGURE; ALLOW USER TO NOT PRINT OR SAVE BUT ONLY HAVE IT RETURNED
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    elif show_fig:
        plt.show()
    return fig, axes


def plot_heatmap(
    data_mat: np.ndarray,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    colorbar_label: str = "",
    **kwargs: Any,
) -> tuple:
    """
    Generic 2D heatmap (no rate-specific defaults). Pass axis labels and options as needed.
    Original TODO (heatmap): MAKE THIS FUNCTION MORE GENERAL FOR ANY TYPE OF HEATMAP.
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


def plot_distributions(
    data_list: List[Union[np.ndarray, List[float]]],
    ax: Optional[Any] = None,
    style: str = "violin",
    x_tick_labs: Optional[List[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = "Av. rate (Hz)",
    font_size: int = 14,
    figsize: Tuple[float, float] = (5, 4),
    colors: Optional[List[Union[str, Tuple[float, ...]]]] = None,
    edge_color: str = "black",
    yscale: str = "log",
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    yticks: Optional[List[float]] = None,
    yticklabels: Optional[List[str]] = None,
    show_means: bool = False,
    show_medians: bool = True,
    alpha: float = 0.8,
    save_path: Optional[str] = None,
    show_fig: bool = True,
) -> Tuple[Optional[Any], Any]:
    """
    Plot distributions as violin or box plots (one per group in data_list).

    Original distribution TODOs addressed (from old_plots/original_plot_distributions.py):
    - LET USER CHOOSE BETWEEN BOXPLOTS OR VIOLIN PLOTS: style='violin'|'box'.
    - LET USER CUSTOMIZE AXIS LABELS, FIG SIZE, VIOLIN/BOXPLOT COLORING, ETC.: xlabel, ylabel, figsize, colors, edge_color, yscale, ylim, yticks, yticklabels, show_means, show_medians, alpha.
    - MAKE FONT SIZE SPECIFIC TO FIGURE/AX: font_size on ax only below, no rcParams.
    - RETURN FIGURE OR AX; ALLOW USER TO NOT PRINT OR SAVE: returns (fig, ax); save_path, show_fig.

    Parameters
    ----------
    data_list : list of 1D arrays
        One array per distribution (e.g. per-unit values per condition).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    style : "violin" or "box"
        Plot type.
    x_tick_labs : list of str, optional
        Tick labels for each distribution (same length as data_list).
    xlabel, ylabel : str, optional
        Axis labels. Default ylabel "Av. rate (Hz)". None = leave unset.
    font_size : int
        Font size for labels/ticks on this ax only (not global rcParams).
    figsize : (width, height) in inches
        Used when ax is None.
    colors : list of color, optional
        One color per distribution (matplotlib color). If None, default cycle (tab:blue, tab:orange, ...).
    edge_color : str
        Edge color for violin/box elements.
    yscale : "log" or "linear"
    ylim : (ymin, ymax) or None; None in tuple means auto (e.g. (None, 100)).
    yticks, yticklabels : optional
        Y-axis tick positions and labels.
    show_means, show_medians : bool
        For violin: show means/medians. For box: median always shown.
    alpha : float
        Fill alpha for violin/box.
    save_path : str, optional
        If set, save figure and close.
    show_fig : bool
        If True and save_path is None, call plt.show(); else only return.

    Returns
    -------
    fig : Figure or None
        The figure if created, else None.
    ax : Axes
        The axes used.
    """
    data_list = [np.asarray(d).ravel() for d in data_list]
    n = len(data_list)
    positions = list(range(1, n + 1))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        plot_fig = True
    else:
        fig = None
        plot_fig = False

    if colors is None:
        default_colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
        ]
        colors = [default_colors[i % len(default_colors)] for i in range(n)]
    elif len(colors) < n:
        colors = list(colors) + [colors[-1]] * (n - len(colors))

    # Original TODO: LET USER CHOOSE BETWEEN BOXPLOTS OR VIOLIN PLOTS
    if style == "violin":
        vplot = ax.violinplot(
            data_list,
            positions=positions,
            showmeans=show_means,
            showmedians=show_medians,
        )
        for i, body in enumerate(vplot["bodies"]):
            body.set_facecolor(colors[i] if i < len(colors) else "gray")
            body.set_edgecolor(edge_color)
            body.set_alpha(alpha)
        for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
            if key in vplot and vplot[key] is not None:
                vplot[key].set_color(edge_color)
    elif style == "box":
        bplot = ax.boxplot(
            data_list,
            positions=positions,
            patch_artist=True,
            showmeans=show_means,
            medianprops=dict(color=edge_color),
        )
        for i, patch in enumerate(bplot["boxes"]):
            patch.set_facecolor(colors[i] if i < len(colors) else "gray")
            patch.set_alpha(alpha)
            patch.set_edgecolor(edge_color)
        for key in ["whiskers", "caps", "fliers"]:
            if key in bplot and bplot[key] is not None:
                for el in bplot[key]:
                    el.set_color(edge_color)
    else:
        raise ValueError(f"style must be 'violin' or 'box', got {style!r}")

    if x_tick_labs is not None:
        ax.set_xticks(positions)
        ax.set_xticklabels(x_tick_labs)
    # Original TODO: LET USER CUSTOMIZE AXIS LABELS (and figsize, colors above)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=font_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=font_size)
    # Original TODO: MAKE FONT SIZE SPECIFIC TO FIGURE/AX (not global rcParams)
    ax.tick_params(axis="both", labelsize=font_size)

    ax.set_yscale(yscale)
    if ylim is not None:
        ymin, ymax = ylim
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    elif yscale == "log" and yticks is None:
        ax.set_yticks([0.01, 0.1, 1, 10, 100])
        ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])

    plt.tight_layout()
    # Original TODO: RETURN FIGURE OR AX; ALLOW USER TO NOT PRINT OR SAVE BUT ONLY HAVE IT RETURNED
    if plot_fig:
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        elif show_fig:
            plt.show()
    return fig, ax
