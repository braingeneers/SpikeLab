"""
Plotting utilities for SpikeLab.

Provides ``plot_recording`` for assembling multi-panel figures from SpikeData
objects, ``plot_heatmap`` for standalone 2-D heatmaps, ``plot_distribution``
for comparing per-unit metrics across conditions, ``plot_pvalue_matrix`` for
significance heatmaps (standalone or as inset), ``plot_scatter`` for pairwise
comparisons with optional regression, ``plot_lines`` for multi-trace line
plots, ``plot_burst_sensitivity`` for threshold sensitivity curves,
``plot_scatter_with_marginals`` for scatter plots with marginal histograms,
``plot_aligned_slice_single_unit`` for event-aligned single-unit raster plots,
and ``plot_spatial_network`` for MEA spatial network visualisation.

Requires ``matplotlib`` (optional dependency).
"""

import numpy as np


def _import_matplotlib():
    """Import matplotlib and return (plt, mticker). Raises ImportError with message."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        return plt, mticker
    except ImportError as e:
        raise ImportError(
            "plot_utils requires 'matplotlib'. " "Install with: pip install matplotlib"
        ) from e


def _add_colorbar(im, ax, label="", font_size=14):
    """Add a colorbar on a dedicated axes so the parent axes width is unchanged.

    Uses ``make_axes_locatable`` to append a thin axes to the right of *ax*.
    This avoids the width-stealing behaviour of ``fig.colorbar(im, ax=ax)``.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=font_size)
    cb.ax.tick_params(labelsize=font_size)
    return cb


def _apply_font_size(ax, font_size):
    """Apply font_size to axis labels and tick labels."""
    ax.xaxis.label.set_fontsize(font_size)
    ax.yaxis.label.set_fontsize(font_size)
    ax.tick_params(axis="both", labelsize=font_size)


# ---------------------------------------------------------------------------
# plot_distribution
# ---------------------------------------------------------------------------


def plot_distribution(
    ax,
    metric_data,
    labels=None,
    colors=None,
    ylabel="",
    xlabel="",
    style="violin",
    show_median=True,
    show_quartiles=True,
    show_data=False,
    data_alpha=0.3,
    data_size=4,
    log_scale=False,
    font_size=None,
):
    """
    Plot distributions of a per-unit metric across multiple groups/conditions.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes (caller creates).
        metric_data (dict[str, np.ndarray] or list[np.ndarray]): Condition-labelled
            or ordered collection of per-unit value arrays. NaN values are
            stripped automatically before plotting.
        labels (list[str] or None): Ordered condition labels. If None, uses
            dict keys (for dict input) or integer indices (for list input).
        colors (list[str] or None): Per-condition colours. If None, uses
            the default matplotlib colour cycle.
        ylabel (str): Y-axis label.
        xlabel (str): X-axis label.
        style (str): ``"violin"`` (default) or ``"boxplot"``.
        show_median (bool): Overlay a median dot on each distribution.
        show_quartiles (bool): Overlay IQR lines (25th–75th percentile) on
            each distribution.
        show_data (bool): Overlay individual data points on each distribution,
            jittered horizontally to reduce overlap.
        data_alpha (float): Alpha transparency for overlaid data points.
        data_size (float): Marker size for overlaid data points.
        log_scale (bool): Use a log scale on the y-axis.
        font_size (int or None): Font size for labels and ticks. If None,
            uses current rcParams.

    Returns:
        parts (dict): The violin or boxplot artist dict returned by
            matplotlib (``violinplot`` or ``boxplot``).

    Notes:
        - In violin mode, groups with fewer than 2 data points cannot produce
          a kernel density estimate. These groups are rendered as individual
          scatter points instead and excluded from the violin plot.
    """
    _import_matplotlib()

    # --- Normalise input to list-of-arrays + labels -----------------------
    if isinstance(metric_data, dict):
        keys = list(metric_data.keys())
        data_arrays = [np.asarray(metric_data[k]) for k in keys]
        if labels is None:
            labels = keys
    else:
        data_arrays = [np.asarray(a) for a in metric_data]
        if labels is None:
            labels = [str(i) for i in range(len(data_arrays))]

    # Strip NaNs from each array
    clean_data = []
    for arr in data_arrays:
        flat = arr.ravel()
        clean_data.append(flat[~np.isnan(flat)])

    n = len(clean_data)
    positions = list(range(n))

    # --- Resolve colours --------------------------------------------------
    if colors is None:
        import matplotlib.pyplot as _plt

        cycle_colors = _plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = [cycle_colors[i % len(cycle_colors)] for i in range(n)]

    # --- Draw distribution ------------------------------------------------
    if style == "boxplot":
        parts = ax.boxplot(
            clean_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=True,
        )
        for i, box in enumerate(parts["boxes"]):
            box.set_facecolor(colors[i])
            box.set_alpha(0.8)
    elif style == "violin":
        # Separate groups with enough points for KDE from sparse groups
        violin_positions = []
        violin_data = []
        sparse_groups = []  # (position, data, color) for groups with < 2 points
        for i, d in enumerate(clean_data):
            if len(d) >= 2:
                violin_positions.append(positions[i])
                violin_data.append(d)
            else:
                sparse_groups.append((positions[i], d, colors[i]))

        parts = {"bodies": []}
        if violin_data:
            parts = ax.violinplot(
                violin_data,
                positions=violin_positions,
                showmeans=False,
                showextrema=False,
            )
            # Map violin bodies back to their colour by position index
            pos_to_color = {p: colors[p] for p in violin_positions}
            for body, pos in zip(parts["bodies"], violin_positions):
                body.set_facecolor(pos_to_color[pos])
                body.set_edgecolor("black")
                body.set_linewidth(0.5)
                body.set_alpha(0.8)

        # Render sparse groups as scatter points
        for pos, d, color in sparse_groups:
            if len(d) > 0:
                ax.scatter(
                    np.full(len(d), pos),
                    d,
                    color=color,
                    s=data_size * 4,
                    zorder=3,
                    edgecolors="black",
                    linewidths=0.5,
                )
    else:
        raise ValueError(f"Unknown style '{style}'. Use 'violin' or 'boxplot'.")

    # --- Median dot + IQR lines -------------------------------------------
    if show_median or show_quartiles:
        for i, d in enumerate(clean_data):
            if len(d) == 0:
                continue
            q25, median, q75 = np.nanpercentile(d, [25, 50, 75])
            if show_median:
                ax.scatter(
                    i,
                    median,
                    color="white",
                    s=15,
                    zorder=4,
                    edgecolors="black",
                    linewidths=0.5,
                )
            if show_quartiles:
                ax.vlines(i, q25, q75, color="black", linewidth=1.5, zorder=3)

    # --- Overlay individual data points -----------------------------------
    if show_data:
        rng = np.random.default_rng(0)
        for i, d in enumerate(clean_data):
            if len(d) == 0:
                continue
            jitter = rng.uniform(-0.15, 0.15, size=len(d))
            ax.scatter(
                positions[i] + jitter,
                d,
                color=colors[i],
                s=data_size,
                alpha=data_alpha,
                zorder=2,
                edgecolors="none",
            )

    # --- Axes formatting --------------------------------------------------
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if log_scale:
        ax.set_yscale("log")

    if font_size is not None:
        _apply_font_size(ax, font_size)

    return parts


# ---------------------------------------------------------------------------
# plot_pvalue_matrix
# ---------------------------------------------------------------------------


def plot_pvalue_matrix(
    pval_matrix,
    sig_matrix=None,
    labels=None,
    ax=None,
    parent_ax=None,
    inset_loc="upper left",
    inset_size="30%",
    inset_offset=0.08,
    cmap="viridis",
    sig_marker_color="red",
    sig_marker_size=2.5,
    show_colorbar=True,
    font_size=None,
):
    """
    Display a pairwise p-value matrix as a ``-log10(p)`` heatmap.

    Supports two rendering modes (mutually exclusive):

    - **Standalone**: pass ``ax`` to plot directly on existing axes.
    - **Inset**: pass ``parent_ax`` to create a small inset axes on a parent
      plot (e.g. a violin or distribution plot).

    Exactly one of ``ax`` or ``parent_ax`` must be provided.

    Parameters:
        pval_matrix (np.ndarray): (K, K) p-value matrix. Diagonal entries
            should be NaN (they are rendered in black).
        sig_matrix (np.ndarray or None): (K, K) boolean — True where the
            comparison is significant. If None, computed as
            ``pval_matrix < 0.05``.
        labels (list[str] or None): Tick labels for each group. If None,
            integer indices are used.
        ax (matplotlib.axes.Axes or None): Target axes for standalone mode.
        parent_ax (matplotlib.axes.Axes or None): Parent axes on which to
            create an inset.
        inset_loc (str): Location string for ``inset_axes`` (e.g.
            ``"upper left"``, ``"lower left"``). Only used in inset mode.
        inset_size (str): Width and height of the inset as a percentage of
            the parent (e.g. ``"30%"``). Only used in inset mode.
        inset_offset (float): Horizontal offset of the inset bounding box
            from the parent axes edge. Only used in inset mode.
        cmap (str): Matplotlib colormap name.
        sig_marker_color (str): Colour for significance markers.
        sig_marker_size (float): Marker size for significance dots.
        show_colorbar (bool): Show a ``-log10(P)`` colorbar.
        font_size (int or None): Font size for labels and ticks. If None,
            uses current rcParams.

    Returns:
        target_ax (matplotlib.axes.Axes): The axes the matrix was drawn on
            (either ``ax`` or the newly created inset axes).
    """
    plt, _ = _import_matplotlib()
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    pval_matrix = np.asarray(pval_matrix, dtype=float)
    K = pval_matrix.shape[0]

    if sig_matrix is None:
        sig_matrix = pval_matrix < 0.05
    sig_matrix = np.asarray(sig_matrix, dtype=bool)

    if labels is None:
        labels = [str(i) for i in range(K)]

    # --- Resolve target axes ----------------------------------------------
    if ax is not None and parent_ax is not None:
        raise ValueError("Provide either 'ax' or 'parent_ax', not both.")
    if ax is None and parent_ax is None:
        raise ValueError("Provide either 'ax' (standalone) or 'parent_ax' (inset).")

    if parent_ax is not None:
        target_ax = inset_axes(
            parent_ax,
            width=inset_size,
            height=inset_size,
            loc=inset_loc,
            bbox_to_anchor=(inset_offset, 0, 1, 1),
            bbox_transform=parent_ax.transAxes,
            borderpad=1.0,
        )
    else:
        target_ax = ax

    # --- Compute -log10(p) ------------------------------------------------
    neg_log_p = -np.log10(pval_matrix)

    # Cap infinite values for display
    finite_vals = neg_log_p[np.isfinite(neg_log_p) & ~np.isnan(neg_log_p)]
    vmax = np.max(finite_vals) if len(finite_vals) > 0 else 1
    neg_log_p = np.where(np.isfinite(neg_log_p), neg_log_p, vmax)

    # Diagonal → NaN (rendered as black via set_bad)
    np.fill_diagonal(neg_log_p, np.nan)

    import matplotlib as mpl

    try:
        colormap = mpl.colormaps[cmap].copy()
    except (AttributeError, TypeError, KeyError, ValueError):
        # Matplotlib < 3.7 / older registry API
        colormap = mpl.cm.get_cmap(cmap).copy()
    colormap.set_bad(color="black")

    im = target_ax.imshow(
        neg_log_p,
        cmap=colormap,
        aspect="equal",
        interpolation="none",
        vmin=0,
        vmax=vmax,
    )

    # --- Significance markers ---------------------------------------------
    for i in range(K):
        for j in range(K):
            if i != j and sig_matrix[i, j]:
                target_ax.plot(
                    j,
                    i,
                    "o",
                    color=sig_marker_color,
                    markersize=sig_marker_size,
                    markeredgewidth=0,
                )

    # --- Tick labels ------------------------------------------------------
    fs = font_size
    if fs is None:
        fs = plt.rcParams.get("xtick.labelsize", 10)
        if not isinstance(fs, (int, float)):
            fs = 10
    tick_fs = fs - 1 if parent_ax is not None else fs
    target_ax.set_xticks(range(K))
    target_ax.set_xticklabels(labels, fontsize=tick_fs)
    target_ax.set_yticks(range(K))
    target_ax.set_yticklabels(labels, fontsize=tick_fs)

    for spine in target_ax.spines.values():
        spine.set_linewidth(0.5)

    # --- Colorbar ---------------------------------------------------------
    if show_colorbar:
        cbar_ax = inset_axes(
            target_ax,
            width="8%",
            height="100%",
            loc="center right",
            bbox_to_anchor=(0.18, 0, 1, 1),
            bbox_transform=target_ax.transAxes,
            borderpad=0,
        )
        cbar = target_ax.figure.colorbar(im, cax=cbar_ax)
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=tick_fs, width=0.5, length=1.5)
        cbar_label_fs = tick_fs if parent_ax is not None else fs
        cbar.set_label(r"$-\log_{10}(\mathrm{P})$", fontsize=cbar_label_fs)

    return target_ax


# ---------------------------------------------------------------------------
# plot_scatter
# ---------------------------------------------------------------------------


def plot_scatter(
    ax,
    x,
    y,
    xlabel="",
    ylabel="",
    color_vals=None,
    color_label="",
    cmap="viridis",
    vmin=None,
    vmax=None,
    show_identity=False,
    show_colorbar=True,
    fit=None,
    show_ci=False,
    show_r2=False,
    marker_size=8,
    alpha=0.7,
    groups=None,
    group_labels=None,
    group_colors=None,
    show_legend=True,
    font_size=None,
):
    """
    Scatter plot comparing two arrays with optional color coding and regression.

    Supports two colouring modes (mutually exclusive):

    - **Continuous**: pass ``color_vals`` for a colormap-based colour scale
      with an optional colorbar.
    - **Discrete groups**: pass ``groups`` (integer index per point) to colour
      each group separately with its own legend entry.

    When ``groups`` is provided, ``color_vals``, ``cmap``, ``vmin``, ``vmax``,
    ``color_label``, and ``show_colorbar`` are ignored.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes (caller creates).
        x (np.ndarray): X-axis values.
        y (np.ndarray): Y-axis values.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        color_vals (np.ndarray or str or None): Per-point values for continuous
            color mapping. Pass the string ``"density"`` to auto-compute KDE
            density and sort points so dense regions render on top (requires
            scipy). If None and ``groups`` is also None, all points are drawn
            in a uniform colour.
        color_label (str): Colorbar label (continuous mode only).
        cmap (str): Matplotlib colormap name (continuous mode only).
        vmin (float or None): Colormap minimum (continuous mode only).
        vmax (float or None): Colormap maximum (continuous mode only).
        show_identity (bool): Plot the x = y identity line.
        show_colorbar (bool): Add a colorbar when *color_vals* is provided
            (continuous mode only).
        fit (str or None): Regression to overlay. ``"linear"`` or None.
        show_ci (bool): Show confidence interval band on the fit.
        show_r2 (bool): Annotate R-squared on the plot.
        marker_size (float): Scatter marker size.
        alpha (float): Scatter alpha.
        groups (array-like or None): Per-point integer group index for discrete
            colouring. Each unique value is rendered as a separate scatter
            series with its own colour and legend entry.
        group_labels (list[str] or None): Label for each unique group value,
            ordered by ``np.unique(groups)``. If None, the group values are
            used as labels.
        group_colors (list[str] or None): Colour for each unique group value,
            ordered by ``np.unique(groups)``. If None, uses the default
            matplotlib colour cycle.
        show_legend (bool): Show legend when ``groups`` is provided. Default
            True.
        font_size (int or None): Font size for labels/ticks. If None, uses
            current rcParams.

    Returns:
        sc (PathCollection or list[PathCollection]): In continuous mode, the
            single scatter artist (useful for shared colorbars). In group
            mode, a list of scatter artists (one per group).
    """
    _import_matplotlib()

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    sc = None

    if groups is not None:
        # --- Discrete group colouring -------------------------------------
        groups = np.asarray(groups).ravel()
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if group_colors is None:
            import matplotlib.pyplot as _plt

            cycle_colors = _plt.rcParams["axes.prop_cycle"].by_key()["color"]
            group_colors = [
                cycle_colors[i % len(cycle_colors)] for i in range(n_groups)
            ]

        if group_labels is None:
            group_labels = [str(g) for g in unique_groups]

        sc = []
        for i, g in enumerate(unique_groups):
            mask = groups == g
            s = ax.scatter(
                x[mask],
                y[mask],
                c=group_colors[i],
                s=marker_size,
                alpha=alpha,
                edgecolors="none",
                label=group_labels[i],
                zorder=2,
            )
            sc.append(s)

        if show_legend:
            ax.legend()
    else:
        # --- Continuous / uniform colouring -------------------------------
        scatter_kw = dict(s=marker_size, alpha=alpha, edgecolors="none", zorder=2)
        if color_vals is not None:
            if isinstance(color_vals, str) and color_vals == "density":
                try:
                    from scipy.stats import gaussian_kde
                except ImportError as e:
                    raise ImportError(
                        "color_vals='density' requires scipy. "
                        "Install with: pip install scipy"
                    ) from e
                valid = np.isfinite(x) & np.isfinite(y)
                xy = np.vstack([x[valid], y[valid]])
                kde = gaussian_kde(xy)
                density = kde(xy)
                sort_idx = density.argsort()
                x = x[valid][sort_idx]
                y = y[valid][sort_idx]
                color_vals = density[sort_idx]
            else:
                color_vals = np.asarray(color_vals, dtype=float).ravel()
            scatter_kw.update(c=color_vals, cmap=cmap)
            if vmin is not None:
                scatter_kw["vmin"] = vmin
            if vmax is not None:
                scatter_kw["vmax"] = vmax
        else:
            scatter_kw["c"] = "black"

        sc = ax.scatter(x, y, **scatter_kw)

    # --- Identity line ----------------------------------------------------
    if show_identity:
        lo = min(np.nanmin(x), np.nanmin(y))
        hi = max(np.nanmax(x), np.nanmax(y))
        ax.plot([lo, hi], [lo, hi], ls="--", color="grey", linewidth=0.8, zorder=1)

    # --- Regression fit ---------------------------------------------------
    if fit == "linear":
        from .stat_utils import linear_regression

        reg = linear_regression(x, y)
        ax.plot(reg["x_fit"], reg["y_fit"], color="red", linewidth=1.2, zorder=3)
        if show_ci:
            ax.fill_between(
                reg["x_fit"],
                reg["ci_lower"],
                reg["ci_upper"],
                color="red",
                alpha=0.15,
                zorder=1,
            )
        if show_r2:
            ax.annotate(
                f"$R^2 = {reg['r_squared']:.3f}$",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=font_size or 10,
            )
    elif fit is not None:
        raise ValueError(f"Unknown fit '{fit}'. Use 'linear' or None.")

    # --- Colorbar ---------------------------------------------------------
    if groups is None and color_vals is not None and show_colorbar:
        _add_colorbar(sc, ax, label=color_label, font_size=font_size or 14)

    # --- Axes formatting --------------------------------------------------
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if font_size is not None:
        _apply_font_size(ax, font_size)

    return sc


# ---------------------------------------------------------------------------
# plot_scatter_with_marginals
# ---------------------------------------------------------------------------


def plot_scatter_with_marginals(
    gs_slot,
    fig,
    x,
    y,
    xlabel="",
    ylabel="",
    marginal_bins=60,
    marginal_color="0.4",
    show_zero_lines=False,
    height_ratios=None,
    width_ratios=None,
    **scatter_kwargs,
):
    """
    Scatter plot with marginal histograms on the top and right edges.

    Creates a 2x2 sub-GridSpec inside *gs_slot* (top-left: x histogram,
    bottom-left: scatter, bottom-right: y histogram, top-right: empty).
    All scatter options are forwarded to :func:`plot_scatter`.

    Parameters:
        gs_slot: A GridSpec slot (e.g. ``gs[0]``) to place the sub-layout in.
        fig (matplotlib.Figure): Parent figure.
        x (np.ndarray): X-axis values.
        y (np.ndarray): Y-axis values.
        xlabel (str): X-axis label for the scatter.
        ylabel (str): Y-axis label for the scatter.
        marginal_bins (int): Number of histogram bins.
        marginal_color (str): Histogram bar colour.
        show_zero_lines (bool): Draw vertical/horizontal zero reference lines
            on the marginal histograms.
        height_ratios (list or None): ``[hist, scatter]`` height ratios.
            Default ``[1, 4]``.
        width_ratios (list or None): ``[scatter, hist]`` width ratios.
            Default ``[4, 1]``.
        **scatter_kwargs: Additional keyword arguments forwarded to
            :func:`plot_scatter` (e.g. ``color_vals``, ``show_identity``,
            ``marker_size``, ``cmap``).

    Returns:
        ax_scatter (matplotlib.axes.Axes): The main scatter axes.
        ax_histx (matplotlib.axes.Axes): Top marginal histogram axes.
        ax_histy (matplotlib.axes.Axes): Right marginal histogram axes.
        sc: Return value from :func:`plot_scatter`.
    """
    plt, _ = _import_matplotlib()
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    if height_ratios is None:
        height_ratios = [1, 4]
    if width_ratios is None:
        width_ratios = [4, 1]

    inner = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=gs_slot,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.05,
        wspace=0.05,
    )
    ax_scatter = fig.add_subplot(inner[1, 0])
    ax_histx = fig.add_subplot(inner[0, 0], sharex=ax_scatter)
    ax_histy = fig.add_subplot(inner[1, 1], sharey=ax_scatter)
    ax_corner = fig.add_subplot(inner[0, 1])
    ax_corner.axis("off")

    # Plot scatter
    sc = plot_scatter(ax_scatter, x, y, xlabel=xlabel, ylabel=ylabel, **scatter_kwargs)

    # Determine axis range from scatter
    xlim = ax_scatter.get_xlim()
    ylim = ax_scatter.get_ylim()

    # Marginal histograms
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    bins_x = np.linspace(xlim[0], xlim[1], marginal_bins)
    bins_y = np.linspace(ylim[0], ylim[1], marginal_bins)
    ax_histx.hist(x_arr[valid], bins=bins_x, color=marginal_color, edgecolor="none")
    ax_histy.hist(
        y_arr[valid],
        bins=bins_y,
        color=marginal_color,
        edgecolor="none",
        orientation="horizontal",
    )

    # Style marginal axes
    ax_histx.set_yticks([])
    ax_histy.set_xticks([])
    ax_histx.tick_params(labelbottom=False, bottom=False)
    ax_histy.tick_params(labelleft=False, left=False)
    for spine in ["top", "right", "left"]:
        ax_histx.spines[spine].set_visible(False)
    for spine in ["top", "right", "bottom"]:
        ax_histy.spines[spine].set_visible(False)

    if show_zero_lines:
        ax_histx.axvline(0, ls=":", color="red", lw=1.5)
        ax_histy.axhline(0, ls=":", color="red", lw=1.5)

    return ax_scatter, ax_histx, ax_histy, sc


# ---------------------------------------------------------------------------
# plot_manifold
# ---------------------------------------------------------------------------


def plot_manifold(
    ax,
    embedding,
    pc_x=0,
    pc_y=1,
    var_explained=None,
    bg_mask=None,
    bg_color="0.85",
    bg_alpha=0.05,
    bg_size=0.3,
    color_vals=None,
    color_label="",
    cmap="viridis",
    vmin=None,
    vmax=None,
    groups=None,
    group_labels=None,
    group_colors=None,
    marker_size=0.8,
    alpha=0.5,
    show_colorbar=True,
    show_legend=True,
    xlabel=None,
    ylabel=None,
    font_size=None,
):
    """
    Plot a 2-D embedding (PCA, UMAP, etc.) with flexible point coloring.

    Supports three foreground coloring modes (same as :func:`plot_scatter`):

    - **Continuous**: pass ``color_vals`` for colormap-scaled values.
    - **Discrete groups**: pass ``groups`` for per-group colours.
    - **Uniform**: neither provided — all foreground points drawn in black.

    An optional **background mask** renders selected points in a faint colour
    before the foreground, useful for separating non-event from event points
    (e.g. non-burst vs burst time bins).

    Parameters:
        ax (matplotlib.axes.Axes): Target axes (caller creates).
        embedding (np.ndarray): Shape ``(T, >=2)`` embedding coordinates.
        pc_x (int): Column index for the x-axis. Default 0.
        pc_y (int): Column index for the y-axis. Default 1.
        var_explained (np.ndarray or None): Explained variance ratio per
            component. When provided, axis labels are auto-generated as
            ``"PC{n} (X.X%)"``; overridden by explicit ``xlabel``/``ylabel``.
        bg_mask (np.ndarray or None): Boolean mask, shape ``(T,)``. True for
            background points. These are drawn first in ``bg_color``.
        bg_color (str): Colour for background points.
        bg_alpha (float): Alpha for background points.
        bg_size (float): Marker size for background points.
        color_vals (np.ndarray or str or None): Per-point values for
            continuous colour mapping (foreground only). Pass ``"density"``
            for KDE-based density colouring.
        color_label (str): Colorbar label (continuous mode).
        cmap (str): Matplotlib colourmap name (continuous mode).
        vmin (float or None): Colourmap minimum.
        vmax (float or None): Colourmap maximum.
        groups (array-like or None): Per-point integer group index for
            discrete colouring (foreground only).
        group_labels (list[str] or None): Labels per unique group value.
        group_colors (list[str] or None): Colours per unique group value.
        marker_size (float): Marker size for foreground points.
        alpha (float): Alpha for foreground points.
        show_colorbar (bool): Add a colorbar (continuous mode only).
        show_legend (bool): Show a legend (group mode only).
        xlabel (str or None): X-axis label. Overrides auto-label from
            ``var_explained``.
        ylabel (str or None): Y-axis label. Overrides auto-label from
            ``var_explained``.
        font_size (int or None): Font size for labels and ticks. If None,
            uses current rcParams.

    Returns:
        sc: The foreground scatter artist(s) — a single ``PathCollection``
            (continuous/uniform) or a ``list[PathCollection]`` (group mode).
            Useful for adding shared colorbars or custom legends.
    """
    _import_matplotlib()

    embedding = np.asarray(embedding)
    x = embedding[:, pc_x]
    y = embedding[:, pc_y]

    # --- Background points ------------------------------------------------
    if bg_mask is not None:
        bg_mask = np.asarray(bg_mask, dtype=bool)
        ax.scatter(
            x[bg_mask],
            y[bg_mask],
            s=bg_size,
            c=bg_color,
            alpha=bg_alpha,
            rasterized=True,
            edgecolors="none",
            zorder=1,
        )
        fg_mask = ~bg_mask
    else:
        fg_mask = np.ones(len(x), dtype=bool)

    # --- Foreground: delegate to plot_scatter -----------------------------
    fg_x = x[fg_mask]
    fg_y = y[fg_mask]

    scatter_kw = dict(
        marker_size=marker_size,
        alpha=alpha,
        show_colorbar=show_colorbar,
        show_legend=show_legend,
        font_size=font_size,
    )

    if color_vals is not None:
        if not isinstance(color_vals, str):
            color_vals = np.asarray(color_vals).ravel()[fg_mask]
        scatter_kw.update(
            color_vals=color_vals,
            color_label=color_label,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    elif groups is not None:
        groups = np.asarray(groups).ravel()[fg_mask]
        scatter_kw.update(
            groups=groups,
            group_labels=group_labels,
            group_colors=group_colors,
        )

    sc = plot_scatter(ax, fg_x, fg_y, **scatter_kw)

    # --- Axis labels ------------------------------------------------------
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    elif var_explained is not None:
        ax.set_xlabel(f"PC{pc_x + 1} ({var_explained[pc_x]:.1%})")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif var_explained is not None:
        ax.set_ylabel(f"PC{pc_y + 1} ({var_explained[pc_y]:.1%})")

    if font_size is not None:
        _apply_font_size(ax, font_size)

    return sc


# ---------------------------------------------------------------------------
# plot_lines
# ---------------------------------------------------------------------------


def plot_lines(
    ax,
    traces,
    x=None,
    labels=None,
    colors=None,
    xlabel="",
    ylabel="",
    linewidth=1.5,
    show_legend=True,
    font_size=None,
):
    """
    Plot one or more 1-D traces on a shared x-axis.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes (caller creates).
        traces (dict[str, np.ndarray] or list[np.ndarray]): Line data. Dict
            keys are used as labels; for list input, supply ``labels``
            separately.
        x (np.ndarray or None): Shared x-axis values. If None, integer
            indices (``0 … len-1``) are used.
        labels (list[str] or None): Per-trace labels. Required for list
            input; ignored for dict input (keys are used instead).
        colors (list[str] or dict[str, str] or None): Per-trace colours.
            For dict ``traces``, may be a dict keyed by the same labels or a
            list in the same order.  If None, uses the default matplotlib
            colour cycle.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        linewidth (float): Line width for all traces.
        show_legend (bool): Show legend. Default True.
        font_size (int or None): Font size for labels and ticks. If None,
            uses current rcParams.

    Returns:
        lines (list[Line2D]): The line artists.
    """
    _import_matplotlib()

    # --- Normalise input to ordered (label, array) pairs ------------------
    if isinstance(traces, dict):
        ordered_labels = list(traces.keys())
        ordered_data = [np.asarray(traces[k]).ravel() for k in ordered_labels]
    else:
        ordered_data = [np.asarray(a).ravel() for a in traces]
        if labels is not None:
            ordered_labels = list(labels)
        else:
            ordered_labels = [str(i) for i in range(len(ordered_data))]

    n = len(ordered_data)

    # --- Resolve x-axis ---------------------------------------------------
    if x is None:
        x = np.arange(len(ordered_data[0]))
    else:
        x = np.asarray(x).ravel()

    # --- Resolve colours --------------------------------------------------
    if colors is None:
        import matplotlib.pyplot as _plt

        cycle_colors = _plt.rcParams["axes.prop_cycle"].by_key()["color"]
        resolved_colors = [cycle_colors[i % len(cycle_colors)] for i in range(n)]
    elif isinstance(colors, dict):
        resolved_colors = [colors[lbl] for lbl in ordered_labels]
    else:
        resolved_colors = list(colors)

    # --- Draw lines -------------------------------------------------------
    lines = []
    for i in range(n):
        (line,) = ax.plot(
            x,
            ordered_data[i],
            color=resolved_colors[i],
            linewidth=linewidth,
            label=ordered_labels[i],
        )
        lines.append(line)

    # --- Axes formatting --------------------------------------------------
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_legend:
        ax.legend()

    if font_size is not None:
        _apply_font_size(ax, font_size)

    return lines


# ---------------------------------------------------------------------------
# plot_percentile_bands
# ---------------------------------------------------------------------------


def plot_percentile_bands(
    ax,
    metric_data,
    labels=None,
    normalize=False,
    summary="mean",
    bands=None,
    band_color="0.3",
    band_alphas=None,
    style="bands",
    line_color="0.5",
    line_alpha=0.3,
    line_width=0.5,
    summary_color="black",
    summary_linewidth=1.5,
    show_zero_line=True,
    xlabel="",
    ylabel="",
    ylim_range=None,
    show_legend=False,
    font_size=None,
):
    """
    Plot percentile bands or individual lines across ordered groups/conditions.

    For each unit a value is computed per condition. Optionally, values are
    normalized relative to the first group using symmetric normalization:
    ``N = (x' - d0') / (x' + d0')`` where ``x' = max(x, 0)``.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes (caller creates).
        metric_data (dict[str, np.ndarray] or list[np.ndarray]): Per-condition
            1-D arrays of per-unit values. Dict keys or ``labels`` define the
            x-axis order.
        labels (list[str] or None): Ordered condition labels. If None, uses
            dict keys (for dict input) or integer indices (for list input).
        normalize (bool): Apply symmetric normalization to the first group.
            Units with non-positive or NaN baseline values are excluded.
        summary (str): Summary line type: ``"mean"`` (default) or ``"median"``.
        bands (list[tuple[int, int]] or None): Percentile band definitions as
            ``(lo, hi)`` pairs, ordered from widest to narrowest. Default is
            ``[(5, 95), (10, 90), (25, 75)]``.
        band_color (str): Fill colour for all bands.
        band_alphas (list[float] or None): Alpha transparency per band. Must
            match length of ``bands``. Default is linearly increasing from
            0.15 to 0.40.
        style (str): ``"bands"`` (default) draws shaded percentile regions;
            ``"lines"`` draws one line per unit.
        line_color (str): Line colour when ``style="lines"``.
        line_alpha (float): Line alpha when ``style="lines"``.
        line_width (float): Line width when ``style="lines"``.
        summary_color (str): Colour for the summary line.
        summary_linewidth (float): Line width for the summary line.
        show_zero_line (bool): Draw a dashed horizontal line at y=0 when
            ``normalize=True``.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        ylim_range (float or None): Symmetric y-axis limits ``(-val, val)``.
            If None and ``normalize=True``, derived from the 5th/95th
            percentile of the data.
        show_legend (bool): Show legend.
        font_size (int or None): Font size for labels and ticks. If None,
            uses current rcParams.

    Returns:
        artists (dict): Keys ``"summary"`` (Line2D), and either ``"bands"``
            (list of PolyCollection) or ``"lines"`` (list of Line2D).
    """
    _import_matplotlib()

    # --- Normalise input to list-of-arrays + labels -----------------------
    if isinstance(metric_data, dict):
        keys = list(metric_data.keys())
        data_arrays = [np.asarray(metric_data[k]).ravel() for k in keys]
        if labels is None:
            labels = keys
    else:
        data_arrays = [np.asarray(a).ravel() for a in metric_data]
        if labels is None:
            labels = [str(i) for i in range(len(data_arrays))]

    n_groups = len(data_arrays)
    x = np.arange(n_groups)

    # --- Build (n_units, n_groups) matrix, optionally normalized ----------
    if normalize:
        d0 = np.maximum(data_arrays[0], 0)
        valid = (d0 > 0) & ~np.isnan(data_arrays[0])
        for arr in data_arrays[1:]:
            valid &= ~np.isnan(arr)

        n_units = int(np.sum(valid))
        mat = np.zeros((n_units, n_groups))
        for j, arr in enumerate(data_arrays):
            vals = np.maximum(arr[valid], 0)
            mat[:, j] = (vals - d0[valid]) / (vals + d0[valid])
    else:
        # Keep all non-NaN across every group
        valid = np.ones(len(data_arrays[0]), dtype=bool)
        for arr in data_arrays:
            valid &= ~np.isnan(arr)

        n_units = int(np.sum(valid))
        mat = np.column_stack([arr[valid] for arr in data_arrays])

    # --- Plot bands or individual lines -----------------------------------
    artists = {}

    if style == "bands":
        if bands is None:
            bands = [(5, 95), (10, 90), (25, 75)]
        if band_alphas is None:
            n_bands = len(bands)
            band_alphas = [
                0.15 + (0.40 - 0.15) * i / max(n_bands - 1, 1) for i in range(n_bands)
            ]

        band_artists = []
        for (lo_pct, hi_pct), alpha in zip(bands, band_alphas):
            lo_vals = np.nanpercentile(mat, lo_pct, axis=0)
            hi_vals = np.nanpercentile(mat, hi_pct, axis=0)
            label = f"{lo_pct}\u2013{hi_pct}th"
            fill = ax.fill_between(
                x,
                lo_vals,
                hi_vals,
                color=band_color,
                alpha=alpha,
                zorder=1,
                label=label,
            )
            band_artists.append(fill)
        artists["bands"] = band_artists

    elif style == "lines":
        line_artists = []
        for i in range(n_units):
            (ln,) = ax.plot(
                x,
                mat[i, :],
                color=line_color,
                alpha=line_alpha,
                linewidth=line_width,
                zorder=1,
            )
            line_artists.append(ln)
        artists["lines"] = line_artists

    else:
        raise ValueError(f"style must be 'bands' or 'lines', got {style!r}")

    # --- Summary line -----------------------------------------------------
    if summary == "mean":
        summary_vals = np.nanmean(mat, axis=0)
    elif summary == "median":
        summary_vals = np.nanmedian(mat, axis=0)
    else:
        raise ValueError(f"summary must be 'mean' or 'median', got {summary!r}")

    (summary_line,) = ax.plot(
        x,
        summary_vals,
        color=summary_color,
        linewidth=summary_linewidth,
        zorder=3,
        label=summary.capitalize(),
    )
    artists["summary"] = summary_line

    # --- Zero reference line ----------------------------------------------
    if show_zero_line and normalize:
        ax.axhline(0, color="0.4", linewidth=0.5, linestyle="--", zorder=0)

    # --- Axes formatting --------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if ylim_range is not None:
        ax.set_ylim(-ylim_range, ylim_range)
    elif normalize and mat.size > 0:
        p5 = np.nanpercentile(mat, 5, axis=0)
        p95 = np.nanpercentile(mat, 95, axis=0)
        ylim = max(abs(p5.min()), abs(p95.max())) * 1.15
        if np.isfinite(ylim) and ylim > 0:
            ax.set_ylim(-ylim, ylim)

    if show_legend:
        handles = [summary_line]
        if style == "bands":
            handles += artists["bands"]
        ax.legend(handles=handles, loc="upper left")

    if font_size is not None:
        _apply_font_size(ax, font_size)

    return artists


# ---------------------------------------------------------------------------
# plot_burst_sensitivity
# ---------------------------------------------------------------------------


def plot_burst_sensitivity(
    ax,
    thresholds,
    burst_counts,
    dist_values=None,
    labels=None,
    colors=None,
    xlabel="RMS mult.",
    ylabel="Number of bursts",
    show_legend=True,
    show_colorbar=True,
    cmap="hot",
    font_size=None,
):
    """
    Plot burst detection sensitivity as line plots or heatmaps.

    Automatically selects the visualisation based on the dimensionality of the
    burst count arrays:

    - **1-D** arrays (single-parameter sweep): one line per condition on a
      shared axes.
    - **2-D** arrays (two-parameter sweep over thresholds x distances, as
      returned by ``SpikeData.burst_sensitivity()``): one heatmap per
      condition via :func:`plot_heatmap`. A single condition is plotted on
      *ax*; multiple conditions produce a row of subplots on a new figure.

    Parameters:
        ax (matplotlib.axes.Axes or None): Target axes. Used directly for 1-D
            line plots and single-condition 2-D heatmaps. For multi-condition
            2-D heatmaps this parameter is ignored and a new figure is created
            (pass None explicitly in that case).
        thresholds (np.ndarray): 1-D array of threshold values (x-axis).
        burst_counts (dict[str, np.ndarray] or np.ndarray): Burst counts per
            condition. Dict mapping condition labels to arrays, or a bare
            ``np.ndarray`` for a single unnamed condition. Arrays can be 1-D
            (line plot) or 2-D of shape
            ``(len(thresholds), len(dist_values))`` (heatmap).
        dist_values (np.ndarray or None): 1-D array of distance parameter
            values. Required when burst counts are 2-D (used as y-axis tick
            labels on the heatmap). Ignored for 1-D line plots.
        labels (list[str] or None): Ordered condition labels. If None, uses
            dict keys.
        colors (list[str] or None): Per-condition line colours (line plots
            only). If None, uses the default matplotlib colour cycle.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label. For 2-D heatmaps, defaults to
            ``"Min. burst dist. (bins)"`` when not explicitly changed from the
            default.
        show_legend (bool): Whether to show a legend (line plots only).
        show_colorbar (bool): Whether to show a colorbar (heatmaps only).
        cmap (str): Colormap for heatmaps.
        font_size (int or None): Font size for labels/ticks. If None, uses
            current rcParams.

    Returns:
        result: For 1-D line plots: ``list[Line2D]`` artists. For a single
            2-D heatmap: the axes returned by :func:`plot_heatmap`. For
            multiple 2-D heatmaps: ``(fig, axes_list)`` where *axes_list*
            contains one axes per condition.
    """
    plt, _ = _import_matplotlib()

    thresholds = np.asarray(thresholds).ravel()

    # --- Normalise burst_counts to an ordered dict ------------------------
    if isinstance(burst_counts, np.ndarray):
        burst_counts = {"": burst_counts}
    elif not isinstance(burst_counts, dict):
        burst_counts = {"": np.asarray(burst_counts).ravel()}

    if labels is None:
        labels = list(burst_counts.keys())

    first_val = np.asarray(burst_counts[labels[0]])
    is_2d = first_val.ndim == 2

    # --- 2-D heatmap path -------------------------------------------------
    if is_2d:
        if dist_values is None:
            raise ValueError(
                "dist_values is required for 2-D burst sensitivity heatmaps."
            )
        dist_values = np.asarray(dist_values).ravel()
        heatmap_ylabel = (
            "Min. burst dist. (bins)" if ylabel == "Number of bursts" else ylabel
        )
        n_thr = len(thresholds)
        n_dist = len(dist_values)
        extent = (
            thresholds[0],
            thresholds[-1],
            dist_values[0],
            dist_values[-1],
        )
        xticks = (
            np.linspace(thresholds[0], thresholds[-1], min(n_thr, 6)),
            [
                f"{v:.1f}"
                for v in np.linspace(thresholds[0], thresholds[-1], min(n_thr, 6))
            ],
        )
        yticks = (
            np.linspace(dist_values[0], dist_values[-1], min(n_dist, 6)),
            [
                f"{v:.0f}"
                for v in np.linspace(dist_values[0], dist_values[-1], min(n_dist, 6))
            ],
        )
        fs = font_size or 14

        # Single condition — plot on the provided ax
        if len(labels) == 1:
            return plot_heatmap(
                np.asarray(burst_counts[labels[0]]).T,
                ax=ax,
                cmap=cmap,
                origin="lower",
                extent=extent,
                xlabel=xlabel,
                ylabel=heatmap_ylabel,
                show_colorbar=show_colorbar,
                colorbar_label="Burst count",
                xticks=xticks,
                yticks=yticks,
                font_size=fs,
            )

        # Multiple conditions — create a subplot row
        n_cond = len(labels)
        fig, axes_row = plt.subplots(1, n_cond, figsize=(5 * n_cond, 4), squeeze=False)
        axes_row = axes_row[0]

        # Compute shared colour range across all conditions
        all_arrays = [np.asarray(burst_counts[l]) for l in labels]
        shared_vmin = min(a.min() for a in all_arrays)
        shared_vmax = max(a.max() for a in all_arrays)

        for i, label in enumerate(labels):
            plot_heatmap(
                all_arrays[i].T,
                ax=axes_row[i],
                cmap=cmap,
                origin="lower",
                vmin=shared_vmin,
                vmax=shared_vmax,
                extent=extent,
                xlabel=xlabel,
                ylabel=heatmap_ylabel if i == 0 else "",
                show_colorbar=show_colorbar and i == n_cond - 1,
                colorbar_label="Burst count",
                xticks=xticks,
                yticks=yticks if i == 0 else (yticks[0], [""] * len(yticks[1])),
                font_size=fs,
            )
            axes_row[i].set_title(label, fontsize=fs)

        fig.tight_layout()
        return fig, list(axes_row)

    # --- 1-D line plot path -----------------------------------------------
    traces = {label: np.asarray(burst_counts[label]).ravel() for label in labels}
    return plot_lines(
        ax,
        traces,
        x=thresholds,
        colors=colors,
        xlabel=xlabel,
        ylabel=ylabel,
        show_legend=show_legend,
        font_size=font_size,
    )


# ---------------------------------------------------------------------------
# plot_aligned_slice_single_unit
# ---------------------------------------------------------------------------


def plot_aligned_slice_single_unit(
    ax,
    spike_times_per_slice,
    color_vals=None,
    color_label="",
    cmap="viridis",
    time_offset=0,
    xlabel="Rel. time (ms)",
    ylabel="Burst",
    x_range=None,
    vlines=None,
    show_colorbar=True,
    marker_size=20,
    font_size=None,
    style="scatter",
    invert_y=False,
    linewidths=0.5,
):
    """
    Raster plot of one unit's spike times across multiple event-aligned slices.

    Each row corresponds to one slice/burst, x-axis is time relative to the
    alignment point. Optionally colour-codes rows by a per-slice variable.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes (caller creates).
        spike_times_per_slice (list[np.ndarray]): List of 1-D arrays, one per
            slice, containing spike times relative to the alignment point.
        color_vals (np.ndarray or None): Per-slice colour values. If None,
            all spikes are drawn in black.
        color_label (str): Colorbar label.
        cmap (str): Matplotlib colormap name.
        time_offset (float): Value subtracted from spike times for display.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        x_range (tuple or None): ``(xmin, xmax)`` for the x-axis. If None,
            auto-scales to the data.
        vlines (list[float] or None): X positions for vertical reference
            lines (e.g. burst onset).
        show_colorbar (bool): Add a colorbar when *color_vals* is provided.
        marker_size (float): Scatter marker size (only used when
            ``style="scatter"``).
        font_size (int or None): Font size for labels/ticks. If None, uses
            current rcParams.
        style (str): ``"scatter"`` for dot markers (default), or
            ``"eventplot"`` for vertical line markers.
        invert_y (bool): If True, the first slice is plotted at the top and
            the last at the bottom. Default False (first slice at bottom).
        linewidths (float): Line width for eventplot markers (only used when
            ``style="eventplot"``). Default 0.5.

    Returns:
        sc (PathCollection or None): The scatter artist when *color_vals* is
            provided and ``style="scatter"``, otherwise None.
    """
    _import_matplotlib()

    n_slices = len(spike_times_per_slice)

    # Shift spike times
    shifted_per_slice = []
    for times in spike_times_per_slice:
        times = np.asarray(times, dtype=float).ravel()
        shifted_per_slice.append(times - time_offset)

    sc = None

    if style == "eventplot":
        ax.eventplot(
            shifted_per_slice,
            colors="black",
            linewidths=linewidths,
        )
    else:
        # Flatten spike times into (x, y, c) arrays for scatter
        x_all = []
        y_all = []
        c_all = []
        for idx, shifted in enumerate(shifted_per_slice):
            x_all.append(shifted)
            y_all.append(np.full(len(shifted), idx))
            if color_vals is not None:
                c_all.append(np.full(len(shifted), color_vals[idx]))

        if len(x_all) == 0:
            return None

        x_all = np.concatenate(x_all)
        y_all = np.concatenate(y_all)

        if color_vals is not None and len(c_all) > 0:
            c_all = np.concatenate(c_all)
            sc = ax.scatter(x_all, y_all, c=c_all, cmap=cmap, s=marker_size, zorder=2)
            if show_colorbar:
                _add_colorbar(sc, ax, label=color_label, font_size=font_size or 14)
        else:
            ax.scatter(x_all, y_all, c="black", s=marker_size, zorder=2)

    # --- Vertical reference lines -----------------------------------------
    if vlines is not None:
        for xv in vlines:
            ax.axvline(x=xv, color="red", linestyle="--", linewidth=1.5, zorder=0)

    # --- Axes formatting --------------------------------------------------
    ax.set_ylim(0, n_slices)
    if invert_y:
        ax.invert_yaxis()
    if x_range is not None:
        ax.set_xlim(x_range)
    else:
        non_empty = [s for s in shifted_per_slice if len(s) > 0]
        if non_empty:
            all_shifted = np.concatenate(non_empty)
            ax.set_xlim(np.min(all_shifted), np.max(all_shifted))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if font_size is not None:
        _apply_font_size(ax, font_size)

    return sc


# ---------------------------------------------------------------------------
# plot_heatmap
# ---------------------------------------------------------------------------


def plot_heatmap(
    data_mat,
    ax=None,
    norm=False,
    vmin=None,
    vmax=None,
    cmap="hot",
    aspect="auto",
    origin="upper",
    extent=None,
    xlabel="Time (ms)",
    ylabel="Unit",
    xticks=None,
    yticks=None,
    vlines=None,
    hlines=None,
    show_colorbar=True,
    colorbar_label="Rate (Hz)",
    font_size=14,
    save_path=None,
):
    """
    Plot a 2-D matrix as a heatmap.

    Parameters:
        data_mat (np.ndarray): 2-D array to display, shape ``(rows, cols)``.
        ax (matplotlib.axes.Axes or None): Target axes. If None a standalone
            figure is created.
        norm (bool or str): Row normalisation. ``False`` for none, ``'row'``
            to scale each row to [0, 1].
        vmin (float or None): Colormap minimum.
        vmax (float or None): Colormap maximum.
        cmap (str): Matplotlib colormap name.
        aspect (str): Aspect ratio passed to ``imshow``. ``"auto"`` stretches
            to fill the axes, ``"equal"`` preserves square pixels.
        origin (str): Origin for ``imshow``. ``"upper"`` places row 0 at the
            top (default), ``"lower"`` places row 0 at the bottom.
        extent (tuple or None): ``(left, right, bottom, top)`` passed to
            ``imshow``. If None, pixel indices are used.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        xticks (tuple or None): ``(locations, labels)`` for x-axis ticks.
        yticks (tuple or None): ``(locations, labels)`` for y-axis ticks.
        vlines (list[dict] or None): Vertical lines. Each dict may contain
            ``'x'``, ``'color'``, ``'linestyle'``, ``'linewidth'``.
        hlines (list[dict] or None): Horizontal lines (same keys as vlines
            but with ``'y'``).
        show_colorbar (bool): Whether to draw a colorbar.
        colorbar_label (str): Label for the colorbar.
        font_size (int): Font size for labels and ticks.
        save_path (str or None): If provided (and ``ax`` is None), save the
            figure to this path and close it.

    Returns:
        result: ``(fig, ax)`` when ``ax`` is None, otherwise just ``ax``.
    """
    plt, _ = _import_matplotlib()

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Normalise
    if norm == "row":
        result = np.zeros_like(data_mat, dtype=float)
        for i in range(data_mat.shape[0]):
            row_min, row_max = data_mat[i].min(), data_mat[i].max()
            if row_max > row_min:
                result[i] = (data_mat[i] - row_min) / (row_max - row_min)
            else:
                result[i] = data_mat[i]
        data_mat = result
        colorbar_label = "Norm. " + colorbar_label
        if vmax is None:
            vmax = 1.0

    im_kwargs = dict(cmap=cmap, aspect=aspect, origin=origin, interpolation="none")
    if extent is not None:
        im_kwargs["extent"] = extent
    if vmin is not None:
        im_kwargs["vmin"] = vmin
    if vmax is not None:
        im_kwargs["vmax"] = vmax

    im = ax.imshow(data_mat, **im_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if yticks is not None:
        ax.set_yticks(yticks[0])
        ax.set_yticklabels(yticks[1])
    else:
        ax.set_yticks([0, data_mat.shape[0] - 1])
        ax.set_yticklabels([1, data_mat.shape[0]])

    if xticks is not None:
        ax.set_xticks(xticks[0])
        ax.set_xticklabels(xticks[1])

    if vlines:
        for vl in vlines:
            ax.axvline(
                x=vl["x"],
                color=vl.get("color", "green"),
                linestyle=vl.get("linestyle", "dotted"),
                linewidth=vl.get("linewidth", 2),
            )

    if hlines:
        for hl in hlines:
            ax.axhline(
                y=hl["y"],
                color=hl.get("color", "green"),
                linestyle=hl.get("linestyle", "dotted"),
                linewidth=hl.get("linewidth", 2),
            )

    if show_colorbar:
        _add_colorbar(im, ax, label=colorbar_label, font_size=font_size)

    _apply_font_size(ax, font_size)

    if standalone:
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
            plt.close(fig)
        return fig, ax

    return ax


# ---------------------------------------------------------------------------
# plot_recording
# ---------------------------------------------------------------------------


def plot_recording(
    sd,
    # --- panel toggles ---
    show_raster=True,
    show_pop_rate=False,
    show_fr_rates=False,
    show_model_states=False,
    # --- data inputs (None = auto-compute where possible) ---
    pop_rate=None,
    pop_rate_params=None,
    fr_rates=None,
    fr_rate_sigma_ms=10.0,
    model_states=None,
    cont_prob=None,
    gplvm_result=None,
    # --- display options ---
    time_range=None,
    sort_indices=None,
    raster_style="eventplot",
    raster_bin_size_ms=1.0,
    raster_vmax=3,
    burst_times=None,
    burst_edges=None,
    # --- heatmap options ---
    vmin_heatmap=None,
    vmax_heatmap=None,
    norm_heatmap=False,
    model_states_cmap="viridis",
    model_states_vmin=0,
    model_states_vmax=1,
    # --- layout ---
    axes=None,
    figsize=None,
    height_ratios=None,
    absolute_xticks=True,
    font_size=14,
    show=True,
    save_path=None,
):
    """
    Assemble a multi-panel column figure from a SpikeData object.

    Each panel is optional — the caller selects which panels to include and
    they are stacked vertically with a shared x-axis. Available panels (in
    display order):

    1. **Spike raster** — eventplot or binned-count image.
    2. **Population rate** — smoothed firing rate curve; if ``cont_prob`` is
       also provided it is overlaid on the same axes.
    3. **Firing-rate heatmap** — per-unit instantaneous rates as a colour map.
    4. **Model states** — latent-state posterior from a GPLVM fit.

    Parameters:
        sd (SpikeData): Source spike data object.
        show_raster (bool): Include the spike raster panel.
        show_pop_rate (bool): Include the population-rate panel. Automatically
            enabled when ``pop_rate`` or ``cont_prob`` is provided.
        show_fr_rates (bool): Include the firing-rate heatmap. Automatically
            enabled when ``fr_rates`` is provided.
        show_model_states (bool): Include the model-states panel.
            Automatically enabled when ``model_states`` is provided.
        pop_rate (np.ndarray or None): Pre-computed smoothed population rate,
            shape ``(T,)``. If None and panel is enabled, computed via
            ``sd.get_pop_rate()``.
        pop_rate_params (dict or None): Keyword arguments forwarded to
            ``sd.get_pop_rate()`` when ``pop_rate`` is None. Defaults:
            ``square_width=5, gauss_sigma=5``.
        fr_rates (np.ndarray or None): Pre-computed per-unit instantaneous
            firing rates, shape ``(U, T)``. If None and ``show_fr_rates`` is
            True, computed via ``sd.resampled_isi()``.
        fr_rate_sigma_ms (float): Gaussian sigma in ms for
            ``sd.resampled_isi()`` when auto-computing firing rates.
        model_states (np.ndarray or None): Latent-state posterior, shape
            ``(S, T)`` where S is the number of latent states. If None and
            ``show_model_states`` is True, extracted from ``gplvm_result``.
        cont_prob (np.ndarray or None): Continuous-dynamics probability,
            shape ``(T,)``. Overlaid on the population-rate panel. If None
            and ``gplvm_result`` is provided, extracted automatically.
        gplvm_result (dict or None): Result dictionary from
            ``SpikeData.fit_gplvm()``. Used to auto-extract ``model_states``
            and ``cont_prob`` when those are not provided directly.
        time_range (tuple or None): ``(start_ms, end_ms)`` display window.
            None shows the full recording.
        sort_indices (np.ndarray or None): Unit reordering indices applied to
            the raster and firing-rate heatmap.
        raster_style (str): ``'eventplot'`` (default) or ``'imshow'``. Controls
            how the raster panel is rendered.
        raster_bin_size_ms (float): Bin size in ms for the raster (used for
            both eventplot and imshow styles).
        raster_vmax (int): Maximum spike count for colormap when
            ``raster_style='imshow'``.
        burst_times (np.ndarray or None): Burst peak positions as raster bin
            indices (as returned by ``get_bursts``), plotted as markers on the
            population-rate panel. With the default ``raster_bin_size_ms=1.0``,
            bin indices equal milliseconds.
        burst_edges (np.ndarray or None): Per-burst ``[start_bin, end_bin]``
            boundaries as raster bin indices, shape ``(B, 2)``. Plotted as
            shaded spans on the population-rate panel.
        vmin_heatmap (float or None): Colormap minimum for the FR heatmap.
        vmax_heatmap (float or None): Colormap maximum for the FR heatmap.
        norm_heatmap (bool or str): Row normalisation for the FR heatmap
            (``False`` or ``'row'``).
        model_states_cmap (str): Colormap for the model-states panel.
        model_states_vmin (float): Colormap minimum for model states.
        model_states_vmax (float): Colormap maximum for model states.
        axes (list or None): Pre-created axes to plot onto instead of
            creating a new figure. Must be a list of ``(ax_panel, ax_cbar)``
            tuples, one per enabled panel, in display order (raster,
            pop_rate, fr_heatmap, model_states — only those that are
            enabled). ``ax_cbar`` is used for colorbars on heatmap/imshow
            panels; pass a hidden axes if no colorbar is needed for that
            panel. When provided, the function skips figure creation,
            ``tight_layout``, ``show``, and ``save_path``. ``figsize`` and
            ``height_ratios`` are ignored. Raises ``ValueError`` if the
            length does not match the number of enabled panels.
        figsize (tuple or None): Figure size ``(width, height)``.
            Ignored when ``axes`` is provided.
        height_ratios (list or None): Relative panel heights. Length must
            match the number of enabled panels.
        absolute_xticks (bool): If True, x-tick labels show absolute ms from
            recording start. If False, labels are relative to the display
            window.
        font_size (int): Font size for labels and tick labels.
        show (bool): If True and ``save_path`` is None, call ``plt.show()``.
        save_path (str or None): Save figure to this path and close it.

    Returns:
        fig (matplotlib.Figure): The assembled figure.

    Notes:
        - At least one panel must be enabled, otherwise ``ValueError`` is
          raised.
        - When ``gplvm_result`` is provided, ``model_states`` and
          ``cont_prob`` are extracted from the ``decode_res`` sub-dict
          (keys ``posterior_latent_marg`` and ``posterior_dynamics_marg``).
          Arrays with a different time resolution (e.g. GPLVM binned
          output) are automatically cropped to match the ms-based
          ``time_range``.
    """
    plt, mticker = _import_matplotlib()

    # ------------------------------------------------------------------
    # 0. Auto-extract from gplvm_result
    # ------------------------------------------------------------------
    if gplvm_result is not None:
        decode = gplvm_result.get("decode_res", {})
        if model_states is None and "posterior_latent_marg" in decode:
            model_states = np.asarray(decode["posterior_latent_marg"]).T
        if cont_prob is None and "posterior_dynamics_marg" in decode:
            dyn = np.asarray(decode["posterior_dynamics_marg"])
            cont_prob = dyn[:, 0] if dyn.ndim == 2 else dyn

    # ------------------------------------------------------------------
    # 1. Resolve panel flags — auto-enable when data is provided
    # ------------------------------------------------------------------
    if pop_rate is not None or cont_prob is not None:
        show_pop_rate = True
    if fr_rates is not None:
        show_fr_rates = True
    if model_states is not None:
        show_model_states = True

    panels = []
    if show_raster:
        panels.append("raster")
    if show_pop_rate:
        panels.append("pop_rate")
    if show_fr_rates:
        panels.append("fr_heatmap")
    if show_model_states:
        panels.append("model_states")

    if not panels:
        raise ValueError(
            "No panels enabled. Set at least one of show_raster, "
            "show_pop_rate, show_fr_rates, or show_model_states to True, "
            "or provide data for auto-detection."
        )

    n_panels = len(panels)

    # ------------------------------------------------------------------
    # 2. Build spike matrix
    # ------------------------------------------------------------------
    spk_mat = sd.sparse_raster(bin_size=raster_bin_size_ms).toarray()

    # Auto-compute firing rates if requested
    if show_fr_rates and fr_rates is None:
        times_arr = np.arange(sd.start_time, sd.start_time + sd.length, 1.0)
        fr_rates = sd.resampled_isi(times_arr, sigma_ms=fr_rate_sigma_ms)

    # Apply unit reordering
    if sort_indices is not None:
        spk_mat = spk_mat[sort_indices, :]
        if fr_rates is not None:
            fr_rates = fr_rates[sort_indices, :]

    # Auto-compute population rate if requested
    if show_pop_rate and pop_rate is None:
        params = {"square_width": 5, "gauss_sigma": 5}
        if pop_rate_params is not None:
            params.update(pop_rate_params)
        pop_rate = sd.get_pop_rate(**params)

    # ------------------------------------------------------------------
    # 3. Crop to time_range
    # ------------------------------------------------------------------
    if time_range is not None:
        # Convert ms time_range to bin indices relative to the raster.
        # Bin 0 corresponds to sd.start_time.
        start = int((time_range[0] - sd.start_time) / raster_bin_size_ms)
        end = int((time_range[1] - sd.start_time) / raster_bin_size_ms)
        start = max(0, min(start, spk_mat.shape[1]))
        end = max(start, min(end, spk_mat.shape[1]))
    else:
        start, end = 0, spk_mat.shape[1]

    spk_mat_view = spk_mat[:, start:end]
    n_samples = end - start

    # Crop arrays whose time axis matches the raster resolution.
    # Arrays with a different time resolution (e.g. GPLVM binned output)
    # are cropped using proportional index conversion so the correct
    # time window is displayed.
    raster_T = spk_mat.shape[1]

    def _rescaled_range(arr_len):
        """Convert raster-resolution [start, end) to indices for an array of length arr_len."""
        if arr_len == raster_T:
            return start, end
        scale = arr_len / raster_T
        s = max(0, min(int(round(start * scale)), arr_len))
        e = max(s, min(int(round(end * scale)), arr_len))
        return s, e

    def _crop_1d(arr):
        if arr is None:
            return None
        s, e = _rescaled_range(len(arr))
        return arr[s:e]

    def _crop_2d(arr):
        if arr is None:
            return None
        s, e = _rescaled_range(arr.shape[-1])
        return arr[:, s:e]

    pop_rate_view = _crop_1d(pop_rate)
    fr_rates_view = _crop_2d(fr_rates)
    cont_prob_view = _crop_1d(cont_prob)
    model_states_view = _crop_2d(model_states)

    # Burst peaks: keep those inside range, shift to window coords
    burst_times_view = None
    if burst_times is not None:
        mask = (burst_times >= start) & (burst_times <= end)
        burst_times_view = np.round(burst_times[mask] - start).astype(int)

    # Burst edges: clip to range, skip fully-outside bursts
    burst_edges_view = None
    if burst_edges is not None:
        clipped = []
        for t0, t1 in burst_edges:
            if t1 < start or t0 > end:
                continue
            clipped.append((max(t0, start) - start, min(t1, end) - start))
        burst_edges_view = clipped if clipped else None

    # ------------------------------------------------------------------
    # 4. Create figure with two-column GridSpec (panels + colorbars)
    # ------------------------------------------------------------------
    from matplotlib.gridspec import GridSpec

    external_axes = axes is not None

    if external_axes:
        if len(axes) != n_panels:
            raise ValueError(
                f"Expected {n_panels} (ax, cbar_ax) pairs for the enabled "
                f"panels {panels}, got {len(axes)}."
            )
        main_axes = [pair[0] for pair in axes]
        cbar_axes = [pair[1] for pair in axes]
        fig = main_axes[0].figure
    else:
        default_ratio_map = {
            "raster": 2,
            "pop_rate": 1,
            "fr_heatmap": 2,
            "model_states": 2,
        }
        default_ratios = [default_ratio_map[p] for p in panels]
        default_height = sum(default_ratios) * 1.7
        default_figsize = (12, default_height)

        fig = plt.figure(figsize=figsize or default_figsize)
        gs = GridSpec(
            n_panels,
            2,
            figure=fig,
            height_ratios=height_ratios or default_ratios,
            width_ratios=[1, 0.02],
            wspace=0.03,
        )

        # Create main panel axes with shared x
        main_axes = []
        for i in range(n_panels):
            ax = fig.add_subplot(gs[i, 0], sharex=main_axes[0] if main_axes else None)
            main_axes.append(ax)

        # Create colorbar axes (one per row, hidden by default)
        cbar_axes = []
        for i in range(n_panels):
            cax = fig.add_subplot(gs[i, 1])
            cax.axis("off")
            cbar_axes.append(cax)

    panel_axes = dict(zip(panels, main_axes))
    panel_cbar = dict(zip(panels, cbar_axes))

    # ------------------------------------------------------------------
    # 5. Raster panel
    # ------------------------------------------------------------------
    if "raster" in panel_axes:
        ax = panel_axes["raster"]
        if raster_style == "imshow":
            im = ax.imshow(
                spk_mat_view,
                aspect="auto",
                cmap="Greys",
                vmin=0,
                vmax=raster_vmax,
                origin="lower",
            )
            cax = panel_cbar["raster"]
            cax.axis("on")
            fig.colorbar(im, cax=cax, label="Spike Count")
            cax.tick_params(labelsize=font_size)
        else:
            spike_times_list = [
                np.where(spk_mat_view[i, :] >= 1)[0]
                for i in range(spk_mat_view.shape[0])
            ]
            ax.eventplot(spike_times_list, colors="black", linewidths=0.5)
        ax.set_ylim([0, spk_mat_view.shape[0]])
        ax.set_ylabel("Unit")
        _apply_font_size(ax, font_size)

    # ------------------------------------------------------------------
    # 6. Population rate panel (+ cont_prob overlay)
    # ------------------------------------------------------------------
    if "pop_rate" in panel_axes:
        ax = panel_axes["pop_rate"]

        if pop_rate_view is not None:
            x_pop = np.linspace(0, n_samples, len(pop_rate_view), endpoint=False)
            ax.plot(x_pop, pop_rate_view, color="blue", label="Pop. rate")
            ax.set_ylabel("Pop. rate (Hz)")

        if cont_prob_view is not None:
            ax2 = ax.twinx()
            x_cont = np.linspace(0, n_samples, len(cont_prob_view), endpoint=False)
            ax2.plot(x_cont, cont_prob_view, color="red", alpha=0.7, label="P(cont.)")
            ax2.set_ylabel("P(continuous)", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            _apply_font_size(ax2, font_size)

        # Burst overlays
        if burst_times_view is not None and pop_rate_view is not None:
            # Scale burst times from raster-bin coords to pop_rate coords
            scale = len(pop_rate_view) / n_samples
            bt_scaled = np.round(burst_times_view * scale).astype(int)
            valid = bt_scaled < len(pop_rate_view)
            bt_scaled = bt_scaled[valid]
            bt_plot = burst_times_view[valid]  # x position in raster coords
            ax.scatter(bt_plot, pop_rate_view[bt_scaled], c="k", zorder=9)

        if burst_edges_view is not None:
            for t0, t1 in burst_edges_view:
                ax.axvspan(t0, t1, color="b", alpha=0.2)

        _apply_font_size(ax, font_size)

    # ------------------------------------------------------------------
    # 7. Firing-rate heatmap panel
    # ------------------------------------------------------------------
    if "fr_heatmap" in panel_axes:
        ax = panel_axes["fr_heatmap"]
        fr_extent = None
        if fr_rates_view.shape[1] != n_samples:
            fr_extent = (0, n_samples, 0, fr_rates_view.shape[0])
        plot_heatmap(
            fr_rates_view,
            ax=ax,
            norm=norm_heatmap,
            vmin=vmin_heatmap,
            vmax=vmax_heatmap,
            origin="lower",
            extent=fr_extent,
            xlabel="Time (ms)",
            ylabel="Unit",
            show_colorbar=False,
            font_size=font_size,
        )
        cax = panel_cbar["fr_heatmap"]
        cax.axis("on")
        cb_label = "Norm. Rate (Hz)" if norm_heatmap else "Rate (Hz)"
        fig.colorbar(ax.images[0], cax=cax, label=cb_label)
        cax.tick_params(labelsize=font_size)

    # ------------------------------------------------------------------
    # 8. Model states panel
    # ------------------------------------------------------------------
    if "model_states" in panel_axes:
        ax = panel_axes["model_states"]
        ms_extent = None
        if model_states_view.shape[1] != n_samples:
            ms_extent = (0, n_samples, 0, model_states_view.shape[0])
        plot_heatmap(
            model_states_view,
            ax=ax,
            cmap=model_states_cmap,
            vmin=model_states_vmin,
            vmax=model_states_vmax,
            origin="lower",
            extent=ms_extent,
            xlabel="Time (ms)",
            ylabel="State",
            show_colorbar=False,
            font_size=font_size,
        )
        cax = panel_cbar["model_states"]
        cax.axis("on")
        fig.colorbar(ax.images[0], cax=cax, label="Probability")
        cax.tick_params(labelsize=font_size)

    # ------------------------------------------------------------------
    # 9. X-axis formatting
    # ------------------------------------------------------------------
    # Set x limits on all axes (sharex propagates when axes are created
    # internally, but external axes may not be linked)
    for ax in main_axes:
        ax.set_xlim(0, n_samples)

    # Hide tick labels on all but the bottom panel
    for ax in main_axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("")
    main_axes[-1].set_xlabel("Time (ms)")
    _apply_font_size(main_axes[-1], font_size)

    # Shift tick labels to absolute recording time when requested
    # Convert bin indices back to ms for tick labels.
    # Bin 0 in the view corresponds to ms = sd.start_time + start * raster_bin_size_ms.
    ms_offset = sd.start_time + start * raster_bin_size_ms
    if absolute_xticks and ms_offset != 0:
        formatter = mticker.FuncFormatter(
            lambda x, _: f"{int(x * raster_bin_size_ms + ms_offset)}"
        )
        for ax in main_axes:
            ax.xaxis.set_major_formatter(formatter)

    # ------------------------------------------------------------------
    # 10. Output
    # ------------------------------------------------------------------
    if not external_axes:
        gs.tight_layout(fig)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        elif show:
            plt.show()

    return fig


def plot_spatial_network(
    ax,
    positions,
    matrix,
    edge_threshold=None,
    top_pct=None,
    node_size_range=(2, 20),
    node_cmap="viridis",
    node_linewidth=0.2,
    edge_color="red",
    edge_linewidth=0.6,
    edge_alpha_range=(0.15, 1.0),
    scale_bar_um=500,
    font_size=None,
):
    """
    Plot a spatial network of units on their MEA positions.

    Units are drawn as scatter markers sized by their mean pairwise value
    (row mean excluding diagonal) and coloured by the same metric.  Edges
    are drawn between unit pairs whose matrix value exceeds a threshold or
    falls in the top percentile, with alpha proportional to edge strength.

    Exactly one of *edge_threshold* or *top_pct* must be provided.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes.
        positions (np.ndarray): Unit positions, shape ``(N, 2)`` with columns
            ``[x, y]`` in micrometres.
        matrix (np.ndarray): Symmetric ``(N, N)`` pairwise matrix (e.g.
            correlation, STTC).  Diagonal values are ignored.
        edge_threshold (float or None): Minimum matrix value to draw an edge.
        top_pct (float or None): Percentage of top edges to draw (e.g.
            ``1.0`` draws the top 1 %).
        node_size_range (tuple): ``(min_size, max_size)`` in points² for the
            scatter markers.
        node_cmap (str): Matplotlib colourmap for node colour.
        node_linewidth (float): Outline width of node markers.
        edge_color (str): Colour for network edges.
        edge_linewidth (float): Line width for network edges.
        edge_alpha_range (tuple): ``(min_alpha, max_alpha)`` for edge
            transparency, scaled by edge strength.
        scale_bar_um (float): Length of the spatial scale bar in micrometres.
            Set to 0 or None to omit.
        font_size (int or None): Font size for scale bar label.  If None,
            uses the current rcParams default.

    Returns:
        scatter (matplotlib.collections.PathCollection): The scatter artist,
            useful for adding a colorbar.
    """
    _import_matplotlib()

    if edge_threshold is None and top_pct is None:
        raise ValueError("Provide either edge_threshold or top_pct.")
    if edge_threshold is not None and top_pct is not None:
        raise ValueError("Provide only one of edge_threshold or top_pct.")

    positions = np.asarray(positions)
    matrix = np.asarray(matrix, dtype=float)
    n = len(positions)

    if matrix.shape != (n, n):
        raise ValueError(
            f"matrix shape {matrix.shape} does not match " f"positions length {n}."
        )

    x, y = positions[:, 0], positions[:, 1]

    # Mean value per unit (excluding diagonal)
    mat = matrix.copy()
    np.fill_diagonal(mat, np.nan)
    mean_val = np.nanmean(mat, axis=1)

    # Upper-triangle values for edge selection
    triu_i, triu_j = np.triu_indices(n, k=1)
    vals = mat[triu_i, triu_j]
    valid = np.isfinite(vals)
    triu_i, triu_j, vals = triu_i[valid], triu_j[valid], vals[valid]

    # Determine threshold
    if top_pct is not None:
        threshold = np.percentile(vals, 100 - top_pct)
    else:
        threshold = edge_threshold

    edge_mask = vals >= threshold
    edge_vals = vals[edge_mask]
    edge_i = triu_i[edge_mask]
    edge_j = triu_j[edge_mask]

    # Draw edges with alpha proportional to strength
    alpha_lo, alpha_hi = edge_alpha_range
    if len(edge_vals) > 0:
        e_min, e_max = threshold, np.max(edge_vals)
        if e_max > e_min:
            edge_alpha = alpha_lo + (alpha_hi - alpha_lo) * (edge_vals - e_min) / (
                e_max - e_min
            )
        else:
            edge_alpha = np.full_like(edge_vals, (alpha_lo + alpha_hi) / 2)
        edge_alpha = np.clip(edge_alpha, alpha_lo, alpha_hi)

        for idx in range(len(edge_i)):
            i, j = edge_i[idx], edge_j[idx]
            ax.plot(
                [x[i], x[j]],
                [y[i], y[j]],
                color=edge_color,
                linewidth=edge_linewidth,
                alpha=edge_alpha[idx],
                zorder=2,
            )

    # Draw nodes sized by mean value
    size_min, size_max = node_size_range
    mc_min, mc_max = np.nanmin(mean_val), np.nanmax(mean_val)
    if mc_max > mc_min:
        sizes = size_min + (size_max - size_min) * (mean_val - mc_min) / (
            mc_max - mc_min
        )
    else:
        sizes = np.full_like(mean_val, (size_min + size_max) / 2)

    sc = ax.scatter(
        x,
        y,
        s=sizes,
        c=mean_val,
        cmap=node_cmap,
        edgecolors="face",
        linewidths=node_linewidth,
        zorder=1,
    )

    ax.set_aspect("equal")
    ax.axis("off")

    # Scale bar
    if scale_bar_um:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        bar_x_end = xlim[1] - (xlim[1] - xlim[0]) * 0.02
        bar_x_start = bar_x_end - scale_bar_um
        bar_y = ylim[0] + (ylim[1] - ylim[0]) * 0.02
        ax.plot(
            [bar_x_start, bar_x_end],
            [bar_y, bar_y],
            color="black",
            linewidth=2.0,
            clip_on=False,
            solid_capstyle="butt",
        )
        fs = font_size or 8
        ax.text(
            (bar_x_start + bar_x_end) / 2,
            bar_y - (ylim[1] - ylim[0]) * 0.03,
            f"{scale_bar_um}\u00b5m",
            ha="center",
            va="top",
            fontsize=fs,
        )

    return sc
