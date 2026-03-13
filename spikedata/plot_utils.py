"""
Plotting utilities for SpikeLab.

Provides ``plot_recording`` for assembling multi-panel figures from SpikeData
objects and ``plot_heatmap`` for standalone 2-D heatmaps.

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
# plot_heatmap
# ---------------------------------------------------------------------------


def plot_heatmap(
    data_mat,
    ax=None,
    norm=False,
    vmin=None,
    vmax=None,
    cmap="hot",
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

    im_kwargs = dict(cmap=cmap, aspect="auto", origin="lower")
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
        burst_times (np.ndarray or None): Burst peak times in ms, plotted as
            markers on the population-rate panel.
        burst_edges (np.ndarray or None): Per-burst ``[start_ms, end_ms]``
            boundaries, shape ``(B, 2)``. Plotted as shaded spans on the
            population-rate panel.
        vmin_heatmap (float or None): Colormap minimum for the FR heatmap.
        vmax_heatmap (float or None): Colormap maximum for the FR heatmap.
        norm_heatmap (bool or str): Row normalisation for the FR heatmap
            (``False`` or ``'row'``).
        model_states_cmap (str): Colormap for the model-states panel.
        model_states_vmin (float): Colormap minimum for model states.
        model_states_vmax (float): Colormap maximum for model states.
        figsize (tuple or None): Figure size ``(width, height)``.
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
          The binned time axis is used, so ``time_range`` should be
          specified in bin units (not ms) when working with GPLVM output.
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
        times_arr = np.arange(0, sd.length, 1.0)
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
        start, end = int(time_range[0]), int(time_range[1])
    else:
        start, end = 0, spk_mat.shape[1]

    spk_mat_view = spk_mat[:, start:end]
    n_samples = end - start

    # Crop arrays whose time axis matches the raster resolution.
    # Arrays with a different time resolution (e.g. GPLVM binned output)
    # are passed through uncropped — their x-axis is linearly scaled to
    # fit the display window.
    raster_T = spk_mat.shape[1]

    def _crop_1d(arr):
        if arr is None:
            return None
        if len(arr) == raster_T:
            return arr[start:end]
        return arr  # different resolution, pass through

    def _crop_2d(arr):
        if arr is None:
            return None
        if arr.shape[-1] == raster_T:
            return arr[:, start:end]
        return arr  # different resolution, pass through

    pop_rate_view = _crop_1d(pop_rate)
    fr_rates_view = _crop_2d(fr_rates)
    cont_prob_view = _crop_1d(cont_prob)
    model_states_view = _crop_2d(model_states)

    # Burst peaks: keep those inside range, shift to window coords
    burst_times_view = None
    if burst_times is not None:
        mask = (burst_times >= start) & (burst_times <= end)
        burst_times_view = (burst_times[mask] - start).astype(int)

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
    axes = []
    for i in range(n_panels):
        ax = fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
    panel_axes = dict(zip(panels, axes))

    # Create colorbar axes (one per row, hidden by default)
    cbar_axes = []
    for i in range(n_panels):
        cax = fig.add_subplot(gs[i, 1])
        cax.axis("off")
        cbar_axes.append(cax)
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
            ax.set_ylabel("Population rate (Hz)")

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
            bt_scaled = (burst_times_view * scale).astype(int)
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
    # Set x limits on the first axes (propagates via sharex)
    axes[0].set_xlim(0, n_samples)

    # Hide tick labels on all but the bottom panel
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("")
    axes[-1].set_xlabel("Time (ms)")
    _apply_font_size(axes[-1], font_size)

    # Shift tick labels to absolute recording time when requested
    if absolute_xticks and start > 0:
        formatter = mticker.FuncFormatter(lambda x, _: f"{int(x + start)}")
        for ax in axes:
            ax.xaxis.set_major_formatter(formatter)

    # ------------------------------------------------------------------
    # 10. Output
    # ------------------------------------------------------------------
    gs.tight_layout(fig)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    elif show:
        plt.show()

    return fig
