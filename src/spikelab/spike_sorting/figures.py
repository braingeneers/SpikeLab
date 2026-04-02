"""Standalone plotting functions for spike sorting QC figures.

Each function accepts pre-computed data as plain arrays and returns a
``matplotlib.Figure``.  An optional *ax* parameter allows drawing onto
a pre-existing axes instead of creating a new figure.
"""

from math import ceil
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def _import_matplotlib():
    """Lazy import of matplotlib; mirrors spikedata/plot_utils.py."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: "
            "pip install matplotlib"
        ) from e


# ---------------------------------------------------------------------------
# 1. Curation bar plot
# ---------------------------------------------------------------------------


def plot_curation_bar(
    rec_names: Sequence[str],
    n_total: Sequence[int],
    n_selected: Sequence[int],
    *,
    ax=None,
    total_label: str = "First Curation",
    selected_label: str = "Selected Curation",
    x_label: str = "Recording",
    y_label: str = "Number of Units",
    label_rotation: int = 0,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Bar chart comparing total vs. curated unit counts per recording.

    Parameters:
        rec_names (sequence of str): Recording names (one per group).
        n_total (sequence of int): Total unit count per recording.
        n_selected (sequence of int): Selected (curated) unit count per
            recording.
        ax (matplotlib.axes.Axes or None): Pre-existing axes to draw on.
            When None, a new figure is created.
        total_label (str): Legend label for the total-units bar.
        selected_label (str): Legend label for the selected-units bar.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        label_rotation (int): Rotation angle for x-tick labels.
        save_path (str or None): Save figure to this path and close it.
        show (bool): Call ``plt.show()`` when *save_path* is None.

    Returns:
        fig (matplotlib.Figure): The figure containing the bar chart.
    """
    plt = _import_matplotlib()

    external_ax = ax is not None
    if not external_ax:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    x = np.arange(len(rec_names))
    width = 0.35
    ax.bar(x - width / 2, n_total, width, label=total_label)
    ax.bar(x + width / 2, n_selected, width, label=selected_label)
    ax.set_xticks(x)
    ax.set_xticklabels(rec_names, rotation=label_rotation)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="upper right")

    if not external_ax:
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        elif show:
            plt.show()

    return fig


# ---------------------------------------------------------------------------
# 2. STD scatter plot
# ---------------------------------------------------------------------------

_DEFAULT_COLORS = [
    "#f74343",
    "#fccd56",
    "#74fc56",
    "#56fcf6",
    "#1e1efa",
    "#fa1ed2",
]


def plot_std_scatter(
    n_spikes: Dict[str, Dict[str, float]],
    std_norms: Dict[str, Dict[str, float]],
    *,
    ax=None,
    spikes_thresh: Optional[float] = None,
    std_thresh: Optional[float] = None,
    colors: Optional[List[str]] = None,
    alpha: float = 1.0,
    x_label: str = "Number of Spikes",
    y_label: str = "avg. STD / amplitude",
    x_max_buffer: float = 300,
    y_max_buffer: float = 0.2,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Scatter plot of normalised waveform STD vs. spike count per unit.

    Each dot is one unit. Recordings are colour-coded. Optional dashed
    threshold lines show curation cutoffs.

    Parameters:
        n_spikes (dict): ``{rec_name: {unit_id: spike_count}}``.
        std_norms (dict): ``{rec_name: {unit_id: std_norm_value}}``.
        ax (matplotlib.axes.Axes or None): Pre-existing axes to draw on.
            When None, a new figure is created.
        spikes_thresh (float or None): Vertical threshold line for
            minimum spike count.
        std_thresh (float or None): Horizontal threshold line for
            maximum normalised STD.
        colors (list of str or None): Colours for each recording.
            Defaults to a built-in 6-colour palette.
        alpha (float): Marker transparency.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        x_max_buffer (float): Padding added to x-axis maximum.
        y_max_buffer (float): Padding added to y-axis maximum.
        save_path (str or None): Save figure to this path and close it.
        show (bool): Call ``plt.show()`` when *save_path* is None.

    Returns:
        fig (matplotlib.Figure): The figure containing the scatter plot.
    """
    plt = _import_matplotlib()

    external_ax = ax is not None
    if not external_ax:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    palette = list(colors) if colors is not None else list(_DEFAULT_COLORS)
    std_max = -np.inf
    spikes_max = -np.inf

    for rec_name in n_spikes:
        if not palette:
            break
        color = palette.pop(0)
        rec_n = n_spikes[rec_name]
        rec_s = std_norms.get(rec_name, {})

        xs = []
        ys = []
        for uid in rec_n:
            if uid in rec_s:
                xs.append(float(rec_n[uid]))
                ys.append(float(rec_s[uid]))
        if xs:
            ax.scatter(xs, ys, c=color, alpha=alpha, label=rec_name)
            spikes_max = max(spikes_max, max(xs))
            std_max = max(std_max, max(ys))

    x_max = spikes_max + x_max_buffer if np.isfinite(spikes_max) else 1
    y_max = std_max + y_max_buffer if np.isfinite(std_max) else 1

    thresh_kwargs = {"linestyle": "dotted", "linewidth": 1, "c": "#000000"}
    if spikes_thresh is not None:
        ax.axvline(spikes_thresh, **thresh_kwargs)
        ax.text(spikes_thresh, y_max, str(spikes_thresh), ha="center")
    if std_thresh is not None:
        ax.axhline(std_thresh, **thresh_kwargs)
        ax.text(x_max, std_thresh, str(std_thresh), va="center")

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    if len(n_spikes) > 1:
        ax.legend(loc="upper right")

    if not external_ax:
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        elif show:
            plt.show()

    return fig


# ---------------------------------------------------------------------------
# 3. Templates plot
# ---------------------------------------------------------------------------


def plot_templates(
    templates: Sequence[np.ndarray],
    peak_indices: Sequence[int],
    fs_Hz: float,
    is_curated: Sequence[bool],
    has_pos_peak: Sequence[bool],
    *,
    ax=None,
    templates_per_column: int = 50,
    y_spacing: float = 50.0,
    y_lim_buffer: float = 10.0,
    color_curated: str = "#000000",
    color_failed: str = "#FF0000",
    window_ms_before: float = 5.0,
    window_ms_after: float = 5.0,
    line_ms_before: Optional[float] = 1.0,
    line_ms_after: Optional[float] = 4.0,
    x_label: str = "Time Rel. to Peak (ms)",
    sort_by_amplitude: bool = False,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Stacked waveform template overview for sorted units.

    Plots the mean waveform template of every unit in vertically stacked
    columns, split by polarity (negative-peak vs. positive-peak).
    Curated units are drawn in *color_curated*, failed units in
    *color_failed*.

    Parameters:
        templates (sequence of np.ndarray): 1-D waveform arrays, one per
            unit (e.g. the template on the max-amplitude channel).
        peak_indices (sequence of int): Sample index of the peak in each
            template.
        fs_Hz (float): Sampling frequency in Hz.
        is_curated (sequence of bool): Whether each unit passed curation.
        has_pos_peak (sequence of bool): Whether each unit has a positive
            peak (used to split into polarity groups).
        ax (matplotlib.axes.Axes or None): When provided, all templates
            are drawn on this single axes (column splitting is skipped).
        templates_per_column (int): Max templates per subplot column
            (ignored when *ax* is provided).
        y_spacing (float): Vertical offset between stacked templates.
        y_lim_buffer (float): Padding above/below data limits.
        color_curated (str): Colour for curated units.
        color_failed (str): Colour for failed units.
        window_ms_before (float): Display window before peak in ms.
        window_ms_after (float): Display window after peak in ms.
        line_ms_before (float or None): Vertical reference line before
            peak (ms).  None to skip.
        line_ms_after (float or None): Vertical reference line after
            peak (ms).  None to skip.
        x_label (str): X-axis label.
        sort_by_amplitude (bool): Sort units within each polarity group
            by peak amplitude (descending).
        save_path (str or None): Save figure to this path and close it.
        show (bool): Call ``plt.show()`` when *save_path* is None.

    Returns:
        fig (matplotlib.Figure): The figure containing the templates.
    """
    plt = _import_matplotlib()

    # Split into polarity groups
    neg_units = []
    pos_units = []
    for i, pos in enumerate(has_pos_peak):
        entry = (templates[i], peak_indices[i], is_curated[i])
        if pos:
            pos_units.append(entry)
        else:
            neg_units.append(entry)

    if sort_by_amplitude:
        neg_units.sort(key=lambda e: float(np.max(np.abs(e[0]))), reverse=True)
        pos_units.sort(key=lambda e: float(np.max(np.abs(e[0]))), reverse=True)

    external_ax = ax is not None

    if external_ax:
        # Draw everything on the single provided axes
        fig = ax.figure
        all_axes = [ax]
        _draw_templates_on_axes(
            all_axes,
            neg_units + pos_units,
            fs_Hz,
            y_spacing,
            color_curated,
            color_failed,
            templates_per_column=len(neg_units) + len(pos_units),
        )
    else:
        n_col_neg = (
            max(1, ceil(len(neg_units) / templates_per_column)) if neg_units else 0
        )
        n_col_pos = (
            max(1, ceil(len(pos_units) / templates_per_column)) if pos_units else 0
        )
        n_cols = n_col_neg + n_col_pos
        if n_cols == 0:
            n_cols = 1

        fig, axs = plt.subplots(
            1,
            n_cols,
            figsize=(n_cols * 3, templates_per_column / 6),
            tight_layout=True,
        )
        axs = np.atleast_1d(axs)

        # Draw negative-peak columns
        neg_axes = list(axs[:n_col_neg])
        pos_axes = list(axs[n_col_neg:])
        neg_ylims = _draw_templates_on_axes(
            neg_axes,
            neg_units,
            fs_Hz,
            y_spacing,
            color_curated,
            color_failed,
            templates_per_column,
        )
        pos_ylims = _draw_templates_on_axes(
            pos_axes,
            pos_units,
            fs_Hz,
            y_spacing,
            color_curated,
            color_failed,
            templates_per_column,
        )

        # Apply consistent y-limits per polarity group
        for a in neg_axes:
            if neg_ylims[0] < neg_ylims[1]:
                a.set_ylim(neg_ylims[0] - y_lim_buffer, neg_ylims[1] + y_lim_buffer)
        for a in pos_axes:
            if pos_ylims[0] < pos_ylims[1]:
                a.set_ylim(pos_ylims[0] - y_lim_buffer, pos_ylims[1] + y_lim_buffer)

        all_axes = list(axs)

    # Formatting
    window = [-window_ms_before, window_ms_after]
    line_kwargs = {"color": "black", "linestyle": "dotted"}
    for a in all_axes:
        a.set_xlim(*window)
        a.set_xticks(window + [0])
        a.set_xlabel(x_label)
        a.set_yticks([])
        if line_ms_before is not None:
            a.axvline(-line_ms_before, **line_kwargs)
        if line_ms_after is not None:
            a.axvline(line_ms_after, **line_kwargs)

    if not external_ax:
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        elif show:
            plt.show()

    return fig


def _draw_templates_on_axes(
    axes, units, fs_Hz, y_spacing, color_curated, color_failed, templates_per_column
):
    """Draw stacked templates onto one or more axes columns.

    Returns (y_min, y_max) across all drawn templates.
    """
    y_min = np.inf
    y_max = -np.inf
    y_offset = 0.0
    count = 0
    ax_idx = 0

    if not axes or not units:
        return (0.0, 0.0)

    for template, peak_ind, curated in units:
        shifted = template - y_offset
        x_ms = (np.arange(len(template)) - peak_ind) / fs_Hz * 1000.0
        color = color_curated if curated else color_failed
        axes[ax_idx].plot(x_ms, shifted, color=color)

        y_min = min(y_min, float(np.min(shifted)))
        y_max = max(y_max, float(np.max(shifted)))

        y_offset += y_spacing
        count += 1
        if count >= templates_per_column and ax_idx < len(axes) - 1:
            ax_idx += 1
            y_offset = 0.0
            count = 0

    return (y_min, y_max)
