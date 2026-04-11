#!/usr/bin/env python
"""Generate QC figures for a completed sorting run.

Writes the three built-in pipeline QC figures (curation bar, std scatter,
all templates), plus quality-metric distributions, a raster + population
rate for the first 30 s, and per-unit ISI histograms + waveform
footprints, into ``<results_folder>/figures/``.

Usage:
    python generate_sorting_figures.py <results_folder>

Requires in the results folder:
- ``sorted_spikedata_curated.pkl`` (required)
- ``sorted_spikedata.pkl`` (optional — used for pre-curation distributions
  when present; otherwise curated units are used and noted in titles)
- ``sorting_*.log`` (optional — used to parse curation thresholds via the
  script path referenced in its header; falls back to library defaults)
"""

from __future__ import annotations

import argparse
import ast
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Defaults matching the library curation defaults
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "fr_min": 0.05,
    "isi_viol_max": 1.0,  # percent
    "snr_min": 5.0,
    "spikes_min_second": 50,
    "std_norm_max": 1.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _locate_log(results_folder: Path) -> Optional[Path]:
    """Return the most recently modified sorting_*.log, or None."""
    logs = sorted(
        results_folder.glob("sorting_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return logs[0] if logs else None


def _script_path_from_log(log_path: Path) -> Optional[Path]:
    """Extract the `Script:` header from the log file."""
    try:
        with open(log_path) as f:
            for line in f:
                m = re.match(r"^Script:\s+(.+?)\s*$", line)
                if m:
                    return Path(m.group(1))
                if line.startswith("========"):
                    # Header section ends at the banner
                    continue
                if len(line.strip()) == 0 and "Running as unit" in line:
                    break
    except OSError:
        return None
    return None


def _parse_script_thresholds(script_path: Path) -> Dict[str, float]:
    """Extract curation threshold kwargs from a sort_recording call via AST.

    Returns only the kwargs that are explicitly set in the script. Unknown
    or non-literal values are skipped with a warning.
    """
    thresholds: Dict[str, float] = {}
    if not script_path.is_file():
        return thresholds
    try:
        tree = ast.parse(script_path.read_text())
    except SyntaxError:
        return thresholds

    target_names = set(DEFAULT_THRESHOLDS.keys()) | {"spikes_min_first"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name in ("sort_recording", "sort_multistream"):
                for kw in node.keywords:
                    if kw.arg in target_names:
                        try:
                            thresholds[kw.arg] = ast.literal_eval(kw.value)
                        except (ValueError, SyntaxError):
                            pass
    return thresholds


def _load_pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _unit_spike_counts(sd) -> np.ndarray:
    return np.array([len(t) for t in sd.train])


def _unit_firing_rates(sd) -> np.ndarray:
    dur_s = sd.length / 1000.0
    return _unit_spike_counts(sd) / dur_s


def _unit_isi_violations(sd, threshold_ms: float = 2.0) -> np.ndarray:
    """Percentage of ISIs shorter than threshold_ms (Hill method 'percent')."""
    out = []
    for t in sd.train:
        if len(t) < 2:
            out.append(0.0)
            continue
        isis = np.diff(t)
        out.append(np.sum(isis < threshold_ms) / len(isis) * 100)
    return np.array(out)


def _unit_attr(sd, key: str, default: float = np.nan) -> np.ndarray:
    return np.array(
        [a.get(key, default) for a in (sd.neuron_attributes or [])],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Figure 1-3: three built-in pipeline figures
# ---------------------------------------------------------------------------


def generate_builtin_figures(sd_curated, thresholds: Dict[str, float], out_dir: Path):
    """Generate curation_bar, std_scatter, and all_templates figures."""
    from spikelab.spike_sorting.figures import (
        plot_curation_bar,
        plot_std_scatter,
        plot_templates,
    )

    rec_name = sd_curated.metadata.get("source_file", "recording")
    rec_name = Path(rec_name).stem

    n_curated = sd_curated.N
    # If raw pkl is not available, we only know curated counts.
    # Use the raw count from metadata or 'rec_chunks' bookkeeping when
    # available; otherwise fall back to curated == total.
    n_total = n_curated
    # The npz file (sorted.npz) sits next to the pkl and has the full
    # (post-curation) list. The caller passes both if available.
    plot_curation_bar(
        [rec_name],
        [n_total],
        [n_curated],
        save_path=str(out_dir / "curation_bar_plot.png"),
    )
    print(f"  [1/3] curation_bar_plot.png")

    # STD scatter — spike count vs. std_norm per unit, with thresholds
    spikes_per_unit = _unit_spike_counts(sd_curated)
    std_norms = _unit_attr(sd_curated, "std_norm")
    n_spikes_dict = {rec_name: dict(enumerate(spikes_per_unit.tolist()))}
    std_norms_dict = {rec_name: dict(enumerate(std_norms.tolist()))}
    plot_std_scatter(
        n_spikes_dict,
        std_norms_dict,
        spikes_thresh=thresholds.get("spikes_min_second"),
        std_thresh=thresholds.get("std_norm_max"),
        save_path=str(out_dir / "std_scatter_plot.png"),
    )
    print(f"  [2/3] std_scatter_plot.png")

    # Templates plot — use template on max channel
    templates = []
    peak_indices = []
    is_curated = []
    has_pos_peak = []
    for a in sd_curated.neuron_attributes:
        templates.append(a["template"])
        peak_indices.append(int(a.get("template_peak_ind", len(a["template"]) // 2)))
        is_curated.append(True)
        has_pos_peak.append(bool(a.get("has_pos_peak", False)))

    fs_Hz = float(sd_curated.metadata.get("fs_Hz", 20000.0))
    plot_templates(
        templates,
        peak_indices,
        fs_Hz,
        is_curated,
        has_pos_peak,
        y_spacing=120.0,
        save_path=str(out_dir / "all_templates_plot.png"),
    )
    print(f"  [3/3] all_templates_plot.png")


# ---------------------------------------------------------------------------
# Figure 4: quality-metric distributions (from pre-curation units if available)
# ---------------------------------------------------------------------------


def generate_quality_distributions(
    sd_source,
    is_pre_curation: bool,
    thresholds: Dict[str, float],
    out_dir: Path,
):
    """4-panel histogram of SNR, firing rate, spike count, ISI violations."""
    snr = _unit_attr(sd_source, "snr")
    frs = _unit_firing_rates(sd_source)
    spikes = _unit_spike_counts(sd_source)
    isi_viols = _unit_isi_violations(sd_source)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    title_suffix = " (pre-curation)" if is_pre_curation else " (curated only)"
    fig.suptitle(f"Unit quality distributions{title_suffix}", fontsize=14)

    # SNR
    ax = axes[0, 0]
    snr_valid = snr[~np.isnan(snr)]
    if snr_valid.size > 0:
        ax.hist(snr_valid, bins=40, color="#4477aa", edgecolor="none")
    if "snr_min" in thresholds:
        ax.axvline(
            thresholds["snr_min"],
            color="red",
            linestyle="--",
            label=f"snr_min={thresholds['snr_min']}",
        )
        ax.legend()
    ax.set_xlabel("SNR")
    ax.set_ylabel("Number of units")
    ax.set_title(f"SNR (n={snr_valid.size})")
    ax.spines[["top", "right"]].set_visible(False)

    # Firing rate
    ax = axes[0, 1]
    ax.hist(frs, bins=40, color="#4477aa", edgecolor="none")
    if "fr_min" in thresholds:
        ax.axvline(
            thresholds["fr_min"],
            color="red",
            linestyle="--",
            label=f"fr_min={thresholds['fr_min']} Hz",
        )
        ax.legend()
    ax.set_xlabel("Firing rate (Hz)")
    ax.set_ylabel("Number of units")
    ax.set_title(f"Firing rate (n={frs.size})")
    ax.spines[["top", "right"]].set_visible(False)

    # Spike count
    ax = axes[1, 0]
    ax.hist(spikes, bins=40, color="#4477aa", edgecolor="none")
    if "spikes_min_second" in thresholds:
        ax.axvline(
            thresholds["spikes_min_second"],
            color="red",
            linestyle="--",
            label=f"spikes_min={int(thresholds['spikes_min_second'])}",
        )
        ax.legend()
    ax.set_xlabel("Spike count")
    ax.set_ylabel("Number of units")
    ax.set_title(f"Spikes per unit (n={spikes.size})")
    ax.spines[["top", "right"]].set_visible(False)

    # ISI violations
    ax = axes[1, 1]
    ax.hist(isi_viols, bins=40, color="#4477aa", edgecolor="none")
    if "isi_viol_max" in thresholds:
        ax.axvline(
            thresholds["isi_viol_max"],
            color="red",
            linestyle="--",
            label=f"isi_viol_max={thresholds['isi_viol_max']}%",
        )
        ax.legend()
    ax.set_xlabel("ISI violations (% spikes with ISI < 2 ms)")
    ax.set_ylabel("Number of units")
    ax.set_title(f"ISI violations (n={isi_viols.size})")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "quality_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"  quality_distributions.png (n={sd_source.N} units, "
        f"{'pre-curation' if is_pre_curation else 'curated'})"
    )


# ---------------------------------------------------------------------------
# Figure 5: raster + population rate for first 30 s
# ---------------------------------------------------------------------------


def generate_raster_overview(sd_curated, out_dir: Path):
    """Raster + population rate plot for the first 30 s of the recording."""
    dur_ms = sd_curated.length
    end_ms = min(30_000.0, dur_ms)
    fig = sd_curated.plot(
        show_raster=True,
        show_pop_rate=True,
        time_range=(0, end_ms),
        show=False,
    )
    fig.savefig(out_dir / "raster_pop_rate_first30s.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  raster_pop_rate_first30s.png (first {end_ms/1000:.1f}s)")


# ---------------------------------------------------------------------------
# Figure 6: per-unit ISI histogram + waveform footprint + single-channel overlay
# ---------------------------------------------------------------------------


def _estimate_channel_spacing(channel_locations: np.ndarray) -> float:
    """Estimate median nearest-neighbour distance from channel positions."""
    if channel_locations.shape[0] < 2:
        return 20.0
    sample = channel_locations[:: max(1, len(channel_locations) // 100)]
    dists = []
    for i in range(len(sample)):
        d = np.linalg.norm(sample - sample[i], axis=1)
        d = d[d > 0]
        if d.size:
            dists.append(d.min())
    return float(np.median(dists)) if dists else 20.0


def generate_per_unit_figures(
    sd,
    out_dir: Path,
    amp_thresh_uv: float = 15.0,
    w_e_raw=None,
    max_overlay_spikes: int = 100,
):
    """For each unit, generate a 3-panel figure:

    1. ISI histogram (0–100 ms)
    2. Waveform footprint: average waveforms at electrode positions for
       channels where |peak| > ``amp_thresh_uv`` µV
    3. Single-channel overlay: individual spike traces (light grey) with
       the average waveform (bold) on the max-amplitude channel

    When ``w_e_raw`` (WaveformExtractor) is provided, individual spike
    waveforms are loaded from it. Otherwise the overlay panel shows only
    the average template.

    Parameters:
        sd: SpikeData with neuron_attributes containing ``template_full``.
        out_dir (Path): Output directory for per-unit PNGs.
        amp_thresh_uv (float): Minimum |peak amplitude| in µV for a
            channel to be included in the footprint plot.
        w_e_raw: WaveformExtractor with ``get_waveforms(unit_id)`` method.
            Optional — when None, the overlay subplot shows only the
            average waveform.
        max_overlay_spikes (int): Maximum number of individual spike
            traces to draw in the overlay subplot (default 100).
    """
    fs_Hz = float(sd.metadata.get("fs_Hz", 20000.0))
    channel_locations = sd.metadata.get("channel_locations")
    if channel_locations is None:
        print("  (skipping per-unit figures: no channel_locations in metadata)")
        return
    channel_locations = np.asarray(channel_locations)

    out_dir.mkdir(parents=True, exist_ok=True)
    t_isi_edges = np.linspace(0, 100, 101)  # 1 ms bins, 0-100 ms
    chan_spacing = _estimate_channel_spacing(channel_locations)

    n_units = sd.N
    print(f"  generating per-unit figures for {n_units} units...")

    for idx in range(n_units):
        attr = sd.neuron_attributes[idx]
        unit_id = attr.get("unit_id", idx)
        spike_times_ms = sd.train[idx]
        template_full = attr.get("template_full")
        if template_full is None:
            continue

        n_spikes = len(spike_times_ms)
        n_samples = template_full.shape[0]
        t_wf_ms = (np.arange(n_samples) - n_samples / 2) / fs_Hz * 1000.0

        # --- ISI histogram (0-100 ms) ---
        if n_spikes > 1:
            isis = np.diff(spike_times_ms)
            isis_plot = isis[isis <= 100.0]
        else:
            isis_plot = np.array([])

        # --- Active channels for footprint ---
        peak_per_chan = np.max(np.abs(template_full), axis=0)
        active_mask = peak_per_chan > amp_thresh_uv
        active_channels = np.where(active_mask)[0]
        if active_channels.size == 0:
            active_channels = np.array([int(np.argmax(peak_per_chan))])

        max_channel = int(np.argmax(peak_per_chan))

        # --- Load individual waveforms if available ---
        individual_wfs = None
        if w_e_raw is not None:
            try:
                all_wfs = w_e_raw.get_waveforms(unit_id)  # (n_spk, n_samples, n_ch)
                if all_wfs.shape[0] > max_overlay_spikes:
                    rng = np.random.default_rng(unit_id)
                    sel = rng.choice(
                        all_wfs.shape[0], max_overlay_spikes, replace=False
                    )
                    individual_wfs = all_wfs[sel, :, max_channel]
                else:
                    individual_wfs = all_wfs[:, :, max_channel]
            except Exception:
                pass

        # --- Figure: 3 subplots ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: ISI histogram
        ax = axes[0]
        if isis_plot.size > 0:
            ax.hist(isis_plot, bins=t_isi_edges, color="#4477aa", edgecolor="none")
        ax.axvline(2.0, color="red", linestyle="--", linewidth=0.8, label="2 ms")
        ax.set_xlim(0, 100)
        ax.set_xlabel("ISI (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"Unit {unit_id} — ISI (n={n_spikes} spikes)")
        ax.legend(loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

        # Panel 2: waveform footprint (average traces only, no dots)
        # Scale so the tallest waveform spans half the electrode spacing,
        # preventing overlap regardless of absolute amplitude.
        ax = axes[1]
        max_amp = peak_per_chan[active_channels].max()
        time_scale = (chan_spacing * 0.4) / (t_wf_ms.max() - t_wf_ms.min())
        amp_scale = (chan_spacing * 0.45) / max_amp if max_amp > 0 else 1.0

        for ch_idx in active_channels:
            cx, cy = channel_locations[ch_idx]
            wf = template_full[:, ch_idx]
            ax.plot(
                cx + t_wf_ms * time_scale,
                cy + wf * amp_scale,
                color="#222222",
                linewidth=0.8,
            )

        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title(
            f"Footprint ({active_channels.size} ch, |peak| > {amp_thresh_uv:g} µV)"
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.spines[["top", "right"]].set_visible(False)

        # Panel 3: single-channel waveform overlay (individual traces + average)
        ax = axes[2]
        if individual_wfs is not None:
            for i in range(individual_wfs.shape[0]):
                ax.plot(t_wf_ms, individual_wfs[i], color="#cccccc", linewidth=0.3)
        avg_wf = template_full[:, max_channel]
        ax.plot(t_wf_ms, avg_wf, color="#cc0000", linewidth=1.5, label="mean")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        n_overlay = individual_wfs.shape[0] if individual_wfs is not None else 0
        ax.set_title(
            f"Ch {max_channel} — {n_spikes} total spikes"
            + (f" ({n_overlay} shown)" if n_overlay > 0 else "")
        )
        ax.legend(loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_dir / f"unit_{unit_id:04d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"  wrote {n_units} per-unit figures to {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate QC figures for a completed sorting run."
    )
    parser.add_argument(
        "results_folder",
        type=Path,
        help="Path to the sorting results folder (contains sorted_spikedata_curated.pkl and sorting_*.log)",
    )
    parser.add_argument(
        "--amp-thresh-uv",
        type=float,
        default=15.0,
        help="Amplitude threshold (µV) for including a channel in the unit footprint plot (default: 15.0)",
    )
    parser.add_argument(
        "--skip-per-unit",
        action="store_true",
        help="Skip the per-unit ISI + footprint figures (slow for many units)",
    )
    args = parser.parse_args(argv)

    results_folder: Path = args.results_folder.resolve()
    if not results_folder.is_dir():
        print(f"ERROR: Not a directory: {results_folder}", file=sys.stderr)
        return 1

    curated_pkl = results_folder / "sorted_spikedata_curated.pkl"
    raw_pkl = results_folder / "sorted_spikedata.pkl"
    if not curated_pkl.is_file():
        print(f"ERROR: Missing {curated_pkl}", file=sys.stderr)
        return 1

    print(f"Results folder: {results_folder}")
    print(f"Loading curated SpikeData from {curated_pkl.name}...")
    sd_curated = _load_pkl(curated_pkl)

    sd_raw = None
    if raw_pkl.is_file():
        print(f"Loading raw SpikeData from {raw_pkl.name}...")
        sd_raw = _load_pkl(raw_pkl)

    # Parse thresholds from the sorting script if we can find it
    thresholds: Dict[str, float] = dict(DEFAULT_THRESHOLDS)
    log_path = _locate_log(results_folder)
    if log_path is not None:
        print(f"Using log file: {log_path.name}")
        script_path = _script_path_from_log(log_path)
        if script_path is not None and script_path.is_file():
            print(f"Parsing thresholds from: {script_path}")
            overrides = _parse_script_thresholds(script_path)
            if overrides:
                thresholds.update(overrides)
                print(f"  Script overrides: {overrides}")
        else:
            print(f"  (no script found; using default thresholds)")
    else:
        print("No sorting_*.log found; using default thresholds.")
    print(f"Thresholds: {thresholds}")

    figures_dir = results_folder / "figures"
    figures_dir.mkdir(exist_ok=True)

    print(f"\nWriting figures to {figures_dir}/")

    print("\n-- Built-in pipeline figures --")
    generate_builtin_figures(sd_curated, thresholds, figures_dir)

    print("\n-- Quality distributions --")
    sd_for_dists = sd_raw if sd_raw is not None else sd_curated
    generate_quality_distributions(
        sd_for_dists,
        is_pre_curation=(sd_raw is not None),
        thresholds=thresholds,
        out_dir=figures_dir,
    )

    print("\n-- Raster + population rate (first 30 s) --")
    generate_raster_overview(sd_curated, figures_dir)

    if not args.skip_per_unit:
        print("\n-- Per-unit ISI + waveform footprint --")
        units_dir = figures_dir / "units"
        generate_per_unit_figures(
            sd_curated, units_dir, amp_thresh_uv=args.amp_thresh_uv
        )
    else:
        print("\n(skipping per-unit figures — --skip-per-unit)")

    print(f"\nDone. Figures saved to {figures_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
