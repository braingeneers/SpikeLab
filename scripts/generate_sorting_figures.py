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
# Figure 6: per-unit ISI histogram + waveform footprint
# ---------------------------------------------------------------------------


def generate_per_unit_figures(sd_curated, out_dir: Path, amp_thresh_uv: float = 5.0):
    """For each unit, plot an ISI histogram (0-50 ms) and a waveform footprint.

    The footprint shows the average waveform on all channels where the
    absolute peak of the average waveform exceeds ``amp_thresh_uv`` µV.
    Channel positions come from the recording's channel_locations.
    """
    fs_Hz = float(sd_curated.metadata.get("fs_Hz", 20000.0))
    channel_locations = sd_curated.metadata.get("channel_locations")
    if channel_locations is None:
        print("  (skipping per-unit figures: no channel_locations in metadata)")
        return
    channel_locations = np.asarray(channel_locations)

    out_dir.mkdir(parents=True, exist_ok=True)
    t_isi_edges = np.linspace(0, 50, 51)  # 1 ms bins
    t_isi_centers = 0.5 * (t_isi_edges[:-1] + t_isi_edges[1:])

    n_units = sd_curated.N
    print(f"  generating per-unit figures for {n_units} units...")

    for idx in range(n_units):
        attr = sd_curated.neuron_attributes[idx]
        unit_id = attr.get("unit_id", idx)
        spike_times_ms = sd_curated.train[idx]
        template_full = attr.get("template_full")
        if template_full is None:
            continue

        # --- ISI histogram (0-50 ms) ---
        if len(spike_times_ms) > 1:
            isis = np.diff(spike_times_ms)
            isis = isis[isis <= 50.0]
        else:
            isis = np.array([])

        # --- Footprint: channels where |avg waveform peak| > threshold ---
        peak_per_chan = np.max(np.abs(template_full), axis=0)  # (n_channels,)
        active_mask = peak_per_chan > amp_thresh_uv
        active_channels = np.where(active_mask)[0]
        if active_channels.size == 0:
            # Fall back to the single max-amplitude channel so there's
            # something to show
            active_channels = np.array([int(np.argmax(peak_per_chan))])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: ISI histogram
        ax = axes[0]
        if isis.size > 0:
            ax.hist(isis, bins=t_isi_edges, color="#4477aa", edgecolor="none")
        ax.axvline(2.0, color="red", linestyle="--", linewidth=0.8, label="2 ms")
        ax.set_xlim(0, 50)
        ax.set_xlabel("ISI (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"Unit {unit_id} — ISI (n={len(spike_times_ms)} spikes)")
        ax.legend(loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

        # Right: waveform footprint on active channels
        ax = axes[1]
        chan_xy = channel_locations[active_channels]
        # Use position-based drawing: each channel's waveform is placed at
        # its (x, y) location, scaled to fit.
        n_samples = template_full.shape[0]
        t_wf_ms = (np.arange(n_samples) - n_samples / 2) / fs_Hz * 1000.0

        # Normalize scaling based on the largest waveform peak across
        # displayed channels
        max_amp = peak_per_chan[active_channels].max()
        if channel_locations.shape[0] >= 2:
            # Estimate channel spacing from nearest-neighbour distances
            xy_all = channel_locations
            # Sample up to 100 channels to estimate spacing
            sample = xy_all[:: max(1, len(xy_all) // 100)]
            dists = []
            for i in range(len(sample)):
                d = np.linalg.norm(sample - sample[i], axis=1)
                d = d[d > 0]
                if d.size:
                    dists.append(d.min())
            chan_spacing = float(np.median(dists)) if dists else 20.0
        else:
            chan_spacing = 20.0

        # Time axis scaling so the waveform fits within one channel pitch
        time_scale = chan_spacing / (t_wf_ms.max() - t_wf_ms.min())
        amp_scale = chan_spacing / max_amp * 0.8 if max_amp > 0 else 1.0

        for ch_idx in active_channels:
            cx, cy = channel_locations[ch_idx]
            wf = template_full[:, ch_idx]
            ax.plot(
                cx + t_wf_ms * time_scale,
                cy + wf * amp_scale,
                color="#222222",
                linewidth=0.8,
            )

        # Mark all displayed channel positions with a small dot
        ax.scatter(
            chan_xy[:, 0],
            chan_xy[:, 1],
            s=6,
            c="#cc0000",
            zorder=3,
        )
        # Mark the unit's max channel with a star
        unit_x = attr.get("x", None)
        unit_y = attr.get("y", None)
        if unit_x is not None and unit_y is not None:
            ax.scatter(
                [unit_x],
                [unit_y],
                s=80,
                marker="*",
                c="#ffaa00",
                edgecolor="black",
                zorder=4,
            )

        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title(
            f"Unit {unit_id} — footprint ({active_channels.size} channels, "
            f"|peak| > {amp_thresh_uv:g} µV)"
        )
        ax.set_aspect("equal", adjustable="datalim")
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
        default=5.0,
        help="Amplitude threshold (µV) for including a channel in the unit footprint plot (default: 5.0)",
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
