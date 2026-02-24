"""Spike functional connectivity metrics.

Implements spike transmission probability (functional connectivity) following
English et al., Neuron 2017 (PMID 29024669). The main function
`compute_spike_transmission` computes a cross-correlogram (CCG) between a
presynaptic and postsynaptic spike train, estimates baseline, extracts a
monosynaptic short-latency peak, computes transmission probability, and
assesses significance with jittered surrogates.

All functions use numpy and are fully type hinted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TransmissionResult:
    """Result of spike transmission computation.

    Attributes:
        transmission_prob: Estimated spike transmission probability
            (excess postsyn spikes per pres spike).
        z_score: Poisson z-score of the monosynaptic peak vs baseline.
        p_value: Surrogate-based p-value (fraction of jitter surrogates with
            peak >= observed). None if surrogates not requested.
        latency_ms: Latency of the peak in milliseconds (center of peak
            window).
        jitter_ms: Standard deviation of postsyn spike times contributing
            to the monosynaptic peak (ms). NaN if not computable.
        observed_peak_count: Number of postsynaptic spikes observed in the
            monosynaptic window summed across all presyn spikes.
        expected_peak_count: Expected count from baseline estimate.
        n_pres_spikes: Number of presynaptic spikes used.
        ccg_counts: Cross-correlogram counts per bin (numpy array).
        ccg_bin_centers_ms: Bin centers in milliseconds (numpy array).
        surrogate_mean: Mean surrogate peak counts (if surrogates used).
        surrogate_std: Std of surrogate peak counts (if surrogates used).
    """

    transmission_prob: float
    z_score: float
    p_value: Optional[float]
    latency_ms: float
    jitter_ms: float
    observed_peak_count: int
    expected_peak_count: float
    n_pres_spikes: int
    ccg_counts: np.ndarray
    ccg_bin_centers_ms: np.ndarray
    surrogate_mean: Optional[float] = None
    surrogate_std: Optional[float] = None


def _hist_relative_times(
    pres: np.ndarray, posts: np.ndarray, window: float, bin_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histogram of posts spike times relative to pres spikes.

    This function collects all inter-spike times within +/- window of each
    pres spike and returns the histogram counts and bin centers.

    Args:
        pres: 1D array of presynaptic spike times (seconds), sorted.
        posts: 1D array of postsynaptic spike times (seconds), sorted.
        window: Half-window (seconds) around each pres spike to consider.
        bin_size: Bin width in seconds.

    Returns:
        counts: Counts per bin (numpy array of ints).
        bin_centers: Bin centers in seconds (numpy array).
    """
    if pres.size == 0 or posts.size == 0:
        edges = np.arange(-window, window + bin_size, bin_size)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return np.zeros(centers.shape, dtype=int), centers

    edges = np.arange(-window, window + bin_size, bin_size)
    centers = (edges[:-1] + edges[1:]) / 2.0

    rels = []  # will hold relative times from all pres spikes
    # iterate over pres spikes but use searchsorted to limit posts considered
    for t in pres:
        left = np.searchsorted(posts, t - window, side="left")
        right = np.searchsorted(posts, t + window, side="right")
        if right > left:
            rel = posts[left:right] - t
            rels.append(rel)

    if not rels:
        return np.zeros(centers.shape, dtype=int), centers

    all_rel = np.concatenate(rels)
    counts, _ = np.histogram(all_rel, bins=edges)
    return counts.astype(int), centers


def compute_spike_transmission(
    pres_spikes: np.ndarray,
    posts_spikes: np.ndarray,
    bin_size: float = 0.0005,
    window: float = 0.05,
    monosyn_window: Tuple[float, float] = (0.0008, 0.0035),
    baseline_exclude: float = 0.010,
    n_jitter: int = 200,
    jitter_window: float = 0.010,
    min_pres_spikes: int = 50,
    random_seed: Optional[int] = None,
) -> TransmissionResult:
    """Compute spike transmission probability between two spike trains.

    The function follows common approaches for extracellular monosynaptic
    connectivity inference: compute the cross-correlogram (CCG), estimate
    baseline from flanking bins, measure excess counts in a short-latency
    monosynaptic window, and compute transmission probability. Optionally
    performs jittered surrogates to obtain a p-value.

    Args:
        pres_spikes: 1D array of presynaptic spike times (seconds).
        posts_spikes: 1D array of postsynaptic spike times (seconds).
        bin_size: Bin width for the CCG in seconds (default 0.5 ms).
        window: Half-window for CCG in seconds (default 50 ms).
        monosyn_window: Tuple (start, end) in seconds defining the
            monosynaptic short-latency window after the pres spike.
            Default roughly 0.8--3.5 ms.
        baseline_exclude: Exclude central +/- baseline_exclude (s) from
            baseline estimate (default 10 ms).
        n_jitter: Number of jitter surrogates to generate (0 disables
            surrogate testing).
        jitter_window: Jitter amplitude (seconds) for surrogates. Each
            pres spike is moved by U(-jitter_window, jitter_window).
        min_pres_spikes: Minimum number of pres spikes to compute metric;
            below this returns zero transmission with NaN statistics.
        random_seed: Optional seed for reproducible surrogates.

    Returns:
        TransmissionResult dataclass with detailed fields.
    """
    pres = np.asarray(pres_spikes, dtype=float)
    posts = np.asarray(posts_spikes, dtype=float)
    pres.sort()
    posts.sort()

    n_pres = pres.size
    if n_pres < min_pres_spikes:
        # return empty/NaN result but still compute CCG for diagnostics
        counts, centers = _hist_relative_times(pres, posts, window, bin_size)
        return TransmissionResult(
            transmission_prob=0.0,
            z_score=float("nan"),
            p_value=None,
            latency_ms=(np.mean(monosyn_window) * 1000.0),
            jitter_ms=float("nan"),
            observed_peak_count=int(0),
            expected_peak_count=float(0.0),
            n_pres_spikes=int(n_pres),
            ccg_counts=counts,
            ccg_bin_centers_ms=centers * 1000.0,
            surrogate_mean=None,
            surrogate_std=None,
        )

    counts, centers = _hist_relative_times(pres, posts, window, bin_size)
    # identify monosyn bins
    mono_start, mono_end = monosyn_window
    mono_mask = (centers >= mono_start) & (centers <= mono_end)
    observed_peak = int(np.sum(counts[mono_mask]))

    # baseline estimate from flanking bins (exclude central +- baseline_exclude)
    baseline_mask = (np.abs(centers) >= baseline_exclude) & (np.abs(centers) <= window)
    if np.any(baseline_mask):
        mean_baseline_per_bin = float(np.mean(counts[baseline_mask]))
    else:
        mean_baseline_per_bin = 0.0

    n_mono_bins = int(np.sum(mono_mask))
    expected_peak = mean_baseline_per_bin * n_mono_bins

    # transmission probability: excess counts per pres spike
    excess = float(observed_peak) - float(expected_peak)
    transmission_prob = max(0.0, excess / float(n_pres))

    # z-score (Poisson approx)
    z_score = float("nan")
    if expected_peak > 0.0:
        z_score = (float(observed_peak) - expected_peak) / float(np.sqrt(expected_peak))

    # latency: take center of monosyn window in ms
    latency_ms = float((mono_start + mono_end) / 2.0 * 1000.0)

    # jitter (std of contributing relative times)
    jitter_ms = float("nan")
    # compute relative times that fall into monosyn window
    rels_in_mono = []
    for t in pres:
        l = np.searchsorted(posts, t + mono_start, side="left")
        r = np.searchsorted(posts, t + mono_end, side="right")
        if r > l:
            rels_in_mono.append(posts[l:r] - t)
    if rels_in_mono:
        all_rel_mono = np.concatenate(rels_in_mono)
        if all_rel_mono.size > 1:
            jitter_ms = float(np.std(all_rel_mono) * 1000.0)
        else:
            jitter_ms = 0.0

    surrogate_mean: Optional[float] = None
    surrogate_std: Optional[float] = None
    p_value: Optional[float] = None

    if n_jitter and n_jitter > 0:
        rng = np.random.default_rng(random_seed)
        surrogate_peaks = np.empty(n_jitter, dtype=float)
        for i in range(n_jitter):
            jittered_pres = pres + rng.uniform(-jitter_window, jitter_window, size=n_pres)
            jittered_pres.sort()
            # compute surrogate peak by reusing histogram function
            scounts, s_centers = _hist_relative_times(jittered_pres, posts, window, bin_size)
            surrogate_peaks[i] = float(np.sum(scounts[(s_centers >= mono_start) & (s_centers <= mono_end)]))

        surrogate_mean = float(np.mean(surrogate_peaks))
        surrogate_std = float(np.std(surrogate_peaks, ddof=1)) if n_jitter > 1 else 0.0
        # p-value: fraction of surrogate peaks >= observed (conservative)
        p_value = float((np.sum(surrogate_peaks >= observed_peak) + 1) / (n_jitter + 1))

    return TransmissionResult(
        transmission_prob=float(transmission_prob),
        z_score=float(z_score),
        p_value=p_value,
        latency_ms=latency_ms,
        jitter_ms=jitter_ms,
        observed_peak_count=int(observed_peak),
        expected_peak_count=float(expected_peak),
        n_pres_spikes=int(n_pres),
        ccg_counts=counts,
        ccg_bin_centers_ms=centers * 1000.0,
        surrogate_mean=surrogate_mean,
        surrogate_std=surrogate_std,
    )


def plot_ccg(result: TransmissionResult, show: bool = True) -> None:
    """Plot a cross-correlogram from a TransmissionResult.

    Matplotlib is optional; if not installed this function will raise an
    informative ImportError.

    Args:
        result: TransmissionResult returned by compute_spike_transmission.
        show: Whether to call plt.show() after plotting (default True).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting not always used
        raise ImportError("matplotlib is required for plotting CCGs") from exc

    centers = result.ccg_bin_centers_ms
    counts = result.ccg_counts

    plt.bar(centers, counts, width=(centers[1] - centers[0]) if centers.size > 1 else 1.0)
    plt.axvspan(result.latency_ms - 0.5, result.latency_ms + 0.5, color="red", alpha=0.2)
    plt.xlabel("Time lag (ms)")
    plt.ylabel("Counts")
    plt.title(f"Spike Transmission (p={result.p_value})\nTP={result.transmission_prob:.4f}, z={result.z_score:.2f}")
    if show:
        plt.show()
