"""HIPPIE neuron classification adapter for SpikeData.

Requires: pip install spikelab[hippie]

Workflow:
    1. extract_features(sd)  — compute wave/ISI/ACG arrays from a SpikeData object
    2. classify_neurons(sd)  — one-call pipeline: embed → UMAP → HDBSCAN
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np


# ISI histogram parameters (log-spaced, matching HIPPIE training data convention)
_ISI_N_BINS = 100
_ISI_MIN_MS = 1.0
_ISI_MAX_MS = 5000.0

# ACG parameters
_ACG_N_BINS = 100
_ACG_MAX_LAG_MS = 100.0


def _require_hippie():
    try:
        import hippie  # noqa: F401
    except ImportError:
        raise ImportError(
            "HIPPIE is required for neuron classification. "
            "Install it with: pip install spikelab[hippie]"
        )


# ------------------------------------------------------------------
# Per-neuron preprocessing helpers
# ------------------------------------------------------------------

def _preprocess_waveform(wave: np.ndarray, target: int = 50) -> np.ndarray:
    """Resample waveform to target length and min-max normalize to [-1, 1]."""
    import torch
    import torch.nn.functional as F

    t = torch.as_tensor(wave, dtype=torch.float32).view(1, 1, -1)
    t = F.interpolate(t, size=(target,), mode="linear", align_corners=False).squeeze()
    mn, mx = t.min().item(), t.max().item()
    if mx > mn:
        t = (t - mn) / (mx - mn) * 2.0 - 1.0
    return t.numpy().astype(np.float32)


def _isi_histogram(spike_times: np.ndarray, n_bins: int = _ISI_N_BINS) -> np.ndarray:
    """Compute a log-spaced ISI histogram, log(x+1)-transformed and min-max normalized."""
    isis_ms = np.diff(np.sort(spike_times)) * 1000.0
    isis_ms = isis_ms[isis_ms > 0]
    if len(isis_ms) < 2:
        return np.full(n_bins, -1.0, dtype=np.float32)  # return flat -1 for silent neurons

    bins = np.logspace(np.log10(_ISI_MIN_MS), np.log10(_ISI_MAX_MS), n_bins + 1)
    hist, _ = np.histogram(isis_ms, bins=bins, density=True)
    hist = hist.astype(np.float32)

    # log(x+1) transform then min-max to [-1, 1] — matches MultiModalEphysDataset
    hist = np.log1p(hist)
    mn, mx = hist.min(), hist.max()
    if mx > mn:
        hist = (hist - mn) / (mx - mn + 1e-8) * 2.0 - 1.0
    return hist


def _autocorrelogram(
    spike_times: np.ndarray,
    max_lag_ms: float = _ACG_MAX_LAG_MS,
    n_bins: int = _ACG_N_BINS,
) -> np.ndarray:
    """Compute a half-sided autocorrelogram (forward lags only), min-max normalized."""
    if len(spike_times) < 2:
        return np.zeros(n_bins, dtype=np.float32)

    st_ms = np.sort(spike_times) * 1000.0
    bin_edges = np.linspace(0.0, max_lag_ms, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.float64)

    for i in range(len(st_ms)):
        hi = np.searchsorted(st_ms, st_ms[i] + max_lag_ms, side="right")
        lo = i + 1
        if lo < hi:
            diffs = st_ms[lo:hi] - st_ms[i]
            counts += np.histogram(diffs, bins=bin_edges)[0]

    total = counts.sum()
    if total > 0:
        counts /= total

    acg = counts.astype(np.float32)
    mn, mx = acg.min(), acg.max()
    if mx > mn:
        acg = (acg - mn) / (mx - mn + 1e-8) * 2.0 - 1.0
    return acg


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def extract_features(
    sd,
    isi_bins: int = _ISI_N_BINS,
    acg_bins: int = _ACG_N_BINS,
    acg_max_lag_ms: float = _ACG_MAX_LAG_MS,
) -> dict:
    """Extract waveform, ISI, and ACG features from a SpikeData object.

    Waveforms are read from ``neuron_attributes["avg_waveform"]``.  Call
    ``sd.get_waveform_traces()`` first if raw_data is available and
    avg_waveform has not yet been computed.

    Args:
        sd: SpikeData instance with spike trains and avg_waveform attributes.
        isi_bins: Number of log-spaced ISI histogram bins.
        acg_bins: Number of autocorrelogram bins.
        acg_max_lag_ms: Maximum lag for the autocorrelogram (milliseconds).

    Returns:
        dict with keys:
            - "wave": (N, 50)  min-max normalized waveforms
            - "isi":  (N, 100) log-transformed, normalized ISI histograms
            - "acg":  (N, 100) normalized autocorrelograms
    """
    waves = sd.get_neuron_attribute("avg_waveform")
    if waves is None or any(w is None for w in waves):
        raise ValueError(
            "avg_waveform not found in neuron_attributes. "
            "Call sd.get_waveform_traces() first, or set avg_waveform manually."
        )

    wave_arr = np.stack([_preprocess_waveform(np.asarray(w)) for w in waves])
    isi_arr = np.stack([_isi_histogram(t, n_bins=isi_bins) for t in sd.train])
    acg_arr = np.stack([
        _autocorrelogram(t, max_lag_ms=acg_max_lag_ms, n_bins=acg_bins)
        for t in sd.train
    ])

    return {"wave": wave_arr, "isi": isi_arr, "acg": acg_arr}


def classify_neurons(
    sd,
    repo_id: str = "Jesusgf23/hippie",
    tech_id: Union[int, str] = 0,
    device: str = "cpu",
    run_umap: bool = True,
    run_hdbscan: bool = True,
    umap_kwargs: Optional[dict] = None,
    hdbscan_kwargs: Optional[dict] = None,
    batch_size: int = 256,
    cache_dir: Optional[str] = None,
) -> dict:
    """Classify neurons in a SpikeData object using HIPPIE.

    Downloads the pretrained HIPPIE checkpoint, encodes all neurons into
    the latent space, and optionally runs UMAP dimensionality reduction
    followed by HDBSCAN clustering.

    Args:
        sd: SpikeData with spike trains and avg_waveform in neuron_attributes.
        repo_id: HuggingFace repository ID for the HIPPIE checkpoint.
        tech_id: Recording technology — int index or one of:
                 "neuropixels" (0), "silicon_probe" (1),
                 "juxtacellular" (2), "tetrodes" (3).
        device: "cuda" or "cpu".
        run_umap: Compute 2-D UMAP projection of the embeddings.
        run_hdbscan: Cluster with HDBSCAN (applied on UMAP coords when
                     run_umap=True, otherwise on raw embeddings).
        umap_kwargs: Extra keyword arguments for HIPPIEClassifier.umap_reduce().
        hdbscan_kwargs: Extra keyword arguments for HIPPIEClassifier.hdbscan_cluster().
        batch_size: Neurons per forward pass.
        cache_dir: Local directory to cache the downloaded checkpoint.

    Returns:
        dict with keys:
            - "embeddings":    (N, 30) latent z_mean vectors
            - "umap_coords":   (N, 2)  UMAP coordinates  (present if run_umap=True)
            - "cluster_labels":(N,)    HDBSCAN labels, -1=noise (present if run_hdbscan=True)

    Example:
        >>> from spikelab.spikedata.hippie_adapter import classify_neurons
        >>> result = classify_neurons(sd, tech_id="neuropixels")
        >>> sd.set_neuron_attribute("hippie_cluster", result["cluster_labels"])
        >>> sd.set_neuron_attribute("hippie_embedding", result["embeddings"])
    """
    _require_hippie()
    from hippie.inference import HIPPIEClassifier

    features = extract_features(sd)

    clf = HIPPIEClassifier.from_pretrained(
        repo_id=repo_id, device=device, cache_dir=cache_dir
    )
    embeddings = clf.get_embeddings(
        features["wave"],
        features["isi"],
        features["acg"],
        tech_id=tech_id,
        batch_size=batch_size,
    )

    result: dict = {"embeddings": embeddings}

    if run_umap:
        result["umap_coords"] = clf.umap_reduce(embeddings, **(umap_kwargs or {}))

    if run_hdbscan:
        cluster_input = result.get("umap_coords", embeddings)
        result["cluster_labels"] = clf.hdbscan_cluster(
            cluster_input, **(hdbscan_kwargs or {})
        )

    return result
