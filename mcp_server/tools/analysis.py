"""
MCP tools for analyzing spike data.

All tools are workspace-centric: inputs are loaded from an AnalysisWorkspace
and all outputs are stored back to the workspace. No bulk data is returned
inline to the agent.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from mcp_server.sessions import get_session_manager
from spikedata.pairwise import PairwiseCompMatrixStack
from spikedata.ratedata import RateData
from spikedata.rateslicestack import RateSliceStack
from spikedata.spikeslicestack import SpikeSliceStack
from spikedata.spikedata import SpikeData
from spikedata.utils import (
    PCA_reduction,
    UMAP_graph_communities,
    UMAP_reduction,
    compute_cosine_similarity_with_lag,
    compute_cross_correlation_with_lag,
    extract_lower_triangle_features as _extract_lower_triangle,
)
from workspace.workspace import get_workspace_manager

_COMPARE_FUNCS = {
    "cross_correlation": compute_cross_correlation_with_lag,
    "cosine_similarity": compute_cosine_similarity_with_lag,
}

_SPIKEDATA_KEY = "spikedata"


def _to_list(arr):
    """Convert a numpy array to a nested Python list for JSON serialization."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def _get_workspace(workspace_id: str):
    """Get AnalysisWorkspace by ID, raising ValueError if not found."""
    ws = get_workspace_manager().get_workspace(workspace_id)
    if ws is None:
        raise ValueError(f"Workspace not found: {workspace_id}")
    return ws


def _get_spikedata(ws, namespace: str) -> SpikeData:
    """
    Load SpikeData from (namespace, 'spikedata') in the workspace.

    Raises ValueError with tool suggestions if not found.
    """
    sd = ws.get(namespace, _SPIKEDATA_KEY)
    if sd is None or not isinstance(sd, SpikeData):
        raise ValueError(
            f"No SpikeData found at ({namespace!r}, {_SPIKEDATA_KEY!r}). "
            "Load a recording first using one of: "
            "load_from_hdf5, load_from_nwb, load_from_kilosort, "
            "load_from_pickle, load_from_hdf5_thresholded."
        )
    return sd


def _get_ratedata(ws, namespace: str, key: str) -> RateData:
    """
    Load RateData from (namespace, key) in the workspace.

    Raises ValueError with tool suggestions if not found.
    """
    rd = ws.get(namespace, key)
    if rd is None or not isinstance(rd, RateData):
        raise ValueError(
            f"No RateData found at ({namespace!r}, {key!r}). "
            "Compute instantaneous firing rates first using: compute_resampled_isi."
        )
    return rd


def _get_rateslicestack(ws, namespace: str, key: str) -> RateSliceStack:
    """
    Load RateSliceStack from (namespace, key) in the workspace.

    Raises ValueError with tool suggestions if not found.
    """
    rss = ws.get(namespace, key)
    if rss is None or not isinstance(rss, RateSliceStack):
        raise ValueError(
            f"No RateSliceStack found at ({namespace!r}, {key!r}). "
            "Build event-aligned rate slices first using: "
            "create_rate_slice_stack or frames_rate_data."
        )
    return rss


def _get_spikeslicestack(ws, namespace: str, key: str) -> SpikeSliceStack:
    """
    Load SpikeSliceStack from (namespace, key) in the workspace.

    Raises ValueError with tool suggestions if not found.
    """
    sss = ws.get(namespace, key)
    if sss is None or not isinstance(sss, SpikeSliceStack):
        raise ValueError(
            f"No SpikeSliceStack found at ({namespace!r}, {key!r}). "
            "Build spike slices first using: "
            "frames_spike_data or create_spike_slice_stack."
        )
    return sss


def _pad_ragged(arrays) -> np.ndarray:
    """Pad a list of 1-D arrays to the same length with NaN, returning (N, max_len)."""
    max_len = max((len(a) for a in arrays), default=0)
    result = np.full((len(arrays), max_len), np.nan, dtype=np.float64)
    for i, a in enumerate(arrays):
        result[i, : len(a)] = a
    return result


# ---------------------------------------------------------------------------
# Basic analysis — SpikeData → ndarray stored in workspace
# ---------------------------------------------------------------------------


async def compute_rates(
    workspace_id: str,
    namespace: str,
    key: str,
    unit: str = "kHz",
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    rates = sd.rates(unit=unit)
    ws.store(namespace, key, rates)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "unit": unit,
        "info": ws.get_info(namespace, key),
    }


async def compute_binned(
    workspace_id: str,
    namespace: str,
    key: str,
    bin_size: float = 40.0,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    binned = sd.binned(bin_size=bin_size)
    ws.store(namespace, key, binned)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "bin_size": bin_size,
        "info": ws.get_info(namespace, key),
    }


async def compute_binned_meanrate(
    workspace_id: str,
    namespace: str,
    key: str,
    bin_size: float = 40.0,
    unit: str = "kHz",
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    meanrate = sd.binned_meanrate(bin_size=bin_size, unit=unit)
    ws.store(namespace, key, meanrate)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "bin_size": bin_size,
        "unit": unit,
        "info": ws.get_info(namespace, key),
    }


async def compute_raster(
    workspace_id: str,
    namespace: str,
    key: str,
    bin_size: float = 20.0,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    raster = sd.raster(bin_size=bin_size)
    ws.store(namespace, key, raster)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "bin_size": bin_size,
        "info": ws.get_info(namespace, key),
    }


async def compute_sparse_raster(
    workspace_id: str,
    namespace: str,
    key: str,
    bin_size: float = 20.0,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    raster = sd.sparse_raster(bin_size=bin_size).toarray()
    ws.store(namespace, key, raster)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "bin_size": bin_size,
        "info": ws.get_info(namespace, key),
    }


async def compute_channel_raster(
    workspace_id: str,
    namespace: str,
    key: str,
    bin_size: float = 20.0,
    channel_attr: Optional[str] = None,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    raster = sd.channel_raster(bin_size=bin_size, channel_attr=channel_attr)
    ws.store(namespace, key, raster)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "bin_size": bin_size,
        "info": ws.get_info(namespace, key),
    }


async def compute_interspike_intervals(
    workspace_id: str,
    namespace: str,
    key: str,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    isis = sd.interspike_intervals()
    arr = _pad_ragged(isis)
    ws.store(namespace, key, arr)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "info": ws.get_info(namespace, key),
        "note": "NaN-padded (U, max_isi_count) array; rows = units",
    }


async def compute_resampled_isi(
    workspace_id: str,
    namespace: str,
    key: str,
    times: List[float],
    sigma_ms: float = 10.0,
) -> Dict[str, Any]:
    """
    Compute instantaneous firing rates via the resampled ISI method and store
    the result as a RateData object in the workspace.
    """
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    rate_matrix = sd.resampled_isi(times=np.array(times), sigma_ms=sigma_ms)
    rd = RateData(rate_matrix, np.array(times))
    ws.store(namespace, key, rd)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "sigma_ms": sigma_ms,
        "n_timepoints": len(times),
        "info": ws.get_info(namespace, key),
    }


async def compute_spike_time_tiling(
    workspace_id: str,
    namespace: str,
    key: str,
    neuron_i: int,
    neuron_j: int,
    delt: float = 20.0,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    sttc = sd.spike_time_tiling(neuron_i, neuron_j, delt=delt)
    ws.store(namespace, key, np.array([sttc]))
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "neuron_i": neuron_i,
        "neuron_j": neuron_j,
        "delt": delt,
        "info": ws.get_info(namespace, key),
    }


async def compute_spike_time_tilings(
    workspace_id: str,
    namespace: str,
    key: str,
    delt: float = 20.0,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    pcm = sd.spike_time_tilings(delt=delt)
    ws.store(namespace, key, pcm.matrix)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "delt": delt,
        "info": ws.get_info(namespace, key),
    }


async def threshold_spike_time_tilings(
    workspace_id: str,
    namespace: str,
    key: str,
    threshold: float,
    delt: float = 20.0,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    pcm = sd.spike_time_tilings(delt=delt)
    binary_pcm = pcm.threshold(threshold)
    ws.store(namespace, key, binary_pcm.matrix)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "threshold": threshold,
        "delt": delt,
        "info": ws.get_info(namespace, key),
    }


async def compute_latencies(
    workspace_id: str,
    namespace: str,
    key: str,
    times: List[float],
    window_ms: float = 100.0,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    latencies = sd.latencies(times, window_ms=window_ms)
    arr = _pad_ragged(latencies)
    ws.store(namespace, key, arr)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "window_ms": window_ms,
        "info": ws.get_info(namespace, key),
        "note": "NaN-padded (U, max_latency_count) array; rows = units",
    }


async def compute_latencies_to_index(
    workspace_id: str,
    namespace: str,
    key: str,
    neuron_index: int,
    window_ms: float = 100.0,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    latencies = sd.latencies_to_index(neuron_index, window_ms=window_ms)
    arr = _pad_ragged(latencies)
    ws.store(namespace, key, arr)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "neuron_index": neuron_index,
        "window_ms": window_ms,
        "info": ws.get_info(namespace, key),
        "note": "NaN-padded (U, max_latency_count) array; rows = units",
    }


async def get_pop_rate(
    workspace_id: str,
    namespace: str,
    key: str,
    square_width: int = 20,
    gauss_sigma: int = 100,
    raster_bin_size_ms: float = 1.0,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    pop_rate = sd.get_pop_rate(
        square_width=square_width,
        gauss_sigma=gauss_sigma,
        raster_bin_size_ms=raster_bin_size_ms,
    )
    ws.store(namespace, key, pop_rate)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "raster_bin_size_ms": raster_bin_size_ms,
        "info": ws.get_info(namespace, key),
    }


async def compute_spike_trig_pop_rate(
    workspace_id: str,
    namespace: str,
    key: str,
    key_lags: str,
    key_coupling: str,
    window_ms: int = 80,
    cutoff_hz: float = 20,
    fs: float = 1000,
    bin_size: float = 1,
    cut_outer: int = 10,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    stPR_filtered, coupling_zero_lag, coupling_max, delays, lags = (
        sd.compute_spike_trig_pop_rate(
            window_ms=window_ms,
            cutoff_hz=cutoff_hz,
            fs=fs,
            bin_size=bin_size,
            cut_outer=cut_outer,
        )
    )
    # Store stPR (U, T) and lags (T,) separately; combine coupling stats as (3, U)
    coupling_stack = np.stack(
        [
            np.asarray(coupling_zero_lag, dtype=np.float64),
            np.asarray(coupling_max, dtype=np.float64),
            np.asarray(delays, dtype=np.float64),
        ],
        axis=0,
    )
    ws.store(namespace, key, np.asarray(stPR_filtered, dtype=np.float64))
    ws.store(namespace, key_lags, np.asarray(lags, dtype=np.float64))
    ws.store(namespace, key_coupling, coupling_stack)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "key_lags": key_lags,
        "key_coupling": key_coupling,
        "info": ws.get_info(namespace, key),
        "info_lags": ws.get_info(namespace, key_lags),
        "info_coupling": ws.get_info(namespace, key_coupling),
        "note": (
            f"key_coupling is (3, U): row 0 = coupling_zero_lag, "
            "row 1 = coupling_max, row 2 = delays"
        ),
    }


async def get_bursts(
    workspace_id: str,
    namespace: str,
    key_tburst: str,
    key_edges: str,
    key_amp: str,
    thr_burst: float,
    min_burst_diff: int,
    burst_edge_mult_thresh: float,
    square_width: int = 20,
    gauss_sigma: int = 100,
    acc_square_width: int = 5,
    acc_gauss_sigma: int = 5,
    raster_bin_size_ms: float = 1.0,
    peak_to_trough: bool = True,
    pop_rms_override: Optional[float] = None,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    tburst, edges, peak_amp = sd.get_bursts(
        thr_burst,
        min_burst_diff,
        burst_edge_mult_thresh,
        square_width=square_width,
        gauss_sigma=gauss_sigma,
        acc_square_width=acc_square_width,
        acc_gauss_sigma=acc_gauss_sigma,
        raster_bin_size_ms=raster_bin_size_ms,
        peak_to_trough=peak_to_trough,
        pop_rms_override=pop_rms_override,
    )
    ws.store(namespace, key_tburst, np.asarray(tburst, dtype=np.float64))
    ws.store(namespace, key_edges, np.asarray(edges, dtype=np.float64))
    ws.store(namespace, key_amp, np.asarray(peak_amp, dtype=np.float64))
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key_tburst": key_tburst,
        "key_edges": key_edges,
        "key_amp": key_amp,
        "n_bursts": int(len(tburst)),
        "info_tburst": ws.get_info(namespace, key_tburst),
        "info_edges": ws.get_info(namespace, key_edges),
        "info_amp": ws.get_info(namespace, key_amp),
    }


async def get_frac_active(
    workspace_id: str,
    namespace: str,
    edges_key: str,
    key_frac_unit: str,
    key_frac_burst: str,
    key_backbone: str,
    min_spikes: int,
    backbone_threshold: float,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    edges_obj = ws.get(namespace, edges_key)
    if edges_obj is None or not isinstance(edges_obj, np.ndarray):
        raise ValueError(
            f"No edges array found at ({namespace!r}, {edges_key!r}). "
            "Run get_bursts first to compute burst edges."
        )
    frac_per_unit, frac_per_burst, backbone_units = sd.get_frac_active(
        edges_obj, min_spikes, backbone_threshold
    )
    ws.store(namespace, key_frac_unit, np.asarray(frac_per_unit, dtype=np.float64))
    ws.store(namespace, key_frac_burst, np.asarray(frac_per_burst, dtype=np.float64))
    ws.store(namespace, key_backbone, np.asarray(backbone_units, dtype=np.float64))
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key_frac_unit": key_frac_unit,
        "key_frac_burst": key_frac_burst,
        "key_backbone": key_backbone,
        "info_frac_unit": ws.get_info(namespace, key_frac_unit),
        "info_frac_burst": ws.get_info(namespace, key_frac_burst),
        "info_backbone": ws.get_info(namespace, key_backbone),
    }


# ---------------------------------------------------------------------------
# Metadata queries — return inline (no large arrays)
# ---------------------------------------------------------------------------


async def get_data_info(
    workspace_id: str,
    namespace: str,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    return {
        "num_neurons": sd.N,
        "length_ms": sd.length,
        "metadata": sd.metadata,
    }


async def list_neurons(
    workspace_id: str,
    namespace: str,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    if sd.neuron_attributes is None:
        neurons = [{"index": i} for i in range(sd.N)]
    else:
        neurons = [
            {"index": i, **attrs} for i, attrs in enumerate(sd.neuron_attributes)
        ]
    return {"neurons": neurons}


async def get_neuron_attribute(
    workspace_id: str,
    namespace: str,
    key: str,
    default=None,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    values = sd.get_neuron_attribute(key, default=default)
    return {"key": key, "values": values}


async def set_neuron_attribute(
    workspace_id: str,
    namespace: str,
    key: str,
    values,
    neuron_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    sd.set_neuron_attribute(key, values, neuron_indices=neuron_indices)
    # Re-store to refresh the workspace index summary
    ws.store(namespace, _SPIKEDATA_KEY, sd)
    return {"workspace_id": workspace_id, "namespace": namespace, "key": key}


async def get_neuron_to_channel_map(
    workspace_id: str,
    namespace: str,
    channel_attr: Optional[str] = None,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    mapping = sd.neuron_to_channel_map(channel_attr=channel_attr)
    return {"mapping": {str(k): v for k, v in mapping.items()}}


# ---------------------------------------------------------------------------
# SpikeData transforms — output stored as SpikeData in workspace
# ---------------------------------------------------------------------------


async def subtime(
    workspace_id: str,
    namespace: str,
    start: float,
    end: float,
    out_namespace: str = "",
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    new_sd = sd.subtime(start, end)
    target_ns = out_namespace if out_namespace else namespace
    ws.store(target_ns, _SPIKEDATA_KEY, new_sd)
    return {
        "workspace_id": workspace_id,
        "namespace": target_ns,
        "workspace_key": _SPIKEDATA_KEY,
        "info": ws.get_info(target_ns, _SPIKEDATA_KEY),
    }


async def subset(
    workspace_id: str,
    namespace: str,
    units: List[int],
    by: Optional[str] = None,
    out_namespace: str = "",
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    new_sd = sd.subset(units, by=by)
    target_ns = out_namespace if out_namespace else namespace
    ws.store(target_ns, _SPIKEDATA_KEY, new_sd)
    return {
        "workspace_id": workspace_id,
        "namespace": target_ns,
        "workspace_key": _SPIKEDATA_KEY,
        "info": ws.get_info(target_ns, _SPIKEDATA_KEY),
    }


async def append_session(
    workspace_id: str,
    namespace_a: str,
    namespace_b: str,
    out_namespace: str = "",
    offset: float = 0.0,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd_a = _get_spikedata(ws, namespace_a)
    sd_b = _get_spikedata(ws, namespace_b)
    new_sd = sd_a.append(sd_b, offset=offset)
    target_ns = out_namespace if out_namespace else namespace_a
    ws.store(target_ns, _SPIKEDATA_KEY, new_sd)
    return {
        "workspace_id": workspace_id,
        "namespace": target_ns,
        "workspace_key": _SPIKEDATA_KEY,
        "info": ws.get_info(target_ns, _SPIKEDATA_KEY),
    }


async def concatenate_units(
    workspace_id: str,
    namespace_a: str,
    namespace_b: str,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd_a = _get_spikedata(ws, namespace_a)
    sd_b = _get_spikedata(ws, namespace_b)
    sd_a.concatenate_spike_data(sd_b)
    # Re-store to refresh the workspace index summary
    ws.store(namespace_a, _SPIKEDATA_KEY, sd_a)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace_a,
        "workspace_key": _SPIKEDATA_KEY,
        "info": ws.get_info(namespace_a, _SPIKEDATA_KEY),
    }


# ---------------------------------------------------------------------------
# RateData-based analysis — load RateData from workspace
# ---------------------------------------------------------------------------


async def compute_pairwise_fr_corr(
    workspace_id: str,
    namespace: str,
    rate_key: str,
    key_corr: str,
    key_lag: str,
    max_lag: int = 10,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    rd = _get_ratedata(ws, namespace, rate_key)
    corr_matrix, lag_matrix = rd.get_pairwise_fr_corr(max_lag=max_lag)
    ws.store(namespace, key_corr, corr_matrix)
    ws.store(namespace, key_lag, lag_matrix)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key_corr": key_corr,
        "key_lag": key_lag,
        "info_corr": ws.get_info(namespace, key_corr),
        "info_lag": ws.get_info(namespace, key_lag),
    }


async def compute_rate_manifold(
    workspace_id: str,
    namespace: str,
    rate_key: str,
    key: str,
    method: str = "PCA",
    n_components: int = 2,
    n_neighbors: Optional[int] = None,
    min_dist: Optional[float] = None,
    metric: Optional[str] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    rd = _get_ratedata(ws, namespace, rate_key)
    umap_kwargs: Dict[str, Any] = {}
    if n_neighbors is not None:
        umap_kwargs["n_neighbors"] = n_neighbors
    if min_dist is not None:
        umap_kwargs["min_dist"] = min_dist
    if metric is not None:
        umap_kwargs["metric"] = metric
    if random_state is not None:
        umap_kwargs["random_state"] = random_state
    embedding = rd.get_manifold(method=method, n_components=n_components, **umap_kwargs)
    ws.store(namespace, key, embedding)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "info": ws.get_info(namespace, key),
    }


async def frames_rate_data(
    workspace_id: str,
    namespace: str,
    rate_key: str,
    key: str,
    length: float,
    overlap: float = 0.0,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    rd = _get_ratedata(ws, namespace, rate_key)
    rss = rd.frames(length, overlap=overlap)
    ws.store(namespace, key, rss)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "n_frames": len(rss.times),
        "frame_length_ms": length,
        "step_size_ms": rss.step_size,
        "info": ws.get_info(namespace, key),
    }


# ---------------------------------------------------------------------------
# SpikeData → RateSliceStack / SpikeSliceStack (creation tools)
# ---------------------------------------------------------------------------


async def create_rate_slice_stack(
    workspace_id: str,
    namespace: str,
    key: str,
    times_start_to_end: List[List[float]],
    sigma_ms: float = 10.0,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    time_tuples = [tuple(t) for t in times_start_to_end]
    rss = RateSliceStack(sd, times_start_to_end=time_tuples, sigma_ms=sigma_ms)
    ws.store(namespace, key, rss)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "info": ws.get_info(namespace, key),
    }


async def frames_spike_data(
    workspace_id: str,
    namespace: str,
    key: str,
    length: float,
    overlap: float = 0.0,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    sss = sd.frames(length, overlap=overlap)
    ws.store(namespace, key, sss)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "n_frames": len(sss.times),
        "frame_length_ms": length,
        "info": ws.get_info(namespace, key),
    }


async def create_spike_slice_stack(
    workspace_id: str,
    namespace: str,
    key: str,
    times_start_to_end: List[List[float]],
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    time_tuples = [tuple(t) for t in times_start_to_end]
    sss = SpikeSliceStack(sd, times_start_to_end=time_tuples)
    ws.store(namespace, key, sss)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "info": ws.get_info(namespace, key),
    }


async def spike_slice_to_sparse(
    workspace_id: str,
    namespace: str,
    stack_key: str,
    key: str,
    bin_size: float = 1.0,
) -> Dict[str, Any]:
    """
    Convert a SpikeSliceStack stored in the workspace to a (U, T, S) binary
    sparse raster ndarray and store the result in the workspace.
    """
    ws = _get_workspace(workspace_id)
    sss = _get_spikeslicestack(ws, namespace, stack_key)
    sparse_list = [
        spike_slice.sparse_raster(bin_size=bin_size) for spike_slice in sss.spike_stack
    ]
    sparse_stack = np.stack(sparse_list, axis=2)
    ws.store(namespace, key, sparse_stack)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "bin_size": bin_size,
        "info": ws.get_info(namespace, key),
    }


# ---------------------------------------------------------------------------
# RateSliceStack-based analysis — load from workspace
# ---------------------------------------------------------------------------


async def compute_rate_slice_unit_corr(
    workspace_id: str,
    namespace: str,
    stack_key: str,
    out_key: str,
    min_rate_threshold: float = 0.1,
    min_frac: float = 0.3,
    max_lag: int = 10,
    compare_func: str = "cross_correlation",
) -> Dict[str, Any]:
    if compare_func not in _COMPARE_FUNCS:
        raise ValueError(f"compare_func must be one of {list(_COMPARE_FUNCS.keys())}")
    ws = _get_workspace(workspace_id)
    rss = _get_rateslicestack(ws, namespace, stack_key)
    pcm_stack, av_corr = rss.get_slice_to_slice_unit_corr_from_stack(
        compare_func=_COMPARE_FUNCS[compare_func],
        MIN_RATE_THRESHOLD=min_rate_threshold,
        MIN_FRAC=min_frac,
        max_lag=max_lag,
    )
    ws.store(namespace, out_key, pcm_stack)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "av_corr": _to_list(av_corr),
        "info": ws.get_info(namespace, out_key),
    }


async def compute_rate_slice_time_corr(
    workspace_id: str,
    namespace: str,
    stack_key: str,
    out_key: str,
    max_lag: int = 0,
    compare_func: str = "cosine_similarity",
) -> Dict[str, Any]:
    if compare_func not in _COMPARE_FUNCS:
        raise ValueError(f"compare_func must be one of {list(_COMPARE_FUNCS.keys())}")
    ws = _get_workspace(workspace_id)
    rss = _get_rateslicestack(ws, namespace, stack_key)
    pcm_stack, av_corr = rss.get_slice_to_slice_time_corr_from_stack(
        compare_func=_COMPARE_FUNCS[compare_func],
        max_lag=max_lag,
    )
    ws.store(namespace, out_key, pcm_stack)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "av_corr": _to_list(av_corr),
        "info": ws.get_info(namespace, out_key),
    }


async def compute_unit_to_unit_slice_corr(
    workspace_id: str,
    namespace: str,
    stack_key: str,
    out_key_corr: str,
    out_key_lag: str,
    max_lag: int = 10,
    compare_func: str = "cross_correlation",
) -> Dict[str, Any]:
    if compare_func not in _COMPARE_FUNCS:
        raise ValueError(f"compare_func must be one of {list(_COMPARE_FUNCS.keys())}")
    ws = _get_workspace(workspace_id)
    rss = _get_rateslicestack(ws, namespace, stack_key)
    corr_stack, lag_stack, av_max_corr, av_max_corr_lag = rss.unit_to_unit_correlation(
        compare_func=_COMPARE_FUNCS[compare_func],
        max_lag=max_lag,
    )
    ws.store(namespace, out_key_corr, corr_stack)
    ws.store(namespace, out_key_lag, lag_stack)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key_corr": out_key_corr,
        "key_lag": out_key_lag,
        "av_max_corr": _to_list(av_max_corr),
        "av_max_corr_lag": _to_list(av_max_corr_lag),
        "info_corr": ws.get_info(namespace, out_key_corr),
        "info_lag": ws.get_info(namespace, out_key_lag),
    }


async def compute_rate_slice_unit_order(
    workspace_id: str,
    namespace: str,
    stack_key: str,
    agg_func: str = "median",
    min_rate_threshold: float = 0.1,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    rss = _get_rateslicestack(ws, namespace, stack_key)
    _, unit_ids_in_order, unit_std_indices, unit_peak_times = (
        rss.order_units_across_slices(agg_func, MIN_RATE_THRESHOLD=min_rate_threshold)
    )
    return {
        "unit_ids_in_order": _to_list(unit_ids_in_order),
        "unit_std_indices": _to_list(unit_std_indices),
        "unit_peak_times": _to_list(unit_peak_times),
    }


# ---------------------------------------------------------------------------
# Other workspace-based tools
# ---------------------------------------------------------------------------


async def get_idces_times(
    workspace_id: str,
    namespace: str,
    key: str,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    idces, times = sd.idces_times()
    stacked = np.stack([idces.astype(np.float64), times.astype(np.float64)], axis=0)
    ws.store(namespace, key, stacked)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "n_spikes": int(len(times)),
        "info": ws.get_info(namespace, key),
    }


async def get_waveform_traces(
    workspace_id: str,
    namespace: str,
    key: str,
    unit: int,
    ms_before: float = 1.0,
    ms_after: float = 2.0,
    bandpass_low_hz: Optional[float] = None,
    bandpass_high_hz: Optional[float] = None,
    filter_order: int = 3,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    bandpass = None
    if bandpass_low_hz is not None or bandpass_high_hz is not None:
        bandpass = (bandpass_low_hz, bandpass_high_hz)
    waveforms, meta = sd.get_waveform_traces(
        unit=unit,
        ms_before=ms_before,
        ms_after=ms_after,
        bandpass=bandpass,
        filter_order=filter_order,
        store=False,
        return_channel_waveforms=False,
        return_avg_waveform=True,
    )
    ws.store(namespace, key, waveforms)
    avg_waveform = None
    if meta.get("avg_waveforms") and len(meta["avg_waveforms"]) > 0:
        avg_waveform = meta["avg_waveforms"][0].tolist()
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "channels": meta["channels"][0] if meta.get("channels") else [],
        "spike_times_ms": (
            meta["spike_times_ms"][0].tolist() if meta.get("spike_times_ms") else []
        ),
        "avg_waveform": avg_waveform,
        "fs_kHz": meta.get("fs_kHz"),
        "info": ws.get_info(namespace, key),
    }


# ---------------------------------------------------------------------------
# Dimensionality reduction pipeline (workspace-native, unchanged)
# ---------------------------------------------------------------------------


async def extract_lower_triangle_features(
    workspace_id: str,
    namespace: str,
    key: str,
    out_key: str,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    obj = ws.get(namespace, key)
    if obj is None:
        raise ValueError(f"Item not found: ({namespace!r}, {key!r})")
    if isinstance(obj, PairwiseCompMatrixStack):
        array = obj.stack
    elif isinstance(obj, np.ndarray) and obj.ndim == 3 and obj.shape[0] == obj.shape[1]:
        array = obj
    else:
        raise ValueError(
            f"Expected PairwiseCompMatrixStack or (N, N, S) ndarray at "
            f"({namespace!r}, {key!r}), got {type(obj).__name__}"
        )
    features = _extract_lower_triangle(array)
    ws.store(namespace, out_key, features)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "info": ws.get_info(namespace, out_key),
    }


async def pca_on_lower_triangle(
    workspace_id: str,
    namespace: str,
    key: str,
    out_key: str,
    n_components: int = 2,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    obj = ws.get(namespace, key)
    if obj is None:
        raise ValueError(f"Item not found: ({namespace!r}, {key!r})")
    if isinstance(obj, PairwiseCompMatrixStack):
        array = obj.stack
    elif isinstance(obj, np.ndarray) and obj.ndim == 3 and obj.shape[0] == obj.shape[1]:
        array = obj
    else:
        raise ValueError(
            f"Expected PairwiseCompMatrixStack or (N, N, S) ndarray at "
            f"({namespace!r}, {key!r}), got {type(obj).__name__}"
        )
    lower_tri = _extract_lower_triangle(array)
    embedding = PCA_reduction(lower_tri, n_components=n_components)
    ws.store(namespace, out_key, embedding)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "info": ws.get_info(namespace, out_key),
    }


async def pca_on_workspace_item(
    workspace_id: str,
    namespace: str,
    key: str,
    out_key: str,
    n_components: int = 2,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    obj = ws.get(namespace, key)
    if obj is None:
        raise ValueError(f"Item not found: ({namespace!r}, {key!r})")
    if not isinstance(obj, np.ndarray) or obj.ndim != 2:
        raise ValueError(
            f"Expected 2D ndarray at ({namespace!r}, {key!r}), "
            f"got {type(obj).__name__}"
            + (f" with ndim={obj.ndim}" if isinstance(obj, np.ndarray) else "")
        )
    embedding = PCA_reduction(obj, n_components=n_components)
    ws.store(namespace, out_key, embedding)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "info": ws.get_info(namespace, out_key),
    }


async def umap_reduction(
    workspace_id: str,
    namespace: str,
    key: str,
    out_key: str,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    obj = ws.get(namespace, key)
    if obj is None:
        raise ValueError(f"Item not found: ({namespace!r}, {key!r})")
    if not isinstance(obj, np.ndarray) or obj.ndim != 2:
        raise ValueError(
            f"Expected 2D ndarray at ({namespace!r}, {key!r}), "
            f"got {type(obj).__name__}"
            + (f" with ndim={obj.ndim}" if isinstance(obj, np.ndarray) else "")
        )
    embedding = UMAP_reduction(
        obj,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    ws.store(namespace, out_key, embedding)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "info": ws.get_info(namespace, out_key),
    }


async def umap_graph_communities(
    workspace_id: str,
    namespace: str,
    key: str,
    out_key: str,
    n_components: int = 2,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    obj = ws.get(namespace, key)
    if obj is None:
        raise ValueError(f"Item not found: ({namespace!r}, {key!r})")
    if not isinstance(obj, np.ndarray) or obj.ndim != 2:
        raise ValueError(
            f"Expected 2D ndarray at ({namespace!r}, {key!r}), "
            f"got {type(obj).__name__}"
            + (f" with ndim={obj.ndim}" if isinstance(obj, np.ndarray) else "")
        )
    embedding, labels = UMAP_graph_communities(
        obj,
        n_components=n_components,
        resolution=resolution,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    ws.store(namespace, out_key, embedding)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "labels": labels.tolist(),
        "info": ws.get_info(namespace, out_key),
    }


# ---------------------------------------------------------------------------
# Session management (kept for backwards compatibility)
# ---------------------------------------------------------------------------


async def list_sessions() -> Dict[str, Any]:
    session_ids = get_session_manager().list_sessions()
    return {"sessions": session_ids, "count": len(session_ids)}


async def delete_session(session_id: str) -> Dict[str, Any]:
    deleted = get_session_manager().delete_session(session_id)
    return {"deleted": deleted, "session_id": session_id}


# ---------------------------------------------------------------------------
# Workspace management
# ---------------------------------------------------------------------------


async def create_workspace(
    name: Optional[str] = None, lazy: bool = False
) -> Dict[str, Any]:
    wm = get_workspace_manager()
    workspace_id = wm.create_workspace(name=name, lazy=lazy)
    ws = wm.get_workspace(workspace_id)
    return {"workspace_id": workspace_id, "name": ws.name, "lazy": lazy}


async def delete_workspace(workspace_id: str) -> Dict[str, Any]:
    deleted = get_workspace_manager().delete_workspace(workspace_id)
    return {"deleted": deleted, "workspace_id": workspace_id}


async def list_workspaces() -> Dict[str, Any]:
    workspaces = get_workspace_manager().list_workspaces()
    return {"workspaces": workspaces, "count": len(workspaces)}


async def describe_workspace(workspace_id: str) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    return {"workspace_id": workspace_id, "index": ws.describe()}


async def workspace_get_info(
    workspace_id: str,
    namespace: str,
    key: str,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    info = ws.get_info(namespace, key)
    if info is None:
        raise ValueError(f"Item not found: ({namespace!r}, {key!r})")
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "info": info,
    }


async def rename_workspace_item(
    workspace_id: str,
    namespace: str,
    old_key: str,
    new_key: str,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    success = ws.rename(namespace, old_key, new_key)
    return {
        "success": success,
        "workspace_id": workspace_id,
        "namespace": namespace,
        "new_key": new_key,
    }


async def add_workspace_note(
    workspace_id: str,
    namespace: str,
    key: str,
    note: str,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    success = ws.add_note(namespace, key, note)
    return {"success": success}


async def delete_workspace_item(
    workspace_id: str,
    namespace: str,
    key: Optional[str] = None,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    deleted = ws.delete(namespace, key)
    return {"deleted": deleted}


async def save_workspace(workspace_id: str, path: str) -> Dict[str, Any]:
    wm = get_workspace_manager()
    if wm.get_workspace(workspace_id) is None:
        raise ValueError(f"Workspace not found: {workspace_id}")
    wm.save_workspace(workspace_id, path)
    return {
        "saved": True,
        "workspace_id": workspace_id,
        "h5_path": f"{path}.h5",
        "json_path": f"{path}.json",
    }


async def load_workspace(path: str) -> Dict[str, Any]:
    wm = get_workspace_manager()
    workspace_id = wm.load_workspace(path)
    ws = wm.get_workspace(workspace_id)
    item_count = sum(len(v) for v in ws._items.values())
    return {
        "workspace_id": workspace_id,
        "name": ws.name,
        "namespace_count": len(ws._items),
        "item_count": item_count,
    }


async def load_workspace_item(
    path: str,
    namespace: str,
    key: str,
    workspace_id: str,
) -> Dict[str, Any]:
    wm = get_workspace_manager()
    if wm.get_workspace(workspace_id) is None:
        raise ValueError(f"Workspace not found: {workspace_id}")
    wm.load_workspace_item(path, namespace, key, workspace_id)
    ws = wm.get_workspace(workspace_id)
    info = ws.get_info(namespace, key)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "info": info,
    }


async def fetch_workspace_item(
    workspace_id: str,
    namespace: str,
    key: str,
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    obj = ws.get(namespace, key)
    if obj is None:
        raise ValueError(f"Item not found: ({namespace!r}, {key!r})")
    info = ws.get_info(namespace, key)
    if isinstance(obj, np.ndarray):
        return {
            "workspace_id": workspace_id,
            "namespace": namespace,
            "key": key,
            "data": obj.tolist(),
            "info": info,
        }
    if isinstance(obj, PairwiseCompMatrixStack):
        return {
            "workspace_id": workspace_id,
            "namespace": namespace,
            "key": key,
            "data": obj.stack.tolist(),
            "info": info,
        }
    raise ValueError(
        f"fetch_workspace_item supports ndarray and PairwiseCompMatrixStack; "
        f"got {type(obj).__name__}. Use type-specific tools for other object types."
    )
