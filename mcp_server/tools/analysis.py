"""
MCP tools for analyzing spike data.

All tools are workspace-centric: inputs are loaded from an AnalysisWorkspace
and all outputs are stored back to the workspace. No bulk data is returned
inline to the agent.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ...spikedata.pairwise import PairwiseCompMatrix, PairwiseCompMatrixStack
from ...spikedata.ratedata import RateData
from ...spikedata.rateslicestack import RateSliceStack
from ...spikedata.spikeslicestack import SpikeSliceStack
from ...spikedata.spikedata import SpikeData
from ...spikedata.utils import (
    PCA_reduction,
    UMAP_graph_communities,
    UMAP_reduction,
    compute_cosine_similarity_with_lag,
    compute_cross_correlation_with_lag,
    consecutive_durations,
    gplvm_average_state_probability,
    gplvm_continuity_prob,
    gplvm_state_entropy,
)
from ...workspace.workspace import AnalysisWorkspace, get_workspace_manager

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
    """Compute mean firing rates for each unit and store to workspace."""
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
    """Compute binned spike counts and store to workspace."""
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
    """Compute binned mean firing rate across units and store to workspace."""
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
    """Compute a dense spike raster array and store to workspace."""
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


async def compute_channel_raster(
    workspace_id: str,
    namespace: str,
    key: str,
    bin_size: float = 20.0,
    channel_attr: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute a channel-grouped spike raster and store to workspace."""
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
    """Compute interspike intervals for all units and store NaN-padded array to workspace."""
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
    """Compute spike time tiling coefficient for a neuron pair and store to workspace."""
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
    """Compute pairwise spike time tiling coefficients for all units and store to workspace."""
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    pcm = sd.spike_time_tilings(delt=delt)
    ws.store(namespace, key, pcm)
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
    """Compute and threshold pairwise STTC matrix and store binary result to workspace."""
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    pcm = sd.spike_time_tilings(delt=delt)
    binary_pcm = pcm.threshold(threshold)
    ws.store(namespace, key, binary_pcm)
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
    """Compute spike latencies relative to event times and store NaN-padded array to workspace."""
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
    """Compute spike latencies relative to a reference neuron and store to workspace."""
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
    """Compute smoothed population firing rate and store to workspace."""
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
    """Compute spike-triggered population rate and coupling stats and store to workspace."""
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
    acc_square_width: int = 8,
    acc_gauss_sigma: int = 8,
    raster_bin_size_ms: float = 1.0,
    peak_to_trough: bool = True,
    pop_rms_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Detect population bursts and store burst times, edges, and amplitudes to workspace."""
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


async def burst_sensitivity(
    workspace_id: str,
    namespace: str,
    key: str,
    thr_values: List[float],
    dist_values: List[float],
    burst_edge_mult_thresh: float,
    square_width: int = 20,
    gauss_sigma: int = 100,
    acc_square_width: int = 8,
    acc_gauss_sigma: int = 8,
    raster_bin_size_ms: float = 1.0,
    peak_to_trough: bool = True,
    pop_rms_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute burst count sensitivity over threshold and distance grids and store to workspace."""
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    burst_counts = sd.burst_sensitivity(
        thr_values=np.asarray(thr_values),
        dist_values=np.asarray(dist_values),
        burst_edge_mult_thresh=burst_edge_mult_thresh,
        square_width=square_width,
        gauss_sigma=gauss_sigma,
        acc_square_width=acc_square_width,
        acc_gauss_sigma=acc_gauss_sigma,
        raster_bin_size_ms=raster_bin_size_ms,
        peak_to_trough=peak_to_trough,
        pop_rms_override=pop_rms_override,
    )
    ws.store(namespace, key, burst_counts)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "shape": list(burst_counts.shape),
        "info": ws.get_info(namespace, key),
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
    """Compute fraction of bursts each unit is active in and store to workspace."""
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
    """Return SpikeData metadata including neuron count and recording length inline."""
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    # Convert metadata values to JSON-safe types (numpy arrays/scalars → lists/floats)
    safe_metadata = {}
    for k, v in sd.metadata.items():
        if isinstance(v, np.ndarray):
            safe_metadata[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating, np.bool_)):
            safe_metadata[k] = v.item()
        else:
            safe_metadata[k] = v

    return {
        "num_neurons": sd.N,
        "length_ms": sd.length,
        "metadata": safe_metadata,
    }


async def list_neurons(
    workspace_id: str,
    namespace: str,
) -> Dict[str, Any]:
    """List all neurons with their attributes and return inline."""
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
    """Retrieve a single neuron attribute by key and return inline."""
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
    """Set a neuron attribute on the SpikeData and re-store to workspace."""
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
    """Return the neuron-to-channel mapping inline."""
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
    """Extract a time-windowed SpikeData subset and store to workspace."""
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
    """Extract a unit subset of SpikeData and store to workspace."""
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
    """Append two SpikeData sessions in time and store the result to workspace."""
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
    """Concatenate units from two SpikeData objects and store to workspace."""
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
    """Compute pairwise firing rate correlations from RateData and store to workspace."""
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


async def compute_pairwise_ccg(
    workspace_id: str,
    namespace: str,
    key_corr: str,
    key_lag: str,
    bin_size: float = 1.0,
    max_lag: float = 350,
    compare_func: str = "cross_correlation",
) -> Dict[str, Any]:
    """Compute pairwise cross-correlograms from SpikeData and store to workspace."""
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)

    if compare_func not in _COMPARE_FUNCS:
        raise ValueError(
            f"Unknown compare_func {compare_func!r}. "
            f"Choose one of {list(_COMPARE_FUNCS.keys())}."
        )
    func = _COMPARE_FUNCS[compare_func]

    corr_matrix, lag_matrix = sd.get_pairwise_ccg(
        compare_func=func, bin_size=bin_size, max_lag=max_lag
    )
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


async def compute_pairwise_latencies(
    workspace_id: str,
    namespace: str,
    key_mean: str,
    key_std: str,
    window_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute pairwise mean and std spike latencies and store to workspace."""
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    mean_lat, std_lat = sd.get_pairwise_latencies(window_ms=window_ms)
    ws.store(namespace, key_mean, mean_lat)
    ws.store(namespace, key_std, std_lat)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key_mean": key_mean,
        "key_std": key_std,
        "info_mean": ws.get_info(namespace, key_mean),
        "info_std": ws.get_info(namespace, key_std),
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
    store_pca_details: bool = False,
) -> Dict[str, Any]:
    """Compute a low-dimensional manifold embedding from RateData and store to workspace."""
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
    manifold_result = rd.get_manifold(
        method=method, n_components=n_components, **umap_kwargs
    )
    # PCA returns (embedding, var_ratio, components); UMAP returns (embedding, trustworthiness)
    if method.upper() == "PCA":
        embedding, var_ratio, components = manifold_result
        tw = None
    else:
        embedding, tw = manifold_result
        var_ratio = None
    ws.store(namespace, key, embedding)
    result: Dict[str, Any] = {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "info": ws.get_info(namespace, key),
    }
    if var_ratio is not None:
        result["explained_variance_ratio"] = var_ratio.tolist()
        if store_pca_details:
            var_key = f"{key}_variance"
            comp_key = f"{key}_components"
            ws.store(namespace, var_key, var_ratio)
            ws.store(namespace, comp_key, components)
            result["key_variance"] = var_key
            result["key_components"] = comp_key
    if tw is not None:
        result["trustworthiness"] = tw
    return result


async def frames_rate_data(
    workspace_id: str,
    namespace: str,
    rate_key: str,
    key: str,
    length: float,
    overlap: float = 0.0,
) -> Dict[str, Any]:
    """Slice RateData into overlapping frames as a RateSliceStack and store to workspace."""
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
    """Build a RateSliceStack from event time windows and store to workspace."""
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
    """Slice SpikeData into overlapping frames as a SpikeSliceStack and store to workspace."""
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
    """Build a SpikeSliceStack from event time windows and store to workspace."""
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


async def spike_slice_to_raster(
    workspace_id: str,
    namespace: str,
    stack_key: str,
    key: str,
    bin_size: float = 1.0,
) -> Dict[str, Any]:
    """
    Convert a SpikeSliceStack stored in the workspace to a (U, T, S) spike
    count raster ndarray and store the result in the workspace.
    """
    ws = _get_workspace(workspace_id)
    sss = _get_spikeslicestack(ws, namespace, stack_key)
    raster_stack = sss.to_raster_array(bin_size=bin_size)
    ws.store(namespace, key, raster_stack)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "bin_size": bin_size,
        "info": ws.get_info(namespace, key),
    }


async def align_to_events(
    workspace_id: str,
    namespace: str,
    key: str,
    events,
    pre_ms: float,
    post_ms: float,
    kind: str = "spike",
    bin_size_ms: float = 1.0,
    sigma_ms: float = 10.0,
) -> Dict[str, Any]:
    """
    Create an event-aligned slice stack from SpikeData and store it in the workspace.

    Reads SpikeData from (namespace, 'spikedata'). Events can be a list of
    times in ms or a string key into SpikeData.metadata. Stores either a
    SpikeSliceStack (kind='spike') or a RateSliceStack (kind='rate') at
    (namespace, key). Out-of-bounds events are dropped with a warning.
    """
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)

    if isinstance(events, list):
        events = np.array(events, dtype=float)

    result = sd.align_to_events(
        events,
        pre_ms,
        post_ms,
        kind=kind,
        bin_size_ms=bin_size_ms,
        sigma_ms=sigma_ms,
    )
    ws.store(namespace, key, result)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "kind": kind,
        "n_slices": (
            len(result.spike_stack) if kind == "spike" else result.event_stack.shape[2]
        ),
        "info": ws.get_info(namespace, key),
    }


# ---------------------------------------------------------------------------
# RateSliceStack-based analysis — load from workspace
# ---------------------------------------------------------------------------


def _get_optional_frac_active(ws, namespace, frac_active_key):
    """Load an optional frac_active (U,) ndarray from the workspace."""
    if frac_active_key is None:
        return None
    arr = ws.get(namespace, frac_active_key)
    if arr is None or not isinstance(arr, np.ndarray):
        raise ValueError(
            f"No ndarray found at ({namespace!r}, {frac_active_key!r}). "
            "Compute activity fractions first using: compute_frac_active "
            "or get_frac_active."
        )
    return arr


async def compute_rate_slice_unit_corr(
    workspace_id: str,
    namespace: str,
    stack_key: str,
    out_key: str,
    min_rate_threshold: float = 0.1,
    min_frac: float = 0.3,
    max_lag: int = 10,
    compare_func: str = "cross_correlation",
    frac_active_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute slice-to-slice unit correlations from a RateSliceStack and store to workspace."""
    if compare_func not in _COMPARE_FUNCS:
        raise ValueError(f"compare_func must be one of {list(_COMPARE_FUNCS.keys())}")
    ws = _get_workspace(workspace_id)
    rss = _get_rateslicestack(ws, namespace, stack_key)
    frac_active = _get_optional_frac_active(ws, namespace, frac_active_key)
    pcm_stack, av_corr = rss.get_slice_to_slice_unit_corr_from_stack(
        compare_func=_COMPARE_FUNCS[compare_func],
        MIN_RATE_THRESHOLD=min_rate_threshold,
        MIN_FRAC=min_frac,
        max_lag=max_lag,
        frac_active=frac_active,
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
    """Compute slice-to-slice time correlations from a RateSliceStack and store to workspace."""
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
    """Compute unit-to-unit correlations across RateSliceStack slices and store to workspace."""
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
    min_frac_active: float = 0.0,
    frac_active_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute unit activation ordering across RateSliceStack slices and return inline."""
    ws = _get_workspace(workspace_id)
    rss = _get_rateslicestack(ws, namespace, stack_key)
    frac_active = _get_optional_frac_active(ws, namespace, frac_active_key)
    _, unit_ids_in_order, unit_std_indices, unit_peak_times, unit_frac_active = (
        rss.order_units_across_slices(
            agg_func,
            MIN_RATE_THRESHOLD=min_rate_threshold,
            MIN_FRAC_ACTIVE=min_frac_active,
            frac_active=frac_active,
        )
    )
    # Each element is a tuple of two arrays (highly_active, low_active)
    return {
        "highly_active": {
            "unit_ids_in_order": _to_list(unit_ids_in_order[0]),
            "unit_std_indices": _to_list(unit_std_indices[0]),
            "unit_peak_times": _to_list(unit_peak_times[0]),
            "unit_frac_active": _to_list(unit_frac_active[0]),
        },
        "low_active": {
            "unit_ids_in_order": _to_list(unit_ids_in_order[1]),
            "unit_std_indices": _to_list(unit_std_indices[1]),
            "unit_peak_times": _to_list(unit_peak_times[1]),
            "unit_frac_active": _to_list(unit_frac_active[1]),
        },
    }


# ---------------------------------------------------------------------------
# Other workspace-based tools
# ---------------------------------------------------------------------------


async def get_idces_times(
    workspace_id: str,
    namespace: str,
    key: str,
) -> Dict[str, Any]:
    """Extract flat spike indices and times arrays and store to workspace."""
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
    """Extract raw waveform traces for a unit and store to workspace."""
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
    """Extract lower-triangle features from a PairwiseCompMatrixStack and store to workspace."""
    ws = _get_workspace(workspace_id)
    obj = ws.get(namespace, key)
    if obj is None:
        raise ValueError(f"Item not found: ({namespace!r}, {key!r})")
    if isinstance(obj, PairwiseCompMatrixStack):
        stack = obj
    elif isinstance(obj, np.ndarray) and obj.ndim == 3 and obj.shape[0] == obj.shape[1]:
        stack = PairwiseCompMatrixStack(stack=obj)
    else:
        raise ValueError(
            f"Expected PairwiseCompMatrixStack or (N, N, S) ndarray at "
            f"({namespace!r}, {key!r}), got {type(obj).__name__}"
        )
    features = stack.extract_lower_triangle_features()
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
    store_pca_details: bool = False,
) -> Dict[str, Any]:
    """Run PCA on lower-triangle features of a PairwiseCompMatrixStack and store to workspace."""
    ws = _get_workspace(workspace_id)
    obj = ws.get(namespace, key)
    if obj is None:
        raise ValueError(f"Item not found: ({namespace!r}, {key!r})")
    if isinstance(obj, PairwiseCompMatrixStack):
        stack = obj
    elif isinstance(obj, np.ndarray) and obj.ndim == 3 and obj.shape[0] == obj.shape[1]:
        stack = PairwiseCompMatrixStack(stack=obj)
    else:
        raise ValueError(
            f"Expected PairwiseCompMatrixStack or (N, N, S) ndarray at "
            f"({namespace!r}, {key!r}), got {type(obj).__name__}"
        )
    lower_tri = stack.extract_lower_triangle_features()
    embedding, var_ratio, components = PCA_reduction(
        lower_tri, n_components=n_components
    )
    ws.store(namespace, out_key, embedding)
    result: Dict[str, Any] = {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "info": ws.get_info(namespace, out_key),
        "explained_variance_ratio": var_ratio.tolist(),
    }
    if store_pca_details:
        var_key = f"{out_key}_variance"
        comp_key = f"{out_key}_components"
        ws.store(namespace, var_key, var_ratio)
        ws.store(namespace, comp_key, components)
        result["key_variance"] = var_key
        result["key_components"] = comp_key
    return result


async def pca_on_workspace_item(
    workspace_id: str,
    namespace: str,
    key: str,
    out_key: str,
    n_components: int = 2,
    store_pca_details: bool = False,
) -> Dict[str, Any]:
    """Run PCA on a 2D workspace array and store the embedding to workspace."""
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
    embedding, var_ratio, components = PCA_reduction(obj, n_components=n_components)
    ws.store(namespace, out_key, embedding)
    result: Dict[str, Any] = {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "info": ws.get_info(namespace, out_key),
        "explained_variance_ratio": var_ratio.tolist(),
    }
    if store_pca_details:
        var_key = f"{out_key}_variance"
        comp_key = f"{out_key}_components"
        ws.store(namespace, var_key, var_ratio)
        ws.store(namespace, comp_key, components)
        result["key_variance"] = var_key
        result["key_components"] = comp_key
    return result


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
    """Run UMAP dimensionality reduction on a 2D workspace array and store to workspace."""
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
    embedding, tw = UMAP_reduction(
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
        "trustworthiness": tw,
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
    """Run UMAP with Louvain community detection and store embedding to workspace."""
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
    embedding, labels, tw = UMAP_graph_communities(
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
        "trustworthiness": tw,
    }


# ---------------------------------------------------------------------------
# Workspace management
# ---------------------------------------------------------------------------


async def create_workspace(
    name: Optional[str] = None, lazy: bool = False
) -> Dict[str, Any]:
    """Create a new AnalysisWorkspace and return its ID inline."""
    wm = get_workspace_manager()
    workspace_id = wm.create_workspace(name=name, lazy=lazy)
    ws = wm.get_workspace(workspace_id)
    return {"workspace_id": workspace_id, "name": ws.name, "lazy": lazy}


async def delete_workspace(workspace_id: str) -> Dict[str, Any]:
    """Delete a workspace by ID and return confirmation inline."""
    deleted = get_workspace_manager().delete_workspace(workspace_id)
    return {"deleted": deleted, "workspace_id": workspace_id}


async def list_workspaces() -> Dict[str, Any]:
    """List all active workspaces and return inline."""
    workspaces = get_workspace_manager().list_workspaces()
    return {"workspaces": workspaces, "count": len(workspaces)}


async def describe_workspace(workspace_id: str) -> Dict[str, Any]:
    """Return the full workspace index describing all stored items inline."""
    ws = _get_workspace(workspace_id)
    return {"workspace_id": workspace_id, "index": ws.describe()}


async def workspace_get_info(
    workspace_id: str,
    namespace: str,
    key: str,
) -> Dict[str, Any]:
    """Return metadata for a single workspace item inline."""
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
    """Rename a workspace item's key within its namespace."""
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
    """Attach a text note to a workspace item."""
    ws = _get_workspace(workspace_id)
    success = ws.add_note(namespace, key, note)
    return {"success": success}


async def delete_workspace_item(
    workspace_id: str,
    namespace: str,
    key: Optional[str] = None,
) -> Dict[str, Any]:
    """Delete an item or entire namespace from a workspace."""
    ws = _get_workspace(workspace_id)
    deleted = ws.delete(namespace, key)
    return {"deleted": deleted}


async def save_workspace(workspace_id: str, path: str) -> Dict[str, Any]:
    """Serialize a workspace to HDF5 and JSON files on disk."""
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
    """Load a workspace from HDF5/JSON files and return its ID inline."""
    wm = get_workspace_manager()
    workspace_id = wm.load_workspace(path)
    ws = wm.get_workspace(workspace_id)
    item_count = sum(len(keys) for keys in ws.list_keys().values())
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
    """Load a single item from a saved workspace file into an existing workspace."""
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


async def merge_workspace(
    workspace_id: str,
    path: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Merge all items from a saved workspace file into an existing workspace."""
    ws = _get_workspace(workspace_id)
    other = AnalysisWorkspace.load(path)
    result = ws.merge_from(other, overwrite=overwrite)
    return {
        "workspace_id": workspace_id,
        "merged": result["merged"],
        "skipped": result["skipped"],
        "skipped_keys": [
            {"namespace": ns, "key": k} for ns, k in result["skipped_keys"]
        ],
    }


async def fetch_workspace_item(
    workspace_id: str,
    namespace: str,
    key: str,
) -> Dict[str, Any]:
    """Fetch the raw data of a workspace item and return it inline as a list."""
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


# ---------------------------------------------------------------------------
# Pairwise matrix conditioning tools
# ---------------------------------------------------------------------------


async def remove_by_condition(
    workspace_id: str,
    namespace: str,
    target_key: str,
    condition_key: str,
    out_key: str,
    op: str,
    threshold: float,
    fill: float = float("nan"),
    condition_namespace: Optional[str] = None,
) -> Dict[str, Any]:
    """Mask entries in a pairwise matrix by a condition matrix and store to workspace."""
    ws = _get_workspace(workspace_id)
    target = ws.get(namespace, target_key)
    if target is None or not isinstance(
        target, (PairwiseCompMatrixStack, PairwiseCompMatrix)
    ):
        raise ValueError(
            f"No PairwiseCompMatrix or PairwiseCompMatrixStack found at "
            f"({namespace!r}, {target_key!r}). "
            "Compute a pairwise matrix first using: compute_spike_time_tilings, "
            "compute_pairwise_ccg, compute_pairwise_latencies, "
            "spike_unit_to_unit_comparison, or similar."
        )

    cond_ns = condition_namespace if condition_namespace is not None else namespace
    condition = ws.get(cond_ns, condition_key)
    if condition is None or not isinstance(
        condition, (PairwiseCompMatrixStack, PairwiseCompMatrix)
    ):
        raise ValueError(
            f"No PairwiseCompMatrix or PairwiseCompMatrixStack found at "
            f"({cond_ns!r}, {condition_key!r}). "
            "Compute the condition matrix first using: compute_pairwise_latencies, "
            "compute_spike_time_tilings, compute_pairwise_ccg, or similar."
        )

    result = target.remove_by_condition(
        condition=condition, op=op, threshold=threshold, fill=fill
    )
    ws.store(namespace, out_key, result)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "info": ws.get_info(namespace, out_key),
    }


# ---------------------------------------------------------------------------
# SpikeSliceStack comparison tools
# ---------------------------------------------------------------------------


async def spike_unit_to_unit_comparison(
    workspace_id: str,
    namespace: str,
    stack_key: str,
    out_key_corr: str,
    out_key_lag: str,
    metric: str = "ccg",
    delt: float = 20.0,
    bin_size: float = 1.0,
    max_lag: float = 350,
) -> Dict[str, Any]:
    """Compute unit-to-unit pairwise comparisons from a SpikeSliceStack and store to workspace."""
    ws = _get_workspace(workspace_id)
    sss = _get_spikeslicestack(ws, namespace, stack_key)
    corr_stack, lag_stack, av_corr, av_lag = sss.unit_to_unit_comparison(
        metric=metric,
        delt=delt,
        bin_size=bin_size,
        max_lag=max_lag,
    )
    ws.store(namespace, out_key_corr, corr_stack)
    result: Dict[str, Any] = {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key_corr": out_key_corr,
        "av_corr": _to_list(av_corr),
        "info_corr": ws.get_info(namespace, out_key_corr),
    }
    if lag_stack is not None:
        ws.store(namespace, out_key_lag, lag_stack)
        result["key_lag"] = out_key_lag
        result["av_lag"] = _to_list(av_lag)
        result["info_lag"] = ws.get_info(namespace, out_key_lag)
    else:
        result["key_lag"] = None
        result["av_lag"] = None
    return result


async def spike_slice_to_slice_unit_comparison(
    workspace_id: str,
    namespace: str,
    stack_key: str,
    out_key_corr: str,
    out_key_lag: str,
    metric: str = "ccg",
    delt: float = 20.0,
    bin_size: float = 1.0,
    max_lag: float = 350,
    min_spikes: int = 2,
    min_frac: float = 0.3,
    frac_active_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute per-unit slice-to-slice comparisons from a SpikeSliceStack and store to workspace."""
    ws = _get_workspace(workspace_id)
    sss = _get_spikeslicestack(ws, namespace, stack_key)
    frac_active = _get_optional_frac_active(ws, namespace, frac_active_key)
    all_corr, all_lag, av_corr, av_lag = sss.get_slice_to_slice_unit_comparison(
        metric=metric,
        delt=delt,
        bin_size=bin_size,
        max_lag=max_lag,
        min_spikes=min_spikes,
        min_frac=min_frac,
        frac_active=frac_active,
    )
    ws.store(namespace, out_key_corr, all_corr)
    result: Dict[str, Any] = {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key_corr": out_key_corr,
        "av_corr": _to_list(av_corr),
        "info_corr": ws.get_info(namespace, out_key_corr),
    }
    if all_lag is not None:
        ws.store(namespace, out_key_lag, all_lag)
        result["key_lag"] = out_key_lag
        result["av_lag"] = _to_list(av_lag)
        result["info_lag"] = ws.get_info(namespace, out_key_lag)
    else:
        result["key_lag"] = None
        result["av_lag"] = None
    return result


async def compute_frac_active(
    workspace_id: str,
    namespace: str,
    stack_key: str,
    out_key: str,
    min_spikes: int = 2,
) -> Dict[str, Any]:
    """Compute fraction of slices each unit is active in and store to workspace."""
    ws = _get_workspace(workspace_id)
    sss = _get_spikeslicestack(ws, namespace, stack_key)
    frac = sss.compute_frac_active(min_spikes=min_spikes)
    ws.store(namespace, out_key, frac)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "info": ws.get_info(namespace, out_key),
    }


async def spike_order_units_across_slices(
    workspace_id: str,
    namespace: str,
    stack_key: str,
    agg_func: str = "median",
    timing: str = "median",
    min_spikes: int = 2,
    min_frac_active: float = 0.0,
    frac_active_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute unit activation ordering across SpikeSliceStack slices and return inline."""
    ws = _get_workspace(workspace_id)
    sss = _get_spikeslicestack(ws, namespace, stack_key)
    frac_active = _get_optional_frac_active(ws, namespace, frac_active_key)
    _, unit_ids, unit_std, unit_times, unit_frac = sss.order_units_across_slices(
        agg_func=agg_func,
        timing=timing,
        min_spikes=min_spikes,
        min_frac_active=min_frac_active,
        frac_active=frac_active,
    )
    return {
        "highly_active": {
            "unit_ids_in_order": _to_list(unit_ids[0]),
            "unit_std": _to_list(unit_std[0]),
            "unit_peak_times_ms": _to_list(unit_times[0]),
            "unit_frac_active": _to_list(unit_frac[0]),
        },
        "low_active": {
            "unit_ids_in_order": _to_list(unit_ids[1]),
            "unit_std": _to_list(unit_std[1]),
            "unit_peak_times_ms": _to_list(unit_times[1]),
            "unit_frac_active": _to_list(unit_frac[1]),
        },
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Unit timing and rank-order correlation tools
# ---------------------------------------------------------------------------


async def get_unit_timing_per_slice_spike(
    workspace_id: str,
    namespace: str,
    stack_key: str,
    out_key: str,
    timing: str = "median",
    min_spikes: int = 2,
) -> Dict[str, Any]:
    """Compute per-unit timing (median/mean/first spike) per slice from a SpikeSliceStack and store to workspace."""
    ws = _get_workspace(workspace_id)
    sss = _get_spikeslicestack(ws, namespace, stack_key)
    tm = sss.get_unit_timing_per_slice(timing=timing, min_spikes=min_spikes)
    ws.store(namespace, out_key, tm)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "info": ws.get_info(namespace, out_key),
    }


async def get_unit_timing_per_slice_rate(
    workspace_id: str,
    namespace: str,
    stack_key: str,
    out_key: str,
    min_rate_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Compute per-unit peak timing per slice from a RateSliceStack and store to workspace."""
    ws = _get_workspace(workspace_id)
    rss = _get_rateslicestack(ws, namespace, stack_key)
    tm = rss.get_unit_timing_per_slice(MIN_RATE_THRESHOLD=min_rate_threshold)
    ws.store(namespace, out_key, tm)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "info": ws.get_info(namespace, out_key),
    }


async def rank_order_correlation_spike(
    workspace_id: str,
    namespace: str,
    stack_key: str,
    out_key_corr: str,
    out_key_overlap: str,
    timing_key: Optional[str] = None,
    timing: str = "median",
    min_spikes: int = 2,
    min_overlap: int = 3,
    min_overlap_frac: Optional[float] = None,
    n_shuffles: int = 100,
    seed: int = 1,
) -> Dict[str, Any]:
    """Compute Spearman rank-order correlation across slices from a SpikeSliceStack with optional shuffle z-scoring."""
    ws = _get_workspace(workspace_id)
    sss = _get_spikeslicestack(ws, namespace, stack_key)
    timing_matrix = None
    if timing_key is not None:
        timing_matrix = ws.get(namespace, timing_key)
        if timing_matrix is None or not isinstance(timing_matrix, np.ndarray):
            raise ValueError(
                f"No ndarray found at ({namespace!r}, {timing_key!r}). "
                "Compute timing first using: get_unit_timing_per_slice_spike."
            )
    corr, av_corr, overlap = sss.rank_order_correlation(
        timing_matrix=timing_matrix,
        timing=timing,
        min_spikes=min_spikes,
        min_overlap=min_overlap,
        min_overlap_frac=min_overlap_frac,
        n_shuffles=n_shuffles,
        seed=seed,
    )
    ws.store(namespace, out_key_corr, corr)
    ws.store(namespace, out_key_overlap, overlap)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key_corr": out_key_corr,
        "key_overlap": out_key_overlap,
        "av_corr": av_corr,
        "n_shuffles": n_shuffles,
        "info_corr": ws.get_info(namespace, out_key_corr),
        "info_overlap": ws.get_info(namespace, out_key_overlap),
    }


async def rank_order_correlation_rate(
    workspace_id: str,
    namespace: str,
    stack_key: str,
    out_key_corr: str,
    out_key_overlap: str,
    timing_key: Optional[str] = None,
    min_rate_threshold: float = 0.1,
    min_overlap: int = 3,
    min_overlap_frac: Optional[float] = None,
    n_shuffles: int = 100,
    seed: int = 1,
) -> Dict[str, Any]:
    """Compute Spearman rank-order correlation across slices from a RateSliceStack with optional shuffle z-scoring."""
    ws = _get_workspace(workspace_id)
    rss = _get_rateslicestack(ws, namespace, stack_key)
    timing_matrix = None
    if timing_key is not None:
        timing_matrix = ws.get(namespace, timing_key)
        if timing_matrix is None or not isinstance(timing_matrix, np.ndarray):
            raise ValueError(
                f"No ndarray found at ({namespace!r}, {timing_key!r}). "
                "Compute timing first using: get_unit_timing_per_slice_rate."
            )
    corr, av_corr, overlap = rss.rank_order_correlation(
        timing_matrix=timing_matrix,
        MIN_RATE_THRESHOLD=min_rate_threshold,
        min_overlap=min_overlap,
        min_overlap_frac=min_overlap_frac,
        n_shuffles=n_shuffles,
        seed=seed,
    )
    ws.store(namespace, out_key_corr, corr)
    ws.store(namespace, out_key_overlap, overlap)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key_corr": out_key_corr,
        "key_overlap": out_key_overlap,
        "av_corr": av_corr,
        "n_shuffles": n_shuffles,
        "info_corr": ws.get_info(namespace, out_key_corr),
        "info_overlap": ws.get_info(namespace, out_key_overlap),
    }


# GPLVM tools
# ---------------------------------------------------------------------------


async def fit_gplvm(
    workspace_id: str,
    namespace: str,
    key: str,
    key_reorder: str,
    key_binned: str,
    bin_size_ms: float = 50.0,
    movement_variance: float = 1.0,
    tuning_lengthscale: float = 10.0,
    n_latent_bin: int = 100,
    n_iter: int = 20,
    n_time_per_chunk: int = 10000,
    random_seed: int = 3,
) -> Dict[str, Any]:
    """Fit a GPLVM model on SpikeData and store decode results to workspace."""
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    result = sd.fit_gplvm(
        bin_size_ms=bin_size_ms,
        movement_variance=movement_variance,
        tuning_lengthscale=tuning_lengthscale,
        n_latent_bin=n_latent_bin,
        n_iter=n_iter,
        n_time_per_chunk=n_time_per_chunk,
        random_seed=random_seed,
    )
    # Store decode_res dict, reorder indices, and binned spike counts
    ws.store(namespace, key, result["decode_res"])
    ws.store(namespace, key_reorder, result["reorder_indices"])
    ws.store(namespace, key_binned, result["binned_spike_counts"])
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": key,
        "key_reorder": key_reorder,
        "key_binned": key_binned,
        "log_marginal_l": _to_list(result["log_marginal_l"]),
        "bin_size_ms": result["bin_size_ms"],
        "n_time_bins": result["binned_spike_counts"].shape[0],
        "n_units": result["binned_spike_counts"].shape[1],
        "note": (
            f"decode_res dict stored at key={key!r}; "
            f"reorder_indices (U,) at key={key_reorder!r}; "
            f"binned_spike_counts (T, U) at key={key_binned!r}. "
            "The fitted model object is not stored (not serializable)."
        ),
    }


async def compute_gplvm_state_entropy(
    workspace_id: str,
    namespace: str,
    key: str,
    out_key: str,
) -> Dict[str, Any]:
    """Compute state entropy from GPLVM posterior and store to workspace."""
    ws = _get_workspace(workspace_id)
    decode_res = ws.get(namespace, key)
    if decode_res is None or not isinstance(decode_res, dict):
        raise ValueError(
            f"No decode_res dict found at ({namespace!r}, {key!r}). "
            "Fit a GPLVM first using: fit_gplvm."
        )
    posterior = np.asarray(decode_res["posterior_latent_marg"])
    ent = gplvm_state_entropy(posterior)
    ws.store(namespace, out_key, ent)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "info": ws.get_info(namespace, out_key),
    }


async def compute_gplvm_continuity_prob(
    workspace_id: str,
    namespace: str,
    key: str,
    out_key: str,
) -> Dict[str, Any]:
    """Compute GPLVM state continuity probability and store to workspace."""
    ws = _get_workspace(workspace_id)
    decode_res = ws.get(namespace, key)
    if decode_res is None or not isinstance(decode_res, dict):
        raise ValueError(
            f"No decode_res dict found at ({namespace!r}, {key!r}). "
            "Fit a GPLVM first using: fit_gplvm."
        )
    cont_prob = gplvm_continuity_prob(decode_res)
    ws.store(namespace, out_key, cont_prob)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "info": ws.get_info(namespace, out_key),
    }


async def compute_gplvm_avg_state_prob(
    workspace_id: str,
    namespace: str,
    key: str,
    out_key: str,
) -> Dict[str, Any]:
    """Compute average state probability from GPLVM posterior and store to workspace."""
    ws = _get_workspace(workspace_id)
    decode_res = ws.get(namespace, key)
    if decode_res is None or not isinstance(decode_res, dict):
        raise ValueError(
            f"No decode_res dict found at ({namespace!r}, {key!r}). "
            "Fit a GPLVM first using: fit_gplvm."
        )
    posterior = np.asarray(decode_res["posterior_latent_marg"])
    avg_prob = gplvm_average_state_probability(posterior)
    ws.store(namespace, out_key, avg_prob)
    return {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "info": ws.get_info(namespace, out_key),
    }


async def compute_gplvm_consecutive_durations(
    workspace_id: str,
    namespace: str,
    key: str,
    out_key: str,
    threshold: float,
    mode: str = "above",
    min_dur: int = 1,
) -> Dict[str, Any]:
    """Compute consecutive durations above or below a threshold and store to workspace."""
    ws = _get_workspace(workspace_id)
    signal = ws.get(namespace, key)
    if signal is None or not isinstance(signal, np.ndarray):
        raise ValueError(
            f"No ndarray found at ({namespace!r}, {key!r}). "
            "Compute the signal first using: compute_gplvm_continuity_prob "
            "or another tool that stores a 1-D array."
        )
    durations = consecutive_durations(signal, threshold, mode=mode, min_dur=min_dur)
    ws.store(namespace, out_key, durations)
    result: Dict[str, Any] = {
        "workspace_id": workspace_id,
        "namespace": namespace,
        "key": out_key,
        "n_durations": int(durations.size),
    }
    if durations.size > 0:
        result["mean_duration"] = float(np.mean(durations))
        result["median_duration"] = float(np.median(durations))
    return result
