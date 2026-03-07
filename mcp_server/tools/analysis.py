"""
MCP tools for analyzing spike data.

Provides async wrappers around SpikeData analysis methods.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from mcp_server.results import get_result_store
from mcp_server.sessions import get_session_manager
from spikedata.ratedata import RateData
from spikedata.rateslicestack import RateSliceStack
from spikedata.spikeslicestack import SpikeSliceStack
from spikedata.utils import (
    compute_cosine_similarity_with_lag,
    compute_cross_correlation_with_lag,
    extract_lower_triangle_features,
    PCA_reduction,
    UMAP_reduction,
    UMAP_graph_communities,
)

_COMPARE_FUNCS = {
    "cross_correlation": compute_cross_correlation_with_lag,
    "cosine_similarity": compute_cosine_similarity_with_lag,
}


def _to_list(arr):
    """Convert a numpy array to a nested Python list for JSON serialization."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def _get_session(session_id: str):
    """Get SpikeData from session, raising ValueError if not found."""
    sd = _get_session(session_id)
    return sd


async def compute_rates(
    session_id: str,
    unit: str = "kHz",
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    rates = sd.rates(unit=unit)
    return {"rates": _to_list(rates), "unit": unit}


async def compute_binned(
    session_id: str,
    bin_size: float = 40.0,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    binned = sd.binned(bin_size=bin_size)
    return {"binned": _to_list(binned), "bin_size": bin_size}


async def compute_binned_meanrate(
    session_id: str,
    bin_size: float = 40.0,
    unit: str = "kHz",
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    meanrate = sd.binned_meanrate(bin_size=bin_size, unit=unit)
    return {"meanrate": _to_list(meanrate), "bin_size": bin_size, "unit": unit}


async def compute_raster(
    session_id: str,
    bin_size: float = 20.0,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    raster = sd.raster(bin_size=bin_size)
    return {
        "raster": _to_list(raster),
        "shape": list(raster.shape),
        "bin_size": bin_size,
    }


async def compute_sparse_raster(
    session_id: str,
    bin_size: float = 20.0,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    raster = sd.sparse_raster(bin_size=bin_size).toarray()
    return {
        "raster": _to_list(raster),
        "shape": list(raster.shape),
        "bin_size": bin_size,
    }


async def compute_channel_raster(
    session_id: str,
    bin_size: float = 20.0,
    channel_attr: Optional[str] = None,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    raster = sd.channel_raster(bin_size=bin_size, channel_attr=channel_attr)
    return {
        "raster": _to_list(raster),
        "shape": list(raster.shape),
        "bin_size": bin_size,
    }


async def compute_interspike_intervals(
    session_id: str,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    isis = sd.interspike_intervals()
    return {"isis": [_to_list(isi) for isi in isis]}


async def compute_resampled_isi(
    session_id: str,
    times: List[float],
    sigma_ms: float = 10.0,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    rates = sd.resampled_isi(times=np.array(times), sigma_ms=sigma_ms)
    return {
        "rates": _to_list(rates),
        "times": times,
        "shape": list(rates.shape),
    }


async def subtime(
    session_id: str,
    start: float,
    end: float,
    create_new_session: bool = False,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    sm = get_session_manager()
    new_sd = sd.subtime(start, end)
    if create_new_session:
        new_session_id = sm.create_session(new_sd)
    else:
        sm.update_session(session_id, new_sd)
        new_session_id = session_id
    return {
        "session_id": new_session_id,
        "info": {
            "num_neurons": new_sd.N,
            "length_ms": new_sd.length,
            "metadata": new_sd.metadata,
        },
    }


async def subset(
    session_id: str,
    units: List[int],
    by: Optional[str] = None,
    create_new_session: bool = False,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    sm = get_session_manager()
    new_sd = sd.subset(units, by=by)
    if create_new_session:
        new_session_id = sm.create_session(new_sd)
    else:
        sm.update_session(session_id, new_sd)
        new_session_id = session_id
    return {
        "session_id": new_session_id,
        "info": {
            "num_neurons": new_sd.N,
            "length_ms": new_sd.length,
            "metadata": new_sd.metadata,
        },
    }


async def compute_spike_time_tiling(
    session_id: str,
    neuron_i: int,
    neuron_j: int,
    delt: float = 20.0,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    sttc = sd.spike_time_tiling(neuron_i, neuron_j, delt=delt)
    return {
        "sttc": float(sttc),
        "neuron_i": neuron_i,
        "neuron_j": neuron_j,
        "delt": delt,
    }


async def compute_spike_time_tilings(
    session_id: str,
    delt: float = 20.0,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    pcm = sd.spike_time_tilings(delt=delt)
    return {
        "matrix": _to_list(pcm.matrix),
        "delt": delt,
        "shape": list(pcm.matrix.shape),
    }


async def compute_latencies(
    session_id: str,
    times: List[float],
    window_ms: float = 100.0,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    latencies = sd.latencies(times, window_ms=window_ms)
    return {
        "latencies": [list(row) for row in latencies],
        "window_ms": window_ms,
    }


async def compute_latencies_to_index(
    session_id: str,
    neuron_index: int,
    window_ms: float = 100.0,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    latencies = sd.latencies_to_index(neuron_index, window_ms=window_ms)
    return {
        "latencies": [list(row) for row in latencies],
        "neuron_index": neuron_index,
        "window_ms": window_ms,
    }


async def get_frac_active(
    session_id: str,
    edges: List[List[float]],
    min_spikes: int,
    backbone_threshold: float,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    edges_arr = np.array(edges)
    frac_per_unit, frac_per_burst, backbone_units = sd.get_frac_active(
        edges_arr, min_spikes, backbone_threshold
    )
    return {
        "frac_per_unit": _to_list(frac_per_unit),
        "frac_per_burst": _to_list(frac_per_burst),
        "backbone_units": _to_list(backbone_units),
    }


async def get_data_info(
    session_id: str,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    return {
        "num_neurons": sd.N,
        "length_ms": sd.length,
        "metadata": sd.metadata,
    }


async def list_neurons(
    session_id: str,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    if sd.neuron_attributes is None:
        neurons = [{"index": i} for i in range(sd.N)]
    else:
        neurons = [
            {"index": i, **attrs} for i, attrs in enumerate(sd.neuron_attributes)
        ]
    return {"neurons": neurons}


async def get_pop_rate(
    session_id: str,
    square_width: int = 20,
    gauss_sigma: int = 100,
    raster_bin_size_ms: float = 1.0,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    pop_rate = sd.get_pop_rate(
        square_width=square_width,
        gauss_sigma=gauss_sigma,
        raster_bin_size_ms=raster_bin_size_ms,
    )
    return {
        "pop_rate": _to_list(pop_rate),
        "raster_bin_size_ms": raster_bin_size_ms,
    }


async def compute_spike_trig_pop_rate(
    session_id: str,
    window_ms: int = 80,
    cutoff_hz: float = 20,
    fs: float = 1000,
    bin_size: float = 1,
    cut_outer: int = 10,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    stPR_filtered, coupling_zero_lag, coupling_max, delays, lags = (
        sd.compute_spike_trig_pop_rate(
            window_ms=window_ms,
            cutoff_hz=cutoff_hz,
            fs=fs,
            bin_size=bin_size,
            cut_outer=cut_outer,
        )
    )
    return {
        "stPR_filtered": _to_list(stPR_filtered),
        "coupling_strengths_zero_lag": _to_list(coupling_zero_lag),
        "coupling_strengths_max": _to_list(coupling_max),
        "delays": _to_list(delays),
        "lags": _to_list(lags),
    }


async def get_bursts(
    session_id: str,
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
    sd = _get_session(session_id)
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
    return {
        "tburst": _to_list(tburst),
        "edges": _to_list(edges),
        "peak_amp": _to_list(peak_amp),
    }


async def threshold_spike_time_tilings(
    session_id: str,
    threshold: float,
    delt: float = 20.0,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    pcm = sd.spike_time_tilings(delt=delt)
    binary_pcm = pcm.threshold(threshold)
    return {
        "matrix": _to_list(binary_pcm.matrix),
        "threshold": threshold,
        "delt": delt,
        "shape": list(binary_pcm.matrix.shape),
    }


async def compute_pairwise_fr_corr(
    session_id: str,
    times: List[float],
    sigma_ms: float = 10.0,
    max_lag: int = 10,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    rate_matrix = sd.resampled_isi(times=np.array(times), sigma_ms=sigma_ms)
    rd = RateData(rate_matrix, np.array(times))
    corr_matrix, lag_matrix = rd.get_pairwise_fr_corr(max_lag=max_lag)
    result_store = get_result_store()
    result_id_corr = result_store.store(corr_matrix, session_ids=[session_id])
    result_id_lag = result_store.store(lag_matrix, session_ids=[session_id])
    return {
        "result_id_corr": result_id_corr,
        "result_id_lag": result_id_lag,
        "shape": list(corr_matrix.shape),
        "ttl_seconds": result_store.default_ttl,
    }


async def compute_rate_manifold(
    session_id: str,
    times: List[float],
    sigma_ms: float = 10.0,
    method: str = "PCA",
    n_components: int = 2,
    n_neighbors: Optional[int] = None,
    min_dist: Optional[float] = None,
    metric: Optional[str] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    rate_matrix = sd.resampled_isi(times=np.array(times), sigma_ms=sigma_ms)
    rd = RateData(rate_matrix, np.array(times))
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
    result_store = get_result_store()
    result_id = result_store.store(embedding, session_ids=[session_id])
    return {
        "result_id": result_id,
        "times": times,
        "shape": list(embedding.shape),
        "dtype": str(embedding.dtype),
        "ttl_seconds": result_store.default_ttl,
    }


async def compute_rate_slice_unit_corr(
    session_id: str,
    times_start_to_end: List[List[float]],
    sigma_ms: float = 10.0,
    min_rate_threshold: float = 0.1,
    min_frac: float = 0.3,
    max_lag: int = 10,
    compare_func: str = "cross_correlation",
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    if compare_func not in _COMPARE_FUNCS:
        raise ValueError(f"compare_func must be one of {list(_COMPARE_FUNCS.keys())}")
    time_tuples = [tuple(t) for t in times_start_to_end]
    rss = RateSliceStack(sd, times_start_to_end=time_tuples, sigma_ms=sigma_ms)
    pcm_stack, av_corr = rss.get_slice_to_slice_unit_corr_from_stack(
        compare_func=_COMPARE_FUNCS[compare_func],
        MIN_RATE_THRESHOLD=min_rate_threshold,
        MIN_FRAC=min_frac,
        max_lag=max_lag,
    )
    result_store = get_result_store()
    result_id = result_store.store(pcm_stack.stack, session_ids=[session_id])
    return {
        "result_id": result_id,
        "av_corr": _to_list(av_corr),
        "shape": list(pcm_stack.stack.shape),
        "dtype": str(pcm_stack.stack.dtype),
        "ttl_seconds": result_store.default_ttl,
    }


async def compute_rate_slice_time_corr(
    session_id: str,
    times_start_to_end: List[List[float]],
    sigma_ms: float = 10.0,
    max_lag: int = 0,
    compare_func: str = "cosine_similarity",
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    if compare_func not in _COMPARE_FUNCS:
        raise ValueError(f"compare_func must be one of {list(_COMPARE_FUNCS.keys())}")
    time_tuples = [tuple(t) for t in times_start_to_end]
    rss = RateSliceStack(sd, times_start_to_end=time_tuples, sigma_ms=sigma_ms)
    pcm_stack, av_corr = rss.get_slice_to_slice_time_corr_from_stack(
        compare_func=_COMPARE_FUNCS[compare_func],
        max_lag=max_lag,
    )
    result_store = get_result_store()
    result_id = result_store.store(pcm_stack.stack, session_ids=[session_id])
    return {
        "result_id": result_id,
        "av_corr": _to_list(av_corr),
        "shape": list(pcm_stack.stack.shape),
        "dtype": str(pcm_stack.stack.dtype),
        "ttl_seconds": result_store.default_ttl,
    }


async def compute_unit_to_unit_slice_corr(
    session_id: str,
    times_start_to_end: List[List[float]],
    sigma_ms: float = 10.0,
    max_lag: int = 10,
    compare_func: str = "cross_correlation",
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    if compare_func not in _COMPARE_FUNCS:
        raise ValueError(f"compare_func must be one of {list(_COMPARE_FUNCS.keys())}")
    time_tuples = [tuple(t) for t in times_start_to_end]
    rss = RateSliceStack(sd, times_start_to_end=time_tuples, sigma_ms=sigma_ms)
    corr_stack, lag_stack, av_max_corr, av_max_corr_lag = rss.unit_to_unit_correlation(
        compare_func=_COMPARE_FUNCS[compare_func],
        max_lag=max_lag,
    )
    result_store = get_result_store()
    result_id_corr = result_store.store(corr_stack.stack, session_ids=[session_id])
    result_id_lag = result_store.store(lag_stack.stack, session_ids=[session_id])
    return {
        "result_id_corr": result_id_corr,
        "result_id_lag": result_id_lag,
        "av_max_corr": _to_list(av_max_corr),
        "av_max_corr_lag": _to_list(av_max_corr_lag),
        "shape": list(corr_stack.stack.shape),
        "dtype": str(corr_stack.stack.dtype),
        "ttl_seconds": result_store.default_ttl,
    }


def _load_rate_slice_stack(result_id: str):
    """Reconstruct a RateSliceStack from a stored event_stack result."""
    entry = get_result_store().get(result_id)
    if entry is None:
        raise ValueError(f"Result not found or expired: {result_id}")
    array, meta = entry
    if array.ndim != 3:
        raise ValueError(
            f"Expected 3D event_stack (U, T, S), got shape {list(array.shape)}"
        )
    extra = meta.get("extra_meta") or {}
    times = extra.get("times")
    if times is None:
        raise ValueError(
            "Stored result has no 'times' metadata — use create_rate_slice_stack to create a cacheable stack"
        )
    step_size = extra.get("step_size")
    time_tuples = [tuple(t) for t in times]
    rss = RateSliceStack(
        data_obj=None,
        event_matrix=array,
        times_start_to_end=time_tuples,
        step_size=step_size,
    )
    return rss, meta


async def create_rate_slice_stack(
    session_id: str,
    times_start_to_end: List[List[float]],
    sigma_ms: float = 10.0,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    time_tuples = [tuple(t) for t in times_start_to_end]
    rss = RateSliceStack(sd, times_start_to_end=time_tuples, sigma_ms=sigma_ms)
    result_store = get_result_store()
    result_id = result_store.store(
        rss.event_stack,
        session_ids=[session_id],
        extra_meta={"times": times_start_to_end, "step_size": rss.step_size},
    )
    return {
        "result_id": result_id,
        "shape": list(rss.event_stack.shape),
        "dtype": str(rss.event_stack.dtype),
        "n_units": rss.event_stack.shape[0],
        "n_time_bins": rss.event_stack.shape[1],
        "n_slices": rss.event_stack.shape[2],
        "step_size_ms": rss.step_size,
        "ttl_seconds": result_store.default_ttl,
    }


async def compute_rate_slice_unit_corr_from_stack(
    result_id: str,
    min_rate_threshold: float = 0.1,
    min_frac: float = 0.3,
    max_lag: int = 10,
    compare_func: str = "cross_correlation",
) -> Dict[str, Any]:
    if compare_func not in _COMPARE_FUNCS:
        raise ValueError(f"compare_func must be one of {list(_COMPARE_FUNCS.keys())}")
    rss, meta = _load_rate_slice_stack(result_id)
    inherited_session_ids = meta.get("session_ids")
    pcm_stack, av_corr = rss.get_slice_to_slice_unit_corr_from_stack(
        compare_func=_COMPARE_FUNCS[compare_func],
        MIN_RATE_THRESHOLD=min_rate_threshold,
        MIN_FRAC=min_frac,
        max_lag=max_lag,
    )
    result_store = get_result_store()
    result_id_out = result_store.store(
        pcm_stack.stack, session_ids=inherited_session_ids
    )
    return {
        "result_id": result_id_out,
        "av_corr": _to_list(av_corr),
        "shape": list(pcm_stack.stack.shape),
        "dtype": str(pcm_stack.stack.dtype),
        "ttl_seconds": result_store.default_ttl,
    }


async def compute_rate_slice_time_corr_from_stack(
    result_id: str,
    max_lag: int = 0,
    compare_func: str = "cosine_similarity",
) -> Dict[str, Any]:
    if compare_func not in _COMPARE_FUNCS:
        raise ValueError(f"compare_func must be one of {list(_COMPARE_FUNCS.keys())}")
    rss, meta = _load_rate_slice_stack(result_id)
    inherited_session_ids = meta.get("session_ids")
    pcm_stack, av_corr = rss.get_slice_to_slice_time_corr_from_stack(
        compare_func=_COMPARE_FUNCS[compare_func],
        max_lag=max_lag,
    )
    result_store = get_result_store()
    result_id_out = result_store.store(
        pcm_stack.stack, session_ids=inherited_session_ids
    )
    return {
        "result_id": result_id_out,
        "av_corr": _to_list(av_corr),
        "shape": list(pcm_stack.stack.shape),
        "dtype": str(pcm_stack.stack.dtype),
        "ttl_seconds": result_store.default_ttl,
    }


async def compute_unit_to_unit_slice_corr_from_stack(
    result_id: str,
    max_lag: int = 10,
    compare_func: str = "cross_correlation",
) -> Dict[str, Any]:
    if compare_func not in _COMPARE_FUNCS:
        raise ValueError(f"compare_func must be one of {list(_COMPARE_FUNCS.keys())}")
    rss, meta = _load_rate_slice_stack(result_id)
    inherited_session_ids = meta.get("session_ids")
    corr_stack, lag_stack, av_max_corr, av_max_corr_lag = rss.unit_to_unit_correlation(
        compare_func=_COMPARE_FUNCS[compare_func],
        max_lag=max_lag,
    )
    result_store = get_result_store()
    result_id_corr = result_store.store(
        corr_stack.stack, session_ids=inherited_session_ids
    )
    result_id_lag = result_store.store(
        lag_stack.stack, session_ids=inherited_session_ids
    )
    return {
        "result_id_corr": result_id_corr,
        "result_id_lag": result_id_lag,
        "av_max_corr": _to_list(av_max_corr),
        "av_max_corr_lag": _to_list(av_max_corr_lag),
        "shape": list(corr_stack.stack.shape),
        "dtype": str(corr_stack.stack.dtype),
        "ttl_seconds": result_store.default_ttl,
    }


async def compute_rate_slice_unit_order_from_stack(
    result_id: str,
    agg_func: str = "median",
    min_rate_threshold: float = 0.1,
) -> Dict[str, Any]:
    rss, _ = _load_rate_slice_stack(result_id)
    _, unit_ids_in_order, unit_std_indices, unit_peak_times = (
        rss.order_units_across_slices(agg_func, MIN_RATE_THRESHOLD=min_rate_threshold)
    )
    return {
        "unit_ids_in_order": _to_list(unit_ids_in_order),
        "unit_std_indices": _to_list(unit_std_indices),
        "unit_peak_times": _to_list(unit_peak_times),
    }


async def compute_rate_slice_unit_order(
    session_id: str,
    times_start_to_end: List[List[float]],
    sigma_ms: float = 10.0,
    agg_func: str = "median",
    min_rate_threshold: float = 0.1,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    time_tuples = [tuple(t) for t in times_start_to_end]
    rss = RateSliceStack(sd, times_start_to_end=time_tuples, sigma_ms=sigma_ms)
    _, unit_ids_in_order, unit_std_indices, unit_peak_times = (
        rss.order_units_across_slices(agg_func, MIN_RATE_THRESHOLD=min_rate_threshold)
    )
    return {
        "unit_ids_in_order": _to_list(unit_ids_in_order),
        "unit_std_indices": _to_list(unit_std_indices),
        "unit_peak_times": _to_list(unit_peak_times),
    }


async def compute_spike_slice_sparse_matrices(
    session_id: str,
    times_start_to_end: List[List[float]],
    bin_size: float = 1.0,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    time_tuples = [tuple(t) for t in times_start_to_end]
    sss = SpikeSliceStack(sd, times_start_to_end=time_tuples)
    sparse_list = [
        spike_obj_slice.sparse_raster(bin_size=bin_size)
        for spike_obj_slice in sss.spike_stack
    ]
    sparse_stack = np.stack(sparse_list, axis=2)
    result_store = get_result_store()
    result_id = result_store.store(sparse_stack, session_ids=[session_id])
    return {
        "result_id": result_id,
        "shape": list(sparse_stack.shape),
        "dtype": str(sparse_stack.dtype),
        "bin_size_ms": bin_size,
        "ttl_seconds": result_store.default_ttl,
    }


async def frames_spike_data(
    session_id: str,
    length: float,
    overlap: float = 0.0,
    bin_size: float = 1.0,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    sss = sd.frames(length, overlap=overlap)
    sparse_list = [
        spike_obj_slice.sparse_raster(bin_size=bin_size)
        for spike_obj_slice in sss.spike_stack
    ]
    sparse_stack = np.stack(sparse_list, axis=2)
    result_store = get_result_store()
    result_id = result_store.store(
        sparse_stack,
        session_ids=[session_id],
        extra_meta={"times": [list(t) for t in sss.times], "bin_size_ms": bin_size},
    )
    return {
        "result_id": result_id,
        "shape": list(sparse_stack.shape),
        "dtype": str(sparse_stack.dtype),
        "n_frames": len(sss.times),
        "frame_length_ms": length,
        "step_size_ms": length - overlap,
        "bin_size_ms": bin_size,
        "ttl_seconds": result_store.default_ttl,
    }


async def frames_rate_data(
    session_id: str,
    isi_times: List[float],
    length: float,
    overlap: float = 0.0,
    sigma_ms: float = 10.0,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    rate_matrix = sd.resampled_isi(times=np.array(isi_times), sigma_ms=sigma_ms)
    rd = RateData(rate_matrix, np.array(isi_times))
    rss = rd.frames(length, overlap=overlap)
    result_store = get_result_store()
    result_id = result_store.store(
        rss.event_stack,
        session_ids=[session_id],
        extra_meta={"times": [list(t) for t in rss.times], "step_size": rss.step_size},
    )
    return {
        "result_id": result_id,
        "shape": list(rss.event_stack.shape),
        "dtype": str(rss.event_stack.dtype),
        "n_units": rss.event_stack.shape[0],
        "n_time_bins": rss.event_stack.shape[1],
        "n_slices": rss.event_stack.shape[2],
        "step_size_ms": rss.step_size,
        "ttl_seconds": result_store.default_ttl,
    }


async def fetch_result(result_id: str) -> Dict[str, Any]:
    entry = get_result_store().get(result_id)
    if entry is None:
        raise ValueError(f"Result not found or expired: {result_id}")
    array, meta = entry
    return {
        "result_id": result_id,
        "data": array.tolist(),
        "shape": meta["shape"],
        "dtype": meta["dtype"],
    }


async def delete_result(result_id: str) -> Dict[str, Any]:
    deleted = get_result_store().delete(result_id)
    return {"deleted": deleted, "result_id": result_id}


async def list_results(session_id: str) -> Dict[str, Any]:
    results = get_result_store().list_by_session(session_id)
    return {"session_id": session_id, "results": results, "count": len(results)}


async def list_sessions() -> Dict[str, Any]:
    session_ids = get_session_manager().list_sessions()
    return {"sessions": session_ids, "count": len(session_ids)}


async def delete_session(session_id: str) -> Dict[str, Any]:
    deleted = get_session_manager().delete_session(session_id)
    return {"deleted": deleted, "session_id": session_id}


async def get_neuron_attribute(
    session_id: str,
    key: str,
    default=None,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    values = sd.get_neuron_attribute(key, default=default)
    return {"key": key, "values": values}


async def set_neuron_attribute(
    session_id: str,
    key: str,
    values,
    neuron_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    sd.set_neuron_attribute(key, values, neuron_indices=neuron_indices)
    return {"session_id": session_id, "key": key}


async def get_neuron_to_channel_map(
    session_id: str,
    channel_attr: Optional[str] = None,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
    mapping = sd.neuron_to_channel_map(channel_attr=channel_attr)
    return {"mapping": {str(k): v for k, v in mapping.items()}}


async def get_idces_times(session_id: str) -> Dict[str, Any]:
    sd = _get_session(session_id)
    idces, times = sd.idces_times()
    # Store as (2, n_spikes) float64: row 0 = unit indices, row 1 = spike times
    stacked = np.stack([idces.astype(np.float64), times.astype(np.float64)], axis=0)
    result_store = get_result_store()
    result_id = result_store.store(stacked, session_ids=[session_id])
    return {
        "result_id": result_id,
        "shape": list(stacked.shape),
        "dtype": str(stacked.dtype),
        "ttl_seconds": result_store.default_ttl,
        "n_spikes": int(len(times)),
    }


async def append_session(
    session_id: str,
    other_session_id: str,
    offset: float = 0.0,
    create_new_session: bool = True,
) -> Dict[str, Any]:
    sm = get_session_manager()
    sd = sm.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
    other_sd = sm.get_session(other_session_id)
    if other_sd is None:
        raise ValueError("other_session_id not found")
    new_sd = sd.append(other_sd, offset=offset)
    if create_new_session:
        new_session_id = sm.create_session(new_sd)
    else:
        sm.update_session(session_id, new_sd)
        new_session_id = session_id
    return {
        "session_id": new_session_id,
        "info": {
            "num_neurons": new_sd.N,
            "length_ms": new_sd.length,
            "metadata": new_sd.metadata,
        },
    }


async def concatenate_units(
    session_id: str,
    other_session_id: str,
) -> Dict[str, Any]:
    sm = get_session_manager()
    sd = sm.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
    other_sd = sm.get_session(other_session_id)
    if other_sd is None:
        raise ValueError("other_session_id not found")
    sd.concatenate_spike_data(other_sd)
    sm.update_session(session_id, sd)
    return {
        "session_id": session_id,
        "info": {
            "num_neurons": sd.N,
            "length_ms": sd.length,
            "metadata": sd.metadata,
        },
    }


async def get_waveform_traces(
    session_id: str,
    unit: int,
    ms_before: float = 1.0,
    ms_after: float = 2.0,
    bandpass_low_hz: Optional[float] = None,
    bandpass_high_hz: Optional[float] = None,
    filter_order: int = 3,
) -> Dict[str, Any]:
    sd = _get_session(session_id)
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
    result_store = get_result_store()
    result_id = result_store.store(waveforms, session_ids=[session_id])
    avg_waveform = None
    if meta.get("avg_waveforms") and len(meta["avg_waveforms"]) > 0:
        avg_waveform = meta["avg_waveforms"][0].tolist()
    return {
        "result_id": result_id,
        "shape": list(waveforms.shape),
        "dtype": str(waveforms.dtype),
        "ttl_seconds": result_store.default_ttl,
        "channels": meta["channels"][0] if meta.get("channels") else [],
        "spike_times_ms": (
            meta["spike_times_ms"][0].tolist() if meta.get("spike_times_ms") else []
        ),
        "avg_waveform": avg_waveform,
        "fs_kHz": meta.get("fs_kHz"),
    }


async def extract_lower_triangle_features_from_result(
    result_id: str,
) -> Dict[str, Any]:
    entry = get_result_store().get(result_id)
    if entry is None:
        raise ValueError(f"Result not found or expired: {result_id}")
    array, meta = entry
    if array.ndim != 3 or array.shape[0] != array.shape[1]:
        raise ValueError(f"Expected (N, N, S) array, got shape {list(array.shape)}")
    inherited_session_ids = meta.get("session_ids")
    features = extract_lower_triangle_features(array)
    result_store = get_result_store()
    result_id_out = result_store.store(features, session_ids=inherited_session_ids)
    return {
        "result_id": result_id_out,
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "ttl_seconds": result_store.default_ttl,
    }


async def pca_on_lower_triangle(
    result_id: str,
    n_components: int = 2,
) -> Dict[str, Any]:
    entry = get_result_store().get(result_id)
    if entry is None:
        raise ValueError(f"Result not found or expired: {result_id}")
    array, meta = entry
    if array.ndim != 3 or array.shape[0] != array.shape[1]:
        raise ValueError(f"Expected (N, N, S) array, got shape {list(array.shape)}")
    inherited_session_ids = meta.get("session_ids")
    lower_tri = extract_lower_triangle_features(array)
    embedding = PCA_reduction(lower_tri, n_components=n_components)
    result_store = get_result_store()
    result_id_out = result_store.store(embedding, session_ids=inherited_session_ids)
    return {
        "result_id": result_id_out,
        "shape": list(embedding.shape),
        "dtype": str(embedding.dtype),
        "ttl_seconds": result_store.default_ttl,
    }


async def pca_on_result(
    result_id: str,
    n_components: int = 2,
) -> Dict[str, Any]:
    entry = get_result_store().get(result_id)
    if entry is None:
        raise ValueError(f"Result not found or expired: {result_id}")
    array, meta = entry
    if array.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got {array.ndim}D shape {list(array.shape)}"
        )
    inherited_session_ids = meta.get("session_ids")
    embedding = PCA_reduction(array, n_components=n_components)
    result_store = get_result_store()
    result_id_out = result_store.store(embedding, session_ids=inherited_session_ids)
    return {
        "result_id": result_id_out,
        "shape": list(embedding.shape),
        "dtype": str(embedding.dtype),
        "ttl_seconds": result_store.default_ttl,
    }


async def umap_reduction_on_result(
    result_id: str,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    entry = get_result_store().get(result_id)
    if entry is None:
        raise ValueError(f"Result not found or expired: {result_id}")
    array, meta = entry
    if array.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got {array.ndim}D shape {list(array.shape)}"
        )
    inherited_session_ids = meta.get("session_ids")
    embedding = UMAP_reduction(
        array,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    result_store = get_result_store()
    result_id_out = result_store.store(embedding, session_ids=inherited_session_ids)
    return {
        "result_id": result_id_out,
        "shape": list(embedding.shape),
        "dtype": str(embedding.dtype),
        "ttl_seconds": result_store.default_ttl,
    }


async def umap_graph_communities_on_result(
    result_id: str,
    n_components: int = 2,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    entry = get_result_store().get(result_id)
    if entry is None:
        raise ValueError(f"Result not found or expired: {result_id}")
    array, meta = entry
    if array.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got {array.ndim}D shape {list(array.shape)}"
        )
    inherited_session_ids = meta.get("session_ids")
    embedding, labels = UMAP_graph_communities(
        array,
        n_components=n_components,
        resolution=resolution,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    result_store = get_result_store()
    result_id_out = result_store.store(embedding, session_ids=inherited_session_ids)
    return {
        "result_id": result_id_out,
        "shape": list(embedding.shape),
        "dtype": str(embedding.dtype),
        "ttl_seconds": result_store.default_ttl,
        "labels": labels.tolist(),
    }
