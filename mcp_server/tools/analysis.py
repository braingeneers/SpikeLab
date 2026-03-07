"""
MCP tools for analyzing spike data.

Provides async wrappers around SpikeData analysis methods.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from mcp_server.sessions import get_session_manager
from spikedata.ratedata import RateData
from spikedata.rateslicestack import RateSliceStack
from spikedata.spikeslicestack import SpikeSliceStack
from spikedata.utils import (
    compute_cosine_similarity_with_lag,
    compute_cross_correlation_with_lag,
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


async def compute_rates(
    session_id: str,
    unit: str = "kHz",
) -> Dict[str, Any]:
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
    rates = sd.rates(unit=unit)
    return {"rates": _to_list(rates), "unit": unit}


async def compute_binned(
    session_id: str,
    bin_size: float = 40.0,
) -> Dict[str, Any]:
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
    binned = sd.binned(bin_size=bin_size)
    return {"binned": _to_list(binned), "bin_size": bin_size}


async def compute_binned_meanrate(
    session_id: str,
    bin_size: float = 40.0,
    unit: str = "kHz",
) -> Dict[str, Any]:
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
    meanrate = sd.binned_meanrate(bin_size=bin_size, unit=unit)
    return {"meanrate": _to_list(meanrate), "bin_size": bin_size, "unit": unit}


async def compute_raster(
    session_id: str,
    bin_size: float = 20.0,
) -> Dict[str, Any]:
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
    raster = sd.channel_raster(bin_size=bin_size, channel_attr=channel_attr)
    return {
        "raster": _to_list(raster),
        "shape": list(raster.shape),
        "bin_size": bin_size,
    }


async def compute_interspike_intervals(
    session_id: str,
) -> Dict[str, Any]:
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
    isis = sd.interspike_intervals()
    return {"isis": [_to_list(isi) for isi in isis]}


async def compute_resampled_isi(
    session_id: str,
    times: List[float],
    sigma_ms: float = 10.0,
) -> Dict[str, Any]:
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
    new_sd = sd.subtime(start, end)
    if create_new_session:
        new_session_id = session_manager.create_session(new_sd)
    else:
        session_manager.update_session(session_id, new_sd)
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
    new_sd = sd.subset(units, by=by)
    if create_new_session:
        new_session_id = session_manager.create_session(new_sd)
    else:
        session_manager.update_session(session_id, new_sd)
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
    return {
        "num_neurons": sd.N,
        "length_ms": sd.length,
        "metadata": sd.metadata,
    }


async def list_neurons(
    session_id: str,
) -> Dict[str, Any]:
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
    rate_matrix = sd.resampled_isi(times=np.array(times), sigma_ms=sigma_ms)
    rd = RateData(rate_matrix, np.array(times))
    corr_matrix, lag_matrix = rd.get_pairwise_fr_corr(max_lag=max_lag)
    return {
        "corr_matrix": _to_list(corr_matrix),
        "lag_matrix": _to_list(lag_matrix),
        "shape": list(corr_matrix.shape),
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    return {
        "embedding": _to_list(embedding),
        "times": times,
        "shape": list(embedding.shape),
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    return {
        "corr_stack": _to_list(pcm_stack.stack),
        "av_corr": _to_list(av_corr),
        "shape": list(pcm_stack.stack.shape),
    }


async def compute_rate_slice_time_corr(
    session_id: str,
    times_start_to_end: List[List[float]],
    sigma_ms: float = 10.0,
    max_lag: int = 0,
    compare_func: str = "cosine_similarity",
) -> Dict[str, Any]:
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
    if compare_func not in _COMPARE_FUNCS:
        raise ValueError(f"compare_func must be one of {list(_COMPARE_FUNCS.keys())}")
    time_tuples = [tuple(t) for t in times_start_to_end]
    rss = RateSliceStack(sd, times_start_to_end=time_tuples, sigma_ms=sigma_ms)
    pcm_stack, av_corr = rss.get_slice_to_slice_time_corr_from_stack(
        compare_func=_COMPARE_FUNCS[compare_func],
        max_lag=max_lag,
    )
    return {
        "corr_stack": _to_list(pcm_stack.stack),
        "av_corr": _to_list(av_corr),
        "shape": list(pcm_stack.stack.shape),
    }


async def compute_unit_to_unit_slice_corr(
    session_id: str,
    times_start_to_end: List[List[float]],
    sigma_ms: float = 10.0,
    max_lag: int = 10,
    compare_func: str = "cross_correlation",
) -> Dict[str, Any]:
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
    if compare_func not in _COMPARE_FUNCS:
        raise ValueError(f"compare_func must be one of {list(_COMPARE_FUNCS.keys())}")
    time_tuples = [tuple(t) for t in times_start_to_end]
    rss = RateSliceStack(sd, times_start_to_end=time_tuples, sigma_ms=sigma_ms)
    corr_stack, lag_stack, av_max_corr, av_max_corr_lag = rss.unit_to_unit_correlation(
        compare_func=_COMPARE_FUNCS[compare_func],
        max_lag=max_lag,
    )
    return {
        "corr_stack": _to_list(corr_stack.stack),
        "lag_stack": _to_list(lag_stack.stack),
        "av_max_corr": _to_list(av_max_corr),
        "av_max_corr_lag": _to_list(av_max_corr_lag),
        "shape": list(corr_stack.stack.shape),
    }


async def compute_rate_slice_unit_order(
    session_id: str,
    times_start_to_end: List[List[float]],
    sigma_ms: float = 10.0,
    agg_func: str = "median",
    min_rate_threshold: float = 0.1,
) -> Dict[str, Any]:
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
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
    session_manager = get_session_manager()
    sd = session_manager.get_session(session_id)
    if sd is None:
        raise ValueError("Session not found")
    time_tuples = [tuple(t) for t in times_start_to_end]
    sss = SpikeSliceStack(sd, times_start_to_end=time_tuples)
    sparse_list = [
        spike_obj_slice.sparse_raster(bin_size=bin_size)
        for spike_obj_slice in sss.spike_stack
    ]
    sparse_stack = np.stack(sparse_list, axis=2)
    return {
        "sparse_matrix": _to_list(sparse_stack),
        "shape": list(sparse_stack.shape),
        "bin_size_ms": bin_size,
    }
