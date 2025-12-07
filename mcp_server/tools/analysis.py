"""
MCP tools for analyzing spike data.

Exposes comprehensive SpikeData analysis methods as MCP tools.
Uses helper functions to avoid duplicating SpikeData method logic.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union

from mcp_server.sessions import get_session_manager


def _convert_array_to_list(arr: np.ndarray) -> Union[List, Any]:
    """Convert numpy array to list for JSON serialization."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def _get_spikedata(session_id: str):
    """Get SpikeData from session, raising error if not found."""
    session_manager = get_session_manager()
    spikedata = session_manager.get_session(session_id)
    if spikedata is None:
        raise ValueError(f"Session not found: {session_id}")
    return spikedata


def _call_spikedata_method(
    session_id: str, method_name: str, result_key: str, **method_kwargs
) -> Dict[str, Any]:
    """
    Generic helper to call a SpikeData method and return serialized result.

    Args:
        session_id: Session ID
        method_name: Name of SpikeData method to call
        result_key: Key name for result in response dict
        **method_kwargs: Arguments to pass to the method

    Returns:
        Dict with result_key containing serialized result
    """
    spikedata = _get_spikedata(session_id)
    method = getattr(spikedata, method_name)
    result = method(**method_kwargs)
    return {
        result_key: (
            _convert_array_to_list(result) if isinstance(result, np.ndarray) else result
        )
    }


# Analysis tools - these are thin wrappers that call SpikeData methods directly
async def compute_rates(session_id: str, unit: str = "kHz") -> Dict[str, Any]:
    """Calculate the mean firing rate of each neuron."""
    spikedata = _get_spikedata(session_id)
    rates = spikedata.rates(unit=unit)
    return {
        "rates": _convert_array_to_list(rates),
        "unit": unit,
        "num_neurons": len(rates),
    }


async def compute_binned(session_id: str, bin_size: float = 40.0) -> Dict[str, Any]:
    """Get binned spike counts."""
    spikedata = _get_spikedata(session_id)
    binned = spikedata.binned(bin_size=bin_size)
    return {
        "binned_counts": _convert_array_to_list(binned),
        "bin_size_ms": bin_size,
        "num_bins": len(binned),
    }


async def compute_binned_meanrate(
    session_id: str, bin_size: float = 40.0, unit: str = "kHz"
) -> Dict[str, Any]:
    """Calculate the mean firing rate across the population in each time bin."""
    spikedata = _get_spikedata(session_id)
    mean_rates = spikedata.binned_meanrate(bin_size=bin_size, unit=unit)
    return {
        "mean_rates": _convert_array_to_list(mean_rates),
        "bin_size_ms": bin_size,
        "unit": unit,
        "num_bins": len(mean_rates),
    }


async def compute_raster(session_id: str, bin_size: float = 20.0) -> Dict[str, Any]:
    """Generate a dense spike raster matrix."""
    spikedata = _get_spikedata(session_id)
    raster = spikedata.raster(bin_size=bin_size)
    return {
        "raster": _convert_array_to_list(raster),
        "bin_size_ms": bin_size,
        "shape": list(raster.shape),
    }


async def compute_sparse_raster(
    session_id: str, bin_size: float = 20.0
) -> Dict[str, Any]:
    """Generate a sparse spike raster matrix."""
    spikedata = _get_spikedata(session_id)
    sparse_raster = spikedata.sparse_raster(bin_size=bin_size)
    raster = sparse_raster.toarray()
    return {
        "raster": _convert_array_to_list(raster),
        "bin_size_ms": bin_size,
        "shape": list(raster.shape),
    }


async def compute_channel_raster(
    session_id: str,
    bin_size: float = 20.0,
    channel_attr: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a channel-aggregated raster matrix."""
    spikedata = _get_spikedata(session_id)
    channel_raster = spikedata.channel_raster(
        bin_size=bin_size, channel_attr=channel_attr
    )
    return {
        "channel_raster": _convert_array_to_list(channel_raster),
        "bin_size_ms": bin_size,
        "shape": list(channel_raster.shape),
    }


async def compute_interspike_intervals(session_id: str) -> Dict[str, Any]:
    """Calculate interspike intervals for each neuron."""
    spikedata = _get_spikedata(session_id)
    isis = spikedata.interspike_intervals()
    return {
        "isis": [_convert_array_to_list(isi) for isi in isis],
        "num_neurons": len(isis),
    }


async def compute_resampled_isi(
    session_id: str,
    times: List[float],
    sigma_ms: float = 10.0,
) -> Dict[str, Any]:
    """Calculate firing rate at specific times using resampled ISI method."""
    spikedata = _get_spikedata(session_id)
    times_array = np.array(times, dtype=float)
    rates = spikedata.resampled_isi(times_array, sigma_ms=sigma_ms)
    return {
        "rates": _convert_array_to_list(rates),
        "times": times,
        "sigma_ms": sigma_ms,
        "shape": list(rates.shape),
    }


async def subtime(
    session_id: str,
    start: float,
    end: float,
    create_new_session: bool = False,
) -> Dict[str, Any]:
    """Extract a time window from the spike data."""
    session_manager = get_session_manager()
    spikedata = _get_spikedata(session_id)
    subset = spikedata.subtime(start, end)

    if create_new_session:
        new_session_id = session_manager.create_session(subset)
        return {
            "session_id": new_session_id,
            "info": {
                "num_neurons": subset.N,
                "length_ms": subset.length,
                "start_ms": start,
                "end_ms": end,
            },
        }
    else:
        session_manager.update_session(session_id, subset)
        return {
            "session_id": session_id,
            "info": {
                "num_neurons": subset.N,
                "length_ms": subset.length,
                "start_ms": start,
                "end_ms": end,
            },
        }


async def subset(
    session_id: str,
    units: List[int],
    by: Optional[str] = None,
    create_new_session: bool = False,
) -> Dict[str, Any]:
    """Select specific neurons from the spike data."""
    session_manager = get_session_manager()
    spikedata = _get_spikedata(session_id)
    subset = spikedata.subset(units, by=by)

    if create_new_session:
        new_session_id = session_manager.create_session(subset)
        return {
            "session_id": new_session_id,
            "info": {
                "num_neurons": subset.N,
                "length_ms": subset.length,
                "selected_units": units,
            },
        }
    else:
        session_manager.update_session(session_id, subset)
        return {
            "session_id": session_id,
            "info": {
                "num_neurons": subset.N,
                "length_ms": subset.length,
                "selected_units": units,
            },
        }


async def compute_spike_time_tiling(
    session_id: str,
    neuron_i: int,
    neuron_j: int,
    delt: float = 20.0,
) -> Dict[str, Any]:
    """Calculate spike time tiling coefficient between two neurons."""
    spikedata = _get_spikedata(session_id)
    if neuron_i < 0 or neuron_i >= spikedata.N:
        raise ValueError(
            f"Invalid neuron_i: {neuron_i} (valid range: 0-{spikedata.N-1})"
        )
    if neuron_j < 0 or neuron_j >= spikedata.N:
        raise ValueError(
            f"Invalid neuron_j: {neuron_j} (valid range: 0-{spikedata.N-1})"
        )

    sttc = spikedata.spike_time_tiling(neuron_i, neuron_j, delt=delt)
    return {
        "sttc": float(sttc),
        "neuron_i": neuron_i,
        "neuron_j": neuron_j,
        "delt_ms": delt,
    }


async def compute_spike_time_tilings(
    session_id: str, delt: float = 20.0
) -> Dict[str, Any]:
    """Compute the full spike time tiling coefficient matrix for all neuron pairs."""
    spikedata = _get_spikedata(session_id)
    sttc_matrix = spikedata.spike_time_tilings(delt=delt)
    return {
        "sttc_matrix": _convert_array_to_list(sttc_matrix),
        "delt_ms": delt,
        "shape": list(sttc_matrix.shape),
    }


async def compute_latencies(
    session_id: str,
    times: List[float],
    window_ms: float = 100.0,
) -> Dict[str, Any]:
    """Compute latencies from reference times to spikes in each neuron."""
    spikedata = _get_spikedata(session_id)
    times_array = np.array(times, dtype=float)
    latencies = spikedata.latencies(times_array, window_ms=window_ms)
    return {
        "latencies": [[float(l) for l in lat_list] for lat_list in latencies],
        "times": times,
        "window_ms": window_ms,
        "num_neurons": len(latencies),
    }


async def compute_latencies_to_index(
    session_id: str,
    neuron_index: int,
    window_ms: float = 100.0,
) -> Dict[str, Any]:
    """Compute latencies from a specific neuron to all other neurons."""
    spikedata = _get_spikedata(session_id)
    if neuron_index < 0 or neuron_index >= spikedata.N:
        raise ValueError(
            f"Invalid neuron_index: {neuron_index} (valid range: 0-{spikedata.N-1})"
        )

    latencies = spikedata.latencies_to_index(neuron_index, window_ms=window_ms)
    return {
        "latencies": [[float(l) for l in lat_list] for lat_list in latencies],
        "neuron_index": neuron_index,
        "window_ms": window_ms,
        "num_neurons": len(latencies),
    }


async def get_frac_active(
    session_id: str,
    edges: List[List[float]],
    min_spikes: int,
    backbone_threshold: float,
) -> Dict[str, Any]:
    """Calculate fraction of active neurons/units in bursts."""
    spikedata = _get_spikedata(session_id)
    edges_array = np.array(edges, dtype=int)
    frac_per_unit, frac_per_burst, backbone_units = spikedata.get_frac_active(
        edges_array, min_spikes, backbone_threshold
    )

    return {
        "frac_per_unit": _convert_array_to_list(frac_per_unit),
        "frac_per_burst": _convert_array_to_list(frac_per_burst),
        "backbone_units": _convert_array_to_list(backbone_units),
        "num_bursts": len(edges),
    }


async def get_data_info(session_id: str) -> Dict[str, Any]:
    """Get information about the spike data in a session."""
    spikedata = _get_spikedata(session_id)
    info = {
        "num_neurons": spikedata.N,
        "length_ms": spikedata.length,
        "metadata": spikedata.metadata,
        "has_neuron_attributes": spikedata.neuron_attributes is not None,
        "has_raw_data": hasattr(spikedata, "raw_data") and spikedata.raw_data.size > 0,
    }

    if spikedata.neuron_attributes:
        info["neuron_attributes_sample"] = (
            str(spikedata.neuron_attributes[0])
            if len(spikedata.neuron_attributes) > 0
            else None
        )

    return info


async def list_neurons(session_id: str) -> Dict[str, Any]:
    """List available neurons with their attributes."""
    spikedata = _get_spikedata(session_id)
    neurons = []
    for i in range(spikedata.N):
        neuron_info = {
            "index": i,
            "num_spikes": len(spikedata.train[i]),
        }
        if spikedata.neuron_attributes and i < len(spikedata.neuron_attributes):
            # Convert attributes to dict if possible
            attrs = spikedata.neuron_attributes[i]
            if hasattr(attrs, "__dict__"):
                neuron_info["attributes"] = attrs.__dict__
            else:
                neuron_info["attributes"] = str(attrs)
        neurons.append(neuron_info)

    return {
        "neurons": neurons,
        "num_neurons": len(neurons),
    }
