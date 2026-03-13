"""
Raster plotting: spike raster + optional population rate + optional per-unit rate heatmap.

Supports:
- SpikeData.plot_raster(...) when IntegratedAnalysisTools is on PYTHONPATH.
- plot_raster_from_workspace(workspace_id, namespace, save_path, ...) to load
  SpikeData (and optional RateData) from a workspace and plot.
- Run from the IntegratedAnalysisTools directory: python viz/new_plot_raster.py (or set PYTHONPATH to IAT).

Original TODOs from old_plots/original_plot_raster.py (addressed in IAT plot_utils.plot_raster + SpikeData.plot_raster):
  (1) MAKE THIS A SPIKEDATA METHOD (SCRIPT IN plot_utils.py) -> spikedata/spikedata.py SpikeData.plot_raster(); spikedata/plot_utils.plot_raster().
  (2) MAKE FONT SIZE SPECIFIC TO FIGURE -> font_size on ax only, no rcParams.
  (3) ALLOW fr_rates COMPUTED FROM SPIKEDATA -> fr_rate_bin_ms in SpikeData.plot_raster().
  (4) PRECOMPUTED POP_RATE; USER CHOOSE PARAMS -> pop_rate optional; pop_rate_square_width, pop_rate_gauss_sigma.
  (5) LET USER CHOOSE FIGURE SIZE AND SUBPLOT RATIOS -> figsize, height_ratios.
  (6) SEPARATE EVENT PERIODS FROM EVENT TIMES; LIST OF (START, END); CLIP TO time_range -> event_periods.
  (7) USE FUNCTION FROM plot_heatmap FOR fr_rates -> plot_av_rate() for rate subplot.
  (8) LET USER PASS vmin AND vmax (NONE BY DEFAULT) for rate heatmap -> fr_vmin, fr_vmax.
  (9) SET X LIMITS FOR ALL AXES FROM time_range -> ax.set_xlim(0, n_bins).
  (10) X-TICK LABELS: time_range[0]..time_range[1] OR 0..duration -> time_axis_absolute.
  (11) RETURN FIGURE; ALLOW NOT PRINT OR SAVE -> returns (fig, axes); save_path, show_fig.
"""

from typing import Any, List, Optional, Tuple

try:
    from spikedata import SpikeData
    from spikedata.plot_utils import plot_raster
    _HAS_IAT = True
except ImportError:
    _HAS_IAT = False
    SpikeData = None  # type: ignore
    plot_raster = None  # type: ignore


def plot_raster_from_workspace(
    workspace_id: str,
    namespace: str,
    save_path: str,
    start_ms: Optional[float] = None,
    end_ms: Optional[float] = None,
    n_units: Optional[int] = None,
    unit_indices: Optional[List[int]] = None,
    raster_bin_size_ms: float = 1.0,
    rate_key: Optional[str] = None,
    fr_rate_bin_ms: Optional[float] = None,
    pop_rate_square_width: int = 5,
    pop_rate_gauss_sigma: int = 5,
    event_times_ms: Optional[List[float]] = None,
    event_periods: Optional[List[Tuple[float, float]]] = None,
    figsize: Tuple[float, float] = (12, 6),
    font_size: int = 14,
    time_axis_absolute: bool = True,
    fr_vmin: Optional[float] = None,
    fr_vmax: Optional[float] = None,
    **kwargs: Any,
) -> Tuple[Optional[Any], List[Any]]:
    """
    Load SpikeData (and optional RateData) from an IntegratedAnalysisTools workspace
    and plot raster + optional population rate + optional rate heatmap; save to save_path.

    Requires workspace package and spikedata (run with IntegratedAnalysisTools on
    PYTHONPATH, or from that repo).

    Parameters
    ----------
    workspace_id, namespace : str
        Workspace ID and recording namespace (e.g. from load_from_pickle / load_from_hdf5).
    save_path : str
        Path to save the figure (e.g. .png).
    start_ms, end_ms : float, optional
        Time window to plot.
    n_units, unit_indices : optional
        Restrict to first n units or to given unit indices.
    raster_bin_size_ms : float
        Bin size in ms for the spike raster.
    rate_key : str, optional
        If set, load RateData from (namespace, rate_key) and use as fr_rates for
        the bottom heatmap (e.g. key from compute_resampled_isi).
    fr_rate_bin_ms : float, optional
        If rate_key is not set and this is set, compute binned rate matrix from
        spikes for the bottom heatmap.
    pop_rate_square_width, pop_rate_gauss_sigma : int
        Parameters for get_pop_rate when pop_rate is not precomputed.
    event_times_ms, event_periods : optional
        Event markers and shaded periods (see plot_raster).
    **kwargs
        Passed to SpikeData.plot_raster.

    Returns
    -------
    fig, axes
    """
    if not _HAS_IAT:
        raise ImportError(
            "plot_raster_from_workspace requires IntegratedAnalysisTools (spikedata). "
            "Run from the IntegratedAnalysisTools directory or set PYTHONPATH to it."
        )
    from workspace.workspace import get_workspace_manager

    wm = get_workspace_manager()
    ws = wm.get_workspace(workspace_id)
    if ws is None:
        raise ValueError(f"Workspace not found: {workspace_id}")

    sd = ws.get(namespace, "spikedata")
    if sd is None:
        raise ValueError(
            f"No SpikeData at ({namespace!r}, 'spikedata'). "
            "Load data first (e.g. load_from_pickle, load_from_hdf5)."
        )
    if not isinstance(sd, SpikeData):
        raise TypeError(f"Item (namespace={namespace!r}, key='spikedata') is not SpikeData.")

    if start_ms is not None or end_ms is not None:
        t_start = start_ms if start_ms is not None else 0.0
        t_end = end_ms if end_ms is not None else float(sd.length)
        sd = sd.subtime(t_start, t_end)
    if unit_indices is not None:
        sd = sd.subset(unit_indices)
    elif n_units is not None:
        n = min(n_units, sd.N)
        sd = sd.subset(list(range(n)))

    fr_rates = None
    if rate_key is not None:
        rd = ws.get(namespace, rate_key)
        if rd is not None and hasattr(rd, "inst_Frate_data"):
            if start_ms is not None or end_ms is not None:
                t_start = start_ms if start_ms is not None else 0.0
                t_end = end_ms if end_ms is not None else float(sd.length)
                rd = rd.subtime(t_start, t_end, shift_time=True)
            if unit_indices is not None:
                rd = rd.subset(unit_indices)
            elif n_units is not None:
                n = min(n_units, rd.N)
                rd = rd.subset(list(range(n)))
            fr_rates = rd.inst_Frate_data

    time_range = None
    if start_ms is not None or end_ms is not None:
        t_start = start_ms if start_ms is not None else 0.0
        t_end = end_ms if end_ms is not None else float(sd.length)
        time_range = (t_start, t_end)

    return sd.plot_raster(
        time_range=time_range,
        raster_bin_size_ms=raster_bin_size_ms,
        fr_rates=fr_rates,
        fr_rate_bin_ms=fr_rate_bin_ms,
        pop_rate_square_width=pop_rate_square_width,
        pop_rate_gauss_sigma=pop_rate_gauss_sigma,
        event_times_ms=event_times_ms,
        event_periods=event_periods,
        figsize=figsize,
        font_size=font_size,
        time_axis_absolute=time_axis_absolute,
        fr_vmin=fr_vmin,
        fr_vmax=fr_vmax,
        save_path=save_path,
        show_fig=False,
        **kwargs,
    )


if __name__ == "__main__":
    if not _HAS_IAT:
        print("Run with IntegratedAnalysisTools on PYTHONPATH to use plot_raster / plot_raster_from_workspace.")
        print("Example: cd IntegratedAnalysisTools && python viz/new_plot_raster.py")
    else:
        print("plot_raster and SpikeData.plot_raster are available.")
        print("Example: plot_raster_from_workspace('my_ws', 'rec1', 'raster.png', rate_key='ratedata')")
