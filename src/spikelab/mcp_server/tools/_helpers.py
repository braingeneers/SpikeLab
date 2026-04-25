"""Shared helpers for MCP tool modules."""

from ...spikedata.spikedata import SpikeData
from ...spikedata.ratedata import RateData
from ...workspace.workspace import get_workspace_manager

SPIKEDATA_KEY = "spikedata"


def get_workspace(workspace_id: str):
    """Get AnalysisWorkspace by ID, raising ValueError if not found."""
    ws = get_workspace_manager().get_workspace(workspace_id)
    if ws is None:
        raise ValueError(f"Workspace not found: {workspace_id}")
    return ws


def resolve_workspace(workspace_id: str, name: str | None = None):
    """
    Get or create a workspace.

    Returns (workspace, workspace_id). Creates a new workspace when
    workspace_id is empty; retrieves an existing one otherwise.
    """
    wm = get_workspace_manager()
    if workspace_id:
        ws = wm.get_workspace(workspace_id)
        if ws is None:
            raise ValueError(f"Workspace not found: {workspace_id}")
        return ws, workspace_id
    new_id = wm.create_workspace(name=name)
    return wm.get_workspace(new_id), new_id


def get_spikedata(ws, namespace: str) -> SpikeData:
    """Load SpikeData from (namespace, 'spikedata') in the workspace.

    Raises ValueError with tool suggestions if not found.
    """
    sd = ws.get(namespace, SPIKEDATA_KEY)
    if sd is None or not isinstance(sd, SpikeData):
        raise ValueError(
            f"No SpikeData found at ({namespace!r}, {SPIKEDATA_KEY!r}). "
            "Load a recording first using one of: "
            "load_from_hdf5_raster, load_from_hdf5_ragged, load_from_hdf5_group, "
            "load_from_hdf5_paired, load_from_nwb, load_from_kilosort, "
            "load_from_pickle, load_from_hdf5_thresholded, load_from_ibl, "
            "load_from_spikelab_sorted_npz."
        )
    return sd


def get_ratedata(ws, namespace: str, key: str) -> RateData:
    """Load RateData from (namespace, key) in the workspace.

    Raises ValueError with tool suggestions if not found.
    """
    rd = ws.get(namespace, key)
    if rd is None or not isinstance(rd, RateData):
        raise ValueError(
            f"No RateData found at ({namespace!r}, {key!r}). "
            "Compute instantaneous firing rates first using: compute_resampled_isi."
        )
    return rd
