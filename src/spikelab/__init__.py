"""SpikeLab — spike train analysis library."""

from .spikedata import SpikeData, RateData, RateSliceStack, SpikeSliceStack
from .spikedata.pairwise import PairwiseCompMatrix, PairwiseCompMatrixStack
from .workspace import AnalysisWorkspace, WorkspaceManager, get_workspace_manager

__version__ = "0.1.0"

__all__ = [
    "SpikeData",
    "RateData",
    "RateSliceStack",
    "SpikeSliceStack",
    "PairwiseCompMatrix",
    "PairwiseCompMatrixStack",
    "AnalysisWorkspace",
    "WorkspaceManager",
    "get_workspace_manager",
]
