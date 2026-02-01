from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DesignDoc(BaseModel):
    title: str
    content: str
    milestones: List[str]
    approved: bool = False


class SwarmState(BaseModel):
    objective: str
    user_id: Optional[str] = None
    current_step: str = "research"
    research_notes: Optional[str] = None
    design_doc: Optional[DesignDoc] = None
    implementation_status: Dict[str, str] = Field(default_factory=dict)
    test_results: Optional[str] = None
    human_feedback: List[str] = Field(default_factory=list)
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list, description="Recent chat history with the user"
    )
    artifacts_path: str = "agent_swarm/artifacts/"

    # New Improvements
    blackboard: Dict[str, Any] = Field(
        default_factory=dict, description="Shared space for inter-agent communication"
    )
    artifact_registry: List[Dict[str, str]] = Field(
        default_factory=list, description="Log of all saved artifacts"
    )

    def update_step(self, step: str):
        self.current_step = step

    def add_feedback(self, feedback: str):
        self.human_feedback.append(feedback)

    def register_artifact(self, filename: str, description: str):
        self.artifact_registry.append(
            {"filename": filename, "description": description}
        )
        # Automatically update blackboard with new artifact info
        self.blackboard[f"artifact_{filename}"] = description

    def update_blackboard(self, key: str, value: Any):
        self.blackboard[key] = value
