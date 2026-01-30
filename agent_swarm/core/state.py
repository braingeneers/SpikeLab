from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DesignDoc(BaseModel):
    title: str
    content: str
    milestones: List[str]
    approved: bool = False


class SwarmState(BaseModel):
    objective: str
    current_step: str = "research"
    research_notes: Optional[str] = None
    design_doc: Optional[DesignDoc] = None
    implementation_status: Dict[str, str] = Field(default_factory=dict)
    test_results: Optional[str] = None
    human_feedback: List[str] = Field(default_factory=list)
    artifacts_path: str = "agent_swarm/artifacts/"

    def update_step(self, step: str):
        self.current_step = step

    def add_feedback(self, feedback: str):
        self.human_feedback.append(feedback)
