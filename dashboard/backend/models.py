"""Pydantic models for the Auto App Generation pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    DONE = "done"


class PipelineStep(str, Enum):
    ENVISIONING = "envisioning"
    BLUEPRINTING = "blueprinting"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"
    FEEDBACK = "feedback"


class LogPrefix(str, Enum):
    SONNET = "SNT"
    OPUS = "OPS"
    SYSTEM = "SYS"
    USER = "USR"
    ERROR = "ERR"


class PipelineState(BaseModel):
    current_step: PipelineStep = PipelineStep.ENVISIONING
    steps: dict[str, StepStatus] = Field(default_factory=lambda: {
        "envisioning": StepStatus.WAITING,
        "blueprinting": StepStatus.WAITING,
        "implementation": StepStatus.WAITING,
        "review": StepStatus.WAITING,
        "feedback": StepStatus.WAITING,
    })

    def advance_to(self, step: PipelineStep) -> None:
        # Mark all previous steps as done
        step_order = list(PipelineStep)
        target_idx = step_order.index(step)
        for i, s in enumerate(step_order):
            if i < target_idx:
                self.steps[s.value] = StepStatus.DONE
            elif i == target_idx:
                self.steps[s.value] = StepStatus.ACTIVE
            # Leave future steps as-is (waiting)
        self.current_step = step


class LogEntry(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    prefix: LogPrefix
    content: str
    agent: str  # "sonnet" or "claude"


class StartRequest(BaseModel):
    prompt: str
    project_name: Optional[str] = None


class AgentMessage(BaseModel):
    agent: str  # "sonnet" or "claude"
    message: str


class FileNode(BaseModel):
    name: str
    type: str = "file"  # "file" or "directory"
    children: list[FileNode] = Field(default_factory=list)
    is_new: bool = False


class Artifact(BaseModel):
    title: str
    description: str
    file_path: str
    size: str
    created_at: str
    created_by: str  # "sonnet" or "claude"
    icon_type: str = "code"  # "md", "code", "review"
