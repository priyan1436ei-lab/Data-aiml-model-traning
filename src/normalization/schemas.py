"""
Canonical LLM Training Schemas
Pydantic schemas conforming to SFT (ChatML), Pretraining, FIM, Debugging, Reasoning, Repo Tasks, and DPO Preference.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class SFTMessage(BaseModel):
    role: str = Field(..., description="user, assistant, or system")
    content: str = Field(..., description="Message text or code")

class SFTRecord(BaseModel):
    messages: List[SFTMessage]
    language: str
    task_type: str = "code_generation"
    domain: Optional[str] = None
    source: str
    license: str

class PretrainingRecord(BaseModel):
    content: str
    language: str
    file_type: str = "source_code"
    domain: Optional[str] = None
    source: str
    license: str
    repo_name: Optional[str] = None
    path: Optional[str] = None

class FIMRecord(BaseModel):
    prefix: str
    middle: str
    suffix: str
    fim_type: str = "PSM"  # PSM (Prefix-Suffix-Middle) or SPM (Suffix-Prefix-Middle)
    formatted_prompt: str
    language: str
    source: str
    license: str

class DebuggingRecord(BaseModel):
    problem: str
    code_before: str
    analysis: str
    code_after: str
    tests: List[str] = Field(default_factory=list)
    language: str
    source: str
    license: Optional[str] = "mit"

class ReasoningRecord(BaseModel):
    problem: str
    reasoning_steps: str
    complexity: Dict[str, str] = Field(default_factory=dict)
    solution: str
    language: str
    source: str
    license: Optional[str] = "apache-2.0"

class RepositoryTaskRecord(BaseModel):
    repo_name: str
    base_commit: str
    issue: str
    patch: str
    tests: List[str] = Field(default_factory=list)
    language: str
    source: str
    license: Optional[str] = "mit"

class PreferenceRecord(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    reason: str
    source: str
    license: Optional[str] = "mit"
    language: Optional[str] = "python"

class AgentTrajectoryRecord(BaseModel):
    goal: str
    system_prompt: str
    trajectory: List[Dict[str, Any]]
    final_status: str
    domain: str = "autonomous_software_engineering"
    source: str
    license: Optional[str] = "apache-2.0"

