"""
Discovery Data Models for Dataset Inventory and Metadata
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class TrainingCategory(str, Enum):
    PRETRAINING = "pretraining"
    SFT = "sft"
    COMPLETION = "completion"
    DEBUGGING = "debugging"
    REASONING = "reasoning"
    REPOSITORY_TASKS = "repository_tasks"
    AGENT_TRAJECTORIES = "agent_trajectories"
    PREFERENCE = "preference"
    BENCHMARK = "benchmark"

class LicenseCategory(str, Enum):
    PERMISSIVE = "permissive"
    COPYLEFT = "copyleft"
    NON_COMMERCIAL = "non_commercial"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"

class RedistributionPermission(str, Enum):
    ALLOW = "allow"
    ALLOW_WITH_ATTRIBUTION = "allow_with_attribution"
    NON_COMMERCIAL_ONLY = "non_commercial_only"
    RESTRICTED = "restricted"
    FORBIDDEN = "forbidden"

class DatasetSource(BaseModel):
    dataset_id: str = Field(..., description="Unique dataset identifier, e.g. bigcode/the-stack-v2")
    name: str = Field(..., description="Display name of the dataset")
    source_url: str = Field(..., description="Official URL or repository link")
    publisher: str = Field(..., description="Author, consortium or publishing organization")
    license: str = Field(..., description="SPDX license string or identifier")
    license_category: LicenseCategory = Field(default=LicenseCategory.PERMISSIVE)
    access_permissions: str = Field(default="public", description="public, gated, authenticated, etc.")
    redistribution_permissions: RedistributionPermission = Field(default=RedistributionPermission.ALLOW_WITH_ATTRIBUTION)
    languages: List[str] = Field(default_factory=list, description="List of programming languages")
    domains: List[str] = Field(default_factory=list, description="List of SE domains covered")
    dataset_format: str = Field(default="parquet", description="parquet, jsonl, arrow, git, etc.")
    example_count: int = Field(default=0, description="Total number of code examples/records")
    storage_size_bytes: int = Field(default=0, description="Approximate raw storage size in bytes")
    token_count: Optional[int] = Field(default=None, description="Reported or estimated token count")
    collection_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    version: str = Field(default="1.0.0")
    training_type: TrainingCategory = Field(default=TrainingCategory.PRETRAINING)
    quality_information: str = Field(default="Standard open-source curation", description="Information regarding data filtering and curation")
    known_limitations: List[str] = Field(default_factory=list, description="Limitations, biases, or constraints")
    is_curated: bool = Field(default=True)
    verification_status: str = Field(default="verified")
    source_ecosystem: str = Field(default="huggingface", description="huggingface, github, kaggle, academic, zenodo, software_heritage")

    def to_dict(self) -> Dict:
        return self.model_dump()

