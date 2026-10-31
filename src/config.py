"""
Universal Global Coding Dataset - Global Configuration
Author: Senior AI Research Scientist & Dataset Architect
"""
import os
from pathlib import Path
from typing import Dict, List, Set

# Base Workspace Directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_ROOT = BASE_DIR / "universal-global-coding-dataset"

# Output Directory Hierarchy
PATHS: Dict[str, Path] = {
    "root": DATASET_ROOT,
    "raw": DATASET_ROOT / "raw",
    "metadata": DATASET_ROOT / "metadata",
    "manifests": DATASET_ROOT / "manifests",
    # Pretraining Corpora
    "pretraining": DATASET_ROOT / "pretraining",
    "pretraining_source_code": DATASET_ROOT / "pretraining" / "source_code",
    "pretraining_docs": DATASET_ROOT / "pretraining" / "documentation",
    "pretraining_repos": DATASET_ROOT / "pretraining" / "repositories",
    "pretraining_config": DATASET_ROOT / "pretraining" / "configuration",
    # SFT Corpora
    "sft": DATASET_ROOT / "sft",
    "sft_code_generation": DATASET_ROOT / "sft" / "code_generation",
    "sft_code_explanation": DATASET_ROOT / "sft" / "code_explanation",
    "sft_code_translation": DATASET_ROOT / "sft" / "code_translation",
    "sft_code_debugging": DATASET_ROOT / "sft" / "code_debugging",
    "sft_software_engineering": DATASET_ROOT / "sft" / "software_engineering",
    # Completion Corpora
    "completion": DATASET_ROOT / "completion",
    "completion_fim": DATASET_ROOT / "completion" / "fim",
    "completion_prefix": DATASET_ROOT / "completion" / "prefix_completion",
    "completion_context": DATASET_ROOT / "completion" / "context_completion",
    # Advanced Training Corpora
    "reasoning": DATASET_ROOT / "reasoning",
    "repository_tasks": DATASET_ROOT / "repository_tasks",
    "agent_trajectories": DATASET_ROOT / "agent_trajectories",
    "preference": DATASET_ROOT / "preference",
    "benchmarks": DATASET_ROOT / "benchmarks",
    # Processing Tiers
    "cleaned": DATASET_ROOT / "cleaned",
    "deduplicated": DATASET_ROOT / "deduplicated",
    "high_quality": DATASET_ROOT / "high_quality",
    "quarantine": DATASET_ROOT / "quarantine",
    "quarantine_secrets": DATASET_ROOT / "quarantine" / "secrets",
    "quarantine_malware": DATASET_ROOT / "quarantine" / "malware",
    "quarantine_contaminated": DATASET_ROOT / "quarantine" / "contaminated",
    "rejected": DATASET_ROOT / "rejected",
    # Tokenized & Partitions
    "tokenized": DATASET_ROOT / "tokenized",
    "train": DATASET_ROOT / "train",
    "validation": DATASET_ROOT / "validation",
    "test": DATASET_ROOT / "test",
    # Operations
    "reports": DATASET_ROOT / "reports",
    "logs": DATASET_ROOT / "logs",
    "scripts": DATASET_ROOT / "scripts",
    "configs": DATASET_ROOT / "configs",
    "tests": DATASET_ROOT / "tests",
}

def init_workspace():
    """Ensure all required directories exist."""
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)

# Supported Programming Languages (Comprehensive coverage - 36+ languages)
SUPPORTED_LANGUAGES: List[str] = [
    "python", "javascript", "typescript", "java", "c", "cpp", "csharp", "go",
    "rust", "php", "ruby", "swift", "kotlin", "dart", "scala", "r", "julia",
    "matlab", "perl", "lua", "elixir", "haskell", "ocaml", "clojure", "objc",
    "assembly", "solidity", "cuda", "sql", "html", "css", "shell", "bash",
    "powershell", "verilog", "vhdl"
]

# Language File Extensions Mapping
LANGUAGE_EXTENSIONS: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".dart": "dart",
    ".scala": "scala",
    ".r": "r",
    ".jl": "julia",
    ".m": "matlab",
    ".pl": "perl",
    ".pm": "perl",
    ".lua": "lua",
    ".ex": "elixir",
    ".exs": "elixir",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".clj": "clojure",
    ".s": "assembly",
    ".asm": "assembly",
    ".sol": "solidity",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".sh": "shell",
    ".bash": "bash",
    ".ps1": "powershell",
    ".v": "verilog",
    ".vhd": "vhdl",
    ".vhdl": "vhdl",
}

# Software Engineering Knowledge Domains (35 complete domains)
CODING_DOMAINS: List[str] = [
    "basic_programming",
    "advanced_programming",
    "data_structures",
    "algorithms",
    "competitive_programming",
    "object_oriented_programming",
    "functional_programming",
    "design_patterns",
    "system_design",
    "software_architecture",
    "frontend_development",
    "backend_development",
    "fullstack_development",
    "mobile_development",
    "web_development",
    "api_development",
    "database_engineering",
    "cloud_computing",
    "devops",
    "distributed_systems",
    "machine_learning_engineering",
    "deep_learning_engineering",
    "ai_application_development",
    "data_engineering",
    "cybersecurity",
    "secure_coding",
    "testing_and_debugging",
    "performance_optimization",
    "code_refactoring",
    "code_documentation",
    "repository_understanding",
    "multi_file_code_generation",
    "bug_fixing",
    "real_world_issue_resolution",
    "autonomous_software_engineering",
]

# Permitted & Tracked Open Source Licenses
PERMISSIVE_LICENSES: Set[str] = {
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense",
    "cc0-1.0", "cc-by-4.0", "artistic-2.0", "zlib", "boost-1.0"
}
COPYLEFT_LICENSES: Set[str] = {
    "gpl-2.0", "gpl-3.0", "lgpl-2.1", "lgpl-3.0", "agpl-3.0", "mpl-2.0", "epl-2.0"
}
RESTRICTED_LICENSES: Set[str] = {
    "cc-by-nc-4.0", "cc-by-nc-nd-4.0", "proprietary", "custom", "unknown"
}

# Quality Engine Thresholds
QUALITY_THRESHOLDS = {
    "min_line_count": 3,
    "max_line_count": 10000,
    "min_alpha_ratio": 0.25,
    "max_line_length_mean": 120,
    "max_line_length_absolute": 1000,
    "premium_score_cutoff": 0.85,
    "standard_score_cutoff": 0.65,
    "minhash_num_perm": 128,
    "minhash_threshold": 0.85,
    "ngram_decontamination_size": 13,
}

# FIM Special Tokens (StarCoder / DeepSeek-Coder compatible)
FIM_TOKENS = {
    "prefix": "<|fim_prefix|>",
    "suffix": "<|fim_suffix|>",
    "middle": "<|fim_middle|>",
    "eot": "<|endoftext|>",
    "file_separator": "<|filename|>",
}

