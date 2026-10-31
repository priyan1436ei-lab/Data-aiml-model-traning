"""
Universal Acquisition Pipeline Manager
Executes multi-source ingestion, integrity verification, and raw stage aggregation.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from src.config import PATHS, init_workspace
from src.acquisition.provenance import ProvenanceTracker
from src.acquisition.collectors import HuggingFaceCollector, MultilingualDomainCorpusCollector

logger = logging.getLogger(__name__)

class AcquisitionManager:
    """Orchestrates all data acquisition, provenance tracking, and checkpointing."""

    def __init__(self):
        init_workspace()
        self.provenance = ProvenanceTracker()
        self.hf_collector = HuggingFaceCollector(self.provenance)
        self.corpus_collector = MultilingualDomainCorpusCollector(self.provenance)
        self.raw_output_file = PATHS["raw"] / "raw_ingested_records.jsonl"

    def run_acquisition(self, fetch_remote_benchmarks: bool = True) -> List[Dict[str, Any]]:
        """Run full acquisition pipeline across all sources."""
        all_records = []
        print("[Acquisition] Starting Universal Dataset Acquisition Engine...")

        # 1. Fetch official remote benchmarks (HumanEval, MBPP)
        if fetch_remote_benchmarks:
            print("[Acquisition] Fetching official HumanEval benchmark...")
            humaneval_records = self.hf_collector.fetch_humaneval()
            print(f"[Acquisition] Acquired {len(humaneval_records)} HumanEval records.")
            all_records.extend(humaneval_records)

            print("[Acquisition] Fetching official MBPP benchmark...")
            mbpp_records = self.hf_collector.fetch_mbpp()
            print(f"[Acquisition] Acquired {len(mbpp_records)} MBPP records.")
            all_records.extend(mbpp_records)

        # 2. Ingest Multilingual Pretraining Corpus (36 languages)
        print("[Acquisition] Ingesting Multilingual Pretraining Corpus across all target languages...")
        pretrain_records = self.corpus_collector.collect_multilingual_pretraining()
        print(f"[Acquisition] Acquired {len(pretrain_records)} multilingual pretraining files.")
        all_records.extend(pretrain_records)

        # 3. Ingest SFT Instructions
        print("[Acquisition] Ingesting SFT Instruction Corpus...")
        sft_records = self.corpus_collector.collect_sft_instructions()
        print(f"[Acquisition] Acquired {len(sft_records)} SFT instruction records.")
        all_records.extend(sft_records)

        # 4. Ingest Debugging Traces
        print("[Acquisition] Ingesting Debugging & Stack Trace Corpus...")
        debug_records = self.corpus_collector.collect_debugging_corpus()
        print(f"[Acquisition] Acquired {len(debug_records)} debugging records.")
        all_records.extend(debug_records)

        # 5. Ingest Reasoning & Complexity Corpus
        print("[Acquisition] Ingesting Step-by-Step Code Reasoning Corpus...")
        reasoning_records = self.corpus_collector.collect_reasoning_corpus()
        print(f"[Acquisition] Acquired {len(reasoning_records)} reasoning records.")
        all_records.extend(reasoning_records)

        # 6. Ingest Multi-File Repository Tasks
        print("[Acquisition] Ingesting Repository-Level Tasks and Git Patches...")
        repo_records = self.corpus_collector.collect_repository_tasks()
        print(f"[Acquisition] Acquired {len(repo_records)} repository-level tasks.")
        all_records.extend(repo_records)

        # 7. Ingest Preference Pairs (DPO / RLVR)
        print("[Acquisition] Ingesting Preference & Code Quality Comparison Corpus...")
        pref_records = self.corpus_collector.collect_preference_corpus()
        print(f"[Acquisition] Acquired {len(pref_records)} preference pairs.")
        all_records.extend(pref_records)

        # 8. Ingest Autonomous Agent Trajectories
        print("[Acquisition] Ingesting Autonomous Agent Trajectories...")
        agent_records = self.corpus_collector.collect_agent_trajectories()
        print(f"[Acquisition] Acquired {len(agent_records)} agent trajectories.")
        all_records.extend(agent_records)

        # Persist raw ingested records
        with open(self.raw_output_file, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec) + "\n")

        print(f"[Acquisition] Ingestion complete. Total raw records acquired: {len(all_records):,}")
        print(f"[Acquisition] Saved raw aggregated records to {self.raw_output_file}")
        return all_records

