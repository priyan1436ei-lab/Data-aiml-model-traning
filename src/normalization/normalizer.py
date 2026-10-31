"""
Training Data Normalization and Routing Engine
Transforms deduplicated records into canonical training schemas and distributes them into dedicated corpora.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.config import PATHS, FIM_TOKENS, init_workspace
from src.normalization.schemas import (
    SFTRecord, PretrainingRecord, FIMRecord, DebuggingRecord,
    ReasoningRecord, RepositoryTaskRecord, PreferenceRecord, AgentTrajectoryRecord
)

logger = logging.getLogger(__name__)

class TrainingCorpusNormalizer:
    """Normalizes and routes records into training-specific subdirectories."""

    def __init__(self):
        init_workspace()
        self.dedup_file = PATHS["deduplicated"] / "deduplicated_records.jsonl"

    def run_normalization(self) -> Dict[str, int]:
        """Normalize and route deduplicated records into target training directories."""
        print("[Normalization] Normalizing and distributing records into training corpora...")
        if not self.dedup_file.exists():
            print(f"[Normalization] Deduplicated file {self.dedup_file} not found. Run deduplicate first.")
            return {}

        counts = {
            "pretraining": 0,
            "sft": 0,
            "fim": 0,
            "debugging": 0,
            "reasoning": 0,
            "repository_tasks": 0,
            "agent_trajectories": 0,
            "preference": 0,
            "benchmarks": 0
        }

        # Writers for each corpus
        f_pretrain = open(PATHS["pretraining_source_code"] / "pretraining_corpus.jsonl", "w", encoding="utf-8")
        f_sft_gen = open(PATHS["sft_code_generation"] / "sft_generation_corpus.jsonl", "w", encoding="utf-8")
        f_sft_exp = open(PATHS["sft_code_explanation"] / "sft_explanation_corpus.jsonl", "w", encoding="utf-8")
        f_fim = open(PATHS["completion_fim"] / "fim_corpus.jsonl", "w", encoding="utf-8")
        f_debug = open(PATHS["sft_code_debugging"] / "debugging_corpus.jsonl", "w", encoding="utf-8")
        f_reason = open(PATHS["reasoning"] / "reasoning_corpus.jsonl", "w", encoding="utf-8")
        f_repo = open(PATHS["repository_tasks"] / "repository_tasks_corpus.jsonl", "w", encoding="utf-8")
        f_agent = open(PATHS["agent_trajectories"] / "agent_trajectories_corpus.jsonl", "w", encoding="utf-8")
        f_pref = open(PATHS["preference"] / "preference_corpus.jsonl", "w", encoding="utf-8")
        f_bench = open(PATHS["benchmarks"] / "benchmarks_corpus.jsonl", "w", encoding="utf-8")

        try:
            with open(self.dedup_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    t_type = record.get("training_type", "pretraining")

                    if t_type == "pretraining":
                        norm_rec = PretrainingRecord(
                            content=record["content"],
                            language=record.get("language", "python"),
                            domain=record.get("domain", "general"),
                            source=record.get("source", "open-source"),
                            license=record.get("license", "mit"),
                            repo_name=record.get("repo_name", "corpus"),
                            path=record.get("path", "")
                        )
                        f_pretrain.write(norm_rec.model_dump_json() + "\n")
                        counts["pretraining"] += 1

                        # Generate FIM variant for pretraining code
                        fim_rec = self._create_fim_transformation(record)
                        if fim_rec:
                            f_fim.write(fim_rec.model_dump_json() + "\n")
                            counts["fim"] += 1

                    elif t_type == "sft":
                        messages = record.get("messages", [])
                        task_type = record.get("task_type", "code_generation")
                        norm_rec = SFTRecord(
                            messages=messages,
                            language=record.get("language", "python"),
                            task_type=task_type,
                            domain=record.get("domain", "algorithms"),
                            source=record.get("source", "sft-corpus"),
                            license=record.get("license", "apache-2.0")
                        )
                        if task_type == "code_explanation":
                            f_sft_exp.write(norm_rec.model_dump_json() + "\n")
                        else:
                            f_sft_gen.write(norm_rec.model_dump_json() + "\n")
                        counts["sft"] += 1

                    elif t_type == "debugging":
                        norm_rec = DebuggingRecord(
                            problem=record.get("problem", ""),
                            code_before=record.get("code_before", ""),
                            analysis=record.get("analysis", ""),
                            code_after=record.get("code_after", ""),
                            tests=record.get("tests", []),
                            language=record.get("language", "python"),
                            source=record.get("source", "debugging-archive"),
                            license=record.get("license", "mit")
                        )
                        f_debug.write(norm_rec.model_dump_json() + "\n")
                        counts["debugging"] += 1

                    elif t_type == "reasoning":
                        norm_rec = ReasoningRecord(
                            problem=record.get("problem", ""),
                            reasoning_steps=record.get("reasoning_steps", ""),
                            complexity=record.get("complexity", {}),
                            solution=record.get("solution", ""),
                            language=record.get("language", "python"),
                            source=record.get("source", "reasoning-corpus"),
                            license=record.get("license", "apache-2.0")
                        )
                        f_reason.write(norm_rec.model_dump_json() + "\n")
                        counts["reasoning"] += 1

                    elif t_type == "repository_tasks":
                        norm_rec = RepositoryTaskRecord(
                            repo_name=record.get("repo_name", ""),
                            base_commit=record.get("base_commit", ""),
                            issue=record.get("issue", ""),
                            patch=record.get("patch", ""),
                            tests=record.get("tests", []),
                            language=record.get("language", "python"),
                            source=record.get("source", "swe-bench"),
                            license=record.get("license", "mit")
                        )
                        f_repo.write(norm_rec.model_dump_json() + "\n")
                        counts["repository_tasks"] += 1

                    elif t_type == "preference":
                        norm_rec = PreferenceRecord(
                            prompt=record.get("prompt", ""),
                            chosen=record.get("chosen", ""),
                            rejected=record.get("rejected", ""),
                            reason=record.get("reason", ""),
                            source=record.get("source", "feedback-dpo"),
                            license=record.get("license", "mit"),
                            language=record.get("language", "python")
                        )
                        f_pref.write(norm_rec.model_dump_json() + "\n")
                        counts["preference"] += 1

                    elif t_type == "agent_trajectories":
                        norm_rec = AgentTrajectoryRecord(
                            goal=record.get("goal", ""),
                            system_prompt=record.get("system_prompt", ""),
                            trajectory=record.get("trajectory", []),
                            final_status=record.get("final_status", "success"),
                            domain=record.get("domain", "autonomous_software_engineering"),
                            source=record.get("source", "agent-bench"),
                            license=record.get("license", "apache-2.0")
                        )
                        f_agent.write(norm_rec.model_dump_json() + "\n")
                        counts["agent_trajectories"] += 1

                    elif t_type == "benchmark":
                        f_bench.write(json.dumps(record) + "\n")
                        counts["benchmarks"] += 1

        finally:
            f_pretrain.close()
            f_sft_gen.close()
            f_sft_exp.close()
            f_fim.close()
            f_debug.close()
            f_reason.close()
            f_repo.close()
            f_agent.close()
            f_pref.close()
            f_bench.close()

        print("[Normalization] Distribution complete:")
        for k, v in counts.items():
            print(f"  - {k}: {v:,} records")

        return counts

    def _create_fim_transformation(self, record: Dict[str, Any]) -> Optional[FIMRecord]:
        """Transform code snippet into Fill-In-The-Middle (PSM) training format."""
        code = record.get("content", "")
        lines = code.split("\n")
        if len(lines) < 6:
            return None

        # Split into prefix, middle, suffix
        p_end = len(lines) // 3
        m_end = 2 * (len(lines) // 3)

        prefix = "\n".join(lines[:p_end]) + "\n"
        middle = "\n".join(lines[p_end:m_end]) + "\n"
        suffix = "\n".join(lines[m_end:])

        # PSM prompt: <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|>
        formatted_prompt = (
            f"{FIM_TOKENS['prefix']}{prefix}"
            f"{FIM_TOKENS['suffix']}{suffix}"
            f"{FIM_TOKENS['middle']}{middle}{FIM_TOKENS['eot']}"
        )

        return FIMRecord(
            prefix=prefix,
            middle=middle,
            suffix=suffix,
            fim_type="PSM",
            formatted_prompt=formatted_prompt,
            language=record.get("language", "python"),
            source=record.get("source", "fim-transformation"),
            license=record.get("license", "mit")
        )
