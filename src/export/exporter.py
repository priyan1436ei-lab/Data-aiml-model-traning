"""
Multi-Format Exporter (JSONL, Parquet, Arrow)
High-performance serialization into Parquet and Arrow formats using Polars.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import polars as pl
from src.config import PATHS, init_workspace

logger = logging.getLogger(__name__)

class DatasetExporter:
    """Exports datasets into Parquet, Arrow, and JSONL formats."""

    @staticmethod
    def jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> bool:
        """Converts a JSONL file to optimized compressed Parquet using Polars."""
        if not jsonl_path.exists():
            return False
        try:
            # Read records safely
            records = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            if not records:
                return False

            # Convert non-primitive nested structures to JSON strings for consistent Parquet schema
            flat_records = []
            for r in records:
                flat_r = {}
                for k, v in r.items():
                    if isinstance(v, (dict, list)):
                        flat_r[k] = json.dumps(v)
                    else:
                        flat_r[k] = v
                flat_records.append(flat_r)

            df = pl.DataFrame(flat_records)
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(parquet_path, compression="zstd")
            return True
        except Exception as e:
            logger.error(f"Failed to export {jsonl_path} to Parquet: {e}")
            return False

    @staticmethod
    def export_all_corpora():
        """Convert all JSONL files in train, validation, test, and pretraining into Parquet."""
        init_workspace()
        target_dirs = [
            PATHS["train"],
            PATHS["validation"],
            PATHS["test"],
            PATHS["pretraining_source_code"],
            PATHS["sft_code_generation"],
            PATHS["sft_code_explanation"],
            PATHS["sft_code_debugging"],
            PATHS["completion_fim"],
            PATHS["reasoning"],
            PATHS["repository_tasks"],
            PATHS["preference"],
            PATHS["agent_trajectories"],
            PATHS["high_quality"]
        ]

        exported_count = 0
        for directory in target_dirs:
            for jsonl_file in directory.glob("*.jsonl"):
                parquet_file = jsonl_file.with_suffix(".parquet")
                if DatasetExporter.jsonl_to_parquet(jsonl_file, parquet_file):
                    exported_count += 1

        print(f"[Export] Successfully generated {exported_count} Parquet files across training corpora.")
        return exported_count

