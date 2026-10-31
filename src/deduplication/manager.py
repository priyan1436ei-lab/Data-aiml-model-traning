"""
Deduplication Pipeline Orchestrator
Applies multi-stage exact, normalized, and MinHash LSH deduplication across cross-source corpora.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from src.config import PATHS, init_workspace
from src.deduplication.exact_dedup import ExactDeduplicator
from src.deduplication.normalized_dedup import NormalizedDeduplicator
from src.deduplication.minhash_lsh import MinHashLSHDeduplicator

logger = logging.getLogger(__name__)

class DeduplicationManager:
    """Coordinates multi-stage exact, normalized, and fuzzy near-deduplication."""

    def __init__(self):
        init_workspace()
        self.exact_engine = ExactDeduplicator()
        self.normalized_engine = NormalizedDeduplicator()
        self.minhash_engine = MinHashLSHDeduplicator()
        self.cleaned_file = PATHS["cleaned"] / "cleaned_records.jsonl"
        self.dedup_output_file = PATHS["deduplicated"] / "deduplicated_records.jsonl"
        self.report_file = PATHS["reports"] / "deduplication_summary.json"

    def run_deduplication(self) -> Dict[str, Any]:
        """Execute full deduplication pipeline."""
        print("[Deduplication] Starting Multi-Stage Deduplication Engine...")
        if not self.cleaned_file.exists():
            print(f"[Deduplication] Input cleaned file {self.cleaned_file} does not exist. Run clean first.")
            return {}

        total_input = 0
        exact_duplicates = 0
        normalized_duplicates = 0
        fuzzy_duplicates = 0
        retained_records = []

        with open(self.cleaned_file, "r", encoding="utf-8") as f_in, \
             open(self.dedup_output_file, "w", encoding="utf-8") as f_out:

            for line in f_in:
                if not line.strip():
                    continue
                record = json.loads(line)
                total_input += 1
                content = record.get("content", "")
                rec_id = record.get("id", f"rec_{total_input}")

                # 1. Exact SHA-256 Deduplication
                is_exact, exact_hash = self.exact_engine.is_duplicate(content)
                if is_exact:
                    exact_duplicates += 1
                    continue

                # 2. Normalized Text Deduplication
                is_norm, norm_hash = self.normalized_engine.is_duplicate(content)
                if is_norm:
                    normalized_duplicates += 1
                    continue

                # 3. MinHash LSH Fuzzy Near-Deduplication (Threshold >= 0.85)
                is_fuzzy, orig_id, sim = self.minhash_engine.is_near_duplicate(rec_id, content)
                if is_fuzzy:
                    fuzzy_duplicates += 1
                    continue

                # Retained
                record["sha256"] = exact_hash
                record["normalized_hash"] = norm_hash
                f_out.write(json.dumps(record) + "\n")
                retained_records.append(record)

        total_duplicates = exact_duplicates + normalized_duplicates + fuzzy_duplicates
        reduction_pct = round((total_duplicates / total_input) * 100, 2) if total_input > 0 else 0.0

        summary = {
            "total_input_records": total_input,
            "retained_unique_records": len(retained_records),
            "exact_duplicates_removed": exact_duplicates,
            "normalized_duplicates_removed": normalized_duplicates,
            "fuzzy_near_duplicates_removed": fuzzy_duplicates,
            "total_duplicates_removed": total_duplicates,
            "duplicate_reduction_percentage": reduction_pct
        }

        with open(self.report_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[Deduplication] Input: {total_input} | Unique Retained: {len(retained_records)} | Removed: {total_duplicates} ({reduction_pct}%)")
        print(f"[Deduplication] Exact dupes: {exact_duplicates} | Normalized dupes: {normalized_duplicates} | Fuzzy dupes: {fuzzy_duplicates}")
        print(f"[Deduplication] Output saved to: {self.dedup_output_file}")
        return summary
