"""
Quality Pipeline Orchestrator
Processes raw records, applies quality filters, segregates tiers, and updates audit records.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from src.config import PATHS, init_workspace
from src.quality.classifier import QualityAssessmentEngine, QualityTier

logger = logging.getLogger(__name__)

class QualityManager:
    """Manages full cleaning, quality filtering, and quarantine segregation."""

    def __init__(self):
        init_workspace()
        self.engine = QualityAssessmentEngine()
        self.raw_file = PATHS["raw"] / "raw_ingested_records.jsonl"
        self.cleaned_file = PATHS["cleaned"] / "cleaned_records.jsonl"
        self.high_quality_file = PATHS["high_quality"] / "premium_records.jsonl"
        self.quarantine_secrets_file = PATHS["quarantine_secrets"] / "quarantined_secrets.jsonl"
        self.quarantine_malware_file = PATHS["quarantine_malware"] / "quarantined_malware.jsonl"
        self.quarantine_contaminated_file = PATHS["quarantine_contaminated"] / "quarantined_contaminated.jsonl"
        self.rejected_file = PATHS["rejected"] / "rejected_records.jsonl"

    def run_cleaning_and_quality_assessment(self) -> Dict[str, Any]:
        """Process all raw records through the quality engine."""
        print("[Quality] Starting Data Quality Assessment & Cleaning Pipeline...")
        if not self.raw_file.exists():
            print(f"[Quality] Raw input file {self.raw_file} not found. Please run collect first.")
            return {}

        tier_counts = {t.value: 0 for t in QualityTier}
        cleaned_records = []
        premium_records = []

        # Open writer files
        f_clean = open(self.cleaned_file, "w", encoding="utf-8")
        f_prem = open(self.high_quality_file, "w", encoding="utf-8")
        f_sec = open(self.quarantine_secrets_file, "w", encoding="utf-8")
        f_mal = open(self.quarantine_malware_file, "w", encoding="utf-8")
        f_cont = open(self.quarantine_contaminated_file, "w", encoding="utf-8")
        f_rej = open(self.rejected_file, "w", encoding="utf-8")

        total_processed = 0

        try:
            with open(self.raw_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    total_processed += 1

                    tier, score, audit = self.engine.evaluate_record(record)
                    tier_counts[tier.value] += 1

                    record["quality_tier"] = tier.value
                    record["quality_score"] = score
                    record_line = json.dumps(record) + "\n"

                    if tier in [QualityTier.PREMIUM, QualityTier.STANDARD, QualityTier.UNVERIFIED]:
                        f_clean.write(record_line)
                        cleaned_records.append(record)
                        if tier == QualityTier.PREMIUM:
                            f_prem.write(record_line)
                            premium_records.append(record)
                    elif tier == QualityTier.QUARANTINE_SECRETS:
                        f_sec.write(record_line)
                    elif tier == QualityTier.QUARANTINE_MALWARE:
                        f_mal.write(record_line)
                    elif tier == QualityTier.QUARANTINE_CONTAMINATED:
                        f_cont.write(record_line)
                    elif tier in [QualityTier.REJECTED_SYNTAX, QualityTier.REJECTED_EMPTY]:
                        f_rej.write(record_line)

        finally:
            f_clean.close()
            f_prem.close()
            f_sec.close()
            f_mal.close()
            f_cont.close()
            f_rej.close()

        summary = {
            "total_processed": total_processed,
            "total_cleaned": len(cleaned_records),
            "total_premium": len(premium_records),
            "tier_distribution": tier_counts,
            "pass_rate_percentage": round(len(cleaned_records) / total_processed * 100, 2) if total_processed > 0 else 0.0
        }

        # Save summary report
        summary_path = PATHS["reports"] / "quality_assessment_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[Quality] Processed: {total_processed} | Cleaned: {len(cleaned_records)} ({summary['pass_rate_percentage']}%) | Premium: {len(premium_records)}")
        print(f"[Quality] Tier breakdown: {tier_counts}")
        print(f"[Quality] Cleaned corpus saved to: {self.cleaned_file}")
        return summary

