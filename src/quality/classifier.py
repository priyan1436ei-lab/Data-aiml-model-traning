"""
Quality Assessment Engine & Tier Classifier
Classifies records into Premium, Standard, Unverified, Quarantined, and Rejected tiers with full audit trail.
"""
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from src.config import PATHS, QUALITY_THRESHOLDS
from src.quality.syntax import SyntaxValidator
from src.quality.readability import ReadabilityEvaluator
from src.quality.security_scanner import SecurityScanner
from src.quality.decontamination import BenchmarkDecontaminationEngine

class QualityTier(str, Enum):
    PREMIUM = "premium"
    STANDARD = "standard"
    UNVERIFIED = "unverified"
    QUARANTINE_SECRETS = "quarantine_secrets"
    QUARANTINE_MALWARE = "quarantine_malware"
    QUARANTINE_CONTAMINATED = "quarantine_contaminated"
    REJECTED_SYNTAX = "rejected_syntax"
    REJECTED_EMPTY = "rejected_empty"

class QualityAssessmentEngine:
    """Evaluates syntactic validity, readability, secrets, malware, and decontamination."""

    def __init__(self):
        self.decontamination_engine = BenchmarkDecontaminationEngine(
            ngram_size=QUALITY_THRESHOLDS["ngram_decontamination_size"]
        )
        self.audit_log_path = PATHS["logs"] / "quality_audit.jsonl"
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate_record(self, record: Dict[str, Any]) -> Tuple[QualityTier, float, Dict[str, Any]]:
        """
        Evaluate a single dataset record.
        Returns (QualityTier, composite_quality_score, audit_details).
        """
        content = record.get("content", "")
        language = record.get("language", "generic_code")
        training_type = record.get("training_type", "pretraining")
        record_id = record.get("id", "unknown")

        # 1. Check Empty/Corrupted
        if not content or not content.strip():
            audit = {"reason": "Empty content"}
            self._log_audit(record_id, QualityTier.REJECTED_EMPTY, 0.0, audit)
            return QualityTier.REJECTED_EMPTY, 0.0, audit

        # 2. Security Scan (Secrets & Malware)
        is_clean, violations, violation_cat = SecurityScanner.scan(content)
        if not is_clean:
            if violation_cat == "SECRETS":
                tier = QualityTier.QUARANTINE_SECRETS
            else:
                tier = QualityTier.QUARANTINE_MALWARE
            audit = {"violations": violations, "category": violation_cat}
            self._log_audit(record_id, tier, 0.0, audit)
            return tier, 0.0, audit

        # 3. Benchmark Decontamination Check
        # Only quarantine if it's meant for pretraining or SFT, not if it's a benchmark itself!
        if training_type in ["pretraining", "sft", "completion"]:
            is_contaminated, matching_ngrams = self.decontamination_engine.check_contamination(content)
            if is_contaminated:
                tier = QualityTier.QUARANTINE_CONTAMINATED
                audit = {"contaminated_ngrams": matching_ngrams}
                self._log_audit(record_id, tier, 0.0, audit)
                return tier, 0.0, audit

        # 4. Syntax Validation
        # For structured training types (debugging, reasoning, preference, SFT), extract the actual target code
        code_to_validate = content
        if training_type == "debugging" and record.get("code_after"):
            code_to_validate = record.get("code_after")
        elif training_type == "reasoning" and record.get("solution"):
            code_to_validate = record.get("solution")
        elif training_type == "preference" and record.get("chosen"):
            code_to_validate = record.get("chosen")
        elif training_type == "repository_tasks":
            code_to_validate = None  # Unified diff / multi-file patch

        if code_to_validate is not None:
            is_syntax_valid, syntax_err = SyntaxValidator.validate(code_to_validate, language)
        else:
            is_syntax_valid, syntax_err = True, None

        if not is_syntax_valid:
            tier = QualityTier.REJECTED_SYNTAX
            audit = {"syntax_error": syntax_err}
            self._log_audit(record_id, tier, 0.1, audit)
            return tier, 0.1, audit

        # 5. Readability & Code Metrics
        readability = ReadabilityEvaluator.evaluate(content)
        readability_score = readability["readability_score"]

        # Composite Score Calculation
        composite_score = readability_score
        if is_syntax_valid:
            composite_score = min(1.0, composite_score + 0.1)

        # Tier Assignment
        if composite_score >= QUALITY_THRESHOLDS["premium_score_cutoff"]:
            tier = QualityTier.PREMIUM
        elif composite_score >= QUALITY_THRESHOLDS["standard_score_cutoff"]:
            tier = QualityTier.STANDARD
        else:
            tier = QualityTier.UNVERIFIED

        audit = {
            "readability": readability,
            "syntax_valid": is_syntax_valid,
            "composite_score": composite_score
        }
        self._log_audit(record_id, tier, composite_score, audit)
        return tier, composite_score, audit

    def _log_audit(self, record_id: str, tier: QualityTier, score: float, details: Dict):
        entry = {
            "record_id": record_id,
            "tier": tier.value,
            "score": round(score, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details
        }
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
