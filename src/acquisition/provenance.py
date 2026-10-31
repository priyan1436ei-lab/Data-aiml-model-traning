"""
Cryptographic Provenance and Ingestion Ledger
Tracks the lineage, license, source URL, and SHA-256 hash of every acquired dataset and record.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from src.config import PATHS

class ProvenanceTracker:
    """Maintains an append-only cryptographic provenance ledger."""

    def __init__(self, ledger_path: Optional[Path] = None):
        self.ledger_path = ledger_path or (PATHS["manifests"] / "provenance_ledger.jsonl")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def record_ingestion(
        self,
        record_id: str,
        source_id: str,
        source_url: str,
        license_str: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Calculate SHA-256 and append record to provenance ledger."""
        sha256_hash = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()
        entry = {
            "record_id": record_id,
            "source_id": source_id,
            "source_url": source_url,
            "license": license_str,
            "sha256": sha256_hash,
            "byte_size": len(content.encode("utf-8", errors="ignore")),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

