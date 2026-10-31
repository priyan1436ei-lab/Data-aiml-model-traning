"""
Exact SHA-256 Deduplication Engine
"""
import hashlib
from typing import Set, Tuple

class ExactDeduplicator:
    """Detects byte-for-byte identical records using SHA-256 hashing."""

    def __init__(self):
        self.seen_hashes: Set[str] = set()

    def is_duplicate(self, text: str) -> Tuple[bool, str]:
        """Check if exact content has been seen. Returns (is_duplicate, sha256_hash)."""
        content_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        if content_hash in self.seen_hashes:
            return True, content_hash
        self.seen_hashes.add(content_hash)
        return False, content_hash

