"""
Normalized Text Deduplication Engine
Strips comments, docstrings, and whitespace to catch formatting variations.
"""
import hashlib
import re
from typing import Set, Tuple

class NormalizedDeduplicator:
    """Normalizes code by stripping comments and standardizing whitespace."""

    def __init__(self):
        self.seen_normalized_hashes: Set[str] = set()

    @staticmethod
    def normalize_code(code: str) -> str:
        """Strip comments, quotes, and collapse whitespace."""
        # 1. Remove multiline docstrings
        code = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)
        # 2. Remove block comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # 3. Remove single-line comments
        code = re.sub(r'(#|//|--).*?$', '', code, flags=re.MULTILINE)
        # 4. Standardize quotes
        code = code.replace("'", '"')
        # 5. Collapse whitespace and newlines
        code = re.sub(r'\s+', ' ', code).strip()
        return code

    def is_duplicate(self, text: str) -> Tuple[bool, str]:
        """Check if normalized content is duplicate. Returns (is_duplicate, norm_hash)."""
        normalized = self.normalize_code(text)
        norm_hash = hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()
        if norm_hash in self.seen_normalized_hashes:
            return True, norm_hash
        self.seen_normalized_hashes.add(norm_hash)
        return False, norm_hash

