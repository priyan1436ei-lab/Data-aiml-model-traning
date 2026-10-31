"""
MinHash and Locality-Sensitive Hashing (LSH) Near-Deduplication Engine
Industrial implementation of token shingle MinHash for code similarity at threshold >= 0.85.
"""
import hashlib
import random
import re
from typing import Dict, List, Set, Tuple, Optional

class MinHashLSHDeduplicator:
    """MinHash LSH index for near-duplicate code detection."""

    def __init__(self, num_perm: int = 128, threshold: float = 0.85, num_bands: int = 16):
        self.num_perm = num_perm
        self.threshold = threshold
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self.prime = 4294967311  # Large 32-bit prime

        # Precompute deterministic pseudo-random hash coefficients
        rng = random.Random(42)
        self.a = [rng.randint(1, self.prime - 1) for _ in range(num_perm)]
        self.b = [rng.randint(0, self.prime - 1) for _ in range(num_perm)]

        # LSH Hash Buckets: band_idx -> bucket_key -> list of (doc_id, signature)
        self.buckets: Dict[int, Dict[str, List[Tuple[str, List[int]]]]] = {
            i: {} for i in range(num_bands)
        }
        self.signatures: Dict[str, List[int]] = {}

    def _tokenize(self, text: str) -> Set[str]:
        """Extract word unigrams and bigrams for robust code similarity."""
        words = re.findall(r"\w+", text.lower())
        if not words:
            return {text}
        tokens = set(words)
        if len(words) >= 2:
            tokens.update(" ".join(words[i:i + 2]) for i in range(len(words) - 1))
        return tokens

    def compute_signature(self, text: str) -> List[int]:
        """Compute MinHash signature vector of length num_perm."""
        tokens = self._tokenize(text)
        if not tokens:
            return [0] * self.num_perm

        token_hashes = [
            int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
            for t in tokens
        ]

        sig = []
        for i in range(self.num_perm):
            min_val = self.prime
            a_i, b_i = self.a[i], self.b[i]
            for h in token_hashes:
                val = (a_i * h + b_i) % self.prime
                if val < min_val:
                    min_val = val
            sig.append(min_val)
        return sig

    def is_near_duplicate(self, doc_id: str, text: str) -> Tuple[bool, Optional[str], float]:
        """
        Check if document is near-duplicate with any indexed document.
        Returns (is_duplicate, duplicate_of_id, estimated_jaccard).
        """
        sig = self.compute_signature(text)
        candidate_ids = set()

        # Query bands
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band_hashes = tuple(sig[start:start + self.rows_per_band])
            bucket_key = hashlib.md5(str(band_hashes).encode("utf-8")).hexdigest()

            if bucket_key in self.buckets[band_idx]:
                for match_id, _ in self.buckets[band_idx][bucket_key]:
                    candidate_ids.add(match_id)

        # Check Jaccard similarity for candidates
        for cand_id in candidate_ids:
            cand_sig = self.signatures[cand_id]
            matches = sum(1 for s1, s2 in zip(sig, cand_sig) if s1 == s2)
            similarity = matches / self.num_perm
            if similarity >= self.threshold:
                return True, cand_id, similarity

        # Not duplicate: index into LSH
        self.signatures[doc_id] = sig
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band_hashes = tuple(sig[start:start + self.rows_per_band])
            bucket_key = hashlib.md5(str(band_hashes).encode("utf-8")).hexdigest()
            if bucket_key not in self.buckets[band_idx]:
                self.buckets[band_idx][bucket_key] = []
            self.buckets[band_idx][bucket_key].append((doc_id, sig))

        return False, None, 0.0
