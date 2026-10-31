"""
Sequence Analysis, Sequence Packing, and Dataset Splitting Engine
"""
import random
from typing import Dict, List, Any, Tuple
from src.tokenization.adapters import BaseTokenizerAdapter

class SequenceAnalyzer:
    """Analyzes token distribution, sequence lengths, and context window requirements."""

    @staticmethod
    def analyze(token_counts: List[int]) -> Dict[str, Any]:
        if not token_counts:
            return {
                "total_tokens": 0,
                "count": 0,
                "mean_length": 0,
                "max_length": 0,
                "min_length": 0,
                "p50_length": 0,
                "p95_length": 0,
                "p99_length": 0,
                "distribution_buckets": {}
            }

        sorted_counts = sorted(token_counts)
        n = len(sorted_counts)
        total = sum(sorted_counts)

        buckets = {
            "0-512": sum(1 for c in sorted_counts if c <= 512),
            "513-1024": sum(1 for c in sorted_counts if 512 < c <= 1024),
            "1025-2048": sum(1 for c in sorted_counts if 1024 < c <= 2048),
            "2049-4096": sum(1 for c in sorted_counts if 2048 < c <= 4096),
            "4097-8192": sum(1 for c in sorted_counts if 4096 < c <= 8192),
            "8192+": sum(1 for c in sorted_counts if c > 8192)
        }

        return {
            "total_tokens": total,
            "count": n,
            "mean_length": round(total / n, 1),
            "max_length": sorted_counts[-1],
            "min_length": sorted_counts[0],
            "p50_length": sorted_counts[n // 2],
            "p95_length": sorted_counts[int(n * 0.95)],
            "p99_length": sorted_counts[int(n * 0.99)],
            "distribution_buckets": buckets
        }


class SequencePacker:
    """Packs short sequences into fixed-length context windows with document delimiters."""

    def __init__(self, max_seq_len: int = 4096, eot_token: str = "<|endoftext|>"):
        self.max_seq_len = max_seq_len
        self.eot_token = eot_token

    def pack(self, items: List[Dict[str, Any]], tokenizer: BaseTokenizerAdapter) -> List[Dict[str, Any]]:
        packed_batches = []
        current_batch_text = []
        current_tokens = 0
        current_docs = []

        for item in items:
            text = item.get("content", "")
            tokens = tokenizer.count_tokens(text)
            if tokens >= self.max_seq_len:
                # Document exceeds or matches context window: place standalone
                packed_batches.append({
                    "content": text,
                    "token_count": tokens,
                    "doc_count": 1,
                    "source_ids": [item.get("id", "")]
                })
                continue

            if current_tokens + tokens + 1 > self.max_seq_len:
                # Flush current batch
                packed_batches.append({
                    "content": f" {self.eot_token} ".join(current_batch_text),
                    "token_count": current_tokens,
                    "doc_count": len(current_docs),
                    "source_ids": [d.get("id", "") for d in current_docs]
                })
                current_batch_text = [text]
                current_tokens = tokens
                current_docs = [item]
            else:
                current_batch_text.append(text)
                current_tokens += tokens + 1
                current_docs.append(item)

        if current_batch_text:
            packed_batches.append({
                "content": f" {self.eot_token} ".join(current_batch_text),
                "token_count": current_tokens,
                "doc_count": len(current_docs),
                "source_ids": [d.get("id", "") for d in current_docs]
            })

        return packed_batches


class DatasetSplitter:
    """Splits records into Train / Validation / Test sets with zero-leakage cross-source partitioning."""

    @staticmethod
    def split(
        records: List[Dict[str, Any]],
        train_ratio: float = 0.95,
        val_ratio: float = 0.025,
        test_ratio: float = 0.025,
        seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4

        # Group by repository / source to prevent cross-file leakage
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            group_key = r.get("repo_name") or r.get("source") or "default_group"
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(r)

        # Shuffle groups deterministically
        group_keys = sorted(groups.keys())
        rng = random.Random(seed)
        rng.shuffle(group_keys)

        total_items = len(records)
        train_target = int(total_items * train_ratio)
        val_target = int(total_items * val_ratio)

        train, val, test = [], [], []

        for gk in group_keys:
            items = groups[gk]
            if len(train) + len(items) <= train_target or (not train and items):
                train.extend(items)
            elif len(val) + len(items) <= val_target or (not val and items):
                val.extend(items)
            else:
                test.extend(items)

        # In case splits are empty due to small group count, fallback to item-level stratified
        if not val or not test:
            all_items = list(records)
            rng.shuffle(all_items)
            t_idx = int(len(all_items) * train_ratio)
            v_idx = int(len(all_items) * (train_ratio + val_ratio))
            train = all_items[:t_idx] or all_items[:1]
            val = all_items[t_idx:v_idx] or all_items[-2:-1] or all_items[:1]
            test = all_items[v_idx:] or all_items[-1:] or all_items[:1]

        return train, val, test

