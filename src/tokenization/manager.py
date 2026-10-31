"""
Tokenization and Model Preparation Manager
Orchestrates token counting, distribution profiling, sequence packing, and train/val/test splitting.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from src.config import PATHS, init_workspace
from src.tokenization.adapters import FastByteBPEAdapter, HuggingFaceTokenizerAdapter
from src.tokenization.sequence_analyzer import SequenceAnalyzer, SequencePacker, DatasetSplitter

logger = logging.getLogger(__name__)

class TokenizationManager:
    """Manages token counting, context length profiling, sequence packing, and dataset splitting."""

    def __init__(self, tokenizer_model: str = "fast_bpe"):
        init_workspace()
        if tokenizer_model == "fast_bpe":
            self.tokenizer = FastByteBPEAdapter()
        else:
            self.tokenizer = HuggingFaceTokenizerAdapter(tokenizer_model)

        self.dedup_file = PATHS["deduplicated"] / "deduplicated_records.jsonl"
        self.train_file = PATHS["train"] / "train.jsonl"
        self.val_file = PATHS["validation"] / "validation.jsonl"
        self.test_file = PATHS["test"] / "test.jsonl"
        self.packed_file = PATHS["tokenized"] / "packed_train_context_4096.jsonl"
        self.report_file = PATHS["reports"] / "tokenization_report.json"

    def run_tokenization(self) -> Dict[str, Any]:
        """Execute full tokenization, profiling, packing, and splitting pipeline."""
        print("[Tokenization] Starting Tokenization and Sequence Preparation Engine...")
        if not self.dedup_file.exists():
            print(f"[Tokenization] Deduplicated file {self.dedup_file} not found. Run deduplicate first.")
            return {}

        records = []
        token_counts = []
        language_tokens: Dict[str, int] = {}
        category_tokens: Dict[str, int] = {}

        with open(self.dedup_file, "r", encoding="utf-8") as f_in:
            for line in f_in:
                if not line.strip():
                    continue
                record = json.loads(line)
                content = record.get("content", "")
                tokens = self.tokenizer.count_tokens(content)

                record["token_count"] = tokens
                records.append(record)
                token_counts.append(tokens)

                # Track per language tokens
                lang = record.get("language", "generic_code")
                language_tokens[lang] = language_tokens.get(lang, 0) + tokens

                # Track per training category tokens
                cat = record.get("training_type", "pretraining")
                category_tokens[cat] = category_tokens.get(cat, 0) + tokens

        # 1. Sequence Length Analysis
        seq_analysis = SequenceAnalyzer.analyze(token_counts)

        # 2. Sequence Packing (Context window 4096)
        packer = SequencePacker(max_seq_len=4096)
        packed_batches = packer.pack(records, self.tokenizer)
        with open(self.packed_file, "w", encoding="utf-8") as f_packed:
            for b in packed_batches:
                f_packed.write(json.dumps(b) + "\n")

        # 3. Train / Validation / Test Splitting
        train, val, test = DatasetSplitter.split(records, train_ratio=0.90, val_ratio=0.05, test_ratio=0.05)

        with open(self.train_file, "w", encoding="utf-8") as f_tr:
            for r in train:
                f_tr.write(json.dumps(r) + "\n")

        with open(self.val_file, "w", encoding="utf-8") as f_vl:
            for r in val:
                f_vl.write(json.dumps(r) + "\n")

        with open(self.test_file, "w", encoding="utf-8") as f_ts:
            for r in test:
                f_ts.write(json.dumps(r) + "\n")

        report = {
            "total_records": len(records),
            "total_tokens": seq_analysis["total_tokens"],
            "sequence_statistics": seq_analysis,
            "language_token_distribution": dict(sorted(language_tokens.items(), key=lambda x: x[1], reverse=True)),
            "category_token_distribution": dict(sorted(category_tokens.items(), key=lambda x: x[1], reverse=True)),
            "splits": {
                "train_records": len(train),
                "train_tokens": sum(r["token_count"] for r in train),
                "val_records": len(val),
                "val_tokens": sum(r["token_count"] for r in val),
                "test_records": len(test),
                "test_tokens": sum(r["token_count"] for r in test)
            },
            "packed_context_batches": len(packed_batches)
        }

        with open(self.report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"[Tokenization] Total tokens measured: {seq_analysis['total_tokens']:,}")
        print(f"[Tokenization] Mean sequence length: {seq_analysis['mean_length']} tokens | Max: {seq_analysis['max_length']} tokens")
        print(f"[Tokenization] Splits -> Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
        print(f"[Tokenization] Sequence packed batches saved to {self.packed_file}")
        return report

