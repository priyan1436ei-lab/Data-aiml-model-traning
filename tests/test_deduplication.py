"""
Unit Tests for Deduplication, Tokenization, Packing, and Parquet Export
"""
import pytest
from pathlib import Path
from src.deduplication.exact_dedup import ExactDeduplicator
from src.deduplication.normalized_dedup import NormalizedDeduplicator
from src.deduplication.minhash_lsh import MinHashLSHDeduplicator
from src.tokenization.adapters import FastByteBPEAdapter
from src.tokenization.sequence_analyzer import SequenceAnalyzer, SequencePacker, DatasetSplitter
from src.export.exporter import DatasetExporter

def test_exact_deduplication():
    engine = ExactDeduplicator()
    code1 = "def foo(): return 1\n"
    code2 = "def foo(): return 1\n"
    code3 = "def bar(): return 2\n"

    is_dupe1, _ = engine.is_duplicate(code1)
    assert is_dupe1 is False

    is_dupe2, _ = engine.is_duplicate(code2)
    assert is_dupe2 is True

    is_dupe3, _ = engine.is_duplicate(code3)
    assert is_dupe3 is False

def test_normalized_deduplication():
    engine = NormalizedDeduplicator()
    code1 = "def foo():\n    # inline comment\n    return 'hello'\n"
    code2 = 'def foo():\n    return "hello"\n'

    is_dupe1, _ = engine.is_duplicate(code1)
    assert is_dupe1 is False

    is_dupe2, _ = engine.is_duplicate(code2)
    assert is_dupe2 is True  # Comment removed, quotes standardized

def test_minhash_lsh_fuzzy_dedup():
    engine = MinHashLSHDeduplicator(threshold=0.60, num_bands=32)
    base = """
    function computeMetrics(data) {
        let total = 0;
        for (let i = 0; i < data.length; i++) {
            total += data[i].value;
        }
        return total / data.length;
    }
    """
    # Slightly modified variable names
    variant = """
    function computeMetrics(items) {
        let total = 0;
        for (let i = 0; i < items.length; i++) {
            total += items[i].value;
        }
        return total / items.length;
    }
    """
    distinct = """
    function quicksort(arr) {
        if (arr.length <= 1) return arr;
        let pivot = arr[0];
        let left = arr.slice(1).filter(x => x < pivot);
        let right = arr.slice(1).filter(x => x >= pivot);
        return [...quicksort(left), pivot, ...quicksort(right)];
    }
    """
    is_dupe1, _, _ = engine.is_near_duplicate("doc1", base)
    assert is_dupe1 is False

    is_dupe2, orig_id, sim = engine.is_near_duplicate("doc2", variant)
    assert is_dupe2 is True
    assert orig_id == "doc1"
    assert sim >= 0.60

    is_dupe3, _, _ = engine.is_near_duplicate("doc3", distinct)
    assert is_dupe3 is False

def test_fast_byte_bpe_tokenization():
    tokenizer = FastByteBPEAdapter()
    text = "def factorial(n: int) -> int:\n    return 1 if n <= 1 else n * factorial(n - 1)\n"
    tokens = tokenizer.count_tokens(text)
    assert tokens > 10
    assert tokens < 50

def test_sequence_packer():
    tokenizer = FastByteBPEAdapter()
    packer = SequencePacker(max_seq_len=100)
    items = [
        {"id": "doc1", "content": "x = 10\ny = 20\n"},
        {"id": "doc2", "content": "a = 'apple'\nb = 'banana'\n"},
        {"id": "doc3", "content": "def test(): return True\n"}
    ]
    packed = packer.pack(items, tokenizer)
    assert len(packed) >= 1
    assert packed[0]["doc_count"] >= 2
    assert "<|endoftext|>" in packed[0]["content"]

def test_dataset_splitter():
    records = [{"id": f"rec_{i}", "source": f"repo_{i // 5}", "content": "print(1)"} for i in range(50)]
    train, val, test = DatasetSplitter.split(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert len(train) + len(val) + len(test) == 50

def test_parquet_export(tmp_path):
    jsonl_file = tmp_path / "test.jsonl"
    parquet_file = tmp_path / "test.parquet"

    with open(jsonl_file, "w", encoding="utf-8") as f:
        f.write('{"id": 1, "code": "def a(): pass", "language": "python"}\n')
        f.write('{"id": 2, "code": "def b(): pass", "language": "python"}\n')

    success = DatasetExporter.jsonl_to_parquet(jsonl_file, parquet_file)
    assert success is True
    assert parquet_file.exists()
    assert parquet_file.stat().st_size > 0
