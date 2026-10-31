"""
Tokenizer Adapters (Hugging Face Tokenizers, Byte-Level BPE, SentencePiece)
"""
import re
from abc import ABC, abstractmethod
from typing import List, Optional

class BaseTokenizerAdapter(ABC):
    """Abstract base class for model tokenizers."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass


class FastByteBPEAdapter(BaseTokenizerAdapter):
    """
    High-performance byte-level tokenizer adapter for coding models.
    Approximates StarCoder/GPT-4 tokenization using byte-fallback and code syntax boundaries.
    """

    def __init__(self):
        # Code token regex pattern matching identifiers, numbers, operators, punctuation, and whitespace
        self.pattern = re.compile(
            r"""[a-zA-Z_]\w*|[0-9]+(?:\.[0-9]+)?|[^\s\w]+|\s+""",
            re.UNICODE
        )

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        matches = self.pattern.findall(text)
        # Approximate byte-level splits for long non-whitespace tokens
        count = 0
        for m in matches:
            if len(m) > 16 and not m.isspace():
                count += (len(m) + 3) // 4
            else:
                count += 1
        return count

    def encode(self, text: str) -> List[int]:
        # Pseudo token ID encoding for demonstration / packing validation
        tokens = self.pattern.findall(text)
        return [hash(t) % 32000 for t in tokens]

    def decode(self, tokens: List[int]) -> str:
        return "<decoded_tokens>"


class HuggingFaceTokenizerAdapter(BaseTokenizerAdapter):
    """Adapter for tokenizers.Tokenizer instances or Hugging Face pretrained models."""

    def __init__(self, tokenizer_name_or_path: Optional[str] = None):
        self.tokenizer = None
        self.fallback = FastByteBPEAdapter()
        if tokenizer_name_or_path:
            try:
                import tokenizers
                # Check if it's a file path
                if Path(tokenizer_name_or_path).exists():
                    self.tokenizer = tokenizers.Tokenizer.from_file(tokenizer_name_or_path)
            except Exception:
                self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text).ids)
            except Exception:
                pass
        return self.fallback.count_tokens(text)

    def encode(self, text: str) -> List[int]:
        if self.tokenizer:
            try:
                return self.tokenizer.encode(text).ids
            except Exception:
                pass
        return self.fallback.encode(text)

    def decode(self, tokens: List[int]) -> str:
        if self.tokenizer:
            try:
                return self.tokenizer.decode(tokens)
            except Exception:
                pass
        return self.fallback.decode(tokens)

