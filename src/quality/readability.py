"""
Language Identification and Readability Metrics Engine
"""
import re
from typing import Dict, Tuple
from src.config import LANGUAGE_EXTENSIONS, SUPPORTED_LANGUAGES

class LanguageIdentifier:
    """Identifies programming language using path extensions, shebangs, and keyword signatures."""

    LANGUAGE_SIGNATURES: Dict[str, list] = {
        "python": ["def ", "import ", "class ", "elif ", "self.", "print("],
        "javascript": ["const ", "let ", "function", "console.log", "=>", "module.exports"],
        "typescript": ["interface ", "type ", ": string", ": number", ": boolean", "export "],
        "rust": ["fn ", "let mut ", "impl ", "pub struct", "match ", "println!"],
        "go": ["func ", "package ", "import (", "chan ", "go func()", "struct {"],
        "cpp": ["#include <", "std::", "template <", "nullptr", "cout <<"],
        "c": ["#include <", "malloc(", "free(", "printf(", "size_t "],
        "java": ["public class ", "private static ", "System.out.println", "package com."],
        "csharp": ["using System;", "namespace ", "public class ", "Console.WriteLine"],
        "solidity": ["pragma solidity", "contract ", "mapping(", "uint256", "msg.sender"],
        "cuda": ["__global__", "__device__", "__shared__", "blockIdx", "threadIdx"],
        "sql": ["SELECT ", "FROM ", "WHERE ", "JOIN ", "GROUP BY ", "ORDER BY "],
        "shell": ["#!/bin/bash", "#!/bin/sh", "echo ", "fi\n", "done\n", "exit 0"],
        "powershell": ["[CmdletBinding()]", "param (", "Write-Host", "$PSScriptRoot"],
        "verilog": ["module ", "endmodule", "input wire", "output reg", "always @"],
        "vhdl": ["entity ", "architecture ", "port (", "signal "],
        "ruby": ["def ", "end\n", "attr_accessor", "require '", "class "],
        "php": ["<?php", "public function ", "echo $", "$this->"],
        "swift": ["import Foundation", "func ", "guard let ", "override func"],
        "kotlin": ["fun ", "val ", "var ", "println(", "companion object"],
        "scala": ["object ", "def ", "val ", "case class ", "extends "],
        "r": ["<- function", "library(", "data.frame(", "ggplot("],
        "julia": ["function ", "end\n", "using ", "@threads", "@simd"],
        "haskell": ["module ", "where", ":: ", "data ", "deriving "],
        "elixir": ["defmodule ", "def ", "do\n", "use ", "@doc"],
        "clojure": ["(defn ", "(def ", "(ns ", "(let ["],
        "lua": ["function ", "local ", "end\n", "then\n"],
        "perl": ["use strict;", "use warnings;", "my $", "sub "],
        "dart": ["void main()", "class ", "Widget build", "final "],
    }

    @classmethod
    def identify(cls, code: str, file_path: str = "") -> Tuple[str, float]:
        """Identify language from path or code heuristics. Returns (language, confidence)."""
        # 1. Path extension check
        if file_path:
            for ext, lang in LANGUAGE_EXTENSIONS.items():
                if file_path.endswith(ext):
                    return lang, 0.95

        # 2. Shebang line
        lines = code.strip().split("\n")
        if lines and lines[0].startswith("#!"):
            shebang = lines[0].lower()
            if "python" in shebang:
                return "python", 0.99
            elif "bash" in shebang or "sh" in shebang:
                return "shell", 0.95
            elif "node" in shebang:
                return "javascript", 0.95
            elif "perl" in shebang:
                return "perl", 0.95
            elif "ruby" in shebang:
                return "ruby", 0.95

        # 3. Keyword matching scores
        scores: Dict[str, int] = {}
        for lang, signatures in cls.LANGUAGE_SIGNATURES.items():
            matches = sum(1 for sig in signatures if sig in code)
            if matches > 0:
                scores[lang] = matches

        if scores:
            best_lang = max(scores, key=scores.get)
            confidence = min(0.9, scores[best_lang] * 0.25)
            return best_lang, confidence

        return "generic_code", 0.3


class ReadabilityEvaluator:
    """Evaluates code cleanliness, comment ratio, and structural consistency."""

    @staticmethod
    def evaluate(code: str) -> Dict[str, float]:
        if not code or not code.strip():
            return {
                "readability_score": 0.0,
                "line_count": 0,
                "mean_line_length": 0.0,
                "alpha_ratio": 0.0,
                "comment_ratio": 0.0
            }

        lines = code.split("\n")
        line_count = len(lines)
        non_empty = [l for l in lines if l.strip()]
        
        if not non_empty:
            return {"readability_score": 0.0, "line_count": 0, "mean_line_length": 0.0, "alpha_ratio": 0.0, "comment_ratio": 0.0}

        mean_length = sum(len(l) for l in non_empty) / len(non_empty)
        max_length = max(len(l) for l in lines)

        # Alphanumeric ratio
        total_chars = len(code)
        alpha_chars = sum(1 for c in code if c.isalnum() or c in " _")
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0.0

        # Comment lines heuristic
        comment_lines = sum(
            1 for l in non_empty
            if l.strip().startswith(("#", "//", "/*", "*", "--", ";", "%"))
        )
        comment_ratio = comment_lines / len(non_empty) if non_empty else 0.0

        # Scoring heuristics
        score = 1.0

        # Penalize overly dense minified lines
        if mean_length > 150:
            score -= 0.3
        elif mean_length < 15:
            score -= 0.2

        if max_length > 1000:
            score -= 0.4

        if alpha_ratio < 0.30:
            score -= 0.4

        score = max(0.0, min(1.0, score))

        return {
            "readability_score": round(score, 3),
            "line_count": line_count,
            "mean_line_length": round(mean_length, 1),
            "max_line_length": max_length,
            "alpha_ratio": round(alpha_ratio, 3),
            "comment_ratio": round(comment_ratio, 3)
        }

