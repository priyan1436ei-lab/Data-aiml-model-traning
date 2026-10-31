"""
Multi-Language Syntax and Structural Validator
Performs AST parsing and grammatical balance checks across 36 languages.
"""
import ast
import json
from typing import Tuple, Optional

class SyntaxValidator:
    """Validates syntactic correctness across languages."""

    @staticmethod
    def validate(code: str, language: str) -> Tuple[bool, Optional[str]]:
        """Validate code syntax. Returns (is_valid, error_description)."""
        lang = language.lower()
        if not code or not code.strip():
            return False, "Empty code snippet"

        if lang == "python":
            return SyntaxValidator._validate_python(code)
        elif lang in ["json"]:
            return SyntaxValidator._validate_json(code)
        elif lang in [
            "c", "cpp", "csharp", "java", "javascript", "typescript",
            "go", "rust", "php", "kotlin", "swift", "solidity", "dart", "scala"
        ]:
            return SyntaxValidator._validate_c_family_delimiters(code)
        elif lang == "sql":
            return SyntaxValidator._validate_sql(code)
        elif lang in ["shell", "bash", "powershell"]:
            return SyntaxValidator._validate_shell(code)
        
        # Generic structural validation
        return SyntaxValidator._validate_c_family_delimiters(code)

    @staticmethod
    def _validate_python(code: str) -> Tuple[bool, Optional[str]]:
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Python SyntaxError at line {e.lineno}: {e.msg}"

    @staticmethod
    def _validate_json(code: str) -> Tuple[bool, Optional[str]]:
        try:
            json.loads(code)
            return True, None
        except Exception as e:
            return False, f"JSON parse error: {str(e)}"

    @staticmethod
    def _validate_c_family_delimiters(code: str) -> Tuple[bool, Optional[str]]:
        """Verify balanced braces {}, brackets [], and parentheses (), ignoring strings/comments."""
        stack = []
        pairs = {')': '(', '}': '{', ']': '['}
        in_single_quote = False
        in_double_quote = False
        in_line_comment = False
        in_block_comment = False
        escaped = False

        chars = list(code)
        i = 0
        n = len(chars)

        while i < n:
            ch = chars[i]
            next_ch = chars[i+1] if i + 1 < n else ''

            if in_line_comment:
                if ch == '\n':
                    in_line_comment = False
                i += 1
                continue

            if in_block_comment:
                if ch == '*' and next_ch == '/':
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue

            if escaped:
                escaped = False
                i += 1
                continue

            if ch == '\\':
                escaped = True
                i += 1
                continue

            if ch == '/' and next_ch == '/' and not in_single_quote and not in_double_quote:
                in_line_comment = True
                i += 2
                continue

            if ch == '/' and next_ch == '*' and not in_single_quote and not in_double_quote:
                in_block_comment = True
                i += 2
                continue

            if ch == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
                i += 1
                continue

            if ch == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
                i += 1
                continue

            if not in_single_quote and not in_double_quote:
                if ch in "({[":
                    stack.append((ch, i))
                elif ch in ")}]":
                    if not stack:
                        return False, f"Unmatched closing delimiter '{ch}' at character {i}"
                    top, _ = stack.pop()
                    if pairs[ch] != top:
                        return False, f"Mismatched delimiter '{ch}' doesn't close '{top}'"

            i += 1

        if stack:
            unclosed = [s[0] for s in stack]
            return False, f"Unclosed delimiters remaining: {unclosed}"

        return True, None

    @staticmethod
    def _validate_sql(code: str) -> Tuple[bool, Optional[str]]:
        upper = code.upper()
        keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "WITH"]
        if not any(k in upper for k in keywords):
            return False, "SQL statement missing fundamental query keywords"
        # Balanced parentheses check
        if code.count('(') != code.count(')'):
            return False, "Unbalanced parentheses in SQL statement"
        return True, None

    @staticmethod
    def _validate_shell(code: str) -> Tuple[bool, Optional[str]]:
        if code.count('"') % 2 != 0:
            return False, "Unmatched double quotes in shell script"
        if code.count("'") % 2 != 0:
            return False, "Unmatched single quotes in shell script"
        return True, None

