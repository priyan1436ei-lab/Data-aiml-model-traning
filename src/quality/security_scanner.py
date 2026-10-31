"""
Security, Secrets, and Malicious Payload Scanner
Detects AWS keys, GitHub tokens, private keys, reverse shells, and malicious exploits.
"""
import re
from typing import Dict, List, Tuple

class SecurityScanner:
    """Scans code for PII, hardcoded credentials, secrets, and malicious execution patterns."""

    SECRET_PATTERNS = [
        ("AWS_ACCESS_KEY", re.compile(r"(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}")),
        ("AWS_SECRET_KEY", re.compile(r"aws_secret_access_key\s*=\s*['\"][A-Za-z0-9/+=]{40}['\"]", re.IGNORECASE)),
        ("GITHUB_PAT", re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,255}")),
        ("GITHUB_FINE_GRAINED", re.compile(r"github_pat_[A-Za-z0-9_]{82}")),
        ("PRIVATE_RSA_KEY", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
        ("DATABASE_URI_PASSWORD", re.compile(r"(?:postgres|postgresql|mysql|mongodb(?:\+srv)?)://[^:]+:([^@]+)@", re.IGNORECASE)),
        ("SLACK_TOKEN", re.compile(r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24,32}")),
        ("STRIPE_KEY", re.compile(r"(?:sk|pk)_(?:test|live)_[0-9a-zA-Z]{24}")),
        ("GENERIC_API_KEY", re.compile(r"(?:api_key|secret_key|authToken)\s*[:=]\s*['\"][0-9a-zA-Z\-_]{24,}['\"]", re.IGNORECASE)),
    ]

    MALICIOUS_PATTERNS = [
        ("REVERSE_SHELL_NETCAT", re.compile(r"nc\s+.*(?:-[ec]\s+/(?:bin/)?(?:ba)?sh|-(?:e|c)\s+[0-9\.]+)", re.IGNORECASE)),
        ("REVERSE_SHELL_BASH", re.compile(r"/bin/(?:bash|sh)\s+-i\s+>&\s+/dev/tcp/")),
        ("REVERSE_SHELL_PYTHON", re.compile(r"socket\.socket.*connect\(.*os\.dup2", re.DOTALL)),
        ("FORK_BOMB", re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:")),
        ("DANGEROUS_EVAL_BASE64", re.compile(r"eval\s*\(\s*(?:base64\.b64decode|atob)\s*\(", re.IGNORECASE)),
        ("DESTRUCTIVE_ROOT_RM", re.compile(r"rm\s+-rf\s+/(?:\s|$|\*)")),
    ]

    @classmethod
    def scan(cls, content: str) -> Tuple[bool, List[str], str]:
        """
        Scan content for security violations.
        Returns (is_clean, violations_list, category).
        category can be 'CLEAN', 'SECRETS', or 'MALWARE'.
        """
        violations = []

        # 1. Check for Secrets
        for name, pattern in cls.SECRET_PATTERNS:
            matches = pattern.findall(content)
            if matches:
                violations.append(f"Hardcoded secret detected: {name} (count: {len(matches)})")

        if violations:
            return False, violations, "SECRETS"

        # 2. Check for Malicious Exploits
        for name, pattern in cls.MALICIOUS_PATTERNS:
            if pattern.search(content):
                violations.append(f"Malicious exploit pattern detected: {name}")

        if violations:
            return False, violations, "MALWARE"

        return True, [], "CLEAN"
