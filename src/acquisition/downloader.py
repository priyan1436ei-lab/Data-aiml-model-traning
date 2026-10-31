"""
Resilient Streaming Downloader with Exponential Backoff and Integrity Verification
"""
import hashlib
import logging
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class ResilientDownloader:
    """Handles rate-limited, retrying, chunked streaming file acquisition."""

    def __init__(self, max_retries: int = 4, backoff_factor: float = 2.0, timeout: int = 30):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout

    def download_file(
        self,
        url: str,
        dest_path: Path,
        expected_sha256: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """Download file with chunked streaming, retry backoff, and SHA-256 verification."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

        attempt = 0
        delay = 1.0

        headers = {
            "User-Agent": "Universal-Global-Coding-Dataset-Collector/1.0"
        }

        while attempt < self.max_retries:
            attempt += 1
            hasher = hashlib.sha256()
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    total_size = int(resp.headers.get("Content-Length", 0))
                    downloaded = 0
                    with open(temp_path, "wb") as f:
                        while True:
                            chunk = resp.read(65536)  # 64 KB chunks
                            if not chunk:
                                break
                            f.write(chunk)
                            hasher.update(chunk)
                            downloaded += len(chunk)
                            if progress_callback and total_size > 0:
                                progress_callback(downloaded, total_size)

                computed_hash = hasher.hexdigest()
                if expected_sha256 and computed_hash.lower() != expected_sha256.lower():
                    logger.error(f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {computed_hash}")
                    if temp_path.exists():
                        temp_path.unlink()
                    return False

                # Atomic replace
                if temp_path.exists():
                    if dest_path.exists():
                        dest_path.unlink()
                    temp_path.rename(dest_path)
                return True

            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                logger.warning(f"Download attempt {attempt} failed for {url}: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= self.backoff_factor

        logger.error(f"Exceeded max retries ({self.max_retries}) downloading {url}")
        if temp_path.exists():
            temp_path.unlink()
        return False

