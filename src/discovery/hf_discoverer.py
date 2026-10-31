"""
Hugging Face Dataset Discovery Engine
Queries Hugging Face API dynamically to discover public coding datasets.
"""
import json
import logging
import urllib.request
import urllib.parse
from typing import List, Optional
from src.discovery.models import DatasetSource, TrainingCategory, LicenseCategory, RedistributionPermission

logger = logging.getLogger(__name__)

class HuggingFaceDiscoverer:
    """Discovers coding datasets from Hugging Face Datasets Hub."""
    API_URL = "https://huggingface.co/api/datasets"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def search(self, query: str = "code", limit: int = 20) -> List[DatasetSource]:
        """Search HF Hub for coding datasets matching query."""
        params = {
            "search": query,
            "limit": limit,
            "full": "true"
        }
        url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Universal-Global-Coding-Dataset/1.0"}
        )
        discovered = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    for item in data:
                        source = self._parse_hf_item(item)
                        if source:
                            discovered.append(source)
        except Exception as e:
            logger.warning(f"Failed to query Hugging Face API: {e}")
        return discovered

    def _parse_hf_item(self, item: dict) -> Optional[DatasetSource]:
        try:
            repo_id = item.get("id") or item.get("_id")
            if not repo_id:
                return None
            tags = item.get("tags", [])
            card_data = item.get("cardData", {}) or {}
            license_str = card_data.get("license") or "unknown"
            if isinstance(license_str, list) and license_str:
                license_str = license_str[0]
            license_str = str(license_str).lower()

            # Classify license
            if any(l in license_str for l in ["mit", "apache", "bsd", "cc0", "unlicense", "openrail"]):
                lic_cat = LicenseCategory.PERMISSIVE
                redist = RedistributionPermission.ALLOW_WITH_ATTRIBUTION
            elif any(l in license_str for l in ["gpl", "agpl", "lgpl", "mpl"]):
                lic_cat = LicenseCategory.COPYLEFT
                redist = RedistributionPermission.ALLOW_WITH_ATTRIBUTION
            elif "nc" in license_str or "non-commercial" in license_str:
                lic_cat = LicenseCategory.NON_COMMERCIAL
                redist = RedistributionPermission.NON_COMMERCIAL_ONLY
            else:
                lic_cat = LicenseCategory.UNKNOWN
                redist = RedistributionPermission.RESTRICTED

            # Determine languages from tags
            languages = []
            for tag in tags:
                if tag.startswith("language:"):
                    lang = tag.split(":", 1)[1].strip()
                    languages.append(lang)
            if not languages:
                languages = ["python", "generic_code"]

            # Infer training type
            task_type = TrainingCategory.PRETRAINING
            if any(t in repo_id.lower() or t in str(tags).lower() for t in ["instruct", "sft", "alpaca"]):
                task_type = TrainingCategory.SFT
            elif any(t in repo_id.lower() for t in ["dpo", "preference", "feedback", "rlhf"]):
                task_type = TrainingCategory.PREFERENCE
            elif any(t in repo_id.lower() for t in ["eval", "bench"]):
                task_type = TrainingCategory.BENCHMARK

            downloads = item.get("downloads", 0)
            likes = item.get("likes", 0)

            return DatasetSource(
                dataset_id=repo_id,
                name=repo_id.split("/")[-1].replace("-", " ").title(),
                source_url=f"https://huggingface.co/datasets/{repo_id}",
                publisher=repo_id.split("/")[0] if "/" in repo_id else "Community",
                license=license_str,
                license_category=lic_cat,
                access_permissions="public" if not item.get("private") else "gated",
                redistribution_permissions=redist,
                languages=languages,
                domains=["code_generation", "basic_programming"],
                dataset_format="parquet/jsonl",
                example_count=downloads if downloads > 0 else 1000,
                storage_size_bytes=100000000,
                token_count=None,
                version="latest",
                training_type=task_type,
                quality_information=f"HF Hub Community dataset with {likes} likes and {downloads} downloads",
                known_limitations=["Community uploaded dataset; verify license and purity before pretraining"],
                is_curated=False,
                verification_status="discovered",
                source_ecosystem="huggingface"
            )
        except Exception as e:
            logger.debug(f"Error parsing HF item: {e}")
            return None

