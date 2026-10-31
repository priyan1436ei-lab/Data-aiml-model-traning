"""
GitHub Repository and Dataset Discovery Engine
Discovers high-quality open-source coding repositories, issue datasets, and benchmarks.
"""
import json
import logging
import urllib.request
import urllib.parse
from typing import List, Optional
from src.discovery.models import DatasetSource, TrainingCategory, LicenseCategory, RedistributionPermission

logger = logging.getLogger(__name__)

class GitHubDiscoverer:
    """Discovers curated code repositories and dataset repos from GitHub."""
    API_URL = "https://api.github.com/search/repositories"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def search_repos(self, topic: str = "coding-dataset", limit: int = 10) -> List[DatasetSource]:
        query = f"topic:{topic}+stars:>50"
        url = f"{self.API_URL}?q={urllib.parse.quote(query)}&per_page={limit}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Universal-Global-Coding-Dataset/1.0",
                "Accept": "application/vnd.github.v3+json"
            }
        )
        discovered = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    for item in data.get("items", []):
                        source = self._parse_repo_item(item)
                        if source:
                            discovered.append(source)
        except Exception as e:
            logger.warning(f"GitHub search failed: {e}")
        return discovered

    def _parse_repo_item(self, item: dict) -> Optional[DatasetSource]:
        try:
            full_name = item.get("full_name")
            if not full_name:
                return None
            lic = item.get("license") or {}
            spdx_id = (lic.get("spdx_id") or "unknown").lower()
            
            if spdx_id in ["mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0"]:
                lic_cat = LicenseCategory.PERMISSIVE
                redist = RedistributionPermission.ALLOW_WITH_ATTRIBUTION
            elif spdx_id in ["gpl-2.0", "gpl-3.0", "agpl-3.0", "lgpl-3.0"]:
                lic_cat = LicenseCategory.COPYLEFT
                redist = RedistributionPermission.ALLOW_WITH_ATTRIBUTION
            else:
                lic_cat = LicenseCategory.UNKNOWN
                redist = RedistributionPermission.RESTRICTED

            lang = (item.get("language") or "generic_code").lower()
            stars = item.get("stargazers_count", 0)

            return DatasetSource(
                dataset_id=f"github/{full_name}",
                name=item.get("name", "").replace("-", " ").title(),
                source_url=item.get("html_url", f"https://github.com/{full_name}"),
                publisher=item.get("owner", {}).get("login", "GitHub Community"),
                license=spdx_id,
                license_category=lic_cat,
                access_permissions="public",
                redistribution_permissions=redist,
                languages=[lang] if lang != "generic_code" else ["python", "javascript"],
                domains=["repository_understanding", "multi_file_code_generation", "software_architecture"],
                dataset_format="git",
                example_count=stars * 10,
                storage_size_bytes=item.get("size", 1024) * 1024,
                token_count=None,
                version=item.get("default_branch", "main"),
                training_type=TrainingCategory.REPOSITORY_TASKS,
                quality_information=f"GitHub repository with {stars} stars, description: {item.get('description') or 'None'}",
                known_limitations=["Git repository snapshot; subject to developer changes"],
                is_curated=False,
                verification_status="discovered",
                source_ecosystem="github"
            )
        except Exception as e:
            logger.debug(f"Error parsing GitHub repo: {e}")
            return None

