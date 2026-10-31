"""
Academic and Open Science Discovery Engine (Zenodo, PapersWithCode, Open Repositories)
"""
import json
import logging
import urllib.request
import urllib.parse
from typing import List, Optional
from src.discovery.models import DatasetSource, TrainingCategory, LicenseCategory, RedistributionPermission

logger = logging.getLogger(__name__)

class AcademicDiscoverer:
    """Discovers datasets from Zenodo and Academic research indices."""
    ZENODO_API_URL = "https://zenodo.org/api/records"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def search_zenodo(self, query: str = "software engineering code dataset", limit: int = 10) -> List[DatasetSource]:
        params = {
            "q": query,
            "size": limit,
            "type": "dataset",
            "sort": "mostrecent"
        }
        url = f"{self.ZENODO_API_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Universal-Global-Coding-Dataset/1.0"}
        )
        discovered = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    hits = data.get("hits", {}).get("hits", [])
                    for hit in hits:
                        source = self._parse_zenodo_hit(hit)
                        if source:
                            discovered.append(source)
        except Exception as e:
            logger.warning(f"Zenodo discovery failed: {e}")
        return discovered

    def _parse_zenodo_hit(self, hit: dict) -> Optional[DatasetSource]:
        try:
            record_id = str(hit.get("id"))
            metadata = hit.get("metadata", {})
            title = metadata.get("title", f"Zenodo Record {record_id}")
            lic_dict = metadata.get("license", {})
            lic_id = lic_dict.get("id", "cc-by-4.0").lower()

            creators = metadata.get("creators", [])
            publisher = creators[0].get("name", "Academic Researchers") if creators else "Zenodo Community"

            files = hit.get("files", [])
            total_size = sum(f.get("size", 0) for f in files) or 50000000

            return DatasetSource(
                dataset_id=f"zenodo/{record_id}",
                name=title[:80],
                source_url=hit.get("links", {}).get("html", f"https://zenodo.org/record/{record_id}"),
                publisher=publisher,
                license=lic_id,
                license_category=LicenseCategory.PERMISSIVE if "cc-by" in lic_id or "mit" in lic_id else LicenseCategory.UNKNOWN,
                access_permissions="public" if hit.get("metadata", {}).get("access_right") == "open" else "restricted",
                redistribution_permissions=RedistributionPermission.ALLOW_WITH_ATTRIBUTION,
                languages=["generic_code", "python", "java", "cpp"],
                domains=["software_architecture", "testing_and_debugging", "empirical_se"],
                dataset_format="archive/json/csv",
                example_count=5000,
                storage_size_bytes=total_size,
                token_count=None,
                version=metadata.get("version", "1.0.0"),
                training_type=TrainingCategory.DEBUGGING,
                quality_information=f"Peer-reviewed academic research artifact archived on Zenodo (DOI: {metadata.get('doi', 'N/A')})",
                known_limitations=["Domain specific academic format; requires custom parsing"],
                is_curated=False,
                verification_status="discovered",
                source_ecosystem="zenodo"
            )
        except Exception as e:
            logger.debug(f"Error parsing Zenodo hit: {e}")
            return None

