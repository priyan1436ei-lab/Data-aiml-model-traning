"""
Universal Discovery Manager
Coordinates curated catalog and dynamic API discovery across Hugging Face, GitHub, and Zenodo.
Generates comprehensive dataset inventories, SOURCES.md, and LICENSES.md.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from src.config import PATHS, init_workspace
from src.discovery.models import DatasetSource, LicenseCategory
from src.discovery.catalog import CURATED_DATASETS
from src.discovery.hf_discoverer import HuggingFaceDiscoverer
from src.discovery.github_discoverer import GitHubDiscoverer
from src.discovery.academic_discoverer import AcademicDiscoverer

logger = logging.getLogger(__name__)

class DiscoveryManager:
    """Manages full dataset discovery, verification, and documentation generation."""

    def __init__(self):
        init_workspace()
        self.curated_datasets = list(CURATED_DATASETS)
        self.hf_discoverer = HuggingFaceDiscoverer()
        self.github_discoverer = GitHubDiscoverer()
        self.academic_discoverer = AcademicDiscoverer()

    def discover_all(self, enable_live_apis: bool = True, query_limit: int = 15) -> List[DatasetSource]:
        """Run complete discovery across curated catalog and live ecosystems."""
        datasets: Dict[str, DatasetSource] = {}

        # 1. Load curated authoritative datasets
        for d in self.curated_datasets:
            datasets[d.dataset_id] = d
        print(f"[Discovery] Loaded {len(self.curated_datasets)} authoritative curated datasets.")

        # 2. Live API discovery if enabled
        if enable_live_apis:
            print("[Discovery] Querying Hugging Face Datasets API...")
            hf_results = self.hf_discoverer.search(query="code instruction", limit=query_limit)
            hf_results += self.hf_discoverer.search(query="starcoder", limit=query_limit // 2)
            for d in hf_results:
                if d.dataset_id not in datasets:
                    datasets[d.dataset_id] = d
            print(f"[Discovery] Discovered {len(hf_results)} datasets from Hugging Face Hub.")

            print("[Discovery] Querying GitHub Search API...")
            gh_results = self.github_discoverer.search_repos(topic="code-dataset", limit=10)
            gh_results += self.github_discoverer.search_repos(topic="coding-benchmark", limit=10)
            for d in gh_results:
                if d.dataset_id not in datasets:
                    datasets[d.dataset_id] = d
            print(f"[Discovery] Discovered {len(gh_results)} repositories from GitHub.")

            print("[Discovery] Querying Zenodo Academic Archive...")
            zenodo_results = self.academic_discoverer.search_zenodo(query="software engineering dataset", limit=10)
            for d in zenodo_results:
                if d.dataset_id not in datasets:
                    datasets[d.dataset_id] = d
            print(f"[Discovery] Discovered {len(zenodo_results)} records from Zenodo.")

        all_sources = list(datasets.values())

        # Save machine-readable inventory
        inventory_file = PATHS["metadata"] / "discovered_datasets.json"
        with open(inventory_file, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in all_sources], f, indent=2)
        print(f"[Discovery] Saved complete inventory ({len(all_sources)} sources) to {inventory_file}")

        # Generate markdown documents
        self.generate_sources_md(all_sources)
        self.generate_licenses_md(all_sources)

        return all_sources

    def generate_sources_md(self, sources: List[DatasetSource]):
        """Generate comprehensive SOURCES.md with complete details required by prompt."""
        path = PATHS["root"] / "SOURCES.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write("# UNIVERSAL GLOBAL CODING DATASET - SOURCES REGISTRY\n\n")
            f.write("Comprehensive inventory of all researched, curated, and discovered global coding datasets.\n\n")
            f.write("| # | Dataset Name | Dataset ID | Publisher | License | Ecosystem | Training Type | Est. Size | Status |\n")
            f.write("|---|--------------|------------|-----------|---------|-----------|---------------|-----------|--------|\n")
            for idx, s in enumerate(sources, 1):
                size_mb = f"{s.storage_size_bytes / (1024*1024):.1f} MB" if s.storage_size_bytes < 1024**3 else f"{s.storage_size_bytes / (1024**3):.1f} GB"
                if s.storage_size_bytes >= 1024**4:
                    size_mb = f"{s.storage_size_bytes / (1024**4):.1f} TB"
                f.write(f"| {idx} | [{s.name}]({s.source_url}) | `{s.dataset_id}` | {s.publisher} | `{s.license}` | {s.source_ecosystem} | {s.training_type.value} | {size_mb} | {s.verification_status} |\n")

            f.write("\n\n## Detailed Dataset Metadata Dossiers\n\n")
            for s in sources:
                f.write(f"### {s.name} (`{s.dataset_id}`)\n")
                f.write(f"- **Official URL**: [{s.source_url}]({s.source_url})\n")
                f.write(f"- **Publisher**: {s.publisher}\n")
                f.write(f"- **License**: `{s.license}` (Category: `{s.license_category.value}`)\n")
                f.write(f"- **Access Permissions**: {s.access_permissions}\n")
                f.write(f"- **Redistribution Rights**: {s.redistribution_permissions.value}\n")
                f.write(f"- **Covered Languages**: {', '.join(s.languages[:12])}{' (and more)' if len(s.languages) > 12 else ''}\n")
                f.write(f"- **Domains**: {', '.join(s.domains[:8])}\n")
                f.write(f"- **Format**: {s.dataset_format} | **Examples**: {s.example_count:,} | **Tokens**: {f'{s.token_count:,}' if s.token_count else 'Not measured'}\n")
                f.write(f"- **Quality Curation**: {s.quality_information}\n")
                f.write(f"- **Known Limitations**: {'; '.join(s.known_limitations)}\n\n")
        print(f"[Discovery] Generated {path}")

    def generate_licenses_md(self, sources: List[DatasetSource]):
        """Generate comprehensive LICENSES.md with legal attribution & compatibility matrix."""
        path = PATHS["root"] / "LICENSES.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write("# UNIVERSAL GLOBAL CODING DATASET - LICENSING & COMPLIANCE MATRIX\n\n")
            f.write("## Legal Verification Framework\n\n")
            f.write("Datasets are classified into strict licensing tiers to ensure total legal compliance for LLM pretraining, fine-tuning, and redistribution.\n\n")
            f.write("### 1. Permissive Licenses (Commercially Trainable & Redistributable)\n")
            f.write("- **Licenses**: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Unlicense, CC0-1.0, CC-BY-4.0\n")
            f.write("- **Usage**: Fully approved for foundation model pretraining, fine-tuning, commercial deployment, and redistribution with copyright attribution.\n\n")
            f.write("### 2. Copyleft Licenses (Viral Terms - Strict Segregation)\n")
            f.write("- **Licenses**: GPL-2.0, GPL-3.0, AGPL-3.0, LGPL-2.1/3.0, MPL-2.0\n")
            f.write("- **Policy**: Kept in distinct training splits. In jurisdictions where model weights are considered derivative works, users can selectively filter out copyleft splits.\n\n")
            f.write("### 3. Non-Commercial / Research-Only Licenses\n")
            f.write("- **Licenses**: CC-BY-NC-4.0, Stanford Alpaca Terms, Academic Research Only agreements\n")
            f.write("- **Policy**: Segregated into non-commercial evaluation and academic research partitions.\n\n")
            f.write("### 4. Restricted / Proprietary / Opted-Out\n")
            f.write("- **Policy**: Quarantined and permanently excluded from all clean training partitions.\n\n")
            f.write("## License Distribution Across Catalog\n\n")
            f.write("| License Category | Permitted For Pretraining | Permitted For SFT | Redistribution Allowed | Count |\n")
            f.write("|---|---|---|---|---|\n")

            cats: Dict[str, int] = {}
            for s in sources:
                cat = s.license_category.value
                cats[cat] = cats.get(cat, 0) + 1

            f.write(f"| Permissive | YES | YES | YES (Attribution required) | {cats.get('permissive', 0)} |\n")
            f.write(f"| Copyleft | Configurable (Check jurisdiction) | Segregated | Under Copyleft terms | {cats.get('copyleft', 0)} |\n")
            f.write(f"| Non-Commercial | Academic Only | Academic Only | Academic Non-Commercial | {cats.get('non_commercial', 0)} |\n")
            f.write(f"| Unknown / Restricted | QUARANTINED | QUARANTINED | NO | {cats.get('unknown', 0) + cats.get('restricted', 0)} |\n")
        print(f"[Discovery] Generated {path}")

