"""
Unit Tests for Discovery Module and Dataset Models
"""
import pytest
from src.discovery.models import DatasetSource, TrainingCategory, LicenseCategory, RedistributionPermission
from src.discovery.catalog import CURATED_DATASETS
from src.discovery.manager import DiscoveryManager

def test_curated_catalog_integrity():
    assert len(CURATED_DATASETS) >= 25
    for dataset in CURATED_DATASETS:
        assert dataset.dataset_id
        assert dataset.name
        assert dataset.source_url.startswith("http")
        assert dataset.publisher
        assert dataset.license
        assert dataset.license_category in LicenseCategory
        assert len(dataset.languages) > 0
        assert dataset.storage_size_bytes > 0
        assert dataset.version

def test_dataset_source_serialization():
    source = DatasetSource(
        dataset_id="test/repo",
        name="Test Dataset",
        source_url="https://github.com/test/repo",
        publisher="Test Author",
        license="mit",
        license_category=LicenseCategory.PERMISSIVE,
        redistribution_permissions=RedistributionPermission.ALLOW,
        languages=["python", "rust"],
        domains=["algorithms"],
        dataset_format="parquet",
        example_count=500,
        storage_size_bytes=1024,
        training_type=TrainingCategory.PRETRAINING
    )
    d = source.to_dict()
    assert d["dataset_id"] == "test/repo"
    assert d["license"] == "mit"
    assert "python" in d["languages"]

def test_discovery_manager_catalog_load():
    manager = DiscoveryManager()
    sources = manager.discover_all(enable_live_apis=False)
    assert len(sources) >= len(CURATED_DATASETS)

