"""
FastAPI REST API and Explorer Service for Universal Global Coding Dataset
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from src.config import PATHS, init_workspace

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_workspace()
    yield

app = FastAPI(
    title="Universal Global Coding Dataset API",
    description="REST API for querying metadata, token distributions, quality tiers, and code snippets.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "dataset_root_exists": PATHS["root"].exists(),
        "train_split_exists": (PATHS["train"] / "train.jsonl").exists(),
        "service": "Universal Global Coding Dataset Engine"
    }

@app.get("/stats")
def get_dataset_statistics():
    manifest_file = PATHS["root"] / "TRAINING_MANIFEST.json"
    if not manifest_file.exists():
        raise HTTPException(status_code=404, detail="Training manifest not found. Run prepare-training first.")
    with open(manifest_file, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/datasets")
def list_datasets(ecosystem: Optional[str] = None):
    inventory_file = PATHS["metadata"] / "discovered_datasets.json"
    if not inventory_file.exists():
        raise HTTPException(status_code=404, detail="Dataset inventory not found. Run discover first.")
    with open(inventory_file, "r", encoding="utf-8") as f:
        datasets = json.load(f)
    if ecosystem:
        datasets = [d for d in datasets if d.get("source_ecosystem") == ecosystem.lower()]
    return {"total": len(datasets), "datasets": datasets}

@app.get("/search")
def search_code(
    q: str = Query(..., description="Keyword or identifier to search"),
    language: Optional[str] = None,
    limit: int = 10
):
    dedup_file = PATHS["deduplicated"] / "deduplicated_records.jsonl"
    if not dedup_file.exists():
        raise HTTPException(status_code=404, detail="Deduplicated dataset not found.")
    
    matches = []
    with open(dedup_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if language and rec.get("language") != language.lower():
                continue
            content = rec.get("content", "")
            if q.lower() in content.lower():
                matches.append({
                    "id": rec.get("id"),
                    "language": rec.get("language"),
                    "domain": rec.get("domain"),
                    "source": rec.get("source"),
                    "license": rec.get("license"),
                    "snippet": content[:300] + ("..." if len(content) > 300 else "")
                })
                if len(matches) >= limit:
                    break

    return {"query": q, "language": language, "count": len(matches), "results": matches}

@app.get("/quality/summary")
def get_quality_summary():
    report_file = PATHS["root"] / "QUALITY_REPORT.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Quality report not found. Run clean first.")
    with open(report_file, "r", encoding="utf-8") as f:
        return json.load(f)
