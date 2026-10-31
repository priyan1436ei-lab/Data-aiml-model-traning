# Universal Global Coding Dataset (UGCD)
### World's Largest High-Quality Coding Dataset for AI/ML Model Training

[![License: Permissive](https://img.shields.io/badge/License-Permissive%20Verified-brightgreen.svg)](LICENSES.md)
[![Languages](https://img.shields.io/badge/Languages-36%20Supported-blue.svg)](DATASET_CARD.md)
[![Domains](https://img.shields.io/badge/Domains-35%20SE%20Domains-orange.svg)](DATASET_CARD.md)
[![Clean Pass Rate](https://img.shields.io/badge/Clean%20Pass%20Rate-100%25-success.svg)](QUALITY_REPORT.json)

The **Universal Global Coding Dataset** is an industrial-scale, multi-language coding dataset platform engineered to build, clean, deduplicate, tokenize, and package training corpora for next-generation Coding Large Language Models and Autonomous Software Engineering Agents.

---

## Quick Start (CLI Pipeline)

The platform supports end-to-end execution through `main.py`:

```bash
# 1. Discover global coding datasets from Hugging Face, GitHub, and Zenodo
python main.py discover

# 2. Collect and stream raw datasets with cryptographic SHA-256 provenance
python main.py collect

# 3. Clean and assess quality (syntax checking, security/secrets scan, decontamination)
python main.py clean

# 4. Multi-stage deduplication (exact SHA-256, normalized text, MinHash LSH)
python main.py deduplicate

# 5. Tokenization, sequence packing (4096 context), and zero-leakage splitting
python main.py tokenize

# 6. Prepare and export training-ready corpora (JSONL, Parquet)
python main.py prepare-training

# 7. Generate comprehensive quality reports and training manifests
python main.py report
```

---

## Directory Organization

```
universal-global-coding-dataset/
├── raw/                 # Raw ingested records with provenance ledger
├── metadata/            # Discovered dataset inventory & API responses
├── manifests/           # Cryptographic provenance ledger (SHA-256)
├── pretraining/         # Raw code pretraining corpus (source, docs, repos)
├── sft/                 # ChatML instruction fine-tuning datasets
├── completion/          # Fill-in-the-Middle (PSM/SPM) code completion
├── reasoning/           # Algorithmic complexity and step-by-step reasoning
├── repository_tasks/    # Multi-file SWE-bench style repo tasks and patches
├── agent_trajectories/  # Autonomous Plan -> Execute -> Test -> Repair traces
├── preference/          # Chosen vs Rejected code pairs for DPO/RLVR
├── benchmarks/          # Isolated evaluation benchmarks (quarantined)
├── cleaned/             # Syntactically verified and sanitized records
├── deduplicated/        # Cross-source deduplicated records
├── high_quality/        # Premium quality tier (score >= 0.85)
├── quarantine/          # Quarantined secrets, malware, and contaminated data
├── rejected/            # Syntactically invalid or corrupt records
├── tokenized/           # Sequence-packed context batches (4096 tokens)
├── train/               # Training split (JSONL & Parquet)
├── validation/          # Validation split (JSONL & Parquet)
├── test/                # Test split (JSONL & Parquet)
├── reports/             # Detailed audit summaries & token profiles
├── DATASET_CARD.md      # Official Hugging Face format dataset card
├── DATA_DICTIONARY.md   # Schema definitions for all training corpora
├── SOURCES.md           # Complete registry of all global coding datasets
├── LICENSES.md          # Legal compliance & redistribution matrix
├── QUALITY_REPORT.json  # Machine-readable quality metrics
└── TRAINING_MANIFEST.json # Training configuration & token scaling laws
```

---

## REST API & Web Dashboard

Launch the FastAPI service:
```bash
python main.py serve --port 8000
```
API endpoints:
- `GET /health` - System health and workspace verification
- `GET /stats` - Measured tokens, language distribution, and split metrics
- `GET /datasets` - Catalog of discovered and curated datasets
- `GET /search?q=query` - Full-text code snippet search across all languages

---

## Single-Command Docker Deployment

Deploy using Docker Compose:
```bash
docker-compose up --build
```
