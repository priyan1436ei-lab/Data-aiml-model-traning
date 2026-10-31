# UNIVERSAL GLOBAL CODING DATASET - LICENSING & COMPLIANCE MATRIX

## Legal Verification Framework

Datasets are classified into strict licensing tiers to ensure total legal compliance for LLM pretraining, fine-tuning, and redistribution.

### 1. Permissive Licenses (Commercially Trainable & Redistributable)
- **Licenses**: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Unlicense, CC0-1.0, CC-BY-4.0
- **Usage**: Fully approved for foundation model pretraining, fine-tuning, commercial deployment, and redistribution with copyright attribution.

### 2. Copyleft Licenses (Viral Terms - Strict Segregation)
- **Licenses**: GPL-2.0, GPL-3.0, AGPL-3.0, LGPL-2.1/3.0, MPL-2.0
- **Policy**: Kept in distinct training splits. In jurisdictions where model weights are considered derivative works, users can selectively filter out copyleft splits.

### 3. Non-Commercial / Research-Only Licenses
- **Licenses**: CC-BY-NC-4.0, Stanford Alpaca Terms, Academic Research Only agreements
- **Policy**: Segregated into non-commercial evaluation and academic research partitions.

### 4. Restricted / Proprietary / Opted-Out
- **Policy**: Quarantined and permanently excluded from all clean training partitions.

## License Distribution Across Catalog

| License Category | Permitted For Pretraining | Permitted For SFT | Redistribution Allowed | Count |
|---|---|---|---|---|
| Permissive | YES | YES | YES (Attribution required) | 28 |
| Copyleft | Configurable (Check jurisdiction) | Segregated | Under Copyleft terms | 0 |
| Non-Commercial | Academic Only | Academic Only | Academic Non-Commercial | 1 |
| Unknown / Restricted | QUARANTINED | QUARANTINED | NO | 0 |
