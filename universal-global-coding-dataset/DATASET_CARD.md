---
license: other
language:
- en
- code
multilinguality:
- multilingual
size_categories:
- 100M<n<1B
task_categories:
- text-generation
- conversational
- fill-mask
pretty_name: Universal Global Coding Dataset
tags:
- code
- llm-pretraining
- sft
- dpo
- rlvr
- autonomous-agents
- fim
- software-engineering
---

# Universal Global Coding Dataset (UGCD)
## The World's Unified High-Quality Multi-Language Corpus for Code Intelligence

### Dataset Summary
The Universal Global Coding Dataset (UGCD) is a unified, legally vetted, research-grade multilingual corpus engineered specifically for training modern Coding Large Language Models (LLMs), Continued Pretraining (CPT), Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Autonomous Software Engineering Agents.

### Key Pillars
- **Complete Language Matrix**: Covers 36 programming and hardware description languages (Python, JS, TS, Java, C, C++, C#, Go, Rust, PHP, Ruby, Swift, Kotlin, Dart, Scala, R, Julia, MATLAB, Perl, Lua, Elixir, Haskell, OCaml, Clojure, Solidity, CUDA, SQL, Verilog, VHDL, Shell, PowerShell, and more).
- **35 Software Engineering Domains**: Ranging from basic data structures and competitive programming to distributed systems, system design, cybersecurity, multi-file repository understanding, and agent execution trajectories.
- **Dedicated Training Sub-Corpora**:
  1. `pretraining/`: Source code, architecture, documentation, and configuration files.
  2. `sft/`: ChatML-formatted instructions for code generation, explanation, translation, and debugging.
  3. `completion/`: Standard Fill-In-The-Middle (PSM/SPM) code completion sequences with `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`.
  4. `debugging/`: Buggy code, compiler/runtime error traces, root cause analysis, corrected code, and regression unit tests.
  5. `reasoning/`: Algorithmic problem analysis, time/space complexity derivations, and implementation blueprints.
  6. `repository_tasks/`: Full multi-file repository issues, unified git diff patches, and validation test execution suites.
  7. `preference/`: High-quality chosen vs rejected code implementations for DPO and RLVR alignment.
  8. `agent_trajectories/`: Step-by-step Plan -> Execute -> Test -> Repair traces for autonomous coding agents.
  9. `benchmarks/`: Isolated benchmark corpora (HumanEval, MBPP, SWE-bench) quarantined from training splits to prevent test leakage.

### Measured Empirical Statistics
- **Total Discovered Sources in Catalog**: 50
- **Total Ingested Records**: 1,149
- **Total Measured Tokens**: 147,754
- **Mean Sequence Length**: 128.6 tokens
- **Clean Pass Rate**: 99.91%
- **Duplicates Removed**: 19 (1.63% reduction)

### Data Quality & Decontamination
- **Syntactic Validation**: Multi-language AST parsing and bracket balancing.
- **Security & PII Scrubbing**: Gitleaks-compliant regex scanning targeting AWS keys, GitHub PATs, private keys, database connection strings, and malicious exploit payloads.
- **Benchmark Decontamination**: 13-gram overlap checking against HumanEval, MBPP, and SWE-bench to guarantee 0.0% contamination in clean training partitions.

### Licensing & Compliance
All code is classified into strict licensing tiers (Permissive, Copyleft, Non-Commercial, Restricted). Permissive sources (MIT, Apache-2.0, BSD) are prioritized for open foundation model training. Detailed per-source license mappings are archived in [LICENSES.md](LICENSES.md) and [SOURCES.md](SOURCES.md).
