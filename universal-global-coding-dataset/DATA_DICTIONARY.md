# UNIVERSAL GLOBAL CODING DATASET - DATA DICTIONARY

This document specifies the exact JSON schemas and field definitions across all sub-corpora.

## 1. Pretraining Corpus (`pretraining/source_code/`)
```json
{
  "content": "string (raw source code file content)",
  "language": "string (lowercase identifier, e.g. python, rust, go)",
  "file_type": "source_code | documentation | configuration",
  "domain": "string (one of 35 SE domains)",
  "source": "string (origin repository or dataset id)",
  "license": "string (verified SPDX license, e.g. mit, apache-2.0)",
  "repo_name": "string (origin GitHub / Software Heritage repo)",
  "path": "string (relative file path in repository)"
}
```

## 2. Instruction Fine-Tuning Corpus (`sft/`)
Format compatible with OpenAI ChatML format.
```json
{
  "messages": [
    {"role": "user", "content": "natural language instruction"},
    {"role": "assistant", "content": "verified code solution and explanation"}
  ],
  "language": "string",
  "task_type": "code_generation | code_explanation | code_translation | code_debugging | software_engineering",
  "domain": "string",
  "source": "string",
  "license": "string"
}
```

## 3. Fill-in-the-Middle (FIM) Completion Corpus (`completion/fim/`)
```json
{
  "prefix": "string (preceding code context)",
  "middle": "string (target completion code to predict)",
  "suffix": "string (succeeding code context)",
  "fim_type": "PSM | SPM",
  "formatted_prompt": "<|fim_prefix|> ... <|fim_suffix|> ... <|fim_middle|> ... <|endoftext|>",
  "language": "string",
  "source": "string",
  "license": "string"
}
```

## 4. Debugging & Error Repair Corpus (`sft/code_debugging/`)
```json
{
  "problem": "string (compiler error message, runtime exception, or stack trace)",
  "code_before": "string (buggy code implementation)",
  "analysis": "string (root cause diagnosis and defect explanation)",
  "code_after": "string (corrected bug-free code)",
  "tests": ["array of string unit test assertions"],
  "language": "string",
  "source": "string",
  "license": "string"
}
```

## 5. Algorithmic Reasoning Corpus (`reasoning/`)
```json
{
  "problem": "string (problem statement and mathematical constraints)",
  "reasoning_steps": "string (chain-of-thought analysis, strategy selection, trade-offs)",
  "complexity": {
    "time": "O(N log N)",
    "space": "O(N)"
  },
  "solution": "string (verified code implementation)",
  "language": "string",
  "source": "string",
  "license": "string"
}
```

## 6. Repository-Level Engineering Corpus (`repository_tasks/`)
SWE-bench style multi-file software engineering tasks.
```json
{
  "repo_name": "string (e.g. django/django)",
  "base_commit": "string (git commit SHA)",
  "issue": "string (issue description from GitHub)",
  "patch": "string (git unified diff patch resolving the issue)",
  "tests": ["array of test file targets"],
  "language": "string",
  "source": "string",
  "license": "string"
}
```

## 7. Preference & Alignment Corpus (`preference/`)
Format compatible with DPO, KTO, and RLVR trainers.
```json
{
  "prompt": "string (programming instruction or task)",
  "chosen": "string (higher quality, secure, idiomatic implementation)",
  "rejected": "string (suboptimal, buggy, or vulnerable implementation)",
  "reason": "string (detailed explanation of preference criteria)",
  "source": "string",
  "license": "string",
  "language": "string"
}
```

## 8. Autonomous Agent Trajectories (`agent_trajectories/`)
```json
{
  "goal": "string (high-level engineering objective)",
  "system_prompt": "string (agent operational prompt)",
  "trajectory": [
    {
      "thought": "string (agent internal reasoning)",
      "action": "string (tool invocation with parameters)",
      "observation": "string (tool output or compiler stderr)"
    }
  ],
  "final_status": "success | failure",
  "domain": "autonomous_software_engineering",
  "source": "string",
  "license": "string"
}
```
