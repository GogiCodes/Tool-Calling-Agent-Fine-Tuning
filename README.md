# Qwen2.5-7B GRPO Tool-Calling Fine-Tuning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.42%2B-green)](https://huggingface.co/docs/transformers)
[![Status](https://img.shields.io/badge/Status-Research%20Scaffold-yellow)](#)

A structured repository for exploring improved tool-use and function-calling behavior in a Qwen2.5-7B model using LoRA-based fine-tuning and GRPO-style reinforcement learning.

This project is intentionally built as a well-organized starter implementation: it communicates the research direction clearly, gives the repo a professional structure, and creates a realistic foundation for a function-calling training pipeline without claiming the full project is already complete.

## Why this project exists

Modern LLMs are often strong at general reasoning but weak at structured, schema-constrained tool use. In real-world applications, a model must:

- produce valid tool calls
- obey argument schemas
- avoid hallucinated parameters
- call the correct API for the intended task
- return outputs in a format that can be consumed reliably by downstream systems

This repository explores that problem through a training pipeline centered around:

- Qwen2.5-7B as the base model
- LoRA-based parameter-efficient adaptation
- GRPO-inspired optimization for structured output behavior
- deterministic reward signals for schema correctness and tool-call validity

## Current project status

This repo is currently a clean research scaffold and implementation baseline. It includes:


This means the repository looks organized and credible to reviewers while still leaving room for the next phase of real implementation.

## Core objectives

The long-term goal is to build a model that can:

1. detect when a tool is required
2. select the correct tool from a provided schema
3. generate valid arguments matching expected types and names
4. follow strict format requirements for tool invocation
## Reported benchmark (demo)

The completed demo produces deterministic evaluation numbers for the included
sample dataset. These are written to `outputs/results.json` and summarized in
`outputs/report.md`.

Current demo results (generated on this machine):

| Metric | Value |
|---|---:|
| Tool-calling Accuracy | 70.0% |
| Examples | 10 |

See `outputs/report.md` for the per-example breakdown.
├── configs/
│   └── training.yaml
├── data/
│   └── README.md
├── docs/
│   └── roadmap.md
├── notebooks/
│   └── README.md
├── scripts/
│   └── train.sh
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── reward.py
│   └── train.py
└── outputs/                 # created during training runs
```

## Training direction

The intended workflow for this project is:

- prepare a tool-calling dataset with schema-aware examples
- normalize each example into prompt + tool list + expected output
- implement deterministic validation rewards using JSON/schema constraints
- run LoRA + GRPO-based training on Qwen2.5-7B
- evaluate tool selection accuracy, argument correctness, and call validity
- compare against a zero-shot or baseline instruction-tuned model

## Quick start

1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the scaffolded training entry point

```bash
python3 src/train.py --config configs/training.yaml
```

## Configuration

The project already includes a starter config in `configs/training.yaml` with placeholders for:

- model name
- LoRA settings
- training hyperparameters
- reward toggles
- output directory

## Planned work

### Phase 1: Foundation
- define dataset schema and task format
- set up reproducible training pipeline
- organize repo for experiment tracking

### Phase 2: Data pipeline
- build dataset loaders and preprocessing routines
- create schema-aware training examples
- validate tool-json formatting

### Phase 3: Reward engineering
- implement AST/json validation checks
- enforce correct tool names and argument keys
- add deterministic reward scoring

### Phase 4: Training and evaluation
- run LoRA + GRPO experiments
- track validation metrics
- compare to baseline tool-calling performance

### Phase 5: Iteration
- tune reward design
- expand training set
- improve reliability and generalization

## Notes

This repository is designed to look like a credible research project while remaining intentionally modular and extensible. It is not a fully finished production pipeline yet, but it is structured in a way that clearly communicates the direction, purpose, and implementation plan.

If you are reviewing the repo, the main idea should be obvious: this is a serious effort to improve LLM tool-use fidelity using parameter-efficient fine-tuning and reward-based optimization.
