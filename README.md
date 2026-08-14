# 🛠️ Qwen2.5-7B-GRPO-ToolCalling

[![Model](https://img.shields.io/badge/Model-Qwen2.5--7B-blue)](https://huggingface.co/Qwen)
[![Framework](https://img.shields.io/badge/Framework-TRL%20%7C%20PEFT%20%7C%20vLLM-orange)](#)
[![Method](https://img.shields.io/badge/RL-GRPO%20%2B%20LoRA-green)](#)
[![Hardware](https://img.shields.io/badge/Hardware-1x%20NVIDIA%20A100%20(80GB)-purple)](#)
[![Benchmark](https://img.shields.io/badge/BFCL%20Accuracy-75%25%20(%2B8%25)-brightgreen)](#)

A high-efficiency Reinforcement Learning (RL) fine-tuning pipeline for **Qwen2.5-7B** that drastically improves structured function/tool-calling accuracy. By pairing **LoRA** with **Group Relative Policy Optimization (GRPO)** and replacing traditional critic networks with deterministic AST/API rewards, this model achieves reliable, schema-compliant tool calls trained on a single **A100 (80GB)** GPU.

---

## ✨ Key Features & Improvements

* **Critic-Free RL via GRPO:** Eliminates the memory and training instability of learned critic/value networks by leveraging relative reward normalization across group samples.
* **Deterministic AST/API Rewards:** Replaces LLM-based or learned reward models with strict, rule-based Abstract Syntax Tree (AST) parsing and JSON Schema validation.
* **Curated Training Set:** Trained on **500 execution-verified trajectories** filtered from ToolBench to ensure zero false-positive tool invocations.
* **Single-GPU Friendly:** Fully trainable on a single NVIDIA A100 GPU utilizing LoRA adapter parameter efficiency.

---

## 📊 Benchmark Results

Evaluated on a subset of the **Berkeley Function Calling Leaderboard (BFCL)**:

| Model / Configuration | Training Method | BFCL Function Calling Acc |
| :--- | :--- | :---: |
| **Qwen2.5-7B-Instruct (Baseline)** | Zero-Shot Prompting | **67.0%** |
| **Qwen2.5-7B-ToolCalling (Ours)** | **LoRA + GRPO (AST Reward)** | **75.0% (+8.0%)** |

> **Key Takeaway:** Deterministic schema rewards force the policy to stop making hallucinated argument keys and syntax errors, yielding a **+8% absolute improvement** over the baseline model.

---

## 🎯 Reward Engineering Architecture

Traditional RLHF/RLAIF uses a learned critic network that can be noisy or prone to reward hacking. Instead, our GRPO framework passes generated completions through a **deterministic evaluation function**:
