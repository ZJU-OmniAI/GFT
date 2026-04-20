<h1 align="center"> GFT: Group Fine-Tuning </h1>
GFT: From Imitation to Reward Fine-Tuning with Unbiased Group Advantages and Dynamic Coefficient Rectification

<div align="center">

[[📖 Paper]()] [[🤗 Qwen-2.5-Math-1.5B-GFT]()] [[🤗 NuminaMath-Cot-Distillation-10W]()]

</div>

## 🔥 News
- [2025/04/06] Our work is accepted by ACL 2026 Findings.🎇🎇🎇

## 👀 Introduction
Large language models (LLMs) rely heavily on Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). However, standard SFT suffers from two critical flaws when viewed from a training-dynamics perspective:
1. **Single-path dependency:** Implicit reward strictly forces imitation, causing entropy collapse.
2. **Gradient explosion:** Unstable inverse-probability weighting leads to mechanical memorization and catastrophic forgetting.
**Group Fine-Tuning (GFT)** is a unified single-stage post-training framework designed to bridge the gap between efficient knowledge injection and robust exploration. GFT mitigates the intrinsic deficiencies of SFT and serves as an optimal initialization for downstream RL (like GRPO or PPO).
<img src="./docs/intro.png" style="zoom:100%;" />

## 🧠 Method
* **Group Advantage Learning (GAL):** Constructs a diverse response group $\mathcal{G}_x$ (Expert + Teacher Distillation + Self-Sampled) and evaluates candidates via normalized contrastive supervision.
* **Dynamic Coefficient Rectification (DCR):** Adaptively bounds inverse-probability weights $1/\pi_{\theta}(y|x)$ to stabilize optimization without losing the capability to inject new knowledge.
<img src="./docs/method.png" style="zoom:100%;" />

## 🏆 Performance
* **Exceptional Data Efficiency:** With only **10k** training samples, GFT surpasses standard SFT trained on **100k** samples across multiple math-reasoning benchmarks (AMC23, MATH, OlympiadBench, etc.).
* **Solving the Synergy Dilemma:** Conventional `SFT -> GRPO` pipelines often underperform. GFT preserves policy entropy and diverse reasoning paths, making the `SFT -> GFT -> GRPO` pipeline yield a significantly higher performance ceiling.
<img src="./docs/performance_1.png" style="zoom:100%;" />
<img src="./docs/performance_2.png" style="zoom:100%;" />

## ⚙️ Set up

```bash
git https://github.com/ZJU-OmniAI/GFT.git
cd GFT

conda create -n DFT python=3.10 -y
conda activate DFT
cd verl
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

