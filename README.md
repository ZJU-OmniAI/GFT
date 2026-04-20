# GFT
GFT: From Imitation to Reward Fine-Tuning with Unbiased Group Advantages and Dynamic Coefficient Rectification

<div align="center">

[[📖 Paper]()] [[🤗 Qwen-2.5-Math-1.5B-GFT]()] [[🤗 NuminaMath-Cot-Distillation-10W]()]

</div>

## 🔥 News
- [2025/04/06] Our work is accepted by ACL 2026 Findings.🎇🎇🎇

## 👀 Introduction
<img src="./docs/intro.png" style="zoom:100%;" />

## 🧠 Method
<img src="./docs/method.png" style="zoom:100%;" />

## 🏆 Performance
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

