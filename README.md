<div align="center">

# Interpretable Few-Shot Image Classification via Prototypical Concept-Guided Mixture of LoRA Experts

[![arXiv](https://img.shields.io/badge/arXiv-2506.04673-b31b1b.svg)](https://arxiv.org/abs/2506.04673)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#installation)

Zhong Ji, Rongshuai Wei, Jingren Liu, Yanwei Pang, Jungong Han

</div>

---

## 🔥 News
- **[2026-03-09]** Repository initialized.
- **[2026-03-09]** README released.
- **[TODO]** Training and evaluation code will be released.
- **[TODO]** Pretrained checkpoints will be released.
- **[TODO]** Reproducibility guide will be released.

---

## 🧩 Overview

This repository provides the official implementation for:

**Interpretable Few-Shot Image Classification via Prototypical Concept-Guided Mixture of LoRA Experts**

Self-Explainable Models (SEMs) rely on **Prototypical Concept Learning (PCL)** to make visual recognition more interpretable, but they often struggle in few-shot settings due to limited supervision. To address this issue, we propose a **Few-Shot Prototypical Concept Classification (FSPCC)** framework that systematically tackles two core challenges in low-data regimes:

- **parametric imbalance** between the backbone and the concept learning module,
- **representation misalignment** between feature embeddings and concept activations.

Our method introduces:

- a **Mixture of LoRA Experts (MoLE)** for parameter-efficient adaptation,
- **cross-module concept guidance** to align feature representations with prototypical concept activations,
- a **multi-level feature preservation strategy** to fuse spatial and semantic cues from multiple layers,
- a **geometry-aware concept discrimination loss** to reduce concept overlap and improve interpretability.

Experimental results on six popular benchmarks show that our method consistently outperforms existing self-explainable models by a clear margin in few-shot image classification.

---

## ✨ Highlights

- An interpretable few-shot image classification framework built on **prototypical concept learning**
- A **parameter-efficient** adaptation design using **Mixture of LoRA Experts (MoLE)**
- Cross-module concept guidance for better feature-concept alignment
- Multi-level feature preservation for stronger low-data representations
- Geometry-aware concept discrimination for clearer and less-overlapping concepts
- Strong results on **CUB-200-2011, mini-ImageNet, CIFAR-FS, Stanford Cars, FGVC-Aircraft, and DTD**

---
