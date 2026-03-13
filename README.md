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
- **[2026-03-13]** Training and evaluation code will be released.
- **[2026-03-13]** Pretrained checkpoints will be released.
- **[2026-03-13]** Reproducibility guide will be released.

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

## 📁 Repository Structure

The current repository is organized as follows:

```text
LE-ProtoPNet/
├── configs/                 # experiment configuration files
├── data/                    # dataset storage / preprocessing utilities
├── evaluation/              # evaluation metrics and testing utilities
├── models/                  # backbone networks, concept modules, MoLE implementation
├── shared_utils/            # common utilities (logging, helpers, training utils)
├── train_fsl.py             # few-shot training entry point
├── test_fsl.py              # few-shot evaluation script
└── env.yaml                 # conda environment specification
```

---

## ⚙️ Installation

### 1. Create Conda Environment

```bash
conda env create -f env.yaml
conda activate LEProtoPNet
```

---

## 📊 Supported Datasets

Our framework supports several commonly used **few-shot image classification benchmarks**:

* **CUB-200-2011**
* **mini-ImageNet**
* **CIFAR-FS**
* **Stanford Cars**
* **FGVC Aircraft**
* **DTD (Describable Textures Dataset)**

---

### Dataset Directory Structure

After downloading the datasets, organize them as follows:

```text
data/
├── cub/
│   ├── images
│   └── splits
├── mini_imagenet/
│   ├── images
│   └── splits
├── cifar_fs/
│   └── splits
├── stanford_cars/
├── fgvc_aircraft/
└── dtd/
```

Dataset split files should follow the standard **few-shot meta-learning splits** used in prior work.

---

## 🏋️ Training

The training pipeline is implemented in **`train_fsl.py`**.

```bash
python train_fsl.py
```

---

## 🧪 Evaluation

Evaluation is implemented in **`test_fsl.py`**.

```bash
python test_fsl.py
```

---

## 📜 Citation

If you find this repository useful for your research, please cite:

```bibtex
@article{ji2026interpretable,
  title={Interpretable Few-Shot Image Classification via Prototypical Concept-Guided Mixture of LoRA Experts},
  author={Ji, Zhong and Wei, Rongshuai and Liu, Jingren and Pang, Yanwei and Han, Jungong},
  journal={IEEE Transactions on Image Processing},
  year={2026},
  publisher={IEEE}
}
```

---

## ⭐ Star History

<p align="center">
  <img src="https://api.star-history.com/svg?repos=Programmergg/LE-ProtoPNet&type=Date" width="650">
</p>

---

## 📬 Contact

For questions, suggestions, or collaborations:

* **Email:** [jrl0219@tju.edu.cn](mailto:jrl0219@tju.edu.cn)
* **GitHub Issues:** Please open an issue in this repository.

---
