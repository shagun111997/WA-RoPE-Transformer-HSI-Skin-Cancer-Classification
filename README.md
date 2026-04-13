# WA-RoPE-Transformer-HSI-Skin-Cancer-Classification

This repository is archived on Zenodo for reproducibility:
DOI: https://doi.org/10.5281/zenodo.19551472

It provides the implementation of a WA-RoPE Transformer-based framework for hyperspectral skin cancer classification. This repository includes preprocessing, training, and testing modules, along with a dataset link and documentation, enabling reproducibility and facilitating further research and development.

This repository contains code directly associated with a manuscript currently submitted at *The Visual Computer*.

If you use this code in your research, please cite the corresponding manuscript. Full citation details will be updated upon publication.

## 📌 Overview
We propose a **Wavelength-Aware Rotary Positional Embedding (WA-RoPE) Transformer** for hyperspectral skin cancer classification.

Key features:
- SNV preprocessing
- Spectral augmentation (noise + masking)
- Transformer-based feature extraction
- Focal loss for class imbalance

---

## 📂 Dataset Availability

The dataset used in this study is **publicly available**:

🔗 Original Source: https://data.mendeley.com/datasets/j9773cyr3k/1)
Related Data Article: Skin cancer diagnosis using NIR spectroscopy data of skin lesions in vivo using machine learning algorithms
Data Article link: https://arxiv.org/abs/2401.01200 

---

## ⚙️ Installation

```bash
pip install -r requirements.txt

🚀 **How to Run**
Step 1: Preprocessing
python scripts/preprocessing.py
Step 2: Training
python scripts/train.py
Step 3: Testing
python scripts/test.py

🔁 **Reproducibility**
Train-test split: 80/20 (stratified)
Preprocessing: SNV normalization
Augmentation:
Spectral noise injection
Random band masking
Loss: Focal Loss
Optimizer: Adam
Scheduler: Cosine Annealing

📊 **Evaluation Metrics**
Accuracy
Precision
Recall
F1-score
ROC-AUC






