# WA-RoPE-Transformer-HSI-Skin-Cancer-Classification
This repository provides the implementation of a WA-RoPE Transformer-based framework for hyperspectral skin cancer classification. It includes preprocessing, training, and testing modules, along with a dataset link and documentation, enabling reproducibility and facilitating further research and development.

This repository contains the official implementation of our manuscript.

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

📌 **Citation**: If you use this code, please cite:




