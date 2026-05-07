# Selective Hybrid Ensemble for Lemon Leaf Disease Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Journal: Scientific Data](https://img.shields.io/badge/Journal-Scientific%20Data-brightgreen.svg)](https://www.nature.com/sdata/)

Official implementation of a high-performance deep learning framework for the automated classification of lemon leaf diseases. This project introduces a **Selective Weighted Soft Voting Ensemble** that achieves state-of-the-art accuracy through architectural diversity and confidence-based inference.

## 📌 Project Overview
Agricultural productivity is often threatened by pathological infections that are difficult to identify manually. This project provides a robust, end-to-end computer vision pipeline to detect seven distinct classes:
* **Anthracnose**
* **Citrus Canker**
* **Citrus Greening (Huanglongbing)**
* **Citrus Leaf Curl**
* **Citrus Leafminer**
* **Swallowtail Larval Herbivory**
* **Healthy Lemon Leaves**

## 🚀 Key Technical Features
* **Hybrid Ensemble Architecture:** Integrates four diverse backbones: **EfficientNetV2-S**, **EfficientNetB3**, **ResNet-50**, and **DenseNet-121**.
* **Selective Inference Logic:** Implements a confidence-based threshold (0.85). The system defaults to the lead model for high-confidence predictions but invokes the full weighted ensemble for uncertain cases, optimizing the trade-off between latency and precision.
* **Multi-GPU Training:** Fully optimized for NVIDIA Tesla T4 infrastructure using TensorFlow's `MirroredStrategy`.
* **Technical Validation:** Features Spearman correlation analysis between model predictions to statistically justify the ensemble's success.
* **Explainable AI (XAI):** Built-in support for **Grad-CAM** visualizations to provide clinical evidence for pathological diagnosis.

## 📊 Performance Summary
| Model | Accuracy | Precision | Recall | F1-Score | Cohen's Kappa |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hybrid Ensemble** | **96.36%** | **96.48%** | **96.36%** | **96.39%** | **0.9575** |

## 📁 Dataset & Methodology
* **Total Images:** 4,394 images of lemon leaves.
* **Data Splitting:** Two-stage **Stratified Shuffle Split** ensuring identical class distributions across sets:
    * **Training:** 80% (3,515 images)
    * **Validation:** 10% (439 images)
    * **Test (Hold-out):** 10% (440 images)
* **Optimization:** Two-phase fine-tuning with a 0.5 unfreeze fraction and a reduced learning rate ($10^{-4}$).

## 🛠️ Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
   cd YourRepoName
