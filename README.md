# Breast Cancer Diagnostic Classification: KNN Pipeline

## 📌 Project Overview
This repository contains an end-to-end machine learning pipeline that leverages the K-Nearest Neighbors (KNN) algorithm to classify breast mass tumors as Malignant or Benign. Utilizing the Breast Cancer Wisconsin (Diagnostic) dataset, the project covers exploratory data analysis (EDA), custom distance-metric implementation, hyperparameter optimization, and high-dimensional decision boundary visualization.

---

## ⚙️ Core Architecture & Pipeline Stages

The classification pipeline is broken down into structured, production-ready modules:

### 1. Exploratory Data Analysis (EDA) & Profiling
*   Evaluates feature distributions across 30 distinct geometric and texture-based cell nuclei variables.
*   Maps class distributions and handles structural dependencies via multi-variable correlation matrices to identify highly collinear features.

### 2. Standardized Data Preprocessing
*   **Stratified Data Splitting:** Ensures identical class distribution proportions between the training and testing matrices to mitigate dataset bias.
*   **Leakage-Free Feature Scaling:** Implements z-score standardization (`StandardScaler`) calculated exclusively on training partitions to prevent data leakage during distance calculations.

### 3. Core Modeling & Custom Optimization
*   **Custom KNN Engine:** Features a scratch-built KNN classification matrix algorithm constructed without deep learning framework dependencies.
*   **Algorithmic Distance Calibration:** Evaluates the statistical trade-offs and accuracy changes between **Euclidean** and **Manhattan** distance metrics.
*   **Hyperparameter Cross-Validation:** Optimizes the neighborhood parameter ($k$) using a 5-fold cross-validation sweep ($1 \leq k \leq 30$) to manage the bias-variance trade-off.

---

## 📊 Evaluation & Diagnostics Suite

Given the critical nature of medical diagnostic workflows, the system evaluates the model using several precision metrics:

1.  **Confusion Matrix Analysis:** Segregates True Positives against False Negatives to closely monitor and minimize missed clinical cases.
2.  **ROC / AUC Curve Profiling:** Gauges the true positive rate against the false positive rate across multiple thresholds to assess the classifier’s discriminative capacity.
3.  **Dimensional Boundary Projection:** Uses Principal Component Analysis (PCA) to reduce the 30-dimensional feature space to a 2D map, plotting a detailed meshgrid decision boundary overlaid with test results.
4.  **Framework Benchmarking:** Runs a comparative validation check between the custom-built KNN algorithm and standard `scikit-learn` baseline estimators.

---

## 🛠️ Environment & Tooling
*   **Language:** Python 3.x
*   **Data Structures:** `pandas`, `numpy`
*   **Pipeline Utilities:** `scikit-learn` (Preprocessing, Validation splits, PCA modules)
*   **Visualization Engine:** `matplotlib`, `seaborn`

---

## 🚀 Execution & Setup

### 1. Dependencies
Install the required data science packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn ucimlrepo
```

### 2. Running the Pipeline
Execute the main script to run data exploration, model tuning, evaluation metrics, and generate visual plots:
```bash
python main.py
```

---

## 📁 Repository Architecture
*   `main.py` - Core execution script containing EDA workflows, custom KNN loops, and evaluation engines.
*   `README.md` - Technical infrastructure and feature documentation.
*   `.gitignore` - Pre-configured rule tracking blocks preventing runtime system configurations from staining version histories.

