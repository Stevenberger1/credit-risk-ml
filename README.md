#  Credit Risk & Loan Default Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/sklearn-latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Project Overview
This project implements a **binary classification system** designed to distinguish between "good" and "bad" credit risks using the **German Credit Dataset**.

Unlike standard "Kaggle-style" competitions, this project prioritizes **principled ML engineering**. The focus is on designing a robust system that accounts for class imbalance, asymmetric error costs (false negatives vs. false positives), and model interpretability within a real-world financial decision-making context.

###  Key Focus Areas
* **Decision-Centric Evaluation:** Moving beyond accuracy to optimize thresholds based on business impact.
* **Rigorous Pipeline Design:** Preventing data leakage through strict scikit-learn pipelines.
* **Model Interpretability:** Using feature importance and error analysis to justify credit decisions.
* **Reproducibility:** Ensuring consistent results via fixed seeds and modular code architecture.

---

##  Dataset & Target Definition

### Data Source
We utilize the **German Credit Dataset (`credit-g`)** via OpenML. 
* **Access:** Fetched programmatically using `sklearn.datasets.fetch_openml`.
* **Features:** Contains demographic (age, housing), financial (checking account status, savings), and loan-specific (duration, purpose) attributes.

### Target Mapping
In this system, we define the **Positive Class (1) as "Bad Credit Risk"**.

> **Rationale:** In banking, the cost of a Type II error (approving a loan for a "bad" applicant) is significantly higher than the opportunity cost of a Type I error (rejecting a "good" applicant). Treating "Bad Risk" as the positive class aligns our metrics (Recall, Precision) with risk mitigation goals.

---

##  Evaluation Philosophy
We treat machine learning as a tool for **decision quality**, not just label prediction.

| Metric | Importance in this Project |
| :--- | :--- |
| **Recall (TPR)** | **High:** Critical for catching as many "Bad Risks" as possible. |
| **PR-AUC** | **High:** Superior to ROC-AUC for evaluating imbalanced credit data. |
| **Threshold Tuning** | **Critical:** We explicitly tune the classification threshold rather than defaulting to $0.5$. |
| **Cost Analysis** | **High:** Evaluation focuses on the asymmetric cost of False Negatives. |

---

##  Model & Engineering Strategy

###  Candidate Models
We compare a variety of architectures to understand different decision boundaries:
* **Linear:** Logistic Regression (Baseline for interpretability).
* **Ensemble (Bagging):** Random Forest.
* **Ensemble (Boosting):** XGBoost & LightGBM.

### Best Practices
* **Stratified Splitting:** Preserves the minority class ratio across Train/Test sets.
* **Leakage Prevention:** Preprocessing parameters (mean, scaling, etc.) are learned *only* from the training split.
* **Cross-Validation:** Performed exclusively on the training set to guide model selection.

---

## Project Structure

```graphql
credit-risk-ml/
├── data/               # Cached artifacts (no manual downloads)
├── notebooks/
│   └── 01_eda.ipynb    # Visualizing distributions and correlations
├── src/                # Modular Source Code
│   ├── config.py       # Global constants and hyperparams
│   ├── data.py         # Data fetching and stratified splitting
│   ├── preprocessing.py# Scikit-learn Pipelines (scaling, encoding)
│   ├── models.py       # Model factory and definitions
│   ├── train.py        # CV logic and training loops
│   ├── evaluate.py     # Threshold tuning and metric reporting
│   ├── interpret.py    # Feature importance and SHAP/Permutation analysis
│   └── utils.py        # Logging and helper utilities
├── outputs/            # Generated artifacts
│   ├── figures/        # PR Curves, Confusion Matrices, ROC
│   ├── models/         # Serialized (.joblib) model files
│   └── reports/        # Performance summaries
└── requirements.txt    # Dependency management