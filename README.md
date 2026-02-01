# Machine Learning Assignment

This repository contains various Machine Learning algorithms implemented in Python as part of an ML course assignment. The scripts demonstrate a wide range of techniques, from classical supervised learning to modern transformer-based models and unsupervised clustering.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Algorithms Implemented](#algorithms-implemented)
  - [Supervised Learning](#supervised-learning)
  - [Unsupervised Learning](#unsupervised-learning)
  - [Advanced Techniques](#advanced-techniques)
- [How to Run](#how-to-run)

## Overview

The goal of this assignment is to explore and implement different machine learning paradigms using real-world datasets. Each script is self-contained and downloads its required dataset automatically using the `kagglehub` library.

## Requirements

To run these scripts, you will need:

- Python 3.x
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `seaborn`
- `kagglehub` (for dataset management)
- `torch` (for RNN and Transformer models)
- `xgboost`, `catboost` (for ensemble methods)

You can install the dependencies using:

```bash
pip install pandas numpy matplotlib scikit-learn seaborn kagglehub torch xgboost catboost
```

## Datasets

The scripts utilize several public datasets from Kaggle:

- **Iris Dataset**: Used for classification tasks (AdaBoost, CatBoost, XGBoost, etc.).
- **Mall Customers Dataset**: Used for clustering analysis.
- **Date Fruit Dataset**: Used for complex classification with Transformers.

## Algorithms Implemented

### Supervised Learning

- **AdaBoost** (`adaboost.py`): Adaptive Boosting for classification.
- **CatBoost** (`catboost.py`): Categorical Boosting.
- **XGBoost** (`xgboost.py`): Extreme Gradient Boosting.
- **Random Forest** (`rfc.py` & `rfr.py`): Implementation of Forest-based Classification and Regression.
- **SVM** (`svm.py`): Support Vector Machine classification.
- **MLP** (`mlp.py`): Multi-Layer Perceptron neural network.

### Unsupervised Learning

- **K-Means** (`k_means.py` & `modified_k_means.py`): Centroid-based clustering.
- **Fuzzy C-Means** (`fuzzy_c_meas.py`): Soft clustering where points belong to multiple clusters.
- **DBSCAN/HDBSCAN** (`hdbscan_and_dbscan.py`): Density-based spatial clustering.
- **Hierarchical Clustering** (`hierarchical.py`): Connectivity-based clustering.
- **SOM** (`som.py`): Self-Organizing Maps.

### Advanced Techniques

- **RNN** (`rnn.py`): Recurrent Neural Network for sequential data processing.
- **GRNN** (`grnn.py`): Generalized Regression Neural Network.
- **Transformer (LLM-based)** (`llm.py`): A Transformer model implemented in PyTorch for tabular classification.
- **Ensemble Stacking** (`ensemble_stacking.py`): Combining multiple models for better performance.
- **Semi-Supervised Learning** (`semi_supervised.py`): Utilizing both labeled and unlabeled data.
- **HMM** (`hmm.py`): Hidden Markov Models.

## How to Run

Simply execute any script using Python:

```bash
python xgboost.py
```

The script will handle dataset downloading, preprocessing, training, and evaluation.
