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

The scripts utilize several public datasets from Kaggle and other sources:

- **Iris Dataset** (`uciml/iris`): Classic botanical dataset used for multi-class classification (AdaBoost, CatBoost, XGBoost, Random Forest).
- **Mall Customers Dataset** (`vjchoudhary7/customer-segmentation-tutorial-in-python`): Data for identifying customer segments based on annual income and spending score (K-Means).
- **Bangladesh Districts Population** (`msjahid/bangladesh-districts-wise-population`): Geographic and demographic data for clustering districts (Fuzzy C-Means, DBSCAN, HDBSCAN).
- **Animal Crossing Villagers** (`jessicali9530/animal-crossing-new-horizons-nookplaza-dataset`): Categorical data on game characters used for clustering and hierarchical analysis (Hierarchical, Modified K-Means).
- **Pumpkin Seeds Dataset** (`muratkokludataset/pumpkin-seeds-dataset`): Morphological features of pumpkin seeds for classification (GRNN, SOM).
- **European Soccer Database** (`hugomathien/soccer`): Historical match data used for modeling team performance cycles (HMM).
- **Date Fruit Dataset** (`muratkokludataset/date-fruit-datasets`): Image-derived features of date fruits for classification (Transformer).
- **Uber Ride Analytics** (`yashdevladdha/uber-ride-analytics-dashboard`): Booking data used for ride status prediction and time-series forecasting (MLP, RNN).
- **COVID-19 Patient Info** (`kimjihoo/coronavirusdataset`): Patient data used for survival prediction in a semi-supervised setting.
- **California Housing** (Built-in `sample_data`): Geographic and economic data used for housing value classification (SVM).

## Algorithms Implemented

### Supervised Learning

- **AdaBoost** (`adaboost.py`): An adaptive boosting algorithm that combines multiple weak classifiers (Decision Stumps) to form a strong one. Uses the **Iris** dataset.
- **CatBoost** (`catboost.py`): A high-performance gradient boosting library that specifically handles categorical features efficiently. Uses the **Iris** dataset.
- **XGBoost** (`xgboost.py`): An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. Uses the **Iris** dataset.
- **Random Forest Classifier** (`rfc.py`): An ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes. Uses the **Iris** dataset.
- **Random Forest Regressor** (`rfr.py`): Similar to RFC but used for regression tasks (predicted values are rounded for accuracy metrics here). Uses the **Iris** dataset.
- **SVM** (`svm.py`): Support Vector Machine classifier with an RBF kernel, used to predict if housing values are above or below the median. Uses the **California Housing** dataset.
- **MLP** (`mlp.py`): A Multi-Layer Perceptron neural network with two hidden layers (64, 32 units). It predicts Uber booking status based on ride details.

### Unsupervised Learning

- **K-Means** (`k_means.py`): Segments mall customers into 5 distinct groups based on income and spending habits using centroid-based clustering.
- **Modified K-Means** (`modified_k_means.py`): Applies K-Means clustering to **Animal Crossing** villagers based on encoded personality and species features.
- **Fuzzy C-Means** (`fuzzy_c_meas.py`): A "soft" clustering algorithm where each data point can belong to more than one cluster with varying degrees of membership. Clusters **Bangladesh districts**.
- **DBSCAN/HDBSCAN** (`hdbscan_and_dbscan.py`): Compares density-based clustering (DBSCAN) with its hierarchical version (HDBSCAN) to identify clusters of varying density and noise points in **Bangladesh districts**.
- **Hierarchical Clustering** (`hierarchical.py`): Uses Ward's linkage to create a dendrogram of **Animal Crossing** villagers, visualizing their taxonomic relationships.
- **SOM** (`som.py`): Self-Organizing Map (a type of artificial neural network) used for dimensionality reduction and visualization of **Pumpkin Seed** varieties.

### Advanced techniques

- **RNN** (`rnn.py`): A Recurrent Neural Network with SimpleRNN layers and Dropout, used for time-series forecasting of Uber ride counts.
- **GRNN** (`grnn.py`): Generalized Regression Neural Network, a variation of radial basis neural networks, used for pumpkin seed classification.
- **Transformer (LLM-based)** (`llm.py`): A PyTorch implementation of a Transformer encoder for tabular data classification on the **Date Fruit** dataset.
- **Ensemble Stacking** (`ensemble_stacking.py`): A meta-model approach that combines Random Forest, KNN, and SVM using a Logistic Regression meta-classifier, further wrapped in a Voting Classifier.
- **Semi-Supervised Learning** (`semi_supervised.py`): Uses Label Spreading to propagate labels from a small set of known COVID-19 patient outcomes to a larger unlabeled set.
- **HMM** (`hmm.py`): Hidden Markov Model used to identify hidden "form" states (Slump, Average, Peak) in FC Barcelona's match history.

## How to Run

Simply execute any script using Python:

```bash
python xgboost.py
```

The script will handle dataset downloading, preprocessing, training, and evaluation.
