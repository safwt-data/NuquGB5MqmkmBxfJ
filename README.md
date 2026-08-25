<h1 align="center">😊 Customer Happiness Prediction</h1>

<p align="center">
A machine learning project for predicting customer happiness and identifying
the variables that contribute most strongly to customer satisfaction.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-purple" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Model-Random%20Forest-green" alt="Random Forest">
  <img src="https://img.shields.io/badge/Hyperparmeter tunning -Random Search-brightgreen" alt="Status">
</p>
---

## 📑 Table of Contents

- [Overview](#overview)
- [Project Workflow](#project-workflow)
- [Machine Learning Models](#machine-learning-models)
- [Model Optimization](#model-optimization)
- [Feature Importance](#feature-importance)
- [Project Structure](#project-structure)
- [Technologies](#technologies)

---

## 🔎 Overview

The project focuses primarily on machine learning modeling and feature
importance analysis to predict customer happiness in a customer service company.

The analysis begins with Exploratory Data Analysis (EDA), including an
assessment of missing values and a correlation matrix. Feature engineering is
not included because the dataset contains only six variables, providing limited
opportunities to derive meaningful additional features.

---

## 🔄 Project Workflow

The project follows the following workflow:

1. Data loading
2. Exploratory Data Analysis
3. Missing value analysis
4. Correlation analysis
5. Machine learning model training
6. Model comparison
7. Random Forest hyperparameter optimization
8. Feature importance analysis
9. Permutation feature importance

---

## 🤖 Machine Learning Models

Several machine learning models are trained and evaluated to compare their
predictive performance.

The models are evaluated using appropriate classification metrics to determine
which approach performs best at predicting customer happiness.

---

## ⚙️ Model Optimization

The Random Forest model is explored further using `RandomizedSearchCV`.

A predefined number of hyperparameter combinations are randomly selected and
evaluated using cross-validation. The best-performing hyperparameters are then
used to construct the optimized Random Forest model.

---

## 📊 Feature Importance

Feature importance analysis is conducted using the optimized Random Forest
model to identify which variables contribute most and least to its predictions.

Permutation feature importance is also applied to both the training and test
datasets. Comparing the results helps determine whether important variables
provide stable predictive signals rather than reflecting noise.

This analysis provides insight into how strongly the model depends on each
variable when predicting customer happiness.

---

## 📁 Project Structure

    project/
    │
    ├── data/
    ├── models/
    │   └── project1/
    ├── notebooks/
    ├── project_1/
    ├── references/
    ├── reports/
    ├── .gitignore
    └── README.md

---

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Git / GitHub