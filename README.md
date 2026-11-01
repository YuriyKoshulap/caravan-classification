# Caravan Insurance Classification

A small Python machine-learning project using K-nearest neighbours (KNN) to classify whether customers purchased caravan insurance.

This repository is a standalone portfolio version of coursework I completed through the **University of Toronto Data Sciences Institute**. I reorganized and expanded the original assignment to make the analysis easier to follow and more reproducible.

## What this project demonstrates

- exploratory inspection of a tabular dataset
- class-imbalance awareness
- reproducible train/test splitting
- feature scaling with `StandardScaler`
- K-nearest-neighbours classification
- scikit-learn pipelines
- comparison against a majority-class baseline
- model evaluation using accuracy, balanced accuracy, precision, recall, F1 and a confusion matrix

## Dataset

The project uses the `Caravan` dataset distributed with the `ISLP` Python package.

The dataset contains **5,822 observations**, **85 predictor variables**, and a binary response variable, `Purchase`. In the original coursework dataset, only about **6%** of observations are purchasers, making class imbalance an important part of the evaluation.

## Why the portfolio version differs from the original assignment

The original exercise introduced the essential KNN workflow. For this version I made several methodological improvements:

- a fixed, reproducible `random_state`
- a **stratified** 75/25 train/test split
- scaling fitted on the **training data only**
- preprocessing and modelling combined in a scikit-learn `Pipeline`
- evaluation beyond raw accuracy
- comparison with a simple majority-class baseline
- a small sensitivity check across several values of `k`

## Repository structure

```text
.
├── notebooks/
│   └── caravan_knn_classification.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Run locally

Create a virtual environment, install the dependencies, and open the notebook:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Original coursework result

In my original assignment run, the dataset contained 5,822 rows and 85 predictors, with a purchase rate of approximately 5.98%. The assignment compared KNN models using `k=1` and `k=3`.

The portfolio notebook intentionally reruns the analysis using a cleaner reproducible workflow, so its metrics may differ from the original assignment output.

## Provenance

Adapted from coursework completed in the **Machine Learning Software Foundations Certificate, University of Toronto Data Sciences Institute (2024)**.

The original course materials remain credited to the University of Toronto Data Sciences Institute; this repository presents my completed analysis as a separate portfolio project.
