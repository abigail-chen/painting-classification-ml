# What Does a Painting Taste Like?

### Predicting Painting Identity from Survey Responses

**Machine Learning | Python | Random Forest | NLP |Logistic Regression |Feature Engineering**

## Overview

This project explores whether subjective descriptions of artwork can be used to predict painting identity. Using survey responses from 562 students, we developed a multi-class classification pipeline to distinguish among three paintings: *The Persistence of Memory*, *The Starry Night*, and *The Water Lily Pond*.

The dataset contains 1,686 observations and combines numerical ratings, Likert-scale responses, categorical selections, and open-ended text.

We compared Logistic Regression, Multi-Layer Perceptron (MLP), and Random Forest models to evaluate their ability to learn from heterogeneous features.

## Methodology

**1. Data Preprocessing and Feature Engineering**

- Applied median and mode imputation to handle missing values.
- Used logarithmic transformations for highly skewed numerical variables.
- Encoded categorical and multi-select responses.
- Applied TF-IDF vectorization with unigrams and bigrams to extract information from free-text responses.

**2. Model Development**

Implemented and compared three machine learning models:

- Multiclass Logistic Regression
- Multi-Layer Perceptron (MLP)
- Random Forest Classifier

**3. Model Validation**

- Performed student-level data splitting to prevent leakage across responses from the same participant.
- Used 5-fold cross-validation for model selection and hyperparameter tuning.
- Evaluated performance using accuracy, macro-F1, and confusion matrices.

## Results

| Model | Test Accuracy | Test Macro-F1 |
|---|---:|---:|
| Logistic Regression | 68.9% | 0.677 |
| MLP | 86.4% | 0.866 |
| **Random Forest** | **89.1%** | **0.891** |

The Random Forest achieved the best overall performance, benefiting from its ability to capture nonlinear relationships across mixed numerical, categorical, and textual features.

## My Contributions

As part of a four-member team, I was responsible for training and tuning the Random Forest model and documenting its methodology.

My work focused on evaluating model configurations, comparing classification performance, and examining the effectiveness of ensemble learning for heterogeneous survey data.

## Repository Contents

- `code/` — Model development and experimentation
- `inference/` — Prediction script and serialized model parameters
- `project_report.pdf` — Complete team project report

## Inference

The prediction interface is provided in `inference/pred.py`. It accepts survey responses in CSV format and returns predicted painting labels.

Example usage:

```python
from inference.pred import predict_all

predictions = predict_all("path/to/test_data.csv")
```

The input CSV must follow the expected survey-response schema. The original training data are not included in this repository.

## Project Context

CSC311: Introduction to Machine Learning  
University of Toronto, Winter 2026

**Team:** Abigail Xi Chen, Zihan Jin, Qifan Yang, Ruoyu Zhou.
