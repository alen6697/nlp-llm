# Clinical NLP vs LLM for Readmission Prediction

## Overview

This project compares traditional NLP methods and large language models (LLMs) for predicting hospital readmission using clinical notes.

## Models

* TF-IDF + Logistic Regression
* Word2Vec + LSTM
* ClinicalBERT + Logistic Regression

## Dataset

MIMIC-III clinical notes dataset.

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the notebook:

```bash
jupyter notebook experiment.ipynb
```

## Results

* TF-IDF achieved the best performance
* LLM did not outperform traditional NLP

## Project Structure

* `experiment.ipynb`: main notebook
* `src/`:code files
* `requirements.txt`: dependencies

