# Cross-Lingual Sentiment Analysis with Linear Models

## Project Overview

This project investigates cross-lingual sentiment transfer from English to Spanish using linear classification models. We train models on English Amazon reviews and evaluate their zero-shot performance on Spanish reviews.

## Key Results

- **Best Model:** Soft Voting + BoW
- **Best F1-Score:** 0.6997
- **Average Cross-Lingual Gap:** 28.76%
- **Total Models Evaluated:** 14

## Dataset

- **Name:** Amazon Reviews Multi
- **Source:** Kaggle (mexwell)
- **English Samples:** 24,985
- **Spanish Samples:** 24,905
- **Task:** Binary sentiment classification

## Models

### Individual Models
1. Logistic Regression
2. Ridge Classifier
3. Linear SVM
4. SGD Classifier
5. Multinomial Naive Bayes

### Ensemble Methods
1. Hard Voting (TF-IDF)
2. Soft Voting (TF-IDF)
3. Hard Voting (BoW)
4. Soft Voting (BoW)

## Project Structure

```
Cross-Lingual_Sentiment_Robustness/
├── data/
│   ├── raw/
│   ├── processed/
│   └── folds/
├── models/
│   ├── Individual models (.pkl)
│   └── Ensemble models (.pkl)
├── results/
│   ├── metrics/
│   ├── plots/
│   └── analysis/
├── logs/
└── README.md
```

## Key Findings

1. Ensemble models performed best overall
2. TF-IDF outperformed BoW by 2.68%
3. Ensembles improved performance by 3.84%
4. Average cross-lingual drop: 28.76%
5. Most robust: Multinomial Naive Bayes with 22.12% gap

