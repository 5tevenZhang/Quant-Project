# News-Driven Stock Return Prediction

This project studies whether overnight financial news can help predict next-day stock returns. It builds a text-based prediction pipeline using **TF-IDF** and **Bag-of-Words (BoW)** features, then compares several machine learning models on a time-ordered train/test split.

## Project Overview

The workflow:

1. Load and preprocess stock price data and news data from zipped TSV files
2. Filter **overnight news**
3. Align news with the next trading day's return target
4. Convert text into features using:
   - TF-IDF
   - Bag-of-Words
5. Train and tune regression models
6. Evaluate out-of-sample performance
7. Save feature-importance and parameter-impact plots

## Models Used

- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

## Main Features

- End-to-end Colab-friendly notebook
- Text preprocessing with:
  - stopword removal
  - stemming
- Time-series-aware train/test split
- Hyperparameter tuning with cross-validation
- Comparison across multiple models
- Visualization of feature importance and tuning impact

## Example Output

From one TF-IDF run shown in the notebook:

- Sample size: 1032
- Training set: 774
- Test set: 258
- Feature dimension: 100

### Test Performance
- **Ridge Regression**: MAE = 0.1091, MSE = 0.0328
- **Random Forest**: MAE = 0.1107, MSE = 0.0347
- **XGBoost**: MAE = 0.1083, MSE = 0.0333

In this run, Ridge delivered the best MSE, while XGBoost achieved the lowest MAE.

## File Structure

```bash
.
├── News_Driven_Stock_Return_Prediction.ipynb
├── README.md
└── data/                 
