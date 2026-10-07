# Machine Learning and Deep Learning in Stock Market Prediction: Integrating Technical Baselines with Lexicon-Based Sentiment

This repository contains the code, data preprocessing pipelines, and documentation for the MSc FinTech research project. The project investigates the predictive performance of multi-architecture models for forecasting short-term stock price movement direction and realized volatility. It provides a rigorous multi-architecture benchmark that contrasts a strictly technical baseline with a hybrid model enriched by qualitative news sentiment.

## Project Overview
*   **Objective**: To formally investigate whether incorporating general domain news sentiment features (extracted via VADER and TextBlob) improves the prediction of short-term directional movements and realized volatility in S&P 500 stocks.
*   **Methodology**: Applies a mathematically rigorous, look-ahead-bias-free pipeline with strict date-based splitting (70/30 train/test split) and per-ticker grouping to eliminate cross-ticker data leakage. Deep learning models process a 10-day sliding lookback sequence.
*   **Key Findings**:
    *   Technical-only baselines achieve highly predictive, state-of-the-art directional accuracy (up to 70.76% out-of-sample for CNN-LSTM).
    *   Adding general news sentiment acts as destabilizing noise for directional prediction, dropping accuracy across non-linear and deep learning models to approximately 55.6% due to multi-ticker collinearity and overfitting.
    *   Conversely, sentiment features significantly enhance volatility forecasting, boosting Random Forest classification accuracy from 60.4% to 64.8%, proving that public panic and qualitative narrative shocks are strong predictors of market risk.

## Data Acquisition
*   **Market Data**: Daily OHLCV data for all S&P 500 constituent companies from August 2008 to July 2016, retrieved programmatically via `yfinance`.
*   **Sentiment Data**: The Top 25 daily world news headlines sourced from the Kaggle "Daily News for Stock Market Prediction" dataset (originating from the Reddit WorldNews channel).
*   **Sample Size**: A structurally homogeneous panel of 912,424 observations across 486 stocks with complete trading histories.

## Feature Engineering
The raw data is converted into a 16-dimensional scale-invariant, stationary feature space:
*   **Technical Baseline (10 Features)**: Daily Return, Open Return, High Return, Volume Change, Simple Moving Average (5 & 20) Relative Distances, 14-day RSI, Bollinger Band (Upper & Lower) Distances, and MACD Relative Distance.
*   **Sentiment Enrichment (6 Features)**: VADER scores (Positive, Negative, Neutral, Compound) and TextBlob scores (Polarity, Subjectivity).

## Model Architectures
The project benchmarks five traditional machine learning classifiers and three deep learning sequence models:
1.  **Traditional ML**: Logistic Regression, Decision Tree, Random Forest, XGBoost, and Support Vector Machines (SVM) with RBF kernels.
2.  **Deep Learning**:
    *   **LSTM**: Stacked Long Short-Term Memory network (128 and 64 units) with 30% dropout to capture long-range temporal dependencies.
    *   **BiLSTM**: Bidirectional LSTM processing sequences in both forward and backward directions to learn context-dependent pattern structures.
    *   **CNN-LSTM**: Hybrid architecture utilizing two 1D Convolutional layers with batch normalization to extract local spatial patterns (like candlestick structures), followed by an LSTM layer for temporal evolution.


## Dependencies
The deep learning pipeline is built to run on GPUs and requires the following primary libraries:
*   `tensorflow` (v2.20.0)
*   `scikit-learn`
*   `pandas`
*   `numpy`
*   `seaborn` & `matplotlib`
*   `joblib`

## Author
**Ainain Zehra Rizvi**
Master of Science in FinTech, National College of Ireland (2025-2026)
