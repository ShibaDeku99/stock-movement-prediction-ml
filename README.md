# 📈 Quant ML: Stock Movement Prediction & Trading Strategy

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost%20%7C%20LightGBM-orange)
![Finance](https://img.shields.io/badge/Finance-yfinance%20%7C%20ta-green)

## 📌 Overview
A production-grade Machine Learning pipeline designed to predict stock price movements and backtest quantitative trading strategies. 

The system moves beyond simple "up/down tomorrow" predictions by targeting high-confidence multi-day trends (T+5). It encompasses an automated data fetching pipeline, robust feature engineering (130+ features including market sentiment and events), rigorous Time-Series Cross-Validation, and realistic backtesting that accounts for transaction costs and probability-based position sizing.

## 🚀 Key Results (Out-of-Sample Backtest)

The model was backtested on **AAPL** (Out-of-sample: Mar 2024 - Jun 2026) factoring in **0.1% transaction costs per trade**. 

| Strategy | Total Return | Max Drawdown | Sharpe Ratio | Calmar Ratio | Trades | Exposure Time |
|---|---|---|---|---|---|---|
| **Binary Signal** (All-in on signal) | **71.90%** | **-11.40%** | 1.45 | 2.41 | 36 | 30.8% |
| **Proba Sizing** (Capped at 70% capital) | 49.23% | **-7.64%** | **1.51** | **2.58** | 36 | 30.8% |
| **Buy & Hold** (Baseline) | 73.98% | *-33.36%* | 1.03 | 0.84 | 1 | 100.0% |

### 💡 Why this beats the market:
1. **Exceptional Risk Control:** Max Drawdown was slashed by nearly 3x compared to Buy & Hold (from -33% down to -11%).
2. **Capital Efficiency:** The model only exposes capital to the market **30.8% of the time**. For the other ~70% of the time, capital is preserved in cash, avoiding major market crashes.
3. **High Profit Factor (3.37):** Win rate sits at 58.3%, but the average winning trade (+3.97%) is more than double the average losing trade (-1.65%).

## ⚙️ Architecture & Pipeline

### 1. Data Loader (`src/data_loader.py`)
- Automatically fetches OHLCV data using `yfinance`.
- Contextualizes individual stocks by simultaneously fetching **Market Data** (SPY, QQQ, VIX).
- Extracts and aligns **Corporate Events** (Earnings Calendar dates).

### 2. Feature Engineering (`src/features.py`)
Generates 131 highly curated features to capture market inefficiencies:
- **Momentum & Volatility:** 5D/10D/20D Returns, ATR (Average True Range), Intraday Gaps.
- **Money Flow:** Volume Spike detectors, Volume/SMA Ratios.
- **Market Context:** Relative Strength (RS) against SPY/QQQ, VIX Levels & Changes.
- **Market Sentiment Proxy:** A custom composite index combining VIX normalization, SPY momentum, and Volume/Price trends to detect Fear/Greed.
- **Event Proximity:** Distance to upcoming Earnings Reports to capture pre/post-earnings volatility.

### 3. Model Training & Tuning (`src/train.py`)
- Compares Logistic Regression, Random Forest, XGBoost, and LightGBM.
- **Time-Series Cross-Validation:** Strict `TimeSeriesSplit` (5-folds) to completely eliminate look-ahead bias.
- **Threshold Tuning:** The probability threshold is dynamically optimized for F1-Score to tackle class imbalance, rather than relying on a static 0.5 threshold.

### 4. Backtesting & Reporting (`src/backtest.py`)
- Simulates realistic trading with configurable transaction costs.
- **Two Custom Strategies:**
  - `Binary Signal`: Full allocation when probability > threshold.
  - `Aggressive Tiered Proba Sizing`: Dynamically scales position size (50% to 70% max) based on the model's confidence probability, enforcing strict risk limits.
- Generates exhaustive trade logs, monthly return heatmaps, and equity/drawdown curves.

## 📂 Project Structure

```text
├── data/
│   ├── raw/             # Raw CSV data (AAPL, SPY, VIX, Earnings)
│   └── processed/       # Engineered features ready for training
├── models/              # Serialized .pkl models, thresholds, and feature lists
├── reports/             # CSV metrics, Trade Logs, and generated Charts
├── src/
│   ├── data_loader.py   # Data extraction pipeline
│   ├── features.py      # Feature engineering logic
│   ├── train.py         # Model training & CV orchestration
│   └── backtest.py      # Strategy simulation & visualization
└── requirements.txt     # Python dependencies
```

## 🛠️ How to Run

1. **Setup Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Execute the Pipeline:**
   ```bash
   # Step 1: Download raw data
   python src/data_loader.py
   
   # Step 2: Generate features
   python src/features.py
   
   # Step 3: Train model and tune threshold
   python src/train.py
   
   # Step 4: Run backtest and generate charts
   python src/backtest.py
   ```

3. **View Results:**
   Check the `reports/` folder for `backtest_summary.csv`, `trade_log.csv`, and beautifully rendered Matplotlib visualizations of the Equity Curve and Monthly Returns.
