# Arbitraging the Price Space: Funding-Adjusted Cross-Exchange Perpetual Arbitrage
---
This repository contains the quantitative backtesting framework and statistical pipeline for exploiting transient, cross-exchange pricing dislocations in cryptocurrency perpetual futures. The architecture processes high-frequency Best Bid and Offer (BBO) data to validate cointegration, construct ex-ante mean-reverting signals, and evaluate strategy decay under strict execution constraints and toxic liquidity regimes.


## Framework Overview & Core Logic

The architecture simulates a market-neutral statistical arbitrage strategy, utilizing an Ornstein-Uhlenbeck (OU) process to model rolling spreads while actively neutralizing phantom liquidity and asynchronous funding mechanics.


1. **Price Normalization (Funding-Adjusted Space)**
   * Strips funding rate accrual from raw contract prices to isolate pure price-space dislocations and ensure parity across asynchronous exchange mechanisms.
2. **Rolling Ex-Ante $s$-score (Signal Engine)**
   * Models the cross-exchange spread using an Ornstein-Uhlenbeck (OU) process, discretized via a 1440-minute rolling AR(1) regression. This generates dynamic, standardized $s$-scores completely free of look-ahead bias.
3. **The Executable EV Gate (Entry Logic)**
   * Restricts market entries to structural extremes (e.g., $|s| \ge 4.0$). Entry is strictly bound by an Expected Value (EV) threshold to ensure theoretical profitability survives transaction friction:
     $$\text{Net EV} = \text{Gross EV} - \text{Round-Trip Fees} - \text{Expected Spread}$$
4. **Execution Simulation & Illiquidity Guards**
   * Implements Bid-Ask Illiquidity Guards (e.g., 5.0, 10.0, 20.0 bps thresholds) to explicitly block trade execution during toxic, low-depth order book regimes (phantom liquidity).

---

## Repository Directory Structure

* `src/data_pipeline.py`: Ingests Tardis.dev BBO and funding data, handles asynchronous timestamp alignment, and structures the master evaluation matrices.
* `src/statistical_eng.py`: Executes rolling Augmented Dickey-Fuller (ADF) tests for stationarity and fits the OU mean-reversion parameters.
* `src/backtest_eng.py`: Event-driven state machine simulating taker-taker execution, transaction slippage, EV gating, and exit routing.
* `src/evaluation.py`: Generates institutional performance arrays including Active Sharpe, Maximum Drawdown, Gain-to-Pain (G2P) ratio, and empirical win rates.

---

## Reproduction & Execution Guide

To replicate the statistical profiles and sensitivity matrices across thresholds and illiquidity guards exactly as presented in the report, follow these steps:



### 1. Prerequisites & Environment Setup
Ensure you have Python 3.10+ installed. Clone the repository and initialize the virtual environment:

```bash
git clone [https://github.com/yoho369/crypto-perpetual-arbitrage.git](https://github.com/yourusername/crypto-perpetual-arbitrage.git)
cd crypto-perpetual-arbitrage
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Preparation
This project uses large Parquet master matrices which are excluded from this repository for performance. Note that the raw Tardis derivatives ticker data is not available here, but the code can still be run with the the master files.

To run the backtester:
1. **Download the data package:** [Click here to download the required Parquet master files from Google Drive](https://drive.google.com/drive/folders/1QhelaaFTQ-AaweRa47B08oujwlpSotBw?usp=sharing)
2. **Setup:** Place the downloaded files into the `/data` folder in this repository.
3. **Execution:** Ensure the files are named `[TOKEN]_TAGGED_MASTER.parquet`.


---


