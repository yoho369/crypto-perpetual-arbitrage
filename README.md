# Arbitraging the Price Space: Funding-Adjusted Cross-Exchange Perpetual Arbitrage
---
This repository contains the quantitative backtesting framework and statistical pipeline for exploiting transient, cross-exchange pricing dislocations in cryptocurrency perpetual futures. The architecture processes one-minute-level Best Bid and Offer (BBO) data to validate cointegration, construct ex-ante mean-reverting signals, and evaluate strategy decay under strict execution constraints and toxic liquidity regimes.


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
```

For Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For Windows (Command Prompt / PowerShell):
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Preparation
This project uses large Parquet master matrices which are excluded from this repository for performance. Note that the raw Tardis derivatives ticker data is not available here, but the code can still be run with the the master files.

To run the backtester:
1. **Download the data package:** [Click here to download the required Parquet master files from Google Drive](https://drive.google.com/drive/folders/1QhelaaFTQ-AaweRa47B08oujwlpSotBw?usp=sharing)
2. **Setup:** Place the downloaded files into the `/data` folder in this repository.
3. **Execution:** Ensure the files are named `[TOKEN]_TAGGED_MASTER.parquet`.

### 3. Execution & Strategy Evaluation
Once the environment is active and the data is loaded, you can trigger the full baseline execution engine across the entire asset universe.

Run from your terminal:
```bash
python main.py
```

Running Statistical Diagnostics:
If you wish to generate the full suite of statistical diagnostics (e.g., ADF stationarity tests, OU half-life calculations, and variance profiling) during the main run, please uncomment the following lines in the main script:

```python
# Step 2: Statistical Engineering
stat_eng = StatModelEngine(df_master)
stat_eng.execute_all_diagnostics()
```
### 4. Interactive Exploration (Standalone Notebooks)

For modular execution, step-by-step code review, and deeper data exploration, we have provided standalone Jupyter Notebooks in the notebooks/ directory. T
hese are ideal for running individual components of the research pipeline:
* **Data Engineering:** `[TOKEN] data cleaning and master file construction.ipynb`
* **Econometric Modeling:** `[TOKEN] statistical studies.ipynb`
* **Baseline Engine Validation:** `full backtest.ipynb`
* **Hyperparameter & Sensitivity Testing:** `sensitivity test.ipynb`
  
Note: The comprehensive hyperparameter sensitivity sweep (evaluating the strategy across varying $\sigma$ thresholds, $C_{max}$ illiquidity guards, and fee tiers) is computationally intensive and is available exclusively as a standalone research notebook.

---

## Repository Architecture

```text
crypto-perpetual-arbitrage/
├── README.md                           # Project documentation and execution guide
├── arbitrage report.pdf                # Research report
├── main.py                             # Execution entry point
├── requirements.txt                    # Python environment dependencies
├── .gitignore.txt                      # Version control exclusion rules
│
├── src/                                # Core Quantitative Engine (Modular Python)
│   ├── data_pipeline.py                # Data ingestion, normalization, and spread construction
│   ├── statistical_eng.py              # Ornstein-Uhlenbeck (OU) model validation and statistical diagnostics
│   └── backtest_eng.py                 # Chronological execution and microstructure risk routing
│
├── notebooks/                          # Interactive Research & Exploratory Environment
│   ├── full backtest.ipynb             # Baseline engine validation and tear sheets
│   ├── sensitivity test.ipynb          # Multi-dimensional hyperparameter optimization
│   ├── [TOKEN] data cleaning...ipynb   # Asset-specific data parsing workflows
│   └── [TOKEN] statistical...ipynb     # Asset-specific statistical diagnostics
│
└── data/                               # High-Frequency Data & Execution Matrices
    ├── funding_rate_history_*.csv      # Historical funding rate epochs (Global)
    ├── instrument_spec_history.csv     # Exchange-specific derivative parameters
    ├── *_TAGGED_MASTER.parquet         # Merged and normalized execution matrices (Git-ignored)
    └── *_bbo.csv                       # Level-1 Best Bid/Offer snapshots


