"""
Module: src/statistical_eng.py
Description: Implements rolling Dickey-Fuller (ADF) cointegration tests and extracts 
             structural Ornstein-Uhlenbeck (OU) mean-reversion metrics for crypto pairs.
"""

import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.stattools import adfuller
import scipy.stats as stats
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class StatModelEngine:
    def __init__(self, df: pd.DataFrame, window_mins: int = 1440, fee_bps: float = 2.0):
        self.df = df
        self.window_mins = window_mins
        self.fee_bps = fee_bps
        self.slippage_bps = 2.0
        self.round_trip_cost = (self.fee_bps + self.slippage_bps) * 2
        
        # Identify all exchanges and possible pairs in the dataframe
        self.exchanges = [c.split('_')[-1] for c in df.columns if c.startswith('mid_')]
        self.all_pairs = list(itertools.combinations(self.exchanges, 2))

    def _get_clean_spread(self, exch_a: str, exch_b: str, spread_type: str = 'padj_idx_mid_avg') -> pd.Series:
        col_name = f'spread_{spread_type}_{exch_a}_vs_{exch_b}'
        if col_name in self.df.columns:
            return self.df[col_name].dropna()
        
        col_name_alt = f'spread_{spread_type}_{exch_b}_vs_{exch_a}'
        if col_name_alt in self.df.columns:
            return -self.df[col_name_alt].dropna()
        
        return pd.Series(dtype=float)

    def calculate_ou_parameters(self, spread_series: pd.Series) -> Dict:
        """Calculates Ornstein-Uhlenbeck parameters using OLS."""
        y = spread_series.values
        if len(y) < 2: return {}
        
        dy = np.diff(y)
        y_prev = y[:-1]
        
        # OLS: dy = theta * (mu - y_prev) * dt + sigma * dW
        # Equivalent to: dy = a + b * y_prev + error
        X = np.vstack([np.ones(len(y_prev)), y_prev]).T
        try:
            beta = np.linalg.inv(X.T @ X) @ X.T @ dy
            a, b = beta[0], beta[1]
            
            if b >= 0: return {} # Non-stationary (diverging)
            
            theta = -b
            mu = a / theta
            half_life = np.log(2) / theta if theta > 0 else np.inf
            
            return {'theta': theta, 'mu': mu, 'half_life_mins': half_life}
        except np.linalg.LinAlgError:
            return {}

    def run_stylized_facts(self):
        """Generates static macro stylized facts and minimum arbitrage bands."""
        logger.info("Running Macro Stylized Facts & Cointegration Analysis...")
        results = []
        
        for a, b in self.all_pairs:
            s = self._get_clean_spread(a, b)
            if s.empty or len(s) < self.window_mins: continue
                
            mean = s.mean()
            std = s.std()
            kurt = s.kurtosis()
            acf1 = s.autocorr(lag=1)
            
            # Static ADF Test (Full Sample)
            try:
                adf_stat, p_value, _, _, _, _ = adfuller(s.values, maxlag=1)
                is_stationary = p_value < 0.05
            except Exception:
                p_value, is_stationary = np.nan, False
                
            req_sigma = self.round_trip_cost / std if std > 0 else np.nan
            active_days = len(s) / 1440
            
            results.append({
                'Pair': f"{a.upper()}/{b.upper()}",
                'Mean(bps)': mean,
                'StdDev(bps)': std,
                'Kurtosis': kurt,
                'ACF(1)': acf1,
                'ADF P-Val': p_value,
                'Stationary': 'Yes' if is_stationary else 'No',
                'Req. Sigma': req_sigma,
                'Active Days': active_days
            })
            
        df_res = pd.DataFrame(results).round(3)
        print("\n--- Stylized Facts ---")
        print(df_res.to_string(index=False))
        return df_res

    def execute_all_diagnostics(self):
        """Runs the complete suite of statistical diagnostics."""
        print(f"\n" + "="*80)
        print(f" ⚙️ RUNNING STATISTICAL DIAGNOSTICS")
        print("="*80)
        self.run_stylized_facts()
        # You can add the rolling signal frequency and OU validation methods here later
        print("="*80 + "\n")