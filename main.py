"""
Global Execution Script: Capstone Arbitrage Backtester
"""
import pandas as pd
from pathlib import Path
import warnings

# Suppress harmless warnings for clean presentation output
warnings.filterwarnings('ignore')

# Import our customized modules
from src.data_pipeline import build_master_matrix, get_token_config
from src.statistical_eng import StatModelEngine
from src.backtest_eng import PerpetualBacktestEngine, PerformanceReporter

# --- GLOBAL VARIABLES ---
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
EMP_START = pd.Timestamp('2025-03-01 00:00:00') 
EMP_END = pd.Timestamp('2025-12-31 23:59:00')

TOKENS_TO_EVALUATE = ['BTC', 'AVAX', 'BERA', 'KAITO']

TOKEN_CONFIGS = {
    'BTC':   {'entry_threshold': 4.0, 'max_entry_spread_bps': 10.0},
    'AVAX':  {'entry_threshold': 4.0, 'max_entry_spread_bps': 20.0},
    'BERA':  {'entry_threshold': 4.0, 'max_entry_spread_bps': 10.0},
    'KAITO': {'entry_threshold': 4.0, 'max_entry_spread_bps': 5.0}
}

FEES = {
    'binf': 1.70, 'okex': 1.75, 'byde': 3.20, 
    'kusw': 2.50, 'gate': 2.00, 'hype': 1.44
}

NAME_MAP = {
    'binance-futures': 'binf', 'bybit': 'byde', 'okex-swap': 'okex', 
    'gate-io-futures': 'gate', 'hyperliquid': 'hype', 'kucoin-futures': 'kusw'
}

def run_pipeline():
    print("======================================================================================================================")
    print(f" 📘 CAPSTONE BASELINE 0.2.1: ASSET-CUSTOMIZABLE PAIR EXECUTION (PER-TOKEN ROUTING)")
    print("======================================================================================================================\n")

    for token in TOKENS_TO_EVALUATE:
        # Step 1: Data Auditing and Parquet Construction
        # Note: If you have already built the parquets, you can comment this line out to save time.
        # build_master_matrix(token)
        
        # Check if the generated file exists before proceeding
        master_parquet = DATA_DIR / f"{token}_TAGGED_MASTER.parquet"
        if not master_parquet.exists():
            print(f"[!] Warning: Master Parquet for {token} not found. Skipping.")
            continue
            
        df_master = pd.read_parquet(master_parquet).loc[EMP_START:EMP_END]
        config_paths = get_token_config(token)
        
        # Step 2: Statistical Engineering
        # stat_eng = StatModelEngine(df_master)
        # stat_eng.execute_all_diagnostics()
        
        # Step 3: Backtest Engine
        config_params = TOKEN_CONFIGS[token]
        engine = PerpetualBacktestEngine(
            df_master=df_master, 
            token=token, 
            config=config_params, 
            bbo_paths=config_paths['bbo_paths'], 
            name_map=NAME_MAP, 
            fees=FEES
        )
        
        trade_log_df = engine.run_simulation()
        
        # Step 4: Reporting
        PerformanceReporter.generate_tear_sheet(
            tdf=trade_log_df, 
            token=token, 
            config=config_params, 
            start_date=EMP_START, 
            end_date=EMP_END
        )

if __name__ == "__main__":
    run_pipeline()
