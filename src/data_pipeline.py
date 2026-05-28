"""
Module: src/data_pipeline.py
Description: Audits high-frequency BBO vs Tardis datasets and builds structural master matrices.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import itertools
import warnings

warnings.filterwarnings('ignore')

# Detect project root directory dynamically (2 levels up from src/data_pipeline.py)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FUNDING_HISTORY_FILE = DATA_DIR / "funding_rate_history_202604231452.csv"
GLOBAL_START = pd.Timestamp('2025-03-01 00:00:00')
GLOBAL_END = pd.Timestamp('2025-12-31 23:59:00')
STALENESS_THRESHOLD_MINS = 3

EXCH_MAP = {
    'binance-futures': 'binf', 'bybit': 'byde', 'okex-swap': 'okex', 
    'hyperliquid': 'hype', 'gate-io-futures': 'gate', 'kucoin-futures': 'kusw'
}

def get_token_config(token: str):
    """Dynamically generates file paths and symbols for a given token."""
    # Handle slight naming variations across exchanges
    if token == 'KAITO':
        okex_sym = 'KAITO-USDT-SWAP'
    else:
        okex_sym = f'{token}-USDT-SWAP'

    return {
        'output_parquet': DATA_DIR / f"{token}_TAGGED_MASTER.parquet",
        'tardis_symbols': {
            'binance-futures': f'{token}USDT', 'bybit': f'{token}USDT', 'gate-io-futures': f'{token}_USDT',
            'hyperliquid': f'{token}', 'kucoin-futures': f'{token}USDTM', 'okex-swap': okex_sym
        },
        'bbo_paths': {
            'binf': DATA_DIR / f"binf_{token}_bbo.csv",
            'byde': DATA_DIR / f"byde_{token}_bbo.csv",
            'okex': DATA_DIR / f"okex_{token}_bbo.csv",
            'gate': DATA_DIR / f"gate_{token}_bbo.csv",
            'hype': DATA_DIR / f"hype_{token}_bbo.csv",
            'kusw': DATA_DIR / f"kusw_{token}_bbo.csv"
        }
    }

def robust_to_datetime(series):
    s_num = pd.to_numeric(series, errors='coerce')
    med_val = s_num.median()
    if pd.isna(med_val): 
        dt_series = pd.to_datetime(series, errors='coerce', utc=True)
    else:
        if med_val > 1e16:    unit = 'ns'
        elif med_val > 1e13:  unit = 'us'
        elif med_val > 1e10:  unit = 'ms'
        else:                 unit = 's'
        dt_series = pd.to_datetime(s_num, unit=unit, errors='coerce', utc=True)
    return dt_series.dt.tz_localize(None)

def compile_tardis(exch, symbol):
    tardis_base_dir = DATA_DIR / exch / symbol / "derivative_ticker"
    if not tardis_base_dir.exists(): return None
    dfs = []
    
    for file in tardis_base_dir.rglob("*.parquet"):
        try:
            df = pd.read_parquet(file)
            df['timestamp'] = robust_to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            cols = ['funding_rate', 'index_price', 'mark_price']
            dfs.append(df[[c for c in cols if c in df.columns]].resample('1min', label='right').last())
        except: continue
        
    if not dfs: return None
    annual_df = pd.concat(dfs)
    annual_df = annual_df[~annual_df.index.duplicated(keep='last')].sort_index()
    if 'index_price' not in annual_df.columns or annual_df['index_price'].isna().all():
        annual_df['index_price'] = annual_df.get('mark_price', np.nan)
    else: 
        annual_df['index_price'] = annual_df['index_price'].fillna(annual_df.get('mark_price', np.nan))
    return annual_df

def compile_bbo(path):
    if not path.exists(): return None
    df = pd.read_csv(path)
    df['timestamp'] = robust_to_datetime(df['minute'])
    df.set_index('timestamp', inplace=True)
    df = df.resample('1min', label='right').last()
    df.loc[df['bid'] >= df['ask'], ['bid', 'ask']] = np.nan
    df['mid'] = (df['bid'] + df['ask']) / 2
    return df[['bid', 'ask', 'mid']]

def compile_funding(path, exch_code, token):
    df = pd.read_csv(path)
    mask = (df['exchange'] == exch_code) & (df['symbol'].str.contains(token, na=False))
    f_df = df[mask].copy()
    if f_df.empty: return pd.DataFrame()
    
    time_col = next((c for c in f_df.columns if 'time' in c.lower() or c.lower() == 'ts'), None)
    f_df['datetime'] = robust_to_datetime(f_df[time_col])
    f_df = f_df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')
    f_df = f_df[~f_df.index.duplicated(keep='last')]
    
    rate_col = next((c for c in f_df.columns if 'rate' in c.lower() and c.lower() != 'timestamp_ns'), None)
    
    # Structural Interval Inference
    has_changed = f_df[rate_col] != f_df[rate_col].shift(1)
    has_changed.iloc[0] = True 
    change_events = f_df[has_changed].copy()
    
    change_intervals_mins = (change_events.index.to_series().diff().dt.total_seconds() / 60.0).shift(-1)
    change_events['inferred_interval_mins'] = (change_intervals_mins / 60.0).round() * 60.0
    
    fallback_mode = change_events['inferred_interval_mins'].mode()[0] if not change_events.empty else 480.0
    change_events['inferred_interval_mins'] = change_events['inferred_interval_mins'].fillna(fallback_mode)
    
    f_df['interval_mins'] = change_events['inferred_interval_mins']
    f_df['interval_mins'] = f_df['interval_mins'].ffill().bfill().fillna(fallback_mode)
    
    if exch_code in ['byde', 'binf', 'gate', 'okex', 'kusw']:
        f_df['interval_mins'] = np.clip(f_df['interval_mins'], 60.0, 480.0)
    elif exch_code == 'hype':
        f_df['interval_mins'] = 60.0
        
    f_df['epoch_id'] = range(1, len(f_df) + 1)
    return f_df[[rate_col, 'interval_mins', 'epoch_id']].rename(columns={rate_col: 'hist_rate'})

def build_master_matrix(token: str):
    """Orchestrates the data pipeline for a specific token."""
    config = get_token_config(token)
    print(f"\n🚀 Compiling Tagged Master Matrix for: {token}...")
    processed_exchanges = []
    master_idx = pd.date_range(start=GLOBAL_START, end=GLOBAL_END, freq='1min')
    
    for exch, tardis_sym in config['tardis_symbols'].items():
        s_code = EXCH_MAP[exch]
        
        t_df = compile_tardis(exch, tardis_sym)
        b_df = compile_bbo(config['bbo_paths'].get(s_code)) 
        f_df = compile_funding(FUNDING_HISTORY_FILE, s_code, token)
        
        if t_df is None or b_df is None: 
            print(f"  [⚠️ WARNING] Skipping {exch} due to missing raw files.")
            continue
            
        exch_df = t_df.join(b_df, how='outer').reindex(master_idx)
        
        is_flatlining = (exch_df['mid'].diff() == 0)
        is_missing = exch_df['mid'].isna()
        consecutive_stale_mins = is_flatlining.groupby((~is_flatlining).cumsum()).cumsum()
        exch_df[f'stale_{s_code}'] = is_missing | (consecutive_stale_mins > STALENESS_THRESHOLD_MINS)
        
        fill_cols = ['bid', 'ask', 'mid', 'index_price', 'mark_price']
        exch_df[fill_cols] = exch_df[fill_cols].ffill()
        
        exch_df = exch_df.join(f_df, how='left')
        exch_df['epoch_id'] = exch_df['epoch_id'].ffill()
        exch_df['interval_mins'] = exch_df['interval_mins'].ffill()
        
        settlement_times = pd.Series(exch_df.index, index=exch_df.index).where(~exch_df['hist_rate'].isna()).ffill()
        exch_df['fr_v1_raw'] = exch_df['funding_rate'].fillna(exch_df['hist_rate'] if 'hist_rate' in exch_df.columns else np.nan).ffill()
        exch_df['fr_v1_hourly'] = exch_df['fr_v1_raw'] * (60.0 / exch_df['interval_mins'])
        
        if 'mark_price' in exch_df.columns and 'index_price' in exch_df.columns and 'mid' in exch_df.columns:
            exch_df['inst_mark_prem'] = (exch_df['mark_price'] - exch_df['index_price']) / exch_df['index_price']
            exch_df['inst_mid_prem'] = (exch_df['mid'] - exch_df['index_price']) / exch_df['index_price']
            raw_v2 = exch_df.groupby('epoch_id')['inst_mark_prem'].expanding().mean().reset_index(level=0, drop=True)
            raw_v3 = exch_df.groupby('epoch_id')['inst_mid_prem'].expanding().mean().reset_index(level=0, drop=True)
            exch_df['fr_v2_mark_avg'] = raw_v2 * (60.0 / exch_df['interval_mins'])
            exch_df['fr_v3_mid_avg'] = raw_v3 * (60.0 / exch_df['interval_mins'])
        else:
            exch_df['fr_v2_mark_avg'], exch_df['fr_v3_mid_avg'] = exch_df['fr_v1_hourly'], exch_df['fr_v1_hourly']

        delta_t_hr = (exch_df.index - settlement_times).dt.total_seconds() / 3600.0
        delta_t_hr = np.clip(delta_t_hr, 0.0, exch_df['interval_mins'] / 60.0)

        bases = {'idx': 'index_price', 'mark': 'mark_price'}
        rates = {'hourly': 'fr_v1_hourly', 'mark_avg': 'fr_v2_mark_avg', 'mid_avg': 'fr_v3_mid_avg'}
        
        for b_n, b_c in bases.items():
            for r_n, r_c in rates.items():
                if b_c not in exch_df.columns: continue
                accrual = exch_df[b_c] * exch_df[r_c] * delta_t_hr
                exch_df[f'padj_mid_{b_n}_{r_n}_{s_code}'] = exch_df['mid'] + accrual

        exch_df.rename(columns={'epoch_id': f'epoch_id_{s_code}', 'fr_v1_hourly': f'fr_v1_{s_code}'}, inplace=True)
        cols_to_keep = [c for c in exch_df.columns if c.startswith('padj_')] + ['mid', f'stale_{s_code}', f'epoch_id_{s_code}', f'fr_v1_{s_code}']
        processed_exchanges.append(exch_df[cols_to_keep].rename(columns={'mid': f'mid_{s_code}'}))
    
    final_df = pd.concat(processed_exchanges, axis=1)
    
    exchs = [c.split('_')[-1] for c in final_df.columns if c.startswith('mid_')]
    for exch_A, exch_B in itertools.combinations(exchs, 2):
        pair = f"{exch_A}_vs_{exch_B}"
        if f'mid_{exch_A}' in final_df.columns and f'mid_{exch_B}' in final_df.columns:
            final_df[f'spread_raw_mid_{pair}'] = (np.log(final_df[f'mid_{exch_A}']) - np.log(final_df[f'mid_{exch_B}'])) * 10000
            
        for b, r in itertools.product(bases.keys(), rates.keys()):
            cA, cB = f'padj_mid_{b}_{r}_{exch_A}', f'padj_mid_{b}_{r}_{exch_B}'
            if cA in final_df.columns and cB in final_df.columns:
                final_df[f'spread_padj_{b}_{r}_{pair}'] = (np.log(final_df[cA]) - np.log(final_df[cB])) * 10000
                    
    final_df.to_parquet(config['output_parquet'])
    print(f"💾 Saved: {config['output_parquet'].name} | Shape: {final_df.shape}")