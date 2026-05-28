"""
Module: src/backtest_eng.py
Description: Core execution engine simulating statistical arbitrage entry gates, 
             illiquidity filters, and hybrid crypto-funding blackout protocols.
"""

import pandas as pd
import numpy as np
import itertools
from tqdm.auto import tqdm
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class PerpetualBacktestEngine:
    def __init__(self, df_master: pd.DataFrame, token: str, config: Dict, bbo_paths: Dict, name_map: Dict, fees: Dict):
        self.df_master = df_master
        self.token = token
        self.config = config
        self.bbo_paths = bbo_paths
        self.name_map = name_map
        self.fees = fees
        
        self.rolling_window_mins = 1440
        self.trade_timeout_mins = 300
        self.exchanges = list(self.bbo_paths.keys())
        self.pairs = list(itertools.combinations(self.exchanges, 2))
        
        self.timestamps = self.df_master.index
        self.n_steps = len(self.df_master)

    def _robust_to_datetime(self, series: pd.Series) -> pd.Series:
        s_num = pd.to_numeric(series, errors='coerce')
        med_val = s_num.median()
        if pd.isna(med_val): return pd.to_datetime(series, errors='coerce', utc=True).dt.tz_localize(None)
        unit = 'ns' if med_val > 1e16 else ('us' if med_val > 1e13 else ('ms' if med_val > 1e10 else 's'))
        return pd.to_datetime(s_num, unit=unit, errors='coerce', utc=True).dt.floor('1min').dt.tz_localize(None)

    def _load_aligned_bbo(self, exch_a: str, exch_b: str) -> pd.DataFrame:
        df_pair = pd.DataFrame(index=self.df_master.index)
        for exch in [exch_a, exch_b]:
            bbo = pd.read_csv(self.bbo_paths[exch], usecols=['minute', 'bid', 'ask'])
            bbo['timestamp'] = self._robust_to_datetime(bbo['minute'])
            bbo.set_index('timestamp', inplace=True)
            bbo = bbo[['bid', 'ask']].copy()
            
            bbo.loc[bbo['bid'] >= bbo['ask'], ['bid', 'ask']] = np.nan
            bbo = bbo.reindex(self.df_master.index).ffill(limit=3)
            df_pair[f'bid_{exch}'] = bbo['bid']
            df_pair[f'ask_{exch}'] = bbo['ask']
            df_pair[f'spread_{exch}'] = (np.log(bbo['ask']) - np.log(bbo['bid'])) * 10000

        df_pair['combined_spread'] = df_pair[f'spread_{exch_a}'] + df_pair[f'spread_{exch_b}']
        df_pair['liquidity_toll'] = df_pair['combined_spread'].rolling(1440).median().bfill()
        return df_pair

    def _calc_stochastic_ex_ante(self, adj_col: str, raw_col: str) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.DataFrame(index=self.df_master.index)
        df[adj_col] = self.df_master[adj_col]
        df['raw_ma'] = self.df_master[raw_col].shift(1).rolling(self.rolling_window_mins).mean()
        
        y_prev, x_prev = df[adj_col].shift(1), df[adj_col].shift(2)
        roll_cov = y_prev.rolling(self.rolling_window_mins).cov(x_prev)
        roll_var_x = x_prev.rolling(self.rolling_window_mins).var()
        roll_var_y = y_prev.rolling(self.rolling_window_mins).var()
        mean_x, mean_y = x_prev.rolling(self.rolling_window_mins).mean(), y_prev.rolling(self.rolling_window_mins).mean()
        
        phi = (roll_cov / roll_var_x).clip(upper=0.999, lower=-0.999) 
        a = mean_y - phi * mean_x
        m = a / (1 - phi)
        var_e = (roll_var_y - (phi**2) * roll_var_x).clip(lower=1e-8)
        sigma_eq = np.sqrt(var_e / (1 - phi**2))
        
        df['s_score'] = (df[adj_col] - m) / sigma_eq
        return df['s_score'].values, df['raw_ma'].values

    def _get_data_driven_epoch(self, exch_name: str, full_name: str) -> np.ndarray:
        epoch_col = f'epoch_id_{full_name}'
        if epoch_col not in self.df_master.columns:
            epoch_col = f'epoch_id_{exch_name}'
        if epoch_col in self.df_master.columns:
            shift_mask = self.df_master[epoch_col] != self.df_master[epoch_col].shift(-1)
            shift_mask.iloc[-1] = False
            return shift_mask.values
        return np.zeros(self.n_steps, dtype=bool)

    def run_simulation(self) -> pd.DataFrame:
        entry_threshold = self.config['entry_threshold']
        max_entry_spread_bps = self.config['max_entry_spread_bps']
        
        signals, bbo_storage, last_entry_t = {}, {}, {}
        
        # 1. Pre-compute footprints
        for exch_a, exch_b in tqdm(self.pairs, desc=f"Pre-computing {self.token} Signals", leave=False):
            pair_label = f"{exch_a.upper()}/{exch_b.upper()}"
            fname_a = [k for k, v in self.name_map.items() if v == exch_a][0]
            fname_b = [k for k, v in self.name_map.items() if v == exch_b][0]
            
            adj_col = f'spread_padj_idx_mid_avg_{fname_a}_vs_{fname_b}'
            raw_col = f'spread_raw_mid_{fname_a}_vs_{fname_b}'
            
            if adj_col not in self.df_master.columns or raw_col not in self.df_master.columns: continue
                
            bbo_df = self._load_aligned_bbo(exch_a, exch_b)
            s_score, raw_ma = self._calc_stochastic_ex_ante(adj_col, raw_col)
            z_exec_raw = self.df_master[raw_col].values
            
            mask_a = (self.timestamps.minute == 59) if exch_a == 'hype' else self._get_data_driven_epoch(exch_a, fname_a)
            mask_b = (self.timestamps.minute == 59) if exch_b == 'hype' else self._get_data_driven_epoch(exch_b, fname_b)

            signals[pair_label] = {
                's_score': s_score, 'raw_ma': raw_ma, 'raw_mid': z_exec_raw,
                'is_funding': mask_a | mask_b,
                'liquidity_toll': bbo_df['liquidity_toll'].values,
                'round_trip_fee_bps': (self.fees[exch_a] + self.fees[exch_b]) * 2
            }
            bbo_storage[pair_label] = bbo_df
            last_entry_t[pair_label] = -60

        active_positions, trade_log = {}, []

        # 2. Chronological Engine
        for t in tqdm(range(self.rolling_window_mins, self.n_steps), desc=f"Processing {self.token} Execution"):
            current_time = self.timestamps[t]
            
            # EXITS
            keys_to_remove = []
            for pair, pos in active_positions.items():
                sig_data = signals[pair]
                bbo_df = bbo_storage[pair]
                exch_a, exch_b = pair.split('/')
                
                hold_time = t - pos['entry_idx']
                timeout_hit = hold_time >= self.trade_timeout_mins
                funding_hit = sig_data['is_funding'][t] 
                ma_reverted = (sig_data['raw_mid'][t] <= pos['target_ma']) if pos['direction'] == -1 else (sig_data['raw_mid'][t] >= pos['target_ma'])
                
                bid_a_val, ask_a_val = bbo_df[f'bid_{exch_a.lower()}'].iloc[t], bbo_df[f'ask_{exch_a.lower()}'].iloc[t]
                bid_b_val, ask_b_val = bbo_df[f'bid_{exch_b.lower()}'].iloc[t], bbo_df[f'ask_{exch_b.lower()}'].iloc[t]
                is_bbo_invalid = np.isnan(bid_a_val) or np.isnan(ask_a_val) or np.isnan(bid_b_val) or np.isnan(ask_b_val)
                
                must_close = False
                exit_reason = None
                
                if funding_hit: must_close, exit_reason = True, "Funding_Epoch"
                elif timeout_hit: must_close, exit_reason = True, "Timeout"
                elif ma_reverted: must_close, exit_reason = True, "MA_Hit"
                elif pos.get('force_close_pending', False): must_close, exit_reason = True, pos.get('pending_reason')
                
                if must_close:
                    if is_bbo_invalid:
                        if hold_time >= self.trade_timeout_mins or exit_reason in ["Funding_Epoch", "Funding_Delayed"]:
                            valid_bbo = bbo_df.iloc[:t].dropna(subset=[
                                f'bid_{exch_a.lower()}', f'ask_{exch_a.lower()}', 
                                f'bid_{exch_b.lower()}', f'ask_{exch_b.lower()}'
                            ])
                            if not valid_bbo.empty:
                                last_val = valid_bbo.iloc[-1]
                                bid_a_val, ask_a_val = last_val[f'bid_{exch_a.lower()}'], last_val[f'ask_{exch_a.lower()}']
                                bid_b_val, ask_b_val = last_val[f'bid_{exch_b.lower()}'], last_val[f'ask_{exch_b.lower()}']
                                pos['pending_reason'] = "Funding_Imputed" if exit_reason in ["Funding_Epoch", "Funding_Delayed"] else "Timeout_Imputed"
                            else:
                                bid_a_val, ask_a_val, bid_b_val, ask_b_val = 1.0, 1.0, 1.0, 1.0
                        else:
                            pos['force_close_pending'] = True
                            if 'pending_reason' not in pos: pos['pending_reason'] = "MA_Hit_Delayed"
                            continue 
                    
                    if pos.get('force_close_pending', False) or (is_bbo_invalid and (hold_time >= self.trade_timeout_mins or exit_reason in ["Funding_Epoch", "Funding_Delayed"])):
                        exit_reason = pos.get('pending_reason', "Timeout_Imputed")
                    elif funding_hit: exit_reason = "Funding_Epoch"
                    elif timeout_hit: exit_reason = "Timeout"
                    else: exit_reason = "MA_Hit"
                    
                    if pos['direction'] == -1:  
                        current_z_exec = (np.log(ask_a_val) - np.log(bid_b_val)) * 10000
                        gross_pnl = pos['z_entry_exec'] - current_z_exec
                    else:                      
                        current_z_exec = (np.log(bid_a_val) - np.log(ask_b_val)) * 10000
                        gross_pnl = current_z_exec - pos['z_entry_exec']
                    
                    trade_log.append({
                        'Pair': pair, 'Entry_Time': self.timestamps[pos['entry_idx']], 'Exit_Time': current_time, 
                        'Hold_Time': hold_time, 'Gross_PnL': gross_pnl, 'Net_PnL': gross_pnl - pos['round_trip_fee_bps'], 'Reason': exit_reason
                    })
                    keys_to_remove.append(pair)
                    
            for k in keys_to_remove: del active_positions[k]

            # ENTRIES
            for pair, sig_data in signals.items():
                if pair in active_positions or t - last_entry_t[pair] < 60 or sig_data['is_funding'][t]: continue  
                if np.isnan(sig_data['s_score'][t]) or np.abs(sig_data['s_score'][t]) < entry_threshold: continue
                    
                bbo_df = bbo_storage[pair]
                exch_a, exch_b = pair.split('/')
                
                bid_a_val, ask_a_val = bbo_df[f'bid_{exch_a.lower()}'].iloc[t], bbo_df[f'ask_{exch_a.lower()}'].iloc[t]
                bid_b_val, ask_b_val = bbo_df[f'bid_{exch_b.lower()}'].iloc[t], bbo_df[f'ask_{exch_b.lower()}'].iloc[t]
                if np.isnan(bid_a_val) or np.isnan(ask_a_val) or np.isnan(bid_b_val) or np.isnan(ask_b_val): continue
                    
                curr_spread_a = (np.log(ask_a_val) - np.log(bid_a_val)) * 10000
                curr_spread_b = (np.log(ask_b_val) - np.log(bid_b_val)) * 10000
                if (curr_spread_a + curr_spread_b) > max_entry_spread_bps: continue

                direction = -1 if sig_data['s_score'][t] > 0 else 1
                target_ma = sig_data['raw_ma'][t]
                
                if direction == -1:
                    entry_phys = (np.log(bid_a_val) - np.log(ask_b_val)) * 10000
                    gross_ev = entry_phys - target_ma
                else:
                    entry_phys = (np.log(ask_a_val) - np.log(bid_b_val)) * 10000
                    gross_ev = target_ma - entry_phys
                    
                net_ev = gross_ev - sig_data['round_trip_fee_bps'] - sig_data['liquidity_toll'][t]
                if net_ev <= 0: continue  
                    
                active_positions[pair] = {
                    'entry_idx': t, 'z_entry_exec': entry_phys, 
                    'target_ma': target_ma, 'direction': direction, 
                    'round_trip_fee_bps': sig_data['round_trip_fee_bps']
                }
                last_entry_t[pair] = t  

        # TERMINAL LIQUIDATION
        terminal_idx = self.n_steps - 1
        terminal_time = self.timestamps[terminal_idx]
        for pair, pos in list(active_positions.items()):
            bbo_df = bbo_storage[pair]
            exch_a, exch_b = pair.split('/')
            
            hold_time = min(terminal_idx - pos['entry_idx'], self.trade_timeout_mins)
            valid_bbo = bbo_df.dropna(subset=[f'bid_{exch_a.lower()}', f'ask_{exch_a.lower()}', f'bid_{exch_b.lower()}', f'ask_{exch_b.lower()}'])
            if not valid_bbo.empty:
                last_val = valid_bbo.iloc[-1]
                bid_a_val, ask_a_val = last_val[f'bid_{exch_a.lower()}'], last_val[f'ask_{exch_a.lower()}']
                bid_b_val, ask_b_val = last_val[f'bid_{exch_b.lower()}'], last_val[f'ask_{exch_b.lower()}']
            else:
                bid_a_val, ask_a_val, bid_b_val, ask_b_val = 1.0, 1.0, 1.0, 1.0
                
            if pos['direction'] == -1:
                current_z_exec = (np.log(ask_a_val) - np.log(bid_b_val)) * 10000
                gross_pnl = pos['z_entry_exec'] - current_z_exec
            else:
                current_z_exec = (np.log(bid_a_val) - np.log(ask_b_val)) * 10000
                gross_pnl = current_z_exec - pos['z_entry_exec']
                
            trade_log.append({
                'Pair': pair, 'Entry_Time': self.timestamps[pos['entry_idx']], 'Exit_Time': terminal_time,
                'Hold_Time': hold_time, 'Gross_PnL': gross_pnl, 'Net_PnL': gross_pnl - pos['round_trip_fee_bps'], 'Reason': "Terminal_Force_Clean"
            })

        return pd.DataFrame(trade_log)

class PerformanceReporter:
    @staticmethod
    def generate_tear_sheet(tdf: pd.DataFrame, token: str, config: Dict, start_date: pd.Timestamp, end_date: pd.Timestamp):
        if tdf.empty: 
            print(f"\n[!] Portfolio evaluation complete for {token}: 0 trades executed.")
            return
            
        tdf['close_date'] = pd.to_datetime(tdf['Exit_Time']).dt.date
        all_days = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D').date
        days_total = (end_date - start_date).days

        act_daily = tdf.groupby('close_date')['Net_PnL'].sum()
        cal_daily = act_daily.reindex(all_days, fill_value=0.0)
        
        g_cum = cal_daily.sum()
        g_mdd = (cal_daily.cumsum().cummax() - cal_daily.cumsum()).max()
        
        c_sr = (cal_daily.mean() / max(cal_daily.std(), 1e-6)) * np.sqrt(365)
        a_sr = (act_daily.mean() / max(act_daily.std(), 1e-6)) * np.sqrt(365) if len(act_daily) > 1 else 0.0
        
        t_std = tdf['Net_PnL'].std()
        t_sr = (tdf['Net_PnL'].mean() / t_std) if pd.notna(t_std) and t_std > 1e-6 else 0.0
        
        g_wins = tdf.loc[tdf['Net_PnL'] > 0, 'Net_PnL'].sum()
        g_loss = abs(tdf.loc[tdf['Net_PnL'] < 0, 'Net_PnL'].sum())
        g2p = (g_wins / g_loss) if g_loss > 0 else 999.99

        print("\n" + "="*150)
        print(f" 📊 {token} BASELINE 0.2.1 TEAR SHEET | CONFIG: (Max Spread Constraint = {config['max_entry_spread_bps']} bps, Sigma = {config['entry_threshold']}σ)")
        print("="*150)
        print(" [ GLOBAL AGGREGATE TRACK METRICS (Unscaled Basis Points) ]")
        print(f" Total Executed Trades:     {len(tdf):<8}     | Overall Strategy Win Rate: {(tdf['Net_PnL'] > 0).mean() * 100:.2f}%")
        print(f" Cumulative PnL (BPS):      {g_cum:<8.2f}     | Overall MA Reversion Rate: {tdf['Reason'].str.contains('MA').mean() * 100:.2f}%")
        print(f" Calendar Sharpe (Ann.):    {c_sr:<8.2f}     | Trade-Level Sharpe (Trd):  {t_sr:.2f}")
        print(f" Active Sharpe (Ann.):      {a_sr:<8.2f}     | Gain-to-Pain Ratio (G2P):  {g2p:.2f}")
        print(f" Max Single Trade Loss:     {tdf['Net_PnL'].min():<8.2f} bps | Max System Drawdown (BPS): {g_mdd:.2f}")
        print(f" Avg. System Hold Time:     {tdf['Hold_Time'].mean():<8.1f} min   | ")
        
        print("\n [ INDIVIDUAL PAIR FINANCIAL AUDIT ]")
        print(f"{'VENUE PAIR':<12} | {'TRDS':<4} | {'WIN %':<6} | {'MA HIT%':<7} | {'AVG HLD':<7} | {'CUM RET%':<8} | {'MDD %':<7} | {'CAL SR':<6} | {'ACT SR':<6} | {'TRD SR':<6} | {'G2P':<6} | {'MAX LOSS':<9} | {'AVG NET'}")
        print("-" * 150)
        
        pair_metrics = []
        for pair in tdf['Pair'].unique():
            p_tdf = tdf[tdf['Pair'] == pair]
            trds = len(p_tdf)
            win = (p_tdf['Net_PnL'] > 0).mean() * 100
            ma = p_tdf['Reason'].str.contains('MA').mean() * 100
            hld = p_tdf['Hold_Time'].mean()
            net = p_tdf['Net_PnL'].mean()
            mloss = p_tdf['Net_PnL'].min()
            
            p_act = p_tdf.groupby('close_date')['Net_PnL'].sum() / 10000
            p_cal = p_act.reindex(all_days, fill_value=0.0)
            
            ret = p_cal.sum() * 100
            ann_ret = (ret / days_total) * 365.0
            
            c_sr_p = (p_cal.mean() / max(p_cal.std(), 1e-6)) * np.sqrt(365)
            a_sr_p = (p_act.mean() / max(p_act.std(), 1e-6)) * np.sqrt(365) if len(p_act) > 1 else 0.0
            
            std_p = p_tdf['Net_PnL'].std()
            t_sr_p = (p_tdf['Net_PnL'].mean() / std_p) if pd.notna(std_p) and std_p > 1e-6 else 0.0
            
            w_p = p_tdf.loc[p_tdf['Net_PnL'] > 0, 'Net_PnL'].sum()
            l_p = abs(p_tdf.loc[p_tdf['Net_PnL'] < 0, 'Net_PnL'].sum())
            g2p_p = (w_p / l_p) if l_p > 0 else 999.99
            
            mdd_p = (p_cal.cumsum().cummax() - p_cal.cumsum()).max() * 100
            
            pair_metrics.append({
                'Pair': pair, 'Trades': trds, 'Win%': win, 'MA%': ma, 'Hold': hld, 'CumRet': ret, 
                'MDD': mdd_p, 'CalSR': c_sr_p, 'ActSR': a_sr_p, 'TrdSR': t_sr_p, 'G2P': g2p_p, 
                'MaxLoss': mloss, 'AvgNet': net
            })
            
        pair_metrics = sorted(pair_metrics, key=lambda x: x['CumRet'], reverse=True)
        for pm in pair_metrics:
            print(f"{pm['Pair']:<12} | {pm['Trades']:<4.0f} | {pm['Win%']:<5.1f}% | {pm['MA%']:<6.1f}% | {pm['Hold']:<7.1f} | {pm['CumRet']:<7.2f}% | {pm['MDD']:<6.2f}% | {pm['CalSR']:<6.2f} | {pm['ActSR']:<6.2f} | {pm['TrdSR']:<6.2f} | {pm['G2P']:<6.2f} | {pm['MaxLoss']:<9.2f} | {pm['AvgNet']:.2f}")

        print("\n [ TERMINATION ROUTING COUNT ]")
        print(f" Normal MA Hit: {len(tdf[tdf['Reason']=='MA_Hit']):<5} | Delayed MA Hit: {len(tdf[tdf['Reason']=='MA_Hit_Delayed']):<5}")
        print(f" Normal Fundng: {len(tdf[tdf['Reason']=='Funding_Epoch']):<5} | Imputed Fundng: {len(tdf[tdf['Reason']=='Funding_Imputed']):<5}")
        print(f" Normal Timout: {len(tdf[tdf['Reason']=='Timeout']):<5} | Imputed Timout: {len(tdf[tdf['Reason']=='Timeout_Imputed']):<5}")
        term_cleans = len(tdf[tdf['Reason']=='Terminal_Force_Clean'])
        if term_cleans > 0: print(f" Termnl Force C: {term_cleans:<5} | (End of Year Cutoff)")

        print("\n--- HOLD TIME DIAGNOSTIC BY EXIT REASON ---")
        print(tdf.groupby('Reason')['Hold_Time'].agg(['mean', 'count']))
        print("="*150 + "\n")