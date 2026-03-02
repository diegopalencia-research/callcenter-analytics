"""
kpi_calculator.py
Core KPI computation functions used across the dashboard.
All functions receive a filtered DataFrame and return clean metrics.
"""

import pandas as pd
import numpy as np


# ── Period comparison ─────────────────────────────────────────────────────────

def split_periods(df: pd.DataFrame):
    """Split df into current half and previous half by date."""
    mid = df['date'].min() + (df['date'].max() - df['date'].min()) / 2
    current  = df[df['date'] > mid]
    previous = df[df['date'] <= mid]
    return current, previous


def delta(current_val, previous_val):
    """Return percentage change between two values."""
    if previous_val == 0:
        return 0.0
    return (current_val - previous_val) / previous_val


# ── KPI aggregation ───────────────────────────────────────────────────────────

def get_kpi_summary(df: pd.DataFrame) -> dict:
    """
    Returns the 4 core KPIs + delta vs previous period.
    Output used for the executive summary cards.
    """
    current, previous = split_periods(df)

    def avg(frame, col):
        return frame[col].mean() if len(frame) > 0 else 0

    aht_curr     = avg(current,  'aht_seconds')
    aht_prev     = avg(previous, 'aht_seconds')
    csat_curr    = avg(current,  'csat_score')
    csat_prev    = avg(previous, 'csat_score')
    abandon_curr = avg(current,  'abandon_rate')
    abandon_prev = avg(previous, 'abandon_rate')
    fcr_curr     = avg(current,  'fcr_rate')
    fcr_prev     = avg(previous, 'fcr_rate')

    return {
        'aht':     {'value': aht_curr,     'delta': delta(aht_curr,     aht_prev),     'unit': 'sec'},
        'csat':    {'value': csat_curr,    'delta': delta(csat_curr,    csat_prev),    'unit': '/5.0'},
        'abandon': {'value': abandon_curr, 'delta': delta(abandon_curr, abandon_prev), 'unit': '%'},
        'fcr':     {'value': fcr_curr,     'delta': delta(fcr_curr,     fcr_prev),     'unit': '%'},
    }


# ── Time series ───────────────────────────────────────────────────────────────

def get_daily_trends(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Daily aggregated KPIs with rolling averages.
    window: rolling average window in days.
    """
    daily = df.groupby('date').agg(
        aht_seconds  = ('aht_seconds',  'mean'),
        csat_score   = ('csat_score',   'mean'),
        abandon_rate = ('abandon_rate', 'mean'),
        fcr_rate     = ('fcr_rate',     'mean'),
        calls_total  = ('calls_handled', 'sum'),
    ).reset_index()

    for col in ['aht_seconds', 'csat_score', 'abandon_rate', 'fcr_rate']:
        daily[f'{col}_rolling'] = daily[col].rolling(window, min_periods=1).mean()

    return daily


# ── Agent performance ─────────────────────────────────────────────────────────

def get_agent_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks all agents by a composite performance score.
    Score: 30% CSAT + 30% FCR + 20% (inverse AHT) + 20% (inverse abandon rate)
    """
    agent = df.groupby(['agent_id', 'agent_name', 'experience_months']).agg(
        avg_aht          = ('aht_seconds',   'mean'),
        avg_csat         = ('csat_score',    'mean'),
        avg_abandon      = ('abandon_rate',  'mean'),
        avg_fcr          = ('fcr_rate',      'mean'),
        total_calls      = ('calls_handled', 'sum'),
        days_worked      = ('date',          'nunique'),
    ).reset_index()

    # Normalize each metric 0–1
    def norm(series, invert=False):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series), index=series.index)
        n = (series - mn) / (mx - mn)
        return 1 - n if invert else n

    agent['score'] = (
        norm(agent['avg_csat'])    * 0.30 +
        norm(agent['avg_fcr'])     * 0.30 +
        norm(agent['avg_aht'],    invert=True) * 0.20 +
        norm(agent['avg_abandon'], invert=True) * 0.20
    )

    agent['rank'] = agent['score'].rank(ascending=False).astype(int)

    # Quartile flags
    q75 = agent['score'].quantile(0.75)
    q25 = agent['score'].quantile(0.25)
    agent['tier'] = agent['score'].apply(
        lambda s: '🟢 Top'    if s >= q75 else
                  ('🔴 Risk'   if s <= q25 else '⚪ Mid')
    )

    return agent.sort_values('rank').reset_index(drop=True)


# ── Shift comparison ──────────────────────────────────────────────────────────

def get_shift_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate KPIs by shift for comparison."""
    return df.groupby('shift').agg(
        avg_aht     = ('aht_seconds',  'mean'),
        avg_csat    = ('csat_score',   'mean'),
        avg_abandon = ('abandon_rate', 'mean'),
        avg_fcr     = ('fcr_rate',     'mean'),
        total_calls = ('calls_handled', 'sum'),
    ).reset_index().round(4)


# ── Anomaly detection ─────────────────────────────────────────────────────────

def flag_anomalies(daily: pd.DataFrame, col: str, threshold: float = 2.0) -> pd.Series:
    """
    Flag rows where value is more than `threshold` std devs from the mean.
    Returns boolean Series.
    """
    mean = daily[col].mean()
    std  = daily[col].std()
    return (daily[col] - mean).abs() > (threshold * std)
