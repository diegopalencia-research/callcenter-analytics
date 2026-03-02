"""
generate_data.py
Generates a realistic 6-month call center dataset with 20 agents.
Patterns included:
  - Monday mornings: highest abandon rate
  - Night shift: lower CSAT
  - Senior agents: lower AHT, higher FCR
  - Queue depth drives abandon rate
Run this first to create callcenter_data.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
N_AGENTS   = 20
START_DATE = datetime(2024, 7, 1)
END_DATE   = datetime(2024, 12, 31)
SHIFTS     = ['morning', 'afternoon', 'night']

AGENT_NAMES = [
    'Sofia Morales', 'Carlos Pérez', 'Ana López', 'Miguel Torres',
    'Lucía Ramírez', 'Diego Hernández', 'María García', 'José Martínez',
    'Isabella Flores', 'Andrés Castro', 'Valentina Ruiz', 'Eduardo Vega',
    'Camila Ortiz', 'Roberto Díaz', 'Daniela Vargas', 'Fernando Reyes',
    'Gabriela Méndez', 'Pablo Jiménez', 'Laura Sánchez', 'Ricardo Chávez'
]

# Agent experience in months (affects performance)
EXPERIENCE = {
    i: np.random.randint(1, 36) for i in range(1, N_AGENTS + 1)
}


def generate_row(date, agent_id, shift):
    exp     = EXPERIENCE[agent_id]
    dow     = date.weekday()          # 0=Monday, 6=Sunday
    is_mon  = (dow == 0)
    is_fri  = (dow == 4)
    is_night = (shift == 'night')
    is_senior = (exp >= 12)

    # Calls handled
    base_calls = 45 if shift == 'morning' else (40 if shift == 'afternoon' else 30)
    calls = max(10, int(np.random.normal(base_calls, 8)))

    # AHT in seconds: seniors faster, night shift slower
    aht_base = 280 - (exp * 2.5) + (30 if is_night else 0)
    aht = max(120, int(np.random.normal(aht_base, 35)))

    # Queue depth: high on Monday mornings
    queue_base = 18 + (25 if is_mon else 0) + (12 if is_fri else 0) + (-8 if is_night else 0)
    queue = max(0, int(np.random.normal(queue_base, 6)))

    # Abandon rate: driven heavily by queue depth
    abandon_base = 0.03 + (queue * 0.003) + (0.04 if is_mon else 0)
    abandon = float(np.clip(np.random.normal(abandon_base, 0.015), 0.0, 0.25))

    # CSAT: seniors better, night shift worse
    csat_base = 3.8 + (exp * 0.025) - (0.4 if is_night else 0)
    csat = float(np.clip(np.random.normal(csat_base, 0.3), 1.0, 5.0))

    # FCR: seniors better
    fcr_base = 0.55 + (exp * 0.012)
    fcr = float(np.clip(np.random.normal(fcr_base, 0.07), 0.35, 0.95))

    return {
        'date':             date.strftime('%Y-%m-%d'),
        'agent_id':         agent_id,
        'agent_name':       AGENT_NAMES[agent_id - 1],
        'shift':            shift,
        'experience_months': exp,
        'calls_handled':    calls,
        'aht_seconds':      aht,
        'csat_score':       round(csat, 2),
        'abandon_rate':     round(abandon, 4),
        'fcr_rate':         round(fcr, 4),
        'calls_in_queue':   queue,
        'day_of_week':      dow,
        'is_monday':        int(is_mon),
        'is_night_shift':   int(is_night),
    }


def generate_dataset():
    rows  = []
    date  = START_DATE

    while date <= END_DATE:
        if date.weekday() < 5:                         # Mon–Fri only
            for agent_id in range(1, N_AGENTS + 1):
                shift = np.random.choice(
                    SHIFTS, p=[0.45, 0.35, 0.20]       # weighted shift distribution
                )
                rows.append(generate_row(date, agent_id, shift))
        date += timedelta(days=1)

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'agent_id']).reset_index(drop=True)

    print(f"Dataset generated: {len(df):,} rows")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Agents: {df['agent_id'].nunique()}")
    print(f"\nKPI Averages:")
    print(f"  AHT:          {df['aht_seconds'].mean():.0f} sec ({df['aht_seconds'].mean()/60:.1f} min)")
    print(f"  CSAT:         {df['csat_score'].mean():.2f} / 5.0")
    print(f"  Abandon Rate: {df['abandon_rate'].mean():.1%}")
    print(f"  FCR:          {df['fcr_rate'].mean():.1%}")

    return df


if __name__ == '__main__':
    df = generate_dataset()
    df.to_csv('callcenter_data.csv', index=False)
    print("\nSaved → callcenter_data.csv")
