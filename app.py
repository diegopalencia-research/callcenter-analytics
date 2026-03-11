"""
app.py — Call Center Performance Analytics v3.0
Diego José Palencia Robles · 2026
Stack: Streamlit + Supabase + Plotly
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CC Analytics · Palencia",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    'bg':       '#070E19',
    'surface':  '#0C1825',
    'surface2': '#101F2E',
    'surface3': '#152435',
    'border':   '#1A2E45',
    'border2':  '#1F3850',
    'accent':   '#00D4AA',
    'accent2':  '#0EA5E9',
    'accent3':  '#A78BFA',
    'warn':     '#F97316',
    'danger':   '#EF4444',
    'success':  '#22C55E',
    'text':     '#E2EAF0',
    'text2':    '#7A99B0',
    'text3':    '#3D5A70',
    'gold':     '#F59E0B',
}

TEAM_COLORS = {
    'Tech Support': '#0EA5E9',
    'Billing':      '#A78BFA',
    'Sales':        '#22C55E',
    'Retention':    '#F59E0B',
    'General':      '#00D4AA',
}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: {C['bg']};
    color: {C['text']};
  }}
  .stApp {{ background: {C['bg']}; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: {C['surface']};
    border-right: 1px solid {C['border']};
  }}
  section[data-testid="stSidebar"] * {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
  }}

  /* Main container */
  .block-container {{
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1280px;
  }}

  /* Inputs */
  .stSelectbox > div > div,
  .stMultiSelect > div > div {{
    background: {C['surface2']} !important;
    border: 1px solid {C['border2']} !important;
    color: {C['text']} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
  }}
  .stSlider > div {{
    font-family: 'IBM Plex Mono', monospace !important;
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background: {C['surface']};
    border-bottom: 1px solid {C['border']};
    gap: 0;
  }}
  .stTabs [data-baseweb="tab"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.09em;
    color: {C['text2']};
    padding: 0.65rem 1.4rem;
    border: none;
    background: transparent;
    text-transform: uppercase;
  }}
  .stTabs [aria-selected="true"] {{
    color: {C['accent']} !important;
    border-bottom: 2px solid {C['accent']} !important;
    background: transparent !important;
  }}

  /* Dataframe */
  .stDataFrame {{ border: 1px solid {C['border']} !important; }}
  .stDataFrame th {{
    background: {C['surface2']} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    color: {C['text2']} !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }}
  .stDataFrame td {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
  }}

  /* Native metrics */
  [data-testid="metric-container"] {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 1rem !important;
  }}

  hr {{ border-color: {C['border']} !important; margin: 1.2rem 0; }}
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* Plotly toolbar hide */
  .modebar {{ display: none !important; }}
</style>
""", unsafe_allow_html=True)


# ── UI Components ─────────────────────────────────────────────────────────────

def page_header(title, subtitle=None, badge=None):
    badge_html = f'<span style="background:{C["surface3"]};border:1px solid {C["border2"]};font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;letter-spacing:0.1em;color:{C["accent"]};padding:0.2rem 0.6rem;border-radius:2px;margin-left:0.8rem;vertical-align:middle;">{badge}</span>' if badge else ''
    sub = f'<div style="font-size:0.82rem;color:{C["text2"]};margin-top:0.35rem;font-family:\'IBM Plex Mono\',monospace;">{subtitle}</div>' if subtitle else ''
    st.markdown(f"""
    <div style='border-bottom:1px solid {C["border"]};padding-bottom:1.1rem;margin-bottom:1.6rem;'>
      <div style='font-family:"IBM Plex Mono",monospace;font-size:0.58rem;letter-spacing:0.22em;color:{C["accent"]};text-transform:uppercase;margin-bottom:0.35rem;opacity:0.8;'>
        CC-ANALYTICS / PALENCIA-RESEARCH / v3.0
      </div>
      <div style='font-family:"IBM Plex Mono",monospace;font-size:1.6rem;font-weight:600;color:{C["text"]};letter-spacing:-0.02em;'>
        {title}{badge_html}
      </div>{sub}
    </div>""", unsafe_allow_html=True)


def section_title(text, icon=""):
    st.markdown(f"""
    <div style='font-family:"IBM Plex Mono",monospace;font-size:0.65rem;letter-spacing:0.16em;
    color:{C["text3"]};text-transform:uppercase;border-left:2px solid {C["accent"]};
    padding-left:0.75rem;margin:1.8rem 0 0.9rem;'>
      {icon + " " if icon else ""}{text}
    </div>""", unsafe_allow_html=True)


def kpi_card(col, label, value, delta_pct, target=None, invert=False, unit=""):
    is_good = (delta_pct < 0) if invert else (delta_pct > 0)
    arrow = "▲" if delta_pct > 0 else "▼"
    d_color = C['success'] if is_good else C['danger']
    top_color = C['accent'] if is_good else C['warn']
    tgt_html = f'<div style="font-size:0.62rem;color:{C["text3"]};margin-top:0.3rem;font-family:\'IBM Plex Mono\',monospace;">TARGET {target}</div>' if target else ''
    col.markdown(f"""
    <div style='background:{C["surface"]};border:1px solid {C["border"]};
    border-top:2px solid {top_color};border-radius:6px;padding:1.2rem 1.3rem;
    transition:all 0.2s;'>
      <div style='font-family:"IBM Plex Mono",monospace;font-size:0.58rem;
      letter-spacing:0.14em;color:{C["text3"]};text-transform:uppercase;
      margin-bottom:0.55rem;'>{label}</div>
      <div style='font-family:"IBM Plex Mono",monospace;font-size:2.1rem;
      font-weight:600;color:{C["text"]};line-height:1;'>{value}</div>
      <div style='font-family:"IBM Plex Mono",monospace;font-size:0.68rem;
      color:{d_color};margin-top:0.45rem;'>
        {arrow} {abs(delta_pct):.1f}% vs prior period
      </div>{tgt_html}
    </div>""", unsafe_allow_html=True)


def stat_row(cols_data):
    """Render a row of small stat pills."""
    pills = ""
    for label, val, color in cols_data:
        pills += f'<div style="background:{C["surface2"]};border:1px solid {C["border"]};border-radius:4px;padding:0.5rem 1rem;display:inline-block;margin-right:0.5rem;margin-bottom:0.4rem;"><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;color:{C["text3"]};letter-spacing:0.1em;text-transform:uppercase;">{label}&nbsp;&nbsp;</span><span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.85rem;font-weight:600;color:{color};">{val}</span></div>'
    st.markdown(f'<div style="margin:0.5rem 0 1rem;">{pills}</div>', unsafe_allow_html=True)


def alert_box(text, level='warn'):
    cfg = {
        'warn':    (C['warn'],    'rgba(249,115,22,0.07)',  '⚠ ALERT'),
        'danger':  (C['danger'],  'rgba(239,68,68,0.07)',   '[CRITICAL]'),
        'info':    (C['accent'],  'rgba(0,212,170,0.07)',   '✓ INFO'),
        'success': (C['success'], 'rgba(34,197,94,0.07)',   '✓ OK'),
    }
    color, bg, prefix = cfg.get(level, cfg['warn'])
    st.markdown(f"""
    <div style='background:{bg};border-left:3px solid {color};padding:0.75rem 1.1rem;
    margin:0.7rem 0;border-radius:0 5px 5px 0;'>
      <span style='font-family:"IBM Plex Mono",monospace;font-size:0.7rem;
      color:{color};letter-spacing:0.08em;'>{prefix}</span>
      <span style='font-family:"IBM Plex Mono",monospace;font-size:0.77rem;
      color:{C["text2"]};margin-left:0.6rem;'>{text}</span>
    </div>""", unsafe_allow_html=True)


# ── Plotly theme ──────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C['bg'],
    plot_bgcolor=C['surface'],
    font=dict(family='IBM Plex Mono', color=C['text2'], size=11),
    xaxis=dict(gridcolor=C['border'], linecolor=C['border'], tickfont=dict(size=10)),
    yaxis=dict(gridcolor=C['border'], linecolor=C['border'], tickfont=dict(size=10)),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=C['border'], font=dict(size=10)),
    hoverlabel=dict(bgcolor=C['surface3'], bordercolor=C['border2'],
                    font=dict(family='IBM Plex Mono', size=11)),
)

def apply_layout(fig, **kwargs):
    layout = {**PLOTLY_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    return fig


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    """Load from Supabase if credentials available, else fallback to CSV."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        from supabase import create_client
        sb = create_client(url, key)
        # Fetch in batches (Supabase default limit is 1000)
        all_rows = []
        offset = 0
        batch = 1000
        while True:
            res = sb.table('daily_metrics').select('*').range(offset, offset + batch - 1).execute()
            rows = res.data
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < batch:
                break
            offset += batch
        df = pd.DataFrame(all_rows)
        df['date'] = pd.to_datetime(df['date'])
        return df, "supabase"
    except Exception:
        # Fallback to CSV
        base = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base, 'callcenter_data.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=['date'])
            return df, "csv"
        st.error("No data source found. Add Supabase secrets or callcenter_data.csv.")
        st.stop()


@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    mp = os.path.join(base, 'models', 'regressor.pkl')
    mep = os.path.join(base, 'models', 'reg_metrics.json')
    if not os.path.exists(mp):
        return None, None, {}
    with open(mp, 'rb') as f:
        models = pickle.load(f)
    metrics = json.load(open(mep)) if os.path.exists(mep) else {}
    return models['rf'], models['lr'], metrics


def engineer_features(df):
    f = pd.DataFrame(index=df.index)
    le = LabelEncoder()
    le.fit(['afternoon', 'morning', 'night'])
    f['day_of_week']       = df['day_of_week']
    f['is_monday']         = df['is_monday']
    f['is_night_shift']    = df['is_night_shift']
    f['shift_enc']         = le.transform(df['shift'])
    f['calls_in_queue']    = df['calls_in_queue']
    f['aht_seconds']       = df['aht_seconds']
    f['experience_months'] = df['experience_months']
    f['calls_handled']     = df['calls_handled']
    f['csat_score']        = df['csat_score']
    f['fcr_rate']          = df['fcr_rate']
    f['queue_x_monday']    = df['calls_in_queue'] * df['is_monday']
    f['queue_x_night']     = df['calls_in_queue'] * df['is_night_shift']
    return f


# ── KPI helpers ───────────────────────────────────────────────────────────────

def get_kpis(df):
    mid = df['date'].min() + (df['date'].max() - df['date'].min()) / 2
    curr = df[df['date'] > mid]
    prev = df[df['date'] <= mid]
    def avg(d, c): return d[c].mean() if len(d) > 0 else 0
    def delt(c, p): return (c - p) / abs(p) * 100 if p != 0 else 0
    out = {}
    for col in ['aht_seconds', 'csat_score', 'abandon_rate', 'fcr_rate', 'escalation_rate', 'transfer_rate']:
        if col in df.columns:
            c = avg(curr, col); p = avg(prev, col)
            out[col] = {'value': c, 'delta': delt(c, p)}
    return out


def norm_score(series, invert=False):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    n = (series - mn) / (mx - mn)
    return 1 - n if invert else n


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(df):
    st.sidebar.markdown(f"""
    <div style='padding:0.6rem 0 1.2rem;border-bottom:1px solid {C["border"]};margin-bottom:1.1rem;'>
      <div style='font-size:0.55rem;letter-spacing:0.22em;color:{C["accent"]};
      text-transform:uppercase;margin-bottom:0.2rem;opacity:0.8;'>CC-ANALYTICS</div>
      <div style='font-size:0.95rem;font-weight:600;color:{C["text"]};'>PALENCIA RESEARCH</div>
      <div style='font-size:0.62rem;color:{C["text3"]};margin-top:0.2rem;'>
        Call Center Intelligence v3.0
      </div>
    </div>""", unsafe_allow_html=True)

    st.sidebar.markdown(f"<div style='font-size:0.58rem;letter-spacing:0.14em;color:{C['text3']};text-transform:uppercase;margin-bottom:0.5rem;'>NAVIGATION</div>", unsafe_allow_html=True)

    pages = ["OVERVIEW", "TRENDS", "TEAMS", "AGENTS", "PREDICTOR"]
    page = st.sidebar.radio("nav", label_visibility="collapsed", options=pages)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<div style='font-size:0.58rem;letter-spacing:0.14em;color:{C['text3']};text-transform:uppercase;margin-bottom:0.5rem;'>FILTERS</div>", unsafe_allow_html=True)

    min_d = df['date'].min().date()
    max_d = df['date'].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    shifts = st.sidebar.multiselect("Shift", options=['morning', 'afternoon', 'night'],
                                    default=['morning', 'afternoon', 'night'])

    teams = []
    if 'team' in df.columns:
        teams = st.sidebar.multiselect("Team", options=sorted(df['team'].unique()), default=[])

    agents = st.sidebar.multiselect("Agent (all if empty)",
                                    options=sorted(df['agent_name'].unique()), default=[])

    years = []
    if 'year' in df.columns:
        available_years = sorted(df['year'].unique())
        years = st.sidebar.multiselect("Year", options=available_years, default=available_years)

    st.sidebar.markdown("---")

    n_rows = len(df)
    n_agents = df['agent_id'].nunique()
    date_span = f"{df['date'].min().strftime('%b %Y')} → {df['date'].max().strftime('%b %Y')}"
    has_teams = 'team' in df.columns

    st.sidebar.markdown(f"""
    <div style='font-size:0.62rem;color:{C["text3"]};letter-spacing:0.04em;line-height:2;'>
      ROWS&nbsp;&nbsp;&nbsp;&nbsp;{n_rows:,}<br>
      AGENTS&nbsp;&nbsp;{n_agents}<br>
      PERIOD&nbsp;&nbsp;{date_span}<br>
      TEAMS&nbsp;&nbsp;&nbsp;{"5" if has_teams else "—"}<br>
      MODEL&nbsp;&nbsp;&nbsp;RF R² 0.919
    </div>""", unsafe_allow_html=True)

    return page, date_range, shifts, teams, agents, years


def apply_filters(df, date_range, shifts, teams, agents, years):
    if len(date_range) == 2:
        df = df[(df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])]
    if shifts:
        df = df[df['shift'].isin(shifts)]
    if teams and 'team' in df.columns:
        df = df[df['team'].isin(teams)]
    if agents:
        df = df[df['agent_name'].isin(agents)]
    if years and 'year' in df.columns:
        df = df[df['year'].isin(years)]
    return df


# ── PAGE: OVERVIEW ────────────────────────────────────────────────────────────

def page_overview(df):
    has_team = 'team' in df.columns
    date_str = f"{df['date'].min().strftime('%d %b %Y')} → {df['date'].max().strftime('%d %b %Y')}"
    badge = f"{df['year'].nunique()} YRS" if 'year' in df.columns else ""

    page_header("EXECUTIVE OVERVIEW",
                f"{len(df):,} records · {df['agent_id'].nunique()} agents · {date_str}",
                badge=badge)

    kpis = get_kpis(df)

    # KPI Cards row
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "AVG HANDLE TIME",
             f"{kpis['aht_seconds']['value']:.0f}s",
             kpis['aht_seconds']['delta'], "< 300s", invert=True)
    kpi_card(c2, "CSAT SCORE",
             f"{kpis['csat_score']['value']:.2f}",
             kpis['csat_score']['delta'], "> 4.20")
    kpi_card(c3, "ABANDON RATE",
             f"{kpis['abandon_rate']['value']*100:.1f}%",
             kpis['abandon_rate']['delta'], "< 5.0%", invert=True)
    kpi_card(c4, "FIRST CALL RES.",
             f"{kpis['fcr_rate']['value']*100:.1f}%",
             kpis['fcr_rate']['delta'], "> 70%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Extended KPIs if available
    if 'escalation_rate' in kpis and 'transfer_rate' in kpis:
        c5, c6, c7, c8 = st.columns(4)
        kpi_card(c5, "ESCALATION RATE",
                 f"{kpis['escalation_rate']['value']*100:.1f}%",
                 kpis['escalation_rate']['delta'], "< 8%", invert=True)
        kpi_card(c6, "TRANSFER RATE",
                 f"{kpis['transfer_rate']['value']*100:.1f}%",
                 kpis['transfer_rate']['delta'], "< 10%", invert=True)
        total_calls = df['calls_handled'].sum()
        avg_exp = df['experience_months'].mean() if 'experience_months' in df.columns else 0
        kpi_card(c7, "TOTAL CALLS",
                 f"{total_calls:,}", 0)
        kpi_card(c8, "AVG EXPERIENCE",
                 f"{avg_exp:.0f} mo", 0)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two charts side by side
    col_left, col_right = st.columns([3, 2])

    with col_left:
        section_title("WEEKLY CALL VOLUME & ABANDON RATE")
        daily = df.groupby('date').agg(
            calls=('calls_handled', 'sum'),
            abandon=('abandon_rate', 'mean')
        ).reset_index()
        daily['week'] = daily['date'].dt.to_period('W').apply(lambda x: x.start_time)
        weekly = daily.groupby('week').agg(calls=('calls', 'sum'), abandon=('abandon', 'mean')).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=weekly['week'], y=weekly['calls'],
            name='Calls', marker_color=C['accent2'],
            opacity=0.75, marker_line_width=0
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=weekly['week'], y=weekly['abandon'] * 100,
            name='Abandon %', line=dict(color=C['warn'], width=2),
            mode='lines+markers', marker=dict(size=5)
        ), secondary_y=True)
        fig.add_hline(y=5.0, line_dash="dash", line_color=C['danger'],
                      opacity=0.5, secondary_y=True,
                      annotation_text="5% target", annotation_font_size=9,
                      annotation_font_color=C['danger'])
        fig.update_yaxes(title_text="Total Calls", secondary_y=False,
                         gridcolor=C['border'], title_font_size=10)
        fig.update_yaxes(title_text="Abandon Rate %", secondary_y=True,
                         gridcolor='rgba(0,0,0,0)', title_font_size=10)
        apply_layout(fig, height=280)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col_right:
        section_title("SHIFT BREAKDOWN")
        shift_df = df.groupby('shift').agg(
            avg_csat=('csat_score', 'mean'),
            avg_abandon=('abandon_rate', 'mean'),
            avg_fcr=('fcr_rate', 'mean'),
            total_calls=('calls_handled', 'sum')
        ).reset_index()

        fig2 = go.Figure()
        shift_colors = {'morning': C['accent'], 'afternoon': C['accent2'], 'night': C['accent3']}
        for _, row in shift_df.iterrows():
            fig2.add_trace(go.Bar(
                name=row['shift'].upper(),
                x=['CSAT', 'FCR %', 'Abandon %'],
                y=[row['avg_csat'] / 5 * 100,
                   row['avg_fcr'] * 100,
                   row['avg_abandon'] * 100],
                marker_color=shift_colors.get(row['shift'], C['accent']),
                opacity=0.85
            ))
        apply_layout(fig2, height=280, barmode='group',
                     yaxis=dict(title='Score / Rate (%)', gridcolor=C['border']))
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    # Alerts
    abandon_val = kpis['abandon_rate']['value'] * 100
    csat_val = kpis['csat_score']['value']
    if abandon_val > 10:
        alert_box(f"Abandon rate {abandon_val:.1f}% is CRITICAL (>10%). Immediate staffing review required.", level='danger')
    elif abandon_val > 5:
        alert_box(f"Abandon rate {abandon_val:.1f}% exceeds 5% target. Review queue depth on Mondays.", level='warn')
    if csat_val < 3.5:
        alert_box(f"CSAT {csat_val:.2f} is critically low. Night shift coaching recommended.", level='danger')
    elif csat_val < 4.0:
        alert_box(f"CSAT {csat_val:.2f} below 4.0 target. Review night shift performance.", level='warn')

    # Day of week heatmap
    if len(df) > 100:
        section_title("ABANDON RATE BY DAY × SHIFT")
        pivot = df.groupby(['day_of_week', 'shift'])['abandon_rate'].mean().unstack(fill_value=0)
        pivot.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][:len(pivot)]
        fig3 = go.Figure(go.Heatmap(
            z=(pivot.values * 100).round(1),
            x=[c.upper() for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=[[0, C['surface3']], [0.5, C['accent2']], [1, C['danger']]],
            text=(pivot.values * 100).round(1),
            texttemplate="%{text}%",
            textfont=dict(size=11, family='IBM Plex Mono'),
            showscale=True,
            hovertemplate='%{y} · %{x}<br>Abandon: %{z}%<extra></extra>'
        ))
        apply_layout(fig3, height=200,
                     xaxis=dict(side='top', gridcolor='rgba(0,0,0,0)'),
                     yaxis=dict(gridcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})


# ── PAGE: TRENDS ──────────────────────────────────────────────────────────────

def page_trends(df):
    page_header("TEMPORAL ANALYSIS", "Daily KPI trends with rolling average · anomaly detection")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        window = st.selectbox("Rolling window", [3, 7, 14, 30], index=1,
                              format_func=lambda x: f"{x}d avg")
        show_anomalies = st.checkbox("Highlight anomalies", value=True)

    daily = df.groupby('date').agg(
        aht_seconds=('aht_seconds', 'mean'),
        csat_score=('csat_score', 'mean'),
        abandon_rate=('abandon_rate', 'mean'),
        fcr_rate=('fcr_rate', 'mean'),
        calls_total=('calls_handled', 'sum'),
    ).reset_index()

    for col in ['aht_seconds', 'csat_score', 'abandon_rate', 'fcr_rate']:
        daily[f'{col}_roll'] = daily[col].rolling(window, min_periods=1).mean()
        mean_v = daily[col].mean()
        std_v = daily[col].std()
        daily[f'{col}_anom'] = (daily[col] - mean_v).abs() > (2.0 * std_v)

    tab1, tab2, tab3, tab4 = st.tabs(["  AHT  ", "  CSAT  ", "  ABANDON RATE  ", "  FCR  "])

    def trend_chart(col, label, target, color, fmt_pct=False):
        fig = go.Figure()

        # Fill area
        fig.add_trace(go.Scatter(
            x=daily['date'], y=daily[col],
            fill='tozeroy', fillcolor='rgba(0,0,0,0)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False, hoverinfo='skip'
        ))

        # Raw line
        y_vals = daily[col] * 100 if fmt_pct else daily[col]
        fig.add_trace(go.Scatter(
            x=daily['date'], y=y_vals,
            mode='lines', name='Daily',
            line=dict(color=color, width=1, dash='dot'),
            opacity=0.4
        ))

        # Rolling avg
        y_roll = daily[f'{col}_roll'] * 100 if fmt_pct else daily[f'{col}_roll']
        fig.add_trace(go.Scatter(
            x=daily['date'], y=y_roll,
            mode='lines', name=f'{window}d avg',
            line=dict(color=color, width=2.5)
        ))

        # Target line
        t_val = target * 100 if fmt_pct else target
        fig.add_hline(y=t_val, line_dash="dash", line_color=C['danger'],
                      opacity=0.6,
                      annotation_text=f"Target: {t_val:.0f}{'%' if fmt_pct else ''}",
                      annotation_font_size=9, annotation_font_color=C['danger'],
                      annotation_position="bottom right")

        # Anomaly markers
        if show_anomalies:
            anom = daily[daily[f'{col}_anom']]
            a_y = anom[col] * 100 if fmt_pct else anom[col]
            if len(anom) > 0:
                fig.add_trace(go.Scatter(
                    x=anom['date'], y=a_y,
                    mode='markers', name='Anomaly',
                    marker=dict(color=C['warn'], size=8, symbol='x',
                                line=dict(color=C['warn'], width=2))
                ))

        apply_layout(fig, height=300,
                     yaxis=dict(title=label, gridcolor=C['border']))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Stats row
        v = daily[col]
        stat_row([
            ("MIN",  f"{v.min()*100:.1f}%" if fmt_pct else f"{v.min():.1f}", C['success']),
            ("MAX",  f"{v.max()*100:.1f}%" if fmt_pct else f"{v.max():.1f}", C['danger']),
            ("AVG",  f"{v.mean()*100:.1f}%" if fmt_pct else f"{v.mean():.1f}", C['accent']),
            ("STD",  f"{v.std()*100:.2f}%" if fmt_pct else f"{v.std():.2f}", C['text3']),
            ("ANOMALIES", str(daily[f'{col}_anom'].sum()), C['warn']),
        ])

    with tab1: trend_chart('aht_seconds', 'AHT (seconds)', 300, C['accent2'])
    with tab2: trend_chart('csat_score', 'CSAT Score', 4.2, C['accent'])
    with tab3: trend_chart('abandon_rate', 'Abandon Rate', 0.05, C['warn'], fmt_pct=True)
    with tab4: trend_chart('fcr_rate', 'FCR Rate', 0.70, C['success'], fmt_pct=True)


# ── PAGE: TEAMS ───────────────────────────────────────────────────────────────

def page_teams(df):
    if 'team' not in df.columns:
        st.info("Team data not available. Upload the v2.0 dataset with team column.")
        return

    page_header("TEAM PERFORMANCE", "Comparative analysis across 5 operational teams",
                badge="NEW · 2025–2026")

    teams_agg = df.groupby('team').agg(
        avg_aht=('aht_seconds', 'mean'),
        avg_csat=('csat_score', 'mean'),
        avg_abandon=('abandon_rate', 'mean'),
        avg_fcr=('fcr_rate', 'mean'),
        avg_escalation=('escalation_rate', 'mean') if 'escalation_rate' in df.columns else ('fcr_rate', 'count'),
        total_calls=('calls_handled', 'sum'),
        active_agents=('agent_id', 'nunique'),
    ).reset_index()

    # Composite score per team
    teams_agg['score'] = (
        norm_score(teams_agg['avg_csat']) * 0.30 +
        norm_score(teams_agg['avg_fcr']) * 0.30 +
        norm_score(teams_agg['avg_aht'], invert=True) * 0.20 +
        norm_score(teams_agg['avg_abandon'], invert=True) * 0.20
    )
    teams_agg = teams_agg.sort_values('score', ascending=False)

    # Radar chart
    col_l, col_r = st.columns([2, 1])

    with col_l:
        section_title("KPI RADAR — ALL TEAMS")
        categories = ['CSAT', 'FCR', 'Low AHT', 'Low Abandon', 'Score']

        fig = go.Figure()
        for _, row in teams_agg.iterrows():
            vals = [
                row['avg_csat'] / 5,
                row['avg_fcr'],
                1 - (row['avg_aht'] / 500),
                1 - row['avg_abandon'],
                row['score'],
            ]
            vals_pct = [v * 100 for v in vals]
            vals_pct.append(vals_pct[0])
            cats = categories + [categories[0]]

            fig.add_trace(go.Scatterpolar(
                r=vals_pct, theta=cats,
                fill='toself',
                name=row['team'],
                line=dict(color=TEAM_COLORS.get(row['team'], C['accent']), width=2),
                fillcolor=TEAM_COLORS.get(row['team'], C['accent']),
                opacity=0.12,
                opacity=0.9
            ))

        apply_layout(fig, height=380,
                     polar=dict(
                         bgcolor=C['surface'],
                         radialaxis=dict(visible=True, range=[0, 100],
                                         gridcolor=C['border'], tickfont_size=9),
                         angularaxis=dict(gridcolor=C['border'])
                     ))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col_r:
        section_title("COMPOSITE RANKING")
        for i, (_, row) in enumerate(teams_agg.iterrows()):
            medal = f"#{i+1:02d}"
            color = TEAM_COLORS.get(row['team'], C['accent'])
            st.markdown(f"""
            <div style='background:{C["surface"]};border:1px solid {C["border"]};
            border-left:3px solid {color};border-radius:5px;
            padding:0.7rem 1rem;margin-bottom:0.5rem;'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                  <span style='font-size:0.75rem;margin-right:0.4rem;'>{medal}</span>
                  <span style='font-family:"IBM Plex Mono",monospace;font-size:0.8rem;
                  color:{C["text"]};font-weight:500;'>{row['team']}</span>
                </div>
                <span style='font-family:"IBM Plex Mono",monospace;font-size:0.9rem;
                font-weight:600;color:{color};'>{row['score']:.2f}</span>
              </div>
              <div style='font-family:"IBM Plex Mono",monospace;font-size:0.62rem;
              color:{C["text3"]};margin-top:0.25rem;'>
                {row['active_agents']} agents · {row['total_calls']:,} calls
              </div>
            </div>""", unsafe_allow_html=True)

    # Monthly trend by team
    section_title("MONTHLY CSAT TREND BY TEAM")
    if 'year' in df.columns and 'month' in df.columns:
        monthly = df.groupby(['year', 'month', 'team'])['csat_score'].mean().reset_index()
        monthly['period'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
        monthly = monthly.sort_values('period')

        fig2 = go.Figure()
        for team in df['team'].unique():
            t_data = monthly[monthly['team'] == team]
            fig2.add_trace(go.Scatter(
                x=t_data['period'], y=t_data['csat_score'],
                mode='lines+markers',
                name=team,
                line=dict(color=TEAM_COLORS.get(team, C['accent']), width=2),
                marker=dict(size=5)
            ))
        fig2.add_hline(y=4.2, line_dash="dash", line_color=C['danger'], opacity=0.5,
                       annotation_text="4.2 target", annotation_font_size=9,
                       annotation_font_color=C['danger'])
        apply_layout(fig2, height=300,
                     yaxis=dict(title='Avg CSAT', range=[3.0, 5.2], gridcolor=C['border']))
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    # Team comparison table
    section_title("TEAM METRICS TABLE")
    display = teams_agg.copy()
    display['avg_aht'] = display['avg_aht'].round(0).astype(int)
    display['avg_csat'] = display['avg_csat'].round(2)
    display['avg_abandon'] = (display['avg_abandon'] * 100).round(2)
    display['avg_fcr'] = (display['avg_fcr'] * 100).round(2)
    display['score'] = display['score'].round(3)
    display.columns = ['Team', 'AHT (s)', 'CSAT', 'Abandon %', 'FCR %',
                        'Escalation %', 'Total Calls', 'Agents', 'Score']
    if 'escalation_rate' not in df.columns:
        display = display.drop(columns=['Escalation %'])
    st.dataframe(display, use_container_width=True, hide_index=True)


# ── PAGE: AGENTS ──────────────────────────────────────────────────────────────

def page_agents(df):
    page_header("AGENT PERFORMANCE",
                "Composite score: CSAT 30% · FCR 30% · AHT 20% · Abandon 20%")

    group_cols = ['agent_id', 'agent_name', 'experience_months']
    if 'team' in df.columns:
        group_cols.append('team')

    agent = df.groupby(group_cols).agg(
        avg_aht=('aht_seconds', 'mean'),
        avg_csat=('csat_score', 'mean'),
        avg_abandon=('abandon_rate', 'mean'),
        avg_fcr=('fcr_rate', 'mean'),
        total_calls=('calls_handled', 'sum'),
        days_worked=('date', 'nunique'),
    ).reset_index()

    agent['score'] = (
        norm_score(agent['avg_csat']) * 0.30 +
        norm_score(agent['avg_fcr']) * 0.30 +
        norm_score(agent['avg_aht'], invert=True) * 0.20 +
        norm_score(agent['avg_abandon'], invert=True) * 0.20
    )

    q75 = agent['score'].quantile(0.75)
    q25 = agent['score'].quantile(0.25)
    agent['tier'] = agent['score'].apply(
        lambda s: 'TOP' if s >= q75 else ('RISK' if s <= q25 else 'MID')
    )
    agent = agent.sort_values('score', ascending=False).reset_index(drop=True)
    agent.index += 1

    # Agent score bar chart — horizontal, clean
    section_title("COMPOSITE PERFORMANCE SCORE — ALL AGENTS")

    fig_bar = go.Figure()
    tier_colors = {'TOP': C['success'], 'MID': C['accent2'], 'RISK': C['danger']}

    for tier in ['TOP', 'MID', 'RISK']:
        t_data = agent[agent['tier'] == tier].sort_values('score', ascending=True)
        if len(t_data) == 0:
            continue
        label_col = 'agent_name'
        fig_bar.add_trace(go.Bar(
            y=t_data[label_col],
            x=t_data['score'],
            orientation='h',
            name=tier,
            marker_color=tier_colors[tier],
            opacity=0.85,
            text=t_data['score'].apply(lambda v: f"{v:.3f}"),
            textposition='outside',
            textfont=dict(size=9, family='IBM Plex Mono'),
            customdata=t_data[['avg_csat', 'avg_fcr', 'avg_aht', 'total_calls']].values,
            hovertemplate=(
                '<b>%{y}</b><br>'
                'Score: %{x:.3f}<br>'
                'CSAT: %{customdata[0]:.2f}<br>'
                'FCR: %{customdata[1]:.1%}<br>'
                'AHT: %{customdata[2]:.0f}s<br>'
                'Calls: %{customdata[3]:,}<extra></extra>'
            )
        ))

    apply_layout(fig_bar,
                 height=max(400, len(agent) * 22),
                 barmode='stack',
                 xaxis=dict(title='Composite Score (0–1)', gridcolor=C['border'],
                            range=[0, 1.15]),
                 yaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=10)),
                 margin=dict(l=130, r=60, t=20, b=20))
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    # CSAT vs AHT quadrant
    section_title("PERFORMANCE QUADRANT — CSAT vs AHT")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        fig = go.Figure()
        for tier, color in tier_colors.items():
            t_data = agent[agent['tier'] == tier]
            if len(t_data) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=t_data['avg_aht'],
                y=t_data['avg_csat'],
                mode='markers+text',
                name=tier,
                text=t_data['agent_name'].str.split().str[0],
                textposition='top center',
                textfont=dict(size=8, color=C['text3']),
                marker=dict(color=color, size=11, opacity=0.9,
                            line=dict(color=C['bg'], width=1.5)),
                customdata=t_data[['agent_name', 'total_calls', 'avg_fcr', 'score']].values,
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'
                    'AHT: %{x:.0f}s · CSAT: %{y:.2f}<br>'
                    'FCR: %{customdata[2]:.1%} · Calls: %{customdata[1]:,}<br>'
                    'Score: %{customdata[3]:.3f}<extra></extra>'
                )
            ))

        fig.add_vline(x=300, line_dash="dash", line_color=C['border2'], opacity=0.7,
                      annotation_text="AHT target 300s",
                      annotation_font_size=9, annotation_font_color=C['text3'])
        fig.add_hline(y=4.2, line_dash="dash", line_color=C['border2'], opacity=0.7,
                      annotation_text="CSAT target 4.2",
                      annotation_font_size=9, annotation_font_color=C['text3'])

        # Quadrant labels
        for label, x, y in [
            ("HIGH PERF", 120, 4.9), ("HIGH AHT", 400, 4.9),
            ("COACHING", 120, 3.2), ("CRITICAL", 400, 3.2)
        ]:
            fig.add_annotation(x=x, y=y, text=label,
                               font=dict(size=8, color=C['text3'],
                                         family='IBM Plex Mono'),
                               showarrow=False, opacity=0.4)

        apply_layout(fig, height=380,
                     xaxis=dict(title='Avg AHT (seconds)', gridcolor=C['border']),
                     yaxis=dict(title='Avg CSAT Score', gridcolor=C['border']))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col_r:
        section_title("TIER SUMMARY")
        for tier, color in tier_colors.items():
            t_agents = agent[agent['tier'] == tier]
            pct = len(t_agents) / len(agent) * 100
            bar_w = f"{pct:.0f}%"
            st.markdown(f"""
            <div style='background:{C["surface"]};border:1px solid {C["border"]};
            border-radius:5px;padding:0.9rem 1.1rem;margin-bottom:0.6rem;'>
              <div style='display:flex;justify-content:space-between;
              align-items:center;margin-bottom:0.5rem;'>
                <span style='font-family:"IBM Plex Mono",monospace;font-size:0.65rem;
                letter-spacing:0.12em;color:{color};'>{tier}</span>
                <span style='font-family:"IBM Plex Mono",monospace;font-size:0.75rem;
                color:{C["text"]};'>{len(t_agents)} agents</span>
              </div>
              <div style='background:{C["surface3"]};border-radius:2px;height:4px;'>
                <div style='background:{color};width:{bar_w};height:4px;
                border-radius:2px;'></div>
              </div>
              <div style='font-family:"IBM Plex Mono",monospace;font-size:0.58rem;
              color:{C["text3"]};margin-top:0.35rem;'>{pct:.0f}% of team</div>
            </div>""", unsafe_allow_html=True)

        section_title("COACHING FLAGS")
        risk = agent[agent['tier'] == 'RISK']
        if len(risk) == 0:
            alert_box("No agents in RISK tier.", level='success')
        else:
            for _, row in risk.iterrows():
                alert_box(
                    f"{row['agent_name']} — Score {row['score']:.3f} · "
                    f"CSAT {row['avg_csat']:.2f} · Abandon {row['avg_abandon']:.1%}",
                    level='warn'
                )

    # Full ranking table
    section_title("COMPLETE RANKING TABLE")
    display = agent.copy()
    cols = ['agent_name', 'experience_months', 'avg_csat', 'avg_aht',
            'avg_abandon', 'avg_fcr', 'total_calls', 'days_worked', 'tier', 'score']
    if 'team' in agent.columns:
        cols = ['agent_name', 'team', 'experience_months', 'avg_csat', 'avg_aht',
                'avg_abandon', 'avg_fcr', 'total_calls', 'days_worked', 'tier', 'score']
    display = display[cols].copy()
    display['avg_abandon'] = (display['avg_abandon'] * 100).round(2)
    display['avg_fcr'] = (display['avg_fcr'] * 100).round(2)
    display['avg_aht'] = display['avg_aht'].round(0).astype(int)
    display['avg_csat'] = display['avg_csat'].round(2)
    display['score'] = display['score'].round(3)

    col_names = {
        'agent_name': 'Agent', 'team': 'Team', 'experience_months': 'Exp (mo)',
        'avg_csat': 'CSAT', 'avg_aht': 'AHT (s)', 'avg_abandon': 'Abandon %',
        'avg_fcr': 'FCR %', 'total_calls': 'Calls', 'days_worked': 'Days',
        'tier': 'Tier', 'score': 'Score'
    }
    display.rename(columns=col_names, inplace=True)
    st.dataframe(display, use_container_width=True)

    # Experience vs FCR scatter
    section_title("EXPERIENCE vs FCR RATE")
    fig2 = go.Figure()
    team_col = agent.get('team') if 'team' in agent.columns else None
    if 'team' in agent.columns:
        for team in agent['team'].unique():
            t = agent[agent['team'] == team]
            fig2.add_trace(go.Scatter(
                x=t['experience_months'], y=t['avg_fcr'] * 100,
                mode='markers', name=team,
                marker=dict(color=TEAM_COLORS.get(team, C['accent']), size=9, opacity=0.85),
                hovertemplate='%{text}<br>Exp: %{x}mo · FCR: %{y:.1f}%<extra></extra>',
                text=t['agent_name']
            ))
    else:
        fig2.add_trace(go.Scatter(
            x=agent['experience_months'], y=agent['avg_fcr'] * 100,
            mode='markers', marker=dict(color=C['accent'], size=9),
            text=agent['agent_name'],
            hovertemplate='%{text}<br>Exp: %{x}mo · FCR: %{y:.1f}%<extra></extra>'
        ))
    apply_layout(fig2, height=260,
                 xaxis=dict(title='Experience (months)', gridcolor=C['border']),
                 yaxis=dict(title='FCR Rate (%)', gridcolor=C['border']))
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})


# ── PAGE: PREDICTOR ───────────────────────────────────────────────────────────

def page_predictor(df):
    page_header("ML ABANDON RATE PREDICTOR",
                "Random Forest Regressor · R² 0.919 · 5-fold CV · 12 features")

    rf, lr, metrics = load_model()

    if rf is None:
        alert_box("Model file not found. Make sure models/regressor.pkl is in the repo.", level='danger')
        st.info("The model was trained on the 2024 dataset. Re-train with train_model.py using the new 2025-2026 data for improved accuracy.")
        return

    # Model metrics
    section_title("MODEL PERFORMANCE")
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics_display = [
        (c1, "RF R²",         f"{metrics.get('rf_r2', 0):.3f}"),
        (c2, "RF RMSE",       f"{metrics.get('rf_rmse', 0):.4f}"),
        (c3, "LR R²",         f"{metrics.get('lr_r2', 0):.3f}"),
        (c4, "CV R² (5-fold)", f"{metrics.get('cv_mean', 0):.3f}"),
        (c5, "Baseline RMSE", f"{metrics.get('baseline_rmse', 0):.4f}"),
    ]
    for col, label, val in metrics_display:
        col.metric(label, val)

    improvement = metrics.get('baseline_rmse', 1) / max(metrics.get('rf_rmse', 0.01), 0.001)
    alert_box(f"Random Forest achieves {improvement:.1f}x RMSE improvement over naive baseline predictor.", level='info')

    # Feature importance chart
    section_title("FEATURE IMPORTANCE")
    fi = metrics.get('feature_importances', {})
    if fi:
        fi_df = pd.DataFrame({'Feature': list(fi.keys()), 'Importance': list(fi.values())})
        fi_df = fi_df.sort_values('Importance', ascending=True).tail(10)

        fig = go.Figure(go.Bar(
            x=fi_df['Importance'], y=fi_df['Feature'],
            orientation='h',
            marker=dict(
                color=fi_df['Importance'],
                colorscale=[[0, C['surface3']], [0.5, C['accent2']], [1, C['accent']]],
                showscale=False
            ),
            text=fi_df['Importance'].apply(lambda x: f"{x:.3f}"),
            textposition='outside',
            textfont=dict(size=10, family='IBM Plex Mono')
        ))
        apply_layout(fig, height=300,
                     xaxis=dict(title='Importance Score', gridcolor=C['border']),
                     yaxis=dict(gridcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("---")
    section_title("SCENARIO FORECAST SIMULATOR")

    # Inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div style='font-family:IBM Plex Mono;font-size:0.65rem;color:{C['text3']};letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem;'>TEMPORAL</div>", unsafe_allow_html=True)
        dow = st.selectbox("Day of week", [0, 1, 2, 3, 4],
                           format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][x])
        shift = st.selectbox("Shift", ['morning', 'afternoon', 'night'])
    with c2:
        st.markdown(f"<div style='font-family:IBM Plex Mono;font-size:0.65rem;color:{C['text3']};letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem;'>OPERATIONAL</div>", unsafe_allow_html=True)
        queue = st.slider("Queue depth", 0, 70, 18)
        aht = st.slider("Expected AHT (sec)", 120, 500, 280)
    with c3:
        st.markdown(f"<div style='font-family:IBM Plex Mono;font-size:0.65rem;color:{C['text3']};letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem;'>TEAM</div>", unsafe_allow_html=True)
        exp = st.slider("Avg experience (months)", 1, 48, 18)
        calls = st.slider("Expected calls handled", 10, 90, 45)

    # Prediction
    le = LabelEncoder()
    le.fit(['afternoon', 'morning', 'night'])
    inp = pd.DataFrame([{
        'day_of_week': dow, 'is_monday': int(dow == 0),
        'is_night_shift': int(shift == 'night'), 'shift': shift,
        'calls_in_queue': queue, 'aht_seconds': aht,
        'experience_months': exp, 'calls_handled': calls,
        'csat_score': df['csat_score'].mean(),
        'fcr_rate': df['fcr_rate'].mean()
    }])
    X = engineer_features(inp)
    rfp = float(np.clip(rf.predict(X)[0], 0, 1))
    lrp = float(np.clip(lr.predict(X)[0], 0, 1))

    st.markdown("<br>", unsafe_allow_html=True)

    # Forecast cards
    fc1, fc2, fc3 = st.columns(3)
    for col, label, val, note in [
        (fc1, "RANDOM FOREST", rfp, "PRIMARY MODEL"),
        (fc2, "LINEAR REG.", lrp, "BASELINE"),
        (fc3, "ENSEMBLE AVG", (rfp + lrp) / 2, "RF + LR MEAN"),
    ]:
        over = val > 0.05
        critical = val > 0.10
        color = C['danger'] if critical else (C['warn'] if over else C['success'])
        status = "CRITICAL" if critical else ("ABOVE TARGET" if over else "WITHIN TARGET")
        col.markdown(f"""
        <div style='background:{C["surface"]};border:1px solid {C["border"]};
        border-top:3px solid {color};border-radius:6px;padding:1.5rem;text-align:center;'>
          <div style='font-family:"IBM Plex Mono",monospace;font-size:0.58rem;
          letter-spacing:0.14em;color:{C["text3"]};text-transform:uppercase;
          margin-bottom:0.6rem;'>{label}</div>
          <div style='font-family:"IBM Plex Mono",monospace;font-size:2.8rem;
          font-weight:600;color:{color};line-height:1;'>{val*100:.1f}%</div>
          <div style='font-family:"IBM Plex Mono",monospace;font-size:0.65rem;
          color:{color};letter-spacing:0.1em;margin-top:0.5rem;'>{status}</div>
          <div style='font-size:0.62rem;color:{C["text3"]};margin-top:0.3rem;
          font-family:"IBM Plex Mono",monospace;'>TARGET &lt; 5.0% · {note}</div>
        </div>""", unsafe_allow_html=True)

    # Staffing recommendation
    st.markdown("<br>", unsafe_allow_html=True)
    day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][dow]

    if rfp > 0.10:
        extra_agents = int(np.ceil((rfp - 0.05) * 40))
        alert_box(
            f"CRITICAL: {rfp*100:.1f}% forecast on {day_name} {shift.upper()}. "
            f"Recommend adding ~{extra_agents} agents to reduce queue from {queue} to target levels.",
            level='danger'
        )
    elif rfp > 0.05:
        extra_agents = max(1, int(np.ceil((rfp - 0.05) * 25)))
        alert_box(
            f"Forecast {rfp*100:.1f}% exceeds 5% target on {day_name} {shift.upper()}. "
            f"Consider adding {extra_agents} agent(s) or redistributing {queue} queued calls.",
            level='warn'
        )
    else:
        alert_box(
            f"Forecast {rfp*100:.1f}% within target on {day_name} {shift.upper()}. "
            f"Current configuration with queue depth {queue} is sufficient.",
            level='success'
        )

    # Sensitivity: queue depth effect
    section_title("QUEUE DEPTH SENSITIVITY")
    queue_range = np.arange(0, 65, 5)
    rf_preds = []
    for q in queue_range:
        inp_q = inp.copy()
        inp_q['calls_in_queue'] = q
        inp_q['queue_x_monday'] = q * int(dow == 0)
        inp_q['queue_x_night'] = q * int(shift == 'night')
        Xq = engineer_features(inp_q)
        rf_preds.append(float(np.clip(rf.predict(Xq)[0], 0, 1)) * 100)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=queue_range, y=rf_preds,
        mode='lines+markers',
        line=dict(color=C['accent2'], width=2.5),
        marker=dict(size=6),
        fill='tozeroy', fillcolor='rgba(14,165,233,0.08)',
        name='Predicted Abandon %',
        hovertemplate='Queue: %{x}<br>Abandon: %{y:.1f}%<extra></extra>'
    ))
    fig2.add_hline(y=5.0, line_dash="dash", line_color=C['danger'], opacity=0.6,
                   annotation_text="5% target", annotation_font_size=9)
    fig2.add_vline(x=queue, line_dash="dot", line_color=C['warn'], opacity=0.7,
                   annotation_text=f"Current: {queue}", annotation_font_size=9,
                   annotation_font_color=C['warn'])
    apply_layout(fig2, height=250,
                 xaxis=dict(title='Queue Depth', gridcolor=C['border']),
                 yaxis=dict(title='Predicted Abandon Rate (%)', gridcolor=C['border']))
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    df, source = load_data()

    page, date_range, shifts, teams, agents, years = render_sidebar(df)
    df_f = apply_filters(df, date_range, shifts, teams, agents, years)

    if len(df_f) == 0:
        alert_box("No data matches current filters. Try expanding the date range or selections.", level='warn')
        return

    # Data source badge in sidebar
    badge_color = C['accent'] if source == 'supabase' else C['warn']
    st.sidebar.markdown(f"""
    <div style='margin-top:0.8rem;font-family:"IBM Plex Mono",monospace;font-size:0.6rem;
    color:{badge_color};letter-spacing:0.08em;'>
      {'[LIVE] SUPABASE' if source == 'supabase' else '[LOCAL] CSV FALLBACK'}
    </div>""", unsafe_allow_html=True)

    page_key = page.strip()

    if   page_key == "OVERVIEW":  page_overview(df_f)
    elif page_key == "TRENDS":    page_trends(df_f)
    elif page_key == "TEAMS":     page_teams(df_f)
    elif page_key == "AGENTS":    page_agents(df_f)
    elif page_key == "PREDICTOR": page_predictor(df_f)


if __name__ == '__main__':
    main()
