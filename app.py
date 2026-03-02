"""
app.py — Call Center Performance Analytics Dashboard
4-page Streamlit application.

Pages:
  1. Executive Overview   — KPI cards with delta vs previous period
  2. Trends              — time series with rolling averages + anomaly flags
  3. Agent Performance   — ranked table with coaching flags
  4. Prediction          — ML forecast for tomorrow's abandon rate

Run locally:   streamlit run app.py
"""

import os
import json
import pickle

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Call Center Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    'primary':   '#1E3A5F',
    'accent':    '#2ECC71',
    'warning':   '#E74C3C',
    'neutral':   '#95A5A6',
    'light_bg':  '#F8F9FA',
    'green':     '#27AE60',
    'red':       '#C0392B',
    'blue':      '#2980B9',
    'orange':    '#E67E22',
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700;
        color: #1E3A5F; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem; color: #7F8C8D; margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: white; border-radius: 12px;
        padding: 1.2rem 1.5rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #1E3A5F;
    }
    .kpi-label  { font-size: 0.8rem; color: #7F8C8D; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value  { font-size: 2rem; font-weight: 700; color: #1E3A5F; }
    .kpi-delta-pos { font-size: 0.85rem; color: #27AE60; }
    .kpi-delta-neg { font-size: 0.85rem; color: #C0392B; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #2C3E50;
        border-bottom: 2px solid #1E3A5F; padding-bottom: 0.3rem;
        margin: 1.5rem 0 1rem 0;
    }
    .insight-box {
        background: #EBF5FB; border-left: 4px solid #2980B9;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        font-size: 0.9rem; color: #1A5276;
    }
    .warning-box {
        background: #FDEDEC; border-left: 4px solid #E74C3C;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        font-size: 0.9rem; color: #922B21;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, 'callcenter_data.csv')
    df = pd.read_csv(path, parse_dates=['date'])
    return df


@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, 'models', 'regressor.pkl')
    metrics_path = os.path.join(base, 'models', 'reg_metrics.json')
    if not os.path.exists(path):
        return None, None, None
    with open(path, 'rb') as f:
        models = pickle.load(f)
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return models['rf'], models['lr'], metrics


# ── Feature engineering (must match train_model.py) ──────────────────────────
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
    f['queue_x_monday']   = df['calls_in_queue'] * df['is_monday']
    f['queue_x_night']    = df['calls_in_queue'] * df['is_night_shift']
    return f


# ── KPI helpers ───────────────────────────────────────────────────────────────
def get_kpis(df):
    mid = df['date'].min() + (df['date'].max() - df['date'].min()) / 2
    curr = df[df['date'] > mid]
    prev = df[df['date'] <= mid]
    def avg(d, c): return d[c].mean() if len(d) > 0 else 0
    def delta(c, p): return (c - p) / p if p != 0 else 0
    metrics = {}
    for col in ['aht_seconds', 'csat_score', 'abandon_rate', 'fcr_rate']:
        c = avg(curr, col); p = avg(prev, col)
        metrics[col] = {'value': c, 'delta': delta(c, p)}
    return metrics


def render_kpi_card(label, value, delta, format_str, target=None, invert_delta=False):
    delta_pct = delta * 100
    is_good   = (delta_pct < 0) if invert_delta else (delta_pct > 0)
    arrow     = "▲" if delta_pct > 0 else "▼"
    color_cls = "kpi-delta-pos" if is_good else "kpi-delta-neg"
    target_html = f"<div style='font-size:0.75rem;color:#95A5A6;'>Target: {target}</div>" if target else ""
    return f"""
    <div class='kpi-card'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value'>{format_str.format(value)}</div>
        <div class='{color_cls}'>{arrow} {abs(delta_pct):.1f}% vs prev period</div>
        {target_html}
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(df):
    st.sidebar.markdown("### 🎛️ Filters")
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    shifts = st.sidebar.multiselect(
        "Shifts", options=['morning', 'afternoon', 'night'],
        default=['morning', 'afternoon', 'night']
    )

    agents = st.sidebar.multiselect(
        "Agents (all if empty)",
        options=sorted(df['agent_name'].unique()),
        default=[]
    )

    return date_range, shifts, agents


def apply_filters(df, date_range, shifts, agents):
    if len(date_range) == 2:
        df = df[(df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])]
    if shifts:
        df = df[df['shift'].isin(shifts)]
    if agents:
        df = df[df['agent_name'].isin(agents)]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_overview(df):
    st.markdown("<div class='main-title'>📊 Executive Overview</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-title'>{len(df):,} records · {df['agent_id'].nunique()} agents · {df['date'].min().date()} → {df['date'].max().date()}</div>", unsafe_allow_html=True)

    kpis = get_kpis(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_kpi_card(
            "Avg Handle Time",
            kpis['aht_seconds']['value'],
            kpis['aht_seconds']['delta'],
            "{:.0f} sec",
            target="< 300 sec",
            invert_delta=True
        ), unsafe_allow_html=True)

    with c2:
        st.markdown(render_kpi_card(
            "CSAT Score",
            kpis['csat_score']['value'],
            kpis['csat_score']['delta'],
            "{:.2f}",
            target="> 4.2"
        ), unsafe_allow_html=True)

    with c3:
        st.markdown(render_kpi_card(
            "Abandon Rate",
            kpis['abandon_rate']['value'] * 100,
            kpis['abandon_rate']['delta'],
            "{:.1f}%",
            target="< 5%",
            invert_delta=True
        ), unsafe_allow_html=True)

    with c4:
        st.markdown(render_kpi_card(
            "First Call Resolution",
            kpis['fcr_rate']['value'] * 100,
            kpis['fcr_rate']['delta'],
            "{:.1f}%",
            target="> 70%"
        ), unsafe_allow_html=True)

    st.markdown("---")

    # Shift comparison
    st.markdown("<div class='section-header'>📋 Performance by Shift</div>", unsafe_allow_html=True)
    shift_df = df.groupby('shift').agg(
        AHT       = ('aht_seconds',  'mean'),
        CSAT      = ('csat_score',   'mean'),
        Abandon   = ('abandon_rate', 'mean'),
        FCR       = ('fcr_rate',     'mean'),
        Calls     = ('calls_handled', 'sum'),
    ).round(3).reset_index()
    shift_df['Abandon'] = (shift_df['Abandon'] * 100).round(2)
    shift_df['FCR']     = (shift_df['FCR']     * 100).round(2)
    shift_df['AHT']     = shift_df['AHT'].round(0).astype(int)
    st.dataframe(shift_df, use_container_width=True, hide_index=True)

    # Quick insight
    worst_shift = shift_df.loc[shift_df['Abandon'].idxmax(), 'shift']
    st.markdown(f"<div class='warning-box'>⚠️ <strong>{worst_shift.capitalize()} shift</strong> has the highest abandon rate ({shift_df['Abandon'].max():.1f}%). Consider reviewing staffing levels for this shift.</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRENDS
# ══════════════════════════════════════════════════════════════════════════════
def page_trends(df):
    st.markdown("<div class='main-title'>📈 KPI Trends</div>", unsafe_allow_html=True)

    window = st.slider("Rolling average window (days)", 3, 14, 7)

    daily = df.groupby('date').agg(
        aht_seconds  = ('aht_seconds',  'mean'),
        csat_score   = ('csat_score',   'mean'),
        abandon_rate = ('abandon_rate', 'mean'),
        fcr_rate     = ('fcr_rate',     'mean'),
    ).reset_index()

    for col in ['aht_seconds', 'csat_score', 'abandon_rate', 'fcr_rate']:
        daily[f'{col}_roll'] = daily[col].rolling(window, min_periods=1).mean()

    tab1, tab2, tab3, tab4 = st.tabs(["AHT", "CSAT", "Abandon Rate", "FCR"])

    def trend_chart(col, label, target, target_label, color, invert_alarm=False):
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.fill_between(daily['date'], daily[col], alpha=0.15, color=color)
        ax.plot(daily['date'], daily[col], alpha=0.4, linewidth=0.8, color=color)
        ax.plot(daily['date'], daily[f'{col}_roll'], linewidth=2.2, color=color, label=f'{window}-day avg')
        ax.axhline(target, color='#E74C3C', linestyle='--', linewidth=1.2, label=f'Target: {target_label}')
        ax.set_xlabel(''); ax.set_ylabel(label)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab1:
        trend_chart('aht_seconds', 'Seconds', 300, '300 sec', COLORS['primary'])
    with tab2:
        trend_chart('csat_score', 'Score', 4.2, '4.2', COLORS['accent'])
    with tab3:
        trend_chart('abandon_rate', 'Rate', 0.05, '5%', COLORS['warning'])
    with tab4:
        trend_chart('fcr_rate', 'Rate', 0.70, '70%', COLORS['blue'])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AGENT PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
def page_agents(df):
    st.markdown("<div class='main-title'>👥 Agent Performance</div>", unsafe_allow_html=True)

    agent = df.groupby(['agent_id', 'agent_name', 'experience_months']).agg(
        avg_aht     = ('aht_seconds',  'mean'),
        avg_csat    = ('csat_score',   'mean'),
        avg_abandon = ('abandon_rate', 'mean'),
        avg_fcr     = ('fcr_rate',     'mean'),
        total_calls = ('calls_handled', 'sum'),
        days_worked = ('date',         'nunique'),
    ).reset_index()

    # Composite score
    def norm(s, inv=False):
        mn, mx = s.min(), s.max()
        if mx == mn: return pd.Series([0.5]*len(s), index=s.index)
        n = (s - mn) / (mx - mn)
        return 1 - n if inv else n

    agent['score'] = (
        norm(agent['avg_csat'])          * 0.30 +
        norm(agent['avg_fcr'])           * 0.30 +
        norm(agent['avg_aht'], inv=True) * 0.20 +
        norm(agent['avg_abandon'], inv=True) * 0.20
    )
    q75 = agent['score'].quantile(0.75)
    q25 = agent['score'].quantile(0.25)
    agent['Tier'] = agent['score'].apply(
        lambda s: '🟢 Top' if s >= q75 else ('🔴 Risk' if s <= q25 else '⚪ Mid')
    )
    agent = agent.sort_values('score', ascending=False).reset_index(drop=True)
    agent.index += 1

    # Display table
    display = agent[['agent_name', 'experience_months', 'avg_csat', 'avg_aht',
                      'avg_abandon', 'avg_fcr', 'total_calls', 'Tier', 'score']].copy()
    display.columns = ['Agent', 'Exp (mo)', 'CSAT', 'AHT (sec)', 'Abandon %', 'FCR %', 'Total Calls', 'Tier', 'Score']
    display['Abandon %'] = (display['Abandon %'] * 100).round(2)
    display['FCR %']     = (display['FCR %']     * 100).round(2)
    display['AHT (sec)'] = display['AHT (sec)'].round(0).astype(int)
    display['CSAT']      = display['CSAT'].round(2)
    display['Score']     = display['Score'].round(3)

    st.dataframe(display, use_container_width=True)

    # Coaching flags
    risk_agents = agent[agent['Tier'] == '🔴 Risk']
    if len(risk_agents) > 0:
        names = ', '.join(risk_agents['agent_name'].tolist())
        st.markdown(f"<div class='warning-box'>⚠️ <strong>Coaching recommended:</strong> {names} — bottom quartile composite performance.</div>", unsafe_allow_html=True)

    # CSAT vs AHT scatter
    st.markdown("<div class='section-header'>CSAT vs AHT by Agent</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors_map = {'🟢 Top': COLORS['green'], '🔴 Risk': COLORS['red'], '⚪ Mid': COLORS['neutral']}
    for _, row in agent.iterrows():
        ax.scatter(row['avg_aht'], row['avg_csat'],
                   color=colors_map[row['Tier']], s=80, zorder=3)
        ax.annotate(row['agent_name'].split()[0], (row['avg_aht'], row['avg_csat']),
                    fontsize=7, ha='left', va='bottom', xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('Avg AHT (seconds)'); ax.set_ylabel('Avg CSAT')
    ax.grid(alpha=0.3)
    patches = [mpatches.Patch(color=v, label=k) for k, v in colors_map.items()]
    ax.legend(handles=patches, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def page_prediction(df):
    st.markdown("<div class='main-title'>🤖 Abandon Rate Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Forecast tomorrow's abandon rate from operational inputs</div>", unsafe_allow_html=True)

    rf, lr, metrics = load_model()

    if rf is None:
        st.error("Model not trained. Run: `python train_model.py`")
        return

    # ── Model metrics ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📊 Model Performance</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RF R²",       f"{metrics.get('rf_r2', 0):.3f}")
    c2.metric("RF RMSE",     f"{metrics.get('rf_rmse', 0):.4f}")
    c3.metric("Baseline RMSE", f"{metrics.get('baseline_rmse', 0):.4f}")
    c4.metric("CV R² (5-fold)", f"{metrics.get('cv_mean', 0):.3f} ± {metrics.get('cv_std', 0):.3f}")

    # ── Feature importance ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔍 Feature Importance</div>", unsafe_allow_html=True)
    fi = metrics.get('feature_importances', {})
    fi_df = pd.DataFrame({'Feature': list(fi.keys()), 'Importance': list(fi.values())})
    fi_df = fi_df.sort_values('Importance', ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(fi_df['Feature'], fi_df['Importance'], color=COLORS['primary'], alpha=0.8)
    ax.set_xlabel('Importance'); ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Prediction form ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔮 Tomorrow's Forecast</div>", unsafe_allow_html=True)
    st.markdown("Adjust the inputs to simulate different scenarios:")

    c1, c2, c3 = st.columns(3)
    with c1:
        day_of_week     = st.selectbox("Day of week", [0,1,2,3,4], format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri'][x])
        shift           = st.selectbox("Dominant shift", ['morning','afternoon','night'])
    with c2:
        calls_in_queue  = st.slider("Expected queue depth", 0, 60, 18)
        aht_seconds     = st.slider("Expected AHT (sec)", 120, 500, 280)
    with c3:
        experience_avg  = st.slider("Avg team experience (months)", 1, 36, 12)
        calls_handled   = st.slider("Expected calls handled", 10, 80, 45)

    is_monday      = int(day_of_week == 0)
    is_night_shift = int(shift == 'night')
    le = LabelEncoder(); le.fit(['afternoon','morning','night'])

    input_df = pd.DataFrame([{
        'day_of_week':       day_of_week,
        'is_monday':         is_monday,
        'is_night_shift':    is_night_shift,
        'shift':             shift,
        'calls_in_queue':    calls_in_queue,
        'aht_seconds':       aht_seconds,
        'experience_months': experience_avg,
        'calls_handled':     calls_handled,
        'csat_score':        df['csat_score'].mean(),
        'fcr_rate':          df['fcr_rate'].mean(),
    }])
    X_pred = engineer_features(input_df)

    rf_pred = float(rf.predict(X_pred)[0])
    lr_pred = float(lr.predict(X_pred)[0])
    rf_pred = max(0, min(rf_pred, 1))
    lr_pred = max(0, min(lr_pred, 1))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        color = "#C0392B" if rf_pred > 0.05 else "#27AE60"
        st.markdown(f"""
        <div style='background:white;border-radius:12px;padding:1.5rem;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.1);border-top:4px solid {color}'>
            <div style='font-size:0.85rem;color:#7F8C8D;text-transform:uppercase;'>Random Forest Forecast</div>
            <div style='font-size:3rem;font-weight:700;color:{color}'>{rf_pred*100:.1f}%</div>
            <div style='font-size:0.85rem;color:#7F8C8D;'>Target: &lt; 5.0%</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        color2 = "#C0392B" if lr_pred > 0.05 else "#27AE60"
        st.markdown(f"""
        <div style='background:white;border-radius:12px;padding:1.5rem;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.1);border-top:4px solid {color2}'>
            <div style='font-size:0.85rem;color:#7F8C8D;text-transform:uppercase;'>Linear Regression Forecast</div>
            <div style='font-size:3rem;font-weight:700;color:{color2}'>{lr_pred*100:.1f}%</div>
            <div style='font-size:0.85rem;color:#7F8C8D;'>Linear baseline</div>
        </div>""", unsafe_allow_html=True)

    if rf_pred > 0.05:
        st.markdown(f"<div class='warning-box' style='margin-top:1rem'>⚠️ Forecast exceeds 5% target. Consider increasing staffing or redistributing queue load for the {shift} shift.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='insight-box' style='margin-top:1rem'>✅ Forecast is within target range. Current staffing configuration looks sufficient.</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    df = load_data()

    date_range, shifts, agents = render_sidebar(df)
    df_filtered = apply_filters(df, date_range, shifts, agents)

    if len(df_filtered) == 0:
        st.warning("No data for selected filters.")
        return

    page = st.sidebar.radio(
        "Navigate",
        ["📊 Executive Overview", "📈 Trends", "👥 Agent Performance", "🤖 Prediction"],
        label_visibility="collapsed"
    )

    if page == "📊 Executive Overview":
        page_overview(df_filtered)
    elif page == "📈 Trends":
        page_trends(df_filtered)
    elif page == "👥 Agent Performance":
        page_agents(df_filtered)
    elif page == "🤖 Prediction":
        page_prediction(df_filtered)


if __name__ == '__main__':
    main()
