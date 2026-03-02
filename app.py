"""
app.py — Call Center Performance Analytics
Mission Control aesthetic — dark, monospace, zero decoration
"""

import os
import json
import pickle

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="CC Analytics — Palencia Research",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

C = {
    'bg':      '#080D14', 'surface':  '#0D1520', 'surface2': '#111C2A',
    'border':  '#1A2C42', 'border2':  '#243850', 'accent':   '#00C9A7',
    'accent2': '#0096FF', 'warn':     '#FF4C4C', 'text':     '#E8EDF2',
    'text2':   '#8899AA', 'text3':    '#4A6070', 'green':    '#00C9A7',
    'red':     '#FF4C4C', 'mid':      '#4A6070',
}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,600;1,300&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
  html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; background-color: {C['bg']}; color: {C['text']}; }}
  .stApp {{ background: {C['bg']}; }}
  section[data-testid="stSidebar"] {{ background: {C['surface']}; border-right: 1px solid {C['border']}; }}
  section[data-testid="stSidebar"] * {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; }}
  .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1100px; }}
  .stSelectbox > div > div, .stMultiSelect > div > div {{ background: {C['surface2']} !important; border: 1px solid {C['border2']} !important; color: {C['text']} !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.8rem !important; }}
  .stSlider > div {{ font-family: 'IBM Plex Mono', monospace !important; }}
  .stTabs [data-baseweb="tab-list"] {{ background: {C['surface']}; border-bottom: 1px solid {C['border']}; gap: 0; }}
  .stTabs [data-baseweb="tab"] {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; letter-spacing: 0.08em; color: {C['text2']}; padding: 0.6rem 1.2rem; border: none; background: transparent; }}
  .stTabs [aria-selected="true"] {{ color: {C['accent']}; border-bottom: 2px solid {C['accent']}; background: transparent; }}
  .stDataFrame {{ border: 1px solid {C['border']} !important; }}
  .stDataFrame th {{ background: {C['surface2']} !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.72rem !important; color: {C['text2']} !important; letter-spacing: 0.05em; }}
  .stDataFrame td {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; }}
  [data-testid="metric-container"] {{ background: {C['surface']}; border: 1px solid {C['border']}; border-radius: 4px; padding: 1rem !important; }}
  [data-testid="metric-container"] label {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 0.7rem !important; letter-spacing: 0.08em; color: {C['text2']} !important; }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 1.6rem !important; color: {C['text']} !important; }}
  hr {{ border-color: {C['border']} !important; margin: 1.5rem 0; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


def page_header(title, subtitle=None):
    sub = f'<div style="font-size:0.85rem;color:{C["text2"]};margin-top:0.3rem;">{subtitle}</div>' if subtitle else ''
    st.markdown(f"""
    <div style='border-bottom:1px solid {C["border"]};padding-bottom:1rem;margin-bottom:1.5rem;'>
      <div style='font-family:"IBM Plex Mono",monospace;font-size:0.62rem;letter-spacing:0.18em;color:{C["accent"]};text-transform:uppercase;margin-bottom:0.3rem;'>ANALYTICS SYSTEM / CC-PERF-v2</div>
      <div style='font-family:"IBM Plex Mono",monospace;font-size:1.5rem;font-weight:600;color:{C["text"]};letter-spacing:-0.01em;'>{title}</div>{sub}
    </div>""", unsafe_allow_html=True)

def section_title(text):
    st.markdown(f"""<div style='font-family:"IBM Plex Mono",monospace;font-size:0.7rem;letter-spacing:0.12em;color:{C["text2"]};text-transform:uppercase;border-left:2px solid {C["accent"]};padding-left:0.7rem;margin:1.5rem 0 0.8rem;'>{text}</div>""", unsafe_allow_html=True)

def kpi_card(col, label, value, delta_pct, target=None, invert=False):
    is_good = (delta_pct < 0) if invert else (delta_pct > 0)
    arrow   = "+" if delta_pct > 0 else ""
    color   = C['green'] if is_good else C['red']
    tgt     = f'<div style="font-size:0.65rem;color:{C["text3"]};margin-top:0.2rem;">target {target}</div>' if target else ''
    col.markdown(f"""
    <div style='background:{C["surface"]};border:1px solid {C["border"]};border-top:2px solid {C["accent"]};border-radius:3px;padding:1.1rem 1.2rem;'>
      <div style='font-family:"IBM Plex Mono",monospace;font-size:0.62rem;letter-spacing:0.12em;color:{C["text3"]};text-transform:uppercase;margin-bottom:0.5rem;'>{label}</div>
      <div style='font-family:"IBM Plex Mono",monospace;font-size:2rem;font-weight:600;color:{C["text"]};line-height:1;'>{value}</div>
      <div style='font-family:"IBM Plex Mono",monospace;font-size:0.72rem;color:{color};margin-top:0.4rem;'>{arrow}{abs(delta_pct):.1f}% vs prior</div>{tgt}
    </div>""", unsafe_allow_html=True)

def alert_bar(text, level='warn'):
    color  = C['warn'] if level == 'warn' else C['accent']
    bg     = 'rgba(255,76,76,0.06)' if level == 'warn' else 'rgba(0,201,167,0.06)'
    prefix = '[ALERT]' if level == 'warn' else '[INFO]'
    st.markdown(f"""
    <div style='background:{bg};border-left:3px solid {color};padding:0.7rem 1rem;margin:0.8rem 0;border-radius:0 3px 3px 0;'>
      <span style='font-family:"IBM Plex Mono",monospace;font-size:0.75rem;color:{color};letter-spacing:0.06em;'>{prefix}</span>
      <span style='font-family:"IBM Plex Mono",monospace;font-size:0.78rem;color:{C["text2"]};margin-left:0.5rem;'>{text}</span>
    </div>""", unsafe_allow_html=True)

def set_plot_style():
    plt.rcParams.update({
        'figure.facecolor': C['bg'], 'axes.facecolor': C['surface'],
        'axes.edgecolor': C['border'], 'axes.labelcolor': C['text2'],
        'axes.titlecolor': C['text'], 'xtick.color': C['text3'],
        'ytick.color': C['text3'], 'text.color': C['text'],
        'grid.color': C['border'], 'grid.linewidth': 0.5,
        'font.family': 'monospace', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

set_plot_style()

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, 'callcenter_data.csv'), parse_dates=['date'])
    return df

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    mp   = os.path.join(base, 'models', 'regressor.pkl')
    mep  = os.path.join(base, 'models', 'reg_metrics.json')
    if not os.path.exists(mp): return None, None, None
    with open(mp, 'rb') as f: models = pickle.load(f)
    metrics = json.load(open(mep)) if os.path.exists(mep) else {}
    return models['rf'], models['lr'], metrics

def engineer_features(df):
    f  = pd.DataFrame(index=df.index)
    le = LabelEncoder(); le.fit(['afternoon', 'morning', 'night'])
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

def get_kpis(df):
    mid  = df['date'].min() + (df['date'].max() - df['date'].min()) / 2
    curr = df[df['date'] > mid]; prev = df[df['date'] <= mid]
    def avg(d, c): return d[c].mean() if len(d) > 0 else 0
    def delt(c, p): return (c - p) / p if p != 0 else 0
    out = {}
    for col in ['aht_seconds', 'csat_score', 'abandon_rate', 'fcr_rate']:
        c = avg(curr, col); p = avg(prev, col)
        out[col] = {'value': c, 'delta': delt(c, p)}
    return out

def render_sidebar(df):
    st.sidebar.markdown(f"""
    <div style='padding:0.5rem 0 1rem;border-bottom:1px solid {C["border"]};margin-bottom:1rem;'>
      <div style='font-size:0.62rem;letter-spacing:0.18em;color:{C["accent"]};text-transform:uppercase;'>CC-ANALYTICS</div>
      <div style='font-size:0.85rem;font-weight:600;color:{C["text"]};margin-top:0.2rem;'>PALENCIA RESEARCH</div>
    </div>""", unsafe_allow_html=True)

    st.sidebar.markdown(f"<div style='font-size:0.62rem;letter-spacing:0.1em;color:{C['text3']};text-transform:uppercase;margin-bottom:0.4rem;'>NAVIGATION</div>", unsafe_allow_html=True)
    page = st.sidebar.radio("nav", label_visibility="collapsed",
                             options=["OVERVIEW", "TRENDS", "AGENTS", "PREDICTOR"])
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<div style='font-size:0.62rem;letter-spacing:0.1em;color:{C['text3']};text-transform:uppercase;margin-bottom:0.4rem;'>FILTERS</div>", unsafe_allow_html=True)
    min_date = df['date'].min().date(); max_date = df['date'].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    shifts = st.sidebar.multiselect("Shift", options=['morning', 'afternoon', 'night'], default=['morning', 'afternoon', 'night'])
    agents = st.sidebar.multiselect("Agents (all if empty)", options=sorted(df['agent_name'].unique()), default=[])
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""<div style='font-size:0.62rem;color:{C['text3']};letter-spacing:0.06em;line-height:1.8;'>DATASET 2,640 rows<br>AGENTS 20<br>PERIOD Jul–Dec 2024<br>MODEL RF R² 0.919</div>""", unsafe_allow_html=True)
    return page, date_range, shifts, agents

def apply_filters(df, date_range, shifts, agents):
    if len(date_range) == 2:
        df = df[(df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])]
    if shifts: df = df[df['shift'].isin(shifts)]
    if agents: df = df[df['agent_name'].isin(agents)]
    return df

def page_overview(df):
    page_header("EXECUTIVE OVERVIEW", f"{len(df):,} records  /  {df['agent_id'].nunique()} agents  /  {df['date'].min().date()} to {df['date'].max().date()}")
    kpis = get_kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "AVG HANDLE TIME",  f"{kpis['aht_seconds']['value']:.0f}s",  kpis['aht_seconds']['delta']*100,  "< 300s",  invert=True)
    kpi_card(c2, "CSAT SCORE",       f"{kpis['csat_score']['value']:.2f}",     kpis['csat_score']['delta']*100,   "> 4.20")
    kpi_card(c3, "ABANDON RATE",     f"{kpis['abandon_rate']['value']*100:.1f}%", kpis['abandon_rate']['delta']*100, "< 5.0%", invert=True)
    kpi_card(c4, "FIRST CALL RES.",  f"{kpis['fcr_rate']['value']*100:.1f}%", kpis['fcr_rate']['delta']*100,     "> 70%")
    st.markdown("---")
    section_title("PERFORMANCE BY SHIFT")
    shift_df = df.groupby('shift').agg(AHT=('aht_seconds','mean'), CSAT=('csat_score','mean'), Abandon=('abandon_rate','mean'), FCR=('fcr_rate','mean'), Calls=('calls_handled','sum')).reset_index()
    shift_df['Abandon'] = (shift_df['Abandon'] * 100).round(2)
    shift_df['FCR']     = (shift_df['FCR']     * 100).round(2)
    shift_df['AHT']     = shift_df['AHT'].round(0).astype(int)
    shift_df['CSAT']    = shift_df['CSAT'].round(2)
    st.dataframe(shift_df, use_container_width=True, hide_index=True)
    worst = shift_df.loc[shift_df['Abandon'].idxmax(), 'shift'].upper()
    alert_bar(f"{worst} shift — abandon rate {shift_df['Abandon'].max():.1f}% exceeds target threshold. Review staffing allocation.")

def page_trends(df):
    page_header("TEMPORAL ANALYSIS", "Daily KPI trends with rolling average overlay")
    window = st.slider("Rolling window (days)", 3, 14, 7)
    daily = df.groupby('date').agg(aht_seconds=('aht_seconds','mean'), csat_score=('csat_score','mean'), abandon_rate=('abandon_rate','mean'), fcr_rate=('fcr_rate','mean')).reset_index()
    for col in ['aht_seconds','csat_score','abandon_rate','fcr_rate']:
        daily[f'{col}_roll'] = daily[col].rolling(window, min_periods=1).mean()
    tab1, tab2, tab3, tab4 = st.tabs(["AHT", "CSAT", "ABANDON RATE", "FCR"])
    def trend_chart(col, label, target, target_label, color):
        fig, ax = plt.subplots(figsize=(12, 3.2))
        ax.fill_between(daily['date'], daily[col], alpha=0.06, color=color)
        ax.plot(daily['date'], daily[col], alpha=0.2, linewidth=0.8, color=color)
        ax.plot(daily['date'], daily[f'{col}_roll'], linewidth=1.8, color=color, label=f"{window}d avg")
        ax.axhline(target, color=C['warn'], linestyle='--', linewidth=0.9, alpha=0.7, label=f"target {target_label}")
        ax.set_xlabel(''); ax.set_ylabel(label, fontsize=8)
        ax.legend(fontsize=8, framealpha=0); ax.grid(axis='y', alpha=0.4)
        ax.set_facecolor(C['surface']); fig.patch.set_facecolor(C['bg'])
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with tab1: trend_chart('aht_seconds',  'seconds', 300,  '300s',  C['accent2'])
    with tab2: trend_chart('csat_score',   'score',   4.2,  '4.2',   C['accent'])
    with tab3: trend_chart('abandon_rate', 'rate',    0.05, '5.0%',  C['warn'])
    with tab4: trend_chart('fcr_rate',     'rate',    0.70, '70%',   C['accent'])

def page_agents(df):
    page_header("AGENT PERFORMANCE", "Composite ranking — CSAT 30% / FCR 30% / AHT 20% / Abandon 20%")
    agent = df.groupby(['agent_id','agent_name','experience_months']).agg(avg_aht=('aht_seconds','mean'), avg_csat=('csat_score','mean'), avg_abandon=('abandon_rate','mean'), avg_fcr=('fcr_rate','mean'), total_calls=('calls_handled','sum'), days_worked=('date','nunique')).reset_index()
    def norm(s, inv=False):
        mn, mx = s.min(), s.max()
        if mx == mn: return pd.Series([0.5]*len(s), index=s.index)
        n = (s - mn) / (mx - mn); return 1-n if inv else n
    agent['score'] = norm(agent['avg_csat'])*0.30 + norm(agent['avg_fcr'])*0.30 + norm(agent['avg_aht'],inv=True)*0.20 + norm(agent['avg_abandon'],inv=True)*0.20
    q75 = agent['score'].quantile(0.75); q25 = agent['score'].quantile(0.25)
    agent['tier'] = agent['score'].apply(lambda s: 'TOP' if s>=q75 else ('RISK' if s<=q25 else 'MID'))
    agent = agent.sort_values('score', ascending=False).reset_index(drop=True); agent.index += 1
    display = agent[['agent_name','experience_months','avg_csat','avg_aht','avg_abandon','avg_fcr','total_calls','tier','score']].copy()
    display.columns = ['Agent','Exp (mo)','CSAT','AHT (s)','Abandon %','FCR %','Total Calls','Tier','Score']
    display['Abandon %'] = (display['Abandon %']*100).round(2)
    display['FCR %']     = (display['FCR %']*100).round(2)
    display['AHT (s)']   = display['AHT (s)'].round(0).astype(int)
    display['CSAT']      = display['CSAT'].round(2)
    display['Score']     = display['Score'].round(3)
    section_title("RANKED TABLE"); st.dataframe(display, use_container_width=True)
    risk = agent[agent['tier']=='RISK']['agent_name'].tolist()
    if risk: alert_bar(f"COACHING FLAG — {', '.join(risk)} — bottom quartile composite score.")
    section_title("CSAT vs AHT")
    fig, ax = plt.subplots(figsize=(10, 4))
    cm = {'TOP': C['green'], 'RISK': C['red'], 'MID': C['mid']}
    for _, row in agent.iterrows():
        ax.scatter(row['avg_aht'], row['avg_csat'], color=cm[row['tier']], s=60, zorder=3, alpha=0.9)
        ax.annotate(row['agent_name'].split()[0], (row['avg_aht'], row['avg_csat']), fontsize=7, color=C['text2'], xytext=(4,4), textcoords='offset points')
    ax.set_xlabel('Avg AHT (seconds)'); ax.set_ylabel('Avg CSAT'); ax.grid(alpha=0.3)
    patches = [mpatches.Patch(color=v, label=k) for k,v in cm.items()]
    ax.legend(handles=patches, fontsize=8, framealpha=0)
    ax.set_facecolor(C['surface']); fig.patch.set_facecolor(C['bg'])
    fig.tight_layout(pad=1.5); st.pyplot(fig, use_container_width=True); plt.close(fig)

def page_predictor(df):
    page_header("ABANDON RATE PREDICTOR", "Random Forest regression — R² 0.919 / 5-fold CV")
    rf, lr, metrics = load_model()
    if rf is None: st.error("Model not found. Run: python train_model.py"); return
    section_title("MODEL METRICS")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("RF R²",         f"{metrics.get('rf_r2',0):.3f}")
    c2.metric("RF RMSE",       f"{metrics.get('rf_rmse',0):.4f}")
    c3.metric("Baseline RMSE", f"{metrics.get('baseline_rmse',0):.4f}")
    c4.metric("CV R² 5-fold",  f"{metrics.get('cv_mean',0):.3f} ± {metrics.get('cv_std',0):.3f}")
    section_title("FEATURE IMPORTANCE")
    fi    = metrics.get('feature_importances', {})
    fi_df = pd.DataFrame({'Feature': list(fi.keys()), 'Importance': list(fi.values())}).sort_values('Importance', ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(fi_df['Feature'], fi_df['Importance'], color=C['accent'], alpha=0.8, height=0.5)
    ax.set_xlabel('Importance'); ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor(C['surface']); fig.patch.set_facecolor(C['bg'])
    fig.tight_layout(pad=1.5); st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.markdown("---"); section_title("SCENARIO FORECAST")
    c1,c2,c3 = st.columns(3)
    with c1:
        dow   = st.selectbox("Day of week", [0,1,2,3,4], format_func=lambda x: ['Monday','Tuesday','Wednesday','Thursday','Friday'][x])
        shift = st.selectbox("Shift", ['morning','afternoon','night'])
    with c2:
        queue = st.slider("Queue depth", 0, 60, 18)
        aht   = st.slider("Expected AHT (sec)", 120, 500, 280)
    with c3:
        exp   = st.slider("Avg team experience (mo)", 1, 36, 12)
        calls = st.slider("Expected calls handled", 10, 80, 45)
    le = LabelEncoder(); le.fit(['afternoon','morning','night'])
    inp = pd.DataFrame([{'day_of_week':dow,'is_monday':int(dow==0),'is_night_shift':int(shift=='night'),'shift':shift,'calls_in_queue':queue,'aht_seconds':aht,'experience_months':exp,'calls_handled':calls,'csat_score':df['csat_score'].mean(),'fcr_rate':df['fcr_rate'].mean()}])
    X   = engineer_features(inp)
    rfp = float(np.clip(rf.predict(X)[0], 0, 1))
    lrp = float(np.clip(lr.predict(X)[0], 0, 1))
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    def forecast_card(col, label, val, note=""):
        over = val > 0.05
        color = C['warn'] if over else C['accent']
        status = "ABOVE TARGET" if over else "WITHIN TARGET"
        col.markdown(f"""<div style='background:{C["surface"]};border:1px solid {C["border"]};border-top:2px solid {color};border-radius:3px;padding:1.5rem;text-align:center;'><div style='font-family:"IBM Plex Mono",monospace;font-size:0.62rem;letter-spacing:0.12em;color:{C["text3"]};text-transform:uppercase;margin-bottom:0.5rem;'>{label}</div><div style='font-family:"IBM Plex Mono",monospace;font-size:3rem;font-weight:600;color:{color};line-height:1;'>{val*100:.1f}%</div><div style='font-family:"IBM Plex Mono",monospace;font-size:0.68rem;color:{color};letter-spacing:0.08em;margin-top:0.5rem;'>{status}</div><div style='font-size:0.72rem;color:{C["text3"]};margin-top:0.3rem;'>target &lt; 5.0% {note}</div></div>""", unsafe_allow_html=True)
    with col1: forecast_card(col1, "RANDOM FOREST",     rfp, "/ primary model")
    with col2: forecast_card(col2, "LINEAR REGRESSION", lrp, "/ baseline")
    day_name = ['Monday','Tuesday','Wednesday','Thursday','Friday'][dow]
    if rfp > 0.05: alert_bar(f"Forecast {rfp*100:.1f}% exceeds 5.0% target on {day_name} {shift.upper()}. Consider increasing staffing or redistributing queue load.", level='warn')
    else:          alert_bar(f"Forecast {rfp*100:.1f}% within target range on {day_name} {shift.upper()}. Current configuration is sufficient.", level='info')

def main():
    df = load_data()
    page, date_range, shifts, agents = render_sidebar(df)
    df_f = apply_filters(df, date_range, shifts, agents)
    if len(df_f) == 0: st.warning("No data for selected filters."); return
    if   page == "OVERVIEW":  page_overview(df_f)
    elif page == "TRENDS":    page_trends(df_f)
    elif page == "AGENTS":    page_agents(df_f)
    elif page == "PREDICTOR": page_predictor(df_f)

if __name__ == '__main__':
    main()
