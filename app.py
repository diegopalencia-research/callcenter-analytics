"""
app.py — Call Center Intelligence Platform v5.0
Diego José Palencia Robles · 2026
Aesthetic: Mission Control / Space Telemetry
Stack: Streamlit · Supabase · Plotly · scikit-learn
"""
import os, json, pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="CC Intelligence · Palencia",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── PALETTE: Deep Space Telemetry ────────────────────────────────────────────
C = {
    'bg':       '#010307',
    'surface':  '#0a111c',
    'surface2': '#111827',
    'surface3': '#1a2335',
    'border':   '#1e2a44',
    'border2':  '#2a3b5c',
    'lavender': '#a5b4fc',
    'mint':     '#6ee7b7',
    'sky':      '#7dd3fc',
    'rose':     '#fda4af',
    'amber':    '#fcd34d',
    'sage':     '#a7f3d0',
    'accent':   '#a5b4fc',
    'success':  '#6ee7b7',
    'warn':     '#fcd34d',
    'danger':   '#fda4af',
    'text':     '#e0f2fe',
    'text2':    '#94a3b8',
    'text3':    '#475569',
    'glow':     'rgba(165,180,252,0.12)',
    'glow_line':'rgba(125,211,252,0.35)',
}

TEAM_COLORS = {
    'Tech Support': '#7dd3fc',
    'Billing':      '#a5b4fc',
    'Sales':        '#6ee7b7',
    'Retention':    '#fcd34d',
    'General':      '#fda4af',
}
TIER_COLORS = {'TOP': '#6ee7b7', 'MID': '#7dd3fc', 'RISK': '#fda4af'}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');
  html,body,[class*="css"]{{font-family:'Rajdhani',sans-serif;background:{C['bg']};color:{C['text']};}}
  .stApp{{background:{C['bg']};}}
  section[data-testid="stSidebar"]{{background:{C['surface']};border-right:1px solid {C['border']};}}
  section[data-testid="stSidebar"] *{{font-family:'Share Tech Mono',monospace !important;font-size:0.74rem !important;}}
  .block-container{{padding-top:1rem;padding-bottom:2rem;max-width:1300px;}}
  .stSelectbox>div>div,.stMultiSelect>div>div{{background:{C['surface2']} !important;border:1px solid {C['border2']} !important;color:{C['text']} !important;font-family:'Share Tech Mono',monospace !important;font-size:0.76rem !important;border-radius:3px !important;}}
  .stTabs [data-baseweb="tab-list"]{{background:{C['surface']};border-bottom:1px solid {C['border']};gap:0;}}
  .stTabs [data-baseweb="tab"]{{font-family:'Share Tech Mono',monospace;font-size:0.68rem;letter-spacing:0.12em;color:{C['text2']};padding:0.6rem 1.4rem;background:transparent;text-transform:uppercase;}}
  .stTabs [aria-selected="true"]{{color:{C['green']} !important;border-bottom:2px solid {C['green']} !important;background:transparent !important;}}
  .stDataFrame{{border:1px solid {C['border']} !important;border-radius:3px;}}
  .stDataFrame th{{background:{C['surface2']} !important;font-family:'Share Tech Mono',monospace !important;font-size:0.66rem !important;color:{C['text3']} !important;letter-spacing:0.1em;text-transform:uppercase;}}
  .stDataFrame td{{font-family:'Share Tech Mono',monospace !important;font-size:0.74rem !important;}}
  [data-testid="metric-container"]{{background:{C['surface']} !important;border:1px solid {C['border']} !important;border-radius:3px !important;padding:1rem !important;}}
  [data-testid="stMetricValue"]{{font-family:'Rajdhani',sans-serif !important;font-weight:600 !important;}}
  hr{{border-color:{C['border']} !important;margin:1.4rem 0;}}
  #MainMenu,footer,header{{visibility:hidden;}}
  div[data-testid="stCheckbox"] label span{{font-family:'Share Tech Mono',monospace !important;font-size:0.74rem !important;}}
  /* Lock sidebar — prevent accidental collapse */
  [data-testid="stSidebarCollapsedControl"]{{display:none !important;}}
  button[kind="header"]{{display:none !important;}}
</style>""", unsafe_allow_html=True)


# ── UI COMPONENTS ─────────────────────────────────────────────────────────────

def page_header(title, subtitle=None, tag=None):
    tag_html = f'<span style="background:{C["surface3"]};border:1px solid {C["border2"]};font-family:Share Tech Mono,monospace;font-size:0.58rem;letter-spacing:0.1em;color:{C["green"]};padding:0.15rem 0.55rem;border-radius:2px;margin-left:0.8rem;vertical-align:middle;">{tag}</span>' if tag else ''
    sub_html = f'<div style="font-family:Share Tech Mono,monospace;font-size:0.75rem;color:{C["text2"]};margin-top:0.35rem;letter-spacing:0.04em;">{subtitle}</div>' if subtitle else ''
    st.markdown(f"""
    <div style='border-bottom:1px solid {C["border"]};padding-bottom:1rem;margin-bottom:1.6rem;'>
      <div style='font-family:Share Tech Mono,monospace;font-size:0.56rem;letter-spacing:0.26em;color:{C["green"]};text-transform:uppercase;margin-bottom:0.35rem;opacity:0.7;'>SYS // CC-INTELLIGENCE · PALENCIA.RESEARCH · 2026</div>
      <div style='font-family:Rajdhani,sans-serif;font-size:1.9rem;font-weight:700;color:{C["text"]};letter-spacing:0.04em;text-transform:uppercase;line-height:1;'>{title}{tag_html}</div>
      {sub_html}
    </div>""", unsafe_allow_html=True)

def section_label(text):
    st.markdown(f'<div style="font-family:Share Tech Mono,monospace;font-size:0.6rem;letter-spacing:0.2em;color:{C["text3"]};text-transform:uppercase;border-left:2px solid {C["green"]};padding-left:0.75rem;margin:1.8rem 0 0.9rem;">{text}</div>', unsafe_allow_html=True)

def kpi_card(col, label, value, delta_pct, target=None, invert=False):
    is_good = (delta_pct < 0) if invert else (delta_pct > 0)
    arrow = "▲" if delta_pct > 0 else "▼"
    dc = C['lime'] if is_good else C['red']
    bc = C['green'] if is_good else C['amber']
    tgt = f'<div style="font-family:Share Tech Mono,monospace;font-size:0.58rem;color:{C["text3"]};margin-top:0.3rem;">TGT {target}</div>' if target else ''
    col.markdown(f"""
    <div style='background:{C["surface"]};border:1px solid {C["border"]};border-top:1px solid {bc};border-radius:3px;padding:1.1rem 1.2rem;position:relative;overflow:hidden;'>
      <div style='position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,{bc}00,{bc},{bc}00);'></div>
      <div style='font-family:Share Tech Mono,monospace;font-size:0.54rem;letter-spacing:0.18em;color:{C["text3"]};text-transform:uppercase;margin-bottom:0.5rem;'>{label}</div>
      <div style='font-family:Rajdhani,sans-serif;font-size:2.4rem;font-weight:700;color:{C["text"]};line-height:1;letter-spacing:0.02em;'>{value}</div>
      <div style='font-family:Share Tech Mono,monospace;font-size:0.62rem;color:{dc};margin-top:0.4rem;'>{arrow} {abs(delta_pct):.1f}% vs prior</div>
      {tgt}
    </div>""", unsafe_allow_html=True)

def alert_box(text, level='warn'):
    cfg = {
        'warn':    (C['amber'], rgba(C['amber'],0.05), '⚡'),
        'danger':  (C['red'],   rgba(C['red'],0.05),   '■ CRIT'),
        'info':    (C['cyan'],  rgba(C['cyan'],0.05),  '◈ SYS'),
        'success': (C['lime'],  rgba(C['lime'],0.05),  '✓ OK'),
    }
    color,bg,icon = cfg.get(level, cfg['warn'])
    st.markdown(f'<div style="background:{bg};border-left:2px solid {color};padding:0.65rem 1rem;margin:0.5rem 0;border-radius:0 3px 3px 0;"><span style="font-family:Share Tech Mono,monospace;font-size:0.68rem;color:{color};letter-spacing:0.08em;">{icon}</span><span style="font-family:Share Tech Mono,monospace;font-size:0.72rem;color:{C["text2"]};margin-left:0.6rem;">{text}</span></div>', unsafe_allow_html=True)

# Plotly base theme
PL = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor=C['surface'],
    font=dict(family='Share Tech Mono', color=C['text2'], size=10),
    xaxis=dict(gridcolor=C['border'], linecolor=C['border2'], tickfont=dict(size=9), zeroline=False),
    yaxis=dict(gridcolor=C['border'], linecolor=C['border2'], tickfont=dict(size=9), zeroline=False),
    margin=dict(l=8,r=8,t=28,b=8),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=C['border'], font=dict(size=9)),
    hoverlabel=dict(bgcolor=C['surface3'], bordercolor=C['border2'], font=dict(family='Share Tech Mono', size=10)),
)
def ply(fig, h=280, **kw):
    fig.update_layout(**{**PL,'height':h,**kw}); return fig


# ── DATA ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    err = None
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        from supabase import create_client
        sb = create_client(url, key)
        rows, offset, batch = [], 0, 1000
        while True:
            res = sb.table('daily_metrics').select('*').range(offset, offset+batch-1).execute()
            if not res.data: break
            rows.extend(res.data)
            if len(res.data) < batch: break
            offset += batch
        if rows:
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            for c in ['csat_score','abandon_rate','fcr_rate','escalation_rate','transfer_rate']:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            return df, "supabase", None
    except KeyError as e:
        err = f"Missing secret: {e}\n\nFix: Streamlit Cloud → App Settings → Secrets\nAdd:\nSUPABASE_URL = \"https://xxxx.supabase.co\"\nSUPABASE_KEY = \"eyJ...\"  ← must start with eyJ (anon/public key)"
    except ImportError:
        err = "supabase package missing. Add to requirements.txt:\nsupabase>=2.0"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    base = os.path.dirname(os.path.abspath(__file__))
    for name in ['callcenter_data.csv']:
        p = os.path.join(base, name)
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=['date'])
            return df, "csv", err
    st.error("No data source found."); st.stop()

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    mp  = os.path.join(base,'models','regressor.pkl')
    mep = os.path.join(base,'models','reg_metrics.json')
    if not os.path.exists(mp): return None, None, {}
    with open(mp,'rb') as f: models = pickle.load(f)
    metrics = json.load(open(mep)) if os.path.exists(mep) else {}
    return models['rf'], models['lr'], metrics

def engineer_features(df):
    f=pd.DataFrame(index=df.index)
    le=LabelEncoder(); le.fit(['afternoon','morning','night'])
    f['day_of_week']=df['day_of_week']; f['is_monday']=df['is_monday']
    f['is_night_shift']=df['is_night_shift']; f['shift_enc']=le.transform(df['shift'])
    f['calls_in_queue']=df['calls_in_queue']; f['aht_seconds']=df['aht_seconds']
    f['experience_months']=df['experience_months']; f['calls_handled']=df['calls_handled']
    f['csat_score']=df['csat_score']; f['fcr_rate']=df['fcr_rate']
    f['queue_x_monday']=df['calls_in_queue']*df['is_monday']
    f['queue_x_night']=df['calls_in_queue']*df['is_night_shift']
    return f

def get_kpis(df):
    mid=df['date'].min()+(df['date'].max()-df['date'].min())/2
    curr=df[df['date']>mid]; prev=df[df['date']<=mid]
    def avg(d,c): return d[c].mean() if len(d)>0 else 0
    def delt(c,p): return (c-p)/abs(p)*100 if p!=0 else 0
    out={}
    for col in ['aht_seconds','csat_score','abandon_rate','fcr_rate','escalation_rate','transfer_rate']:
        if col in df.columns:
            c=avg(curr,col); p=avg(prev,col)
            out[col]={'value':c,'delta':delt(c,p)}
    return out

def norm_score(s, invert=False):
    mn,mx=s.min(),s.max()
    if mx==mn: return pd.Series([0.5]*len(s),index=s.index)
    n=(s-mn)/(mx-mn); return 1-n if invert else n


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

def render_sidebar(df, source, err):
    st.sidebar.markdown(f"""
    <div style='padding:0.8rem 0 1.2rem;border-bottom:1px solid {C["border"]};margin-bottom:1.1rem;'>
      <div style='font-family:Share Tech Mono,monospace;font-size:0.5rem;letter-spacing:0.28em;color:{C["green"]};text-transform:uppercase;margin-bottom:0.3rem;opacity:0.7;'>◈ MISSION CONTROL</div>
      <div style='font-family:Rajdhani,sans-serif;font-size:1.05rem;font-weight:700;color:{C["text"]};text-transform:uppercase;letter-spacing:0.06em;'>PALENCIA RESEARCH</div>
      <div style='font-family:Share Tech Mono,monospace;font-size:0.6rem;color:{C["text3"]};margin-top:0.15rem;'>CC INTELLIGENCE · v5.0</div>
    </div>""", unsafe_allow_html=True)

    st.sidebar.markdown(f'<div style="font-family:Share Tech Mono;font-size:0.54rem;letter-spacing:0.18em;color:{C["text3"]};text-transform:uppercase;margin-bottom:0.5rem;">NAV</div>', unsafe_allow_html=True)
    page = st.sidebar.radio("nav", label_visibility="collapsed",
                             options=["HOME","OVERVIEW","TRENDS","TEAMS","AGENTS","PREDICTOR"])
    st.sidebar.markdown("---")
    st.sidebar.markdown(f'<div style="font-family:Share Tech Mono;font-size:0.54rem;letter-spacing:0.18em;color:{C["text3"]};text-transform:uppercase;margin-bottom:0.5rem;">FILTERS</div>', unsafe_allow_html=True)

    min_d=df['date'].min().date(); max_d=df['date'].max().date()
    date_range=st.sidebar.date_input("Date range",value=(min_d,max_d),min_value=min_d,max_value=max_d)
    shifts=st.sidebar.multiselect("Shift",options=['morning','afternoon','night'],default=['morning','afternoon','night'])
    teams=[]
    if 'team' in df.columns: teams=st.sidebar.multiselect("Team",options=sorted(df['team'].unique()),default=[])
    agents=st.sidebar.multiselect("Agent",options=sorted(df['agent_name'].unique()),default=[])
    years=[]
    if 'year' in df.columns:
        avail=sorted(df['year'].unique())
        years=st.sidebar.multiselect("Year",options=avail,default=avail)

    st.sidebar.markdown("---")
    sc,sl = (C['lime'],"● SUPABASE LIVE") if source=="supabase" else (C['amber'],"○ CSV FALLBACK")
    st.sidebar.markdown(f'<div style="font-family:Share Tech Mono,monospace;font-size:0.6rem;color:{sc};letter-spacing:0.08em;margin-bottom:0.5rem;">{sl}</div>', unsafe_allow_html=True)

    if source=="csv" and err:
        with st.sidebar.expander("⚡ Why CSV? (debug)"):
            st.markdown(f"""**Fix for Supabase connection:**

The `SUPABASE_KEY` must be the **anon/public key** that starts with `eyJ...`

**Steps:**
1. Go to your Supabase project
2. Project Settings → API
3. Copy the **"anon public"** key (NOT service_role, NOT personal token)
4. Streamlit Cloud → App Settings → Secrets → paste:

```
SUPABASE_URL = "https://pbdtojwwhtqojqhrwjia.supabase.co"
SUPABASE_KEY = "eyJ..."
```""")

    st.sidebar.markdown(f'<div style="font-family:Share Tech Mono,monospace;font-size:0.6rem;color:{C["text3"]};line-height:2;">ROWS&nbsp;&nbsp;&nbsp;{len(df):,}<br>AGENTS&nbsp;{df["agent_id"].nunique()}<br>PERIOD&nbsp;{df["date"].min().strftime("%b %Y")} → {df["date"].max().strftime("%b %Y")}<br>MODEL&nbsp;&nbsp;RF R² 0.919</div>', unsafe_allow_html=True)
    return page, date_range, shifts, teams, agents, years

def apply_filters(df, date_range, shifts, teams, agents, years):
    if len(date_range)==2:
        df=df[(df['date'].dt.date>=date_range[0])&(df['date'].dt.date<=date_range[1])]
    if shifts: df=df[df['shift'].isin(shifts)]
    if teams and 'team' in df.columns: df=df[df['team'].isin(teams)]
    if agents: df=df[df['agent_name'].isin(agents)]
    if years and 'year' in df.columns: df=df[df['year'].isin(years)]
    return df


# ── PAGE: HOME ────────────────────────────────────────────────────────────────

def page_home():
    tech_stack = ["Python 3.12","Streamlit","Supabase","PostgreSQL","Plotly","scikit-learn","Random Forest","Pandas","NumPy"]
    ps = f'background:{C["surface3"]};border:1px solid {C["border2"]};font-family:Share Tech Mono,monospace;font-size:0.58rem;letter-spacing:0.06em;color:{C["text2"]};padding:0.25rem 0.7rem;border-radius:2px;'
    tech_pills = "".join([f'<span style="{ps}">{t}</span>' for t in tech_stack])

    ml_items = [("RF R²","0.919",C['green']),("CV R² 5-fold","0.908",C['cyan']),("RMSE Improvement","3.5×",C['lime']),("Features","12",C['amber'])]
    ml_grid = "".join([f'<div><div style="font-family:Share Tech Mono,monospace;font-size:0.52rem;color:{C["text3"]};letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.25rem;">{l}</div><div style="font-family:Rajdhani,sans-serif;font-size:1.6rem;font-weight:700;color:{c};letter-spacing:0.02em;">{v}</div></div>' for l,v,c in ml_items])

    ds_items = [("Rows","6,829"),("Agents","25"),("Period","Jan 2025 → Mar 2026"),("Teams","5"),("Grain","Agent × Day"),("KPIs","8")]
    row_style = f'display:flex;justify-content:space-between;padding:0.4rem 0;border-bottom:1px solid {C["border"]};'
    ds_rows = "".join([f'<div style="{row_style}"><span style="font-family:Share Tech Mono,monospace;font-size:0.64rem;color:{C["text3"]};">{k}</span><span style="font-family:Share Tech Mono,monospace;font-size:0.64rem;color:{C["text"]};font-weight:500;">{v}</span></div>' for k,v in ds_items])

    kpi_items = [("AHT","< 300s",C['cyan']),("CSAT","> 4.20 / 5.0",C['green']),("Abandon Rate","< 5.0%",C['red']),("FCR","> 70%",C['lime']),("Escalation","< 8%",C['amber']),("Transfer","< 10%",C['purple'])]
    kpi_rows = "".join([f'<div style="{row_style}"><span style="font-family:Rajdhani,sans-serif;font-size:0.85rem;font-weight:600;color:{c};letter-spacing:0.04em;">{k}</span><span style="font-family:Share Tech Mono,monospace;font-size:0.62rem;color:{C["text3"]};">{v}</span></div>' for k,v,c in kpi_items])

    feat_items = [
        (C['green'],  "◈","Executive Overview",   "4–6 live KPI cards with period-over-period deltas. Weekly call volume vs abandon rate, shift breakdown, day×shift heatmap."),
        (C['cyan'],   "◎","Temporal Analysis",     "Daily time series for AHT, CSAT, Abandon Rate and FCR. Configurable rolling averages. Statistical anomaly detection with z-score flagging."),
        (C['lime'],   "◍","Team Intelligence",     "Comparative KPI radar across 5 teams. Monthly CSAT trends by team. Composite ranking. Identifies improving vs declining units."),
        (C['red'],    "◉","Agent Performance",     "Composite score: CSAT 30% + FCR 30% + AHT 20% + Abandon 20%. TOP/MID/RISK tier classification with automatic coaching flags."),
        (C['amber'],  "◊","ML Predictor",          "Random Forest (R² 0.919) predicts abandon rate from 12 engineered features. Queue sensitivity curve. Staffing recommendations."),
        (C['purple'], "◌","Supabase + PostgreSQL", "Production DB with Row Level Security. 5 tables, 3 views, composite indexes. Pre-configured for multi-tenant SaaS scale."),
    ]

    # ── Hero ──
    st.markdown(f"""
    <div style='padding:3rem 1rem 2.5rem;background:radial-gradient(ellipse at 50% -20%,{rgba(C["green"],0.07)} 0%,transparent 60%);border-bottom:1px solid {C["border"]};margin-bottom:2.5rem;position:relative;'>
      <div style='position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,{C["green"]},transparent);opacity:0.4;'></div>
      <div style='text-align:center;'>
        <div style='font-family:Share Tech Mono,monospace;font-size:0.6rem;letter-spacing:0.32em;color:{C["green"]};text-transform:uppercase;margin-bottom:1.2rem;opacity:0.8;'>◈ CALL CENTER INTELLIGENCE PLATFORM · MISSION CONTROL</div>
        <div style='font-family:Rajdhani,sans-serif;font-size:3.6rem;font-weight:700;color:{C["text"]};letter-spacing:0.06em;text-transform:uppercase;line-height:1;margin-bottom:0.6rem;'>OPERATIONAL<br><span style='color:{C["green"]};text-shadow:0 0 30px {rgba(C["green"],0.5)};'>ANALYTICS</span></div>
        <div style='font-family:Share Tech Mono,monospace;font-size:0.78rem;color:{C["text2"]};max-width:560px;margin:1rem auto 1.8rem;line-height:1.8;letter-spacing:0.03em;'>Production-grade BI dashboard for call center operations — real-time KPI telemetry, agent performance ranking, and ML-powered abandon rate prediction.</div>
        <div style='display:flex;justify-content:center;gap:0.5rem;flex-wrap:wrap;'>{tech_pills}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Feature cards ──
    cols = st.columns(3)
    for i,(color,icon,title,desc) in enumerate(feat_items):
        with cols[i%3]:
            st.markdown(f"""
            <div style='background:{C["surface"]};border:1px solid {C["border"]};border-top:1px solid {color};border-radius:3px;padding:1.3rem;margin-bottom:1rem;min-height:175px;position:relative;overflow:hidden;'>
              <div style='position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,{color}00,{color},{color}00);'></div>
              <div style='font-family:Share Tech Mono,monospace;font-size:1.1rem;color:{color};margin-bottom:0.6rem;'>{icon}</div>
              <div style='font-family:Rajdhani,sans-serif;font-size:1rem;font-weight:600;color:{C["text"]};margin-bottom:0.4rem;text-transform:uppercase;letter-spacing:0.04em;'>{title}</div>
              <div style='font-family:Share Tech Mono,monospace;font-size:0.65rem;color:{C["text2"]};line-height:1.7;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ML specs ──
    st.markdown(f"""
    <div style='background:{C["surface"]};border:1px solid {C["border"]};border-radius:3px;padding:1.6rem;margin-bottom:1.2rem;position:relative;overflow:hidden;'>
      <div style='position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,{C["green"]}00,{C["green"]},{C["green"]}00);opacity:0.5;'></div>
      <div style='font-family:Share Tech Mono,monospace;font-size:0.56rem;letter-spacing:0.22em;color:{C["green"]};text-transform:uppercase;margin-bottom:1.2rem;'>ML MODEL SPECIFICATIONS // RANDOM FOREST REGRESSOR</div>
      <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:1.2rem;margin-bottom:1.2rem;'>{ml_grid}</div>
      <div style='font-family:Share Tech Mono,monospace;font-size:0.65rem;color:{C["text3"]};line-height:2;border-top:1px solid {C["border"]};padding-top:1rem;'>
        TOP FEATURES → <span style='color:{C["green"]};'>queue×monday [0.363]</span> · <span style='color:{C["cyan"]};'>day_of_week [0.254]</span> · <span style='color:{C["lime"]};'>calls_in_queue [0.243]</span> · <span style='color:{C["amber"]};'>is_monday [0.133]</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Dataset + KPI targets ──
    col_l,col_r = st.columns(2)
    with col_l:
        st.markdown(f'<div style="background:{C["surface"]};border:1px solid {C["border"]};border-radius:3px;padding:1.4rem;"><div style="font-family:Share Tech Mono,monospace;font-size:0.56rem;letter-spacing:0.22em;color:{C["green"]};text-transform:uppercase;margin-bottom:1rem;">DATASET SPECS</div>{ds_rows}</div>', unsafe_allow_html=True)
    with col_r:
        st.markdown(f'<div style="background:{C["surface"]};border:1px solid {C["border"]};border-radius:3px;padding:1.4rem;"><div style="font-family:Share Tech Mono,monospace;font-size:0.56rem;letter-spacing:0.22em;color:{C["green"]};text-transform:uppercase;margin-bottom:1rem;">KPI TARGETS</div>{kpi_rows}</div>', unsafe_allow_html=True)

    # ── Author ──
    st.markdown(f"""
    <div style='margin-top:1.2rem;padding:1.3rem 1.6rem;background:{C["surface"]};border:1px solid {C["border"]};border-radius:3px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;'>
      <div>
        <div style='font-family:Rajdhani,sans-serif;font-size:1.05rem;font-weight:700;color:{C["text"]};text-transform:uppercase;letter-spacing:0.06em;'>Diego José Palencia Robles</div>
        <div style='font-family:Share Tech Mono,monospace;font-size:0.6rem;color:{C["text3"]};margin-top:0.15rem;letter-spacing:0.04em;'>DATA SCIENCE · NLP · APPLIED AI · MACHINE LEARNING</div>
      </div>
      <div style='display:flex;gap:0.6rem;'>
        <a href="https://github.com/diegopalencia-research" style='text-decoration:none;'><span style='background:{C["surface3"]};border:1px solid {C["border2"]};font-family:Share Tech Mono,monospace;font-size:0.6rem;color:{C["green"]};padding:0.28rem 0.75rem;border-radius:2px;'>GITHUB</span></a>
        <a href="https://linkedin.com/in/diego-jose-palencia-robles" style='text-decoration:none;'><span style='background:{C["surface3"]};border:1px solid {C["border2"]};font-family:Share Tech Mono,monospace;font-size:0.6rem;color:{C["cyan"]};padding:0.28rem 0.75rem;border-radius:2px;'>LINKEDIN</span></a>
        <a href="https://callcenter-analytics.streamlit.app" style='text-decoration:none;'><span style='background:{C["surface3"]};border:1px solid {C["border2"]};font-family:Share Tech Mono,monospace;font-size:0.6rem;color:{C["lime"]};padding:0.28rem 0.75rem;border-radius:2px;'>LIVE APP</span></a>
      </div>
    </div>""", unsafe_allow_html=True)


# ── PAGE: OVERVIEW ────────────────────────────────────────────────────────────

def page_overview(df):
    date_str=f"{df['date'].min().strftime('%d %b %Y')} → {df['date'].max().strftime('%d %b %Y')}"
    page_header("OVERVIEW", f"{len(df):,} RECORDS · {df['agent_id'].nunique()} AGENTS · {date_str}")
    kpis=get_kpis(df)
    c1,c2,c3,c4=st.columns(4)
    kpi_card(c1,"AVG HANDLE TIME",f"{kpis['aht_seconds']['value']:.0f}s",kpis['aht_seconds']['delta'],"< 300s",invert=True)
    kpi_card(c2,"CSAT SCORE",f"{kpis['csat_score']['value']:.2f}",kpis['csat_score']['delta'],"> 4.20")
    kpi_card(c3,"ABANDON RATE",f"{kpis['abandon_rate']['value']*100:.1f}%",kpis['abandon_rate']['delta'],"< 5.0%",invert=True)
    kpi_card(c4,"FIRST CALL RES.",f"{kpis['fcr_rate']['value']*100:.1f}%",kpis['fcr_rate']['delta'],"> 70%")
    st.markdown("<br>", unsafe_allow_html=True)
    if 'escalation_rate' in kpis:
        c5,c6,c7,c8=st.columns(4)
        kpi_card(c5,"ESCALATION",f"{kpis['escalation_rate']['value']*100:.1f}%",kpis['escalation_rate']['delta'],"< 8%",invert=True)
        kpi_card(c6,"TRANSFER RATE",f"{kpis['transfer_rate']['value']*100:.1f}%",kpis['transfer_rate']['delta'],"< 10%",invert=True)
        kpi_card(c7,"TOTAL CALLS",f"{df['calls_handled'].sum():,}",0)
        kpi_card(c8,"AVG EXPERIENCE",f"{df['experience_months'].mean():.0f} MO",0)
    st.markdown("<br>", unsafe_allow_html=True)
    cl,cr=st.columns([3,2])
    with cl:
        section_label("WEEKLY CALL VOLUME // ABANDON RATE TELEMETRY")
        daily=df.groupby('date').agg(calls=('calls_handled','sum'),abandon=('abandon_rate','mean')).reset_index()
        daily['week']=daily['date'].dt.to_period('W').apply(lambda x:x.start_time)
        wk=daily.groupby('week').agg(calls=('calls','sum'),abandon=('abandon','mean')).reset_index()
        fig=make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=wk['week'],y=wk['calls'],name='CALLS',marker_color=C['cyan'],opacity=0.5,marker_line_width=0),secondary_y=False)
        fig.add_trace(go.Scatter(x=wk['week'],y=wk['abandon']*100,name='ABANDON %',line=dict(color=C['green'],width=2),mode='lines+markers',marker=dict(size=4,color=C['green'])),secondary_y=True)
        fig.add_hline(y=5.0,line_dash="dash",line_color=C['amber'],opacity=0.5,secondary_y=True,annotation_text="5% TGT",annotation_font_size=8,annotation_font_color=C['amber'])
        fig.update_yaxes(title_text="CALLS",secondary_y=False,gridcolor=C['border'])
        fig.update_yaxes(title_text="ABANDON %",secondary_y=True,gridcolor='rgba(0,0,0,0)')
        ply(fig,h=260); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    with cr:
        section_label("SHIFT BREAKDOWN")
        sdf=df.groupby('shift').agg(csat=('csat_score','mean'),fcr=('fcr_rate','mean'),abandon=('abandon_rate','mean')).reset_index()
        sc={'morning':C['green'],'afternoon':C['cyan'],'night':C['purple']}
        fig2=go.Figure()
        for _,row in sdf.iterrows():
            fig2.add_trace(go.Bar(name=row['shift'].upper(),x=['CSAT','FCR %','ABANDON %'],
                y=[row['csat']/5*100,row['fcr']*100,row['abandon']*100],
                marker_color=sc.get(row['shift'],C['green']),opacity=0.8))
        ply(fig2,h=260,barmode='group',yaxis=dict(title='%',gridcolor=C['border']))
        st.plotly_chart(fig2,use_container_width=True,config={'displayModeBar':False})
    if len(df)>50:
        section_label("ABANDON RATE MATRIX — DAY × SHIFT")
        pivot=df.groupby(['day_of_week','shift'])['abandon_rate'].mean().unstack(fill_value=0)
        pivot.index=['MON','TUE','WED','THU','FRI'][:len(pivot)]
        fig3=go.Figure(go.Heatmap(
            z=(pivot.values*100).round(1),x=[c.upper() for c in pivot.columns],y=pivot.index.tolist(),
            colorscale=[[0,C['surface3']],[0.4,rgba(C['cyan'],1)],[1,rgba(C['red'],1)]],
            text=(pivot.values*100).round(1),texttemplate="%{text}%",
            textfont=dict(size=10,family='Share Tech Mono'),showscale=True,
            hovertemplate='%{y} · %{x}<br>ABANDON: %{z}%<extra></extra>'))
        ply(fig3,h=185,xaxis=dict(side='top',gridcolor='rgba(0,0,0,0)'),yaxis=dict(gridcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig3,use_container_width=True,config={'displayModeBar':False})
    av=kpis['abandon_rate']['value']*100; cv=kpis['csat_score']['value']
    if av>10: alert_box(f"ABANDON RATE {av:.1f}% — CRITICAL THRESHOLD EXCEEDED. IMMEDIATE STAFFING REVIEW.",'danger')
    elif av>5: alert_box(f"ABANDON RATE {av:.1f}% ABOVE 5% TARGET. QUEUE REVIEW RECOMMENDED.",'warn')
    if cv<4.0: alert_box(f"CSAT {cv:.2f} BELOW 4.0 TARGET. NIGHT SHIFT COACHING REQUIRED.",'warn')


# ── PAGE: TRENDS ──────────────────────────────────────────────────────────────

def page_trends(df):
    page_header("TEMPORAL ANALYSIS","DAILY KPI TELEMETRY · ROLLING AVERAGE · ANOMALY DETECTION")
    ca,_=st.columns([1,3])
    with ca:
        window=st.selectbox("Rolling window",[3,7,14,30],index=1,format_func=lambda x:f"{x}D AVG")
        show_anom=st.checkbox("Flag anomalies",value=True)
    daily=df.groupby('date').agg(aht_seconds=('aht_seconds','mean'),csat_score=('csat_score','mean'),abandon_rate=('abandon_rate','mean'),fcr_rate=('fcr_rate','mean')).reset_index()
    for col in ['aht_seconds','csat_score','abandon_rate','fcr_rate']:
        daily[f'{col}_roll']=daily[col].rolling(window,min_periods=1).mean()
        m,s=daily[col].mean(),daily[col].std()
        daily[f'{col}_anom']=(daily[col]-m).abs()>(2.0*s)

    def trend_chart(col, label, target, color, pct=False):
        fig=go.Figure()
        yv=daily[col]*100 if pct else daily[col]
        yr=daily[f'{col}_roll']*100 if pct else daily[f'{col}_roll']
        tv=target*100 if pct else target
        # Fill area — use rgba() properly
        fig.add_trace(go.Scatter(x=daily['date'],y=yv,fill='tozeroy',
            fillcolor=rgba(color,0.06),
            line=dict(color='rgba(0,0,0,0)'),showlegend=False,hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=daily['date'],y=yv,mode='lines',name='DAILY',
            line=dict(color=rgba(color,0.5),width=1,dash='dot')))
        fig.add_trace(go.Scatter(x=daily['date'],y=yr,mode='lines',name=f'{window}D AVG',
            line=dict(color=color,width=2.5)))
        fig.add_hline(y=tv,line_dash="dash",line_color=C['amber'],opacity=0.5,
            annotation_text=f"TGT {tv:.0f}{'%' if pct else ''}",
            annotation_font_size=8,annotation_font_color=C['amber'],annotation_position="bottom right")
        if show_anom:
            an=daily[daily[f'{col}_anom']]
            ay=an[col]*100 if pct else an[col]
            if len(an):
                fig.add_trace(go.Scatter(x=an['date'],y=ay,mode='markers',name='ANOMALY',
                    marker=dict(color=C['red'],size=7,symbol='x',line=dict(color=C['red'],width=2))))
        ply(fig,h=300,yaxis=dict(title=label,gridcolor=C['border']))
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
        v=daily[col]
        sp = f'background:{C["surface2"]};border:1px solid {C["border"]};font-family:Share Tech Mono,monospace;font-size:0.6rem;padding:0.22rem 0.65rem;border-radius:2px;'
        stats_items=[("MIN",f"{v.min()*100:.1f}%" if pct else f"{v.min():.1f}",C['lime']),
                     ("MAX",f"{v.max()*100:.1f}%" if pct else f"{v.max():.1f}",C['red']),
                     ("AVG",f"{v.mean()*100:.1f}%" if pct else f"{v.mean():.1f}",C['cyan']),
                     ("ANOMALIES",str(daily[f'{col}_anom'].sum()),C['amber'])]
        pills="".join([f'<span style="{sp}color:{c};">{lb} {sv}</span>' for lb,sv,c in stats_items])
        st.markdown(f'<div style="display:flex;gap:0.6rem;flex-wrap:wrap;margin-bottom:0.5rem;">{pills}</div>', unsafe_allow_html=True)

    t1,t2,t3,t4=st.tabs(["  AHT  ","  CSAT  ","  ABANDON RATE  ","  FCR  "])
    with t1: trend_chart('aht_seconds','AHT (S)',300,C['cyan'])
    with t2: trend_chart('csat_score','CSAT SCORE',4.2,C['green'])
    with t3: trend_chart('abandon_rate','ABANDON %',0.05,C['red'],pct=True)
    with t4: trend_chart('fcr_rate','FCR %',0.70,C['lime'],pct=True)


# ── PAGE: TEAMS ───────────────────────────────────────────────────────────────

def page_teams(df):
    if 'team' not in df.columns: alert_box("Team column not in dataset.",'warn'); return
    page_header("TEAM INTELLIGENCE","COMPARATIVE ANALYSIS · 5 OPERATIONAL UNITS",tag="2025–2026")
    agg=df.groupby('team').agg(avg_aht=('aht_seconds','mean'),avg_csat=('csat_score','mean'),avg_abandon=('abandon_rate','mean'),avg_fcr=('fcr_rate','mean'),total_calls=('calls_handled','sum'),agents=('agent_id','nunique')).reset_index()
    agg['score']=(norm_score(agg['avg_csat'])*0.30+norm_score(agg['avg_fcr'])*0.30+norm_score(agg['avg_aht'],invert=True)*0.20+norm_score(agg['avg_abandon'],invert=True)*0.20)
    agg=agg.sort_values('score',ascending=False)
    cl,cr=st.columns([3,2])
    with cl:
        section_label("KPI RADAR // ALL UNITS")
        cats=['CSAT','FCR','LOW AHT','LOW ABANDON','SCORE']
        fig=go.Figure()
        for _,row in agg.iterrows():
            vals=[row['avg_csat']/5,row['avg_fcr'],1-row['avg_aht']/500,1-row['avg_abandon'],row['score']]
            vp=[v*100 for v in vals]+[vals[0]*100]
            tc=TEAM_COLORS.get(row['team'],C['green'])
            fig.add_trace(go.Scatterpolar(r=vp,theta=cats+[cats[0]],fill='toself',name=row['team'],
                line=dict(color=tc,width=2),fillcolor=rgba(tc,0.08),opacity=0.95))
        ply(fig,h=360,polar=dict(bgcolor=C['surface'],radialaxis=dict(visible=True,range=[0,100],gridcolor=C['border'],tickfont_size=8),angularaxis=dict(gridcolor=C['border'])))
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    with cr:
        section_label("UNIT RANKING")
        for i,(_,row) in enumerate(agg.iterrows()):
            c=TEAM_COLORS.get(row['team'],C['green'])
            st.markdown(f'<div style="background:{C["surface"]};border:1px solid {C["border"]};border-left:2px solid {c};border-radius:2px;padding:0.75rem 1rem;margin-bottom:0.45rem;"><div style="display:flex;justify-content:space-between;align-items:center;"><div><span style="font-family:Share Tech Mono,monospace;font-size:0.56rem;color:{C["text3"]};">#{i+1:02d}</span><span style="font-family:Rajdhani,sans-serif;font-size:0.9rem;font-weight:600;color:{C["text"]};margin-left:0.5rem;text-transform:uppercase;letter-spacing:0.04em;">{row["team"]}</span></div><span style="font-family:Rajdhani,sans-serif;font-size:1rem;font-weight:700;color:{c};">{row["score"]:.3f}</span></div><div style="font-family:Share Tech Mono,monospace;font-size:0.58rem;color:{C["text3"]};margin-top:0.2rem;">{int(row["agents"])} AGENTS · {int(row["total_calls"]):,} CALLS</div></div>', unsafe_allow_html=True)
    if 'year' in df.columns and 'month' in df.columns:
        section_label("MONTHLY CSAT TELEMETRY BY TEAM")
        monthly=df.groupby(['year','month','team'])['csat_score'].mean().reset_index()
        monthly['period']=pd.to_datetime(monthly[['year','month']].assign(day=1))
        fig2=go.Figure()
        for team in df['team'].unique():
            t=monthly[monthly['team']==team].sort_values('period')
            tc=TEAM_COLORS.get(team,C['green'])
            fig2.add_trace(go.Scatter(x=t['period'],y=t['csat_score'],mode='lines+markers',name=team,line=dict(color=tc,width=2),marker=dict(size=4,color=tc)))
        fig2.add_hline(y=4.2,line_dash="dash",line_color=C['amber'],opacity=0.4,annotation_text="4.2 TGT",annotation_font_size=8,annotation_font_color=C['amber'])
        ply(fig2,h=280,yaxis=dict(title='AVG CSAT',range=[3.0,5.2],gridcolor=C['border']))
        st.plotly_chart(fig2,use_container_width=True,config={'displayModeBar':False})
    section_label("METRICS TABLE")
    d=agg.copy(); d['avg_aht']=d['avg_aht'].round(0).astype(int); d['avg_csat']=d['avg_csat'].round(2)
    d['avg_abandon']=(d['avg_abandon']*100).round(2); d['avg_fcr']=(d['avg_fcr']*100).round(2); d['score']=d['score'].round(3)
    d.columns=['Team','AHT(s)','CSAT','Abandon%','FCR%','Calls','Agents','Score']
    st.dataframe(d,use_container_width=True,hide_index=True)


# ── PAGE: AGENTS ──────────────────────────────────────────────────────────────

def page_agents(df):
    page_header("AGENT PERFORMANCE","COMPOSITE SCORE: CSAT 30% · FCR 30% · AHT 20% · ABANDON 20%")
    gcols=['agent_id','agent_name','experience_months']
    if 'team' in df.columns: gcols.append('team')
    agent=df.groupby(gcols).agg(avg_aht=('aht_seconds','mean'),avg_csat=('csat_score','mean'),avg_abandon=('abandon_rate','mean'),avg_fcr=('fcr_rate','mean'),total_calls=('calls_handled','sum'),days_worked=('date','nunique')).reset_index()
    agent['score']=(norm_score(agent['avg_csat'])*0.30+norm_score(agent['avg_fcr'])*0.30+norm_score(agent['avg_aht'],invert=True)*0.20+norm_score(agent['avg_abandon'],invert=True)*0.20)
    q75=agent['score'].quantile(0.75); q25=agent['score'].quantile(0.25)
    agent['tier']=agent['score'].apply(lambda s:'TOP' if s>=q75 else ('RISK' if s<=q25 else 'MID'))
    agent=agent.sort_values('score',ascending=False).reset_index(drop=True); agent.index+=1
    n=len(agent); tc=agent['tier'].value_counts()
    top_n=tc.get('TOP',0); mid_n=tc.get('MID',0); risk_n=tc.get('RISK',0)

    # Tier summary — 3 full-width cards
    st.markdown(f"""
    <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1.8rem;'>
      {"".join([f'<div style="background:{C["surface"]};border:1px solid {C["border"]};border-top:1px solid {color};border-radius:3px;padding:1.1rem 1.3rem;text-align:center;position:relative;overflow:hidden;"><div style="position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,{color}00,{color},{color}00);"></div><div style="font-family:Share Tech Mono,monospace;font-size:0.54rem;letter-spacing:0.18em;color:{C["text3"]};text-transform:uppercase;margin-bottom:0.4rem;">{tier} TIER</div><div style="font-family:Rajdhani,sans-serif;font-size:2.8rem;font-weight:700;color:{color};line-height:1;">{count}</div><div style="font-family:Share Tech Mono,monospace;font-size:0.6rem;color:{C["text3"]};margin-top:0.2rem;">{count/n*100:.0f}% · {desc}</div></div>' for tier,count,color,desc in [("TOP",top_n,C['lime'],"TOP QUARTILE"),("MID",mid_n,C['cyan'],"MIDDLE 50%"),("RISK",risk_n,C['red'],"COACHING NEEDED")]])}
    </div>""", unsafe_allow_html=True)

    # Score bar — full width
    section_label("COMPOSITE SCORE // ALL AGENTS")
    fig_bar=go.Figure()
    for tier in ['RISK','MID','TOP']:
        t=agent[agent['tier']==tier].sort_values('score')
        if not len(t): continue
        tc=TIER_COLORS[tier]
        fig_bar.add_trace(go.Bar(y=t['agent_name'],x=t['score'],orientation='h',name=tier,
            marker_color=tc,opacity=0.8,
            text=t['score'].apply(lambda v:f"{v:.3f}"),textposition='outside',
            textfont=dict(size=9,family='Share Tech Mono'),
            customdata=t[['avg_csat','avg_fcr','avg_aht','total_calls']].values,
            hovertemplate='<b>%{y}</b><br>SCORE: %{x:.3f}<br>CSAT: %{customdata[0]:.2f}<br>FCR: %{customdata[1]:.1%}<br>AHT: %{customdata[2]:.0f}s<br>CALLS: %{customdata[3]:,}<extra></extra>'))
    ply(fig_bar,h=max(320,n*18+40),xaxis=dict(title='COMPOSITE SCORE',range=[0,1.12],gridcolor=C['border']),yaxis=dict(gridcolor='rgba(0,0,0,0)'),barmode='stack')
    st.plotly_chart(fig_bar,use_container_width=True,config={'displayModeBar':False})

    # Two scatter charts side by side
    cl,cr=st.columns(2)
    with cl:
        section_label("CSAT vs AHT // AGENT MAP")
        fig2=go.Figure()
        for tier,color in TIER_COLORS.items():
            t=agent[agent['tier']==tier]
            if not len(t): continue
            fig2.add_trace(go.Scatter(x=t['avg_aht'],y=t['avg_csat'],mode='markers+text',name=tier,
                text=t['agent_name'].str.split().str[0],textposition='top right',
                textfont=dict(size=8,color=C['text3']),
                marker=dict(color=color,size=9,opacity=0.9,line=dict(color=C['bg'],width=1.5)),
                customdata=t[['agent_name','total_calls','avg_fcr']].values,
                hovertemplate='<b>%{customdata[0]}</b><br>AHT: %{x:.0f}s · CSAT: %{y:.2f}<br>FCR: %{customdata[2]:.1%} · %{customdata[1]:,} calls<extra></extra>'))
        fig2.add_vline(x=300,line_dash="dash",line_color=C['text3'],opacity=0.3,annotation_text="300s",annotation_font_size=8)
        fig2.add_hline(y=4.2,line_dash="dash",line_color=C['text3'],opacity=0.3,annotation_text="4.2",annotation_font_size=8)
        ply(fig2,h=300,xaxis=dict(title='AVG AHT (S)',gridcolor=C['border']),yaxis=dict(title='AVG CSAT',gridcolor=C['border']))
        st.plotly_chart(fig2,use_container_width=True,config={'displayModeBar':False})
    with cr:
        section_label("EXPERIENCE vs FCR RATE")
        fig3=go.Figure()
        if 'team' in agent.columns:
            for team in agent['team'].unique():
                t=agent[agent['team']==team]
                fig3.add_trace(go.Scatter(x=t['experience_months'],y=t['avg_fcr']*100,mode='markers',name=team,
                    marker=dict(color=TEAM_COLORS.get(team,C['green']),size=9,opacity=0.85),
                    text=t['agent_name'],hovertemplate='%{text}<br>EXP: %{x}mo · FCR: %{y:.1f}%<extra></extra>'))
        else:
            fig3.add_trace(go.Scatter(x=agent['experience_months'],y=agent['avg_fcr']*100,mode='markers',
                marker=dict(color=C['green'],size=9),text=agent['agent_name'],
                hovertemplate='%{text}<br>EXP: %{x}mo · FCR: %{y:.1f}%<extra></extra>'))
        ply(fig3,h=300,xaxis=dict(title='EXPERIENCE (MONTHS)',gridcolor=C['border']),yaxis=dict(title='FCR RATE (%)',gridcolor=C['border']))
        st.plotly_chart(fig3,use_container_width=True,config={'displayModeBar':False})

    # Coaching flags
    risk=agent[agent['tier']=='RISK']
    if len(risk):
        section_label("COACHING FLAGS")
        for _,row in risk.iterrows():
            team_info=f" · {row['team']}" if 'team' in agent.columns else ""
            alert_box(f"{row['agent_name']}{team_info} — SCORE {row['score']:.3f} · CSAT {row['avg_csat']:.2f} · ABANDON {row['avg_abandon']:.1%}",'warn')

    # Full table
    section_label("COMPLETE RANKING TABLE")
    dcols=['agent_name','experience_months','avg_csat','avg_aht','avg_abandon','avg_fcr','total_calls','days_worked','tier','score']
    if 'team' in agent.columns: dcols=['agent_name','team']+dcols[1:]
    d=agent[dcols].copy()
    d['avg_abandon']=(d['avg_abandon']*100).round(2); d['avg_fcr']=(d['avg_fcr']*100).round(2)
    d['avg_aht']=d['avg_aht'].round(0).astype(int); d['avg_csat']=d['avg_csat'].round(2); d['score']=d['score'].round(3)
    d.rename(columns={'agent_name':'AGENT','team':'TEAM','experience_months':'EXP(MO)','avg_csat':'CSAT','avg_aht':'AHT(S)','avg_abandon':'ABANDON%','avg_fcr':'FCR%','total_calls':'CALLS','days_worked':'DAYS','tier':'TIER','score':'SCORE'},inplace=True)
    st.dataframe(d,use_container_width=True)


# ── PAGE: PREDICTOR ───────────────────────────────────────────────────────────

def page_predictor(df):
    page_header("ML PREDICTOR","RANDOM FOREST REGRESSOR · R² 0.919 · 5-FOLD CV · 12 FEATURES")
    rf,lr,metrics=load_model()
    if rf is None: alert_box("models/regressor.pkl NOT FOUND. Ensure models/ folder is in the repo.",'danger'); return

    section_label("MODEL PERFORMANCE METRICS")
    c1,c2,c3,c4,c5=st.columns(5)
    for col,lbl,val in [(c1,"RF R²",f"{metrics.get('rf_r2',0):.3f}"),(c2,"RF RMSE",f"{metrics.get('rf_rmse',0):.4f}"),
                        (c3,"LR R²",f"{metrics.get('lr_r2',0):.3f}"),(c4,"CV R² 5-FOLD",f"{metrics.get('cv_mean',0):.3f}"),
                        (c5,"BASELINE RMSE",f"{metrics.get('baseline_rmse',0):.4f}")]:
        col.metric(lbl,val)
    impr=metrics.get('baseline_rmse',1)/max(metrics.get('rf_rmse',0.01),0.001)
    alert_box(f"RANDOM FOREST: {impr:.1f}× RMSE IMPROVEMENT OVER NAIVE MEAN BASELINE",'info')

    cl,cr=st.columns([2,1])
    with cl:
        section_label("FEATURE IMPORTANCE RANKING")
        fi=metrics.get('feature_importances',{})
        if fi:
            fi_df=pd.DataFrame({'Feature':list(fi.keys()),'Importance':list(fi.values())}).sort_values('Importance',ascending=True).tail(10)
            fig=go.Figure(go.Bar(x=fi_df['Importance'],y=fi_df['Feature'],orientation='h',
                marker=dict(color=fi_df['Importance'],colorscale=[[0,C['surface3']],[0.5,C['cyan']],[1,C['green']]]),
                text=fi_df['Importance'].apply(lambda x:f"{x:.3f}"),textposition='outside',textfont=dict(size=9,family='Share Tech Mono')))
            ply(fig,h=280,xaxis=dict(title='IMPORTANCE',gridcolor=C['border']),yaxis=dict(gridcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    with cr:
        section_label("SIGNAL INTERPRETATION")
        st.markdown(f'<div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;color:{C["text2"]};line-height:1.9;"><span style="color:{C["green"]};">queue×monday [0.363]</span> — Monday backlog amplifies abandons exponentially.<br><br><span style="color:{C["cyan"]};">day_of_week [0.254]</span> — Temporal patterns outperform all agent-level features.<br><br>Implication: staff decisions must use <span style="color:{C["lime"]};">queue forecasting</span> not historical averages.</div>', unsafe_allow_html=True)

    st.markdown("---"); section_label("SCENARIO FORECAST SIMULATOR")
    c1,c2,c3=st.columns(3)
    with c1: dow=st.selectbox("Day of week",[0,1,2,3,4],format_func=lambda x:['Monday','Tuesday','Wednesday','Thursday','Friday'][x]); shift=st.selectbox("Shift",['morning','afternoon','night'])
    with c2: queue=st.slider("Queue depth",0,70,18); aht=st.slider("Expected AHT (s)",120,500,280)
    with c3: exp=st.slider("Avg experience (months)",1,48,18); calls=st.slider("Expected calls handled",10,90,45)

    le=LabelEncoder(); le.fit(['afternoon','morning','night'])
    inp=pd.DataFrame([{'day_of_week':dow,'is_monday':int(dow==0),'is_night_shift':int(shift=='night'),'shift':shift,'calls_in_queue':queue,'aht_seconds':aht,'experience_months':exp,'calls_handled':calls,'csat_score':df['csat_score'].mean(),'fcr_rate':df['fcr_rate'].mean()}])
    X=engineer_features(inp)
    rfp=float(np.clip(rf.predict(X)[0],0,1)); lrp=float(np.clip(lr.predict(X)[0],0,1)); ens=(rfp+lrp)/2

    st.markdown("<br>", unsafe_allow_html=True)
    fc1,fc2,fc3=st.columns(3)
    for col,lbl,val,note in [(fc1,"RANDOM FOREST",rfp,"PRIMARY MODEL"),(fc2,"LINEAR REG.",lrp,"BASELINE"),(fc3,"ENSEMBLE AVG",ens,"RF + LR / 2")]:
        crit=val>0.10; over=val>0.05
        color=C['red'] if crit else (C['amber'] if over else C['lime'])
        status="■ CRITICAL" if crit else ("⚡ ABOVE TGT" if over else "✓ NOMINAL")
        col.markdown(f'<div style="background:{C["surface"]};border:1px solid {C["border"]};border-top:1px solid {color};border-radius:3px;padding:1.5rem;text-align:center;position:relative;overflow:hidden;"><div style="position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,{color}00,{color},{color}00);"></div><div style="font-family:Share Tech Mono,monospace;font-size:0.54rem;letter-spacing:0.16em;color:{C["text3"]};text-transform:uppercase;margin-bottom:0.6rem;">{lbl}</div><div style="font-family:Rajdhani,sans-serif;font-size:3.2rem;font-weight:700;color:{color};line-height:1;text-shadow:0 0 20px {rgba(color,0.4)};">{val*100:.1f}%</div><div style="font-family:Share Tech Mono,monospace;font-size:0.62rem;color:{color};letter-spacing:0.1em;margin-top:0.5rem;">{status}</div><div style="font-family:Share Tech Mono,monospace;font-size:0.56rem;color:{C["text3"]};margin-top:0.25rem;">TGT &lt; 5.0% · {note}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    day_name=['Monday','Tuesday','Wednesday','Thursday','Friday'][dow]
    if rfp>0.10: alert_box(f"CRITICAL ON {day_name.upper()} {shift.upper()}. ADD ~{int(np.ceil((rfp-0.05)*40))} AGENTS. QUEUE {queue} EXCEEDS THRESHOLD.",'danger')
    elif rfp>0.05: alert_box(f"{rfp*100:.1f}% ON {day_name.upper()} {shift.upper()}. CONSIDER +{max(1,int(np.ceil((rfp-0.05)*25)))} AGENT(S).",'warn')
    else: alert_box(f"NOMINAL — {rfp*100:.1f}% ON {day_name.upper()} {shift.upper()}. QUEUE {queue} WITHIN PARAMETERS.",'success')

    section_label("QUEUE DEPTH SENSITIVITY CURVE")
    qr=np.arange(0,71,5); preds=[]
    for q in qr:
        iq=inp.copy(); iq['calls_in_queue']=q; iq['queue_x_monday']=q*int(dow==0); iq['queue_x_night']=q*int(shift=='night')
        preds.append(float(np.clip(rf.predict(engineer_features(iq))[0],0,1))*100)
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=qr,y=preds,mode='lines+markers',
        line=dict(color=C['green'],width=2.5),marker=dict(size=5,color=C['green']),
        fill='tozeroy',fillcolor=rgba(C['green'],0.06),
        hovertemplate='QUEUE %{x} → ABANDON %{y:.1f}%<extra></extra>'))
    fig2.add_hline(y=5.0,line_dash="dash",line_color=C['amber'],opacity=0.5,annotation_text="5% TGT",annotation_font_size=8,annotation_font_color=C['amber'])
    fig2.add_vline(x=queue,line_dash="dot",line_color=C['red'],opacity=0.6,annotation_text=f"NOW: {queue}",annotation_font_size=8,annotation_font_color=C['red'])
    ply(fig2,h=230,xaxis=dict(title='QUEUE DEPTH',gridcolor=C['border']),yaxis=dict(title='PREDICTED ABANDON %',gridcolor=C['border']))
    st.plotly_chart(fig2,use_container_width=True,config={'displayModeBar':False})


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    df,source,err=load_data()
    page,date_range,shifts,teams,agents,years=render_sidebar(df,source,err)
    if page=="HOME": page_home(); return
    df_f=apply_filters(df,date_range,shifts,teams,agents,years)
    if len(df_f)==0: alert_box("NO DATA MATCHES CURRENT FILTERS. EXPAND SELECTIONS.",'warn'); return
    dispatch={"OVERVIEW":page_overview,"TRENDS":page_trends,"TEAMS":page_teams,"AGENTS":page_agents,"PREDICTOR":page_predictor}
    fn=dispatch.get(page)
    if fn: fn(df_f)

if __name__=='__main__':
    main()
