# Call Center Performance Analytics Dashboard

---

A business intelligence dashboard built around the exact KPIs that call center operations teams track daily (AHT, CSAT, abandon rate, FCR) with a predictive layer that forecasts performance before problems escalate.

Built with Python, scikit-learn, and Streamlit. Deployed on Streamlit Cloud.

---

## What It Does

- **Executive Overview** — 4 KPI cards (AHT, CSAT, abandon rate, FCR) with delta vs previous period and shift-level breakdown
- **Trend Analysis** — Daily time series with configurable rolling averages
- **Agent Performance** — Ranked composite score table with tier classification (Top / Mid / Risk) and coaching flags
- **ML Predictor** — Random Forest model forecasts tomorrow's abandon rate from operational inputs

---

## Key Findings

- **Queue × Monday interaction** is the strongest predictor of abandon rate (feature importance: 0.363) — the backlog effect at week start is the dominant driver
- **Day of week** is the second most important feature (0.254) — temporal patterns outperform agent-level features for abandon rate prediction
- **Queue depth alone** accounts for 0.243 importance — operational implication: staffing decisions should be driven by queue forecasting, not historical averages
- **Monday morning** abandon rates are consistently 40–60% above weekly average
- **Night shift** CSAT is 0.3–0.5 points below morning shift average — training gap or fatigue signal
- **Agent experience** correlates strongly with FCR but plateaus on AHT after 12 months

---

## ML Model

```
Algorithm:     Random Forest Regressor + Linear Regression (baseline)
Target:        Abandon rate (continuous regression)
Features:      12 engineered features
Validation:    5-fold cross-validation
Train/Test:    80/20 split (2,112 train / 528 test)

RF R²:         0.919
RF RMSE:       0.0150
RF MAE:        0.0120
LR R²:         0.922
CV R² (mean):  0.908 ± 0.005
Baseline RMSE: 0.0528
Improvement:   3.5x RMSE reduction vs naive baseline
```

**Top features by importance:**
| Feature | Importance |
|---|---|
| queue × monday interaction | 0.363 |
| day_of_week | 0.254 |
| calls_in_queue | 0.243 |
| is_monday | 0.133 |
| calls_handled | 0.002 |

**All 12 features:** day_of_week · is_monday · is_night_shift · shift_encoded · calls_in_queue · aht_seconds · experience_months · calls_handled · csat_score · fcr_rate · queue×monday interaction · queue×night interaction

---

## Dataset

| Property | Value |
|---|---|
| Rows | 2,640 |
| Agents | 20 |
| Date range | July–December 2024 (6 months) |
| Grain | Agent × day |
| Simulation | Realistic distributions with temporal patterns |

**Simulation logic:** Monday morning backlog effect · Friday afternoon rush · Night shift fatigue penalty · Senior agent FCR/AHT improvement curves · Queue depth → abandon rate correlation

---

## Project Structure

```
callcenter-analytics/
├── app.py                  Main Streamlit application (4 pages)
├── generate_data.py        Dataset simulation script
├── train_model.py          Model training + evaluation
├── kpi_calculator.py       KPI aggregation functions
├── models/
│   ├── regressor.pkl       Trained models (RF + LR, serialized)
│   └── reg_metrics.json    Evaluation metrics
├── callcenter_data.csv     Simulated dataset
├── README.md
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/diegopalencia-research/callcenter-analytics.git
cd callcenter-analytics
pip install -r requirements.txt

python generate_data.py    # creates callcenter_data.csv
python train_model.py      # trains model, saves to models/
streamlit run app.py       # launch dashboard
```

---

## KPI Definitions

| KPI | Definition | Industry Target |
|---|---|---|
| **AHT** | Average Handle Time — talk + hold + wrap-up time per call | < 300 sec (5 min) |
| **CSAT** | Customer Satisfaction Score — post-call survey 1–5 | > 4.2 |
| **Abandon Rate** | % callers who hang up before reaching an agent | < 5% |
| **FCR** | First Call Resolution — % resolved without callback | > 70% |

---

## Research Connection

This dashboard operationalizes findings from:
> *Palencia, D. (2024). Computational Feature Extraction for Human Performance Prediction. OSF Preprints.*

The call center context serves as an empirical domain for testing whether temporal and behavioral operational features can predict quality outcomes at the individual agent level a research question extending the phonological prediction framework from Project 01.

---

**Live App:** https://callcenter-analytics.streamlit.app/ &nbsp;·&nbsp; **GitHub:** https://github.com/diegopalencia-research/callcenter-analytics

## Author

**Diego José Palencia Robles**
*Data Science & NLP Projects — Applied AI & Analytics + Machine Learning*

- GitHub; @diegopalencia-research: https://github.com/diegopalencia-research
- LinkedIn: https://www.linkedin.com/in/diego-jose-palencia-robles/

---

## License

MIT License
