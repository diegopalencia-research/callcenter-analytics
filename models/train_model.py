"""
train_model.py
Trains two regression models to predict abandon_rate.
  - LinearRegression   (interpretable baseline)
  - RandomForestRegressor (higher accuracy)

Saves: models/regressor.pkl + models/reg_metrics.json
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, 'callcenter_data.csv')
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
MODEL_PATH   = os.path.join(MODEL_DIR, 'regressor.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'reg_metrics.json')

os.makedirs(MODEL_DIR, exist_ok=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for abandon rate prediction."""
    f = pd.DataFrame(index=df.index)
    le = LabelEncoder()

    f['day_of_week']       = df['day_of_week']
    f['is_monday']         = df['is_monday']
    f['is_night_shift']    = df['is_night_shift']
    f['shift_enc']         = le.fit_transform(df['shift'])
    f['calls_in_queue']    = df['calls_in_queue']
    f['aht_seconds']       = df['aht_seconds']
    f['experience_months'] = df['experience_months']
    f['calls_handled']     = df['calls_handled']
    f['csat_score']        = df['csat_score']
    f['fcr_rate']          = df['fcr_rate']

    # Interaction features
    f['queue_x_monday']   = df['calls_in_queue'] * df['is_monday']
    f['queue_x_night']    = df['calls_in_queue'] * df['is_night_shift']

    return f


def train():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    print(f"  Rows: {len(df):,}")

    X = engineer_features(df)
    y = df['abandon_rate']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # ── Baseline: mean prediction ──────────────────────────────────────────
    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    print(f"\nBaseline RMSE (mean predictor): {baseline_rmse:.4f}")

    # ── Linear Regression ─────────────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_r2   = r2_score(y_test, lr_pred)
    print(f"Linear Regression  — RMSE: {lr_rmse:.4f} | R²: {lr_r2:.3f}")

    # ── Random Forest ─────────────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=8,
        min_samples_leaf=3, random_state=42
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae  = mean_absolute_error(y_test, rf_pred)
    rf_r2   = r2_score(y_test, rf_pred)

    cv = cross_val_score(rf, X, y, cv=5, scoring='r2')
    print(f"Random Forest      — RMSE: {rf_rmse:.4f} | R²: {rf_r2:.3f} | CV R²: {cv.mean():.3f} ± {cv.std():.3f}")

    # ── Feature importance ────────────────────────────────────────────────
    importance = dict(zip(X.columns, [round(float(v), 6) for v in rf.feature_importances_]))
    top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 features:")
    for feat, imp in top5:
        print(f"  {feat:25s} {imp:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    metrics = {
        'baseline_rmse': round(float(baseline_rmse), 4),
        'lr_rmse':       round(float(lr_rmse), 4),
        'lr_r2':         round(float(lr_r2), 4),
        'rf_rmse':       round(float(rf_rmse), 4),
        'rf_mae':        round(float(rf_mae), 4),
        'rf_r2':         round(float(rf_r2), 4),
        'cv_mean':       round(float(cv.mean()), 4),
        'cv_std':        round(float(cv.std()), 4),
        'n_features':    int(X.shape[1]),
        'train_size':    int(len(X_train)),
        'test_size':     int(len(X_test)),
        'feature_importances': importance,
        'feature_names': X.columns.tolist(),
        'lr_coefficients': dict(zip(X.columns, [round(float(v), 6) for v in lr.coef_])),
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'rf': rf, 'lr': lr}, f)

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved → {MODEL_PATH}")
    print(f"Saved → {METRICS_PATH}")
    print("Done.")

    return rf, lr, metrics


if __name__ == '__main__':
    train()
