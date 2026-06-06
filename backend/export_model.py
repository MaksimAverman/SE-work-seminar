"""
Export trained model for the ICU Early Warning System web app.
Trains LGBM_balanced on filtered data, saves model + scaler + metadata.
"""
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score
import lightgbm as lgb

# === Paths ===
# backend/export_model.py -> BACKEND_DIR = .../avoda/backend, PROJECT_ROOT = .../avoda
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'COMPLETE_ICU_RISK_DATASET.csv'
MODELS_DIR = BACKEND_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# === Load & Filter ===
df = pd.read_csv(DATA_PATH)
target_col = 'HOSPITAL_EXPIRE_FLAG'

meas_cols = ['DiasBP_mean','HeartRate_mean','SysBP_mean','DiasBP_min','HeartRate_min',
             'SysBP_min','DiasBP_max','HeartRate_max','SysBP_max','Creatinine_max','Lactate_max']
# === Outlier cleaning ===
# Physiologically impossible values (sensor artifacts, 0 = missing reading,
# negatives, and absurd maxima like SysBP=127105) are set to NaN so they get
# imputed with the median below instead of poisoning the features.
valid_ranges = {
    'HeartRate_mean': (20, 300), 'HeartRate_min': (20, 300), 'HeartRate_max': (20, 300),
    'SysBP_mean': (30, 300),     'SysBP_min': (30, 300),     'SysBP_max': (30, 300),
    'DiasBP_mean': (10, 200),    'DiasBP_min': (10, 200),    'DiasBP_max': (10, 200),
    'Creatinine_max': (0.1, 50), 'Lactate_max': (0.1, 40),
}
n_cleaned = 0
for col, (lo, hi) in valid_ranges.items():
    bad = (df[col] < lo) | (df[col] > hi)
    n_cleaned += int(bad.sum())
    df.loc[bad, col] = np.nan
print(f"Outlier values set to NaN: {n_cleaned:,}")

df['n_missing'] = df[meas_cols].isna().sum(axis=1)
df = df[df['AGE'] >= 3]
df = df[df['n_missing'] <= 5]
print(f"Filtered cohort: {len(df):,} patients")

# === Preprocessing ===
df['INTIME'] = pd.to_datetime(df['INTIME'], errors='coerce')
df['GENDER_M'] = (df['GENDER'] == 'M').astype(int)
for col in ['DiasBP_mean', 'HeartRate_mean', 'SysBP_mean', 'Creatinine_max', 'Lactate_max']:
    df[col + '_missing'] = df[col].isna().astype(int)
fill_cols = ['DiasBP_mean','HeartRate_mean','SysBP_mean','DiasBP_min','HeartRate_min',
             'SysBP_min','DiasBP_max','HeartRate_max','SysBP_max','Creatinine_max','Lactate_max']
for col in fill_cols:
    df[col] = df[col].fillna(df[col].median())

# === Feature Engineering ===
df['BP_range'] = df['DiasBP_max'] - df['DiasBP_min']
df['HR_range'] = df['HeartRate_max'] - df['HeartRate_min']
df['SBP_range'] = df['SysBP_max'] - df['SysBP_min']
df['shock_index'] = df['HeartRate_mean'] / df['SysBP_mean'].replace(0, np.nan)
df['shock_index'] = df['shock_index'].fillna(df['shock_index'].median())
df['pulse_pressure'] = df['SysBP_mean'] - df['DiasBP_mean']
df['MAP'] = df['DiasBP_mean'] + (df['pulse_pressure'] / 3)
df['age_x_hr'] = df['AGE'] * df['HeartRate_mean']
df['age_x_sbp'] = df['AGE'] * df['SysBP_mean']
df['shock_index_high'] = (df['shock_index'] > 0.9).astype(int)
df['age_over_75'] = (df['AGE'] > 75).astype(int)
df['map_low'] = (df['MAP'] < 65).astype(int)
df['tachycardia'] = (df['HeartRate_mean'] > 100).astype(int)
df['hypotension'] = (df['SysBP_min'] < 90).astype(int)
df['admit_hour'] = df['INTIME'].dt.hour
df['admit_dayofweek'] = df['INTIME'].dt.dayofweek
df['is_night'] = ((df['admit_hour'] >= 22) | (df['admit_hour'] <= 6)).astype(int)
df['is_weekend'] = (df['admit_dayofweek'] >= 5).astype(int)

features = [
    'DiasBP_mean','HeartRate_mean','SysBP_mean',
    'DiasBP_min','HeartRate_min','SysBP_min',
    'DiasBP_max','HeartRate_max','SysBP_max',
    'AGE','GENDER_M',
    'BP_range','HR_range','SBP_range',
    'shock_index','pulse_pressure','MAP',
    'age_x_hr','age_x_sbp',
    'DiasBP_mean_missing','HeartRate_mean_missing','SysBP_mean_missing',
    'shock_index_high','age_over_75','map_low','tachycardia','hypotension',
    'admit_hour','admit_dayofweek','is_night','is_weekend',
    'Creatinine_max','Lactate_max','Creatinine_max_missing','Lactate_max_missing',
]

# === Train ===
X = df[features]; y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=features)

imb_ratio = (y_train == 0).sum() / (y_train == 1).sum()
model = lgb.LGBMClassifier(is_unbalance=True, n_estimators=300, max_depth=6,
                            learning_rate=0.05, random_state=42, verbose=-1)
model.fit(X_train_s, y_train)

# === Evaluate ===
y_proba = model.predict_proba(X_test_s)[:, 1]
auc = roc_auc_score(y_test, y_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
j_scores = tpr - fpr
opt_thr = float(thresholds[np.argmax(j_scores)])
y_pred_opt = (y_proba >= opt_thr).astype(int)
sens = recall_score(y_test, y_pred_opt)
mf1 = f1_score(y_test, y_pred_opt, average='macro')

print(f"AUC: {auc:.4f}")
print(f"Optimal threshold (Youden): {opt_thr:.4f}")
print(f"Sensitivity at threshold: {sens:.4f}")
print(f"Macro F1: {mf1:.4f}")

# === Feature ranges for validation ===
ranges = {}
raw_inputs = ['AGE','HeartRate_mean','HeartRate_min','HeartRate_max',
              'SysBP_mean','SysBP_min','SysBP_max',
              'DiasBP_mean','DiasBP_min','DiasBP_max',
              'Creatinine_max','Lactate_max']
for col in raw_inputs:
    ranges[col] = {'min': float(df[col].min()), 'max': float(df[col].max()),
                   'median': float(df[col].median()), 'mean': float(df[col].mean().round(1))}

# === Save ===
joblib.dump(model, MODELS_DIR / 'model.joblib')
joblib.dump(scaler, MODELS_DIR / 'scaler.joblib')

meta = {
    'features': features,
    'threshold': opt_thr,
    'auc': round(auc, 4),
    'sensitivity': round(sens, 4),
    'macro_f1': round(mf1, 4),
    'model_type': 'LGBMClassifier',
    'n_train': int(X_train.shape[0]),
    'n_test': int(X_test.shape[0]),
    'input_ranges': ranges,
}
with open(MODELS_DIR / 'model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\nSaved to", MODELS_DIR)
print("  model.joblib")
print("  scaler.joblib")
print("  model_meta.json")
print("DONE")
