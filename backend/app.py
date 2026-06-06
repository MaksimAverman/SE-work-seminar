"""
ICU Early Warning System — Flask Backend
Predicts deterioration risk from early clinical indicators.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

# === Paths ===
# backend/app.py  ->  BACKEND_DIR = .../avoda/backend ,  PROJECT_ROOT = .../avoda
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = BACKEND_DIR / 'models'
FRONTEND_DIR = PROJECT_ROOT / 'frontend'

# Templates and static assets live in the separate frontend/ folder.
app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR / 'templates'),
    static_folder=str(FRONTEND_DIR / 'static'),
    static_url_path='/static',
)


def to_python(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    return obj

# Load model artifacts
model = joblib.load(MODELS_DIR / 'model.joblib')
scaler = joblib.load(MODELS_DIR / 'scaler.joblib')
with open(MODELS_DIR / 'model_meta.json', 'r') as f:
    meta = json.load(f)

FEATURES = meta['features']
THRESHOLD = meta['threshold']
# Median lab values used to impute when the user leaves a lab field blank.
CREATININE_MEDIAN = meta['input_ranges']['Creatinine_max']['median']
LACTATE_MEDIAN = meta['input_ranges']['Lactate_max']['median']


def compute_features(data):
    """Compute all 31 model features from raw patient inputs."""
    hr_mean = data['heart_rate_mean']
    hr_min = data['heart_rate_min']
    hr_max = data['heart_rate_max']
    sbp_mean = data['systolic_bp_mean']
    sbp_min = data['systolic_bp_min']
    sbp_max = data['systolic_bp_max']
    dbp_mean = data['diastolic_bp_mean']
    dbp_min = data['diastolic_bp_min']
    dbp_max = data['diastolic_bp_max']
    age = data['age']
    gender_m = 1 if data.get('gender', 'M') == 'M' else 0
    admit_hour = data.get('admit_hour', 12)
    admit_dow = data.get('admit_dayofweek', 2)

    # Labs: optional. Blank -> impute with cohort median and flag as missing.
    creat = data.get('creatinine_max')
    lact = data.get('lactate_max')
    creat_missing = 1 if creat is None else 0
    lact_missing = 1 if lact is None else 0
    creat = CREATININE_MEDIAN if creat is None else float(creat)
    lact = LACTATE_MEDIAN if lact is None else float(lact)

    # Derived
    bp_range = dbp_max - dbp_min
    hr_range = hr_max - hr_min
    sbp_range = sbp_max - sbp_min
    shock_index = hr_mean / sbp_mean if sbp_mean > 0 else 1.0
    pulse_pressure = sbp_mean - dbp_mean
    map_val = dbp_mean + (pulse_pressure / 3)
    age_x_hr = age * hr_mean
    age_x_sbp = age * sbp_mean

    # Clinical flags
    shock_index_high = 1 if shock_index > 0.9 else 0
    age_over_75 = 1 if age > 75 else 0
    map_low = 1 if map_val < 65 else 0
    tachycardia = 1 if hr_mean > 100 else 0
    hypotension = 1 if sbp_min < 90 else 0
    is_night = 1 if (admit_hour >= 22 or admit_hour <= 6) else 0
    is_weekend = 1 if admit_dow >= 5 else 0

    feature_dict = {
        'DiasBP_mean': dbp_mean, 'HeartRate_mean': hr_mean, 'SysBP_mean': sbp_mean,
        'DiasBP_min': dbp_min, 'HeartRate_min': hr_min, 'SysBP_min': sbp_min,
        'DiasBP_max': dbp_max, 'HeartRate_max': hr_max, 'SysBP_max': sbp_max,
        'AGE': age, 'GENDER_M': gender_m,
        'BP_range': bp_range, 'HR_range': hr_range, 'SBP_range': sbp_range,
        'shock_index': shock_index, 'pulse_pressure': pulse_pressure, 'MAP': map_val,
        'age_x_hr': age_x_hr, 'age_x_sbp': age_x_sbp,
        'DiasBP_mean_missing': 0, 'HeartRate_mean_missing': 0, 'SysBP_mean_missing': 0,
        'shock_index_high': shock_index_high, 'age_over_75': age_over_75,
        'map_low': map_low, 'tachycardia': tachycardia, 'hypotension': hypotension,
        'admit_hour': admit_hour, 'admit_dayofweek': admit_dow,
        'is_night': is_night, 'is_weekend': is_weekend,
        'Creatinine_max': creat, 'Lactate_max': lact,
        'Creatinine_max_missing': creat_missing, 'Lactate_max_missing': lact_missing,
    }

    computed = {
        'shock_index': round(shock_index, 2),
        'MAP': round(map_val, 1),
        'pulse_pressure': round(pulse_pressure, 1),
        'HR_range': round(hr_range, 1),
        'SBP_range': round(sbp_range, 1),
        'BP_range': round(bp_range, 1),
    }

    return feature_dict, computed


def generate_alerts(data, computed):
    """Generate clinical alert messages."""
    alerts = []
    if data['heart_rate_mean'] > 100:
        alerts.append({'icon': '⚠️', 'text': f"Tachycardia detected (HR = {data['heart_rate_mean']} bpm)", 'level': 'warning'})
    if data['heart_rate_mean'] < 50:
        alerts.append({'icon': '⚠️', 'text': f"Bradycardia detected (HR = {data['heart_rate_mean']} bpm)", 'level': 'warning'})
    if data['systolic_bp_min'] < 90:
        alerts.append({'icon': '🔴', 'text': f"Severe hypotension (SBP min = {data['systolic_bp_min']} mmHg)", 'level': 'critical'})
    if computed['shock_index'] > 0.9:
        alerts.append({'icon': '🔴', 'text': f"Elevated Shock Index ({computed['shock_index']}) — possible compensated shock", 'level': 'critical'})
    elif computed['shock_index'] > 0.7:
        alerts.append({'icon': '⚠️', 'text': f"Shock Index approaching elevated ({computed['shock_index']})", 'level': 'warning'})
    if computed['MAP'] < 65:
        alerts.append({'icon': '🔴', 'text': f"Low MAP ({computed['MAP']} mmHg) — risk of organ hypoperfusion", 'level': 'critical'})
    if data['age'] > 75:
        alerts.append({'icon': 'ℹ️', 'text': f"Age > 75 — reduced physiological reserve", 'level': 'info'})
    if computed['HR_range'] > 40:
        alerts.append({'icon': '⚠️', 'text': f"High heart rate variability (range = {computed['HR_range']}) — hemodynamic instability", 'level': 'warning'})
    if computed['pulse_pressure'] < 25:
        alerts.append({'icon': '⚠️', 'text': f"Narrow pulse pressure ({computed['pulse_pressure']} mmHg) — possible reduced cardiac output", 'level': 'warning'})
    if data.get('lactate_max') is not None and float(data['lactate_max']) > 4:
        alerts.append({'icon': '🔴', 'text': f"Elevated lactate ({data['lactate_max']} mmol/L) — tissue hypoperfusion / possible sepsis", 'level': 'critical'})
    if data.get('creatinine_max') is not None and float(data['creatinine_max']) > 2:
        alerts.append({'icon': '⚠️', 'text': f"Elevated creatinine ({data['creatinine_max']} mg/dL) — renal dysfunction", 'level': 'warning'})
    return alerts


def get_key_factors(feature_dict, feature_importances):
    """Identify top contributing factors for this prediction."""
    factors = []
    imp = dict(zip(FEATURES, feature_importances))

    indicator_labels = {
        'age_x_hr': 'Age × Heart Rate',
        'AGE': 'Age',
        'age_x_sbp': 'Age × Systolic BP',
        'shock_index': 'Shock Index (HR/SBP)',
        'SysBP_min': 'Min Systolic BP',
        'HeartRate_mean': 'Mean Heart Rate',
        'HR_range': 'Heart Rate Variability',
        'MAP': 'Mean Arterial Pressure',
        'SBP_range': 'Systolic BP Variability',
        'DiasBP_min': 'Min Diastolic BP',
        'pulse_pressure': 'Pulse Pressure',
        'HeartRate_max': 'Max Heart Rate',
        'SysBP_mean': 'Mean Systolic BP',
        'BP_range': 'Diastolic BP Variability',
        'HeartRate_min': 'Min Heart Rate',
        'Lactate_max': 'Max Lactate',
        'Creatinine_max': 'Max Creatinine',
    }

    sorted_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    for feat, importance in sorted_feats[:7]:
        if feat in indicator_labels:
            val = feature_dict[feat]
            factors.append({
                'name': indicator_labels[feat],
                'feature': feat,
                'value': round(val, 2) if isinstance(val, float) else val,
                'importance': round(importance, 4),
            })
    return factors


@app.route('/')
def index():
    return render_template('index.html', meta=meta)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate required fields
        required = ['age', 'heart_rate_mean', 'heart_rate_min', 'heart_rate_max',
                     'systolic_bp_mean', 'systolic_bp_min', 'systolic_bp_max',
                     'diastolic_bp_mean', 'diastolic_bp_min', 'diastolic_bp_max']
        for field in required:
            if field not in data or data[field] is None:
                return jsonify({'error': f'Missing required field: {field}'}), 400
            data[field] = float(data[field])

        data['admit_hour'] = int(data.get('admit_hour', 12))
        data['admit_dayofweek'] = int(data.get('admit_dayofweek', 2))
        data['gender'] = data.get('gender', 'M')

        # Optional labs: treat blank/empty as missing.
        for lab in ['creatinine_max', 'lactate_max']:
            if data.get(lab) in (None, ''):
                data[lab] = None

        # Compute features
        feature_dict, computed = compute_features(data)

        # Build feature vector
        X = pd.DataFrame([feature_dict])[FEATURES]
        X_scaled = scaler.transform(X)

        # Predict
        risk_score = float(model.predict_proba(X_scaled)[0, 1])
        prediction = risk_score >= THRESHOLD

        # Risk level
        if risk_score >= 0.6:
            risk_level, risk_color = 'CRITICAL', '#c0392b'
        elif risk_score >= THRESHOLD:
            risk_level, risk_color = 'HIGH', '#e67e22'
        elif risk_score >= 0.25:
            risk_level, risk_color = 'MODERATE', '#f1c40f'
        else:
            risk_level, risk_color = 'LOW', '#27ae60'

        # Key factors
        fi = model.feature_importances_
        key_factors = get_key_factors(feature_dict, fi)

        # Clinical alerts
        alerts = generate_alerts(data, computed)

        return jsonify(to_python({
            'risk_score': round(risk_score, 4),
            'risk_percent': f"{risk_score * 100:.1f}%",
            'risk_level': risk_level,
            'risk_color': risk_color,
            'threshold': THRESHOLD,
            'prediction': 'At Risk of Deterioration' if prediction else 'Low Risk',
            'is_at_risk': prediction,
            'key_factors': key_factors,
            'clinical_alerts': alerts,
            'computed_indicators': computed,
        }))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"Model: {meta['model_type']} (AUC={meta['auc']}, threshold={meta['threshold']:.4f})")
    print(f"Starting ICU Early Warning System on http://localhost:5000")
    app.run(debug=True, port=5000)
