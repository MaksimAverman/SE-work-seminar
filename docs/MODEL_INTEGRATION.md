# Model ↔ Frontend Integration Guide

This document explains **how the trained ML model is wired to the web frontend** in the
ICU Early Warning System: the project layout, the request/response contract, the full
data-flow from a browser form to a risk score, and step-by-step recipes for common
changes (adding an input, retraining, swapping the model).

---

## 1. Project structure

```
avoda/
├── backend/                     # All Python: API + training + artifacts
│   ├── app.py                   # Flask server — serves the UI and the /predict API
│   ├── export_model.py          # Trains the model and writes the artifacts below
│   ├── requirements.txt         # Backend dependencies
│   └── models/                  # Saved model artifacts (consumed by app.py)
│       ├── model.joblib         #   trained LightGBM classifier
│       ├── scaler.joblib        #   StandardScaler fitted on the training set
│       └── model_meta.json      #   feature order, threshold, metrics, input ranges
│
├── frontend/                    # All UI assets (no Python)
│   ├── templates/
│   │   └── index.html           # The single-page form + results layout
│   └── static/
│       ├── app.js               # Collects form data, calls /predict, renders results
│       └── style.css            # Styling
│
├── data/
│   └── COMPLETE_ICU_RISK_DATASET.csv   # Raw training data (MIMIC-derived)
│
├── notebooks/                   # Exploratory analysis & experiments
│   ├── ICU_Risk_Analysis.ipynb
│   ├── ICU_Risk_Analysis_v2.ipynb
│   └── ICU_Risk_Final.ipynb
│
├── images/                      # Generated figures (fig_*.png, viz_*.png)
├── experiments/                 # Experiment logs (experiment_log*.csv)
└── docs/
    └── MODEL_INTEGRATION.md      # (this file)
```

**Key idea:** the backend and frontend live in **separate folders**, but Flask is
configured to find the frontend at startup (see §3). The model is **not** called from
the browser — the browser only talks to the `/predict` HTTP endpoint, and `app.py` owns
the model.

---

## 2. Architecture at a glance

```
 ┌─────────────┐     POST /predict (JSON)      ┌──────────────────────────────┐
 │   Browser   │ ────────────────────────────▶ │           app.py             │
 │ index.html  │                               │                              │
 │  + app.js   │                               │  1. validate inputs          │
 │             │                               │  2. compute_features()       │
 │             │ ◀──────────────────────────── │  3. scaler.transform()       │
 └─────────────┘     JSON result               │  4. model.predict_proba()    │
                                               │  5. threshold + alerts       │
                                               └──────────────────────────────┘
                                                          │  loads at startup
                                                          ▼
                                               backend/models/{model,scaler,meta}
```

The model artifacts are loaded **once** when the Flask process starts, and reused for
every request. There is no per-request disk I/O for the model.

---

## 3. How Flask connects to the separate `frontend/` folder

By default Flask looks for `templates/` and `static/` *next to* `app.py`. Because we
moved the UI into `frontend/`, `app.py` points Flask at the right locations explicitly:

```python
# backend/app.py
from pathlib import Path
from flask import Flask

BACKEND_DIR  = Path(__file__).resolve().parent          # .../avoda/backend
PROJECT_ROOT = BACKEND_DIR.parent                       # .../avoda
MODELS_DIR   = BACKEND_DIR / 'models'
FRONTEND_DIR = PROJECT_ROOT / 'frontend'

app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR / 'templates'),    # render_template('index.html')
    static_folder=str(FRONTEND_DIR / 'static'),         # serves /static/*
    static_url_path='/static',
)
```

- `render_template('index.html')` resolves to `frontend/templates/index.html`.
- `index.html` references `/static/style.css` and `/static/app.js`; Flask serves those
  from `frontend/static/` because of `static_url_path='/static'`.
- All paths are **absolute** (derived from `__file__`), so the app runs correctly no
  matter what the current working directory is.

The model artifacts are loaded the same way — by absolute path:

```python
model  = joblib.load(MODELS_DIR / 'model.joblib')
scaler = joblib.load(MODELS_DIR / 'scaler.joblib')
with open(MODELS_DIR / 'model_meta.json') as f:
    meta = json.load(f)
```

---

## 4. The `/predict` contract

### Request — `POST /predict`, `Content-Type: application/json`

| Field                | Type   | Required | Notes                                        |
|----------------------|--------|----------|----------------------------------------------|
| `age`                | number | ✅       | years                                        |
| `gender`             | string | –        | `"M"` / `"F"` (default `"M"`)                 |
| `heart_rate_mean`    | number | ✅       | bpm                                          |
| `heart_rate_min`     | number | ✅       | bpm                                          |
| `heart_rate_max`     | number | ✅       | bpm                                          |
| `systolic_bp_mean`   | number | ✅       | mmHg                                         |
| `systolic_bp_min`    | number | ✅       | mmHg                                         |
| `systolic_bp_max`    | number | ✅       | mmHg                                         |
| `diastolic_bp_mean`  | number | ✅       | mmHg                                         |
| `diastolic_bp_min`   | number | ✅       | mmHg                                         |
| `diastolic_bp_max`   | number | ✅       | mmHg                                         |
| `admit_hour`         | int    | –        | 0–23 (default 12)                            |
| `admit_dayofweek`    | int    | –        | 0=Mon … 6=Sun (default 2)                    |
| `lactate_max`        | number | –        | **optional** lab; blank → cohort median used |
| `creatinine_max`     | number | –        | **optional** lab; blank → cohort median used |

Missing **required** fields return `400` with `{"error": "Missing required field: ..."}`.

### Response — `200 OK`, JSON

```jsonc
{
  "risk_score": 0.933,                 // model probability (0–1)
  "risk_percent": "93.3%",
  "risk_level": "CRITICAL",            // LOW | MODERATE | HIGH | CRITICAL
  "risk_color": "#c0392b",
  "threshold": 0.4361,                 // decision threshold from model_meta.json
  "prediction": "At Risk of Deterioration",
  "is_at_risk": true,
  "key_factors": [                     // top contributing features for this patient
    {"name": "Age × Heart Rate", "feature": "age_x_hr", "value": 9200, "importance": 0.12}
  ],
  "clinical_alerts": [                 // human-readable warnings
    {"icon": "🔴", "text": "Elevated lactate (6.5 mmol/L) — ...", "level": "critical"}
  ],
  "computed_indicators": {             // derived values shown in the UI
    "shock_index": 1.35, "MAP": 61.7, "pulse_pressure": 35.0,
    "HR_range": 50.0, "SBP_range": 40.0, "BP_range": 25.0
  }
}
```

---

## 5. End-to-end data flow (request lifecycle)

1. **Browser collects inputs** — `frontend/static/app.js` reads the form on submit and
   builds the JSON body. Optional labs are sent **only when filled**:

   ```js
   const lactate = document.getElementById('lactate_max').value;
   if (lactate !== '') data.lactate_max = parseFloat(lactate);
   ```

2. **Validation** — `app.py /predict` checks required fields and casts numeric types.
   Empty-string labs are normalized to `None`.

3. **Feature engineering** — `compute_features(data)` turns ~10 raw inputs into the
   **exact 35 features** the model expects, in the right order. This includes:
   - derived values (`shock_index`, `pulse_pressure`, `MAP`, `age_x_hr`, …),
   - clinical flags (`tachycardia`, `hypotension`, `map_low`, …),
   - missing-indicator flags (`Lactate_max_missing`, …),
   - median imputation for blank labs (medians read from `model_meta.json`).

   > ⚠️ **The single most important integration rule:** the feature computation in
   > `app.py` must mirror the training pipeline in `export_model.py` *exactly* — same
   > formulas, same order. The order is enforced by reindexing to `meta['features']`:
   > `X = pd.DataFrame([feature_dict])[FEATURES]`.

4. **Scaling** — `scaler.transform(X)` applies the same `StandardScaler` that was fit
   during training.

5. **Inference** — `model.predict_proba(X_scaled)[0, 1]` gives the risk probability;
   it is compared against `THRESHOLD` (from `model_meta.json`) to set `is_at_risk`.

6. **Enrichment** — the server attaches `key_factors`, `clinical_alerts`, and
   `computed_indicators`, then returns JSON.

7. **Rendering** — `app.js` `displayResults()` animates the gauge, fills the alert list,
   the key-factor bars, and the computed-indicator grid.

---

## 6. Recipe: add a new input field end-to-end

Suppose you want to add **Temperature** as a model feature. Touch these five places, in
order:

1. **Training** (`backend/export_model.py`)
   - clean it in `valid_ranges` (e.g. `'Temp': (30, 45)`),
   - add a `_missing` indicator + median fill if it can be absent,
   - add `'Temp'` to the `features` list,
   - add it to `raw_inputs` so its median lands in `model_meta.json`,
   - **re-run** `python export_model.py` to regenerate the artifacts.

2. **Backend feature build** (`backend/app.py → compute_features`)
   - read `data['temp']`, add it (and any derived/flag features) to `feature_dict`
     using the **same formula** as training.

3. **Backend validation / parsing** (`backend/app.py → predict`)
   - add it to `required` (if mandatory) or handle blank → median (if optional).

4. **Frontend form** (`frontend/templates/index.html`)
   - add an `<input id="temp" name="temp" ...>` inside a card.

5. **Frontend payload** (`frontend/static/app.js`)
   - add `temp: parseFloat(document.getElementById('temp').value)` to the `data` object.

Mismatch between steps 1 and 2 (different formula or missing feature) is the usual cause
of wrong predictions — keep them in sync.

---

## 7. Recipe: retrain or swap the model

The frontend and `/predict` API are **model-agnostic** — they only depend on
`model_meta.json` (feature list, threshold, input ranges) and the `predict_proba`
interface. To retrain or replace the model:

```bash
cd backend
python export_model.py     # rewrites models/model.joblib, scaler.joblib, model_meta.json
```

Then restart the Flask process. The UI badge AUC, the decision threshold, and the lab
imputation medians all update automatically because they are read from
`model_meta.json` at startup. To use a *different* algorithm, change the estimator in
`export_model.py` — as long as it exposes `predict_proba` and you keep the same feature
list, nothing else needs to change.

> The current model is a `LGBMClassifier` (LightGBM). `app.py` reads
> `model.feature_importances_` for the "key factors" panel — any tree model exposes
> this. If you switch to a model without `feature_importances_` (e.g. plain logistic
> regression), update `get_key_factors()` accordingly.

---

## 8. Running the app

```bash
# from the project root
pip install -r backend/requirements.txt
cd backend
python app.py
# -> http://localhost:5000
```

`app.py` runs with `debug=True` on port 5000 for development. For production, serve it
behind a WSGI server, e.g.:

```bash
cd backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app     # (Linux/macOS; pip install gunicorn)
```

---

## 9. Notes & gotchas

- **Notebook paths:** the notebooks were written when the CSV sat next to them. After
  the reorg the data lives in `../data/`. If you re-run a notebook, update its
  `pd.read_csv(...)` path to `../data/COMPLETE_ICU_RISK_DATASET.csv`.
- **Outlier cleaning:** `export_model.py` replaces physiologically impossible values
  (sensor artifacts, e.g. `SysBP=127105`, negatives, zeros) with `NaN` before median
  imputation, via the `valid_ranges` dict. Keep these ranges consistent with the input
  `min`/`max` you advertise in the UI.
- **Single source of truth:** `model_meta.json` is the contract between training and
  serving. Never hand-edit it — regenerate it via `export_model.py`.
- **Research use only:** this tool is not validated for clinical use.
```
