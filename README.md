# ICU Early Warning System

Predicts ICU patient **deterioration / mortality risk** from early clinical indicators
(heart rate, blood pressure, age, and optional labs). A LightGBM model is called
directly (in-process, no web server) from a desktop GUI built with CustomTkinter.

## Layout

| Folder         | Contents                                                        |
|----------------|-----------------------------------------------------------------|
| `app.py`       | Desktop GUI (CustomTkinter) — the app's entry point               |
| `backend/`     | Inference module (`app.py`: feature computation, alerts, model loading), training script (`export_model.py`), saved model artifacts (`models/`), `requirements.txt` |
| `frontend/`    | Legacy web UI assets (unused by the desktop app)                 |
| `data/`        | Raw dataset (`COMPLETE_ICU_RISK_DATASET.csv`)                    |
| `notebooks/`   | EDA & experiments                                                |
| `images/`      | Generated figures                                               |
| `experiments/` | Experiment logs (CSV)                                            |
| `docs/`        | Documentation                                                   |

## Quick start

Runs locally as a desktop application — no server, no browser.

```bash
pip install -r backend/requirements.txt
pip install customtkinter
python app.py
```

A login window opens. Demo accounts:

| Username  | Password    | Role   |
|-----------|-------------|--------|
| `doctor1` | `doctor123` | Doctor |
| `nurse1`  | `nurse123`  | Nurse  |

## Retrain the model

```bash
cd backend
python export_model.py   # rewrites backend/models/{model,scaler,model_meta}.joblib/json
```

## Current model

LightGBM (cleaned data + labs): **AUC 0.786**, sensitivity 0.73, macro-F1 0.58.

## How the model connects to the UI

`app.py` imports `compute_features`, `generate_alerts`, `get_risk_drivers`, `model`,
`scaler`, `FEATURES`, and `THRESHOLD` straight from `backend/app.py` and calls them
in-process — no HTTP request involved. See
**[docs/MODEL_INTEGRATION.md](docs/MODEL_INTEGRATION.md)** for the full data-flow and
step-by-step recipes for adding inputs, retraining, and swapping the model.

> ⚠️ Research use only. Not validated for clinical use.
