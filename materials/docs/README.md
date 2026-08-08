# ICU Early Warning System

Predicts ICU patient **deterioration / mortality risk** from early clinical indicators
(heart rate, blood pressure, age, and optional labs). A LightGBM model is served through
a Flask web app with a clinical-style UI.

## Layout

| Folder         | Contents                                                        |
|----------------|-----------------------------------------------------------------|
| `backend/`     | Flask API (`app.py`), training script (`export_model.py`), saved model artifacts (`models/`), `requirements.txt` |
| `frontend/`    | UI — `templates/index.html`, `static/app.js`, `static/style.css` |
| `data/`        | Raw dataset (`COMPLETE_ICU_RISK_DATASET.csv`)                    |
| `notebooks/`   | EDA & experiments                                                |
| `images/`      | Generated figures                                               |
| `experiments/` | Experiment logs (CSV)                                            |
| `docs/`        | Documentation                                                   |

## Quick start

```bash
pip install -r backend/requirements.txt
cd backend
python app.py        # -> http://localhost:5000
```

## Retrain the model

```bash
cd backend
python export_model.py   # rewrites backend/models/{model,scaler,model_meta}.joblib/json
```

## Current model

LightGBM (cleaned data + labs): **AUC 0.786**, sensitivity 0.73, macro-F1 0.58.

## How the model connects to the UI

See **[docs/MODEL_INTEGRATION.md](docs/MODEL_INTEGRATION.md)** — full data-flow, the
`/predict` request/response contract, and step-by-step recipes for adding inputs,
retraining, and swapping the model.

> ⚠️ Research use only. Not validated for clinical use.
