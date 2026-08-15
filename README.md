# ICU Early Warning System

Predicts ICU patient **deterioration / mortality risk** from early clinical indicators
(heart rate, blood pressure, age, and optional labs). A LightGBM model powers two
front-ends that share the same inference code in `backend/`:

- a **desktop clinical decision-support app** (`app.py`, built with customtkinter) — the
  main interface, with login, patient dashboard, priority triage, and per-patient risk
  explanations;
- an optional **Flask web app** (`backend/app.py` + `frontend/`) exposing a `/predict` API.

## Run the desktop app (main UI)

```bash
pip install -r backend/requirements.txt
pip install customtkinter Pillow
python app.py
```

Run it from the repository root so the `backend` package is importable. On first launch it
creates a local SQLite database (`icu_system.db`, git-ignored). Load patients with the
**Upload Patients CSV** button, using `materials/data/COMPLETE_ICU_RISK_DATASET.csv`.

Sign in with the **Doctor / Nurse / Admin** demo buttons on the login screen, or the seeded
accounts:

| Role   | Username  | Password    |
|--------|-----------|-------------|
| Doctor | `doctor1` | `doctor123` |
| Nurse  | `nurse1`  | `nurse123`  |
| Admin  | `admin1`  | `admin123`  |

## Run the web app (Flask)

```bash
pip install -r backend/requirements.txt
cd backend
python app.py        # -> http://localhost:5000
```

## Layout

| Path                         | Contents                                                                 |
|------------------------------|--------------------------------------------------------------------------|
| `app.py`                     | Desktop GUI (customtkinter): login, dashboard, triage, risk explanation  |
| `backend/`                   | Model + Flask API: `app.py`, `export_model.py`, `models/`, `requirements.txt` |
| `frontend/`                  | Web UI for the Flask app — `templates/`, `static/`                       |
| `tests/`                     | pytest suite (`test_system.py`, `conftest.py`)                           |
| `materials/`                 | Dataset, notebooks, and docs (`data/`, `notebooks/`, `docs/`)            |
| `images/`                    | Generated figures and UI icons                                          |
| `assets/`, `*.html`, `website/` | Project website (GitHub Pages)                                       |
| `data/`, `notebooks/`, `experiments/`, `docs/` | Dataset, EDA, experiment logs, and integration docs   |

## Tests

```bash
pip install pytest
python -m pytest
```

## Current model

LightGBM (cleaned data + labs): **AUC 0.786**, sensitivity 0.73, macro-F1 0.58 — 35
engineered features, trained on 41,702 and evaluated on 10,426 ICU stays.

## Retrain the model

```bash
cd backend
python export_model.py   # rewrites backend/models/{model,scaler,model_meta}.joblib/json
```

## How the model connects to the UI

See **[docs/MODEL_INTEGRATION.md](docs/MODEL_INTEGRATION.md)** — full data-flow, the
`/predict` request/response contract, and step-by-step recipes for adding inputs,
retraining, and swapping the model.

> ⚠️ Research use only. Not validated for clinical use.
