# AtmoKHI

**Karachi AQI Predictor and Analytics System**

Developed by **Murtaza Aamir** as the final project for the **10Pearls Pakistan SHINE Internship Program**.

**AtmoKHI** monitors live air quality parameters in Karachi and delivers automated **72-hour forward-looking AQI forecasts**. The platform features real-time pollutant tracking, interactive trend analysis, model explainability metrics, a custom scenario simulator, and public health advisories.

* **Location:** Karachi, Sindh, Pakistan (24.86°N, 67.02°E)

---

## Live Demo

The production dashboard is hosted on **Streamlit Community Cloud**.

* **App Link:** *Coming soon — will be updated upon deployment*

---

## Features

### Dashboard Tabs
1. **Live & 3-Day Forecast:** Real-time AQI gauge, pollutant cards, 3-day daily summary cards, and an interactive 72-hour forecast trend chart.
2. **Model Metrics & SHAP:** Cross-validation score performance metrics ($R^2$, $RMSE$, $MAE$) and SHAP feature importance interpretability plots.
3. **Custom Scenario Simulator:** Interactive slider interface to run custom pollutant and lag baseline predictions.
4. **Karachi Data Insights:** Exploratory Data Analysis (EDA) charts showcasing diurnal patterns, seasonality, and model validation curves.
5. **Health Guidelines:** Persona-based risk assessment advisories and standard EPA reference matrices.

### Architecture Highlights
* **Automated Data Ingestion:** Scheduled hourly fetching from Open-Meteo and OpenWeather APIs via GitHub Actions.
* **Feature Engineering:** Autoregressive lag constructs (1h, 24h, 48h, 72h) and temporal rolling averages.
* **Predictive ML Modeling:** XGBoost model selected via `TimeSeriesSplit` cross-validation to prevent temporal data leakage.
* **Resilient API Layer:** FastAPI backend service with immediate automated fallback to local `joblib` binaries and GitHub CSV data feeds.

---

## Tech Stack

| Area | Technologies |
|---|---|
| **Language** | Python 3.11 |
| **Machine Learning** | scikit-learn, XGBoost, PyTorch |
| **Explainability** | SHAP |
| **Dashboard** | Streamlit, Plotly |
| **Backend API** | FastAPI, Uvicorn |
| **Automation & CI** | GitHub Actions |
| **Data Sources** | Open-Meteo Air Quality API, OpenWeather API |
| **Feature Store** | Hopsworks |

---

## Project Structure

```text
AtmoKHI/
├── feature_pipeline/     # Hourly fetch, feature engineering, and ingestion
├── training_pipeline/    # Model training scripts, binaries, and metric logs
├── web_app/              # Streamlit dashboard (app.py) & FastAPI service (api.py)
├── data/                 # Automated dataset storage (aqi_features.csv)
├── notebooks/            # Exploratory analysis & static visualization outputs
└── .github/workflows/    # Scheduled cron automation pipelines