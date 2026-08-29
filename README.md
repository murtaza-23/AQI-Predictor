# AtmoKHI

**Karachi AQI Predictor and Analytics System**

**AtmoKHI** monitors live air quality parameters in Karachi and delivers automated **72-hour forward-looking AQI forecasts**. The platform features real-time pollutant tracking, interactive trend analysis, model explainability metrics, a custom scenario simulator, and public health advisories.

* **Location:** Karachi, Sindh, Pakistan (24.86°N, 67.02°E)

---

## Live Demo

The production dashboard and REST API are deployed live:

* **Streamlit Dashboard:** [[atmokhi.streamlit.app](https://atmokhi.streamlit.app)](https://atmokhi.streamlit.app/)

---

## Features

### Dashboard Tabs
1. **Live & 3-Day Forecast:** Real-time AQI gauge, pollutant cards, 3-day daily summary cards, and an interactive 72-hour forecast trend chart.
2. **Model Metrics & SHAP:** Cross-validation performance metrics ($R^2$, $RMSE$, $MAE$) and SHAP feature importance interpretability plots.
3. **Custom Scenario Simulator:** Interactive slider interface to run custom pollutant and lag baseline predictions.
4. **Karachi Data Insights:** Exploratory Data Analysis (EDA) charts showcasing diurnal patterns, seasonality, and model validation curves.
5. **Health Guidelines:** Persona-based risk assessment advisories and standard EPA reference matrices.

### Architecture Highlights
* **Automated Data Ingestion:** Scheduled hourly fetching from Open-Meteo and OpenWeather APIs via GitHub Actions.
* **Feature Store Integration:** Centralized feature management and historical sync using Hopsworks.
* **Feature Engineering:** Autoregressive lag constructs (1h, 24h, 48h, 72h) and temporal rolling averages.
* **Predictive ML Modeling:** XGBoost model selected via `TimeSeriesSplit` cross-validation to prevent temporal data leakage.
* **Resilient API Layer:** Decoupled FastAPI backend hosted on Render with instant fallback to local `joblib` binaries and cached data to handle free-tier cold starts seamlessly.

---

## Tech Stack

| Area | Technologies |
|---|---|
| **Language** | Python 3.11 |
| **Machine Learning** | scikit-learn, XGBoost, PyTorch |
| **Explainability** | SHAP |
| **Dashboard** | Streamlit, Plotly, Altair |
| **Backend API** | FastAPI, Uvicorn |
| **Feature Store** | Hopsworks |
| **Automation & CI** | GitHub Actions |
| **Data Sources** | Open-Meteo Air Quality API, OpenWeather API |
| **Hosting** | Streamlit Community Cloud (UI), Render (API) |

---

## Project Structure

```text
AtmoKHI/
├── feature_pipeline/    # Hourly fetch, feature engineering, and Hopsworks ingestion
├── training_pipeline/   # Model training scripts, binaries, and metric logs
├── web_app/             # Streamlit dashboard (app.py) & FastAPI service (api.py)
├── data/                # Automated dataset storage (aqi_features.csv)
├── notebooks/           # Exploratory analysis & static visualization outputs
├── .github/workflows/   # Scheduled cron automation pipelines
└── requirements.txt     # Core ML and API dependencies
