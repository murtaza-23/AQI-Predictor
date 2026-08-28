# AtmoKHI

**Karachi AQI Predictor and Analytics**

Hey! I'm Murtaza Aamir, and this is my final project for the **10Pearls Pakistan SHINE Internship**.

**AtmoKHI** tracks live air quality in Karachi and predicts AQI up to **72 hours** ahead. The app shows current pollutant readings, a 3-day forecast, model stats, a scenario simulator, EDA charts, and basic health guidance.

Location: Karachi (24.86°N, 67.02°E)

**GitHub:** [murtaza-23/AtmoKHI](https://github.com/murtaza-23/AtmoKHI)

---

## Live Demo (Streamlit Cloud)

The dashboard will be hosted on **Streamlit Community Cloud**.

**App link:** _Coming soon - will be added here after deployment_

Example format once live:
`https://atmokhi-karachi-aqi.streamlit.app`

> The cloud app reads from the GitHub CSV and local metrics files. If you run the full API + forecast stack locally, see Manual Setup below.

---

## What the app does

**Dashboard (5 tabs)**

1. **Live and 3-Day Forecast** - AQI gauge, pollutant cards, daily summary, 72-hour trend chart
2. **Model Metrics and SHAP** - validation scores and feature importance plot
3. **Custom Scenario Simulator** - change pollutant sliders and get a prediction
4. **Karachi Data Insights** - EDA plots from the notebook and training run
5. **Health Guidelines** - EPA-style advice based on current AQI

**Behind the scenes**

- Hourly data fetch (GitHub Actions) from Open-Meteo + OpenWeather
- Feature engineering with lag features (1h, 24h, 48h, 72h)
- XGBoost model (also tried Ridge, Random Forest, LSTM)
- TimeSeriesSplit cross-validation so we don't leak future data into training
- Optional FastAPI backend for `/current`, `/forecast`, and custom predictions
- Hopsworks feature store + model registry on Linux/CI (local pickle on Windows)

---

## Tech stack

| Area | Tools |
|------|-------|
| Language | Python 3.11 |
| ML | scikit-learn, XGBoost, PyTorch (LSTM baseline) |
| Explainability | SHAP |
| Dashboard | Streamlit, Plotly |
| API | FastAPI, Uvicorn |
| Automation | GitHub Actions |
| Data | Open-Meteo Air Quality API, OpenWeather |
| MLOps (CI/Linux) | Hopsworks |

---

## Project structure

```text
AtmoKHI/
├── feature_pipeline/     # hourly fetch + feature parsing
├── training_pipeline/    # train.py, models/, plots/
├── web_app/              # app.py (Streamlit) + api.py (FastAPI)
├── data/                 # aqi_features.csv (updated by CI)
├── notebooks/            # EDA + static plots
└── .github/workflows/    # hourly fetch, daily training, backfill
```

---

## Manual setup (run on your machine)

### 1. Clone and install

```bash
git clone https://github.com/murtaza-23/AtmoKHI.git
cd AtmoKHI
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -r web_app/requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root:

```env
OPENWEATHER_API_KEY=your_key_here
```

Pollution and AQI mostly come from **Open-Meteo** (no key needed). OpenWeather is used for temperature, humidity, and wind in the live pipeline.

For Hopsworks on Linux/CI, also add:

```env
HOPSWORKS_API_KEY=your_key_here
```

### 3. Train the model (important)

`best_model.pkl` is not in git (it's in `.gitignore`). After cloning, run training once:

```bash
python training_pipeline/train.py
```

This creates:

- `training_pipeline/models/best_model.pkl`
- `training_pipeline/models/metrics.json`
- `training_pipeline/plots/` (SHAP, validation charts)

EDA plots in `notebooks/plots/` are already in the repo.

### 4. Run the dashboard

**Streamlit only (simplest):**

```bash
cd web_app
streamlit run app.py
```

The app will try the FastAPI URL first, then fall back to the local CSV and model if the API is offline.

**Full setup (API + Streamlit):**

```bash
# Terminal 1
cd web_app
uvicorn api:app --reload --port 8000

# Terminal 2
cd web_app
streamlit run app.py
```

### 5. Streamlit Cloud secrets (when deploying)

In the Streamlit Cloud app settings, add secrets if you host the API separately:

```toml
API_URL = "https://your-api-url.com"
```

If you only deploy the Streamlit app, the built-in fallback (GitHub CSV + metrics.json) still works for most of the UI.

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (main branch).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick the repo, and set:
   - **Main file path:** `web_app/app.py`
   - **Python version:** 3.11
4. Add any secrets under **Advanced settings** if needed.
5. Deploy. Copy the public URL and paste it in the **Live Demo** section above.

**Note:** Make sure `metrics.json` and plot PNGs are committed. The `.pkl` model file is gitignored, so either commit a small trained model for demo purposes or run training in CI and document that cloud users see CSV-based current AQI until the model artifact is available.

---

## GitHub Actions

| Workflow | When | What it does |
|----------|------|--------------|
| `feature_pipeline.yml` | Every hour | Fetches data, updates `data/aqi_features.csv`, commits to main |
| `model_training.yml` | Daily (2 AM UTC) | Retrains model, commits metrics and plots |
| `backfill.yml` | Manual | One-time historical backfill |

Commits from Actions use `[skip ci]` so they don't loop forever.

---

## Model notes

- **Target:** predict AQI 1 hour ahead (`aqi_next_1h`)
- **Best model:** XGBoost (picked via TimeSeriesSplit CV)
- **Features:** 32 total (pollutants, time of day, AQI lags, rolling averages)
- **72h forecast:** recursive predictions in the API, with Open-Meteo pollutant forecasts for future hours

Latest scores live in `training_pipeline/models/metrics.json`.

---

## Known quirks

- Project used to be called HawaNama. Everything is **AtmoKHI** now.
- Early live rows from OpenWeather had messy timestamps. Training filters those out.
- Hopsworks is awkward on Windows, so local dev uses CSV + pickle files.

---

## Author

**Murtaza Aamir**

- [LinkedIn](https://www.linkedin.com/in/murtaza-aamir)
- [GitHub](https://github.com/murtaza-23/AtmoKHI)

Built for the 10Pearls Pakistan SHINE Internship Program.
