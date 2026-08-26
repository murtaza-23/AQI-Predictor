import os
import joblib
import numpy as np
import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "training_pipeline", "models", "best_model.pkl")

GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "murtaza-23/HawaNama/main/data/aqi_features.csv"
)

app = FastAPI(
    title="HawaName - Karachi AQI Predictor API",
    description="Real-time Air Quality Monitoring and 72-Hour AQI Forecasting Engine for Karachi",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

FEATURES = [
    "pm2_5", "pm10", "o3", "no2", "co", "so2",
    "hour", "day", "day_of_week", "month", "is_weekend", "hour_category",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h",
    "aqi_change_prev_hour", "aqi_rolling_3h", "aqi_rolling_24h",
    "pm2_5_lag_1h", "pm2_5_lag_24h", "pm10_lag_1h", "pm10_lag_24h",
    "o3_lag_1h", "o3_lag_24h", "no2_lag_1h", "no2_lag_24h",
    "co_lag_1h", "co_lag_24h", "so2_lag_1h", "so2_lag_24h"
]

CUTOFF = pd.Timestamp("2026-08-12 18:00:00")

@app.on_event("startup")
def load_model():
    global model
    import platform

    if platform.system() == "Windows":
        # use local pickle file (avoids Hopsworks Windows issues)
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded locally (Windows dev mode): {MODEL_PATH}")
        else:
            print(f"WARNING: no local model found at {MODEL_PATH}")
    else:
        # Linux (Render deployment) then load from Hopsworks Model Registry
        try:
            cert_folder = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), ".hopsworks_certs"
            )
            os.makedirs(cert_folder, exist_ok=True)
            import hopsworks
            project = hopsworks.login(
                project="aqi_predictor_23",
                host="eu-west.cloud.hopsworks.ai",
                port=443,
                api_key_value=os.getenv("HOPSWORKS_API_KEY"),
                cert_folder=cert_folder,
            )
            mr = project.get_model_registry()
            hw_model = mr.get_model("aqi_predictor")
            model_dir = hw_model.download()
            model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
            print("Model loaded from Hopsworks Model Registry")
        except Exception as e:
            print(f"Hopsworks load failed, falling back to local: {e}")
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)

LOCAL_CSV_PATH = os.path.join(BASE_DIR, "data", "aqi_features.csv")

def load_latest_data() -> pd.DataFrame:
    df = None
    # Try loading directly from GitHub repository (live deployment mode)
    try:
        response = requests.get(GITHUB_CSV_URL, timeout=8)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
    except Exception as e:
        print(f"GitHub CSV fetch fallback to local file: {e}")
        # Local Windows fallback
        if os.path.exists(LOCAL_CSV_PATH):
            df = pd.read_csv(LOCAL_CSV_PATH)
        else:
            raise HTTPException(status_code=503, detail=f"Failed to load data: {e}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    is_clean_hour = ((df["timestamp"].dt.minute == 0) & (df["timestamp"].dt.second == 0))
    is_new_live = df["timestamp"] >= CUTOFF
    df = df[is_clean_hour | is_new_live].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["aqi_change_prev_hour"] = df["aqi"].shift(1) - df["aqi"].shift(2)

    df["aqi_lag_1h"] = df["aqi"].shift(1)
    df["aqi_lag_3h"] = df["aqi"].shift(3)
    df["aqi_lag_24h"] = df["aqi"].shift(24)
    df["aqi_lag_48h"] = df["aqi"].shift(48)
    df["aqi_lag_72h"] = df["aqi"].shift(72)

    df["aqi_rolling_3h"] = df["aqi"].shift(1).rolling(3).mean()
    df["aqi_rolling_24h"] = df["aqi"].shift(1).rolling(24).mean()

    for col in ["pm2_5", "pm10", "o3", "no2", "co", "so2"]:
        df[f"{col}_lag_1h"] = df[col].shift(1)
        df[f"{col}_lag_24h"] = df[col].shift(24)

    df = df.dropna().reset_index(drop=True)

    return df 

def get_hour_category(h: int) -> int:
    if 6 <= h < 10: return 1
    if 10 <= h < 17: return 2
    if 17 <= h < 21: return 3
    return 0  

def get_aqi_category(aqi: float) -> dict:
    if aqi <= 50:
        return {"label": "Good", "color": "#00e400", "emoji": "🟢"}
    elif aqi <= 100:
        return {"label": "Moderate", "color": "#ffff00", "emoji": "🟡"}
    elif aqi <= 150:
        return {"label": "Unhealthy for Sensitive Groups", "color": "#ff7e00", "emoji": "🟠"}
    elif aqi <= 200:
        return {"label": "Unhealthy", "color": "#ff0000", "emoji": "🔴"}
    elif aqi <= 300:
        return {"label": "Very Unhealthy", "color": "#8f3f97", "emoji": "🟣"}
    else:
        return {"label": "Hazardous", "color": "#7e0023", "emoji": "⚫"}

def fetch_open_meteo_hourly_forecast() -> pd.DataFrame:
    try:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": 24.860753,
            "longitude": 67.029503,
            "hourly": ["us_aqi", "pm2_5", "pm10", "ozone", "nitrogen_dioxide", "carbon_monoxide", "sulphur_dioxide"],
            "forecast_days": 3,
            "timezone": "UTC"
        }
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()["hourly"]
        df_f = pd.DataFrame(data)
        df_f["timestamp"] = pd.to_datetime(df_f["time"])
        return df_f
    except Exception as e:
        print(f"Open-Meteo forecast fetch warning (fallback to historical profile): {e}")
        return None

def build_hourly_pollutant_profile(df: pd.DataFrame) -> dict:
    engineered = df.copy()
    engineered["hour"] = pd.to_datetime(engineered["timestamp"]).dt.hour
    profile = engineered.groupby("hour")[
        ["pm2_5", "pm10", "o3", "no2", "co", "so2"]
    ].mean().to_dict(orient="index")
    return profile

# Recursive 72 hour forecast with dynamic pollutant updating
def recursive_forecast(df: pd.DataFrame, hours: int = 72) -> list:
    engineered = engineer_features(df)
    aqi_history = list(engineered["aqi"].values[-72:])
    
    pm2_5_history = list(engineered["pm2_5"].values[-24:])
    pm10_history = list(engineered["pm10"].values[-24:])
    o3_history = list(engineered["o3"].values[-24:])
    no2_history = list(engineered["no2"].values[-24:])
    co_history = list(engineered["co"].values[-24:])
    so2_history = list(engineered["so2"].values[-24:])

    hourly_profile = build_hourly_pollutant_profile(df)
    df_f = fetch_open_meteo_hourly_forecast()
    
    last_timestamp = engineered["timestamp"].iloc[-1]
    forecasts = []

    for h in range(hours):
        future_timestamp = last_timestamp + timedelta(hours=h + 1)
        future_hour = future_timestamp.hour

        # Try live 72h pollutant forecast from Open-Meteo first, else use historical profile
        pm2_5, pm10, o3, no2, co, so2 = None, None, None, None, None, None
        if df_f is not None and not df_f.empty:
            match_row = df_f[df_f["timestamp"] >= future_timestamp]
            if not match_row.empty:
                curr_p = match_row.iloc[0]
                pm2_5 = float(curr_p["pm2_5"])
                pm10 = float(curr_p["pm10"])
                o3 = float(curr_p["ozone"])
                no2 = float(curr_p["nitrogen_dioxide"])
                co = float(curr_p["carbon_monoxide"])
                so2 = float(curr_p["sulphur_dioxide"])

        if pm2_5 is None:
            profile_p = hourly_profile.get(future_hour, {})
            pm2_5 = float(profile_p.get("pm2_5", pm2_5_history[-1]))
            pm10 = float(profile_p.get("pm10", pm10_history[-1]))
            o3 = float(profile_p.get("o3", o3_history[-1]))
            no2 = float(profile_p.get("no2", no2_history[-1]))
            co = float(profile_p.get("co", co_history[-1]))
            so2 = float(profile_p.get("so2", so2_history[-1]))

        aqi_1h = aqi_history[-1]
        aqi_3h = aqi_history[-3] if len(aqi_history) >= 3 else aqi_1h
        aqi_24h = aqi_history[-24] if len(aqi_history) >= 24 else aqi_1h
        aqi_48h = aqi_history[-48] if len(aqi_history) >= 48 else aqi_1h
        aqi_72h = aqi_history[-72] if len(aqi_history) >= 72 else aqi_1h
        aqi_change = aqi_1h - (aqi_history[-2] if len(aqi_history) >= 2 else aqi_1h)
        roll_3h = float(np.mean(aqi_history[-3:]))
        roll_24h = float(np.mean(aqi_history[-24:]))

        row = {
            "pm2_5": pm2_5, "pm10": pm10, "o3": o3, "no2": no2, "co": co, "so2": so2,
            "hour": future_timestamp.hour, "day": future_timestamp.day,
            "day_of_week": future_timestamp.dayofweek, "month": future_timestamp.month,
            "is_weekend": int(future_timestamp.dayofweek >= 5),
            "hour_category": get_hour_category(future_timestamp.hour),
            "aqi_lag_1h": aqi_1h, "aqi_lag_3h": aqi_3h,
            "aqi_lag_24h": aqi_24h, "aqi_lag_48h": aqi_48h, "aqi_lag_72h": aqi_72h,
            "aqi_change_prev_hour": aqi_change,
            "aqi_rolling_3h": roll_3h, "aqi_rolling_24h": roll_24h,
            "pm2_5_lag_1h": pm2_5_history[-1], "pm2_5_lag_24h": pm2_5_history[-24] if len(pm2_5_history) >= 24 else pm2_5_history[-1],
            "pm10_lag_1h": pm10_history[-1], "pm10_lag_24h": pm10_history[-24] if len(pm10_history) >= 24 else pm10_history[-1],
            "o3_lag_1h": o3_history[-1], "o3_lag_24h": o3_history[-24] if len(o3_history) >= 24 else o3_history[-1],
            "no2_lag_1h": no2_history[-1], "no2_lag_24h": no2_history[-24] if len(no2_history) >= 24 else no2_history[-1],
            "co_lag_1h": co_history[-1], "co_lag_24h": co_history[-24] if len(co_history) >= 24 else co_history[-1],
            "so2_lag_1h": so2_history[-1], "so2_lag_24h": so2_history[-24] if len(so2_history) >= 24 else so2_history[-1],
        }

        feature_row = pd.DataFrame([row])[FEATURES]
        predicted_aqi = float(model.predict(feature_row)[0])
        predicted_aqi = max(0, min(500, round(predicted_aqi, 1)))

        aqi_history.append(predicted_aqi)
        pm2_5_history.append(pm2_5)
        pm10_history.append(pm10)
        o3_history.append(o3)
        no2_history.append(no2)
        co_history.append(co)
        so2_history.append(so2)

        forecasts.append({
            "timestamp": future_timestamp.strftime("%Y-%m-%d %H:%M"),
            "predicted_aqi": predicted_aqi,
            "hour": future_timestamp.hour,
            "day": future_timestamp.strftime("%A %b %d"),
        })

    return forecasts


@app.get("/")
def root():
    return {"name": "HawaNama — Karachi AQI Predictor API",
            "version": "1.0.0",
            "endpoints": ["/health", "/current", "/forecast", "/forecast/daily", "/history"]}


@app.get("/health")
def health():
    return {"status": "ok",
            "model_loaded": model is not None,
            "timestamp": datetime.utcnow().isoformat()}


@app.get("/current")
def get_current():
    df = load_latest_data()
    latest = df.iloc[-1]
    aqi = float(latest["aqi"])
    category = get_aqi_category(aqi)
    return {
        "timestamp": str(latest["timestamp"]),
        "aqi": aqi,
        "category": category["label"],
        "color": category["color"],
        "emoji": category["emoji"],
        "pollutants": {
            "pm2_5": float(latest["pm2_5"]), "pm10": float(latest["pm10"]),
            "o3": float(latest["o3"]), "no2": float(latest["no2"]),
            "co": float(latest["co"]), "so2": float(latest["so2"]),
        }
    }

@app.get("/forecast")
def get_forecast(hours: int = 72):
    if not (1 <= hours <= 72):
        raise HTTPException(status_code=400, detail="hours must be 1-72")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = load_latest_data()
    forecasts = recursive_forecast(df, hours=hours)

    for forecast in forecasts:
        category = get_aqi_category(forecast["predicted_aqi"])
        forecast.update(category=category["label"], color=category["color"], emoji=category["emoji"])

    max_aqi = max(forecast["predicted_aqi"] for forecast in forecasts)
    max_category = get_aqi_category(max_aqi)

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "forecast_hours": hours,
        "max_predicted_aqi": max_aqi,
        "max_category": max_category["label"],
        "has_hazard_alert": max_aqi > 150,
        "forecasts": forecasts,
    }


@app.get("/forecast/daily")
def get_daily_forecast():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = load_latest_data()
    forecasts = recursive_forecast(df, hours=72)

    f_df = pd.DataFrame(forecasts)
    f_df["timestamp"] = pd.to_datetime(f_df["timestamp"])
    f_df["date"] = f_df["timestamp"].dt.date

    daily = f_df.groupby("date").agg(
        avg_aqi=("predicted_aqi", "mean"),
        max_aqi=("predicted_aqi", "max"),
        min_aqi=("predicted_aqi", "min"),
    ).reset_index()

    result = []
    for _, row in daily.iterrows():
        category = get_aqi_category(row["avg_aqi"])
        result.append({
            "date": str(row["date"]), 
            "avg_aqi": round(row["avg_aqi"], 1),
            "max_aqi": round(row["max_aqi"], 1),
            "min_aqi": round(row["min_aqi"], 1),
            "category": category["label"],
            "emoji": category["emoji"],
            "color": category["color"],
        })
    return {"daily_forecast": result}

@app.get("/history")
def get_history(hours: int = 168):
    df = load_latest_data()
    recent = df.tail(hours)[["timestamp", "aqi", "pm2_5", "pm10", "o3", "no2", "co", "so2"]].copy()
    recent["timestamp"] = recent["timestamp"].astype(str)
    records = recent.to_dict(orient="records")
    for r in records:
        category = get_aqi_category(r["aqi"])
        r["category"] = category["label"]
        r["emoji"] = category["emoji"]
    return {"history": records, "count": len(records)}

METRICS_JSON_PATH = os.path.join(BASE_DIR, "training_pipeline", "models", "metrics.json")
METRICS_TXT_PATH  = os.path.join(BASE_DIR, "training_pipeline", "models", "metrics.txt")

@app.get("/model/info")
def get_model_info():
    import json
    model_name = "XGBoost"
    rmse = 0.7300
    mae = 0.4890
    r2 = 0.9922
    trained_at = None

    # Try reading structured JSON metrics
    if os.path.exists(METRICS_JSON_PATH):
        try:
            with open(METRICS_JSON_PATH, "r") as f:
                data = json.load(f)
                return {
                    "model_name": data.get("best_model_name", "XGBoost Regressor"),
                    "features_count": len(FEATURES),
                    "features": FEATURES,
                    "metrics": {
                        "r2": data.get("r2", r2),
                        "rmse": data.get("rmse", rmse),
                        "mae": data.get("mae", mae),
                        "cv_folds": 5,
                        "cross_val_strategy": "TimeSeriesSplit",
                        "trained_at": data.get("trained_at")
                    },
                    "model_comparison": data.get("model_comparison", [
                        {"model": "XGBoost", "rmse": round(data.get("rmse", rmse), 4), "mae": round(data.get("mae", mae), 4), "r2": round(data.get("r2", r2), 4), "status": "Best Model"}
                    ])
                }
        except Exception as e:
            print(f"Error reading metrics.json: {e}")

    # Fallback to reading text metrics
    if os.path.exists(METRICS_TXT_PATH):
        try:
            with open(METRICS_TXT_PATH, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "Best model:" in line: model_name = line.split(":")[-1].strip()
                    elif "RMSE:" in line: rmse = float(line.split(":")[-1].strip())
                    elif "MAE:" in line: mae = float(line.split(":")[-1].strip())
                    elif "R²:" in line: r2 = float(line.split(":")[-1].strip())
                    elif "Trained at:" in line: trained_at = line.split("Trained at:")[-1].strip()
        except Exception as e:
            print(f"Error parsing metrics.txt: {e}")

    return {
        "model_name": model_name,
        "features_count": len(FEATURES),
        "features": FEATURES,
        "metrics": {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "cv_folds": 5,
            "cross_val_strategy": "TimeSeriesSplit",
            "trained_at": trained_at
        },
        "model_comparison": [
            {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2, "status": "Best Model"}
        ]
    }

class PredictRequest(BaseModel):
    pm2_5: float = 25.0
    pm10: float = 50.0
    o3: float = 40.0
    no2: float = 15.0
    co: float = 300.0
    so2: float = 10.0
    hour: int = 14
    aqi_lag_1h: float = 70.0
    aqi_lag_24h: float = 75.0

@app.post("/predict/custom")
def predict_custom(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    h = req.hour
    h_cat = 1 if 6 <= h < 10 else (2 if 10 <= h < 17 else (3 if 17 <= h < 21 else 0))
    is_w = 0

    row = {
        "pm2_5": req.pm2_5, "pm10": req.pm10, "o3": req.o3, "no2": req.no2,
        "co": req.co, "so2": req.so2,
        "hour": h, "day": 15, "day_of_week": 2, "month": 8, "is_weekend": is_w,
        "hour_category": h_cat,
        "aqi_lag_1h": req.aqi_lag_1h, "aqi_lag_3h": req.aqi_lag_1h,
        "aqi_lag_24h": req.aqi_lag_24h, "aqi_lag_48h": req.aqi_lag_24h, "aqi_lag_72h": req.aqi_lag_24h,
        "aqi_change_prev_hour": 0.0,
        "aqi_rolling_3h": req.aqi_lag_1h, "aqi_rolling_24h": req.aqi_lag_24h,
        "pm2_5_lag_1h": req.pm2_5, "pm2_5_lag_24h": req.pm2_5,
        "pm10_lag_1h": req.pm10, "pm10_lag_24h": req.pm10,
        "o3_lag_1h": req.o3, "o3_lag_24h": req.o3,
        "no2_lag_1h": req.no2, "no2_lag_24h": req.no2,
        "co_lag_1h": req.co, "co_lag_24h": req.co,
        "so2_lag_1h": req.so2, "so2_lag_24h": req.so2,
    }

    feature_row = pd.DataFrame([row])[FEATURES]
    predicted_aqi = float(model.predict(feature_row)[0])
    predicted_aqi = max(0, min(500, round(predicted_aqi, 1)))
    category = get_aqi_category(predicted_aqi)

    return {
        "predicted_aqi": predicted_aqi,
        "category": category["label"],
        "color": category["color"],
        "emoji": category["emoji"]
    }