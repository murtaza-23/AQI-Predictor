import os
import sys
import platform
import sqlite3
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
import shap

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "aqi_features.csv")
DB_PATH = os.path.join(BASE_DIR, "feature_store", "aqi_features.db")
MODEL_DIR = os.path.join(BASE_DIR, "training_pipeline", "models")
PLOTS_DIR = os.path.join(BASE_DIR, "training_pipeline", "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Weather data excluded since it was not available on free tier plan and did not have much impact on the AQI
FEATURES = ["pm2_5", "pm10", "o3", "no2", "co", "so2",
            "hour", "day", "day_of_week", "month", "is_weekend", "hour_category",
            "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h",
            "aqi_change_prev_hour", "aqi_rolling_3h", "aqi_rolling_24h",
            "pm2_5_lag_1h", "pm2_5_lag_24h", "pm10_lag_1h", "pm10_lag_24h",
            "o3_lag_1h", "o3_lag_24h", "no2_lag_1h", "no2_lag_24h",
            "co_lag_1h", "co_lag_24h", "so2_lag_1h", "so2_lag_24h"]

TARGET = "aqi_next_1h"

# Load Data
def load_data() -> pd.DataFrame:

    if platform.system() == "Windows":

        if os.path.exists(CSV_PATH):
            print(f"Loading data from CSV: {CSV_PATH}")
            df = pd.read_csv(CSV_PATH)
        else:
            raise FileNotFoundError(
                f"CSV not found at {CSV_PATH}. "
                "Make sure data/aqi_features.csv exists in your repo."
            )

    else:
        import hopsworks
    
        print("Reading data from Hopsworks...")

        cert_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".hopsworks_certs"
        )
        os.makedirs(cert_folder, exist_ok=True)

        api_key = os.getenv("HOPSWORKS_API_KEY")

        if not api_key:
            raise RuntimeError(
                "HOPSWORKS_API_KEY is not available."
            )

        project = hopsworks.login(
            project="aqi_predictor_23",
            host="eu-west.cloud.hopsworks.ai",
            port=443,
            api_key_value=api_key,
            cert_folder=cert_folder,
            engine="python"
        )

        fs = project.get_feature_store()

        fg = fs.get_feature_group(
            name="aqi_features",
            version=1
        )
    
        df = fg.read(
            dataframe_type="pandas",
            read_options={"use_hive": True}
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"\nLoaded {len(df)} rows")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nAQI stats:\nmin={df['aqi'].min()}")
    print(f"max={df['aqi'].max()}")
    print(f"mean={df['aqi'].mean():.1f}")
    return df

# Clean Data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # Remove live pipeline rows fetched from OpenWeather (Aug 8-12) which have inconsistent pollutant data compared to backfill data (Open-Mateo)
    # Backfill rows have clean hourly timestamps (minute=00, second=00) 
    # Live OpenWeather rows have irregular timestamps (e.g. 10:59:39)
        
    before = len(df)

    # from 12th August 2026 18:00:00 onwards i changed my live pollution data fetch from OpenWeather to Open-Mateo (meaning no inconsistency)
    cutoff = pd.Timestamp("2026-08-12 18:00:00")

    is_exact_hour = ((df["timestamp"].dt.minute == 0) & (df["timestamp"].dt.second == 0))

    is_new_live_data = (df["timestamp"] >= cutoff)

    keep_rows = is_exact_hour | is_new_live_data

    df = df[keep_rows].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    removed = before - len(df)

    print(f"\nRemoved {removed} inconsistent live rows")
    print(f"Clean rows for training: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


# Feature engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    # Add aqi_change_rate which required sorted historical data and could not be computed for a single live row so done here at training time.

    df = df.sort_values("timestamp").reset_index(drop=True)

    df["aqi_change_prev_hour"] = df["aqi"].shift(1) - df["aqi"].shift(2)

    # Lag features (previous hour AQI is legitimate predictors as they come from past)
    df["aqi_lag_1h"] = df["aqi"].shift(1)   # 1 hour ago
    df["aqi_lag_3h"] = df["aqi"].shift(3)   # 3 hours ago
    df["aqi_lag_24h"] = df["aqi"].shift(24)  # same hour yesterday
    df["aqi_lag_48h"] = df["aqi"].shift(48)
    df["aqi_lag_72h"] = df["aqi"].shift(72)

    for col in ["pm2_5", "pm10", "o3", "no2", "co", "so2"]:
        df[f"{col}_lag_1h"] = df[col].shift(1)
        df[f"{col}_lag_24h"] = df[col].shift(24)

    # Rolling averages (smooth out noise)
    df["aqi_rolling_3h"] = df["aqi"].shift(1).rolling(3).mean()
    df["aqi_rolling_24h"] = df["aqi"].shift(1).rolling(24).mean()

    # shift target forward by 1 meaning given everything we know NOW, predict AQI in 1 hour
    df["aqi_next_1h"] = df["aqi"].shift(-1)

    df = df.dropna().reset_index(drop=True)

    print(f"\nRows after feature engineering: {len(df)}")

    return df

# Evaluate
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Overfitting check
    train_r2 = r2_score(y_train, model.predict(X_train))
    gap = abs(train_r2 - r2)

    print(f"\n{name}")
    print(f"RMSE: {rmse:.4f}  MAE: {mae:.4f}  R²: {r2:.4f}")
    print(f"Overfit gap: {gap:.4f}")

    return {
        "name": name,
        "model": model,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "preds": preds,
        "y_test": y_test,
    }

# Plot results
def plot_actual_vs_predicted(results: dict, best_name: str):

    best = results[best_name]
    y_test = np.asarray(best["y_test"])
    preds = np.asarray(best["preds"])

    n = min(200, len(y_test))
    x = np.arange(n)

    plt.figure(figsize=(12, 5))

    plt.plot(x, y_test[:n], label="Actual AQI", color="blue", linewidth=1.5)

    plt.plot(x, preds[:n], label=f"Predicted AQI ({best_name})", color="red", linewidth=1.5, linestyle="--")

    plt.title(f"Actual vs Predicted AQI — {best_name} (first {n} test points)", fontsize=13, fontweight="bold")

    plt.xlabel("Validation Time step")
    plt.ylabel("AQI")

    plt.legend()
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "actual_vs_predicted.png")

    plt.savefig(path, dpi=150)

    plt.close()

    print(f"Saved: {path}")


def plot_model_comparison(results: dict):
    names = list(results.keys())
    rmses = [results[n]["rmse"] for n in names]
    maes = [results[n]["mae"] for n in names]
    r2s = [results[n]["r2"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].bar(names, rmses, color=["blue", "green", "red", "purple"])
    axes[0].set_title("RMSE (lower is better)")
    axes[0].set_ylabel("RMSE")

    axes[1].bar(names, maes, color=["blue", "green", "red", "purple"])
    axes[1].set_title("MAE (lower is better)")
    axes[1].set_ylabel("MAE")

    axes[2].bar(names, r2s, color=["blue", "green", "red", "purple"])
    axes[2].set_title("R² (higher is better)")
    axes[2].set_ylabel("R²")

    plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")

    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "model_comparison.png")

    plt.savefig(path, dpi=150)

    plt.close()

    print(f"Saved: {path}")


def plot_shap(model, X_test, feature_names):

    print("\nComputing SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()

    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)

    plt.title("SHAP Feature Importance", fontsize=13, fontweight="bold")

    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "shap_summary.png")

    plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()

    print(f"Saved: {path}")


# Save model
def save_model_locally(model, name: str, metrics: dict):
    import json
    path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(model, path)
    print(f"Model saved locally: {path}")

    # Save metrics as a text file for reference
    metrics_path = os.path.join(MODEL_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Best model: {name}\n")
        f.write(f"RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"MAE: {metrics['mae']:.4f}\n")
        f.write(f"R²: {metrics['r2']:.4f}\n")
        f.write(f"Trained at: {datetime.utcnow()}\n")

    # Save metrics as structured JSON for web app / API ingestion
    json_path = os.path.join(MODEL_DIR, "metrics.json")
    metrics_data = {
        "best_model_name": name,
        "rmse": round(metrics['rmse'], 4),
        "mae": round(metrics['mae'], 4),
        "r2": round(metrics['r2'], 4),
        "trained_at": str(datetime.utcnow()),
        "features": FEATURES,
        "features_count": len(FEATURES),
        "model_comparison": [
            {"model": name, "rmse": round(metrics['rmse'], 4), "mae": round(metrics['mae'], 4), "r2": round(metrics['r2'], 4), "status": "Best Model"}
        ]
    }
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Metrics saved: {metrics_path} and {json_path}")


def save_model_to_hopsworks(model, name: str, metrics: dict):

    cert_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".hopsworks_certs"
    )

    os.makedirs(cert_folder, exist_ok=True)

    import hopsworks

    project = hopsworks.login(
        project="aqi_predictor_23",
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        cert_folder=cert_folder
    )

    # Save locally first then upload
    local_path = os.path.join(MODEL_DIR, "best_model.pkl")

    joblib.dump(model, local_path)

    mr = project.get_model_registry()

    hw_model = mr.sklearn.create_model(
        name="aqi_predictor",
        metrics={
            "rmse": round(metrics["rmse"], 4),
            "mae": round(metrics["mae"],  4),
            "r2": round(metrics["r2"],   4),
        },
        description=f"Best model: {name}. Trained on 1 year Karachi AQI data."
    )

    hw_model.save(local_path)

    print(f"Model saved to Hopsworks registry: aqi_predictor")


def train():
    # Load
    df = load_data()

    # Clean (remove inconsistent OpenWeather live rows)
    df = clean_data(df)

    # Feature engineering
    df = engineer_features(df)

    # Prepare X and y
    X = df[FEATURES]
    y = df[TARGET]

    print(f"\nFeatures: {FEATURES}")
    print(f"Target: {TARGET}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"AQI range in training: {y.min()} to {y.max()}, mean: {y.mean():.1f}")

    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        ),
    }
    
    # Time series split to prevent data leakage
    # Not use future dates to train and test it on past dates (predict past using future)
    # DOES NOT MAKE SENSE!

    print("\nTime-series cross-validation...")

    tscv = TimeSeriesSplit(n_splits=5)

    selection_metrics = {}
    final_results = {}

    final_train_idx = None
    final_val_idx = None

    # Fold 1: low-data warm-up fold, excluded from model selection.
    # Folds 2-4: model selection.
    # Fold 5: final unseen evaluation.

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):

        if fold == 1:
            continue

        print(f"\nFold {fold}: ")
        print(f"train={len(train_idx)}")
        print(f"Test/validation={len(test_idx)}")
    
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if fold == tscv.n_splits:
            final_train_idx = train_idx
            final_test_idx = test_idx

        for name, model in models.items():

            result = evaluate_model(name, model, X_train, X_test, y_train, y_test)

            baseline_predictions = X_test["aqi_lag_1h"].values
            baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
            print(f"Baseline RMSE: {baseline_rmse:.4f}")

            if fold < tscv.n_splits:
                if name not in selection_metrics:
                    selection_metrics[name] = {
                        "rmses": [],
                        "maes": [],
                        "r2s": [],
                        "baseline_rmses": []
                    }
                selection_metrics[name]["rmses"].append(result["rmse"])
                selection_metrics[name]["maes"].append(result["mae"])
                selection_metrics[name]["r2s"].append(result["r2"])
                selection_metrics[name]["baseline_rmses"].append(baseline_rmse)

            else:
                final_results[name] = result
                final_results[name]["baseline_rmse"] = (baseline_rmse)

    selection_summary = {}

    for name, metrics in selection_metrics.items():

        mean_rmse = np.mean(metrics["rmses"])
        std_rmse = np.std(metrics["rmses"])
        mean_mae = np.mean(metrics["maes"])
        mean_r2 = np.mean(metrics["r2s"])
        mean_baseline_rmse = np.mean(metrics["baseline_rmses"])   

        selection_summary[name] = {
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "mean_mae": mean_mae,
            "mean_r2": mean_r2,
            "mean_baseline_rmse": mean_baseline_rmse
        }

        print(f"\n{name}")
        print(f"Mean RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f}")
        print(f"Mean MAE: {mean_mae:.4f}")
        print(f"Mean R²: {mean_r2:.4f}")
        print(f"Mean baseline RMSE: {mean_baseline_rmse:.4f}")    

    # Select best model by mean CV RMSE
    best_name = min(selection_summary, key=lambda name: selection_summary[name]["mean_rmse"])
    print(f"\nBest model from TimeSeriesSplit (folds 2-4): {best_name}")

    best_final = final_results[best_name]

    final_rmse = best_final["rmse"]
    final_mae = best_final["mae"]
    final_r2 = best_final["r2"]
    final_baseline_rmse = best_final["baseline_rmse"]
    final_improvement = ((final_baseline_rmse - final_rmse) / final_baseline_rmse * 100)

    print(f"Selected model: {best_name}")
    print(f"Final RMSE: {final_rmse:.4f}")
    print(f"Final MAE: {final_mae:.4f}")
    print(f"Final R²: {final_r2:.4f}")
    print(f"Final persistence baseline RMSE: {final_baseline_rmse:.4f}")
    print(f"Improvement over persistence baseline: {final_improvement:.1f}%")

    # Train best final model on all available data
    print(f"Retraining {best_name} on all {len(X)} rows")

    best_model = models[best_name]

    best_model.fit(X, y)

    print("model training complete!")

    # Plots
    print("\nGenerating plots...")
    plot_actual_vs_predicted(final_results, best_name)
    plot_model_comparison(final_results)

    # SHAP (only works on tree models)
    shap_model = best_model

    if best_name == "Ridge":
        # Use Random Forest for SHAP if Ridge is the best model out of all
        shap_model = models["RandomForest"]
        shap_model.fit(X, y)
        print("Note: SHAP uses Random Forest because Ridge is not tree-based")

    # Get raw model from Pipeline if needed
    if hasattr(shap_model, "named_steps"):
        shap_model = shap_model.named_steps["model"]

    X_shap = X.iloc[-min(1000, len(X)):]

    plot_shap(shap_model, X_shap, FEATURES)

    final_metrics = {
        "rmse": final_rmse,
        "mae": final_mae,
        "r2": final_r2
    }

    # Save model
    print("\nSaving model...")
    save_model_locally(best_model, best_name, final_metrics)

    if platform.system() != "Windows":
        save_model_to_hopsworks(best_model, best_name, final_metrics)

if __name__ == "__main__":
    train()