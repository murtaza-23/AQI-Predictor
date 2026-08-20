"""
export_hopsworks.py
-------------------
Reads the 'aqi_features' Feature Group from Hopsworks and exports it to
data/aqi_features.csv so the Streamlit app can use it.

Strategy:
  1. Try reading the FULL feature group via Arrow Flight (fast path).
  2. If Arrow Flight fails (timeout / gRPC error) fall back to a
     regular Spark/HSFS batch read with a longer timeout.
  3. Guarantee the CSV is always sorted by timestamp and deduplicated.
"""

import os
import sys
import time
import traceback

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_NAME = "aqi_predictor_23"
HOST         = "eu-west.cloud.hopsworks.ai"
PORT         = 443
FG_NAME      = "aqi_features"
FG_VERSION   = 1

# Path is relative to the repo root (script is run from feature_pipeline/)
OUTPUT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "aqi_features.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _login():
    """Login to Hopsworks and return the project object."""
    import hopsworks

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY secret is missing — add it to GitHub Secrets.")

    cert_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hopsworks_certs")
    os.makedirs(cert_folder, exist_ok=True)

    print(f"Logging in to Hopsworks project '{PROJECT_NAME}'...")
    project = hopsworks.login(
        project=PROJECT_NAME,
        host=HOST,
        port=PORT,
        api_key_value=api_key,
        cert_folder=cert_folder,
    )
    print("Login successful.")
    return project


def _read_with_arrow_flight(fg, timeout_secs=600):
    """Primary read path: Arrow Flight (fast, streaming). Returns a DataFrame."""
    print(f"Reading feature group via Arrow Flight (timeout={timeout_secs}s)...")
    df = fg.read(
        dataframe_type="pandas",
        read_options={
            "arrow_flight_config": {
                "timeout": timeout_secs,
            }
        },
    )
    print(f"Arrow Flight read complete: {len(df)} rows retrieved.")
    return df


def _read_with_batch(fg):
    """Fallback read path: standard HSFS batch read (no Arrow Flight)."""
    print("Falling back to standard batch read (Arrow Flight unavailable)...")
    df = fg.read(dataframe_type="pandas")
    print(f"Batch read complete: {len(df)} rows retrieved.")
    return df


def _clean_and_save(df):
    """Sort by timestamp, deduplicate, and write CSV."""
    import pandas as pd

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp"], keep="last")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows → {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_feature_group_to_csv(max_retries=2):
    project = _login()
    fs = project.get_feature_store()

    print(f"Fetching feature group: {FG_NAME} v{FG_VERSION}")
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    df = None
    last_error = None

    for attempt in range(1, max_retries + 1):
        print(f"\n--- Read attempt {attempt}/{max_retries} ---")
        try:
            df = _read_with_arrow_flight(fg, timeout_secs=600)
            break  # success
        except Exception as exc:
            last_error = exc
            print(f"Arrow Flight read failed on attempt {attempt}: {exc}")
            traceback.print_exc()

            # Try the non-Arrow-Flight fallback once
            if attempt == max_retries:
                print("All Arrow Flight attempts failed. Trying standard batch read...")
                try:
                    df = _read_with_batch(fg)
                    break
                except Exception as fallback_exc:
                    last_error = fallback_exc
                    print(f"Standard batch read also failed: {fallback_exc}")
                    traceback.print_exc()
            else:
                wait = 30 * attempt
                print(f"Waiting {wait}s before retry...")
                time.sleep(wait)

    if df is None or df.empty:
        print(f"\nERROR: Could not read any data from Hopsworks. Last error: {last_error}")
        sys.exit(1)

    _clean_and_save(df)
    print("\nExport complete.")


if __name__ == "__main__":
    export_feature_group_to_csv()
