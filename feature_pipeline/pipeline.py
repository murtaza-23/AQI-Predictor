import tempfile
import os
import hopsworks
from dotenv import load_dotenv
from fetch_data import fetch_all_data
from parse_features import compute_features

load_dotenv()

def get_feature_group():

    cert_folder = os.path.join(os.getcwd(), ".hopsworks_certs")

    os.makedirs(cert_folder, exist_ok=True)

    project = hopsworks.login(
        project='aqi_predictor_23',
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )

    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=True,
        description="Hourly AQI features - Karachi"
    )
    return fg

def run_pipeline():

    try:
        print(f"Fetching raw data...")
        raw_data = fetch_all_data()
        print(f"AQI data fetched successfully!")

        print(f"Parsing and computing features...")
        df = compute_features(raw_data)
        print(f"Features computed successfully!")

        print(f"Storing data in Hopsworks...")
        fg = get_feature_group()
        print(df.shape)
        print(df.head())

        fg.insert(df.head(1))

        print(f"Data stored in Hopsworks successfully!")

        print(f"AQI = {df.aqi.iloc[0]}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_pipeline()


