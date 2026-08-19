import os
from dotenv import load_dotenv

load_dotenv()

import hopsworks

def export_feature_group_to_csv():

    cert_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".hopsworks_certs"
        )
    os.makedirs(cert_folder, exist_ok=True)

    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY is not available in the environment."
        )
    
    print("Connecting to Hopsworks...")

    project = hopsworks.login(
        project='aqi_predictor_23',
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=api_key,
        cert_folder=cert_folder
    )

    print("Connected to Hopsworks!")

    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name="aqi_features",
        version=1
    )

    print("Reading feature group...")

    df = fg.read(
        dataframe_type="pandas",
        read_options={
            "arrow_flight_config": {
                "timeout": 900
            }
        }
    )      

    print(f"Retrived {len(df)} rows")

    os.makedirs("../data", exist_ok=True)

    output_path = "../data/aqi_features.csv"

    df.to_csv(output_path, index=False)

    print(f"Exported feature data to {output_path}")

if __name__ == "__main__":
    export_feature_group_to_csv()

