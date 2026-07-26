import requests
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

# testing if requests module works
# response = requests.get("https://www.google.com/")
# print(response.status_code)

load_dotenv()

OW_API_KEY = os.getenv("OPENWEATHER_API_KEY") # get the API keys from .env file
AQICN_API_KEY = os.getenv("AQICN_API_KEY")

LAT, LON = 24.860753, 67.029503 # latitude and longitude coordinates for my current location (Karachi)

ow_base_url = "http://api.openweathermap.org/data/2.5"
aqicn_base_url = "https://api.waqi.info/feed"

def fetch_aqicn_data():
    url = f"{aqicn_base_url}/Karachi"

    params = {"token": AQICN_API_KEY}
    response = requests.get(url, params=params)

    response.raise_for_status() # will check immediately for any failure and raise error

    aqicn = response.json()
    data = aqicn["data"]

    if aqicn["status"] != "ok":
        raise ValueError(f"AQICN API error: {aqicn}")

    return {
        "aqi": data["aqi"],    
    }

def fetch_openweather_weather_data():
    url = f"{ow_base_url}/weather"

    params = {"lat": LAT, "lon": LON, "appid": OW_API_KEY, "units": "metric"}
    response = requests.get(url, params=params)

    response.raise_for_status()

    data = response.json()

    return {
        "temperature": data["main"]["temp"],
        "pressure": data["main"]["pressure"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"],
        "wind_direction": data["wind"].get("deg", 0),
    }

def fetch_openweather_air_pollution_data():
    url = f"{ow_base_url}/air_pollution"

    params = {"lat": LAT, "lon": LON, "appid": OW_API_KEY}
    response = requests.get(url, params=params)

    response.raise_for_status()

    data = response.json()    
    components = data["list"][0]["components"]

    return {
        "pm2_5": components["pm2_5"],
        "pm10": components["pm10"],
        "o3": components["o3"],
        "no2": components["no2"],
        "co": components["co"],
        "so2": components["so2"],
    }

def fetch_all_data():
    aqi = fetch_aqicn_data()
    weather = fetch_openweather_weather_data()
    pollution = fetch_openweather_air_pollution_data()

if __name__ == "__main__":
    pass

