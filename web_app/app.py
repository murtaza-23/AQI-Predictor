import os
import requests
import joblib
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime
from io import StringIO

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Page Configuration
st.set_page_config(
    page_title="AtmoKHI — Karachi AQI Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Directory Paths Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "training_pipeline", "models", "best_model.pkl")
CSV_PATH = os.path.join(BASE_DIR, "data", "aqi_features.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "training_pipeline", "plots")
EDA_PLOTS_DIR = os.path.join(BASE_DIR, "notebooks", "plots")

API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000")
GITHUB_CSV_URL = "https://raw.githubusercontent.com/murtaza-23/AtmoKHI/main/data/aqi_features.csv"

# Portfolio Links
LINKEDIN_URL = "https://www.linkedin.com/in/murtaza-aamir"
GITHUB_REPO_URL = "https://github.com/murtaza-23/AtmoKHI"

# Custom CSS — AtmoKHI Design System 
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ── Global Theme ── */
    .stApp {
        background: #F3E8D8;
        color: #0A0908;
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .stApp p, .stApp span, .stApp label, .stApp li {
        color: #0A0908;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #e8dfd0; }
    ::-webkit-scrollbar-thumb { background: #1E249E; border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: #38BDF8; }

    /* ── Header ── */
    .atmokhi-header {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(30, 36, 158, 0.12);
        border-radius: 18px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(10, 9, 8, 0.06);
    }
    .header-title {
        font-family: 'Bebas Neue', Impact, sans-serif;
        font-size: 4.2rem;
        font-weight: 400;
        color: #1E249E;
        letter-spacing: 2px;
        line-height: 1;
        margin: 0;
        text-transform: uppercase;
    }
    .header-subtitle {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        color: #1E249E;
        margin-top: 8px;
        font-weight: 500;
        opacity: 0.85;
    }
    .header-links { margin-top: 10px; font-size: 0.9rem; }
    .header-links a {
        color: #1E249E;
        text-decoration: none;
        font-weight: 600;
        margin-right: 15px;
        transition: color 0.2s;
    }
    .header-links a:hover { color: #1E249E; text-decoration: underline; }

    .header-location-badge {
        display: inline-block;
        background: rgba(30, 36, 158, 0.95);
        border: 1px solid #1E249E;
        padding: 8px 16px;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #FAFAF7;
        box-shadow: 0 4px 14px rgba(30, 36, 158, 0.25);
    }

    /* ── Section Headings — Retro Display ── */
    h1, .stMarkdown h1, [data-testid="stMarkdownContainer"] h1, .header-title {
        font-family: 'Bebas Neue', Impact, sans-serif !important;
        font-size: 3.6rem !important;
        font-weight: 400 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        line-height: 1.05 !important;
    }

    h2, .stMarkdown h2, [data-testid="stMarkdownContainer"] h2,
    [data-testid="stHeader"] {
        font-family: 'Bebas Neue', Impact, sans-serif !important;
        font-size: 2.6rem !important;
        font-weight: 400 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: #1E249E !important;
        -webkit-text-fill-color: #1E249E !important;
        line-height: 1.1 !important;
    }

    h3, h4, h5,
    .stMarkdown h3, .stMarkdown h4, .stMarkdown h5,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5 {
        font-family: 'Bebas Neue', Impact, sans-serif !important;
        font-size: 2rem !important;
        font-weight: 400 !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
        color: #1E249E !important;
        -webkit-text-fill-color: #1E249E !important;
        line-height: 1.15 !important;
    }

    .accent-body, .stMarkdown p.accent-body {
        color: #1E249E !important;
        font-weight: 500 !important;
    }

    [data-testid="stCaptionContainer"] {
        color: #334155 !important;
    }

    [data-testid="stCaptionContainer"] strong {
        color: #1E249E !important;
    }

    /* ── Unified Deep Cobalt Cards ── */
    .glass-card,
    .pollutant-card {
        background: rgba(30, 36, 158, 0.95) !important;
        border: 1px solid rgba(250, 250, 247, 0.15) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        text-align: left;
        margin-bottom: 18px !important;
        box-shadow: 0 4px 20px rgba(10, 9, 8, 0.08) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
    }

    .glass-card:hover,
    .pollutant-card:hover {
        transform: translateY(-3px) !important;
        border-color: #38BDF8 !important;
        box-shadow:
            0 0 22px rgba(56, 189, 248, 0.55),
            0 8px 28px rgba(30, 36, 158, 0.25) !important;
    }

    .card-label,
    .pollutant-card .pollutant-label {
        color: #FAFAF7 !important;
        -webkit-text-fill-color: #FAFAF7 !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 4px !important;
        opacity: 0.92 !important;
    }

    .card-value,
    .pollutant-card .pollutant-value {
        color: #FAFAF7 !important;
        -webkit-text-fill-color: #FAFAF7 !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
        margin: 4px 0 !important;
    }

    .pollutant-card .pollutant-value { font-size: 2.1rem !important; }

    .pollutant-card .pollutant-unit {
        color: #FAFAF7 !important;
        -webkit-text-fill-color: #FAFAF7 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        opacity: 0.85 !important;
    }

    .glass-card h4,
    .glass-card p,
    .glass-card p b {
        color: #FAFAF7 !important;
        -webkit-text-fill-color: #FAFAF7 !important;
    }

    /* ── Forecast & Health Status Cards ── */
    .forecast-day-card,
    .health-status-card {
        background: var(--card-fill, #FEF3C7) !important;
        border: 2px solid var(--card-accent, #EAB308) !important;
        border-radius: 18px !important;
        padding: 22px 20px !important;
        box-shadow: 0 4px 14px var(--card-glow, rgba(234, 179, 8, 0.18)) !important;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease !important;
        margin-bottom: 8px !important;
    }

    .forecast-day-card {
        text-align: center !important;
    }

    .health-status-card {
        text-align: left !important;
    }

    .forecast-day-card:hover,
    .health-status-card:hover {
        transform: translateY(-3px) !important;
        border-color: var(--card-accent) !important;
        box-shadow:
            0 0 10px var(--card-glow),
            0 4px 18px var(--card-glow) !important;
    }

    .forecast-emoji {
        font-size: 2rem !important;
        line-height: 1 !important;
        margin-bottom: 8px !important;
    }

    .health-status-row {
        display: flex !important;
        align-items: center !important;
        gap: 20px !important;
        flex-wrap: nowrap !important;
    }

    .health-status-emoji {
        font-size: 2.4rem !important;
        line-height: 1 !important;
        flex-shrink: 0 !important;
        width: 44px !important;
        text-align: center !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    .health-status-content {
        flex: 1 !important;
        min-width: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        gap: 2px !important;
    }

    .health-status-label {
        font-family: 'Bebas Neue', Impact, sans-serif !important;
        font-size: 1.75rem !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
        line-height: 1.1 !important;
        margin: 0 !important;
        color: var(--card-text, #713F12) !important;
        -webkit-text-fill-color: var(--card-text, #713F12) !important;
        text-align: left !important;
    }

    .health-status-aqi {
        font-family: 'Bebas Neue', Impact, sans-serif !important;
        font-size: 2.2rem !important;
        line-height: 1 !important;
        margin: 0 !important;
        color: var(--card-text, #713F12) !important;
        -webkit-text-fill-color: var(--card-text, #713F12) !important;
        text-align: left !important;
    }

    .health-status-desc {
        margin: 16px 0 0 0 !important;
        padding-top: 14px !important;
        border-top: 1px solid rgba(0, 0, 0, 0.08) !important;
        color: var(--card-text, #713F12) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        opacity: 0.9 !important;
        text-align: left !important;
    }

    /* ── Custom Pill Tab Navigation (st.radio) ── */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 10px !important;
        background: transparent !important;
        border: none !important;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background-color: #FFFFFF !important;
        border: 2px solid #CBD5E1 !important;
        border-radius: 9999px !important;
        padding: 12px 22px !important;
        margin: 0 !important;
        min-height: 46px !important;
        box-shadow: 0 3px 10px rgba(10, 9, 8, 0.08) !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
        flex: 0 1 auto !important;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        border-color: #1E249E !important;
        box-shadow: 0 4px 14px rgba(30, 36, 158, 0.2) !important;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background-color: #1E249E !important;
        border-color: #1E249E !important;
        box-shadow: 0 4px 18px rgba(30, 36, 158, 0.45) !important;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label p,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label span,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label div {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        color: #1E293B !important;
        -webkit-text-fill-color: #1E293B !important;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] p,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] span,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) p,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) span {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* Legacy st.tabs fallback (if used elsewhere) */
    .stTabs [data-baseweb="tab-border"],
    .stTabs [data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"],
    div[data-baseweb="tab-highlight"] {
        display: none !important;
        background: transparent !important;
        height: 0 !important;
    }

    .stTabs [data-baseweb="tab-list"],
    div[data-baseweb="tab-list"] {
        gap: 10px !important;
        flex-wrap: wrap !important;
        border-bottom: none !important;
        padding-bottom: 4px !important;
        margin-bottom: 28px !important;
        background: transparent !important;
    }

    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"],
    .stTabs button[data-baseweb="tab"],
    div[data-baseweb="tab-list"] button[role="tab"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 2px solid #CBD5E1 !important;
        border-radius: 9999px !important;
        padding: 11px 24px !important;
        margin: 0 !important;
        min-height: 46px !important;
        height: auto !important;
        box-shadow: 0 2px 8px rgba(10, 9, 8, 0.08) !important;
        opacity: 1 !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] div,
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] span,
    .stTabs button[data-baseweb="tab"] p,
    .stTabs button[data-baseweb="tab"] div,
    .stTabs button[data-baseweb="tab"] span,
    div[data-baseweb="tab-list"] button[role="tab"] *,
    div[data-baseweb="tab-list"] button[role="tab"] p,
    div[data-baseweb="tab-list"] button[role="tab"] div,
    div[data-baseweb="tab-list"] button[role="tab"] span {
        color: #1E293B !important;
        -webkit-text-fill-color: #1E293B !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        opacity: 1 !important;
    }

    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"],
    .stTabs button[data-baseweb="tab"][aria-selected="true"],
    div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
        background-color: #1E249E !important;
        border-color: #1E249E !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 16px rgba(30, 36, 158, 0.4) !important;
    }

    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] p,
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] div,
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] span,
    .stTabs button[data-baseweb="tab"][aria-selected="true"] p,
    .stTabs button[data-baseweb="tab"][aria-selected="true"] div,
    .stTabs button[data-baseweb="tab"][aria-selected="true"] span,
    div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] *,
    div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] p,
    div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] div,
    div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] span {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    .stTabs [data-baseweb="tab-panel"],
    [data-baseweb="tab-panel"] {
        border: none !important;
        padding-top: 8px !important;
    }

    [data-testid="stTabs"] {
        border: none !important;
    }

    [data-testid="stTabs"] > div > div:first-child {
        border-bottom: none !important;
        background: transparent !important;
    }

    .stTabs button[data-baseweb="tab"]:focus,
    .stTabs button[data-baseweb="tab"]:active,
    div[data-baseweb="tab-list"] button[role="tab"]:focus {
        outline: none !important;
        box-shadow: 0 2px 8px rgba(10, 9, 8, 0.08) !important;
    }

    .stTabs button[data-baseweb="tab"][aria-selected="true"]:focus,
    div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"]:focus {
        box-shadow: 0 4px 16px rgba(30, 36, 158, 0.4) !important;
    }

    /* ── Scenario Simulator Sliders ── */
    .stSlider label, .stNumberInput label {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(1) {
        height: 8px !important;
        border-radius: 4px !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(1) > div {
        background-color: #EF4444 !important;
        height: 8px !important;
        border-radius: 4px !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) > div {
        background-color: #EF4444 !important;
        border-color: #EF4444 !important;
        width: 18px !important;
        height: 18px !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(1) > div > div {
        background-color: #EF4444 !important;
    }

    /* ── Run Scenario Prediction Button ── */
    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #EF4444 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button *,
    div[data-testid="stFormSubmitButton"] > button *,
    div[data-testid="stFormSubmitButton"] > button p,
    div[data-testid="stFormSubmitButton"] > button span {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #DC2626 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(220, 38, 38, 0.35) !important;
    }

    /* ── EPA Reference Table (Tab 5) ── */
    div[data-testid="stDataFrame"],
    div[data-testid="stDataFrame"] > div,
    div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
        background-color: #FFFFFF !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    div[data-testid="stDataFrame"] [class*="glideDataEditor"],
    div[data-testid="stDataFrame"] canvas {
        background-color: #FFFFFF !important;
    }

    div[data-testid="stDataFrame"] th,
    div[data-testid="stDataFrame"] td,
    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataFrame"] [role="gridcell"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border-color: #e2e8f0 !important;
    }

    div[data-testid="stDataFrame"] [role="columnheader"] {
        background-color: #f8fafc !important;
        font-weight: 700 !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Color mapping for AQI categories
def get_aqi_info(aqi):
    if aqi <= 50:
        return {"color": "#22C55E", "label": "Good", "emoji": "🟢",
                "fill": "#E8F7EE", "glow": "rgba(34, 197, 94, 0.28)", "text": "#166534"}
    elif aqi <= 100:
        return {"color": "#EAB308", "label": "Moderate", "emoji": "🟡",
                "fill": "#FEF3C7", "glow": "rgba(234, 179, 8, 0.30)", "text": "#854D0E"}
    elif aqi <= 150:
        return {"color": "#F97316", "label": "Unhealthy for Sensitive Groups", "emoji": "🟠",
                "fill": "#FFEDD5", "glow": "rgba(249, 115, 22, 0.28)", "text": "#9A3412"}
    elif aqi <= 200:
        return {"color": "#EF4444", "label": "Unhealthy", "emoji": "🔴",
                "fill": "#FEE2E2", "glow": "rgba(239, 68, 68, 0.28)", "text": "#991B1B"}
    elif aqi <= 300:
        return {"color": "#A855F7", "label": "Very Unhealthy", "emoji": "🟣",
                "fill": "#F3E8FF", "glow": "rgba(168, 85, 247, 0.28)", "text": "#6B21A8"}
    else:
        return {"color": "#A855F7", "label": "Hazardous", "emoji": "⚫",
                "fill": "#EDE9FE", "glow": "rgba(168, 85, 247, 0.32)", "text": "#581C87"}


# Generic short-lived call helper (3 minutes)
@st.cache_data(ttl=180, show_spinner="Loading latest AQI data...")
def call_api(endpoint, params=None):
    try:
        r = requests.get(f"{API_URL}{endpoint}", params=params, timeout=45)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# Dedicated 1-Hour Global Cache for Daily Summaries
@st.cache_data(ttl=3600, show_spinner=False)
def get_daily_forecast_cached():
    return call_api("/forecast/daily")

# Dedicated 1-Hour Global Cache for 72-Hour Forecast
@st.cache_data(ttl=3600, show_spinner=False)
def get_72h_forecast_cached():
    return call_api("/forecast", params={"hours": 72})

OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

@st.cache_data(ttl=900, show_spinner=False)
def fetch_current_openweather():
    if not OPENWEATHER_API_KEY:
        print("OpenWeather API key missing.")
        return None
        
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": 24.860753,
            "lon": 67.029503,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"  # Temperature in Celsius, wind in m/s
        }
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()
        
        return {
            "temp": round(data["main"]["temp"], 1),
            "humidity": data["main"]["humidity"],
            "wind_speed": round(data["wind"]["speed"] * 3.6, 1),  # Convert m/s to km/h
            "wind_deg": data["wind"]["deg"],
            "description": data["weather"][0]["description"].title()
        }
    except Exception as e:
        print(f"OpenWeather fetch failed: {e}")
        return None


# Standalone model loader fallback
@st.cache_resource
def load_local_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


# Load CSV fallback data
@st.cache_data(ttl=300)
def load_local_data():
    df = None
    try:
        res = requests.get(GITHUB_CSV_URL, timeout=5)
        res.raise_for_status()
        df = pd.read_csv(StringIO(res.text))
    except Exception:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
    if df is not None:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


METRICS_JSON_PATH = os.path.join(BASE_DIR, "training_pipeline", "models", "metrics.json")
METRICS_TXT_PATH = os.path.join(BASE_DIR, "training_pipeline", "models", "metrics.txt")


@st.cache_data(ttl=300)
def load_model_metrics():
    model_info = call_api("/model/info")
    if model_info and model_info.get("metrics", {}).get("r2") is not None:
        return model_info

    if os.path.exists(METRICS_JSON_PATH):
        try:
            with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "model_name": data.get("best_model_name", "XGBoost"),
                    "metrics": {
                        "r2": data.get("r2"),
                        "rmse": data.get("rmse"),
                        "mae": data.get("mae"),
                        "trained_at": data.get("trained_at"),
                    },
                }
        except Exception:
            pass

    return {"model_name": "XGBoost", "metrics": {"r2": None, "rmse": None, "mae": None}}


# Fetch active data feeds using dedicated hourly cached endpoints
current = call_api("/current")
daily_data = get_daily_forecast_cached()
forecast_data = get_72h_forecast_cached()

# Offline fallback if API server is not running
if not current:
    local_df = load_local_data()
    local_model = load_local_model()
    if local_df is not None:
        latest = local_df.iloc[-1]
        aqi_val = float(latest["aqi"])
        aqi_meta = get_aqi_info(aqi_val)
        current = {
            "aqi": aqi_val,
            "category": aqi_meta["label"],
            "color": aqi_meta["color"],
            "timestamp": str(latest["timestamp"]),
            "pollutants": {
                "pm2_5": float(latest["pm2_5"]),
                "pm10": float(latest["pm10"]),
                "o3": float(latest["o3"]),
                "no2": float(latest["no2"]),
                "co": float(latest["co"]),
                "so2": float(latest["so2"]),
            }
        }

metrics_info = load_model_metrics()
m_metrics = metrics_info.get("metrics", {})
model_display_name = metrics_info.get("model_name", "XGBoost")
real_r2 = m_metrics.get("r2")
real_rmse = m_metrics.get("rmse")
real_mae = m_metrics.get("mae")


def _fmt_metric(value):
    return f"{value:.4f}" if value is not None else "N/A"

# Header Section
header_html = f"""
    <div class="atmokhi-header">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
            <div>
                <h1 class="header-title"> AtmoKHI - Karachi AQI Predictor </h1>
                <p class="header-subtitle">
                    Real-Time Air Quality Monitoring & 72-Hour Forecast
                </p>
                <div class="header-links">
                    <a href="{LINKEDIN_URL}" target="_blank">LinkedIn Profile</a>
                    <a href="{GITHUB_REPO_URL}" target="_blank">GitHub Repository</a>
                </div>
            </div>
            <div style="text-align:right;">
                <span class="header-location-badge">
                    📍 Karachi (24.86°N, 67.02°E)
                </span>
            </div>
        </div>
    </div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# AQI Status Warning
if current and current.get("aqi"):
    aqi_v = current["aqi"]
    cur_cat = current['category']
    cur_emoji = get_aqi_info(aqi_v)["emoji"]
    if aqi_v > 200:
        st.error(f"**HAZARD ALERT**: Current AQI is **{aqi_v:.0f}** {cur_emoji} ({cur_cat}). Avoid outdoor activities and wear N95 masks.")
    elif aqi_v > 150:
        st.warning(f"**AIR QUALITY WARNING**: Current AQI is **{aqi_v:.0f}** {cur_emoji} ({cur_cat}). Sensitive groups should remain indoors.")
    elif aqi_v > 100:
        st.info(f"**MODERATE AQI**: Current AQI is **{aqi_v:.0f}** {cur_emoji} ({cur_cat}). Air quality is acceptable for most people.")


# Tab Navigation — pill buttons via horizontal radio
TAB_LABELS = [
    "Live & 3-Day Forecast",
    "Model Metrics & SHAP",
    "Custom Scenario Simulator",
    "Karachi Data Insights",
    "Health Guidelines",
]

active_tab = st.radio(
    "Navigation",
    TAB_LABELS,
    horizontal=True,
    label_visibility="collapsed",
    key="atmokhi_nav",
)

# TAB 1: Live and 3 Day Forecast
if active_tab == TAB_LABELS[0]:
    if current:
        aqi_val = current["aqi"]
        meta = get_aqi_info(aqi_val)
        pollutants = current["pollutants"]

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("### Current Air Quality")
            
            # Semi-gauge indicator chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=aqi_val,
                title={'text': f"{meta['emoji']} {meta['label']}", 'font': {'size': 26, 'color': meta['color'], 'family': 'Bebas Neue, Impact, sans-serif'}},
                number={"font": {"size": 54, "color": "#0F172A", "family": "sans-serif"}},
                gauge={
                    'axis': {'range': [0, 300], 'tickwidth': 2, 'tickcolor': "#64748B", "tickfont": {"color": "#0F172A", "size": 12}},
                    'bar': {'color': meta['color'], "thickness": 0.35},
                    'bgcolor': "#FFFFFF",
                    "borderwidth": 2,
                    'bordercolor': "#CBD5E1",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(34, 197, 94, 0.65)'},
                        {'range': [51, 100], 'color': 'rgba(234, 179, 8, 0.65)'},
                        {'range': [101, 150], 'color': 'rgba(249, 115, 22, 0.65)'},
                        {'range': [151, 200], 'color': 'rgba(239, 68, 68, 0.65)'},
                        {'range': [201, 300], 'color': 'rgba(168, 85, 247, 0.65)'},
                    ],
                    "threshold": {
                        "line": {"color": meta["color"], "width": 6},
                        "thickness": 0.85,
                        "value": aqi_val
                    }
                }
            ))
            fig_gauge.update_layout(
                height=300,
                margin=dict(l=15, r=15, t=40, b=5),
                paper_bgcolor="#F3E8D8",
                plot_bgcolor="#F3E8D8",
                font={'color': "#0F172A", 'family': 'sans-serif'}
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False, "responsive": True})

        # Convert UTC timestamp to Pakistan Standard Time (PKT / UTC+5)
        try:
            utc_dt = pd.to_datetime(current['timestamp'])
            
            # If the string lacks explicit UTC tz info, set it first then convert
            if utc_dt.tzinfo is None:
                utc_dt = utc_dt.tz_localize('UTC')
                
            pkt_dt = utc_dt.tz_convert('Asia/Karachi')
            formatted_time = pkt_dt.strftime('%b %d, %Y at %I:%M %p PKT')
        except Exception:
            formatted_time = f"{current['timestamp']} UTC"

        st.caption(f"Last updated: **{formatted_time}**")

        with col_right:
            st.markdown("### Current Pollutants")
            p_cols = st.columns(3)
            
            p_list = [
                ("PM2.5", pollutants["pm2_5"], "μg/m³"),
                ("PM10", pollutants["pm10"], "μg/m³"),
                ("Ozone (O₃)", pollutants["o3"], "μg/m³"),
                ("NO₂", pollutants["no2"], "μg/m³"),
                ("CO", pollutants["co"], "μg/m³"),
                ("SO₂", pollutants["so2"], "μg/m³")
            ]
            
            for idx, (name, val, unit) in enumerate(p_list):
                with p_cols[idx % 3]:
                    st.markdown(
                    f"""
                    <div class="pollutant-card">
                        <div class="pollutant-label">{name}</div>
                        <div class="pollutant-value">{val:.1f}</div>
                        <div class="pollutant-unit">{unit}</div>
                    </div>
                    """,
                    unsafe_allow_html=True

                )   
                    
        weather = fetch_current_openweather()

        if weather:
            st.markdown("### Current Atmospheric Weather")
            w_col1, w_col2, w_col3, w_col4 = st.columns(4)

            with w_col1:
                st.markdown(f"""
                    <div class="glass-card">
                        <div class="card-label">Temperature</div>
                        <div class="card-value">{weather['temp']} °C</div>
                    </div>
                """, unsafe_allow_html=True)

            with w_col2:
                st.markdown(f"""
                    <div class="glass-card">
                        <div class="card-label">Relative Humidity</div>
                        <div class="card-value">{weather['humidity']} %</div>
                    </div>
                """, unsafe_allow_html=True)

            with w_col3:
                st.markdown(f"""
                    <div class="glass-card">
                        <div class="card-label">Wind Speed</div>
                        <div class="card-value">{weather['wind_speed']} km/h</div>
                    </div>
                """, unsafe_allow_html=True)

            with w_col4:
                st.markdown(f"""
                    <div class="glass-card">
                        <div class="card-label">Conditions</div>
                        <div class="card-value" style="font-size: 1.4rem !important;">{weather['description']}</div>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("3-Day Daily Summary Forecast")
    
    if daily_data and daily_data.get("daily_forecast"):
        df_days = daily_data["daily_forecast"]
        d_cols = st.columns(len(df_days))
        
        for i, day in enumerate(df_days):
            d_meta = get_aqi_info(day["avg_aqi"])
            day_emoji = day.get("emoji") or d_meta["emoji"]
            with d_cols[i]:
                st.markdown(f"""
                    <div class="forecast-day-card" style="--card-accent: {d_meta['color']}; --card-fill: {d_meta['fill']}; --card-glow: {d_meta['glow']};">
                        <div class="forecast-emoji">{day_emoji}</div>
                        <div style="font-family:'Bebas Neue',Impact,sans-serif; font-size:1.15rem; letter-spacing:1.2px; text-transform:uppercase; color:{d_meta['text']}; margin-bottom:4px;">{day['category']}</div>
                        <div style="font-weight:700; font-size:1rem; color:{d_meta['text']}; opacity:0.85; margin:6px 0;">{day['date']}</div>
                        <div style="font-family:'Bebas Neue',Impact,sans-serif; font-size:2.8rem; line-height:1; color:{d_meta['text']}; margin:8px 0;">{day['avg_aqi']:.0f}</div>
                        <div style="font-size:0.82rem; font-weight:700; color:{d_meta['text']}; text-transform:uppercase; letter-spacing:0.5px; opacity:0.9;">AQI Average</div>
                        <div style="font-size:0.88rem; color:{d_meta['text']}; margin-top:10px; font-weight:700; opacity:0.9;">
                            Range: {day['min_aqi']:.0f} – {day['max_aqi']:.0f}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("72-Hour AQI Trend Forecast")

    if forecast_data and forecast_data.get("forecasts"):
      f_df = pd.DataFrame(forecast_data["forecasts"])
      f_df["timestamp"] = pd.to_datetime(f_df["timestamp"])

      fig_line = go.Figure()

      # Add smooth glowing line with translucent area fill under the trend
      fig_line.add_trace(
          go.Scatter(
              x=f_df["timestamp"],
              y=f_df["predicted_aqi"],
              mode="lines+markers",
              name="Forecasted AQI",
              line=dict(color="#38bdf8", width=3, shape="spline"),
              fill="tozeroy",
              fillcolor="rgba(56, 189, 248, 0.08)",
              marker=dict(size=6, color="#0284c7", symbol="circle"),
              hovertemplate=(
                  "<b>%{x|%b %d, %I:%M %p}</b><br>Predicted AQI:"
                  " <b>%{y:.1f}</b><extra></extra>"
              ),
          )
      )

      # Threshold horizontal lines with bold annotations
      thresholds = [
          (50, "Good (50)", "#22C55E"),
          (100, "Moderate (100)", "#EAB308"),
          (150, "Unhealthy Sensitive (150)", "#F97316"),
          (200, "Unhealthy (200)", "#EF4444"),
      ]

      for val, label, color in thresholds:
        fig_line.add_hline(
            y=val,
            line_dash="dash",
            line_color=color,
            line_width=1.5,
            opacity=0.7,
            annotation_text=f" <b>{label}</b>",
            annotation_position="top right",
            annotation_font=dict(color=color, size=11),
        )

      fig_line.update_layout(
          height=420,
          hovermode="x unified",
          margin=dict(l=20, r=20, t=30, b=30),
          paper_bgcolor="#F3E8D8",
          plot_bgcolor="rgba(255, 255, 255, 0.55)",
          font=dict(color="#0F172A", family="sans-serif"),
          xaxis=dict(
              title=dict(
                  text="Date / Time (UTC)",
                  font=dict(color="#0F172A", size=15),
              ),
              tickfont=dict(color="#0F172A", size=13),
              gridcolor="rgba(15, 23, 42, 0.08)",
              zerolinecolor="rgba(15, 23, 42, 0.12)",
              showgrid=True,
          ),
          yaxis=dict(
              title=dict(
                  text="Predicted AQI Index",
                  font=dict(color="#0F172A", size=15),
              ),
              tickfont=dict(color="#0F172A", size=13),
              gridcolor="rgba(15, 23, 42, 0.08)",
              zerolinecolor="rgba(15, 23, 42, 0.12)",
              showgrid=True,
          ),
      )

      st.plotly_chart(
          fig_line,
          use_container_width=True,
          config={"displayModeBar": False, "responsive": True},
      )

# TAB 2: Model Architecture & SHAP
elif active_tab == TAB_LABELS[1]:
    st.markdown("### Model Evaluation Metrics")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
            <div class="glass-card">
                <div class="card-label">Selected Model</div>
                <div class="card-value" style="color:#0284C7;">{model_display_name}</div>
            </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
            <div class="glass-card">
                <div class="card-label">Validation R² Score</div>
                <div class="card-value" style="color:#22C55E;">{_fmt_metric(real_r2)}</div>
            </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
            <div class="glass-card">
                <div class="card-label">Root Mean Square Error</div>
                <div class="card-value" style="color:#EAB308;">{_fmt_metric(real_rmse)}</div>
            </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
            <div class="glass-card">
                <div class="card-label">Mean Absolute Error</div>
                <div class="card-value" style="color:#9333EA;">{_fmt_metric(real_mae)}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Feature Importance & SHAP Values")
    
    shap_img_path = os.path.join(PLOTS_DIR, "shap_summary.png")
    
    col_shap_img, col_shap_desc = st.columns([3, 2])
    with col_shap_img:
        if os.path.exists(shap_img_path):
            st.image(shap_img_path, caption="SHAP Summary Plot", width=620)
        else:
            st.info("SHAP summary plot will render here after running model training.")
    
    with col_shap_desc:
        st.markdown("""
        #### How to interpret SHAP values:
        * **Feature Ranking**: Top features exert the largest overall impact on predictions.
        * **Color Gradient**: Red dots indicate higher feature values; blue dots indicate lower feature values.
        * **Horizontal Impact**: Values to the right increase predicted AQI; values to the left lower predicted AQI.
        
        #### Key Insights:
        1. **Autoregressive Lags**: Previous hour AQI (`aqi_lag_1h`) accounts for the strongest predictive signal.
        2. **Particulate Matter**: PM2.5 and PM10 serve as key chemical features influencing atmospheric shifts.
        3. **Diurnal Time Features**: Hour of the day captures traffic peak times during commuting hours.
        """)

# TAB 3: Interactive Custom Scenario Simulator
elif active_tab == TAB_LABELS[2]:
    st.markdown("### Interactive Custom Scenario Simulator")
    st.markdown('<p class="accent-body">Adjust input sliders to test how custom pollutant levels and temporal lags affect AQI predictions.</p>', unsafe_allow_html=True)

    with st.form("custom_predict_form"):
        st.markdown("##### Pollutant Concentrations")
        sc_col1, sc_col2, sc_col3 = st.columns(3)
        c_pm2_5 = sc_col1.slider("PM2.5 (μg/m³)", 0.0, 300.0, 35.0, 1.0)
        c_pm10 = sc_col2.slider("PM10 (μg/m³)", 0.0, 400.0, 65.0, 1.0)
        c_o3 = sc_col3.slider("Ozone O3 (μg/m³)", 0.0, 200.0, 40.0, 1.0)

        sc_col4, sc_col5, sc_col6 = st.columns(3)
        c_no2 = sc_col4.slider("NO2 (μg/m³)", 0.0, 200.0, 20.0, 1.0)
        c_co = sc_col5.slider("CO (μg/m³)", 0.0, 2000.0, 400.0, 10.0)
        c_so2 = sc_col6.slider("SO2 (μg/m³)", 0.0, 200.0, 12.0, 1.0)

        st.markdown("##### Temporal Factors & Lag Baseline")
        sc_col7, sc_col8 = st.columns(2)
        c_hour = sc_col7.slider("Hour of Day (0-23)", 0, 23, 14)
        c_aqi_lag = sc_col8.slider("Previous Hour AQI Baseline", 0.0, 500.0, 85.0, 1.0)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Run Scenario Prediction")

        if submit_btn:
            payload = {
                "pm2_5": c_pm2_5, "pm10": c_pm10, "o3": c_o3,
                "no2": c_no2, "co": c_co, "so2": c_so2,
                "hour": c_hour, "aqi_lag_1h": c_aqi_lag, "aqi_lag_24h": c_aqi_lag
            }
            try:
                res = requests.post(f"{API_URL}/predict/custom", json=payload, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    pred_val = float(data['predicted_aqi'])
                    p_meta = get_aqi_info(pred_val)
                    
                    st.markdown(f"""
                        <div class="health-status-card" style="margin-top:20px; --card-accent: {p_meta['color']}; --card-fill: {p_meta['fill']}; --card-glow: {p_meta['glow']}; --card-text: {p_meta['text']};">
                            <div class="health-status-row">
                                <div class="health-status-emoji">{p_meta['emoji']}</div>
                                <div class="health-status-content">
                                    <div class="health-status-label">Predicted Scenario Status: {p_meta['label']}</div>
                                    <div class="health-status-aqi">AQI {pred_val:.1f}</div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("API backend unreachable. Using local calculation...")
                    local_model = load_local_model()
                    if local_model:
                        h_cat = 1 if 6 <= c_hour < 10 else (2 if 10 <= c_hour < 17 else (3 if 17 <= c_hour < 21 else 0))
                        feat_dict = {
                            "pm2_5": c_pm2_5, "pm10": c_pm10, "o3": c_o3, "no2": c_no2, "co": c_co, "so2": c_so2,
                            "hour": c_hour, "day": 15, "day_of_week": 2, "month": 8, "is_weekend": 0, "hour_category": h_cat,
                            "aqi_lag_1h": c_aqi_lag, "aqi_lag_3h": c_aqi_lag, "aqi_lag_24h": c_aqi_lag, "aqi_lag_48h": c_aqi_lag, "aqi_lag_72h": c_aqi_lag,
                            "aqi_change_prev_hour": 0.0, "aqi_rolling_3h": c_aqi_lag, "aqi_rolling_24h": c_aqi_lag,
                            "pm2_5_lag_1h": c_pm2_5, "pm2_5_lag_24h": c_pm2_5, "pm10_lag_1h": c_pm10, "pm10_lag_24h": c_pm10,
                            "o3_lag_1h": c_o3, "o3_lag_24h": c_o3, "no2_lag_1h": c_no2, "no2_lag_24h": c_no2,
                            "co_lag_1h": c_co, "co_lag_24h": c_co, "so2_lag_1h": c_so2, "so2_lag_24h": c_so2
                        }
                        f_df = pd.DataFrame([feat_dict])
                        pred_val = float(local_model.predict(f_df)[0])
                        p_meta = get_aqi_info(pred_val)
                        
                        st.markdown(f"""
                            <div class="health-status-card" style="margin-top:20px; --card-accent: {p_meta['color']}; --card-fill: {p_meta['fill']}; --card-glow: {p_meta['glow']}; --card-text: {p_meta['text']};">
                                <div class="health-status-row">
                                    <div class="health-status-emoji">{p_meta['emoji']}</div>
                                    <div class="health-status-content">
                                        <div class="health-status-label">Local Model Prediction: {p_meta['label']}</div>
                                        <div class="health-status-aqi">AQI {pred_val:.1f}</div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("No local model binary found.")
            except Exception as ex:
                st.error(f"Prediction request failed: {ex}")

# TAB 4: Data Insights & Visualizations
elif active_tab == TAB_LABELS[3]:
    st.markdown("### Karachi Air Quality Data Insights")
    st.markdown('<p class="accent-body">Exploratory charts generated from historical environmental data.</p>', unsafe_allow_html=True)

    eda_plots = [
        ("aqi_over_time.png", "1-Year Historical AQI Trend", "Timeline showing AQI variations over the dataset period.", EDA_PLOTS_DIR),
        ("aqi_by_hour.png", "Diurnal Hourly Patterns", "Average AQI fluctuation across hours of the day.", EDA_PLOTS_DIR),
        ("aqi_by_month.png", "Monthly Seasonal Variation", "Distribution of air quality metrics across months.", EDA_PLOTS_DIR),
        ("correlation_heatmap.png", "Feature Correlation Matrix", "Linear correlation heatmap between target AQI and key pollutants.", EDA_PLOTS_DIR),
        ("model_comparison.png", "Model Selection Benchmark", "Time-series cross-validation performance across algorithms.", PLOTS_DIR),
        ("actual_vs_predicted.png", "Actual vs Predicted Validation Curve", "Scatter plot comparing ground truth against model outputs.", PLOTS_DIR),
    ]

    grid_cols = st.columns(2)
    for idx, (filename, title, desc, folder) in enumerate(eda_plots):
        img_path = os.path.join(folder, filename)
        with grid_cols[idx % 2]:
            st.markdown(f"""
                <div class="glass-card" style="margin-bottom: 20px;">
                    <h4 style="margin:0 0 6px 0; color:#FAFAF7;">{title}</h4>
                    <p style="font-size:0.85rem; color:#FAFAF7; margin-bottom:12px; opacity:0.9;">{desc}</p>
            """, unsafe_allow_html=True)
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.info(f"Plot `{filename}` not found in `{folder}`.")
            st.markdown("</div>", unsafe_allow_html=True)

# TAB 5: Health Guidelines
elif active_tab == TAB_LABELS[4]:
    st.markdown("### Public Health Advisories")
    
    current_aqi_val = current["aqi"] if current else 70.0
    c_meta = get_aqi_info(current_aqi_val)

    st.markdown(f"""
        <div class="health-status-card" style="--card-accent: {c_meta['color']}; --card-fill: {c_meta['fill']}; --card-glow: {c_meta['glow']}; --card-text: {c_meta['text']};">
            <div class="health-status-row">
                <div class="health-status-emoji">{c_meta['emoji']}</div>
                <div class="health-status-content">
                    <div class="health-status-label">Current Status: {c_meta['label']}</div>
                    <div class="health-status-aqi">AQI {current_aqi_val:.0f}</div>
                </div>
            </div>
            <p class="health-status-desc">
                General precautionary guidance for Karachi residents based on current atmospheric conditions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    h_col1, h_col2, h_col3 = st.columns(3)
    
    with h_col1:
        st.markdown("""
            <div class="glass-card">
                <h4 style="color:#FAFAF7 !important; -webkit-text-fill-color:#FAFAF7 !important; margin-top:0;"> Sensitive Groups</h4>
                <p><b>Good (0-50):</b> Ideal for outdoor exertion.</p>
                <p><b>Moderate (51-100):</b> Unusually sensitive individuals monitor symptoms.</p>
                <p><b>Unhealthy (101+):</b> Keep rescue inhalers accessible; limit peak hour exposure.</p>
            </div>
        """, unsafe_allow_html=True)
    with h_col2:
        st.markdown("""
            <div class="glass-card">
                <h4 style="color:#FAFAF7 !important; -webkit-text-fill-color:#FAFAF7 !important; margin-top:0;"> Children & Elderly</h4>
                <p><b>Good (0-50):</b> Safe conditions for outdoor play.</p>
                <p><b>Moderate (51-100):</b> Monitor outdoor activity duration.</p>
                <p><b>Unhealthy (101+):</b> Restrict intense outdoor exercise during morning rush hours.</p>
            </div>
        """, unsafe_allow_html=True)
    with h_col3:
        st.markdown("""
            <div class="glass-card">
                <h4 style="color:#FAFAF7 !important; -webkit-text-fill-color:#FAFAF7 !important; margin-top:0;"> Outdoor Workers</h4>
                <p><b>Good (0-50):</b> Optimal outdoor conditions.</p>
                <p><b>Moderate (51-100):</b> Take frequent breaks during intense manual labor.</p>
                <p><b>Unhealthy (101+):</b> Wear protective masks near high-density traffic areas.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")  

    st.markdown("### Standard EPA Air Quality Index Reference")
    scale_df = pd.DataFrame({
        "Category": ["Good", "Moderate", "Unhealthy (Sensitive)", "Unhealthy", "Very Unhealthy", "Hazardous"],
        "AQI Range": ["0-50", "51-100", "101-150", "151-200", "201-300", "301-500"],
        "Health Summary": [
            "Satisfactory air quality with little or no health risk.",
            "Acceptable quality; minor risk for unusually sensitive individuals.",
            "Sensitive groups may experience health effects.",
            "General public may begin experiencing health effects.",
            "Health warning of emergency conditions for the whole population.",
            "Serious health risks for the entire population."
        ]
    })
    st.dataframe(scale_df, use_container_width=True, hide_index=True)

# Footer
st.markdown(f"""
    <hr style="border-color:#cbd5e1; margin-top:40px;">
    <div style="text-align:center; color:#64748b; font-size:0.85rem; padding:12px 0;">
        <b style="color:#1E249E;"> AtmoKHI </b> — <span style="color:#334155;">Developed by Murtaza Aamir</span><br>
        <a href="{GITHUB_REPO_URL}" target="_blank" style="color:#1E249E; text-decoration:none; font-weight:600;">View Source Code on GitHub</a>
    </div>
""", unsafe_allow_html=True)