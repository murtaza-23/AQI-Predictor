# AQI Predictor

An end-to-end serverless machine learning pipeline for forecasting Air Quality Index (AQI) up to 3 days ahead using weather and pollutant data.

## Project Overview

This project is being developed as part of the **10Pearls Pakistan SHINE Internship Program**.

The system automatically:

- Collects weather and AQI data from external APIs
- Builds and stores engineered features
- Trains machine learning models
- Predicts AQI for the next 3 days
- Visualizes predictions through a Streamlit dashboard
- Automates pipelines using GitHub Actions

## Tech Stack

- Python
- Scikit-learn
- TensorFlow
- Streamlit
- Flask
- GitHub Actions
- Hopsworks / Vertex AI
- SHAP
- OpenWeather API / AQICN API

## Project Structure

```text
aqi-predictor/
│
├── feature_pipeline/
├── training_pipeline/
├── web_app/
├── notebooks/
└── .github/
    └── workflows/
```
