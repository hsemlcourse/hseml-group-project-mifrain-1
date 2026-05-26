from __future__ import annotations

import os
from datetime import date

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def request_prediction(payload: dict[str, object]) -> dict[str, object]:
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="Car Price Predictor", page_icon=":car:", layout="centered")

st.title("Business-class car price predictor")

with st.form("prediction_form"):
    dataset_part = st.selectbox("Market segment", options=["used", "new"], index=0)
    brand = st.text_input("Brand", value="BMW")
    model = st.text_input("Model", value="X5")
    trim_or_variant = st.text_input("Trim or variant", value="BMW X5 2020 A/T Turbo SUV")
    model_year = st.number_input("Model year", min_value=1900, max_value=2035, value=2020, step=1)
    snapshot_date = st.date_input("Snapshot date", value=date(2023, 3, 27))
    submitted = st.form_submit_button("Predict price")

if submitted:
    payload = {
        "dataset_part": dataset_part,
        "brand": brand,
        "model": model,
        "trim_or_variant": trim_or_variant,
        "model_year": int(model_year),
        "snapshot_date": snapshot_date.isoformat(),
    }
    try:
        result = request_prediction(payload)
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
    else:
        price = float(result["predicted_price_egp"])
        st.metric("Predicted price", f"{price:,.0f} EGP")
        st.caption(f"Model: {result['model_name']} | Primary metric: {result['primary_metric'].upper()}")
