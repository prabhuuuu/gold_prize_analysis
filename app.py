import streamlit as st
import numpy as np
import pickle
import requests

# -----------------------------
# 1. Load Trained Model
# -----------------------------
model_path = "model.pkl"
model = pickle.load(open(model_path, "rb"))

# -----------------------------
# 2. Function to fetch live USD → INR rate
# -----------------------------
@st.cache_data(ttl=300)  # cache result for 5 minutes
def get_usd_inr_rate():
    """
    Fetch the live USD→INR conversion rate from exchangerate.host API.
    """
    try:
        url = "https://api.exchangerate.host/convert?from=USD&to=INR&amount=1"
        response = requests.get(url, timeout=8)
        data = response.json()
        rate = data.get("result")
        if rate:
            return float(rate)
        else:
            # fallback
            r2 = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=INR").json()
            return float(r2["rates"]["INR"])
    except Exception as e:
        st.warning(f"⚠️ Could not fetch live rate. Using default fallback (₹84.5/USD).")
        return 84.5  # fallback rate

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.set_page_config(page_title="💰 Gold Price Prediction (USD & INR)", layout="centered")
st.title("💰 Gold Price Prediction App")
st.write("Predict gold price in **USD** and **INR** using Machine Learning + Live Exchange Rate API")

st.markdown("---")

# Input fields for model features
col1, col2 = st.columns(2)
with col1:
    spx = st.number_input("S&P 500 (SPX)", value=4200.0)
    uso = st.number_input("Oil ETF (USO)", value=75.0)
with col2:
    eur_usd = st.number_input("EUR/USD Exchange Rate", value=1.08)
    slv = st.number_input("Silver ETF (SLV)", value=22.0)

# Prediction button
if st.button("Predict Gold Price 💰"):
    try:
        # Prepare features
        features = np.array([[spx, uso, eur_usd, slv]])

        # Predict price in USD
        pred_usd = model.predict(features)[0]

        # Fetch live exchange rate
        usd_inr = get_usd_inr_rate()

        # Convert to INR
        pred_inr = pred_usd * usd_inr

        # Display results
        st.success(f"**Predicted Gold Price (USD):** ${pred_usd:,.2f}")
        st.success(f"**Predicted Gold Price (INR):** ₹{pred_inr:,.2f}")
        st.caption(f"Exchange Rate used: 1 USD = ₹{usd_inr:.2f}")

    except Exception as e:
        st.error(f"Error while predicting: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and exchangerate.host API")
