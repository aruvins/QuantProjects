import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

def calculate_var(data, confidence_level):
    returns = data.pct_change().dropna()
    var = np.percentile(returns, 100 - confidence_level)
    return var

def calculate_cvar(data, confidence_level):
    returns = data.pct_change().dropna()
    var = calculate_var(data, confidence_level)
    cvar = returns[returns <= var].mean()
    return cvar

st.title("Quantitative Risk Dashboard")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOG):", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))
confidence_level = st.slider("Confidence Level (%)", min_value=90, max_value=99, value=95)

if st.button("Analyze"):
    with st.spinner("Fetching data and calculating metrics..."):
        try:
            # Fetch historical data
            data = fetch_data(ticker, start_date, end_date)
            st.success("Data fetched successfully!")

            # Calculate risk metrics
            var = calculate_var(data, confidence_level)
            cvar = calculate_cvar(data, confidence_level)

            # Display metrics
            st.subheader(f"Risk Metrics for {ticker}")
            st.write(f"Value at Risk (VaR) at {confidence_level}% confidence: {var:.4f}")
            st.write(f"Conditional VaR (CVaR) at {confidence_level}% confidence: {cvar:.4f}")

            stress_test_data = data * 0.9
            st.subheader("Stress Test: 10% Price Drop Simulation")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Historical Prices'))
            fig.add_trace(go.Scatter(x=stress_test_data.index, y=stress_test_data, mode='lines', name='Stress Test (10% Drop)'))
            fig.update_layout(title="Price History and Stress Test",
                              xaxis_title="Date",
                              yaxis_title="Price (USD)")
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
