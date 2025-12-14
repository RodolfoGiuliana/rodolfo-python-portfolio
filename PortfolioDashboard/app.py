import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Portfolio Analytics Dashboard",
    layout="wide"
)

st.title("Portfolio Analytics Dashboard")
st.write("Upload a portfolio CSV file with columns: **ticker**, **weight**")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload portfolio file",
    type=["csv"]
)

# =========================
# MAIN LOGIC
# =========================
if uploaded_file is not None:
    portfolio = pd.read_csv(uploaded_file)

    if not {"ticker", "weight"}.issubset(portfolio.columns):
        st.error("CSV must contain 'ticker' and 'weight' columns.")
    else:
        tickers = portfolio["ticker"].tolist()
        weights = portfolio["weight"].values

        # =========================
        # DOWNLOAD DATA
        # =========================
        prices = yf.download(
            tickers,
            start="2018-01-01",
            progress=False
        )["Close"]

        returns = prices.pct_change().dropna()

        # =========================
        # PORTFOLIO RETURNS
        # =========================
        portfolio_returns = returns.dot(weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # =========================
        # METRICS
        # =========================
        annual_return = cumulative_returns.iloc[-1] ** (252 / len(cumulative_returns)) - 1
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

        # =========================
        # METRICS DISPLAY
        # =========================
        col1, col2, col3 = st.columns(3)
        col1.metric("Annual Return", f"{annual_return:.2%}")
        col2.metric("Volatility", f"{annual_volatility:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

        # =========================
        # PLOTS
        # =========================
        st.subheader("Cumulative Portfolio Performance")

        fig, ax = plt.subplots()
        ax.plot(cumulative_returns)
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.grid(True)

        st.pyplot(fig)

        # =========================
        # CORRELATION MATRIX
        # =========================
        st.subheader("Asset Correlation Matrix")
        corr = returns.corr()
        st.dataframe(corr.round(2))

else:
    st.info("Please upload a portfolio CSV file to start.")
