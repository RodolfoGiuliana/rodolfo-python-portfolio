import yfinance as yf
import pandas as pd
from datetime import datetime

# =========================
# PARAMETERS
# =========================
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
START_DATE = "2015-01-01"
OUTPUT_FILE = "raw_prices.csv"

# =========================
# DOWNLOAD MARKET DATA
# =========================
def fetch_market_data():
    data = yf.download(
        TICKERS,
        start=START_DATE,
        progress=False
    )["Close"]

    data.to_csv(OUTPUT_FILE)
    print(f"Market data saved to {OUTPUT_FILE}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    fetch_market_data()
