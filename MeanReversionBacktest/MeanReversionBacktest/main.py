import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================
TICKER = "^NDX"          # Nasdaq 100
START_DATE = "2015-01-01"
WINDOW = 20
Z_ENTRY = 1.0

# =========================
# DOWNLOAD DATA
# =========================
data = yf.download(TICKER, start=START_DATE, progress=False)
data = data[["Close"]].rename(columns={"Close": "price"})

# =========================
# INDICATORS
# =========================
data["mean"] = data["price"].rolling(WINDOW).mean()
data["std"] = data["price"].rolling(WINDOW).std()
data["z_score"] = (data["price"] - data["mean"]) / data["std"]

# =========================
# SIGNAL GENERATION
# =========================
data["signal"] = 0
data.loc[data["z_score"] < -Z_ENTRY, "signal"] = 1    # Long
data.loc[data["z_score"] > Z_ENTRY, "signal"] = -1    # Short

# Shift signal to avoid look-ahead bias
data["position"] = data["signal"].shift(1)

# =========================
# RETURNS & STRATEGY
# =========================
data["returns"] = data["price"].pct_change()
data["strategy_returns"] = data["position"] * data["returns"]

# =========================
# PERFORMANCE METRICS
# =========================
equity_curve = (1 + data["strategy_returns"]).cumprod()

cagr = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
volatility = data["strategy_returns"].std() * np.sqrt(252)
sharpe_ratio = cagr / volatility if volatility != 0 else 0

# =========================
# OUTPUT
# =========================
print("===== STRATEGY PERFORMANCE =====")
print(f"CAGR: {cagr:.2%}")
print(f"Annualized Volatility: {volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# =========================
# PLOT
# =========================
plt.figure(figsize=(10, 5))
plt.plot(equity_curve, label="Strategy Equity Curve")
plt.title("Mean Reversion Strategy – Nasdaq 100")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.legend()
plt.grid(True)
plt.show()
