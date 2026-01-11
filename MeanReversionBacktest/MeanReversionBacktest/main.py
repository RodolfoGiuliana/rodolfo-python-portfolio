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
# DOWNLOAD & FIX DATA
# =========================
data = yf.download(TICKER, start=START_DATE, progress=False)

# FIX per MultiIndex: se yfinance restituisce colonne doppie, prendiamo solo 'Close'
if isinstance(data.columns, pd.MultiIndex):
    data = data['Close'][[TICKER]].rename(columns={TICKER: "price"})
else:
    data = data[["Close"]].rename(columns={"Close": "price"})


data = data.dropna()

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
# Long quando il prezzo è basso (z-score < -1)
data.loc[data["z_score"] < -Z_ENTRY, "signal"] = 1    
# Short quando il prezzo è alto (z-score > 1)
data.loc[data["z_score"] > Z_ENTRY, "signal"] = -1   

# Shift per evitare look-ahead bias 
data["position"] = data["signal"].shift(1)

# =========================
# RETURNS & STRATEGY
# =========================
data["returns"] = data["price"].pct_change()
data["strategy_returns"] = data["position"] * data["returns"]

# =========================
# PERFORMANCE METRICS
# =========================
# Equity curve partendo da 1 (base 100%)
# fillna(0) per i giorni senza trading
strategy_rets_clean = data["strategy_returns"].fillna(0)
equity_curve = (1 + strategy_rets_clean).cumprod()

# Calcolo metriche annualizzate
trading_days = 252
years = len(data) / trading_days
final_return = equity_curve.iloc[-1]
cagr = (final_return ** (1 / years)) - 1 if final_return > 0 else -1
volatility = strategy_rets_clean.std() * np.sqrt(trading_days)
sharpe_ratio = cagr / volatility if volatility != 0 else 0

# =========================
# OUTPUT
# =========================
print(f"===== STRATEGY PERFORMANCE: {TICKER} =====")
print(f"CAGR (Rendimento Annuo): {cagr:.2%}")
print(f"Volatilità Annua: {volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# =========================
# PLOT
# =========================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot 1: Price and Bollinger-like Bands
ax1.plot(data.index, data["price"], label="Price", alpha=0.6, color='blue')
ax1.plot(data.index, data["mean"], label="Moving Average", color="orange", linestyle='--')
ax1.fill_between(data.index, 
                 data["mean"] + Z_ENTRY * data["std"], 
                 data["mean"] - Z_ENTRY * data["std"], 
                 color="gray", alpha=0.2, label="Trading Range")
ax1.set_title(f"Price Action & Mean Reversion Bands ({TICKER})", fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Equity Curve
ax2.fill_between(equity_curve.index, 1, equity_curve, color='green', alpha=0.2)
ax2.plot(equity_curve.index, equity_curve, label="Strategy Wealth", color="green", lw=1.5)
ax2.set_title("Equity Curve (Growth of $1)", fontweight='bold')
ax2.set_ylabel("Multiple of Initial Capital")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
