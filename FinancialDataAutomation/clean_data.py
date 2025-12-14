import pandas as pd

# =========================
# PARAMETERS
# =========================
INPUT_FILE = "raw_prices.csv"
OUTPUT_FILE = "clean_prices.csv"

# =========================
# CLEAN DATA
# =========================
def clean_market_data():
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

    # Forward fill missing values
    df = df.fillna(method="ffill")

    # Drop rows still containing NaN
    df = df.dropna()

    df.to_csv(OUTPUT_FILE)
    print(f"Cleaned data saved to {OUTPUT_FILE}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    clean_market_data()
