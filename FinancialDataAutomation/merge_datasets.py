import pandas as pd

# =========================
# PARAMETERS
# =========================
INPUT_FILE = "clean_prices.csv"
OUTPUT_FILE = "final_dataset.csv"

# =========================
# MERGE & EXPORT
# =========================
def create_final_dataset():
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

    # Compute daily returns
    returns = df.pct_change().dropna()

    # Merge prices and returns
    final_df = pd.concat(
        [df.loc[returns.index], returns],
        axis=1,
        keys=["Prices", "Returns"]
    )

    final_df.to_csv(OUTPUT_FILE)
    print(f"Final dataset saved to {OUTPUT_FILE}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    create_final_dataset()
