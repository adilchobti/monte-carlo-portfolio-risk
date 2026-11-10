
import yfinance as yf
import pandas as pd
from pathlib import Path

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "SPY",
    "TLT",
    "GLD"
]

START_DATE = "2021-01-01"
END_DATE = None

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. DOWNLOAD MARKET DATA
# ==========================================

def download_market_data(tickers, start_date, end_date=None):

    print("Downloading historical market data...")

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No market data retrieved.")

    # Extract adjusted closing prices
    prices = data["Close"]

    # Remove assets with missing observations
    prices = prices.dropna(how="all")

    # Forward-fill occasional missing observations
    prices = prices.ffill()

    return prices


# ==========================================
# 3. SAVE MARKET DATA
# ==========================================

def save_market_data(prices):

    output_path = DATA_DIR / "market_prices.csv"

    prices.to_csv(output_path)

    print(f"\nData successfully saved to: {output_path}")


# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    prices = download_market_data(
        TICKERS,
        START_DATE,
        END_DATE
    )

    print("\nHistorical prices:")
    print(prices.tail())

    print("\nDataset information:")
    print(f"Number of observations: {len(prices)}")
    print(f"Number of assets: {len(prices.columns)}")

    save_market_data(prices)
