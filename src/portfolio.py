
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

TRADING_DAYS = 252

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD MARKET DATA
# ==========================================

def load_market_data():

    file_path = DATA_DIR / "market_prices.csv"

    prices = pd.read_csv(
        file_path,
        index_col=0,
        parse_dates=True
    )

    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.sort_index()

    if prices.empty:
        raise ValueError("The market dataset is empty.")

    if prices.columns.duplicated().any():
        raise ValueError("Duplicate asset columns detected.")

    if (prices.dropna() <= 0).any().any():
        raise ValueError("Asset prices must be positive.")

    print("Market data successfully loaded.")
    print(f"Observations: {len(prices)}")
    print(f"Assets: {len(prices.columns)}")

    return prices


# ==========================================
# 3. CALCULATE DAILY RETURNS
# ==========================================

def calculate_returns(prices):

    returns = prices.pct_change(fill_method=None)

    returns = returns.dropna(how="all")

    return returns


# ==========================================
# 4. CALCULATE PERFORMANCE METRICS
# ==========================================

def calculate_performance_metrics(prices, returns):

    mean_daily_returns = returns.mean()
    volatility_daily = returns.std(ddof=1)

    annualized_returns = mean_daily_returns * TRADING_DAYS

    annualized_volatility = (
        volatility_daily * np.sqrt(TRADING_DAYS)
    )

    observations = prices.count()
    elapsed_years = (
        (prices.index[-1] - prices.index[0]).days / 365.25
    )

    cagr = (
        (prices.iloc[-1] / prices.iloc[0])
        ** (1 / elapsed_years)
    ) - 1

    metrics = pd.DataFrame({
        "Annualized Arithmetic Return": annualized_returns,
        "CAGR": cagr,
        "Annualized Volatility": annualized_volatility,
        "Daily Mean Return": mean_daily_returns,
        "Daily Volatility": volatility_daily
    })

    return metrics


# ==========================================
# 5. COVARIANCE AND CORRELATION
# ==========================================

def calculate_risk_matrices(returns):

    covariance_matrix = returns.cov() * TRADING_DAYS

    correlation_matrix = returns.corr()

    return covariance_matrix, correlation_matrix


# ==========================================
# 6. SAVE RESULTS
# ==========================================

def save_results(returns, metrics, covariance, correlation):

    returns.to_csv(RESULTS_DIR / "daily_returns.csv")

    metrics.to_csv(RESULTS_DIR / "performance_metrics.csv")

    covariance.to_csv(RESULTS_DIR / "covariance_matrix.csv")

    correlation.to_csv(RESULTS_DIR / "correlation_matrix.csv")

    print("\nAll statistical results successfully saved.")


# ==========================================
# 7. CORRELATION HEATMAP
# ==========================================

def plot_correlation_matrix(correlation):

    plt.figure(figsize=(11, 8))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1
    )

    plt.title("Portfolio Asset Correlation Matrix")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "correlation_heatmap.png",
        dpi=300
    )

    plt.show()


# ==========================================
# 8. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    prices = load_market_data()

    returns = calculate_returns(prices)

    metrics = calculate_performance_metrics(
        prices,
        returns
    )

    covariance, correlation = calculate_risk_matrices(
        returns
    )

    print("\n===== PERFORMANCE METRICS =====")
    print((metrics * 100).round(2).to_string())

    print("\n===== CORRELATION MATRIX =====")
    print(correlation.round(3).to_string())

    save_results(
        returns,
        metrics,
        covariance,
        correlation
    )

    plot_correlation_matrix(correlation)

    print("\nSTATISTICAL ANALYSIS COMPLETED.")
