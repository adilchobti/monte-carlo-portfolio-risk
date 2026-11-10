
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# ==========================================
# 0. VISUAL IDENTITY — BRITISH RACING GREEN
# ==========================================
CHART_BG = "#142019"
AXES_BG = "#1b2a21"
TEXT_COLOR = "#e7e5dc"
MUTED_COLOR = "#a5a58d"
GRID_COLOR = "#405247"

def apply_portfolio_style():
    """Apply the portfolio's dark green visual identity to all charts."""
    plt.rcParams.update({
        "figure.facecolor": CHART_BG, "axes.facecolor": AXES_BG,
        "savefig.facecolor": CHART_BG, "savefig.edgecolor": CHART_BG,
        "text.color": TEXT_COLOR, "axes.labelcolor": TEXT_COLOR,
        "axes.edgecolor": GRID_COLOR, "xtick.color": MUTED_COLOR,
        "ytick.color": MUTED_COLOR, "font.family": "DejaVu Sans",
        "font.size": 10, "axes.titlesize": 14, "axes.titleweight": "medium",
        "axes.labelsize": 10, "legend.facecolor": AXES_BG,
        "legend.edgecolor": GRID_COLOR, "legend.labelcolor": TEXT_COLOR,
        "grid.color": GRID_COLOR, "grid.alpha": 0.42,
    })

apply_portfolio_style()




# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

N_SIMULATIONS = 10_000
N_DAYS = 252
TRADING_DAYS = 252

INITIAL_PORTFOLIO_VALUE = 100_000

RANDOM_SEED = 42

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. PORTFOLIO CONFIGURATION
# ==========================================

def create_portfolio_weights(assets):

    n_assets = len(assets)

    weights = np.ones(n_assets) / n_assets

    return pd.Series(weights, index=assets)


# ==========================================
# 3. LOAD HISTORICAL DATA
# ==========================================

def load_historical_data():

    file_path = DATA_DIR / "market_prices.csv"

    prices = pd.read_csv(
        file_path,
        index_col=0,
        parse_dates=True
    )

    prices = prices.apply(
        pd.to_numeric,
        errors="coerce"
    )

    prices = prices.sort_index()

    if prices.empty:
        raise ValueError("Historical price dataset is empty.")

    if prices.columns.duplicated().any():
        raise ValueError("Duplicate asset columns detected.")

    if prices.isna().any().any():
        raise ValueError("Missing prices detected.")

    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive.")

    log_returns = np.log(
        prices / prices.shift(1)
    )

    log_returns = log_returns.dropna()

    if len(log_returns) < 100:
        raise ValueError("Insufficient historical observations.")

    print("Historical data successfully loaded.")
    print(f"Observations: {len(log_returns)}")
    print(f"Assets: {len(prices.columns)}")

    return prices, log_returns


# ==========================================
# 4. ESTIMATE MODEL PARAMETERS
# ==========================================

def estimate_model_parameters(log_returns):

    # Annualized mean logarithmic returns
    annual_log_returns = (
        log_returns.mean() * TRADING_DAYS
    )

    # Annualized covariance matrix
    covariance_matrix = (
        log_returns.cov() * TRADING_DAYS
    )

    # Ensure matrix symmetry
    covariance_matrix = (
        covariance_matrix + covariance_matrix.T
    ) / 2

    # Extract individual asset variances
    variances = np.diag(
        covariance_matrix.to_numpy()
    )

    # Convert expected log returns into GBM drift
    mu = annual_log_returns + 0.5 * variances

    return mu, covariance_matrix


# ==========================================
# 5. VALIDATE COVARIANCE MATRIX
# ==========================================

def calculate_cholesky(covariance_matrix):

    covariance = covariance_matrix.to_numpy()

    if not np.all(np.isfinite(covariance)):
        raise ValueError(
            "Covariance matrix contains invalid values."
        )

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvalsh(covariance)

    min_eigenvalue = eigenvalues.min()

    print("\n===== COVARIANCE VALIDATION =====")

    print(
        f"Minimum eigenvalue: {min_eigenvalue:.10f}"
    )

    if min_eigenvalue < -1e-8:
        raise ValueError(
            "Covariance matrix is not positive semidefinite."
        )

    # Numerical adjustment if necessary
    jitter = max(
        0.0,
        1e-10 - min_eigenvalue
    )

    adjusted_covariance = (
        covariance
        + np.eye(len(covariance)) * jitter
    )

    # Cholesky decomposition
    cholesky_matrix = np.linalg.cholesky(
        adjusted_covariance
    )

    print("Cholesky decomposition successful.")

    return cholesky_matrix


# ==========================================
# 6. MONTE CARLO SIMULATION ENGINE
# ==========================================

def run_monte_carlo(
    initial_prices,
    mu,
    cholesky_matrix,
    weights
):

    rng = np.random.default_rng(RANDOM_SEED)

    n_assets = len(initial_prices)

    dt = 1 / TRADING_DAYS

    initial_prices_array = initial_prices.to_numpy()

    # Initialize simulated asset prices
    asset_prices = np.tile(
        initial_prices_array,
        (N_SIMULATIONS, 1)
    )

    # Store portfolio values for all scenarios
    portfolio_values = np.zeros(
        (N_SIMULATIONS, N_DAYS + 1)
    )

    portfolio_values[:, 0] = (
        INITIAL_PORTFOLIO_VALUE
    )

    # Calculate annualized asset variances
    variances = np.sum(
        cholesky_matrix ** 2,
        axis=1
    )

    # GBM drift component
    drift = (
        mu.to_numpy() - 0.5 * variances
    ) * dt

    weights_array = weights.to_numpy()

    print(
        f"\nRunning {N_SIMULATIONS:,} Monte Carlo simulations..."
    )

    # Simulate each trading day
    for day in range(1, N_DAYS + 1):

        # Generate independent Gaussian shocks
        independent_shocks = rng.standard_normal(
            (N_SIMULATIONS, n_assets)
        )

        # Introduce historical correlations
        correlated_shocks = (
            independent_shocks @ cholesky_matrix.T
        )

        # Calculate daily logarithmic returns
        log_returns = (
            drift
            + np.sqrt(dt) * correlated_shocks
        )

        # Update simulated asset prices
        asset_prices *= np.exp(log_returns)

        # Calculate relative asset prices
        relative_prices = (
            asset_prices / initial_prices_array
        )

        # Calculate buy-and-hold portfolio value
        portfolio_values[:, day] = (
            INITIAL_PORTFOLIO_VALUE
            * (relative_prices @ weights_array)
        )

    print("Simulation completed successfully.")

    return portfolio_values


# ==========================================
# 7. SAVE SIMULATION RESULTS
# ==========================================

def save_simulation_results(portfolio_values):

    final_values = portfolio_values[:, -1]

    # Save individual simulation outcomes
    results = pd.DataFrame({
        "Final Portfolio Value": final_values
    })

    results.to_csv(
        RESULTS_DIR / "monte_carlo_results.csv",
        index=False
    )

    # Calculate descriptive statistics
    summary = pd.Series({

        "Initial Portfolio Value":
            INITIAL_PORTFOLIO_VALUE,

        "Mean Final Value":
            np.mean(final_values),

        "Median Final Value":
            np.median(final_values),

        "5th Percentile":
            np.percentile(final_values, 5),

        "95th Percentile":
            np.percentile(final_values, 95),

        "Minimum Final Value":
            np.min(final_values),

        "Maximum Final Value":
            np.max(final_values)

    })

    summary.to_csv(
        RESULTS_DIR / "monte_carlo_summary.csv",
        header=["Value"]
    )

    print("\n===== MONTE CARLO RESULTS =====")

    print(
        summary.round(2).to_string()
    )

    return summary


# ==========================================
# 8. VISUALIZE SIMULATED PATHS
# ==========================================

def plot_simulation_paths(portfolio_values):

    fig, ax = plt.subplots(figsize=(12, 7))

    # Display 100 simulated portfolio paths
    for i in range(100):

        ax.plot(portfolio_values[i], color="#a5a58d", alpha=0.16, linewidth=0.75)

    ax.axhline(INITIAL_PORTFOLIO_VALUE, color="#d4c9a8", linestyle="--", linewidth=1.25, label="Initial Portfolio Value")

    ax.set_title("Monte Carlo Portfolio Simulations", color=TEXT_COLOR, pad=16)
    ax.set_xlabel("Trading Days", labelpad=10)
    ax.set_ylabel("Portfolio Value ($)", labelpad=10)
    ax.legend(framealpha=1, fontsize=9)
    ax.grid(True, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=2)

    plt.savefig(
        RESULTS_DIR / "monte_carlo_paths.png",
        dpi=300
    )

    plt.show()


# ==========================================
# 9. FINAL VALUE DISTRIBUTION
# ==========================================

def plot_final_distribution(portfolio_values):

    final_values = portfolio_values[:, -1]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(final_values, bins=60, density=True, color="#91a98c", edgecolor=AXES_BG, linewidth=0.55, alpha=0.92)

    ax.axvline(INITIAL_PORTFOLIO_VALUE, color="#e7e5dc", linestyle="--", linewidth=1.3, label="Initial Portfolio Value")

    ax.axvline(np.percentile(final_values, 5), color="#c49b75", linestyle="--", linewidth=1.3, label="5th Percentile")

    ax.set_title("Monte Carlo Final Portfolio Value Distribution", color=TEXT_COLOR, pad=16)
    ax.set_xlabel("Final Portfolio Value ($)", labelpad=10)
    ax.set_ylabel("Probability Density", labelpad=10)
    ax.legend(framealpha=1, fontsize=9)
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=2)

    plt.savefig(
        RESULTS_DIR / "monte_carlo_distribution.png",
        dpi=300
    )

    plt.show()


# ==========================================
# 10. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    # Load historical market data
    prices, log_returns = load_historical_data()

    # Define portfolio assets and weights
    assets = prices.columns

    weights = create_portfolio_weights(assets)

    # Estimate annualized model parameters
    mu, covariance = estimate_model_parameters(
        log_returns
    )

    # Validate covariance matrix
    cholesky_matrix = calculate_cholesky(
        covariance
    )

    # Use latest historical prices as initial prices
    initial_prices = prices.iloc[-1]

    print("\n===== PORTFOLIO WEIGHTS =====")

    print(
        (weights * 100).round(2).to_string()
    )

    # Run Monte Carlo simulations
    portfolio_values = run_monte_carlo(
        initial_prices,
        mu,
        cholesky_matrix,
        weights
    )

    # Save numerical results
    save_simulation_results(
        portfolio_values
    )

    # Generate visualizations
    plot_simulation_paths(
        portfolio_values
    )

    plot_final_distribution(
        portfolio_values
    )

    print(
        "\nMONTE CARLO ENGINE COMPLETED SUCCESSFULLY."
    )
