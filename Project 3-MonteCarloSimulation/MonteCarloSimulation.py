import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Geometric Brownian Motion Simulator
def simulate_gbm(S0, mu, sigma, T, dt, simulations):
    steps = int(T / dt)
    time = np.linspace(0, T, steps)
    prices = np.zeros((steps, simulations))
    prices[0] = S0
    for t in range(1, steps):
        dW = np.random.normal(0, np.sqrt(dt), simulations)
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
    return time, prices

def fetch_stock_data_and_metrics(ticker, period="1y"):
    data = yf.download(ticker, period=period)
    historical_prices = data['Close']
    daily_returns = np.log(historical_prices / historical_prices.shift(1)).dropna()
    mu = daily_returns.mean() * 252
    sigma = daily_returns.std() * np.sqrt(252)
    current_price = historical_prices.iloc[-1]
    return current_price, mu, sigma, historical_prices

def plot_historical(historical, current_price):
    plt.figure(figsize=(12, 6))
    plt.plot(historical, color='red', label='Historical Data')
    plt.axhline(current_price, color='blue', linestyle='--', label=f'Current Price: {current_price:.2f}')
    plt.title("Historical Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def plot_simulations(time, simulated, confidence_interval):
    plt.figure(figsize=(12, 6))
    plt.plot(time, simulated, alpha=0.1, color='blue')
    plt.fill_between(time, confidence_interval[0], confidence_interval[1], color='blue', alpha=0.2, label='Confidence Interval')
    plt.title("Simulated Stock Prices")
    plt.xlabel("Time (Years)")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def plot_final_price_distribution(final_prices, current_price):
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(final_prices, bins=100, color='blue', alpha=0.5, edgecolor='black')
    
    mean_price = np.mean(final_prices)
    median_price = np.median(final_prices)
    mode_price = bins[np.argmax(counts)] 
    lower_bound = np.percentile(final_prices, 5)
    upper_bound = np.percentile(final_prices, 95)

    plt.axvline(mean_price, color='red', linestyle='--', label=f'Mean: {mean_price:.2f}')
    plt.axvline(median_price, color='green', linestyle='--', label=f'Median: {median_price:.2f}')
    plt.axvline(mode_price, color='purple', linestyle='--', label=f'Mode: {mode_price:.2f}')
    plt.axvline(lower_bound, color='orange', linestyle='--', label=f'5% CI: {lower_bound:.2f}')
    plt.axvline(upper_bound, color='cyan', linestyle='--', label=f'95% CI: {upper_bound:.2f}')
    plt.axvline(current_price, color='blue', linestyle='-', label=f'Current Price: {current_price:.2f}')
    
    plt.title("Distribution of Final Stock Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# Main Implementation
def monte_carlo_simulation(ticker, T, dt, simulations):
    S0, mu, sigma, historical = fetch_stock_data_and_metrics(ticker)
    time, simulated = simulate_gbm(S0, mu, sigma, T, dt, simulations)

    final_prices = simulated[-1]
    lower_bound = np.percentile(final_prices, 5)
    upper_bound = np.percentile(final_prices, 95)
    # Plot Charts
    plot_historical(historical, S0)
    plot_simulations(time, simulated, (lower_bound, upper_bound))
    plot_final_price_distribution(final_prices, S0)

# Parameters
ticker = "GME"
T = 10/252
dt = 1/252 #number of minutes the stock market is open
simulations = 1000

# Run Monte Carlo Simulation
monte_carlo_simulation(ticker, T, dt, simulations)
