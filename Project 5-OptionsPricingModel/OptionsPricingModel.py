


# # Example data
# # S = 1208                                # Current stock price
# # K = 1210                                # Strike price
# # T = 7/365                               # Time to expiration (1 week converted to years)
# # r = 2.05051/100                            # Risk-free interest rate (US Real Interest Rate)
# # implied_volatility_value = 0.57         # Implied volatility (converted from percentage)
# # historic_volatility_value = 0.47        # Historic volatility (converted from percentage)

# # S = underlying price ($$$ per share)
# # K = strike price ($$$ per share)
# # σ(sigma) = volatility (% p.a.)
# # r = continuously compounded risk-free interest rate (% p.a.)
# # q = continuously compounded dividend yield (% p.a.)
# # t = time to expiration (% of year)



from numpy import exp, sqrt, log
from scipy.stats import norm
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

class BlackScholes:
    def __init__(self, time_to_maturity: float, strike: float, current_price: float, volatility: float, interest_rate: float):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def _calculate_d1_d2(self):
        # Calculate d1 and d2 used in the Black-Scholes formula
        d1 = (np.log(self.current_price / self.strike) + (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / (self.volatility * np.sqrt(self.time_to_maturity))
        d2 = d1 - self.volatility * np.sqrt(self.time_to_maturity)
        return d1, d2

    # Call Option Greeks and Price

    def _call_delta(self, d1):
        # Delta is the first derivative of option price with respect to underlying price
        return norm.cdf(d1)

    def _call_gamma(self, d1):
        # Gamma is the second derivative of option price with respect to underlying price
        return norm.pdf(d1) / (self.current_price * self.volatility * np.sqrt(self.time_to_maturity))

    def _call_theta(self, d1, d2):
        # Theta is the first derivative of option price with respect to time to expiration
        theta = -(self.current_price * norm.pdf(d1) * self.volatility) / (2 * np.sqrt(self.time_to_maturity)) - self.interest_rate * self.strike * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)
        return theta / 365

    def _call_vega(self, d1):
        # Vega is the first derivative of option price with respect to volatility σ(Sigma)
        return self.current_price * norm.pdf(d1) * np.sqrt(self.time_to_maturity) / 100

    def _call_rho(self, d2):
        # Rho is the first derivative of option price with respect to interest rate
        return self.strike * self.time_to_maturity * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)

    # Put Option Greeks and Price

    def _put_delta(self, d1):
        # Delta for put option
        return norm.cdf(d1) - 1

    def _put_gamma(self, d1):
        # Gamma for put option
        return norm.pdf(d1) / (self.current_price * self.volatility * np.sqrt(self.time_to_maturity))

    def _put_theta(self, d1, d2):
        # Theta for put option
        theta = -(self.current_price * norm.pdf(d1) * self.volatility) / (2 * np.sqrt(self.time_to_maturity)) + self.interest_rate * self.strike * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)
        return theta / 365

    def _put_vega(self, d1):
        # Vega for put option
        return self.current_price * norm.pdf(d1) * np.sqrt(self.time_to_maturity) / 100

    def _put_rho(self, d2):
        # Rho for put option
        return -self.strike * self.time_to_maturity * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)

    # Call and Put Prices

    def _call_price(self, d1, d2):
        # Calculate call option price
        return self.current_price * norm.cdf(d1) - (self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2))

    def _put_price(self, d1, d2):
        # Calculate put option price
        return (self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)) - self.current_price * norm.cdf(-d1)

    def calculateEverything(self):
        # Run Black-Scholes calculations and print results
        d1, d2 = self._calculate_d1_d2()

        call_delta = self._call_delta(d1)
        call_gamma = self._call_gamma(d1)
        call_theta = self._call_theta(d1, d2)
        call_vega = self._call_vega(d1)
        call_rho = self._call_rho(d2)
        call_price = self._call_price(d1, d2)

        put_delta = self._put_delta(d1)
        put_gamma = self._put_gamma(d1)
        put_theta = self._put_theta(d1, d2)
        put_vega = self._put_vega(d1)
        put_rho = self._put_rho(d2)
        put_price = self._put_price(d1, d2)

        results = {
            'Call Delta': call_delta,
            'Call Gamma': call_gamma,
            'Call Theta': call_theta,
            'Call Vega': call_vega,
            'Call Rho': call_rho,
            'Call Price': call_price,
            'Put Delta': put_delta,
            'Put Gamma': put_gamma,
            'Put Theta': put_theta,
            'Put Vega': put_vega,
            'Put Rho': put_rho,
            'Put Price': put_price
        }

        return results
    def calculate_prices(self):
        d1, d2 = self._calculate_d1_d2()
        call_price = self._call_price(d1, d2)
        put_price = self._put_price(d1, d2)
        return call_price, put_price

    
    def calculate_call_greeks(self):
        d1, d2 = self._calculate_d1_d2()
        greeks = {
            'Call Delta': round(self._call_delta(d1), 2),
            'Call Gamma': round(self._call_gamma(d1), 2),
            'Call Theta': round(self._call_theta(d1, d2), 2),
            'Call Vega': round(self._call_vega(d1), 2),
            'Call Rho': round(self._call_rho(d2), 2)
        }
        return greeks

    def calculate_put_greeks(self):
        d1, d2 = self._calculate_d1_d2()
        greeks = {
            'Put Delta': round(self._put_delta(d1), 2),
            'Put Gamma': round(self._put_gamma(d1), 2),
            'Put Theta': round(self._put_theta(d1, d2), 2),
            'Put Vega': round(self._put_vega(d1), 2),
            'Put Rho': round(self._put_rho(d2), 2)
        }
        return greeks

def plot_heatmap(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            call_prices[i, j], put_prices[i, j] = bs_temp.calculate_prices()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), 
                yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", 
                cmap="viridis", ax=axes[0])
    axes[0].set_title("CALL Price Heatmap")
    axes[0].set_xlabel("Spot Price")
    axes[0].set_ylabel("Volatility")

    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), 
                yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", 
                cmap="viridis", ax=axes[1])
    axes[1].set_title("PUT Price Heatmap")
    axes[1].set_xlabel("Spot Price")
    axes[1].set_ylabel("Volatility")

    plt.tight_layout()
    plt.show()

def plot_call_greeks_correlation(bs_model, spot_range, vol_range):
    greeks_list = []

    for vol in vol_range:
        for spot in spot_range:
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=bs_model.strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            greeks = bs_temp.calculate_call_greeks()
            greeks['Spot Price'] = spot
            greeks['Volatility'] = vol
            greeks_list.append(greeks)

    greeks_df = pd.DataFrame(greeks_list)
    correlation_matrix = greeks_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Between Greeks and Other Parameters")
    plt.show()

def plot_put_greeks_correlation(bs_model, spot_range, vol_range):
    greeks_list = []

    for vol in vol_range:
        for spot in spot_range:
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=bs_model.strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            greeks = bs_temp.calculate_put_greeks()
            greeks['Spot Price'] = spot
            greeks['Volatility'] = vol
            greeks_list.append(greeks)

    greeks_df = pd.DataFrame(greeks_list)
    correlation_matrix = greeks_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Between Greeks and Other Parameters")
    plt.show()


if __name__ == "__main__":
    bs_model = BlackScholes(time_to_maturity=7/365, strike=1210, current_price=1208, volatility=0.57, interest_rate=2.05/100)
    spot_range = np.linspace(1150, 1250, 5)
    vol_range = np.linspace(0.4, 0.6, 5)

    print(bs_model.calculateEverything())
    # Plot Heatmap
    plot_heatmap(bs_model, spot_range, vol_range, bs_model.strike)

    # Plot Call Greeks Correlation
    plot_call_greeks_correlation(bs_model, spot_range, vol_range)
    plot_put_greeks_correlation(bs_model, spot_range, vol_range)