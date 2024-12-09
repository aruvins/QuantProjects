from matplotlib import pyplot as plt
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

stock1 = yf.download("AAPL", start="2015-01-01", end="2023-01-01")
stock2 = yf.download("MSFT", start="2015-01-01", end="2023-01-01")
prices = pd.DataFrame({
    "stock1": stock1["Close"],
    "stock2": stock2["Close"]
})

score, p_value, _ = coint(prices["stock1"], prices["stock2"])
if p_value < 0.05:
    print("cointegrated")

X = sm.add_constant(prices["stock2"])
model = sm.OLS(prices["stock1"], X).fit()

spread = prices["stock1"] - model.predict(X)
spread_mean = spread.mean()
spread_std = spread.std()

z_score = (spread - spread_mean) / spread_std
entry_threshold = 0.75
exit_threshold = 0

plt.figure(figsize=(10, 6))
plt.plot(spread, label="Spread")
plt.axhline(spread_mean, color="black", linestyle="--", label="Mean")
plt.axhline(spread_mean + entry_threshold * spread_std, color="red", linestyle="--", label="Entry Threshold")
plt.axhline(spread_mean - entry_threshold * spread_std, color="green", linestyle="--", label="Entry Threshold")
plt.legend()
plt.title("Spread and Trading Signals")
plt.show()



# #backtesting
signals = pd.DataFrame(index=spread.index)
signals["long"] = z_score < -entry_threshold
signals["short"] = z_score > entry_threshold
signals["exit"] = abs(z_score) < exit_threshold

capital = 100000
positions = pd.DataFrame(index=spread.index)
portfolio_value = [] 
cash = capital
position_size = 100
long_position = 0
short_position = 0

for date, signal in signals.iterrows():
    if signal["long"] and cash >= position_size * prices["stock1"][date]:
        long_position += position_size
        short_position -= position_size
        cash -= position_size * prices["stock1"][date]
        cash += position_size * prices["stock2"][date]
        
    elif signal["short"] and cash >= position_size * prices["stock2"][date]:
        short_position += position_size
        long_position -= position_size
        cash += position_size * prices["stock1"][date]
        cash -= position_size * prices["stock2"][date]

    elif signal["exit"]:
        cash += long_position * prices["stock1"][date]
        cash -= short_position * prices["stock2"][date]
        long_position = 0
        short_position = 0

    portfolio_value.append(cash + long_position * prices["stock1"][date] + short_position * prices["stock2"][date])

plt.figure(figsize=(10, 6))
plt.plot(portfolio_value, label="Portfolio Value")
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.show()



