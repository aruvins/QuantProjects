from matplotlib import pyplot as plt
import yfinance as yf

stock = 'NVDA'
data = yf.download(stock, start='2020-01-01', end='2024-01-01')
data['SMA_60'] = data['Close'].rolling(window=60).mean()
data['SMA_180'] = data['Close'].rolling(window=180).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()  # Calculate price changes
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # Average gains
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # Average losses
    rs = gain / loss  # Relative Strength
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    return rsi


#buy and hold
data.loc[:, 'Signal'] = 1

#Simple SMA crossing strategy
# data['Signal'] = 0
# data.loc[data['SMA_60'] > data['SMA_180'], 'Signal'] = 1  # Buy signal
# data.loc[data['SMA_0'] < data['SMA_180'], 'Signal'] = -1  # Sell signal

#Buy if price is above SMA and sell if price is below SMA
# data['Signal'] = 0  # Initialize Signal column
# data.loc[data['Close'] > data['SMA_180'], 'Signal'] = 1  # Buy signal if price is above SMA
# data.loc[data['Close'] < data['SMA_180'], 'Signal'] = -1  # Sell signal if price is below SMA

#Buy if RSI is below or equal to 30 and sell if RSI is above 70
# data['RSI'] = calculate_rsi(data)
# data['Signal'] = 0
# data.loc[data['RSI'] <= 30, 'Signal'] = 1  # Buy signal when RSI < 30
# data.loc[data['RSI'] >= 70, 'Signal'] = -1  # Sell signal when RSI > 70

#combine RSI and SMA strategy
# data['RSI'] = calculate_rsi(data)
# data['Signal'] = 0
# data.loc[(data['RSI'] <= 30) | (data['SMA_60'] > data['SMA_180']), 'Signal'] = 1  # Buy signal when RSI < 30
# data.loc[(data['RSI'] >= 70) | (data['SMA_60'] < data['SMA_180']), 'Signal'] = -1  # Sell signal when RSI > 70


#Simulation of strategy
initial_capital = 10000
positions = initial_capital * data['Signal'].shift(1) * data['Close'].pct_change()
portfolio = positions.cumsum() + initial_capital

cagr = ((portfolio[-1] / portfolio[0]) ** (1 / len(data) * 252)) - 1
max_drawdown = (portfolio / portfolio.cummax()).min() - 1


plt.figure(figsize=(12, 6))
plt.plot(portfolio, label='Portfolio Value')
plt.legend()
plt.title('Equity Curve of ' + stock)
plt.xlabel('Time')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.show()