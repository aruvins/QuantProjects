from matplotlib import pyplot as plt
import requests
import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def fetch_data(symbol="bitcoin", currency="usd", days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {"vs_currency": currency, "days": days}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def add_features(df):
    df['SMA_20'] = ta.sma(df['price'], length=20)
    df['EMA_20'] = ta.ema(df['price'], length=20)
    df['volatility'] = df['price'].pct_change().rolling(window=20).std()
    return df

def sharpe_ratio(df, risk_free_rate=0.02):
    daily_returns = df['price'].pct_change()
    avg_return = daily_returns.mean()
    volatility = daily_returns.std()
    sharpe_ratio = (avg_return - risk_free_rate) / volatility
    return sharpe_ratio

crypto_data = fetch_data()
crypto_data = add_features(crypto_data)
crypto_data.dropna(inplace=True)

X = crypto_data[['SMA_20', 'EMA_20', 'volatility']]
y = crypto_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

sharpe_ratio = sharpe_ratio(crypto_data)
print(f"Sharpe Ratio: {sharpe_ratio}")

plt.figure(figsize=(12, 6))
plt.plot(crypto_data.index[-len(y_test):], y_test, label='Actual Prices', color='blue', alpha=0.7)
plt.plot(crypto_data.index[-len(y_pred):], y_pred, label='Predicted Prices', color='orange', alpha=0.7)

plt.title('Actual vs Predicted Cryptocurrency Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()