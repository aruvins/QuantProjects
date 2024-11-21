import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


stockName = 'AAPL'
data = yf.download(stockName, start='2024-01-01')
plt.figure(figsize=(10, 6))
plt.plot(data['Close'])
plt.title(stockName + ' Historical Stock Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()
print(data.describe())


# #ARIMA model
stock_price = data['Close']

model = ARIMA(stock_price, order=(1, 3, 1))
model_fit = model.fit()

forecast_steps = 10
forecast_values = model_fit.forecast(steps=forecast_steps)

forecast_index = pd.date_range(start=stock_price.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')

fitted_values = model_fit.fittedvalues
fitted_values = fitted_values[1:]

plt.figure(figsize=(10, 6))
plt.plot(stock_price, label='Actual')
plt.plot(stock_price.index[1:], fitted_values, color='red', label='Predicted')
plt.plot(forecast_index, forecast_values, color='green', linestyle='dashed', label='Forecasted (Next '+ str(forecast_steps) + ' Days)')
plt.title('ARIMA Model - ' + stockName + ' Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


# # GARCH model
returns = stock_price.pct_change().dropna()
garch_model = arch_model(returns, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit()

print(garch_fit.summary())

volatility_forecast = garch_fit.conditional_volatility
plt.figure(figsize=(10, 6))
plt.plot(volatility_forecast)
plt.title('GARCH(1,1) Model - Volatility Forecast')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.grid(True)
plt.show()


#LSTM Model
data = yf.download(stockName, start='2014-01-01')
stock_price = data['Close']

units = 150
layers = 4
dropout_rate = 0.4
learning_rate = 0.001
epochs = 200
batch_size = 64
look_back=90

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_price.values.reshape(-1, 1))

def create_dataset(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, look_back=60)
X = X.reshape(X.shape[0], X.shape[1], 1)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the model
def create_model(units, layers, dropout_rate, learning_rate):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(dropout_rate))

    for _ in range(layers - 1):
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    model.add(LSTM(units=units))
    model.add(Dense(units=1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model = create_model(units=units, layers=layers, dropout_rate=dropout_rate, learning_rate=learning_rate)
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping])

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Price')
plt.plot(predictions, label='Predicted Price', color='red')
plt.title('LSTM Model - '+ stockName +' Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

rmse = math.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print(f"RMSE for LSTM: {rmse}")
print(f"MAE for LSTM: {mae}")