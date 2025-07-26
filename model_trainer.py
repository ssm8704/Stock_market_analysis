import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import ta
import os


st.title("Reliance Stock Price Prediction")
st.write("This application predicts stock prices using LSTM and technical indicators.")


st.sidebar.header("User Inputs")
ticker = st.sidebar.text_input("Stock Ticker", "RELIANCE.NS")
nifty_ticker = st.sidebar.text_input("Index Ticker", "^NSEI")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-01"))


feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))


model_path = "model.h5"
if not os.path.exists(model_path):
    historical_stock_data = yf.download("RELIANCE.NS", start="2010-01-01", end="2024-01-01")
    historical_nifty_data = yf.download("^NSEI", start="2010-01-01", end="2024-01-01")

    historical_stock_data = historical_stock_data[['Close']].rename(columns={'Close': 'Stock_Close'})
    historical_nifty_data = historical_nifty_data[['Close']].rename(columns={'Close': 'Nifty_Close'})
    historical_data = pd.merge(historical_stock_data, historical_nifty_data, left_index=True, right_index=True)

    # calculate technical indicators
    historical_data['SMA_20'] = historical_data['Stock_Close'].rolling(window=3).mean()
    historical_data['EMA_20'] = historical_data['Stock_Close'].ewm(span=3, adjust=False).mean()
    historical_data['Volatility'] = historical_data['Stock_Close'].rolling(window=20).std()
    historical_data['Bollinger_Upper'] = historical_data['SMA_20'] + (2 * historical_data['Volatility'])
    historical_data['Bollinger_Lower'] = historical_data['SMA_20'] - (2 * historical_data['Volatility'])
    historical_data['Daily_Return'] = historical_data['Stock_Close'].pct_change()
    historical_data['RSI'] = ta.momentum.RSIIndicator(historical_data['Stock_Close'].squeeze(), window=14).rsi()
    historical_data.dropna(inplace=True)
    # Clip Extremes
    mean_return = historical_data['Daily_Return'].mean()
    std_return = historical_data['Daily_Return'].std()
    lower_clip_std = mean_return - 3 * std_return
    upper_clip_std = mean_return + 3 * std_return
    historical_data['Clipped_Return_Std'] = historical_data['Daily_Return'].clip(lower=lower_clip_std, upper=upper_clip_std)
    historical_data['Volatility_Adjusted_Movement'] = historical_data['Clipped_Return_Std'] / historical_data['Volatility']

    # Drop unused columns
    historical_data.drop(columns=['Clipped_Return_Std', 'Daily_Return'], inplace=True)
    historical_data.dropna(inplace=True)
    # prepare and scale historical_data for training
    features = historical_data.drop(columns=['Stock_Close']).values
    target = historical_data['Stock_Close'].values.reshape(-1, 1)

    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target)

    def create_sequences(features, target, window=45, steps=1):
        X, Y = [], []
        for i in range(window, len(features) - steps):
            X.append(features[i-window:i])
            Y.append(target[i+steps])
        return np.array(X), np.array(Y)

    X_train, Y_train = create_sequences(features_scaled, target_scaled, window=45, steps=1)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=1)
    model.save(model_path)


st.write(f"Fetching data for **{ticker}** and **{nifty_ticker}**...")
stock_data = yf.download(ticker, start=start_date, end=end_date)
nifty_data = yf.download(nifty_ticker, start=start_date, end=end_date)

if stock_data.empty or nifty_data.empty:
    st.error("No data found for the selected period.")
else:
    stock_data = stock_data[['Close']].rename(columns={'Close': 'Stock_Close'})
    nifty_data = nifty_data[['Close']].rename(columns={'Close': 'Nifty_Close'})
    merged_data = pd.merge(stock_data, nifty_data, left_index=True, right_index=True)

    # calculate technical indicators
    merged_data['SMA_20'] = merged_data['Stock_Close'].rolling(window=3).mean()
    merged_data['EMA_20'] = merged_data['Stock_Close'].ewm(span=3, adjust=False).mean()
    merged_data['Volatility'] = merged_data['Stock_Close'].rolling(window=20).std()
    merged_data['Bollinger_Upper'] = merged_data['SMA_20'] + (2 * merged_data['Volatility'])
    merged_data['Bollinger_Lower'] = merged_data['SMA_20'] - (2 * merged_data['Volatility'])
    merged_data['Daily_Return'] = merged_data['Stock_Close'].pct_change()
    merged_data['RSI'] = ta.momentum.RSIIndicator(merged_data['Stock_Close'].squeeze(), window=14).rsi()
    mean_return = merged_data['Daily_Return'].mean()
    std_return = merged_data['Daily_Return'].std()
    lower_clip_std = mean_return - 3 * std_return
    upper_clip_std = mean_return + 3 * std_return
    merged_data['Clipped_Return_Std'] = merged_data['Daily_Return'].clip(lower=lower_clip_std, upper=upper_clip_std)
    merged_data['Volatility_Adjusted_Movement'] = merged_data['Clipped_Return_Std'] / merged_data['Volatility']
    merged_data.dropna(inplace=True)

    # prepare historical_data for predictions
    test_features = merged_data.drop(columns=['Stock_Close']).values
    test_target = merged_data['Stock_Close'].values.reshape(-1, 1)

    test_features_scaled = feature_scaler.fit_transform(test_features)
    test_target_scaled = target_scaler.fit_transform(test_target)

    def create_sequences_test(features, target, window=45, steps=1):
        X, Y = [], []
        for i in range(window, len(features) - steps):
            X.append(features[i-window:i])
            Y.append(target[i+steps])
        return np.array(X), np.array(Y)

    X_test, Y_test = create_sequences_test(test_features_scaled, test_target_scaled, window=45, steps=1)

    # load the saved model
    model = load_model(model_path)

    # making predictions on user selected interval
    st.write("Making predictions on user-selected data...")
    predictions = model.predict(X_test)
    predicted_prices = target_scaler.inverse_transform(predictions)[:, 0]
    actual_prices = target_scaler.inverse_transform(Y_test)[:, 0]

    # plotting the results
    st.subheader("Predicted Stock Prices (Adjusted for feature consistency)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(predicted_prices, label="Predicted Prices", color='red', linestyle='dashed')

    ax.set_title(f"Stock Price Prediction ({merged_data.index[45]} to {merged_data.index[-1]})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)



descriptive_data = merged_data.iloc[45:].copy()  
key_features = ['Stock_Close', 'Volatility', 'Daily_Return', 'RSI']


stats_table = descriptive_data[key_features].describe()
stats_table.columns = key_features  
st.subheader("Descriptive Data Table (Key Features)")
st.write(stats_table)

