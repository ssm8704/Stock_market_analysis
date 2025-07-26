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
import mplfinance as mpf

st.title("Reliance Stock Price Prediction")
st.write("This application predicts stock prices using LSTM and technical indicators.")


st.sidebar.header("User Inputs")
ticker = st.sidebar.text_input("Stock Ticker", "RELIANCE.NS")
nifty_ticker = st.sidebar.text_input("Index Ticker", "^NSEI")
#start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
#end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-01"))


feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))
nifty_feature_scaler = MinMaxScaler(feature_range=(0,1))
nifty_target_scaler = MinMaxScaler(feature_range=(0,1))
def indexer(data, name):
    data.to_csv(f"{name}.csv")
    output_df = pd.read_csv(f"{name}.csv")
    # Drop the first two rows as they are now part of the header
    new_header = output_df.iloc[0].combine_first(output_df.iloc[1])
    output_df.columns = new_header
    output_df = output_df[2:]
    # Step 2: Assign consistent column names (using the ones in data.csv as reference)
    output_df.columns = [
        "Date", "Close", "High", "Low", "Open", "Volume"
    ]
    # # Ensure correct data types
    output_df = output_df.reset_index(drop=True)
    output_df["Date"] = pd.to_datetime(output_df["Date"], errors='coerce')
    output_df[["Close", "High", "Low", "Open", "Volume"]] = output_df[[
        "Close", "High", "Low", "Open", "Volume"
    ]].apply(pd.to_numeric, errors='coerce')
    return output_df
def data_preprocessor(data):
    data['SMA_20'] = data['Stock_Close'].rolling(window=3).mean()
    data['EMA_20'] = data['Stock_Close'].ewm(span=3, adjust=False).mean()
    data['Volatility'] = data['Stock_Close'].rolling(window=20).std()
    data['Bollinger_Upper'] = data['SMA_20'] + (2 * data['Volatility'])
    data['Bollinger_Lower'] = data['SMA_20'] - (2 * data['Volatility'])
    data['Daily_Return'] = data['Stock_Close'].pct_change()
    data['RSI'] = ta.momentum.RSIIndicator(data['Stock_Close'].squeeze(), window=14).rsi()

    # Clip Extremes
    mean_return = data['Daily_Return'].mean()
    std_return = data['Daily_Return'].std()
    lower_clip_std = mean_return - 3 * std_return
    upper_clip_std = mean_return + 3 * std_return
    data['Clipped_Return_Std'] = data['Daily_Return'].clip(lower=lower_clip_std, upper=upper_clip_std)
    data['Volatility_Adjusted_Movement'] = data['Clipped_Return_Std'] / data['Volatility']

    # Drop unused columns
    data.drop(columns=['Clipped_Return_Std', 'Daily_Return'], inplace=True)
    data.dropna(inplace=True)
    return data
    
model_path = "model.h5"
if not os.path.exists(model_path):
    historical_stock_data = yf.download("RELIANCE.NS", start="2010-01-01", end="2024-01-01")
    historical_nifty_data = yf.download("^NSEI", start="2010-01-01", end="2024-01-01")
    historical_stock_data = indexer(historical_stock_data,'stock_train')
    historical_nifty_data = indexer(historical_nifty_data,'nifty_train')
    historical_stock_data = historical_stock_data[['Close']].rename(columns={'Close': 'Stock_Close'})
    historical_nifty_data = historical_nifty_data[['Close']].rename(columns={'Close': 'Nifty_Close'})
    historical_data = pd.merge(historical_stock_data, historical_nifty_data, left_index=True, right_index=True)

    # calculate technical indicators
    historical_data = data_preprocessor(historical_data)
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
# ticker = "RELIANCE.NS"
# nifty_ticker = "^NSEI"
end_date = pd.Timestamp.today()
start_date = end_date - pd.DateOffset(months=4)

stock_data = yf.download(ticker, start=start_date, end=end_date)
nifty_data = yf.download(nifty_ticker, start=start_date, end=end_date)

if stock_data.empty or nifty_data.empty:
    st.error("No data found for the selected period.")
else:
    stock_data = indexer(stock_data,'stock_test')
    nifty_data = indexer(nifty_data, 'nifty_test')
    stock_data = stock_data[['Close']].rename(columns={'Close': 'Stock_Close'})
    nifty_data = nifty_data[['Close']].rename(columns={'Close': 'Nifty_Close'})
    merged_data = pd.merge(stock_data, nifty_data, left_index=True, right_index=True)
    
    # calculate technical indicators
    merged_data = data_preprocessor(merged_data)
    features = merged_data.drop(columns=['Stock_Close']).values
    target = merged_data['Stock_Close'].values.reshape(-1, 1)
    nifty_features = merged_data.drop(columns=['Stock_Close', 'Nifty_Close']).values
    nifty_target = merged_data['Nifty_Close'].values.reshape(-1,1)
    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target)
    nifty_features_scaled = nifty_feature_scaler.fit_transform(nifty_features)
    nifty_target_scaled = nifty_target_scaler.fit_transform(nifty_target)
    model = load_model(model_path)
    nifty_model = load_model('nifty_model.keras')
    
    initial_data = merged_data.copy()
    initial_nifty_data = merged_data.copy()
    predictions = []
    num_iterations = 30
    initial_data.dropna(inplace=True)
    initial_nifty_data.dropna(inplace=True)
    feature_columns = list(merged_data.drop(columns=['Stock_Close']).columns)
    feature_nifty_columns = list(merged_data.drop(columns=['Stock_Close','Nifty_Close']).columns)
    features = [feature_columns[i] for i in range(len(feature_columns))]
    nifty_features = [feature_nifty_columns[i] for i in range(len(feature_nifty_columns))]
    print(features ,nifty_features, sep='\n')
    initial_data['Stock_Close'] = pd.to_numeric(initial_data['Stock_Close'].squeeze(), errors='coerce')
    initial_nifty_data['Nifty_Close'] = pd.to_numeric(initial_nifty_data['Nifty_Close'].squeeze(), errors='coerce')
    for _ in range(num_iterations):
        # Predict Nifty Close
        if len(initial_nifty_data) >= 45:
            latest_nifty_features = initial_nifty_data[nifty_features].tail(45).values
            scaled_latest_nifty_features = nifty_feature_scaler.transform(latest_nifty_features)
            nifty_prediction = nifty_model.predict(scaled_latest_nifty_features.reshape(1, 45, scaled_latest_nifty_features.shape[1]))
            predicted_nifty_close = nifty_target_scaler.inverse_transform(nifty_prediction)[0][0]
        else:
            st.warning("Insufficient data for Nifty prediction. Skipping this iteration.")
            continue  # Skip to the next iteration if there's insufficient data

        # Prepare new Nifty row
        new_nifty_row = {'Nifty_Close': float(predicted_nifty_close)}
        new_nifty_row['SMA_20'] = initial_nifty_data['Nifty_Close'].iloc[-20:].mean()
        alpha = 2 / (20 + 1)
        new_nifty_row['EMA_20'] = initial_nifty_data['Nifty_Close'].iloc[-1] * alpha + initial_nifty_data['EMA_20'].iloc[-1] * (1 - alpha)
        new_nifty_row['Volatility'] = initial_nifty_data['Nifty_Close'].iloc[-20:].std()
        new_nifty_row['Bollinger_Upper'] = new_nifty_row['SMA_20'] + (2 * new_nifty_row['Volatility'])
        new_nifty_row['Bollinger_Lower'] = new_nifty_row['SMA_20'] - (2 * new_nifty_row['Volatility'])
        recent_prices = pd.Series([predicted_nifty_close] + list(initial_nifty_data['Nifty_Close'].iloc[-13:].astype(float)))
        new_nifty_row['RSI'] = ta.momentum.RSIIndicator(recent_prices, window=14).rsi().iloc[-1]
        new_nifty_row['Volatility_Adjusted_Movement'] = ((predicted_nifty_close - initial_nifty_data['Nifty_Close'].iloc[-1]) / initial_nifty_data['Nifty_Close'].iloc[-1]) / new_nifty_row['Volatility']
        new_nifty_row_df = pd.DataFrame([new_nifty_row])
        new_nifty_row_df = new_nifty_row_df[['Nifty_Close'] + nifty_features]
        initial_nifty_data = pd.concat([initial_nifty_data, new_nifty_row_df], ignore_index=True)
        initial_nifty_data = initial_nifty_data.iloc[-(45 + 1):]

        # Predict Stock Close
        if len(initial_data) >= 45:
            latest_features = initial_data[features].tail(45).values
            scaled_latest_features = feature_scaler.transform(latest_features)
            stock_prediction = model.predict(scaled_latest_features.reshape(1, 45, scaled_latest_features.shape[1]))
            predicted_close = target_scaler.inverse_transform(stock_prediction)[0][0]
            predictions.append(predicted_close)
        else:
            st.warning("Insufficient data for Stock prediction. Skipping this iteration.")
            continue  # Skip to the next iteration if there's insufficient data

        # Prepare new Stock row
        new_row = {'Stock_Close': float(predicted_close)}
        new_row['SMA_20'] = initial_data['Stock_Close'].iloc[-20:].mean()
        alpha = 2 / (20 + 1)
        new_row['EMA_20'] = initial_data['Stock_Close'].iloc[-1] * alpha + initial_data['EMA_20'].iloc[-1] * (1 - alpha)
        new_row['Volatility'] = initial_data['Stock_Close'].iloc[-20:].std()
        new_row['Bollinger_Upper'] = new_row['SMA_20'] + (2 * new_row['Volatility'])
        new_row['Bollinger_Lower'] = new_row['SMA_20'] - (2 * new_row['Volatility'])
        recent_prices = pd.Series([predicted_close] + list(initial_data['Stock_Close'].iloc[-13:].astype(float)))
        new_row['RSI'] = ta.momentum.RSIIndicator(recent_prices, window=14).rsi().iloc[-1]
        new_row['Volatility_Adjusted_Movement'] = ((predicted_close - initial_data['Stock_Close'].iloc[-1]) / initial_data['Stock_Close'].iloc[-1]) / new_row['Volatility']
        new_row['Nifty_Close'] = predicted_nifty_close
        new_row_df = pd.DataFrame([new_row])
        new_row_df = new_row_df[['Stock_Close'] + features]
        initial_data = pd.concat([initial_data, new_row_df], ignore_index=True)
        initial_data = initial_data.iloc[-(45 + 1):]
    plot_data = yf.download(ticker, period='3mo')
    # Candlestick and MACD Plot
    st.subheader("Recent Trends in the stock price")

    # Convert data to NumPy arrays
    open_array = np.array(plot_data['Open'].squeeze())
    high_array = np.array(plot_data['High'].squeeze())
    low_array = np.array(plot_data['Low'].squeeze())
    close_array = np.array(plot_data['Close'].squeeze())
    volume_array = np.array(plot_data['Volume'].squeeze())

    # Create a new DataFrame for mplfinance
    candlestick_df = pd.DataFrame({
        'Open': open_array,
        'High': high_array,
        'Low': low_array,
        'Close': close_array,
        'Volume': volume_array
    }, index=plot_data.index)

    # Drop NaN values
    candlestick_df.dropna(inplace=True)

    # Calculate MACD
    def MACD(df, window_slow, window_fast, window_signal):
        macd = pd.DataFrame()
        macd['ema_slow'] = df['Close'].ewm(span=window_slow).mean()
        macd['ema_fast'] = df['Close'].ewm(span=window_fast).mean()
        macd['macd'] = macd['ema_fast'] - macd['ema_slow']
        macd['signal'] = macd['macd'].ewm(span=window_signal).mean()
        macd['diff'] = macd['macd'] - macd['signal']
        macd['bar_positive'] = macd['diff'].map(lambda x: x if x > 0 else 0)
        macd['bar_negative'] = macd['diff'].map(lambda x: x if x < 0 else 0)
        return macd

    macd = MACD(candlestick_df, 26, 12, 9)

    # Define additional plots for MACD
    plots = [
        mpf.make_addplot(macd['macd'], color='#606060', panel=1, ylabel='MACD (12,26,9)'),
        mpf.make_addplot(macd['signal'], color='#1f77b4', panel=1),
        mpf.make_addplot(macd['bar_positive'], type='bar', color='#4dc790', panel=1),
        mpf.make_addplot(macd['bar_negative'], type='bar', color='#fd6b6c', panel=1),
    ]

    # Generate the candlestick chart with MACD
    fig, axlist = mpf.plot(
        candlestick_df,
        type='candle',
        style='yahoo',
        mav=(5, 20),
        volume=False,
        addplot=plots,
        panel_ratios=(3, 1),
        figscale=1.5,
        returnfig=True
    )

    # Render the plot in Streamlit
    st.pyplot(fig)
    st.write("Making predictions for next 30 days...")
    # Predictions Plot
    st.subheader("Predicted Stock Prices for next 30 days")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(predictions, label="Predicted Prices", color='red', linestyle='dashed')

    ax.set_title(f"Stock Price Prediction for Next 30 Days")
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)




descriptive_data = merged_data.iloc[-45:].copy()  
key_features = ['Stock_Close', 'Volatility', 'RSI']


stats_table = descriptive_data[key_features].describe()
stats_table.columns = key_features  
st.subheader("Descriptive Data Table (Key Features)")
st.write(stats_table)

