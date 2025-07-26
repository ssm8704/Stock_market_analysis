# Stock Market Analysis

## Project Overview

This application:

- Loads historical stock data for Reliance and the Nifty Index using yfinance
- Preprocesses the data using technical indicators like:
  - SMA
  - EMA
  - Bollinger Bands
  - RSI
  - Volatility
- Uses pre-trained LSTM models to predict:
  - Reliance stock price for the next 30 days
  - Nifty index values as an influencing factor
- Visualizes:
  - Historical candlestick chart with MACD
  - Predicted stock prices
  - Descriptive statistics of key features

## Model Details

- **Base Model**: LSTM (Long Short Term Memory) Neural Networks to predict the stock price using technical indicators and Index price.
  - **Input**: 45 days of historical data
  - **Preprocessing**: Technical indicators using 3 day time window (14 for RSI)
  - **Output**: Predicted stock price for next 30 days

- **Nifty Model**: Calculates Nifty index price which has an influence on the stock price. Same model structure used
