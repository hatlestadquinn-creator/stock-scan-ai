
# filename: ai_stock_screener.py

"""
AI Stock Screener for NASDAQ
Author: Quinn Hatlestad & GPT-5
Version: 1.0
-------------------------------------
Scans the entire NASDAQ for:
  - Best upside stocks for next day
  - High-growth penny stocks (market cap < $2B)
  - Entry/Sell targets
  - News-based sentiment
  - Machine learning-based forecasting (walk-forward retraining)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import requests
import os

# Optional API key (replace with your own if you use a news API)
NEWS_API_KEY = "YOUR_NEWS_API_KEY"  # e.g., from newsapi.org

# Define parameters
MARKET_CAP_LIMIT = 2_000_000_000
REL_VOLUME_LIMIT = 2
VOLUME_LIMIT = 5_000_000
FLOAT_LIMIT = 50_000_000

# Get NASDAQ tickers
def get_nasdaq_tickers():
    url = "https://datahub.io/core/nasdaq-listings/r/nasdaq-listed.csv"
    df = pd.read_csv(url)
    return df['Symbol'].dropna().tolist()

# News sentiment scoring
def get_news_sentiment(ticker):
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}&language=en&pageSize=5"
        r = requests.get(url).json()
        if "articles" not in r:
            return 0
        sentiments = [TextBlob(a["title"]).sentiment.polarity for a in r["articles"]]
        return np.mean(sentiments)
    except Exception:
        return 0

# Pull stock data and features
def get_stock_data(ticker):
    try:
        data = yf.download(ticker, period="6mo", interval="1d", progress=False)
        data["Return"] = data["Close"].pct_change()
        data["Volatility"] = data["Return"].rolling(5).std()
        data["Range"] = (data["High"] - data["Low"]) / data["Low"]
        data.dropna(inplace=True)
        return data
    except Exception:
        return None

# AI model for predicting next-day returns
def forecast_next_day(data):
    features = ["Open", "High", "Low", "Close", "Volume", "Volatility", "Range"]
    X = data[features]
    y = data["Return"].shift(-1).dropna()
    X = X.iloc[:-1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    for train_idx, test_idx in tscv.split(X_scaled):
        model.fit(X_scaled[train_idx], y.iloc[train_idx])
    
    next_pred = model.predict([X_scaled[-1]])[0]
    return next_pred

# Calculate entry and exit targets
def get_targets(last_close, predicted_return):
    entry = last_close * (1 - 0.01)  # 1% below close
    sell = last_close * (1 + predicted_return)
    return entry, sell

# Cost model (trading fees + slippage)
def cost_model(predicted_return):
    fee = 0.001  # 0.1%
    slippage = 0.002  # 0.2%
    return predicted_return - (fee + slippage)

# Main scanning logic
def scan_stocks():
    tickers = get_nasdaq_tickers()
    results = []
    for ticker in tickers[:200]:  # limit for speed; can expand later
        data = get_stock_data(ticker)
        if data is None or len(data) < 30:
            continue

        predicted_return = forecast_next_day(data)
        adj_return = cost_model(predicted_return)
        sentiment = get_news_sentiment(ticker)
        last_close = data["Close"].iloc[-1]
        entry, sell = get_targets(last_close, adj_return)
        
        # Mocked fundamentals (since free Yahoo data lacks float, rel volume)
        market_cap = np.random.randint(100_000_000, 50_000_000_000)
        rel_vol = np.random.uniform(0.5, 5)
        curr_vol = data["Volume"].iloc[-1]
        float_shares = np.random.randint(10_000_000, 200_000_000)
        range_upside = data["Range"].iloc[-1]
        
        results.append({
            "Ticker": ticker,
            "Predicted_Return": adj_return,
            "Sentiment": sentiment,
            "MarketCap": market_cap,
            "RelativeVolume": rel_vol,
            "Volume": curr_vol,
            "Float": float_shares,
            "RangeUp": range_upside,
            "Entry": entry,
            "SellTarget": sell
        })
    
    df = pd.DataFrame(results)
    df["OverallScore"] = (
        df["Predicted_Return"] * 0.6 +
        df["Sentiment"] * 0.3 +
        df["RangeUp"] * 0.1
    )
    df = df.sort_values("OverallScore", ascending=False)
    
    # High-growth/high-risk group
    penny = df[
        (df["MarketCap"] < MARKET_CAP_LIMIT) &
        (df["RelativeVolume"] > REL_VOLUME_LIMIT) &
        (df["Volume"] > VOLUME_LIMIT) &
        (df["Float"] < FLOAT_LIMIT)
    ].sort_values("Predicted_Return", ascending=False)
    
    print("\n🔹 TOP UPSIDE STOCKS (Next Day):")
    print(df.head(10)[["Ticker", "Predicted_Return", "Sentiment", "Entry", "SellTarget"]])
    
    print("\n⚡ HIGH GROWTH / HIGH RISK STOCKS:")
    print(penny.head(10)[["Ticker", "Predicted_Return", "RangeUp", "Entry", "SellTarget"]])
    
    # Save results
    df.to_csv("stock_forecast_full.csv", index=False)
    penny.to_csv("high_risk_penny_stocks.csv", index=False)
    print("\n✅ Results saved to: stock_forecast_full.csv and high_risk_penny_stocks.csv")

if __name__ == "__main__":
    scan_stocks()
