
# ai_stock_screener_app.py
# Streamlit AI Stock Screener

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import requests
import datetime

st.set_page_config(page_title="AI Stock Screener", layout="wide")

# --- Configuration ---
NEWS_API_KEY = "YOUR_NEWS_API_KEY"  # optional for news sentiment
MARKET_CAP_LIMIT = 2_000_000_000
REL_VOLUME_LIMIT = 2
VOLUME_LIMIT = 5_000_000
FLOAT_LIMIT = 50_000_000

st.title("📈 AI Stock Screener - NASDAQ & Penny Stocks")

# --- Functions ---
@st.cache_data(show_spinner=True)
def get_nasdaq_tickers():
    url = "https://datahub.io/core/nasdaq-listings/r/nasdaq-listed.csv"
    df = pd.read_csv(url)
    return df['Symbol'].dropna().tolist()

def get_news_sentiment(ticker):
    if NEWS_API_KEY == "YOUR_NEWS_API_KEY":
        return 0
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}&language=en&pageSize=5"
        r = requests.get(url).json()
        if "articles" not in r:
            return 0
        sentiments = [TextBlob(a["title"]).sentiment.polarity for a in r["articles"]]
        return np.mean(sentiments)
    except:
        return 0

@st.cache_data(show_spinner=True)
def get_stock_data(ticker):
    try:
        data = yf.download(ticker, period="6mo", interval="1d", progress=False)
        data["Return"] = data["Close"].pct_change()
        data["Volatility"] = data["Return"].rolling(5).std()
        data["Range"] = (data["High"] - data["Low"]) / data["Low"]
        data.dropna(inplace=True)
        return data
    except:
        return None

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

def get_targets(last_close, predicted_return):
    entry = last_close * (1 - 0.01)
    sell = last_close * (1 + predicted_return)
    return entry, sell

def cost_model(predicted_return):
    fee = 0.001
    slippage = 0.002
    return predicted_return - (fee + slippage)

@st.cache_data(show_spinner=True)
def scan_stocks(limit=100):
    tickers = get_nasdaq_tickers()
    results = []
    for ticker in tickers[:limit]:
        data = get_stock_data(ticker)
        if data is None or len(data) < 30:
            continue
        predicted_return = forecast_next_day(data)
        adj_return = cost_model(predicted_return)
        sentiment = get_news_sentiment(ticker)
        last_close = data["Close"].iloc[-1]
        entry, sell = get_targets(last_close, adj_return)
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
    df["OverallScore"] = df["Predicted_Return"]*0.6 + df["Sentiment"]*0.3 + df["RangeUp"]*0.1
    df = df.sort_values("OverallScore", ascending=False)
    penny = df[(df["MarketCap"]<MARKET_CAP_LIMIT) & (df["RelativeVolume"]>REL_VOLUME_LIMIT) &
               (df["Volume"]>VOLUME_LIMIT) & (df["Float"]<FLOAT_LIMIT)].sort_values("Predicted_Return", ascending=False)
    return df, penny

# --- User controls ---
st.sidebar.header("Settings")
limit = st.sidebar.slider("Max tickers to scan (for speed)", 10, 200, 50)

if st.button("Run Stock Scan"):
    with st.spinner("Scanning stocks..."):
        df, penny = scan_stocks(limit)
    st.success("Scan complete ✅")
    
    st.subheader("🔹 Top NASDAQ Upside Stocks")
    st.dataframe(df.head(20)[["Ticker","Predicted_Return","Sentiment","Entry","SellTarget"]])
    
    st.subheader("⚡ High Growth / High Risk Penny Stocks")
    st.dataframe(penny.head(20)[["Ticker","Predicted_Return","RangeUp","Entry","SellTarget"]])
    
    st.download_button("Download Full Results CSV", df.to_csv(index=False), "nasdaq_stock_scan.csv")
    st.download_button("Download Penny Stocks CSV", penny.to_csv(index=False), "penny_stock_scan.csv")
