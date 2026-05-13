import yfinance as yf
import pandas as pd
import ta
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

@st.cache_data
def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="2y", interval="1d")
    
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    
    df = df.dropna()
    return df


def train_model(df):
    df = df.copy()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    
    X = df[['Return', 'MA5', 'MA10', 'RSI']]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    
    return model, accuracy

def backtest_model(model, df):

    correct = 0
    total = 0

    for i in range(50, len(df)-1):
        sample = df[['Return','MA5','MA10','RSI']].iloc[i:i+1]
        pred = model.predict(sample)[0]

        actual = 1 if df['Close'].iloc[i+1] > df['Close'].iloc[i] else 0

        if pred == actual:
            correct += 1

        total += 1

    return correct / total