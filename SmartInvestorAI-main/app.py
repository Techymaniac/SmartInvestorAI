
import streamlit as st
from model import get_stock_data, train_model, backtest_model
from sentiment import get_sentiment
from news import get_news
from lstm_model import train_lstm, predict_next
import plotly.graph_objects as go
from report import generate_report
from decision import make_decision, get_risk
from io import BytesIO

# ---------------- CACHE ----------------
@st.cache_resource
def get_trained_model(df):
    return train_model(df)

@st.cache_resource
def get_lstm_prediction(df):
    model, scaler = train_lstm(df)
    return predict_next(model, scaler, df)

@st.cache_data
def get_news_cached(symbol):
    return get_news(symbol)

# ---------------- UI ----------------
st.set_page_config(layout="wide")
st.title("📈 Stock Market Prediction using AI & Sentiment Analysis")

# ---------------- SESSION ----------------
if "result" not in st.session_state:
    st.session_state.result = None

symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS or AAPL)")

# ---------------- ANALYZE ----------------
if st.button("Analyze"):

    df = get_stock_data(symbol)

    if df.empty:
        st.error("Invalid stock symbol")
        st.stop()

    currency = "₹" if ".NS" in symbol else "$"
    market = "India (NSE)" if ".NS" in symbol else "USA"

    model, acc = get_trained_model(df)

    latest = df[['Return', 'MA5', 'MA10', 'RSI']].iloc[-1:]
    prediction = model.predict(latest)[0]
    proba = model.predict_proba(latest)[0]
    confidence = max(proba) * 100

    clean_symbol = symbol.replace(".NS", "")
    news = get_news_cached(clean_symbol)
    sentiment_score = get_sentiment(news)

    decision = make_decision(prediction, sentiment_score, confidence)
    risk = get_risk(confidence)

    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]

    future_price = get_lstm_prediction(df)

    st.session_state.result = {
        "df": df,
        "symbol": symbol,
        "market": market,
        "currency": currency,
        "prediction": prediction,
        "confidence": confidence,
        "decision": decision,
        "risk": risk,
        "current_price": current_price,
        "prev_price": prev_price,
        "future_price": future_price,
        "news": news,
        "acc": acc,
        "model": model,
        "sentiment_score": sentiment_score
    }

# ---------------- DISPLAY ----------------
if st.session_state.result:

    r = st.session_state.result
    df = r["df"]

    current_price = r["current_price"]
    prev_price = r["prev_price"]

    change = current_price - prev_price
    change_percent = (change / prev_price) * 100
    arrow = "↑" if change > 0 else "↓"

    st.markdown(f"## {r['symbol']} ({r['market']})")

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📰 News", "📈 Chart"])

    # -------- OVERVIEW --------
    with tab1:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Price", f"{r['currency']}{round(current_price,2)}",
                    f"{arrow} {round(change,2)} ({round(change_percent,2)}%)")

        col2.metric("Prediction", "UP 📈" if r["prediction"] == 1 else "DOWN 📉")
        col3.metric("Confidence", f"{round(r['confidence'],2)}%")
        col4.metric("Risk", r["risk"])

        strength = "Strong" if r["confidence"] > 75 else "Moderate" if r["confidence"] > 60 else "Weak"
        st.success(f"AI Recommendation: {r['decision']} ({strength})")

        st.markdown("### 🔮 Future Price Prediction")

        future_price = r["future_price"]

        if future_price:
            change_pred = future_price - current_price
            change_percent_pred = (change_pred / current_price) * 100
            arrow_pred = "📈" if change_pred > 0 else "📉"

            st.metric(
                "🔮 Predicted Price",
                f"{r['currency']}{future_price:.2f}",
                f"{arrow_pred} {round(change_pred,2)} ({round(change_percent_pred,2)}%)"
            )
        else:
            st.warning("Not enough data for prediction")

    # -------- NEWS --------
    with tab2:
        if r["news"]:
            for n in r["news"]:
                st.write("-", n)
        else:
            st.write("No news available")

    # -------- CHART --------
    with tab3:
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        ))

        st.plotly_chart(fig, width="stretch")

    # -------- ACCURACY --------
    st.write("Model Accuracy:", round(r["acc"] * 100, 2), "%")

    bt_acc = backtest_model(r["model"], df)
    st.metric("Backtest Accuracy", f"{bt_acc*100:.2f}%")

    # -------- REPORT --------
    report_data = {
        "Stock": r["symbol"],
        "Market": r["market"],
        "Current Price": f"{r['currency']}{round(current_price,2)}",
        "Prediction": "UP" if r["prediction"] == 1 else "DOWN",
        "Confidence": f"{round(r['confidence'],2)}%",
        "Risk": r["risk"],
        "Recommendation": r["decision"],
        "Sentiment Score": round(r["sentiment_score"],2),
        "Predicted Price": f"{r['currency']}{round(r['future_price'],2)}"
    }

    buffer = BytesIO()
    generate_report(buffer, report_data)

    st.download_button(
        "📥 Download Report",
        buffer,
        file_name="Stock_Report.pdf"
    )

