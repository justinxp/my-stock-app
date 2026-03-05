import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd
from datetime import date
import plotly.graph_objects as go
import time
import random

# --- 1. 網頁配置 ---
st.set_page_config(page_title="AI 股市旗艦分析系統", page_icon="📈", layout="wide")

# --- 2. 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 系統核心設定")
    ticker_input = st.text_input("🚀 輸入股票代碼 (多個請用逗號隔開):", value="0050", placeholder="例如: 2330, 0050")
    tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
    st.markdown("---")
    train_years = st.slider("訓練數據年數:", 1, 10, 5)
    predict_days = st.slider("預測未來天數:", 30, 365, 90)
    st.caption("版本：v9.4 雲端穩定版")

# --- 3. 核心數據函數 ---
@st.cache_data(ttl=300)
def fetch_stock_data(ticker, years):
    full_ticker = f"{ticker}.TW" if ticker.isdigit() else ticker
    try:
        # 下載歷史數據
        df = yf.download(full_ticker, start=f"{date.today().year - years}-01-01", auto_adjust=False, progress=False)
        if df.empty or len(df) < 20: return None
        
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # 指標計算
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Upper'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
        df['Lower'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)
        
        # 抓取即時參考價 (修正 0050 漲跌)
        tk = yf.Ticker(full_ticker)
        info = tk.info
        actual_last = info.get('regularMarketPrice') if info.get('regularMarketPrice') else float(df['Close'].iloc[-1])
        actual_prev = info.get('regularMarketPreviousClose') if info.get('regularMarketPreviousClose') else float(df['Close'].iloc[-2])
        
        return {
            "df": df.dropna(subset=['Upper']).reset_index(drop=True),
            "last_p": actual_last,
            "prev_p": actual_prev,
            "name": info.get('longName', full_ticker)
        }
    except:
        return None

# --- 4. 主畫面邏輯 ---
st.title("🛡️ AI 旗艦級股市分析系統")

if not tickers:
    st.info("💡 請在左側輸入代碼後按 Enter 開始。")
else:
    all_forecasts = []
    for ticker in tickers:
        data = fetch_stock_data(ticker, train_years)
        if data:
            df, last_p, prev_p = data["df"], data["last_p"], data["prev_p"]
            change = last_p - prev_p
            pct = (change / prev_p) * 100

            st.subheader(f"📊 {data['name']} ({ticker})")
            
            c1, c2 = st.columns(2)
            c1.metric("當前價格", f"{last_p:.2f}", f"{change:+.2f} ({pct:+.2f}%)", delta_color="inverse")
            c2.metric("20日均線 (MA20)", f"{df['MA20'].iloc[-1]:.2f}", f"標準差: {df['Close'].rolling(20).std().iloc[-1]:.2f}", delta_color="off")

            # AI 建議語句 (保留原功能)
            with st.spinner(f"AI 正在運算 {ticker}..."):
                m = Prophet(daily_seasonality=True).fit(df[['Date', 'Close']].rename(columns={"Date":"ds", "Close":"y"}))
                fc = m.predict(m.make_future_dataframe(periods=predict_days))
                ai_val = ((fc['yhat'].iloc[-1] - last_p) / last_p) * 100

            color = "red" if ai_val >= 0 else "green"
            st.markdown(f"🧭 AI 進出場建議：**{'建議佈局' if ai_val > 5 else '建議避險' if ai_val < -5 else '持股觀望'}**")
            st.markdown(f"📈 預估 {predict_days} 天變動：<span style='color:{color}; font-weight:bold;'>{ai_val:+.2f}%</span>", unsafe_allow_html=True)

            t1, t2, t3 = st.tabs(["📈 趨勢布林", "📊 MACD 動能", "🔮 AI 預測圖"])
            with t1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Upper'], name='上軌', line=dict(dash='dash', color='gray')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Lower'], name='下軌', line=dict(dash='dash', color='gray'), fill='tonexty'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='收盤', line=dict(color='red' if change >= 0 else 'green')))
                st.plotly_chart(fig, use_container_width=True)
            with t2:
                # 簡單 MACD 顯示
                macd = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
                st.area_chart(macd)
            with t3:
                st.plotly_chart(plot_plotly(m, fc), use_container_width=True)

st.markdown("---")
st.caption("⚠️ 本工具僅供 AI 技術研究參考，不構成投資建議，投資人應獨立評估風險。數據來源：Yahoo Finance")
