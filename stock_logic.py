import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd
from datetime import date
import plotly.graph_objects as go

# --- 1. 網頁配置與美化 ---
st.set_page_config(
    page_title="AI 股市旗艦分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 強化視覺專業感
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. 側邊欄：參數設定 ---
with st.sidebar:
    st.header("⚙️ 參數設定")

    # 核心代碼輸入（支援多股）
    ticker_input = st.text_input("🚀 輸入股票代碼 (多個請用逗號隔開):", "0050")
    tickers = [t.strip() for t in ticker_input.split(",")]

    st.markdown("---")

    st.subheader("📊 模型設定")
    train_years = st.slider("訓練數據年數:", 1, 10, 5)
    predict_days = st.slider("預測未來天數:", 30, 365, 90)

    st.markdown("---")
    st.caption(f"數據更新日期: {date.today()}\n版本：v3.5 專業版")


# --- 3. 數據抓取邏輯 (含上市/櫃自動偵測) ---
@st.cache_data(ttl=300)
def load_data(ticker, years):
    start = date(date.today().year - years, 1, 1).strftime("%Y-%m-%d")
    df = pd.DataFrame()
    # 自動偵測：上市(.TW) -> 上櫃(.TWO) -> 美股(無後綴)
    for suffix in [".TW", ".TWO", ""]:
        full_ticker = f"{ticker}{suffix}" if suffix else ticker
        try:
            # auto_adjust=False 確保價格對齊 Yahoo 原始收盤價 (如 0050 = 75.60)
            df = yf.download(full_ticker, start=start, auto_adjust=False, progress=False)
            if not df.empty: break
        except:
            continue

    if df.empty: return None

    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# --- 4. 主畫面標題 ---
st.title("🛡️ AI 旗艦級股市分析系統")
st.markdown("🚀 **整合 Meta Prophet AI 預測模型與多重技術指標 (RSI, Bollinger, MACD)**")

all_forecasts = []

# --- 5. 股票循環分析區 ---
for ticker in tickers:
    data = load_data(ticker, train_years)

    if data is not None:
        df = data.dropna(subset=['Close']).copy()

        # --- A. 專業數據計算 ---
        # 價格資訊
        last_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        high_price = float(df['High'].iloc[-1])
        low_price = float(df['Low'].iloc[-1])
        change = last_price - prev_price
        change_pct = (change / prev_price) * 100

        # RSI (14日)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        current_rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]

        # 布林通道 (20日)
        df['MA20'] = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        df['Upper'] = df['MA20'] + (std * 2)
        df['Lower'] = df['MA20'] - (std * 2)

        # MACD (12, 26, 9)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']

        # --- B. UI 佈局：指標快報 ---
        st.subheader(f"📊 {ticker} 即時盤態")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("收盤價", f"{last_price:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
        m2.metric("今日最高", f"{high_price:.2f}")
        m3.metric("今日最低", f"{low_price:.2f}")
        m4.metric("今日振幅", f"{((high_price - low_price) / low_price * 100):.2f}%")
        m5.metric("RSI 指標", f"{current_rsi:.2f}")

        # --- C. UI 佈局：互動分頁圖表 ---
        t1, t2, t3 = st.tabs(["📈 趨勢與布林", "📊 MACD 動能", "🔮 AI 深度預測"])

        with t1:
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Upper'], name='上軌 (壓力)',
                                        line=dict(dash='dash', color='rgba(200,200,200,0.3)')))
            fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Lower'], name='下軌 (支撐)',
                                        line=dict(dash='dash', color='rgba(200,200,200,0.3)'), fill='tonexty'))
            fig_bb.add_trace(
                go.Scatter(x=df['Date'], y=df['Close'], name='收盤價', line=dict(color='#1f77b4', width=2)))
            fig_bb.add_trace(
                go.Scatter(x=df['Date'], y=df['MA20'], name='月線 (MA20)', line=dict(color='#ff7f0e', width=1.5)))
            fig_bb.update_layout(height=500, hovermode="x unified", margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig_bb, use_container_width=True)

        with t2:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(x=df['Date'], y=df['Hist'], name='MACD 柱狀體', marker_color='gray'))
            fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='快線 (DIF)', line=dict(color='blue')))
            fig_macd.add_trace(
                go.Scatter(x=df['Date'], y=df['Signal'], name='慢線 (Signal)', line=dict(color='orange')))
            fig_macd.update_layout(height=400, margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig_macd, use_container_width=True)

        with t3:
            with st.spinner(f"正在執行 Meta Prophet AI 對 {ticker} 的深度運算..."):
                df_train = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
                df_train['ds'] = pd.to_datetime(df_train['ds']).dt.tz_localize(None)
                m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.08)
                m.fit(df_train)
                future = m.make_future_dataframe(periods=predict_days)
                forecast = m.predict(future)

                fig_ai = plot_plotly(m, forecast)
                fig_ai.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=30))
                st.plotly_chart(fig_ai, use_container_width=True)

                # 報酬率數據存儲 (用於多股對比)
                base = forecast.iloc[-predict_days - 1]['yhat']
                forecast['return_pct'] = (forecast['yhat'] - base) / base * 100
                temp_f = forecast[['ds', 'return_pct']].tail(predict_days).copy()
                temp_f['ticker'] = ticker
                all_forecasts.append(temp_f)
    else:
        st.error(f"❌ 無法獲取代碼 `{ticker}` 的數據。請檢查代碼是否正確。")

# --- 6. 多股對比總結 ---
if len(all_forecasts) > 1:
    st.markdown("---")
    st.header("🏁 多股 AI 預期報酬率對比")
    st.info("💡 說明：此圖比較各標的從今日起往後預測走勢。")

    comp_df = pd.concat(all_forecasts)
    fig_comp = go.Figure()
    for t in tickers:
        t_data = comp_df[comp_df['ticker'] == t]
        fig_comp.add_trace(go.Scatter(x=t_data['ds'], y=t_data['return_pct'], mode='lines', name=f"{t} 預期報酬"))

    fig_comp.update_layout(
        xaxis_title="預測日期",
        yaxis_title="報酬率 (%)",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# --- 7. 免責聲明 ---
st.markdown("---")
st.caption("⚠️ 免責聲明：本系統僅供 AI 技術研究參考，不構成投資建議。投資人應獨立判斷並自負盈虧。")
