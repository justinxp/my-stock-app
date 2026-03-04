import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd
from datetime import date
import plotly.graph_objects as go

# 1. 網頁配置
st.set_page_config(page_title="AI 股市旗艦分析系統", layout="wide")

# 2. 側邊欄設定
st.sidebar.header("🔍 進階投資參數")
ticker_input = st.sidebar.text_input("輸入台股代碼 (多個請用逗號隔開):", "0050")
tickers = [t.strip() for t in ticker_input.split(",")]

train_years = st.sidebar.slider("訓練數據年數:", 1, 10, 5)
predict_days = st.sidebar.slider("預測未來天數:", 30, 365, 90)


@st.cache_data(ttl=300)  # 籌碼與技術面建議 5 分鐘更新一次
def load_data(ticker, years):
    full_ticker = f"{ticker}.TW"
    start = date(date.today().year - years, 1, 1).strftime("%Y-%m-%d")
    df = yf.download(full_ticker, start=start, auto_adjust=False, actions=True)
    if df.empty: return None
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# 3. 主畫面
st.title("🛡️ AI 旗艦級股市分析系統 (含多重技術曲線)")

all_forecasts = []

for ticker in tickers:
    st.divider()
    st.header(f"📈 {ticker} 深度技術分析")

    data = load_data(ticker, train_years)

    if data is not None:
        df = data.dropna(subset=['Close']).copy()

        # --- A. 強化版儀表板 ---
        last_price = float(df['Close'].iloc[-1])
        high_price = float(df['High'].iloc[-1])
        low_price = float(df['Low'].iloc[-1])
        vol = float(df['Volume'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        change = last_price - prev_price

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("當前收盤", f"{last_price:.2f}", f"{change:.2f}")
        c2.metric("今日最高", f"{high_price:.2f}")
        c3.metric("今日最低", f"{low_price:.2f}")
        c4.metric("成交量", f"{int(vol):,}")

        # 計算 RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        current_rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]
        c5.metric("RSI 指標", f"{current_rsi:.2f}")

        # --- B. 技術指標計算 (布林通道, MACD) ---
        # 布林通道 (20日)
        df['MA20'] = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        df['Upper'] = df['MA20'] + (std * 2)
        df['Lower'] = df['MA20'] - (std * 2)

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']

        # --- C. 繪製互動式技術圖表 (Plotly) ---
        tab1, tab2, tab3 = st.tabs(["布林趨勢圖", "MACD 動能圖", "AI 深度預測"])

        with tab1:
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Upper'], name='壓力線 (上軌)',
                                        line=dict(dash='dash', color='rgba(200,200,200,0.5)')))
            fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Lower'], name='支撐線 (下軌)',
                                        line=dict(dash='dash', color='rgba(200,200,200,0.5)'), fill='tonexty'))
            fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='收盤價', line=dict(color='blue')))
            fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name='月線 (MA20)', line=dict(color='orange')))
            fig_bb.update_layout(title=f"{ticker} 布林通道圖", hovermode="x unified", height=500)
            st.plotly_chart(fig_bb, width='stretch')
            st.caption("💡 註：股價觸碰下軌通常有反彈機會，觸碰上軌則需小心回檔。")

        with tab2:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(x=df['Date'], y=df['Hist'], name='MACD 柱狀體'))
            fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='DIF 快線'))
            fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], name='MACD 慢線'))
            fig_macd.update_layout(title=f"{ticker} MACD 指標", height=400)
            st.plotly_chart(fig_macd, width='stretch')

        with tab3:
            # AI 預測邏輯
            df_train = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
            df_train['ds'] = pd.to_datetime(df_train['ds']).dt.tz_localize(None)
            m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.08)
            m.fit(df_train)
            future = m.make_future_dataframe(periods=predict_days)
            forecast = m.predict(future)

            fig_ai = plot_plotly(m, forecast)
            st.plotly_chart(fig_ai, width='stretch')

            # 準備對比數據
            base = forecast.iloc[-predict_days - 1]['yhat']
            forecast['return_pct'] = (forecast['yhat'] - base) / base * 100
            temp = forecast[['ds', 'return_pct']].tail(predict_days).copy()
            temp['ticker'] = ticker
            all_forecasts.append(temp)

# --- D. 多股對比總結 ---
if len(all_forecasts) > 1:
    st.divider()
    st.header("🏁 全標的 AI 報酬率對比")
    comp_df = pd.concat(all_forecasts)
    fig_comp = go.Figure()
    for t in tickers:
        t_data = comp_df[comp_df['ticker'] == t]
        fig_comp.add_trace(go.Scatter(x=t_data['ds'], y=t_data['return_pct'], mode='lines', name=t))
    fig_comp.update_layout(yaxis_title="預期報酬率 (%)", height=500)
    st.plotly_chart(fig_comp, width='stretch')

st.sidebar.markdown("---")
st.sidebar.info(
    "📢 外資數據提示：\n目前的收盤價與成交量已包含法人行為。若需精確外資買賣超張數，請參考證交所每日 15:00 公布之數據。")