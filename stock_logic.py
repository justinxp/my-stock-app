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
st.sidebar.header("🔍 投資標的與參數")
ticker_input = st.sidebar.text_input("輸入台股代碼 (多個請用逗號隔開):", "0050")
tickers = [t.strip() for t in ticker_input.split(",")]

train_years = st.sidebar.slider("訓練數據年數:", 1, 10, 5)
predict_days = st.sidebar.slider("預測未來天數:", 30, 365, 90)

@st.cache_data(ttl=300)
def load_data(ticker, years):
    start = date(date.today().year - years, 1, 1).strftime("%Y-%m-%d")
    
    # 策略：自動嘗試上市 (.TW) 與 上櫃 (.TWO)
    df = pd.DataFrame()
    for suffix in [".TW", ".TWO"]:
        full_ticker = f"{ticker}{suffix}"
        df = yf.download(full_ticker, start=start, auto_adjust=False)
        if not df.empty:
            break
            
    if df.empty: return None
    
    df.reset_index(inplace=True)
    # 處理新版 yfinance 多層索引問題
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# 3. 主畫面標題
st.title("🛡️ AI 旗艦級股市分析系統")
st.caption("整合 RSI、布林通道、MACD 與 Meta Prophet AI 預測")

all_forecasts = []

# 遍歷每一支股票進行深度分析
for ticker in tickers:
    st.divider()
    st.header(f"📊 {ticker} 深度技術分析")
    
    data = load_data(ticker, train_years)
    
    if data is not None:
        df = data.dropna(subset=['Close']).copy()
        
        # --- A. 專業儀表板 ---
        last_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        high_price = float(df['High'].iloc[-1])
        low_price = float(df['Low'].iloc[-1])
        change = last_price - prev_price
        change_pct = (change / prev_price) * 100
        
        # 計算 RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        current_rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("當前收盤", f"{last_price:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
        c2.metric("今日最高", f"{high_price:.2f}")
        c3.metric("今日最低", f"{low_price:.2f}")
        c4.metric("今日振幅", f"{((high_price-low_price)/low_price*100):.2f}%")
        c5.metric("RSI 指標", f"{current_rsi:.2f}")

        # --- B. 技術指標計算 ---
        # 布林通道
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

        # --- C. 互動式分頁圖表 ---
        tab1, tab2, tab3 = st.tabs(["📈 布林趨勢與均線", "📊 MACD 動能", "🔮 AI 深度預測"])
        
        with tab1:
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Upper'], name='上軌 (壓力)', line=dict(dash='dash', color='rgba(200,200,200,0.5)')))
            fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Lower'], name='下軌 (支撐)', line=dict(dash='dash', color='rgba(200,200,200,0.5)'), fill='tonexty'))
            fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='收盤價', line=dict(color='blue')))
            fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name='月線 (MA20)', line=dict(color='orange')))
            fig_bb.update_layout(hovermode="x unified", height=500)
            st.plotly_chart(fig_bb, width='stretch')
            
            if current_rsi < 35:
                st.success(f"💡 提醒：{ticker} RSI 偏低且接近布林下軌，短線具備支撐潛力。")
            elif current_rsi > 65:
                st.warning(f"💡 提醒：{ticker} RSI 偏高且接近布林上軌，請留意過熱回檔。")

        with tab2:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(x=df['Date'], y=df['Hist'], name='柱狀體'))
            fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='快線 (DIF)'))
            fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], name='慢線 (Signal)'))
            fig_macd.update_layout(height=400)
            st.plotly_chart(fig_macd, width='stretch')

        with tab3:
            with st.spinner(f"AI 正在運算 {ticker} 模型..."):
                df_train = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
                df_train['ds'] = pd.to_datetime(df_train['ds']).dt.tz_localize(None)
                m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.08)
                m.fit(df_train)
                future = m.make_future_dataframe(periods=predict_days)
                forecast = m.predict(future)
                
                fig_ai = plot_plotly(m, forecast)
                st.plotly_chart(fig_ai, width='stretch')
                
                # 為對比圖準備數據
                base = forecast.iloc[-predict_days-1]['yhat']
                forecast['return_pct'] = (forecast['yhat'] - base) / base * 100
                temp = forecast[['ds', 'return_pct']].tail(predict_days).copy()
                temp['ticker'] = ticker
                all_forecasts.append(temp)
    else:
        st.error(f"❌ 找不到代碼 {ticker}。請確認輸入正確（例如上市 2330, 上櫃 8069）。")

# --- D. 多股對比報酬率 ---
if len(all_forecasts) > 1:
    st.divider()
    st.header("🏁 多股 AI 預期報酬率對比 (%)")
    comp_df = pd.concat(all_forecasts)
    fig_comp = go.Figure()
    for t in tickers:
        t_data = comp_df[comp_df['ticker'] == t]
        fig_comp.add_trace(go.Scatter(x=t_data['ds'], y=t_data['return_pct'], mode='lines', name=t))
    fig_comp.update_layout(yaxis_title="預期報酬率 (%)", hovermode="x unified", height=500)
    st.plotly_chart(fig_comp, width='stretch')

st.divider()
st.caption(f"數據更新時間: {date.today()} | 來源: Yahoo Finance | 本工具不構成投資建議。")
