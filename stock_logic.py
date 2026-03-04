import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd
from datetime import date
import plotly.graph_objects as go

# --- 1. 網頁配置 ---
st.set_page_config(
    page_title="AI 股市旗艦分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 系統核心設定")
    ticker_input = st.text_input("🚀 輸入股票代碼 (多個請用逗號隔開):", "2330, 0050")
    tickers = [t.strip() for t in ticker_input.split(",")]
    st.markdown("---")
    train_years = st.slider("訓練數據年數:", 1, 10, 5)
    predict_days = st.slider("預測未來天數:", 30, 365, 90)
    st.caption(f"數據更新日期: {date.today()}\n版本：v6.1 修復穩定版")


# --- 3. 數據抓取邏輯 (修正快取報錯問題) ---

# 僅快取財務文字資訊 (Info 是 dict，可以被 pickle)
@st.cache_data(ttl=3600)
def get_stock_financials(ticker):
    for suffix in [".TW", ".TWO", ""]:
        full_ticker = f"{ticker}{suffix}" if suffix else ticker
        try:
            s = yf.Ticker(full_ticker)
            info = s.info
            if 'symbol' in info:
                return info, full_ticker
        except:
            continue
    return None, None


# 快取歷史價格數據 (DataFrame 可以被 pickle)
@st.cache_data(ttl=300)
def load_historical_data(full_ticker, years):
    start = date(date.today().year - years, 1, 1).strftime("%Y-%m-%d")
    try:
        df = yf.download(full_ticker, start=start, auto_adjust=False, progress=False)
        if df.empty: return None
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return None


# --- 4. 主畫面 ---
st.title("🛡️ AI 旗艦級股市分析系統")

all_forecasts = []

for ticker in tickers:
    # 步驟 1: 獲取財務資訊與正確的帶後綴代碼
    info, full_ticker = get_stock_financials(ticker)

    if info and full_ticker:
        # 步驟 2: 獲取歷史數據
        df = load_historical_data(full_ticker, train_years)

        if df is not None:
            # --- A. 指標計算 ---
            last_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            change = last_price - prev_price
            change_pct = (change / prev_price) * 100

            # 財務指標
            eps = info.get('trailingEps', 'N/A')
            div_yield = info.get('trailingAnnualDividendYield', 0) * 100
            pe_ratio = info.get('trailingPE', 'N/A')

            # 技術指標 (RSI, 布林)
            df['MA20'] = df['Close'].rolling(20).mean()
            std = df['Close'].rolling(20).std()
            df['Upper'] = df['MA20'] + (std * 2)
            df['Lower'] = df['MA20'] - (std * 2)

            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]

            # --- B. AI 預測 ---
            with st.spinner(f"正在分析 {ticker}..."):
                df_p = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
                df_p['ds'] = pd.to_datetime(df_p['ds']).dt.tz_localize(None)
                m = Prophet(daily_seasonality=True).fit(df_p)
                future = m.make_future_dataframe(periods=predict_days)
                fc = m.predict(future)
                future_trend = ((fc['yhat'].iloc[-1] - last_price) / last_price) * 100

            # --- C. 🧭 綜合診斷報告 ---
            st.divider()
            st.subheader(f"📊 {ticker} 綜合診斷與財務概況")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("收盤價", f"{last_price:.2f}", f"{change_pct:.2f}%")
            c2.metric("當日高/低", f"{df['High'].iloc[-1]:.1f}", f"{df['Low'].iloc[-1]:.1f}", delta_color="off")
            c3.metric("EPS (盈餘)", f"{eps}")
            c4.metric("股息殖利率", f"{div_yield:.2f}%")
            c5.metric("本益比", f"{pe_ratio}")

            # 導航儀建議
            st.info(f"**💡 時機建議：** " +
                    ("建議分批進場" if future_trend > 5 and rsi < 40 else
                     "建議分批獲利" if future_trend < -3 and rsi > 65 else
                     "持股觀望"))

            # --- D. 圖表分頁 ---
            t1, t2, t3 = st.tabs(["📈 趨勢/布林", "📊 MACD", "🔮 AI 預測"])
            with t1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Upper'], name='上軌',
                                         line=dict(dash='dash', color='rgba(200,200,200,0.3)')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Lower'], name='下軌',
                                         line=dict(dash='dash', color='rgba(200,200,200,0.3)'), fill='tonexty'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='收盤價', line=dict(color='#1f77b4')))
                fig.update_layout(height=450, margin=dict(l=0, r=0, b=0, t=20))
                st.plotly_chart(fig, width='stretch')

            with t2:
                # 快速計算 MACD
                short_ema = df['Close'].ewm(span=12).mean()
                long_ema = df['Close'].ewm(span=26).mean()
                macd_line = short_ema - long_ema
                signal_line = macd_line.ewm(span=9).mean()
                fig_m = go.Figure()
                fig_m.add_trace(go.Bar(x=df['Date'], y=macd_line - signal_line, name='Hist'))
                fig_m.add_trace(go.Scatter(x=df['Date'], y=macd_line, name='DIF'))
                fig_m.update_layout(height=400)
                st.plotly_chart(fig_m, width='stretch')

            with t3:
                fig_ai = plot_plotly(m, fc)
                st.plotly_chart(fig_ai, width='stretch')

                # 多股對比準備
                base = fc.iloc[-predict_days - 1]['yhat']
                fc['return_pct'] = (fc['yhat'] - base) / base * 100
                temp_f = fc[['ds', 'return_pct']].tail(predict_days).copy()
                temp_f['ticker'] = ticker
                all_forecasts.append(temp_f)
    else:
        st.error(f"無法獲取 {ticker} 的資訊。")

# --- 6. 多股對比 ---
if len(all_forecasts) > 1:
    st.divider()
    st.header("🏁 多股 AI 預期報酬率對比")
    comp_df = pd.concat(all_forecasts)
    fig_comp = go.Figure()
    for t in tickers:
        t_data = comp_df[comp_df['ticker'] == t]
        fig_comp.add_trace(go.Scatter(x=t_data['ds'], y=t_data['return_pct'], name=t))
    st.plotly_chart(fig_comp, width='stretch')
