import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd
from datetime import date
import plotly.graph_objects as go

# --- 1. 網頁配置 ---
st.set_page_config(page_title="AI 股市旗艦分析系統", page_icon="📈", layout="wide")

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
    # 修改點：移除預設代碼，改為空白提示
    ticker_input = st.text_input("🚀 輸入股票代碼 (多個請用逗號隔開):", value="", placeholder="例如: 2330, 0050, TSLA")
    tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]

    st.markdown("---")
    st.subheader("📊 AI 運算參數")
    train_years = st.slider("訓練數據年數:", 1, 10, 5)
    predict_days = st.slider("預測未來天數:", 30, 365, 90)
    st.caption(f"版本：v7.0 專業純淨版")


# --- 3. 核心數據函數 ---

@st.cache_data(ttl=300)
def fetch_stock_data(ticker, years):
    """抓取歷史價格並處理後綴與欄位"""
    for suffix in [".TW", ".TWO", ""]:
        full_ticker = f"{ticker}{suffix}" if suffix else ticker
        try:
            df = yf.download(full_ticker, start=f"{date.today().year - years}-01-01", auto_adjust=False, progress=False)
            if not df.empty:
                df.reset_index(inplace=True)
                # 拍平多重索引確保不噴 KeyError
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df, full_ticker
        except:
            continue
    return None, None


@st.cache_data(ttl=3600)
def fetch_financials(full_ticker):
    """抓取基本面數據 (EPS/股利)"""
    try:
        info = yf.Ticker(full_ticker).info
        return {
            "eps": info.get('trailingEps', "N/A"),
            "div": info.get('trailingAnnualDividendYield', 0) * 100,
            "pe": info.get('trailingPE', "N/A"),
            "name": info.get('longName', full_ticker)
        }
    except:
        return {"eps": "N/A", "div": 0, "pe": "N/A", "name": full_ticker}


# --- 4. 主畫面邏輯 ---
st.title("🛡️ AI 旗艦級股市分析系統")

if not tickers:
    st.info("💡 請在左側輸入股票代碼以開始分析（例如輸入 `2330` 後按 Enter）。")
else:
    all_forecasts = []
    for ticker in tickers:
        df, full_name = fetch_stock_data(ticker, train_years)

        if df is not None:
            # 計算技術指標
            df['MA20'] = df['Close'].rolling(20).mean()
            std = df['Close'].rolling(20).std()
            df['Upper'] = df['MA20'] + (std * 2)
            df['Lower'] = df['MA20'] - (std * 2)

            # 漲幅計算
            last_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            day_change = last_price - prev_price
            day_pct = (day_change / prev_price) * 100

            f_info = fetch_financials(full_name)

            # AI 預測
            with st.spinner(f"正在分析 {ticker}..."):
                df_p = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
                df_p['ds'] = pd.to_datetime(df_p['ds']).dt.tz_localize(None)
                m = Prophet(daily_seasonality=True).fit(df_p)
                future = m.make_future_dataframe(periods=predict_days)
                fc = m.predict(future)
                ai_trend = ((fc['yhat'].iloc[-1] - last_price) / last_price) * 100

            # --- 顯示診斷報告 ---
            st.divider()
            st.subheader(f"📊 {f_info['name']} ({ticker}) 綜合分析")

            col1, col2, col3, col4, col5 = st.columns(5)
            # 強化漲幅 % 顯示
            col1.metric("當前收盤", f"{last_price:.2f}", f"{day_change:+.2f} ({day_pct:+.2f}%)")
            col2.metric("當日高/低", f"{df['High'].iloc[-1]:.1f}", f"{df['Low'].iloc[-1]:.1f}", delta_color="off")
            col3.metric("EPS (盈餘)", f"{f_info['eps']}")
            col4.metric("股息殖利率", f"{f_info['div']:.2f}%")
            col5.metric("本益比 (P/E)", f"{f_info['pe']}")

            # 交易導航
            st.write(f"**🧭 智能建議：** " +
                     ("🟢 建議佈局 (預期漲幅大)" if ai_trend > 6 else
                      "🔴 建議減碼 (預期修正)" if ai_trend < -4 else "⚪ 持股觀望"))

            # 分頁圖表
            t1, t2, t3 = st.tabs(["📈 趨勢/布林", "📊 MACD 動能", "🔮 AI 深度預測"])
            with t1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Upper'], name='布林上軌',
                                         line=dict(dash='dash', color='rgba(150,150,150,0.5)')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Lower'], name='布林下軌',
                                         line=dict(dash='dash', color='rgba(150,150,150,0.5)'), fill='tonexty'))
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=df['Close'], name='收盤價', line=dict(color='#1f77b4', width=2)))
                fig.update_layout(height=450, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

            with t2:
                short = df['Close'].ewm(span=12).mean()
                long = df['Close'].ewm(span=26).mean()
                macd = short - long
                sig = macd.ewm(span=9).mean()
                fig_m = go.Figure()
                fig_m.add_trace(go.Bar(x=df['Date'], y=macd - sig, name='柱狀體'))
                fig_m.add_trace(go.Scatter(x=df['Date'], y=macd, name='DIF快線'))
                fig_m.update_layout(height=400)
                st.plotly_chart(fig_m, use_container_width=True)

            with t3:
                st.plotly_chart(plot_plotly(m, fc), use_container_width=True)
                # 報酬率收集
                base = fc.iloc[-predict_days - 1]['yhat']
                fc['return_pct'] = (fc['yhat'] - base) / base * 100
                temp_f = fc[['ds', 'return_pct']].tail(predict_days).copy()
                temp_f['ticker'] = ticker
                all_forecasts.append(temp_f)
        else:
            st.error(f"無法獲取代碼 {ticker} 的歷史數據，請檢查格式。")

    # 多股對比
    if len(all_forecasts) > 1:
        st.divider()
        st.header("🏁 多股 AI 預期報酬率對比 (%)")
        comp_df = pd.concat(all_forecasts)
        fig_comp = go.Figure()
        for t in tickers:
            t_data = comp_df[comp_df['ticker'] == t]
            fig_comp.add_trace(go.Scatter(x=t_data['ds'], y=t_data['return_pct'], name=t))
        st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")
st.caption("⚠️ 本工具僅供參考，不代表投資建議。數據來源：Yahoo Finance")
