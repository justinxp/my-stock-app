import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd
from datetime import date
import plotly.graph_objects as go
import requests
import time
import random

# --- 1. 網頁配置 ---
st.set_page_config(page_title="AI 股市旗艦分析系統", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 系統核心設定")
    ticker_input = st.text_input("🚀 輸入股票代碼 (多個請用逗號隔開):", value="", placeholder="例如: 2330, 0050, TSLA")
    tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]

    st.markdown("---")
    st.subheader("📊 AI 運算參數")
    train_years = st.slider("訓練數據年數:", 1, 10, 5)
    predict_days = st.slider("預測未來天數:", 30, 365, 90)
    st.caption(f"最後更新: {date.today()} | v8.0 穩定整合版")


# --- 3. 核心數據函數 (雲端穩定優化) ---

@st.cache_data(ttl=300)
def fetch_stock_data(ticker, years):
    """抓取歷史價格，加入 User-Agent 防止雲端阻擋"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # 台股邏輯自動修正
    search_list = []
    if ticker.isdigit():
        search_list = [f"{ticker}.TW", f"{ticker}.TWO"]
    else:
        search_list = [ticker]

    for full_ticker in search_list:
        try:
            # 增加少許隨機延遲，防止雲端併發請求被擋
            time.sleep(random.uniform(0.5, 1.5))

            # 使用 yfinance 的下載功能
            df = yf.download(full_ticker, start=f"{date.today().year - years}-01-01",
                             auto_adjust=False, progress=False)

            if not df.empty:
                df.reset_index(inplace=True)
                # 拍平多重索引 (處理 yfinance v0.2.x 結構)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # 強制轉換日期格式，解決 Prophet 報錯
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                return df, full_ticker
        except Exception as e:
            continue
    return None, None


@st.cache_data(ttl=3600)
def fetch_financials(full_ticker):
    """加強版：多重來源抓取財務指標"""
    try:
        tk = yf.Ticker(full_ticker)
        # 優先嘗試 info
        info = tk.info
        
        # 備援機制：如果 info 為空或關鍵欄位缺失，嘗試從 fast_info 獲取
        eps = info.get('trailingEps')
        if eps is None or eps == "N/A":
            # 某些標的 (如 ETF) 本來就沒有 EPS，顯示 0.0 或 --
            eps = "N/A"
            
        div = info.get('trailingAnnualDividendYield')
        if div is None:
            # 嘗試抓取股利歷史紀錄來估算
            try:
                div_history = tk.dividends
                if not div_history.empty:
                    # 取最近一年的股利總和
                    last_year_div = div_history.tail(4).sum()
                    price = info.get('regularMarketPrice', 1)
                    div = (last_year_div / price) if price else 0
                else:
                    div = 0
            except:
                div = 0
                
        pe = info.get('trailingPE', "N/A")
        
        return {
            "eps": eps,
            "div": div * 100 if isinstance(div, (int, float)) else 0,
            "pe": pe,
            "name": info.get('longName', full_ticker)
        }
    except Exception as e:
        # 最終保險門檻
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
            # --- 技術指標計算 ---
            df['MA20'] = df['Close'].rolling(20).mean()
            std = df['Close'].rolling(20).std()
            df['Upper'] = df['MA20'] + (std * 2)
            df['Lower'] = df['MA20'] - (std * 2)

            last_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            day_change = last_price - prev_price
            day_pct = (day_change / prev_price) * 100

            f_info = fetch_financials(full_name)

            # --- AI 預測 ---
            with st.spinner(f"正在分析 {ticker} 之未來趨勢..."):
                df_p = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
                m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
                m.fit(df_p)
                future = m.make_future_dataframe(periods=predict_days)
                fc = m.predict(future)
                ai_trend = ((fc['yhat'].iloc[-1] - last_price) / last_price) * 100

            # --- 顯示診斷報告 ---
            st.divider()
            st.subheader(f"📊 {f_info['name']} ({ticker}) 綜合分析")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("當前收盤", f"{last_price:.2f}", f"{day_change:+.2f} ({day_pct:+.2f}%)")
            col2.metric("當日高/低", f"{df['High'].iloc[-1]:.2f}", f"{df['Low'].iloc[-1]:.2f}", delta_color="off")
            col3.metric("每股盈餘 (EPS)", f"{f_info['eps']}")
            col4.metric("股息殖利率", f"{f_info['div']:.2f}%")
            col5.metric("本益比 (P/E)", f"{f_info['pe']}")

            # --- 智能交易導航 ---
            st.write(f"**🧭 AI 進出場建議：** " +
                     ("🟢 **建議佈局** (AI 預測強勁漲幅)" if ai_trend > 6 else
                      "🔴 **建議減碼** (AI 預測短線走弱)" if ai_trend < -4 else "⚪ **持股觀望** (趨勢盤整中)"))
            st.write(f"ℹ️ AI 預估未來 {predict_days} 天變動率：`{ai_trend:.2f}%`")

            # --- 分頁圖表 ---
            t1, t2, t3 = st.tabs(["📈 趨勢與布林", "📊 MACD 動能", "🔮 AI 深度預測"])

            with t1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Upper'], name='布林上軌',
                                         line=dict(dash='dash', color='rgba(150,150,150,0.5)')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Lower'], name='布林下軌',
                                         line=dict(dash='dash', color='rgba(150,150,150,0.5)'), fill='tonexty'))
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=df['Close'], name='收盤價', line=dict(color='#1f77b4', width=2)))
                fig.update_layout(height=450, hovermode="x unified", margin=dict(l=0, r=0, b=0, t=10))
                st.plotly_chart(fig, use_container_width=True)

            with t2:
                short = df['Close'].ewm(span=12).mean()
                long = df['Close'].ewm(span=26).mean()
                macd = short - long
                sig = macd.ewm(span=9).mean()
                fig_m = go.Figure()
                fig_m.add_trace(go.Bar(x=df['Date'], y=macd - sig, name='柱狀體'))
                fig_m.add_trace(go.Scatter(x=df['Date'], y=macd, name='DIF快線'))
                fig_m.update_layout(height=400, margin=dict(l=0, r=0, b=0, t=10))
                st.plotly_chart(fig_m, use_container_width=True)

            with t3:
                fig_ai = plot_plotly(m, fc)
                fig_ai.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=10))
                st.plotly_chart(fig_ai, use_container_width=True)

                # 報酬率收集
                base_price = fc.iloc[-predict_days - 1]['yhat']
                fc['return_pct'] = (fc['yhat'] - base_price) / base_price * 100
                temp_f = fc[['ds', 'return_pct']].tail(predict_days).copy()
                temp_f['ticker'] = ticker
                all_forecasts.append(temp_f)
        else:
            st.error(f"❌ 無法獲取 {ticker} 的歷史數據。請確認代碼是否正確，或稍後再試。")

    # --- 多股對比 ---
    if len(all_forecasts) > 1:
        st.divider()
        st.header("🏁 多股 AI 預期報酬率對比 (%)")
        st.caption("說明：此圖表比較各標的從今日起往後的 AI 預測漲跌百分比。")
        comp_df = pd.concat(all_forecasts)
        fig_comp = go.Figure()
        for t in tickers:
            t_data = comp_df[comp_df['ticker'] == t]
            fig_comp.add_trace(go.Scatter(x=t_data['ds'], y=t_data['return_pct'], name=t))
        fig_comp.update_layout(height=500, yaxis_title="預期報酬率 (%)", hovermode="x unified")
        st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")
st.caption("⚠️ 免責聲明：本工具基於 AI 模型預測，僅供學術研究參考，不構成任何投資建議。投資一定有風險。")

