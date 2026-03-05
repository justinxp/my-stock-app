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
    ticker_input = st.text_input("🚀 輸入股票代碼 (多個請用逗號隔開):", value="0050", placeholder="例如: 2330, 0050, TSLA")
    tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
    st.markdown("---")
    train_years = st.slider("訓練數據年數:", 1, 10, 5)
    predict_days = st.slider("預測未來天數:", 30, 365, 90)
    st.caption(f"版本：v9.3 功能完整整合版")


# --- 3. 核心數據函數 (修正 KeyError 與 0050 漲跌) ---

@st.cache_data(ttl=300)
def fetch_stock_data(ticker, years):
    search_list = [f"{ticker}.TW", f"{ticker}.TWO"] if ticker.isdigit() else [ticker]
    for full_ticker in search_list:
        try:
            time.sleep(random.uniform(0.5, 1.0))
            df = yf.download(full_ticker, start=f"{date.today().year - years}-01-01", auto_adjust=False, progress=False)
            if not df.empty and len(df) > 20:
                df.reset_index(inplace=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                
                # 強制計算布林通道欄位，徹底解決 KeyError
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['Std'] = df['Close'].rolling(window=20).std()
                df['Upper'] = df['MA20'] + (df['Std'] * 2)
                df['Lower'] = df['MA20'] - (df['Std'] * 2)
                
                # 取得 Yahoo 官方參考價 (解決 0050 漲跌相反)
                tk = yf.Ticker(full_ticker)
                info = tk.info
                # 優先使用官方參考價，若無則用歷史數據最後兩筆
                actual_last = info.get('regularMarketPrice') if info.get('regularMarketPrice') else float(df['Close'].iloc[-1])
                actual_prev = info.get('regularMarketPreviousClose') if info.get('regularMarketPreviousClose') else float(df['Close'].iloc[-2])
                
                # 確保歷史數據最後一筆與現價同步
                df.loc[df.index[-1], 'Close'] = actual_last
                
                return {
                    "df": df.dropna(subset=['Upper']).reset_index(drop=True),
                    "last_p": actual_last,
                    "prev_p": actual_prev,
                    "name": info.get('longName', full_ticker)
                }, full_ticker
        except:
            continue
    return None, None


# --- 4. 主畫面邏輯 ---
st.title("🛡️ AI 旗艦級股市分析系統")

if not tickers:
    st.info("💡 請在左側輸入股票代碼（如 `2330`）後按 Enter 開始分析。")
else:
    all_forecasts = []
    for ticker in tickers:
        result, full_name = fetch_stock_data(ticker, train_years)

        if result:
            df = result["df"]
            last_price = result["last_p"]
            prev_price = result["prev_p"]
            display_name = result["name"]

            # 計算漲跌
            day_change = last_price - prev_price
            day_pct = (day_change / prev_price) * 100

            with st.spinner(f"正在分析 {ticker}..."):
                # Prophet 預測邏輯
                df_p = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
                m = Prophet(daily_seasonality=True).fit(df_p)
                fc = m.predict(m.make_future_dataframe(periods=predict_days))
                ai_trend = ((fc['yhat'].iloc[-1] - last_price) / last_price) * 100

            st.divider()
            st.subheader(f"📊 {display_name} ({ticker}) 綜合分析")

            # 原有 Metric 功能
            col1, col2 = st.columns(2)
            col1.metric("當前收盤價", f"{last_price:.2f}", f"{day_change:+.2f} ({day_pct:+.2f}%)", delta_color="inverse")
            col2.metric("今日成交區間 (高/低)", f"{df['High'].iloc[-1]:.2f}", f"{df['Low'].iloc[-1]:.2f}", delta_color="off")

            # --- 原有 AI 建議區功能 ---
            trend_color_ball = "🔴" if ai_trend >= 0 else "🟢"
            trend_html_color = "red" if ai_trend >= 0 else "green"

            st.markdown(f"**🧭 AI 進出場建議：**")
            if ai_trend > 5:
                st.success(f"{trend_color_ball} **建議佈局**：AI 預估未來具備較強漲升潛力。")
            elif ai_trend < -5:
                st.error(f"{trend_color_ball} **建議避險**：AI 預估未來面臨較大修正壓力。")
            else:
                st.info(f"⚪ **持股觀望**：預測趨勢處於區間震盪。")

            st.markdown(f"""
                <div style="font-size:1.1em; font-weight:bold; margin-bottom: 20px;">
                    📈 AI 預估未來 <span style="color:blue;">{predict_days}</span> 天變動率：
                    <span style="color:{trend_html_color}; font-size:1.25em;">
                        {ai_trend:+.2f}% {trend_color_ball}
                    </span>
                </div>
            """, unsafe_allow_html=True)

            # --- 分頁功能 (包含完整的 MACD 與 布林) ---
            t1, t2, t3 = st.tabs(["📈 趨勢布林", "📊 MACD 動能", "🔮 AI 預測圖"])

            with t1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Upper'], name='上軌', line=dict(dash='dash', color='rgba(150,150,150,0.3)')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Lower'], name='下軌', line=dict(dash='dash', color='rgba(150,150,150,0.3)'), fill='tonexty'))
                l_color = 'red' if day_change >= 0 else 'green'
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='收盤價', line=dict(color=l_color, width=2)))
                fig.update_layout(height=450, hovermode="x unified", margin=dict(l=0,r=0,b=0,t=10))
                st.plotly_chart(fig, use_container_width=True)

            with t2:
                # 恢復原本的 MACD 繪圖功能
                short, long = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
                macd, sig = short - long, (short - long).ewm(span=9).mean()
                hist = macd - sig
                colors = ['red' if x >= 0 else 'green' for x in hist]
                fig_m = go.Figure()
                fig_m.add_trace(go.Bar(x=df['Date'], y=hist, name='柱狀體', marker_color=colors))
                fig_m.add_trace(go.Scatter(x=df['Date'], y=macd, name='DIF', line=dict(color='orange')))
                fig_m.add_trace(go.Scatter(x=df['Date'], y=sig, name='MACD', line=dict(color='cyan')))
                fig_m.update_layout(height=400, margin=dict(l=0, r=0, b=0, t=10), hovermode="x unified")
                st.plotly_chart(fig_m, use_container_width=True)

            with t3:
                fig_ai = plot_plotly(m, fc)
                fig_ai.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=10))
                st.plotly_chart(fig_ai, use_container_width=True)

                # 準備多股對比數據
                base_p = fc.iloc[-predict_days - 1]['yhat']
                fc['return_pct'] = (fc['yhat'] - base_p) / base_p * 100
                temp_f = fc[['ds', 'return_pct']].tail(predict_days).copy()
                temp_f['ticker'] = ticker
                all_forecasts.append(temp_f)

    # --- 原有多股對比功能 ---
    if len(all_forecasts) > 1:
        st.divider()
        st.header("🏁 多股預期報酬對比 (%)")
        comp_df = pd.concat(all_forecasts)
        fig_comp = go.Figure()
        for t in [f.ticker.iloc[0] for f in all_forecasts]:
            t_data = comp_df[comp_df['ticker'] == t]
            fig_comp.add_trace(go.Scatter(x=t_data['ds'], y=t_data['return_pct'], name=t))
        fig_comp.update_layout(height=500, hovermode="x unified", yaxis_title="預期變動 (%)")
        st.plotly_chart(fig_comp, use_container_width=True)

# --- 原有警告語 ---
st.markdown("---")
st.caption("⚠️ 本工具僅供 AI 技術研究參考，不構成投資建議，投資人應獨立評估風險。數據來源：Yahoo Finance")
