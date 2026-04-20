import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
from functools import lru_cache
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. 頁面配置與主題設定
# ============================================================================
st.set_page_config(
    page_title="Professional Stock Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 專業配色系統
COLORS = {
    "primary": "#0052CC",
    "success": "#34C759",
    "danger": "#FF3B30",
    "warning": "#FF9500",
    "neutral": "#8E8E93",
    "bg_dark": "#0A0E27",
    "bg_light": "#F8F9FA",
    "text_primary": "#1F2937",
    "text_secondary": "#6B7280",
    "border": "#E5E7EB"
}

# 自訂 CSS 美化
st.markdown(f"""
<style>
    * {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    .main {{
        background-color: {COLORS['bg_light']};
    }}

    .stMetric {{
        background-color: white;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid {COLORS['primary']};
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}

    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] {{
        font-weight: 600;
        font-size: 0.95rem;
    }}

    .analysis-card {{
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid {COLORS['border']};
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}

    .signal-bullish {{
        color: {COLORS['success']};
        font-weight: 600;
    }}

    .signal-bearish {{
        color: {COLORS['danger']};
        font-weight: 600;
    }}

    .signal-neutral {{
        color: {COLORS['warning']};
        font-weight: 600;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. 側邊欄配置 - 專業級參數設定
# ============================================================================
logo_path = "stock.png"  # 替換為你的檔案路徑

with st.sidebar:
    if logo_path:
        try:
            st.image(logo_path, width=200)
        except:
            st.info("Logo 載入失敗")
    st.title("⚙️ Analysis Hub")

    st.divider()

    # 股票輸入
    col_input = st.container()
    with col_input:
        st.subheader("📊 Portfolio Management")
        ticker_input = st.text_input(
            "Enter Tickers",
            value="2330,0050,2454",
            placeholder="e.g. 2330, AAPL, 0050",
            label_visibility="collapsed"
        )
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    st.divider()

    # 分析參數
    st.subheader("🔍 Analysis Parameters")

    col1, col2 = st.columns(2)
    with col1:
        train_years = st.selectbox(
            "Training Period",
            [1, 2, 3, 5, 10],
            index=2,
            label_visibility="collapsed"
        )
    with col2:
        predict_days = st.selectbox(
            "Forecast Days",
            [30, 60, 90, 180, 365],
            index=2,
            label_visibility="collapsed"
        )

    # 高級選項
    with st.expander("🎯 Advanced Settings"):
        confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
        min_volume = st.number_input("Min Volume Threshold", 1000000, 100000000, 5000000, step=1000000)
        risk_level = st.select_slider(
            "Risk Tolerance",
            ["Conservative", "Moderate", "Aggressive"],
            value="Moderate"
        )

    st.divider()
    st.caption("📌 Professional Platform v2.0")


# ============================================================================
# 3. 核心數據函數 - 增強的數據獲取與計算
# ============================================================================

@st.cache_data(ttl=600)
def fetch_stock_data(ticker, years):
    """獲取股票數據並計算技術指標"""
    search_list = [f"{ticker}.TW", f"{ticker}.TWO"] if ticker.isdigit() else [ticker]

    for full_ticker in search_list:
        try:
            time.sleep(random.uniform(0.3, 0.8))

            # 獲取數據
            df = yf.download(
                full_ticker,
                start=f"{date.today().year - years}-01-01",
                auto_adjust=False,
                progress=False
            )

            if df.empty or len(df) < 20:
                continue

            # 清理多層列索引
            df.reset_index(inplace=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

            # ====== 計算所有技術指標 ======

            # 布林通道
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['Std'] = df['Close'].rolling(window=20).std()
            df['Upper'] = df['MA20'] + (df['Std'] * 2)
            df['Lower'] = df['MA20'] - (df['Std'] * 2)

            # MACD
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema12 - ema26
            df['Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal']

            # RSI (相對強度指數)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # ATR (真實波幅)
            df['TR'] = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    abs(df['High'] - df['Close'].shift()),
                    abs(df['Low'] - df['Close'].shift())
                )
            )
            df['ATR'] = df['TR'].rolling(window=14).mean()

            # 成交量移動平均
            df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()

            # 獲取最新價格信息
            tk = yf.Ticker(full_ticker)
            info = tk.info

            actual_last = float(info.get('regularMarketPrice', df['Close'].iloc[-1]))
            actual_prev = float(info.get('regularMarketPreviousClose', df['Close'].iloc[-2]))

            df.loc[df.index[-1], 'Close'] = actual_last

            return {
                "df": df.dropna(subset=['MACD_Hist', 'RSI']).reset_index(drop=True),
                "last_p": actual_last,
                "prev_p": actual_prev,
                "name": info.get('longName', full_ticker),
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', 0),
                "pb_ratio": info.get('priceToBook', 0),
                "dividend_yield": info.get('dividendYield', 0)
            }, full_ticker

        except Exception as e:
            continue

    return None, None


# ============================================================================
# 4. 技術指標與信號分析
# ============================================================================

def calculate_technical_scores(df):
    """計算綜合技術評分 (0-100)"""
    last = df.iloc[-1]
    scores = {}

    # 趨勢評分 (布林通道)
    if last['Close'] > last['Upper']:
        scores['trend'] = 20  # 超買
    elif last['Close'] < last['Lower']:
        scores['trend'] = 80  # 超賣
    else:
        scores['trend'] = 50 + (last['Close'] - last['Lower']) / (last['Upper'] - last['Lower']) * 50

    # 動能評分 (MACD)
    macd_score = 50
    if last['MACD'] > last['Signal'] and last['MACD_Hist'] > 0:
        macd_score = 75  # 強勢
    elif last['MACD'] < last['Signal'] and last['MACD_Hist'] < 0:
        macd_score = 25  # 弱勢
    scores['momentum'] = macd_score

    # 超買超賣評分 (RSI)
    scores['rsi'] = last['RSI'] if 0 <= last['RSI'] <= 100 else 50

    # 成交量評分
    current_vol = df['Volume'].iloc[-1]
    vol_avg = df['Vol_MA20'].iloc[-1]
    vol_score = min(100, (current_vol / vol_avg) * 50 + 25) if vol_avg > 0 else 50
    scores['volume'] = vol_score

    # 加權綜合分數
    composite = (
            scores['trend'] * 0.35 +
            scores['momentum'] * 0.30 +
            scores['rsi'] * 0.20 +
            scores['volume'] * 0.15
    )

    return {
        'composite': composite,
        'trend': scores['trend'],
        'momentum': scores['momentum'],
        'rsi': scores['rsi'],
        'volume': scores['volume']
    }


def get_signal_recommendation(composite_score, ai_trend):
    """根據分數生成信號與建議"""
    if composite_score >= 70 and ai_trend > 5:
        return {
            'signal': '🟢 STRONG BUY',
            'color': 'success',
            'recommendation': '強勢買進信號 - 技術面與 AI 預測均呈樂觀',
            'confidence': 'HIGH'
        }
    elif composite_score >= 60 and ai_trend > 0:
        return {
            'signal': '🔵 BUY',
            'color': 'info',
            'recommendation': '適度買進 - 累積部位的好時機',
            'confidence': 'MODERATE'
        }
    elif composite_score <= 30 and ai_trend < -5:
        return {
            'signal': '🔴 STRONG SELL',
            'color': 'danger',
            'recommendation': '強勢賣出信號 - 建議規避或減碼',
            'confidence': 'HIGH'
        }
    elif composite_score <= 40 and ai_trend < 0:
        return {
            'signal': '🟠 SELL',
            'color': 'warning',
            'recommendation': '考慮減碼 - 風險提升',
            'confidence': 'MODERATE'
        }
    else:
        return {
            'signal': '⚪ HOLD',
            'color': 'secondary',
            'recommendation': '持股觀望 - 等待更明確的方向',
            'confidence': 'NEUTRAL'
        }


def calculate_risk_metrics(df, last_price):
    """計算風險指標"""
    returns = df['Close'].pct_change().dropna()

    # 波動率 (年化)
    volatility = returns.std() * np.sqrt(252) * 100

    # 最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # 夏普比率 (假設無風險率 2%)
    risk_free_rate = 0.02
    annual_return = returns.mean() * 252
    sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

    # VaR (95% 信心水準)
    var_95 = np.percentile(returns, 5) * 100

    return {
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'var_95': var_95,
        'avg_daily_change': returns.mean() * 100
    }


# ============================================================================
# 5. AI 預測與趨勢分析
# ============================================================================

@st.cache_data(ttl=600)
def forecast_stock_price(df, predict_days):
    """Prophet 時間序列預測"""
    try:
        df_p = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

        # 配置 Prophet (針對財務數據優化)
        m = Prophet(
            daily_seasonality=False,
            yearly_seasonality=True,
            weekly_seasonality=True,
            interval_width=0.95
        )
        m.fit(df_p)

        future = m.make_future_dataframe(periods=predict_days)
        forecast = m.predict(future)

        return m, forecast
    except Exception as e:
        st.warning(f"預測生成失敗: {str(e)}")
        return None, None


# ============================================================================
# 6. 主程式邏輯
# ============================================================================

def main():
    # 標題區域
    col_title, col_info = st.columns([3, 1])
    with col_title:
        st.title("📈 Professional Stock Analysis Platform")
        st.markdown("*Institutional-Grade Technical & Quantitative Analysis*")

    with col_info:
        st.metric("Platform Version", "2.0 PRO")

    if not tickers:
        st.info("👈 請在左側輸入股票代碼開始分析")
        return

    # 準備多股數據
    all_data = {}
    all_forecasts = []

    for ticker in tickers:
        result, full_ticker = fetch_stock_data(ticker, train_years)
        if result:
            all_data[ticker] = {
                'result': result,
                'full_ticker': full_ticker
            }

    if not all_data:
        st.error("❌ 無法獲取數據，請檢查股票代碼")
        return

    # ====== 單股詳細分析 ======
    for ticker in tickers:
        if ticker not in all_data:
            continue

        data = all_data[ticker]
        result = data['result']
        df = result["df"]
        last_price = result["last_p"]
        prev_price = result["prev_p"]
        display_name = result["name"]

        # 計算所有指標
        day_change = last_price - prev_price
        day_pct = (day_change / prev_price) * 100

        technical_scores = calculate_technical_scores(df)
        risk_metrics = calculate_risk_metrics(df, last_price)

        # AI 預測
        with st.spinner(f"分析 {ticker}..."):
            m, forecast = forecast_stock_price(df, predict_days)
            if m is not None and forecast is not None:
                ai_trend = ((forecast['yhat'].iloc[-1] - last_price) / last_price) * 100
            else:
                ai_trend = 0

        signal = get_signal_recommendation(technical_scores['composite'], ai_trend)

        # ====== UI 呈現 ======
        with st.container():
            st.markdown(f"""
            <div class="analysis-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin:0;">{display_name} <code style="background:#f0f0f0; padding:0.2em 0.4em;">{ticker}</code></h2>
                    </div>
                    <div style="text-align: right;">
                        <h1 style="margin:0; color:{'#34C759' if day_change >= 0 else '#FF3B30'};">
                            NT${last_price:.2f}
                        </h1>
                        <p style="margin:0; font-size:1.1em; color:{'#34C759' if day_change >= 0 else '#FF3B30'};">
                            {day_change:+.2f} ({day_pct:+.2f}%)
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 信號區域
        col_signal = st.columns(1)[0]
        with col_signal:
            signal_color = {
                'success': '🟢',
                'danger': '🔴',
                'warning': '🟠',
                'secondary': '⚪'
            }

            st.markdown(f"""
            <div style="background: {'#f0fdf4' if signal['color'] == 'success' else '#fef2f2' if signal['color'] == 'danger' else '#fffbeb'}; 
                        border-left: 4px solid {'#34C759' if signal['color'] == 'success' else '#FF3B30' if signal['color'] == 'danger' else '#FF9500'}; 
                        padding: 1rem; border-radius: 8px;">
                <h3 style="margin-top:0;">{signal['signal']}</h3>
                <p style="margin: 0.5rem 0;">{signal['recommendation']}</p>
                <p style="margin: 0; font-size: 0.9em; color: #6B7280;">
                    信心度: <strong>{signal['confidence']}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # 關鍵指標區域
        st.subheader("📊 Key Metrics")

        metric_cols = st.columns(5)

        with metric_cols[0]:
            st.metric(
                "Technical Score",
                f"{technical_scores['composite']:.0f}/100",
                f"{technical_scores['composite'] - 50:+.0f}"
            )

        with metric_cols[1]:
            st.metric(
                "RSI (14)",
                f"{technical_scores['rsi']:.1f}",
                "超買" if technical_scores['rsi'] > 70 else "超賣" if technical_scores['rsi'] < 30 else "中立"
            )

        with metric_cols[2]:
            st.metric(
                "Volatility (年化)",
                f"{risk_metrics['volatility']:.2f}%",
                f"{risk_metrics['avg_daily_change']:+.3f}%/日"
            )

        with metric_cols[3]:
            st.metric(
                "Max Drawdown",
                f"{risk_metrics['max_drawdown']:.2f}%",
                "風險指標"
            )

        with metric_cols[4]:
            st.metric(
                "AI 預測 (7-365天)",
                f"{ai_trend:+.2f}%",
                "預期變動"
            )

        st.divider()

        # 圖表標籤
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 技術面",
            "📊 動能分析",
            "🔮 AI 預測",
            "⚠️ 風險報告"
        ])

        # ====== 技術面圖表 ======
        with tab1:
            fig_tech = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )

            # 主圖: 價格 + 布林
            fig_tech.add_trace(
                go.Scatter(
                    x=df['Date'], y=df['Upper'],
                    name='Upper Band',
                    line=dict(color='rgba(200,200,200,0.3)'),
                    fill=None
                ),
                row=1, col=1
            )
            fig_tech.add_trace(
                go.Scatter(
                    x=df['Date'], y=df['Lower'],
                    name='Lower Band',
                    line=dict(color='rgba(200,200,200,0.3)'),
                    fill='tonexty',
                    fillcolor='rgba(200,200,200,0.1)'
                ),
                row=1, col=1
            )
            fig_tech.add_trace(
                go.Scatter(
                    x=df['Date'], y=df['Close'],
                    name='Close Price',
                    line=dict(color=COLORS['primary'], width=2)
                ),
                row=1, col=1
            )
            fig_tech.add_trace(
                go.Scatter(
                    x=df['Date'], y=df['MA20'],
                    name='MA 20',
                    line=dict(color=COLORS['warning'], width=1, dash='dash')
                ),
                row=1, col=1
            )

            # 副圖: 成交量
            colors_vol = [COLORS['success'] if df['Close'].iloc[i] >= df['Close'].iloc[i - 1] else COLORS['danger']
                          for i in range(1, len(df))]
            fig_tech.add_trace(
                go.Bar(
                    x=df['Date'], y=df['Volume'],
                    name='Volume',
                    marker=dict(color=colors_vol)
                ),
                row=2, col=1
            )

            fig_tech.update_layout(
                height=600,
                hovermode='x unified',
                margin=dict(l=0, r=0, b=0, t=10)
            )
            st.plotly_chart(fig_tech, use_container_width=True)

        # ====== MACD 與 RSI ======
        with tab2:
            fig_macd = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.12,
                row_heights=[0.5, 0.5]
            )

            # MACD
            colors_macd = [COLORS['success'] if h >= 0 else COLORS['danger'] for h in df['MACD_Hist']]
            fig_macd.add_trace(
                go.Bar(x=df['Date'], y=df['MACD_Hist'], name='MACD Histogram',
                       marker=dict(color=colors_macd), showlegend=False),
                row=1, col=1
            )
            fig_macd.add_trace(
                go.Scatter(x=df['Date'], y=df['MACD'], name='MACD',
                           line=dict(color=COLORS['primary'])),
                row=1, col=1
            )
            fig_macd.add_trace(
                go.Scatter(x=df['Date'], y=df['Signal'], name='Signal',
                           line=dict(color=COLORS['warning'])),
                row=1, col=1
            )

            # RSI
            fig_macd.add_hline(y=70, line_dash="dash", line_color="red",
                               annotation_text="Overbought", row=2, col=1)
            fig_macd.add_hline(y=30, line_dash="dash", line_color="green",
                               annotation_text="Oversold", row=2, col=1)
            fig_macd.add_trace(
                go.Scatter(x=df['Date'], y=df['RSI'], name='RSI (14)',
                           line=dict(color=COLORS['primary'], width=2),
                           fill='tozeroy'),
                row=2, col=1
            )

            fig_macd.update_layout(height=550, hovermode='x unified', margin=dict(l=0, r=0, b=0, t=10))
            st.plotly_chart(fig_macd, use_container_width=True)

        # ====== AI 預測 ======
        with tab3:
            if m is not None and forecast is not None:
                fig_ai = go.Figure()

                # 歷史價格
                fig_ai.add_trace(
                    go.Scatter(
                        x=df['Date'], y=df['Close'],
                        name='Historical Price',
                        line=dict(color=COLORS['primary'], width=2)
                    )
                )

                # 預測價格
                forecast_dates = forecast[forecast['ds'] >= df['Date'].max()]['ds']
                forecast_values = forecast[forecast['ds'] >= df['Date'].max()]['yhat']
                forecast_upper = forecast[forecast['ds'] >= df['Date'].max()]['yhat_upper']
                forecast_lower = forecast[forecast['ds'] >= df['Date'].max()]['yhat_lower']

                fig_ai.add_trace(
                    go.Scatter(
                        x=forecast_dates, y=forecast_upper,
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,100,200,0)',
                        name='95% Confidence Upper'
                    )
                )
                fig_ai.add_trace(
                    go.Scatter(
                        x=forecast_dates, y=forecast_lower,
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,100,200,0)',
                        name='95% Confidence Lower',
                        fillcolor='rgba(0,100,200,0.2)'
                    )
                )
                fig_ai.add_trace(
                    go.Scatter(
                        x=forecast_dates, y=forecast_values,
                        name='Forecast',
                        line=dict(color=COLORS['warning'], width=2, dash='dash')
                    )
                )

                fig_ai.update_layout(
                    height=500,
                    hovermode='x unified',
                    margin=dict(l=0, r=0, b=0, t=10),
                    title=f"AI 預測未來 {predict_days} 天價格走勢"
                )
                st.plotly_chart(fig_ai, use_container_width=True)

                # 預測統計
                col_pred = st.columns(3)
                with col_pred[0]:
                    st.metric("目標價 (未來)", f"NT${forecast_values.iloc[-1]:.2f}",
                              f"{ai_trend:+.2f}%")
                with col_pred[1]:
                    st.metric("預測區間", f"NT${forecast_lower.min():.2f} - {forecast_upper.max():.2f}")
                with col_pred[2]:
                    st.metric("置信度", "95%", "統計區間")

        # ====== 風險報告 ======
        with tab4:
            col_risk = st.columns(2)

            with col_risk[0]:
                st.metric("年化波動率", f"{risk_metrics['volatility']:.2f}%",
                          "市場風險度量")
                st.metric("夏普比率", f"{risk_metrics['sharpe_ratio']:.3f}",
                          "風險調整後報酬")

            with col_risk[1]:
                st.metric("最大回撤", f"{risk_metrics['max_drawdown']:.2f}%",
                          "歷史最大虧損")
                st.metric("VaR (95%)", f"{risk_metrics['var_95']:.3f}%",
                          "單日風險值")

            # 風險警示
            st.subheader("⚠️ Risk Assessment")

            risk_assessment = ""
            if risk_metrics['volatility'] > 30:
                risk_assessment += "🔴 **高波動性**: 股價變動劇烈\n\n"
            elif risk_metrics['volatility'] > 20:
                risk_assessment += "🟠 **中等波動性**: 風險適中\n\n"
            else:
                risk_assessment += "🟢 **低波動性**: 相對穩定\n\n"

            if risk_metrics['sharpe_ratio'] > 1:
                risk_assessment += "🟢 **良好的風險調整報酬**\n\n"
            elif risk_metrics['sharpe_ratio'] > 0:
                risk_assessment += "🟠 **可接受的風險調整報酬**\n\n"
            else:
                risk_assessment += "🔴 **差的風險調整報酬**\n\n"

            if abs(risk_metrics['max_drawdown']) > 30:
                risk_assessment += "⚠️ **歷史回撤偏大**，需注意下跌風險"

            st.markdown(risk_assessment)

        st.divider()

    # ====== 多股對比 ======
    if len(tickers) > 1:
        st.header("🏆 Portfolio Comparison")

        comparison_data = []
        for ticker in tickers:
            if ticker in all_data:
                result = all_data[ticker]['result']
                df = result["df"]
                day_change = result["last_p"] - result["prev_p"]
                technical_scores = calculate_technical_scores(df)
                risk_metrics = calculate_risk_metrics(df, result["last_p"])

                comparison_data.append({
                    'Stock': ticker,
                    'Price': result["last_p"],
                    'Change %': (day_change / result["prev_p"]) * 100,
                    'Technical Score': technical_scores['composite'],
                    'Volatility': risk_metrics['volatility'],
                    'Sharpe Ratio': risk_metrics['sharpe_ratio']
                })

        comp_df = pd.DataFrame(comparison_data)

        # 表格展示
        st.dataframe(
            comp_df.style.format({
                'Price': '${:.2f}',
                'Change %': '{:.2f}%',
                'Technical Score': '{:.0f}',
                'Volatility': '{:.2f}%',
                'Sharpe Ratio': '{:.3f}'
            }).highlight_max(subset=['Technical Score'], color='lightgreen')
            .highlight_min(subset=['Volatility'], color='lightblue'),
            use_container_width=True
        )

        # 比較圖表
        fig_comp = go.Figure()

        fig_comp.add_trace(
            go.Bar(
                x=comp_df['Stock'],
                y=comp_df['Technical Score'],
                name='Technical Score',
                marker_color=COLORS['primary']
            )
        )

        fig_comp.update_layout(
            height=400,
            title="Technical Score 對比",
            yaxis_title="Score (0-100)",
            margin=dict(l=0, r=0, b=0, t=30)
        )

        st.plotly_chart(fig_comp, use_container_width=True)


# ============================================================================
# 7. 免責聲明與頁腳
# ============================================================================

st.markdown("---")
col_footer = st.columns(3)

with col_footer[0]:
    st.caption("🔐 Data Source: Yahoo Finance API")

with col_footer[1]:
    st.caption("📊 Analysis Engine: Prophet + Custom Indicators")

with col_footer[2]:
    st.caption("⚠️ Disclaimer: Not investment advice. For research only.")

# ============================================================================
# 運行主程式
# ============================================================================

if __name__ == "__main__":
    main()
