# 🚀 Professional Stock Analysis Platform - Quick Start Guide

## ⚡ 5 分鐘快速開始

### 步驟 1: 環境準備 (1 分鐘)

```bash
# 創建虛擬環境
python -m venv venv

# 激活虛擬環境
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### 步驟 2: 安裝依賴 (2 分鐘)

```bash
pip install -r requirements.txt
```

### 步驟 3: 運行應用 (1 分鐘)

```bash
streamlit run professional_stock_analyzer.py
```

✅ 應用將在 http://localhost:8501 打開！

---

## 📖 基本使用流程

### 第一次運行

1. **打開應用** → 自動載入頁面
2. **在左側邊欄輸入股票代碼**
   - 例: `2330` (台積電)
   - 例: `0050` (元大 50)
   - 例: `AAPL` (蘋果)
3. **按 Enter 鍵** → 系統開始分析
4. **等待數據加載** (通常 5-10 秒)
5. **查看分析結果**

### 分析結果解讀

```
📊 股票概覽卡片
├─ 股票名稱與代碼
├─ 當前價格
└─ 今日漲跌幅

🧭 交易信號區
├─ STRONG BUY / BUY / HOLD / SELL / STRONG SELL
├─ 具體建議
└─ 信心度等級

📈 關鍵指標 (5 個核心指標)
├─ 綜合技術評分 (0-100)
├─ RSI 指數
├─ 年化波動率
├─ 最大回撤
└─ AI 預測變動率

📊 詳細圖表 (4 個標籤)
├─ 技術面 (價格 + 布林通道 + 成交量)
├─ 動能分析 (MACD + RSI)
├─ AI 預測 (Prophet 模型)
└─ 風險報告 (波動率、夏普比率、VaR)
```

---

## 🎯 常用操作

### 單股分析

**推薦設置 (中期投資)**:
```
訓練期: 5 年
預測天數: 90 天
風險偏好: Moderate
```

**查看流程**:
1. 輸入 `2330`
2. 查看 MACD 動能 (是否有黃金交叉?)
3. 確認技術評分 (> 70 算良好)
4. 查看風險報告 (波動率是否可接受?)
5. 結合 AI 預測 (方向是否樂觀?)

### 多股對比

**如何操作**:
```
1. 在「Enter Tickers」中輸入多個代碼
   例: 2330,0050,2454,TSLA

2. 系統自動分析每一檔

3. 最後會生成「Portfolio Comparison」
   ├─ 表格對比 (價格、漲幅、評分)
   └─ 圖表展示 (技術評分對比)
```

### 不同策略參數

#### 短期交易 (日線/週線)
```
Training Period: 1-2 年
Forecast Days: 30 天
關注指標: MACD 柱狀圖、RSI
```

#### 中期波段 (月線)
```
Training Period: 5 年 ⭐ 推薦
Forecast Days: 90 天
關注指標: 布林通道、技術評分
```

#### 長期配置 (季線/年線)
```
Training Period: 10 年
Forecast Days: 180-365 天
關注指標: 夏普比率、最大回撤
```

---

## 💻 常見命令

### 清除緩存

Prophet 和數據會被緩存 600 秒以提高性能。如需強制更新:

```bash
# 方法 1: 刷新瀏覽器 Ctrl+F5

# 方法 2: 在終端重啟應用
# 按 Ctrl+C 停止
# 再次運行:
streamlit run professional_stock_analyzer.py --logger.level=debug
```

### 調整埠號

默認埠為 8501，如需更改:

```bash
streamlit run professional_stock_analyzer.py --server.port 8502
```

### 部署到雲端

#### Streamlit Cloud (免費)
```bash
# 1. 推送代碼到 GitHub
git push origin main

# 2. 訪問 https://share.streamlit.io
# 3. 選擇 repo → main → professional_stock_analyzer.py
# 4. 部署完成！
```

#### Heroku 部署
```bash
# 創建 Procfile:
echo "web: streamlit run professional_stock_analyzer.py" > Procfile

# 部署:
heroku login
heroku create your-app-name
git push heroku main
```

---

## 🎨 自訂設置

### 修改配色

編輯文件中的 `COLORS` 字典:

```python
COLORS = {
    "primary": "#0052CC",      # 主色 (藍色)
    "success": "#34C759",      # 上升色 (綠色)
    "danger": "#FF3B30",       # 下降色 (紅色)
    "warning": "#FF9500",      # 警告色 (橙色)
    ...
}
```

### 修改技術評分權重

編輯 `calculate_technical_scores()` 函數:

```python
composite = (
    scores['trend'] * 0.35 +      # ← 修改這些數值
    scores['momentum'] * 0.30 +
    scores['rsi'] * 0.20 +
    scores['volume'] * 0.15
)
```

### 修改信號臨界值

編輯 `get_signal_recommendation()` 函數:

```python
if composite_score >= 70 and ai_trend > 5:  # ← 調整臨界值
    return {
        'signal': '🟢 STRONG BUY',
        ...
    }
```

---

## 📚 數據來源與注意事項

### Yahoo Finance 數據特性

| 特性 | 說明 |
|-----|------|
| 延遲 | 15-20 分鐘 (實時行情需付費) |
| 準確度 | 美股優於台股 |
| 歷史數據 | 一般可追溯 20+ 年 |
| 成交量 | 單位因市場而異 |

### 台灣股票代碼格式

```
系統自動轉換:
輸入 2330      → 自動搜索 2330.TW 與 2330.TWO
輸入 AAPL      → 直接搜索 AAPL (美股)

手動指定:
2330.TW        → 台灣證交所上市股票
2330.TWO       → 台灣證交所櫃檯股票
```

---

## 🔍 故障排除

### 問題 1: 「No module named 'streamlit'」

**原因**: 依賴未正確安裝

**解決**:
```bash
# 確保在虛擬環境中
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 問題 2: 數據載入失敗

**可能原因**:
- 網絡連接問題
- Yahoo Finance API 限流
- 股票代碼不存在

**解決**:
```python
# 測試連接
import yfinance as yf
data = yf.download("2330.TW", period="1y")
print(data.head())
```

### 問題 3: Prophet 預測出現 NaN 值

**原因**: 數據點不足

**解決**:
- 增加 Training Period 至 3+ 年
- 確保股票有足夠的交易歷史

### 問題 4: 圖表無法顯示

**原因**: Plotly 版本兼容性

**解決**:
```bash
pip install --upgrade plotly
```

---

## 📊 最佳實踐

### ✅ 推薦做法

1. **定期檢視** - 每週檢查一次您的投資組合
2. **多角度分析** - 同時查看技術面、動能面、風險面
3. **結合新聞** - 技術分析 + 基本面分析 + 新聞動向
4. **風險管理** - 檢查波動率與最大回撤，調整倉位
5. **長期視角** - 不要過度交易，相信長期趨勢

### ❌ 避免做法

1. ❌ 盲目追逐漲跌幅
2. ❌ 忽視風險指標
3. ❌ 交易不流動的股票
4. ❌ 違背 AI 信號短期內頻繁操作
5. ❌ 只看一個指標決策

---

## 🎓 進階學習

### 理解技術指標

**布林通道 (Bollinger Bands)**
- 用途: 識別超買超賣
- 原理: 價格在均線 ±2 個標準差
- 信號: 價格碰布林上軌 → 考慮減碼

**MACD (移動平均聚散指標)**
- 用途: 判斷動能與趨勢
- 原理: DIF > Signal 為正信號
- 信號: 黃金交叉 (DIF ↗ Signal) → 買進

**RSI (相對強度指數)**
- 用途: 判斷超買超賣
- 原理: 0-100 的相對強度衡量
- 信號: RSI > 70 超買，< 30 超賣

### 理解風險指標

**波動率**
- 衡量: 股價波動劇烈程度
- 用途: 評估風險大小
- 應用: 波動率高 → 倉位應更小

**夏普比率**
- 衡量: 單位風險報酬
- 用途: 比較投資品質
- 應用: > 1 為優秀投資

**最大回撤**
- 衡量: 歷史最大虧損
- 用途: 評估最壞情況
- 應用: 控制風險承受範圍

---

## 📞 支援資訊

### 常見問題

**Q: 系統是否實時更新?**
A: 數據緩存 600 秒以提高性能。可刷新瀏覽器獲得最新數據。

**Q: 能否用於實際交易?**
A: 可以參考，但應結合其他分析與專業建議。本工具僅供研究。

**Q: 支援多少檔股票同時分析?**
A: 理論上無限制，但建議 ≤ 10 檔以保持性能。

**Q: 歷史數據可追溯多久?**
A: 一般 10+ 年，但某些新股票或小型股可能更短。

---

## ⚠️ 免責聲明

```
本工具僅供教育與研究目的。
- 不構成投資建議
- 過去表現不保證未來結果
- 使用者需自行承擔投資風險
- 建議諮詢專業財務顧問
```

---

祝您投資順利！🚀📈

有任何問題，查看完整的 **DEPLOYMENT_GUIDE.md** 文檔。
