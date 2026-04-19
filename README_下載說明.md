# 台股資料下載說明

## 執行順序

### 第一步：安裝套件
```bash
pip install FinMind tqdm pandas requests
```

### 第二步：下載資料（約 20–30 分鐘）
```bash
python download_twse_data.py
```
- 自動從 TWSE 取得所有上市股票清單（約 900 支）
- 每支股票下載 2010–2024 年完整日K資料
- **支援中斷後繼續**：重新執行會跳過已完成的股票
- 預估完成時間：約 20–30 分鐘

### 第三步：過濾品質不佳的股票
```bash
python filter_data.py
```
- 自動剔除資料缺損、停牌過多、殭屍股等
- 輸出乾淨的訓練資料集

---

## 輸出目錄結構

```
twse_data/
├── raw/               ← 所有原始 CSV（一支股票一個檔案）
│   ├── 2330.csv       ← 台積電
│   ├── 2317.csv       ← 鴻海
│   └── ...
├── filtered/          ← 過濾後的乾淨資料
├── download_log.csv   ← 下載進度紀錄（支援斷點續傳）
├── data_quality.csv   ← 各股資料品質明細
├── passed_stocks.csv  ← 通過過濾的股票清單
├── rejected_stocks.csv ← 被剔除的股票及原因
├── summary.txt        ← 下載完成報告
└── filter_report.txt  ← 過濾完成報告
```

---

## CSV 欄位說明

| 欄位 | 說明 | Kronos 用途 |
|------|------|-------------|
| date | 日期 | 時間索引 |
| open | 開盤價 | 核心特徵 |
| high | 最高價 | 核心特徵 |
| low | 最低價 | 核心特徵 |
| close | 收盤價 | 核心特徵 |
| volume | 成交股數 | 核心特徵 |
| amount | 成交金額（元） | 核心特徵 |
| vwap | 成交量加權平均價 | 由 amount/volume 計算 |

---

## 常見問題

**Q: 下載中斷了怎麼辦？**
A: 直接重新執行 `python download_twse_data.py`，已完成的股票會自動跳過。

**Q: 某些股票一直失敗？**
A: 可能是新上市（資料不足）或已下市股票，這些會自動記錄在 `download_log.csv` 中，之後的 `filter_data.py` 也會過濾掉。

**Q: 過濾後股票數量不夠怎麼辦？**
A: 開啟 `filter_data.py`，調整以下參數放寬標準：
```python
MIN_YEARS = 5        # 改為 5 年（原本 8 年）
MAX_NULL_PCT = 5.0   # 改為 5%（原本 3%）
MIN_AVG_VOLUME = 100 # 改為 100 張（原本 1000 張）
```
