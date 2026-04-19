"""
TWSE 台股歷史日K資料下載腳本
================================
下載所有 TWSE 上市股票 15 年歷史 OHLCV 資料，用於 Kronos fine-tune 訓練。

資料來源：
  主要：FinMind API（每支股票一次請求，最有效率）
  備援：TWSE 官方 API（免費無限制，但速度較慢）

【重要】FinMind 免費額度限制說明：
  - 未登入：每日約 300 次請求
  - 免費註冊帳號（推薦）：每日 600 次請求
  - 付費方案：不限次數

  → 建議先在 https://finmindtrade.com/ 免費註冊取得 token
  → 將 token 填入下方 FINMIND_TOKEN 變數
  → 若不填 token，腳本會在額度耗盡時自動切換為 TWSE 備援來源繼續下載

使用方式：
    pip install pandas requests
    python download_twse_data.py

輸出：
    ./twse_data/raw/              每支股票一個 CSV 檔
    ./twse_data/download_log.csv  下載進度紀錄（支援中斷後繼續）
    ./twse_data/summary.txt       最終統計報告
"""

import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

try:
    from dateutil.relativedelta import relativedelta
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "python-dateutil", "--break-system-packages", "-q"])
    from dateutil.relativedelta import relativedelta

# ─────────────────────────── 【請填入你的設定】 ───────────────────────────
# 免費註冊網址：https://finmindtrade.com/  → 登入後在帳號頁面複製 API Token
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidmVuY2VuMjY5IiwiZW1haWwiOiJ0YWxhbnQyOEBnbWFpbC5jb20ifQ.3R12rXb2kwtd10clUYN4AByKbfVgC3ARNQ6I8JaqKqo"

START_DATE = "2010-01-01"
END_DATE   = "2024-12-31"
# ──────────────────────────────────────────────────────────────────────────

OUTPUT_DIR   = Path("./twse_data/raw")
LOG_FILE     = Path("./twse_data/download_log.csv")
SUMMARY_FILE = Path("./twse_data/summary.txt")

# 速率控制
DELAY_BETWEEN_STOCKS  = 1.5    # FinMind 模式：每支股票間隔
DELAY_TWSE_PER_MONTH  = 1.8    # TWSE 備援模式：每次月份請求間隔（≤ 3 req/5s）
DELAY_ON_402          = 65.0   # 收到 402 後等待 65 秒（讓額度窗口重置）
MAX_RETRIES           = 3
BATCH_SIZE            = 50

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
TWSE_URL    = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"

# 全域：追蹤是否已切換到 TWSE 備援模式
_use_twse_fallback = False

# ─────────────────────────── 工具函數 ────────────────────────────

def get_twse_stock_list():
    """從 TWSE 官方 OpenAPI 取得所有上市股票代號"""
    print("📋 正在從 TWSE 取得上市股票清單...")
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    try:
        resp = requests.get(url, timeout=15, headers={"Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        # 只取純股票（代號為 4 位數字），排除 ETF、特別股等
        stocks = []
        for item in data:
            code = item.get("公司代號", "").strip()
            name = item.get("公司簡稱", "").strip()
            if code.isdigit() and len(code) == 4:
                stocks.append({"code": code, "name": name})
        print(f"✅ 取得 {len(stocks)} 支上市股票（純股票，已排除 ETF）")
        return stocks
    except Exception as e:
        print(f"⚠️  TWSE API 失敗: {e}，改用備用清單")
        return get_fallback_stock_list()


def get_finmind_stock_list():
    """備援：從 FinMind 取得上市股票清單"""
    print("📋 從 FinMind 取得股票清單...")
    try:
        params = {
            "dataset": "TaiwanStockInfo",
            "token": ""
        }
        resp = requests.get(FINMIND_URL, params=params, timeout=15)
        data = resp.json().get("data", [])
        stocks = []
        for item in data:
            code = item.get("stock_id", "").strip()
            name = item.get("stock_name", "").strip()
            market = item.get("type", "")
            # 只取上市（twse），且代號為 4 位數字
            if code.isdigit() and len(code) == 4 and "twse" in market.lower():
                stocks.append({"code": code, "name": name})
        print(f"✅ FinMind 取得 {len(stocks)} 支上市股票")
        return stocks
    except Exception as e:
        print(f"⚠️  FinMind 清單也失敗: {e}，使用硬編碼備用清單")
        return get_fallback_stock_list()


def get_fallback_stock_list():
    """最終備援：常見台股代號清單（前 100 大市值）"""
    codes = [
        "2330","2317","2454","2308","2303","2882","2881","2412","1301","2002",
        "1303","2886","2891","2884","2887","2892","5880","2885","2880","2883",
        "3711","2379","2609","2207","1216","2345","3034","2395","4904","2357",
        "2382","1101","2327","2353","2408","2474","3045","2301","2337","1326",
        "2363","6505","1402","2105","2912","3008","2801","1304","2347","9910",
        "2371","1605","2498","2392","9917","1802","2485","2388","2368","2376",
        "3481","2915","2049","1590","3702","2404","2059","2344","6415","2201",
        "1102","5876","2385","2449","2204","8046","3533","2356","4938","2441",
        "6669","3考","1210","2614","5871","8�","2376","3考","2618","6220",
        "2006","1319","2328","2231","1909","2313","3037","1514","2360","1476",
    ]
    # 過濾掉非數字
    codes = [c for c in codes if c.isdigit() and len(c) == 4]
    stocks = [{"code": c, "name": f"股票{c}"} for c in codes]
    print(f"⚠️  使用備用清單：{len(stocks)} 支股票")
    return stocks


def load_download_log():
    """載入下載進度紀錄（支援中斷後繼續）"""
    if LOG_FILE.exists():
        df = pd.read_csv(LOG_FILE, dtype=str)
        done = set(df[df["status"] == "ok"]["code"].tolist())
        failed = set(df[df["status"] == "failed"]["code"].tolist())
        print(f"📂 讀取進度紀錄：已完成 {len(done)} 支，失敗 {len(failed)} 支")
        return done, failed
    return set(), set()


def save_log_entry(code, name, status, rows=0, note=""):
    """新增一筆下載紀錄"""
    entry = pd.DataFrame([{
        "code": code, "name": name, "status": status,
        "rows": rows, "note": note,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    if LOG_FILE.exists():
        entry.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        entry.to_csv(LOG_FILE, index=False)


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """統一欄位名稱、計算 VWAP、轉換型別，供兩個資料源共用"""
    rename_map = {
        "max":              "high",
        "min":              "low",
        "Trading_Volume":   "volume",
        "Trading_money":    "amount",
        "Trading_turnover": "turnover",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # VWAP：用成交金額 / 成交股數（最精確），或近似值
    if "amount" in df.columns and "volume" in df.columns:
        df["vwap"] = pd.to_numeric(df["amount"], errors="coerce") / \
                     pd.to_numeric(df["volume"], errors="coerce").replace(0, float("nan"))
    else:
        df["vwap"] = (pd.to_numeric(df["high"], errors="coerce") +
                      pd.to_numeric(df["low"],  errors="coerce") +
                      pd.to_numeric(df["close"],errors="coerce")) / 3

    for col in ["open", "high", "low", "close", "volume", "amount", "vwap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    keep = ["date", "open", "high", "low", "close", "volume", "amount", "vwap"]
    return df[[c for c in keep if c in df.columns]]


def download_stock_finmind(code, start_date, end_date):
    """FinMind API：每次請求拿完整歷史，效率最高"""
    params = {
        "dataset":    "TaiwanStockPrice",
        "data_id":    code,
        "start_date": start_date,
        "end_date":   end_date,
        "token":      FINMIND_TOKEN,
    }
    resp = requests.get(FINMIND_URL, params=params, timeout=30)

    # 402 = 當日額度耗盡，讓呼叫端決定如何處理
    if resp.status_code == 402:
        raise requests.exceptions.HTTPError("402 Payment Required", response=resp)

    resp.raise_for_status()
    result = resp.json()

    if result.get("status") != 200:
        raise ValueError(f"FinMind error: {result.get('msg', 'unknown')}")

    data = result.get("data", [])
    if not data:
        raise ValueError("No data returned from FinMind")

    return _normalize_df(pd.DataFrame(data))


def download_stock_twse(code, start_date, end_date):
    """
    TWSE 官方 API 備援：逐月抓取，無額度限制
    速度較慢（每月一次請求），但完全免費不會 402
    """
    all_rows = []
    cur = datetime.strptime(start_date, "%Y-%m-%d").replace(day=1)
    end = datetime.strptime(end_date,   "%Y-%m-%d")

    month_count = 0
    while cur <= end:
        date_str = cur.strftime("%Y%m%d")
        try:
            resp = requests.get(
                TWSE_URL,
                params={"response": "json", "date": date_str, "stockNo": code},
                timeout=20,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            if resp.status_code == 200:
                raw = resp.json()
                if raw.get("stat") == "OK" and "data" in raw:
                    fields = raw.get("fields", [])
                    for row in raw["data"]:
                        row_dict = dict(zip(fields, row))
                        all_rows.append(row_dict)
        except Exception:
            pass  # 單月失敗繼續，不中斷整支股票

        # 逐月前進
        cur = (cur + relativedelta(months=1)).replace(day=1)
        month_count += 1
        time.sleep(DELAY_TWSE_PER_MONTH)

    if not all_rows:
        raise ValueError("TWSE: no data for this stock")

    df = pd.DataFrame(all_rows)

    # TWSE 欄位名稱為中文，統一轉換
    twse_rename = {
        "日期": "date", "開盤價": "open", "最高價": "high",
        "最低價": "low", "收盤價": "close",
        "成交股數": "volume", "成交金額": "amount",
        "成交筆數": "turnover",
    }
    df = df.rename(columns={k: v for k, v in twse_rename.items() if k in df.columns})

    # 民國年 → 西元年
    def roc_to_date(s):
        try:
            parts = str(s).replace("/", "-").split("-")
            y = int(parts[0]) + 1911
            return f"{y}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
        except:
            return None

    if "date" in df.columns:
        df["date"] = df["date"].apply(roc_to_date)
        df = df.dropna(subset=["date"])

    # 去除千分位逗號
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").str.replace("--", "")

    return _normalize_df(df)


# ─────────────────────────── 主程式 ────────────────────────────

def main():
    print("=" * 60)
    print("  TWSE 台股歷史日K資料下載")
    print(f"  資料期間：{START_DATE} ~ {END_DATE}")
    print("=" * 60)

    # 建立輸出目錄
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 1. 取得股票清單
    stocks = get_twse_stock_list()
    if not stocks:
        stocks = get_finmind_stock_list()

    # 2. 讀取已完成的進度（支援中斷後繼續）
    done_set, failed_set = load_download_log()

    # 過濾掉已完成的
    remaining = [s for s in stocks if s["code"] not in done_set]
    print(f"\n📊 總計：{len(stocks)} 支 | 待下載：{len(remaining)} 支 | 已完成：{len(done_set)} 支")
    print(f"⏱️  預估時間：約 {len(remaining) * DELAY_BETWEEN_STOCKS / 60:.0f} 分鐘\n")

    success_count = 0
    fail_count    = 0
    fail_list     = []
    global _use_twse_fallback

    for i, stock in enumerate(remaining):
        code   = stock["code"]
        name   = stock["name"]
        prefix = f"[{i+1:4d}/{len(remaining)}]"

        downloaded = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # ── 選擇資料源 ──
                if _use_twse_fallback:
                    source = "TWSE"
                    df = download_stock_twse(code, START_DATE, END_DATE)
                else:
                    source = "FinMind"
                    df = download_stock_finmind(code, START_DATE, END_DATE)

                if len(df) < 100:
                    raise ValueError(f"資料量不足（{len(df)} 筆）")

                out_path = OUTPUT_DIR / f"{code}.csv"
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
                save_log_entry(code, name, "ok", rows=len(df), note=source)

                src_tag = "🟢FM" if source == "FinMind" else "🔵TW"
                print(f"{prefix} ✅ {src_tag} {code} {name:<12} {len(df):,} 筆")
                success_count += 1
                downloaded = True
                break

            except requests.exceptions.HTTPError as e:
                if "402" in str(e):
                    # FinMind 額度耗盡 → 自動切換 TWSE 備援
                    if not _use_twse_fallback:
                        print(f"\n{'='*60}")
                        print(f"  ⚠️  FinMind 免費額度已耗盡（402）")
                        print(f"  🔄 自動切換為 TWSE 官方 API 備援模式")
                        print(f"  ℹ️  備援模式速度較慢（每支約 3–5 分鐘）但不限量")
                        if not FINMIND_TOKEN:
                            print(f"  💡 建議：前往 https://finmindtrade.com 免費註冊")
                            print(f"     取得 token 後填入腳本頂端 FINMIND_TOKEN")
                            print(f"     可大幅提升每日額度（下次執行即生效）")
                        print(f"{'='*60}\n")
                        _use_twse_fallback = True
                    # 直接重試（不等待），這次用 TWSE
                    continue
                else:
                    err_msg = str(e)[:80]
                    if attempt < MAX_RETRIES:
                        print(f"{prefix} ⚠️  {code} {name} 第{attempt}次失敗：{err_msg}")
                        time.sleep(10)
                    else:
                        print(f"{prefix} ❌ {code} {name} 失敗：{err_msg}")
                        save_log_entry(code, name, "failed", note=err_msg)
                        fail_count += 1
                        fail_list.append(f"{code} {name}")

            except Exception as e:
                err_msg = str(e)[:80]
                if attempt < MAX_RETRIES:
                    print(f"{prefix} ⚠️  {code} {name} 第{attempt}次失敗：{err_msg}")
                    time.sleep(8)
                else:
                    print(f"{prefix} ❌ {code} {name} 失敗：{err_msg}")
                    save_log_entry(code, name, "failed", note=err_msg)
                    fail_count += 1
                    fail_list.append(f"{code} {name}")

        # 速率控制（TWSE 模式已在函數內部 sleep）
        if not _use_twse_fallback:
            time.sleep(DELAY_BETWEEN_STOCKS)

        # 每 50 支顯示一次進度摘要
        if (i + 1) % BATCH_SIZE == 0:
            elapsed_min = (i + 1) * DELAY_BETWEEN_STOCKS / 60
            remaining_min = (len(remaining) - i - 1) * DELAY_BETWEEN_STOCKS / 60
            print(f"\n  ─── 進度：{i+1}/{len(remaining)} | "
                  f"成功:{success_count} 失敗:{fail_count} | "
                  f"已用:{elapsed_min:.0f}分 剩餘:{remaining_min:.0f}分 ───\n")

    # ─── 最終統計報告 ───
    print("\n" + "=" * 60)
    print("  下載完成！統計報告")
    print("=" * 60)

    # 分析資料品質
    csv_files = list(OUTPUT_DIR.glob("*.csv"))
    stats = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df["date"] = pd.to_datetime(df["date"])
            null_ratio = df.isnull().mean().mean()
            stats.append({
                "code": f.stem,
                "rows": len(df),
                "date_start": df["date"].min().strftime("%Y-%m-%d"),
                "date_end": df["date"].max().strftime("%Y-%m-%d"),
                "null_pct": round(null_ratio * 100, 2),
                "years": (df["date"].max() - df["date"].min()).days / 365
            })
        except:
            pass

    stats_df = pd.DataFrame(stats)
    if not stats_df.empty:
        stats_df.to_csv(Path("./twse_data/data_quality.csv"), index=False)

        # 分類統計
        full_data    = stats_df[stats_df["years"] >= 10]
        medium_data  = stats_df[(stats_df["years"] >= 5) & (stats_df["years"] < 10)]
        short_data   = stats_df[stats_df["years"] < 5]
        high_missing = stats_df[stats_df["null_pct"] > 5]

        summary = f"""
TWSE 台股資料下載報告
產生時間：{datetime.now().strftime("%Y-%m-%d %H:%M")}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
下載範圍：{START_DATE} ~ {END_DATE}

【下載結果】
  成功下載：{success_count + len(done_set)} 支
  本次新增：{success_count} 支
  失敗跳過：{fail_count} 支
  CSV 檔案數：{len(csv_files)} 個

【資料年限分佈】
  ≥ 10 年完整資料：{len(full_data)} 支  ← 建議優先使用
  5–10 年資料：    {len(medium_data)} 支
  < 5 年資料：     {len(short_data)} 支（新上市居多）

【資料品質】
  缺失率 > 5% 的股票：{len(high_missing)} 支  ← 建議 fine-tune 時剔除
  平均缺失率：{stats_df["null_pct"].mean():.2f}%

【訓練樣本估算（lookback=90, predict=10）】
  ≥ 10 年的股票池：{len(full_data)} 支
  每支股票平均樣本：~{int(stats_df[stats_df['years']>=10]['rows'].mean()) - 100:,} 個
  估計總訓練樣本：~{len(full_data) * (int(stats_df[stats_df['years']>=10]['rows'].mean()) - 100):,} 個

【失敗清單】
{chr(10).join(fail_list[:20]) if fail_list else "  無"}

【後續步驟建議】
1. 執行 filter_data.py 剔除缺損嚴重的個股
2. 執行 convert_to_qlib.py 轉換成 Qlib 格式
3. 開始 Kronos fine-tune 訓練
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        print(summary)
        SUMMARY_FILE.write_text(summary, encoding="utf-8")
        print(f"📄 完整報告已儲存至：{SUMMARY_FILE}")
        print(f"📊 資料品質明細：./twse_data/data_quality.csv")

    print(f"\n📁 資料儲存位置：{OUTPUT_DIR.resolve()}")
    print("🎉 下載完成！")


if __name__ == "__main__":
    main()
