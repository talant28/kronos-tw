"""
Kronos-TW 台股預測推論腳本
============================
使用方式：
    python inference.py
    # 或直接指定股票代碼：
    python inference.py --stock 2330

必要套件：
    pip install finmind torch pandas numpy matplotlib
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

# ── 路徑設定：從 kronos-tw 根目錄執行 ──────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
KRONOS_DIR = BASE_DIR / "Kronos"

if not KRONOS_DIR.exists():
    print("❌ 找不到 Kronos/ 資料夾，請先執行：")
    print("   git clone https://github.com/shiyu-coder/Kronos.git Kronos")
    sys.exit(1)

sys.path.insert(0, str(KRONOS_DIR))
sys.path.insert(0, str(KRONOS_DIR / "finetune"))

from model.kronos import Kronos, KronosTokenizer, KronosPredictor

# ── 模型路徑 ────────────────────────────────────────────────────
TOKENIZER_PATH = BASE_DIR / "checkpoints" / "tokenizer" / "checkpoints" / "best_model"
PREDICTOR_PATH = BASE_DIR / "checkpoints" / "predictor" / "checkpoints" / "best_model"

# ── 推論設定 ────────────────────────────────────────────────────
LOOKBACK_DAYS = 90   # 輸入過去 90 個交易日
PRED_DAYS     = 5    # 預測未來 5 個交易日
SAMPLE_COUNT  = 10   # 自回歸抽樣次數（越多越穩，但越慢）
TOP_P         = 0.9
TEMPERATURE   = 1.0


# ════════════════════════════════════════════════════════════════
# 1. 資料抓取
# ════════════════════════════════════════════════════════════════

def fetch_stock_data(stock_id: str, lookback: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    從 FinMind 抓取台股日K資料，回傳最近 lookback 個交易日的 DataFrame。
    columns: open, high, low, close, volume, amount
    index:   datetime (pd.DatetimeIndex)
    """
    try:
        from FinMind.data import DataLoader as FinMindLoader
    except ImportError:
        print("❌ 缺少 FinMind 套件，請執行：pip install finmind")
        sys.exit(1)

    # 多抓一些日曆天，確保取得足夠交易日
    start_date = (datetime.today() - timedelta(days=int(lookback * 1.8))).strftime("%Y-%m-%d")
    end_date   = datetime.today().strftime("%Y-%m-%d")

    print(f"   📡 從 FinMind 抓取 {stock_id} 股價資料（{start_date} ~ {end_date}）...")

    fl = FinMindLoader()
    raw = fl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)

    if raw is None or len(raw) == 0:
        print(f"❌ 找不到股票代碼 {stock_id} 的資料，請確認代碼是否正確。")
        sys.exit(1)

    # FinMind 欄位對應
    raw = raw.rename(columns={
        "max":             "high",
        "min":             "low",
        "Trading_Volume":  "volume",
        "Trading_money":   "amount",
    })

    raw["datetime"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("datetime").set_index("datetime")
    raw = raw[["open", "high", "low", "close", "volume", "amount"]].astype(float)

    # 取最近 lookback 個交易日
    if len(raw) < lookback:
        print(f"⚠️  只找到 {len(raw)} 個交易日（需要 {lookback}），將以現有資料繼續。")
    else:
        raw = raw.iloc[-lookback:]

    return raw


# ════════════════════════════════════════════════════════════════
# 2. 未來交易日推算
# ════════════════════════════════════════════════════════════════

def next_trading_days(last_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """
    推算 last_date 之後的 n 個台股交易日（排除週六日）。
    注意：無法自動排除國定假日，若預測區間有假日，日期可能需要手動調整。
    """
    days = []
    current = last_date + timedelta(days=1)
    while len(days) < n:
        if current.weekday() < 5:   # 0=Mon … 4=Fri
            days.append(current)
        current += timedelta(days=1)
    return pd.DatetimeIndex(days)


# ════════════════════════════════════════════════════════════════
# 3. 模型載入
# ════════════════════════════════════════════════════════════════

def load_predictor(device: str) -> KronosPredictor:
    if not TOKENIZER_PATH.exists():
        print(f"❌ 找不到 Tokenizer checkpoint：{TOKENIZER_PATH}")
        print("   請先完成訓練或執行 upload_to_huggingface.py 確認模型存在。")
        sys.exit(1)
    if not PREDICTOR_PATH.exists():
        print(f"❌ 找不到 Predictor checkpoint：{PREDICTOR_PATH}")
        sys.exit(1)

    print(f"   🤖 載入 Tokenizer from {TOKENIZER_PATH} ...")
    tokenizer = KronosTokenizer.from_pretrained(str(TOKENIZER_PATH)).eval()

    print(f"   🤖 載入 Predictor from {PREDICTOR_PATH} ...")
    model = Kronos.from_pretrained(str(PREDICTOR_PATH)).eval()

    return KronosPredictor(model=model, tokenizer=tokenizer, device=device, max_context=512)


# ════════════════════════════════════════════════════════════════
# 4. 結果輸出
# ════════════════════════════════════════════════════════════════

def print_results(stock_id: str, hist_df: pd.DataFrame, pred_df: pd.DataFrame):
    last_close = hist_df["close"].iloc[-1]
    print()
    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  Kronos-TW 預測結果  ·  股票代碼：{stock_id:<6}             ║")
    print(f"╠══════════════════════════════════════════════════════╣")
    print(f"║  最後收盤價（{hist_df.index[-1].date()}）：{last_close:>10.2f} 元       ║")
    print(f"╠══════╦════════╦════════╦════════╦════════╦══════════╣")
    print(f"║  日期  ║  開盤  ║  最高  ║  最低  ║  收盤  ║  漲跌幅  ║")
    print(f"╠══════╬════════╬════════╬════════╬════════╬══════════╣")

    prev_close = last_close
    for date, row in pred_df.iterrows():
        chg = (row["close"] - prev_close) / prev_close * 100
        sign = "▲" if chg >= 0 else "▼"
        print(f"║ {str(date.date())} ║ {row['open']:>6.1f} ║ {row['high']:>6.1f} "
              f"║ {row['low']:>6.1f} ║ {row['close']:>6.1f} ║ {sign}{abs(chg):>5.2f}%  ║")
        prev_close = row["close"]

    print(f"╚══════╩════════╩════════╩════════╩════════╩══════════╝")
    print()
    print("⚠️  本預測僅供研究參考，不構成投資建議。")
    print()


def plot_results(stock_id: str, hist_df: pd.DataFrame, pred_df: pd.DataFrame):
    """繪製歷史收盤價 + 預測區間圖"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        # 設定中文字體（Windows 微軟正黑體，備選微軟雅黑）
        matplotlib.rcParams["font.family"] = ["Microsoft JhengHei", "Microsoft YaHei", "sans-serif"]
        matplotlib.rcParams["axes.unicode_minus"] = False

        # 只顯示最近 30 個交易日的歷史
        hist_plot = hist_df.iloc[-30:]

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        # 歷史收盤
        ax.plot(hist_plot.index, hist_plot["close"],
                color="#00e5ff", lw=1.8, label="歷史收盤價")

        # 連接線（最後歷史點 → 第一個預測點）
        connect_dates  = [hist_plot.index[-1], pred_df.index[0]]
        connect_prices = [hist_plot["close"].iloc[-1], pred_df["close"].iloc[0]]
        ax.plot(connect_dates, connect_prices, color="#888", lw=1.2, linestyle="--")

        # 預測 close
        ax.plot(pred_df.index, pred_df["close"],
                color="#ffd700", lw=2.2, marker="o", markersize=5, label="預測收盤價")

        # 預測 high/low 填色區域
        ax.fill_between(pred_df.index, pred_df["low"], pred_df["high"],
                        color="#ffd700", alpha=0.15, label="預測最高/最低區間")

        # 格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.tick_params(colors="#ccc", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#444")
        ax.grid(True, color="#2a2a4a", lw=0.6)
        ax.set_xlabel("日期", color="#aaa", fontsize=10)
        ax.set_ylabel("股價（元）", color="#aaa", fontsize=10)
        ax.set_title(f"Kronos-TW 股票預測 · {stock_id}", color="white", fontsize=13, fontweight="bold")
        ax.legend(facecolor="#0f3460", edgecolor="#555", labelcolor="white", fontsize=9)

        plt.tight_layout()
        out = BASE_DIR / f"prediction_{stock_id}.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"   📊 圖表已存至：{out}")

    except Exception as e:
        print(f"   ⚠️  圖表繪製失敗（{e}），跳過。")


# ════════════════════════════════════════════════════════════════
# 5. 主程式
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Kronos-TW 台股預測推論")
    parser.add_argument("--stock",   type=str, default=None,    help="台股代碼（例如 2330）")
    parser.add_argument("--days",    type=int, default=PRED_DAYS,    help="預測天數（預設 5）")
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT, help="抽樣次數（越多越穩，預設 10）")
    parser.add_argument("--device",  type=str, default=None,    help="運算裝置（cuda:0 / cpu，預設自動偵測）")
    parser.add_argument("--no-plot", action="store_true",       help="不輸出圖表")
    args = parser.parse_args()

    # ── 股票代碼 ───────────────────────────────────────────────
    stock_id = args.stock
    if not stock_id:
        stock_id = input("請輸入台股代碼（例如 2330）：").strip()
    if not stock_id:
        print("❌ 未輸入股票代碼，結束。")
        sys.exit(1)

    # ── 裝置偵測 ───────────────────────────────────────────────
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    print()
    print(f"🚀 Kronos-TW 推論啟動")
    print(f"   股票代碼 : {stock_id}")
    print(f"   預測天數 : {args.days}")
    print(f"   抽樣次數 : {args.samples}")
    print(f"   運算裝置 : {device}")
    print()

    # ── Step 1：抓取歷史資料 ───────────────────────────────────
    print("【Step 1/3】抓取歷史股價...")
    hist_df = fetch_stock_data(stock_id, lookback=LOOKBACK_DAYS)
    print(f"   ✅ 取得 {len(hist_df)} 個交易日資料（{hist_df.index[0].date()} ~ {hist_df.index[-1].date()}）")

    # ── Step 2：載入模型 ───────────────────────────────────────
    print("\n【Step 2/3】載入模型...")
    predictor = load_predictor(device)
    print("   ✅ 模型載入完成")

    # ── Step 3：推論 ───────────────────────────────────────────
    print(f"\n【Step 3/3】預測未來 {args.days} 個交易日...")

    x_timestamp = hist_df.index.to_series().reset_index(drop=True)
    y_dates     = next_trading_days(hist_df.index[-1], args.days)
    y_timestamp = pd.Series(pd.DatetimeIndex(y_dates))

    pred_df = predictor.predict(
        df          = hist_df,
        x_timestamp = x_timestamp,
        y_timestamp = y_timestamp,
        pred_len    = args.days,
        T           = TEMPERATURE,
        top_p       = TOP_P,
        sample_count= args.samples,
        verbose     = True,
    )

    # 只保留 open/high/low/close 四欄顯示
    pred_df = pred_df[["open", "high", "low", "close"]]

    print("   ✅ 推論完成\n")

    # ── 輸出結果 ───────────────────────────────────────────────
    print_results(stock_id, hist_df, pred_df)

    if not args.no_plot:
        plot_results(stock_id, hist_df, pred_df)


if __name__ == "__main__":
    main()
