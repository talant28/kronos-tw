"""
將 filtered CSV 轉換為 Qlib 二進位格式
=======================================
Qlib 需要自己的二進位資料格式（.bin 檔）才能高效讀取。
此腳本把 twse_data/filtered/ 下的 CSV 全部轉換過去。

使用方式：
    pip install pyqlib
    python convert_to_qlib.py

輸出：
    ./qlib_data/tw_data/     Qlib 標準資料目錄
      ├── calendars/day.txt  交易日曆
      ├── instruments/       股票清單
      │   └── tw_all.txt
      └── features/          每支股票的 .bin 特徵檔
          ├── 2330/
          │   ├── open.day.bin
          │   ├── high.day.bin
          │   └── ...
          └── ...
"""

import struct
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ─────────────── 路徑設定 ───────────────
BASE_DIR     = Path(__file__).parent.resolve()
FILTERED_DIR = BASE_DIR / "twse_data" / "filtered"
QLIB_DIR     = BASE_DIR / "qlib_data" / "tw_data"

FIELDS = ["open", "high", "low", "close", "volume", "vwap", "amount"]


def write_bin(data: np.ndarray, path: Path):
    """寫入 Qlib 的 .bin 格式（float32 陣列，前綴為起始日期 index）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        # Qlib bin 格式：第一個 float32 是起始 index（此處不用，寫 0）
        arr = data.astype(np.float32)
        f.write(struct.pack("<f", 0.0))  # placeholder
        f.write(arr.tobytes())


def get_all_trading_days(filtered_dir: Path) -> list:
    """從所有 CSV 收集完整交易日曆"""
    all_dates = set()
    for f in filtered_dir.glob("*.csv"):
        df = pd.read_csv(f, usecols=["date"])
        dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").tolist()
        all_dates.update(dates)
    return sorted(all_dates)


def convert_stock(code: str, df: pd.DataFrame, date_to_idx: dict, total_days: int):
    """轉換單支股票的所有欄位為 .bin 檔"""
    feat_dir = QLIB_DIR / "features" / code.lower()
    feat_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")

    for field in FIELDS:
        if field not in df.columns:
            continue

        # 建立對齊到完整日曆的陣列（缺的日期填 NaN）
        arr = np.full(total_days, np.nan, dtype=np.float32)
        for date_str, val in df[field].items():
            if date_str in date_to_idx:
                idx = date_to_idx[date_str]
                try:
                    arr[idx] = float(val)
                except (ValueError, TypeError):
                    arr[idx] = np.nan

        # 找第一個有效資料的位置（Qlib 需要起始 offset）
        valid_mask = ~np.isnan(arr)
        if not valid_mask.any():
            continue
        start_idx = int(np.argmax(valid_mask))

        # 只寫入從第一個有效資料開始的部分
        data_slice = arr[start_idx:]

        bin_path = feat_dir / f"{field}.day.bin"
        with open(bin_path, "wb") as f:
            # Qlib 格式：先寫 start_index（uint32），再寫 float32 資料
            f.write(struct.pack("<I", start_idx))
            f.write(data_slice.astype(np.float32).tobytes())


def main():
    print("=" * 60)
    print("  CSV → Qlib 格式轉換")
    print("=" * 60)

    csv_files = sorted(FILTERED_DIR.glob("*.csv"))
    if not csv_files:
        print(f"❌ 找不到資料：{FILTERED_DIR}")
        return

    print(f"📂 來源：{FILTERED_DIR}（{len(csv_files)} 支股票）")
    print(f"📁 目標：{QLIB_DIR}")

    # ── 1. 建立完整交易日曆 ──
    print("\n🗓️  建立交易日曆...")
    trading_days = get_all_trading_days(FILTERED_DIR)
    date_to_idx  = {d: i for i, d in enumerate(trading_days)}
    total_days   = len(trading_days)
    print(f"  共 {total_days} 個交易日（{trading_days[0]} ~ {trading_days[-1]}）")

    # 寫入 calendars/day.txt
    cal_dir = QLIB_DIR / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / "day.txt").write_text("\n".join(trading_days) + "\n")
    print("  ✅ calendars/day.txt 寫入完成")

    # ── 2. 轉換每支股票 ──
    print("\n🔄 轉換股票資料...")
    stock_list = []

    for i, f in enumerate(csv_files):
        code = f.stem
        try:
            df = pd.read_csv(f)
            convert_stock(code, df, date_to_idx, total_days)

            # 記錄有效日期範圍（用於 instruments 檔）
            df["date"] = pd.to_datetime(df["date"])
            start = df["date"].min().strftime("%Y-%m-%d")
            end   = df["date"].max().strftime("%Y-%m-%d")
            stock_list.append((code, start, end))

            if (i + 1) % 50 == 0 or (i + 1) == len(csv_files):
                print(f"  進度：{i+1}/{len(csv_files)}")

        except Exception as e:
            print(f"  ⚠️  {code} 失敗：{e}")

    # ── 3. 建立 instruments 清單 ──
    inst_dir = QLIB_DIR / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)

    # Qlib instruments 格式：symbol\tstart_date\tend_date
    lines = [f"{code.upper()}\t{start}\t{end}" for code, start, end in stock_list]
    (inst_dir / "tw_all.txt").write_text("\n".join(lines) + "\n")

    # 也建立一個 all.txt（部分 Qlib 版本需要）
    (inst_dir / "all.txt").write_text("\n".join(lines) + "\n")

    print(f"\n  ✅ instruments/tw_all.txt（{len(stock_list)} 支）寫入完成")

    print(f"""
{'='*60}
✅ Qlib 格式轉換完成！

  交易日數：{total_days}
  股票數量：{len(stock_list)}
  資料目錄：{QLIB_DIR}

  目錄結構：
    {QLIB_DIR}/
    ├── calendars/day.txt
    ├── instruments/tw_all.txt
    └── features/<stock_code>/
        ├── open.day.bin
        ├── high.day.bin
        ├── low.day.bin
        ├── close.day.bin
        ├── volume.day.bin
        ├── vwap.day.bin
        └── amount.day.bin

  下一步：執行 setup_finetune.py 完成 Kronos 訓練環境設定
{'='*60}
""")


if __name__ == "__main__":
    main()
