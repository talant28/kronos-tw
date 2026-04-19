"""
台股資料品質過濾腳本
====================
在 download_twse_data.py 執行完後使用此腳本
自動剔除資料缺損嚴重的個股，輸出適合 Kronos fine-tune 的乾淨資料集

使用方式：
    python filter_data.py

輸出：
    ./twse_data/filtered/   過濾後的乾淨 CSV 檔案
    ./twse_data/filter_report.txt  過濾報告
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ─────────────────────── 過濾條件設定 ───────────────────────
RAW_DIR       = Path("./twse_data/raw")
FILTERED_DIR  = Path("./twse_data/filtered")
REPORT_FILE   = Path("./twse_data/filter_report.txt")

MIN_YEARS       = 8      # 至少需要 8 年資料（2,000 個交易日以上）
MAX_NULL_PCT    = 3.0    # 任一欄位缺失率不能超過 3%
MAX_GAP_DAYS    = 30     # 最長連續停牌不能超過 30 個交易日
MIN_AVG_VOLUME  = 200   # 平均日成交量至少 1,000 張（避免殭屍股）
MIN_PRICE       = 1.0    # 收盤價不能長期低於 1 元（排除風險股）

# Kronos 訓練視窗設定
LOOKBACK   = 90
PRED_LEN   = 10
MIN_ROWS   = LOOKBACK + PRED_LEN + 50  # 至少要能切出 50 個以上的訓練樣本

# ─────────────────────── 主要函數 ───────────────────────

def check_stock(code: str, df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    檢查一支股票是否符合訓練資料品質標準
    回傳 (通過, [不通過原因清單])
    """
    reasons = []

    # 1. 資料年限
    df["date"] = pd.to_datetime(df["date"])
    years = (df["date"].max() - df["date"].min()).days / 365
    if years < MIN_YEARS:
        reasons.append(f"資料年限不足（{years:.1f} 年 < {MIN_YEARS} 年）")

    # 2. 最小資料筆數
    if len(df) < MIN_ROWS:
        reasons.append(f"資料筆數不足（{len(df)} < {MIN_ROWS}）")

    # 3. 缺失率檢查（逐欄位）
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            reasons.append(f"缺少必要欄位：{col}")
            continue
        null_pct = df[col].isnull().mean() * 100
        if null_pct > MAX_NULL_PCT:
            reasons.append(f"{col} 缺失率過高（{null_pct:.1f}% > {MAX_NULL_PCT}%）")

    # 4. 連續停牌偵測（以日期間隔判斷）
    df_sorted = df.sort_values("date")
    date_diffs = df_sorted["date"].diff().dt.days.dropna()
    max_gap = date_diffs.max()
    if max_gap > MAX_GAP_DAYS:
        reasons.append(f"有長期停牌（最長間隔 {max_gap:.0f} 天）")

    # 5. 成交量檢查（殭屍股）
    if "volume" in df.columns:
        # volume 單位為「股」，台股一張 = 1000 股
        avg_vol_lots = df["volume"].median() / 1000
        if avg_vol_lots < MIN_AVG_VOLUME:
            reasons.append(f"成交量過低（中位數 {avg_vol_lots:.0f} 張 < {MIN_AVG_VOLUME} 張）")

    # 6. 低價股/問題股
    if "close" in df.columns:
        low_price_pct = (df["close"] < MIN_PRICE).mean() * 100
        if low_price_pct > 20:
            reasons.append(f"長期低於 {MIN_PRICE} 元（占 {low_price_pct:.1f}%）")

    # 7. 異常值檢查（高低價倒置）
    if "high" in df.columns and "low" in df.columns:
        anomaly = (df["high"] < df["low"]).sum()
        if anomaly > 0:
            reasons.append(f"高低價異常（{anomaly} 筆高價 < 低價）")

    return len(reasons) == 0, reasons


def estimate_training_samples(df: pd.DataFrame) -> int:
    """估算這支股票能產生的訓練樣本數"""
    return max(0, len(df) - LOOKBACK - PRED_LEN)


def main():
    print("=" * 60)
    print("  台股資料品質過濾")
    print("=" * 60)

    FILTERED_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(f"❌ 找不到資料：{RAW_DIR}，請先執行 download_twse_data.py")
        return

    print(f"📂 原始資料：{len(csv_files)} 支股票")
    print(f"📋 過濾標準：年限≥{MIN_YEARS}年 | 缺失率≤{MAX_NULL_PCT}% | 成交量≥{MIN_AVG_VOLUME}張\n")

    passed = []
    rejected = []

    for i, f in enumerate(csv_files):
        code = f.stem
        try:
            df = pd.read_csv(f)
            ok, reasons = check_stock(code, df)
            samples = estimate_training_samples(df)

            if ok:
                # 儲存過濾後的資料
                out_path = FILTERED_DIR / f"{code}.csv"
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
                passed.append({
                    "code": code,
                    "rows": len(df),
                    "samples": samples,
                    "years": round((pd.to_datetime(df["date"]).max() -
                                    pd.to_datetime(df["date"]).min()).days / 365, 1)
                })
                if (i + 1) % 100 == 0:
                    print(f"  進度：{i+1}/{len(csv_files)}，已通過 {len(passed)} 支...")
            else:
                rejected.append({"code": code, "reasons": "; ".join(reasons)})

        except Exception as e:
            rejected.append({"code": code, "reasons": f"讀取錯誤: {e}"})

    # ─── 統計報告 ───
    passed_df   = pd.DataFrame(passed)
    rejected_df = pd.DataFrame(rejected)

    total_samples = passed_df["samples"].sum() if not passed_df.empty else 0

    print("\n" + "=" * 60)
    print("  過濾結果統計")
    print("=" * 60)

    report = f"""
台股資料品質過濾報告
產生時間：{datetime.now().strftime("%Y-%m-%d %H:%M")}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【過濾標準】
  最少資料年限：{MIN_YEARS} 年
  最大缺失率：  {MAX_NULL_PCT}%
  最低成交量：  {MIN_AVG_VOLUME} 張/日（中位數）
  最長停牌間隔：{MAX_GAP_DAYS} 天

【過濾結果】
  原始股票數：  {len(csv_files)} 支
  通過過濾：    {len(passed)} 支  ✅
  被剔除：      {len(rejected)} 支  ❌
  通過率：      {len(passed)/len(csv_files)*100:.1f}%

【通過股票統計】"""

    if not passed_df.empty:
        report += f"""
  資料筆數範圍：{passed_df['rows'].min():,} ~ {passed_df['rows'].max():,} 筆
  資料年限範圍：{passed_df['years'].min():.1f} ~ {passed_df['years'].max():.1f} 年
  年限 ≥ 10 年：{(passed_df['years'] >= 10).sum()} 支
  年限 8–10 年：{((passed_df['years'] >= 8) & (passed_df['years'] < 10)).sum()} 支

【Kronos 訓練樣本估算（lookback={LOOKBACK}, predict={PRED_LEN}）】
  總訓練樣本數：約 {total_samples:,} 個
  平均每股樣本：{total_samples // len(passed):,} 個

  對照官方 CSI300 規模（~600,000 樣本）：
  {"✅ 超過官方規模，資料充足！" if total_samples > 600000 else
   "🟡 接近官方規模" if total_samples > 300000 else
   "⚠️  低於官方規模，建議放寬過濾條件或補充資料"}
"""

    report += f"""
【被剔除股票前 20 名原因】
"""
    if not rejected_df.empty:
        for _, row in rejected_df.head(20).iterrows():
            report += f"  {row['code']}: {row['reasons']}\n"

    report += """
【後續步驟】
1. 確認過濾後股票數是否符合預期
2. 如通過數量不足，可調整 filter_data.py 中的 MIN_YEARS 或 MAX_NULL_PCT
3. 執行 convert_to_qlib.py 將過濾後的資料轉換為 Qlib 格式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    print(report)
    REPORT_FILE.write_text(report, encoding="utf-8")

    if not passed_df.empty:
        passed_df.to_csv(Path("./twse_data/passed_stocks.csv"), index=False)
        print(f"✅ 通過股票清單：./twse_data/passed_stocks.csv")
    if not rejected_df.empty:
        rejected_df.to_csv(Path("./twse_data/rejected_stocks.csv"), index=False)
        print(f"❌ 被剔除股票清單：./twse_data/rejected_stocks.csv")

    print(f"📄 過濾報告：{REPORT_FILE}")
    print(f"📁 乾淨資料：{FILTERED_DIR.resolve()}")
    print("\n🎉 過濾完成！")


if __name__ == "__main__":
    main()
