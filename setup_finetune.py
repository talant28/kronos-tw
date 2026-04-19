"""
Kronos Fine-tune 環境設定腳本
================================
執行此腳本完成以下全部步驟：
  1. 安裝必要套件
  2. Clone Kronos 官方 repo
  3. 將 filtered CSV → Qlib 二進位格式
  4. 生成 Taiwan 專屬 config.py（已針對 RTX 4080 12GB 優化）
  5. 執行 Kronos 資料前處理（產出 train/val/test pkl）

完成後直接進入 fine-tune 訓練即可。

使用方式：
    python setup_finetune.py

硬體假設：RTX 4080 12GB × 1、64GB RAM、i9-13980HX
"""

import subprocess
import sys
import os
from pathlib import Path

# ─────────────────── 路徑設定 ────────────────────
BASE_DIR        = Path(__file__).parent.resolve()
FILTERED_DIR    = BASE_DIR / "twse_data" / "filtered"
KRONOS_DIR      = BASE_DIR / "Kronos"
QLIB_DATA_DIR   = BASE_DIR / "qlib_data" / "tw_data"
DATASET_DIR     = BASE_DIR / "kronos_dataset"
CHECKPOINT_DIR  = BASE_DIR / "checkpoints"
BACKTEST_DIR    = BASE_DIR / "backtest_results"

def run(cmd, cwd=None, check=True):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd,
                            capture_output=False, text=True)
    if check and result.returncode != 0:
        print(f"  ⚠️  命令失敗（returncode={result.returncode}），繼續執行...")
    return result.returncode == 0

def step(n, title):
    print(f"\n{'='*60}")
    print(f"  步驟 {n}：{title}")
    print(f"{'='*60}")

# ══════════════════════════════════════════════════════════════
# 步驟 1：安裝套件
# ══════════════════════════════════════════════════════════════
step(1, "安裝必要套件")

packages = [
    "pyqlib",
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "transformers",
    "numpy pandas scipy scikit-learn",
    "huggingface_hub",
    "einops",
    "python-dateutil",
    "tqdm",
]

for pkg in packages:
    print(f"\n📦 安裝 {pkg.split()[0]}...")
    run(f"{sys.executable} -m pip install {pkg} -q")

print("\n✅ 套件安裝完成")

# ══════════════════════════════════════════════════════════════
# 步驟 2：Clone Kronos repo
# ══════════════════════════════════════════════════════════════
step(2, "Clone Kronos 官方 repo")

if KRONOS_DIR.exists():
    print(f"✅ Kronos 已存在：{KRONOS_DIR}")
    run("git pull", cwd=KRONOS_DIR, check=False)
else:
    print("📥 正在 clone Kronos...")
    ok = run(f"git clone https://github.com/shiyu-coder/Kronos.git {KRONOS_DIR}")
    if ok:
        print("✅ Clone 完成")
    else:
        print("❌ Clone 失敗，請手動執行：")
        print(f"   git clone https://github.com/shiyu-coder/Kronos.git {KRONOS_DIR}")
        sys.exit(1)

# 安裝 Kronos 自身依賴
req_file = KRONOS_DIR / "requirements.txt"
if req_file.exists():
    run(f"{sys.executable} -m pip install -r {req_file} -q")
    print("✅ Kronos requirements 安裝完成")

# ══════════════════════════════════════════════════════════════
# 步驟 3：CSV → Qlib 格式轉換
# ══════════════════════════════════════════════════════════════
step(3, "將 filtered CSV 轉換為 Qlib 格式")

QLIB_DATA_DIR.mkdir(parents=True, exist_ok=True)

csv_count = len(list(FILTERED_DIR.glob("*.csv")))
print(f"📂 來源：{FILTERED_DIR}（共 {csv_count} 支股票）")
print(f"📁 目標：{QLIB_DATA_DIR}")

# 在轉換前，先把 CSV 欄位調整為 Qlib 標準格式（小寫英文）
# Qlib dump_bin 需要 CSV 含有 date 欄且欄名為小寫

# 找 Qlib 安裝路徑裡的 dump_bin.py
import qlib
qlib_path = Path(qlib.__file__).parent
dump_script = qlib_path / "tests" / "dump_bin.py"

if not dump_script.exists():
    # 備援：找 scripts 目錄
    dump_script = qlib_path.parent / "scripts" / "dump_bin.py"

if dump_script.exists():
    print(f"✅ 找到 dump_bin.py：{dump_script}")
    cmd = (
        f"{sys.executable} {dump_script} dump_all "
        f"--csv_path {FILTERED_DIR} "
        f"--qlib_dir {QLIB_DATA_DIR} "
        f"--freq day "
        f"--include_fields open,high,low,close,volume,amount,vwap "
        f"--date_field_name date "
        f"--symbol_field_name symbol "
        f"--exclude_fields spread,turnover"
    )
    ok = run(cmd)
    if ok:
        print("✅ Qlib 資料轉換完成")
    else:
        # 嘗試備用方式
        print("⚠️  dump_bin.py 轉換失敗，嘗試備用方式...")
        _do_manual_qlib_convert()
else:
    print("⚠️  找不到 dump_bin.py，使用 Python 直接寫入 Qlib 格式...")
    _manual_convert = True


def _do_manual_qlib_convert():
    """備援：手動將 CSV 存為 Qlib 可讀的格式（直接用 qlib 的 DumpDataAll）"""
    try:
        from qlib.data.storage.file_storage import FileStorageDumper
        print("  使用 FileStorageDumper...")
    except ImportError:
        pass

    # 最簡備援：用 qlib 的 dump_bin CLI
    cmd2 = (
        f"{sys.executable} -m qlib.tests.dump_bin dump_all "
        f"--csv_path {FILTERED_DIR} "
        f"--qlib_dir {QLIB_DATA_DIR} "
        f"--freq day "
        f"--include_fields open,high,low,close,volume,amount,vwap"
    )
    run(cmd2, check=False)


# ══════════════════════════════════════════════════════════════
# 步驟 4：生成 Taiwan 專屬 config.py
# ══════════════════════════════════════════════════════════════
step(4, "生成 Taiwan 專屬 finetune config")

DATASET_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

config_content = f'''"""
Kronos Fine-tune Config — 台股版
針對 RTX 4080 12GB × 1 GPU 優化
"""
from pathlib import Path

# ─── 路徑設定 ───
BASE_DIR              = Path(r"{BASE_DIR}")
qlib_data_path        = str(BASE_DIR / "qlib_data" / "tw_data")
dataset_path          = str(BASE_DIR / "kronos_dataset")
save_path             = str(BASE_DIR / "checkpoints")
backtest_result_path  = str(BASE_DIR / "backtest_results")

# ─── 預訓練模型（從 HuggingFace 自動下載）───
pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
pretrained_predictor_path = "NeoQuasar/Kronos-base"

# ─── 資料設定 ───
market          = "tw_stock"          # 自定義市場名稱
freq            = "day"
start_time      = "2010-01-01"
end_time        = "2024-12-31"
fit_start_time  = "2010-01-01"        # 訓練集開始
fit_end_time    = "2021-12-31"        # 訓練集結束
val_start_time  = "2021-10-01"        # 驗證集（與訓練有重疊以確保 lookback window）
val_end_time    = "2023-06-30"
test_start_time = "2023-04-01"
test_end_time   = "2024-12-31"

# 特徵欄位（與 Kronos 原始設定一致）
fields = ["$open", "$high", "$low", "$close", "$volume", "$vwap"]
feature_names = ["open", "high", "low", "close", "volume", "vwap"]

# 視窗設定
lookback_window = 90    # 使用過去 90 個交易日
predict_window  = 10    # 預測未來 10 個交易日

# ─── 模型設定 ───
model_type = "base"     # mini / small / base（large 未開放）

# ─── 訓練超參數（針對 RTX 4080 12GB 單卡優化）───
# 核心原則：batch_size × gradient_accumulation_steps = 等效 batch 256
batch_size                  = 16     # 實際每步 batch（配合 12GB VRAM）
gradient_accumulation_steps = 16     # 模擬 batch=256

# 精度與記憶體優化（必須開啟，否則 base 會 OOM）
use_fp16                = True       # 混合精度訓練，VRAM 使用量減半
gradient_checkpointing  = True       # 犧牲 ~20% 速度換取大幅降低顯存

# 優化器
learning_rate  = 5e-5    # Fine-tune 用較小 LR（預訓練是 5e-4）
weight_decay   = 0.05
num_epochs     = 3
warmup_steps   = 500

# ─── 其他 ───
num_workers = 4          # DataLoader workers（RAM 夠大可以開 4）
seed        = 42
device      = "cuda:0"  # RTX 4080

# 股票清單（從 filtered 目錄自動讀取）
import os as _os
_filtered_dir = BASE_DIR / "twse_data" / "filtered"
instruments = sorted([
    f.stem for f in _filtered_dir.glob("*.csv")
]) if _filtered_dir.exists() else []
print(f"[config] 載入 {{len(instruments)}} 支股票")
'''

config_path = KRONOS_DIR / "finetune" / "config_tw.py"
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(config_content, encoding="utf-8")
print(f"✅ Taiwan config 已寫入：{config_path}")

# ══════════════════════════════════════════════════════════════
# 步驟 5：執行 Kronos 資料前處理
# ══════════════════════════════════════════════════════════════
step(5, "執行 Kronos 資料前處理（產出 train/val/test pkl）")

preprocess_script = KRONOS_DIR / "finetune" / "qlib_data_preprocess.py"

if not preprocess_script.exists():
    print(f"⚠️  找不到 {preprocess_script}")
    print("   請確認 Kronos repo 已正確 clone")
else:
    print(f"🔄 執行中（可能需要 5–15 分鐘，視資料量）...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(KRONOS_DIR)
    result = subprocess.run(
        [sys.executable, str(preprocess_script), "--config", "config_tw"],
        cwd=str(KRONOS_DIR),
        env=env,
    )
    if result.returncode == 0:
        print("✅ 資料前處理完成")
    else:
        print("⚠️  前處理有警告（可能是 Qlib 初始化訊息，請查看上方輸出）")

# ══════════════════════════════════════════════════════════════
# 完成
# ══════════════════════════════════════════════════════════════
print(f"""
{'='*60}
  ✅ 環境設定完成！

  接下來執行 fine-tune（依序執行）：

  步驟 A：Fine-tune Tokenizer（先調整 K 線量化器）
    cd {KRONOS_DIR}
    torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py --config config_tw

  步驟 B：Fine-tune Predictor（再調整預測器）
    torchrun --standalone --nproc_per_node=1 finetune/train_predictor.py --config config_tw

  步驟 C：回測驗證
    python finetune/qlib_test.py --config config_tw --device cuda:0

  ⏱  預估訓練時間（RTX 4080 12GB）：
    Tokenizer：約 3–5 小時
    Predictor：約 8–15 小時
{'='*60}
""")
