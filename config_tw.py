"""
Kronos Fine-tune Config — 台股版（config_tw.py）
=================================================
放置位置：將此檔案複製到 Kronos/finetune/ 目錄下

使用方式：
    # 在 train_tokenizer.py / train_predictor.py 頂部改為：
    from config_tw import Config

硬體設定：RTX 4080 12GB × 1 GPU（已針對顯存優化）
"""

from pathlib import Path

# ─── 自動偵測路徑 ────────────────────────────────────────────
# 此檔案放在 Kronos/finetune/ 下，所以 ../ 是 Kronos 根目錄
_THIS_DIR  = Path(__file__).parent.resolve()   # Kronos/finetune/
_BASE_DIR  = _THIS_DIR.parent.parent.resolve() # Kronos 的上一層（你的工作資料夾）


class Config:
    # ─── 資料路徑 ─────────────────────────────────────────────
    qlib_data_path       = str(_BASE_DIR / "qlib_data" / "tw_data")
    dataset_path         = str(_BASE_DIR / "kronos_dataset")
    save_path            = str(_BASE_DIR / "checkpoints")
    backtest_result_path = str(_BASE_DIR / "backtest_results")

    # ─── 預訓練模型（從 HuggingFace 自動下載）──────────────────
    pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
    pretrained_predictor_path = "NeoQuasar/Kronos-base"

    # ─── Fine-tuned Tokenizer 路徑（train_predictor.py 需要）──
    # 指向 train_tokenizer.py 訓練完的最佳權重
    finetuned_tokenizer_path  = str(_BASE_DIR / "checkpoints" / "tokenizer" / "checkpoints" / "best_model")

    # ─── 股票池設定 ───────────────────────────────────────────
    # Qlib instruments 名稱，對應 qlib_data/tw_data/instruments/tw_all.txt
    instrument = "tw_all"

    # ─── 資料時間範圍 ─────────────────────────────────────────
    dataset_begin_time = "2010-01-01"
    dataset_end_time   = "2024-12-31"

    # 訓練 / 驗證 / 測試分割（tuple 格式，qlib_data_preprocess.py 直接讀取）
    # 注意：val_start < train_end，重疊部分確保 lookback window 完整
    train_time_range    = ("2010-01-01", "2021-12-31")
    val_time_range      = ("2021-10-01", "2023-06-30")   # 與訓練重疊 3 個月
    test_time_range     = ("2023-04-01", "2024-12-31")   # 與驗證重疊 3 個月
    backtest_time_range = ("2024-07-01", "2024-12-31")

    # ─── 特徵設定 ─────────────────────────────────────────────
    # 注意：Kronos 內部用 vol（=volume）和 amt（=avg_price × volume）
    # vwap 由 preprocess 腳本從 Qlib 讀取後用於計算 amt
    feature_list      = ["open", "high", "low", "close", "vol", "amt"]
    # 模型 module.py 硬編碼讀取順序：[minute, hour, weekday, day, month]（5個固定index）
    # 日K的 minute/hour 值為 0，但必須提供，否則 index out of bounds
    time_feature_list = ["minute", "hour", "weekday", "day", "month"]

    # ─── 視窗設定 ─────────────────────────────────────────────
    lookback_window = 90    # 過去 90 個交易日
    predict_window  = 10    # 預測未來 10 個交易日
    max_context     = 512   # Kronos-base/small 的最大 context length

    # 每個 epoch 使用的樣本數（從全部 ~90 萬個視窗中隨機抽取）
    # 實際值 = min(n_train_iter, 可用總樣本數)
    n_train_iter = 100000   # 訓練集每 epoch 10 萬樣本
    n_val_iter   = 20000    # 驗證集每 epoch 2 萬樣本

    # Z-score 正規化後的截斷值（±5 sigma 以外視為異常值）
    clip = 5.0

    # ─── 模型設定 ─────────────────────────────────────────────
    model_size = "base"     # 對應 NeoQuasar/Kronos-base

    # ─── 訓練超參數（針對 RTX 4080 12GB 單卡優化）────────────────
    #
    # 等效 batch size = batch_size × gradient_accumulation_steps = 8 × 32 = 256
    # batch_size 從 16 降到 8 → GPU 每步負載降低約 30%，減少功耗避免關機
    # accumulation_steps 從 16 升到 32 → 保持等效 batch 256 不變，訓練品質相同
    #
    batch_size         = 8
    accumulation_steps = 32   # train_tokenizer.py 讀這個 key

    # 混合精度（FP16）+ 梯度檢查點 → Kronos-base 可在 12GB 內執行
    use_fp16               = True
    gradient_checkpointing = True

    # 學習率
    tokenizer_learning_rate = 2e-4
    predictor_learning_rate = 4e-5

    # Adam 優化器參數（train script 直接讀這些 key）
    adam_weight_decay = 0.05
    adam_beta1        = 0.9
    adam_beta2        = 0.999

    epochs       = 3
    warmup_steps = 500
    log_interval = 10   # 每 10 步印一次 loss

    # Checkpoint 資料夾名稱
    tokenizer_save_folder_name = "tokenizer"
    predictor_save_folder_name = "predictor"

    # Comet ML 實驗追蹤（關閉，不需要帳號）
    use_comet  = False
    comet_config = {"api_key": "", "project_name": "", "workspace": ""}
    comet_tag  = ""
    comet_name = ""

    # ─── 系統設定 ─────────────────────────────────────────────
    num_workers = 0   # 0 = single-thread loading，避免多進程 + CUDA 在 Windows 導致當機
    seed        = 42
    device      = "cuda:0"

    # ─── 回測設定 ─────────────────────────────────────────────
    topk   = 10
    n_drop = 5

    def __init__(self):
        # 將所有類別屬性複製為實例屬性，讓 __dict__ 能正確回傳所有設定
        for key, val in self.__class__.__dict__.items():
            if not key.startswith("_") and not callable(val):
                setattr(self, key, val)

    def __repr__(self):
        return (
            f"Config(instrument={self.instrument}, "
            f"model={self.model_size}, "
            f"batch={self.batch_size}×{self.accumulation_steps}, "
            f"fp16={self.use_fp16})"
        )
