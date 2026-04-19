# Kronos-TW：台股財務基礎模型微調

> 繁體中文 | [English](README_en.md)

基於 [Kronos](https://github.com/shiyu-coder/Kronos) 開源金融基礎模型，使用台灣證券交易所（TWSE）上市股票歷史日K資料進行微調，目標為台股股價走勢預測。

---

## 訓練結果

| 模型 | 參數量 | 最佳 Val Loss | 最佳 Epoch | 訓練時間 |
|------|--------|--------------|-----------|---------|
| Tokenizer | ~4M (16MB) | — | — | ~3 hr |
| Predictor | ~102M (409MB) | **3.0293** | Epoch 1 | ~20 hr |

- 硬體：ROG G16，RTX 4080 Laptop 12GB
- 訓練資料：TWSE 上市股票 613 支，2010–2024 年日K（共 15 年）
- Predictor best_model = Epoch 1（Epoch 2、3 val loss 較高，輕微 overfitting）

**訓練好的模型已公開於 HuggingFace，可直接下載使用：**
- 🤗 [talant28/Kronos-TW-Tokenizer](https://huggingface.co/talant28/Kronos-TW-Tokenizer)
- 🤗 [talant28/Kronos-TW-Predictor](https://huggingface.co/talant28/Kronos-TW-Predictor)

---

## 專案特色

- **完整資料管線**：從 TWSE/FinMind API 自動下載 → 品質過濾 → Qlib 格式轉換 → 訓練前處理
- **Windows 相容**：修正 Kronos 官方程式碼的 Windows DDP 問題（nccl → gloo + FileStore）
- **斷點續訓**：每個 epoch 結束自動存 checkpoint，重啟後從最後完成的 epoch 繼續
- **低顯存優化**：RTX 4080 Laptop 12GB 可執行，透過 gradient accumulation 維持等效 batch size = 256
- **Loss CSV 記錄**：訓練過程自動寫入 `training_log.csv`，方便事後分析與繪圖

---

## 硬體需求

| 項目 | 最低需求 | 本專案測試環境 |
|------|----------|----------------|
| GPU  | 8GB VRAM | RTX 4080 Laptop 12GB |
| RAM  | 32GB     | 64GB DDR5 |
| CPU  | 8 核心   | i9-13980HX |
| 儲存 | 10GB     | NVMe SSD |

> ⚠️ 筆電用戶注意：GPU + CPU 同時全速運行時總功耗可能超過變壓器上限，導致系統強制斷電。本專案已將 `batch_size` 調低為 8（等效 batch 維持 256），解決此問題。

---

## 快速開始

### 1. 環境安裝

```bash
pip install pyqlib finmind pandas tqdm requests torch torchvision
pip install "numpy<2.0"   # 解決 qlib 與 NumPy 2.x 不相容問題
```

### 2. Clone Kronos 官方 repo

```bash
git clone https://github.com/shiyu-coder/Kronos.git Kronos
```

### 3. 套用本專案修改（Windows 修正 + Resume + Loss Log 功能）

```bash
copy /Y patches\training_utils.py  Kronos\finetune\utils\training_utils.py
copy /Y patches\train_tokenizer.py Kronos\finetune\train_tokenizer.py
copy /Y patches\train_predictor.py Kronos\finetune\train_predictor.py
```

> 或直接執行：`python setup_finetune.py`

---

## 完整訓練流程

```
TWSE 股票資料
    │
    ▼
Step 1：下載資料
    python download_twse_data.py
    # 約 900 支上市股 × 15 年日K
    # 來源：TWSE 官方 API + FinMind（免費帳號 600 req/day）
    # 耗時約 30 分鐘
    │
    ▼
Step 2：品質過濾
    python filter_data.py
    # 剔除資料缺損、掛牌未滿 2 年、殭屍股等
    # 900+ 支 → 613 支通過過濾
    │
    ▼
Step 3：轉換 Qlib 格式
    python convert_to_qlib.py
    # 產出 qlib_data/tw_data/（Qlib 二進位格式）
    # 包含 open, high, low, close, volume, amount 六個特徵
    │
    ▼
Step 4：Kronos 資料前處理
    cd Kronos
    copy /Y ..\config_tw.py finetune\config_tw.py
    copy /Y ..\config_tw.py finetune\config.py
    python finetune\qlib_data_preprocess.py
    # 產出 kronos_dataset/*.pkl
    # 耗時約 15 分鐘
    │
    ▼
Step 5：開始訓練（自動完成後關機）
    launch_windows.bat
    ├── Stage 1: Fine-tune Tokenizer  (~3–5 小時)
    └── Stage 2: Fine-tune Predictor  (~8–15 小時)
    # 訓練 log 自動寫入 checkpoints/{tokenizer,predictor}/training_log.csv
```

---

## 目錄結構

```
kronos-tw/
├── config_tw.py              # 台股專屬訓練設定（RTX 4080 優化）
├── download_twse_data.py     # TWSE/FinMind 資料下載
├── filter_data.py            # 股票品質過濾
├── convert_to_qlib.py        # CSV → Qlib 二進位格式
├── setup_finetune.py         # 一鍵環境設定（複製 patches → Kronos）
├── launch_windows.bat        # Windows 訓練啟動（含自動關機）
├── plot_predictor_loss.py    # 從 training_log.csv 或 console log 繪製 loss 圖
├── upload_to_huggingface.py  # 上傳訓練好的模型到 HuggingFace
├── patches/                  # Kronos 官方程式的修改補丁
│   ├── training_utils.py     # Windows DDP 修正（gloo + FileStore）
│   ├── train_tokenizer.py    # Resume + per-epoch checkpoint + CSV log
│   └── train_predictor.py    # Resume + per-epoch checkpoint + CSV log
│
├── Kronos/                   # (git clone 取得，不納入版控)
├── twse_data/                # (本地生成，不納入版控)
├── qlib_data/                # (本地生成，不納入版控)
├── kronos_dataset/           # (本地生成，不納入版控)
└── checkpoints/              # (訓練產出，上傳 HuggingFace)
    ├── tokenizer/
    │   ├── training_log.csv  # 每 step train loss + 每 epoch val loss
    │   └── checkpoints/best_model/
    └── predictor/
        ├── training_log.csv
        └── checkpoints/
            ├── epoch_1/ epoch_2/ epoch_3/
            └── best_model/   ← Epoch 1（val loss 3.0293）
```

---

## 主要設定（config_tw.py）

| 參數 | 值 | 說明 |
|------|----|------|
| `instrument` | `tw_all` | 台股全部上市股票 |
| `dataset_begin_time` | `2010-01-01` | 資料起始 |
| `dataset_end_time` | `2024-12-31` | 資料截止 |
| `train_time_range` | `2010–2021` | 訓練集 |
| `val_time_range` | `2021–2023` | 驗證集 |
| `test_time_range` | `2023–2024` | 測試集 |
| `lookback_window` | `90` | 回看過去 90 個交易日 |
| `predict_window` | `10` | 預測未來 10 個交易日 |
| `batch_size` | `8` | 每步批次（顯存 + 功耗優化） |
| `accumulation_steps` | `32` | 梯度累積（等效 batch=256） |
| `epochs` | `3` | 訓練回合數 |

---

## Windows 相容性修正

Kronos 官方程式碼在 Windows 有以下問題，本專案均已修正：

| 問題 | 原因 | 修正方式 |
|------|------|----------|
| `torchrun` 啟動失敗 | PyTorch 2.5.1 未編譯 libuv | 改用環境變數手動設定 DDP |
| `dist.init_process_group` 崩潰 | Windows 不支援 nccl backend | 改用 gloo + FileStore |
| `KeyError: 'seed'` | Config class 屬性不進 `__dict__` | 加入 `__init__` 複製屬性 |
| DataLoader CUDA abort | `pin_memory=True` + multiprocessing | 改為 `pin_memory=False` |
| `KeyError: 'finetuned_tokenizer_path'` | Config 缺少 Predictor 所需欄位 | 在 config_tw.py 補上路徑 |

---

## 斷點續訓機制

訓練中斷（當機 / 手動停止）後，重新執行 `launch_windows.bat` 會自動：

1. **Tokenizer**：偵測 `summary.json` 中的 `completed_epochs`，從上次完成的 epoch 繼續
2. **Predictor**：同上，並從對應的 `epoch_N/` checkpoint 載入權重，快轉 LR scheduler

---

## Loss 圖表繪製

訓練完成後，`training_log.csv` 會自動存在各模型的 checkpoints 資料夾中：

```bash
python plot_predictor_loss.py
# 自動讀取 checkpoints/predictor/training_log.csv
# 輸出 predictor_loss_comparison.png
```

CSV 格式：

```
type,epoch,step,loss,lr
train,1,100,2.4512,0.000040
train,1,200,2.3891,0.000041
val,1,-1,3.0293,
...
```

---

## 訓練完成後：上傳 HuggingFace

```bash
pip install huggingface_hub
huggingface-cli login

python upload_to_huggingface.py --username YOUR_HF_USERNAME
```

這會自動建立兩個 repo 並上傳：

- `YOUR_USERNAME/Kronos-TW-Tokenizer`
- `YOUR_USERNAME/Kronos-TW-Predictor`

---

## 回測

```bash
cd Kronos
python finetune\qlib_test.py --device cuda:0
```

結果存放於 `backtest_results/`，並自動生成回測績效圖。

---

## 資料來源

- **TWSE**：台灣證券交易所官方 API（免費，無需帳號）
- **FinMind**：台灣金融開放資料平台（免費帳號 600 requests/day）
- **基礎模型**：[NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)

---

## 相關連結

- 原始論文：[Kronos: A Foundation Model for Time Series](https://arxiv.org/abs/2504.03249)
- Kronos GitHub：[shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
- Kronos HuggingFace：[NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)

---

## 引用聲明

本專案基於 Kronos 進行微調開發。Kronos 為 NeoQuasar 團隊開發的金融時間序列基礎模型，原始論文如下：

```
Shi, Yu, et al. "Kronos: A Foundation Model for Time Series."
arXiv preprint arXiv:2504.03249 (2025).
https://arxiv.org/abs/2504.03249
```

若引用本專案或衍生成果，請同時引用原始論文。

---

## License

本專案程式碼採 MIT License。訓練資料版權屬 TWSE/FinMind，模型版權屬 NeoQuasar，請遵守各方授權條款。
