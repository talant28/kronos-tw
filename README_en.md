# Kronos-TW: Fine-tuning Kronos on Taiwan Stock Exchange Data

> 🇹🇼 [繁體中文](README.md) | English

Fine-tuning the [Kronos](https://github.com/shiyu-coder/Kronos) financial foundation model on Taiwan Stock Exchange (TWSE) historical daily OHLCV data for stock price prediction.

---

## Training Results

| Model | Parameters | Best Val Loss | Best Epoch | Training Time |
|-------|-----------|--------------|-----------|---------------|
| Tokenizer | ~4M (16MB) | — | — | ~3 hr |
| Predictor | ~102M (409MB) | **3.0293** | Epoch 1 | ~20 hr |

- Hardware: ROG G16, RTX 4080 Laptop 12GB
- Training data: 613 TWSE-listed stocks, daily OHLCV, 2010–2024 (15 years)
- Predictor best_model = Epoch 1 (Epochs 2 & 3 showed slightly higher val loss, mild overfitting)

**Pre-trained models available on HuggingFace:**
- 🤗 [talant28/Kronos-TW-Tokenizer](https://huggingface.co/talant28/Kronos-TW-Tokenizer)
- 🤗 [talant28/Kronos-TW-Predictor](https://huggingface.co/talant28/Kronos-TW-Predictor)

---

## Features

- **Full data pipeline**: Auto-download from TWSE/FinMind API → quality filtering → Qlib format conversion → preprocessing
- **Windows compatibility**: Fixes for Kronos official code's Windows DDP issues (nccl → gloo + FileStore)
- **Resume from checkpoint**: Per-epoch checkpoints saved automatically; training resumes from last completed epoch after interruption
- **Low VRAM optimization**: Runs on RTX 4080 Laptop 12GB via gradient accumulation maintaining effective batch size of 256
- **Loss CSV logging**: Training loss automatically written to `training_log.csv` for analysis and visualization

---

## Hardware Requirements

| Component | Minimum | Test Environment |
|-----------|---------|-----------------|
| GPU | 8GB VRAM | RTX 4080 Laptop 12GB |
| RAM | 32GB | 64GB DDR5 |
| CPU | 8 cores | i9-13980HX |
| Storage | 10GB | NVMe SSD |

> ⚠️ Laptop users: Total power draw (GPU + CPU at full load) may exceed adapter capacity, causing forced shutdown. This project reduces `batch_size` to 8 (maintaining effective batch of 256) to address this issue.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install pyqlib finmind pandas tqdm requests torch torchvision
pip install "numpy<2.0"   # Required: resolves qlib/NumPy 2.x incompatibility
```

### 2. Clone the Official Kronos Repo

```bash
git clone https://github.com/shiyu-coder/Kronos.git Kronos
```

### 3. Apply This Project's Patches (Windows fixes + Resume + Loss Logging)

```bash
copy /Y patches\training_utils.py  Kronos\finetune\utils\training_utils.py
copy /Y patches\train_tokenizer.py Kronos\finetune\train_tokenizer.py
copy /Y patches\train_predictor.py Kronos\finetune\train_predictor.py
```

> Or run: `python setup_finetune.py`

---

## Full Training Pipeline

```
TWSE Stock Data
    │
    ▼
Step 1: Download data
    python download_twse_data.py
    # ~900 listed stocks × 15 years of daily OHLCV
    # Sources: TWSE official API + FinMind (free tier: 600 req/day)
    # Takes ~30 minutes
    │
    ▼
Step 2: Quality filtering
    python filter_data.py
    # Remove stocks with missing data, listed < 2 years, zombie stocks, etc.
    # 900+ → 613 stocks pass filtering
    │
    ▼
Step 3: Convert to Qlib format
    python convert_to_qlib.py
    # Outputs qlib_data/tw_data/ (Qlib binary format)
    # Features: open, high, low, close, volume, amount
    │
    ▼
Step 4: Kronos data preprocessing
    cd Kronos
    copy /Y ..\config_tw.py finetune\config_tw.py
    copy /Y ..\config_tw.py finetune\config.py
    python finetune\qlib_data_preprocess.py
    # Outputs kronos_dataset/*.pkl
    # Takes ~15 minutes
    │
    ▼
Step 5: Training (auto-shutdown on completion)
    launch_windows.bat
    ├── Stage 1: Fine-tune Tokenizer  (~3–5 hours)
    └── Stage 2: Fine-tune Predictor  (~8–15 hours)
    # Loss logs written to checkpoints/{tokenizer,predictor}/training_log.csv
```

---

## Project Structure

```
kronos-tw/
├── config_tw.py              # TWSE-specific training config (RTX 4080 optimized)
├── download_twse_data.py     # TWSE/FinMind data downloader
├── filter_data.py            # Stock quality filtering
├── convert_to_qlib.py        # CSV → Qlib binary format
├── setup_finetune.py         # One-click setup (copies patches → Kronos)
├── launch_windows.bat        # Windows training launcher (with auto-shutdown)
├── plot_predictor_loss.py    # Loss visualization from training_log.csv
├── upload_to_huggingface.py  # Upload trained models to HuggingFace
├── patches/                  # Modified Kronos source files
│   ├── training_utils.py     # Windows DDP fix (gloo + FileStore)
│   ├── train_tokenizer.py    # Resume + per-epoch checkpoint + CSV logging
│   └── train_predictor.py    # Resume + per-epoch checkpoint + CSV logging
│
├── Kronos/                   # (obtained via git clone, excluded from version control)
├── twse_data/                # (generated locally, excluded from version control)
├── qlib_data/                # (generated locally, excluded from version control)
├── kronos_dataset/           # (generated locally, excluded from version control)
└── checkpoints/              # (training output, upload to HuggingFace)
    ├── tokenizer/
    │   ├── training_log.csv  # Per-step train loss + per-epoch val loss
    │   └── checkpoints/best_model/
    └── predictor/
        ├── training_log.csv
        └── checkpoints/
            ├── epoch_1/  epoch_2/  epoch_3/
            └── best_model/   ← Epoch 1 (val loss 3.0293)
```

---

## Key Configuration (config_tw.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `instrument` | `tw_all` | All TWSE-listed stocks |
| `dataset_begin_time` | `2010-01-01` | Data start date |
| `dataset_end_time` | `2024-12-31` | Data end date |
| `train_time_range` | `2010–2021` | Training split |
| `val_time_range` | `2021–2023` | Validation split |
| `test_time_range` | `2023–2024` | Test split |
| `lookback_window` | `90` | Past 90 trading days as input |
| `predict_window` | `10` | Predict next 10 trading days |
| `batch_size` | `8` | Per-step batch (VRAM + power optimized) |
| `accumulation_steps` | `32` | Gradient accumulation (effective batch = 256) |
| `epochs` | `3` | Training epochs |

---

## Windows Compatibility Fixes

The official Kronos code has several issues on Windows, all fixed in this project:

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `torchrun` launch failure | PyTorch 2.5.1 built without libuv | Set DDP env vars manually, run with `python` |
| `dist.init_process_group` crash | Windows doesn't support nccl backend | Switch to gloo + FileStore |
| `KeyError: 'seed'` | Config class attributes not in `__dict__` | Add explicit `__init__` with attribute copy |
| DataLoader CUDA abort | `pin_memory=True` + multiprocessing on Windows | Change to `pin_memory=False` |
| `KeyError: 'finetuned_tokenizer_path'` | Missing field in config for Predictor | Add path to config_tw.py |

---

## Resume from Checkpoint

After an interruption (crash or manual stop), re-running `launch_windows.bat` will automatically:

1. **Tokenizer**: Detect `completed_epochs` in `summary.json` and resume from the last completed epoch
2. **Predictor**: Same, plus load weights from `epoch_N/` checkpoint and fast-forward the LR scheduler

---

## Loss Visualization

Once training is complete, `training_log.csv` is auto-saved in each model's checkpoint folder:

```bash
python plot_predictor_loss.py
# Reads checkpoints/predictor/training_log.csv
# Outputs predictor_loss_comparison.png
```

CSV format:
```
type,epoch,step,loss,lr
train,1,100,2.4512,0.000040
val,1,-1,3.0293,
```

---

## Upload to HuggingFace

```bash
pip install huggingface_hub
huggingface-cli login

python upload_to_huggingface.py --username talant28
```

This creates and uploads two public repos:
- `talant28/Kronos-TW-Tokenizer`
- `talant28/Kronos-TW-Predictor`

---

## Backtesting

```bash
cd Kronos
python finetune\qlib_test.py --device cuda:0
```

Results saved to `backtest_results/` with auto-generated performance charts.

---

## Data Sources

- **TWSE**: Taiwan Stock Exchange official API (free, no account required)
- **FinMind**: Taiwan financial open data platform (free tier: 600 requests/day)
- **Base model**: [NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)

---

## Citation

This project fine-tunes **Kronos**, a foundation model for financial time series developed by the NeoQuasar team. If you use this work or the pre-trained models, please cite the original paper:

```bibtex
@article{shi2025kronos,
  title={Kronos: A Foundation Model for Time Series},
  author={Shi, Yu and others},
  journal={arXiv preprint arXiv:2504.03249},
  year={2025}
}
```

- Original paper: [Kronos: A Foundation Model for Time Series](https://arxiv.org/abs/2504.03249)
- Kronos GitHub: [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
- Kronos HuggingFace: [NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)

---

## License

Project code is released under the MIT License. Training data is subject to TWSE/FinMind terms of use. Base model weights are subject to NeoQuasar's license. Please comply with all applicable terms.
