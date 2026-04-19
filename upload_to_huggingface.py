"""
上傳微調模型到 HuggingFace
============================
訓練完成後執行此腳本：
    python upload_to_huggingface.py --username YOUR_HF_USERNAME

前置步驟：
    pip install huggingface_hub
    huggingface-cli login
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

BASE_DIR = Path(__file__).parent.resolve()

TOKENIZER_PATH = BASE_DIR / "checkpoints" / "tokenizer" / "checkpoints" / "best_model"
PREDICTOR_PATH = BASE_DIR / "checkpoints" / "predictor" / "checkpoints" / "best_model"

TOKENIZER_CARD = """---
language: zh
license: mit
tags:
  - finance
  - time-series
  - taiwan
  - stock
  - kronos
base_model: NeoQuasar/Kronos-Tokenizer-base
---

# Kronos-TW-Tokenizer

Kronos VQ Tokenizer fine-tuned on Taiwan Stock Exchange (TWSE) daily OHLCV data.

## Base Model
[NeoQuasar/Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base)

## Training Data
- Source: TWSE listed stocks (613 stocks after quality filtering)
- Period: 2010-01-01 ~ 2024-12-31
- Features: open, high, low, close, volume, amount
- Lookback: 90 trading days

## Training Config
- Epochs: 3
- Batch size: 8 (effective 256 with gradient accumulation)
- Learning rate: 2e-4
- Hardware: RTX 4080 Laptop 12GB
"""

PREDICTOR_CARD = """---
language: zh
license: mit
tags:
  - finance
  - time-series
  - taiwan
  - stock
  - kronos
base_model: NeoQuasar/Kronos-base
---

# Kronos-TW-Predictor

Kronos Autoregressive Predictor fine-tuned on Taiwan Stock Exchange (TWSE) daily data.

## Base Model
[NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)

## Training Data
- Source: TWSE listed stocks (613 stocks after quality filtering)
- Period: 2010-01-01 ~ 2024-12-31
- Features: open, high, low, close, volume, amount
- Predict window: 10 trading days

## Training Config
- Epochs: 3
- Batch size: 8 (effective 256 with gradient accumulation)
- Learning rate: 4e-5
- Hardware: RTX 4080 Laptop 12GB

## Usage
Use with [Kronos-TW-Tokenizer](https://huggingface.co/talant28/Kronos-TW-Tokenizer)

## Citation
This model is fine-tuned from Kronos. If you use this work, please cite the original paper:

```bibtex
@article{shi2025kronos,
  title={Kronos: A Foundation Model for Time Series},
  author={Shi, Yu and others},
  journal={arXiv preprint arXiv:2504.03249},
  year={2025}
}
```
"""


def upload(username: str, private: bool = False):
    api = HfApi()

    tokenizer_repo = f"{username}/Kronos-TW-Tokenizer"
    predictor_repo = f"{username}/Kronos-TW-Predictor"

    # ── Tokenizer ──────────────────────────────────────────────
    if TOKENIZER_PATH.exists():
        print(f"[1/2] Creating repo: {tokenizer_repo}")
        create_repo(tokenizer_repo, repo_type="model", private=private, exist_ok=True)

        # Write model card
        (TOKENIZER_PATH / "README.md").write_text(TOKENIZER_CARD)

        print(f"      Uploading from {TOKENIZER_PATH} ...")
        api.upload_folder(
            folder_path=str(TOKENIZER_PATH),
            repo_id=tokenizer_repo,
            repo_type="model",
        )
        print(f"      Done: https://huggingface.co/{tokenizer_repo}")
    else:
        print(f"[1/2] SKIP: Tokenizer checkpoint not found at {TOKENIZER_PATH}")

    # ── Predictor ──────────────────────────────────────────────
    if PREDICTOR_PATH.exists():
        print(f"\n[2/2] Creating repo: {predictor_repo}")
        create_repo(predictor_repo, repo_type="model", private=private, exist_ok=True)

        # Write model card
        card = PREDICTOR_CARD.replace("talant28", username)
        (PREDICTOR_PATH / "README.md").write_text(card)

        print(f"      Uploading from {PREDICTOR_PATH} ...")
        api.upload_folder(
            folder_path=str(PREDICTOR_PATH),
            repo_id=predictor_repo,
            repo_type="model",
        )
        print(f"      Done: https://huggingface.co/{predictor_repo}")
    else:
        print(f"[2/2] SKIP: Predictor checkpoint not found at {PREDICTOR_PATH}")

    print("\nAll uploads complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="Your HuggingFace username")
    parser.add_argument("--private", action="store_true", help="Make repos private")
    args = parser.parse_args()
    upload(args.username, args.private)
