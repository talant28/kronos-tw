"""
Predictor 訓練 Loss 對比圖
============================
使用方式：
    1. 把 log 資料填入下方 EPOCH_LOGS
    2. python plot_predictor_loss.py
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── 填入各 Epoch 的 console log（貼上原始文字即可）────────────────
EPOCH_LOGS = {
    1: """
    # 把 Epoch 1 的 log 貼在這裡
    """,
    2: """
    # 把 Epoch 2 的 log 貼在這裡
    """,
    3: """
    # 把 Epoch 3 的 log 貼在這裡
    """,
}

# 各 Epoch 驗證 loss（從 Summary 行讀取）
EPOCH_VAL_LOSS = {
    # 1: 2.3456,
    # 2: 2.1234,
    # 3: 2.0123,
}

# ─────────────────────────────────────────────────────────────────

def parse_log(text):
    """從 console log 解析 step 和 loss"""
    steps, losses = [], []
    pattern = r'Step\s+(\d+)/\d+\].*?Loss:\s*([\d\.\-]+)'
    for m in re.finditer(pattern, text):
        steps.append(int(m.group(1)))
        losses.append(float(m.group(2)))

    # 也解析 val loss
    val_pattern = r'Validation Loss:\s*([\d\.]+)'
    val_match = re.search(val_pattern, text)
    val_loss = float(val_match.group(1)) if val_match else None

    return steps, losses, val_loss


def moving_avg(data, window=30):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    pad = np.pad(data, (window//2, window//2), mode='edge')
    return np.convolve(pad, kernel, mode='valid')[:len(data)]


def plot():
    COLORS = {1: '#00e5ff', 2: '#ff6b6b', 3: '#ffd700'}
    LIGHT  = {1: '#00e5ff44', 2: '#ff6b6b44', 3: '#ffd70044'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#1a1a2e')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#ccc', labelsize=9)
        for sp in ax.spines.values(): sp.set_color('#444')
        ax.grid(True, color='#2a2a4a', lw=0.6)
        ax.set_xlabel('Step', color='#aaa', fontsize=10)

    has_data = False
    val_losses = {}

    for epoch, log_text in EPOCH_LOGS.items():
        steps, losses, val_loss = parse_log(log_text)
        if not steps:
            continue
        has_data = True

        color = COLORS[epoch]
        smooth = moving_avg(losses, window=30)

        # Left: raw + smooth
        ax1.plot(steps, losses, color=color, alpha=0.2, lw=0.8)
        ax1.plot(steps, smooth, color=color, lw=2.0, label=f'Epoch {epoch} (smooth)')

        # Right: smoothed only
        ax2.plot(steps, smooth, color=color, lw=2.2, label=f'Epoch {epoch}')

        # Val loss from log or manual override
        vl = EPOCH_VAL_LOSS.get(epoch, val_loss)
        if vl is not None:
            val_losses[epoch] = vl

    if not has_data:
        for ax in [ax1, ax2]:
            ax.text(0.5, 0.5, 'No data yet\nPaste logs into EPOCH_LOGS',
                    transform=ax.transAxes, ha='center', va='center',
                    color='#888', fontsize=14)
        ax1.set_title('Predictor Training Loss (Raw + Smoothed)', color='white', fontsize=12, fontweight='bold')
        ax2.set_title('Smoothed Loss Comparison', color='white', fontsize=12, fontweight='bold')
    else:
        ax1.set_title('Predictor Training Loss (Raw + Smoothed)', color='white', fontsize=12, fontweight='bold', pad=8)
        ax1.set_ylabel('Loss', color='#aaa', fontsize=10)
        ax1.legend(facecolor='#0f3460', edgecolor='#555', labelcolor='white', fontsize=9)

        ax2.set_title('Smoothed Loss — Epoch Comparison', color='white', fontsize=12, fontweight='bold', pad=8)
        ax2.set_ylabel('Loss (Smoothed)', color='#aaa', fontsize=10)
        ax2.legend(facecolor='#0f3460', edgecolor='#555', labelcolor='white', fontsize=9)

        # Val loss table
        if val_losses:
            table_text = "Validation Loss:\n" + "\n".join(
                [f"  Epoch {e}: {v:.4f}" for e, v in sorted(val_losses.items())]
            )
            ax2.text(0.98, 0.98, table_text, transform=ax2.transAxes,
                     ha='right', va='top', fontsize=9, family='monospace',
                     color='#e0e0e0',
                     bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.85, edgecolor='#555'))

    fig.suptitle('Kronos Predictor Fine-tune — Training Progress',
                 color='white', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    out = 'predictor_loss_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {out}")


if __name__ == '__main__':
    plot()
