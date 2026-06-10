"""Generate train/val loss curves for the three main detectors from logged CSVs.

Outputs three PNGs in ``runs/training_curves/``:
  - Faster R-CNN: ``runs/fasterrcnn/optuna/trial_002/metrics.csv``
    (no val loss recorded; val mAP@50:95 plotted on a twin axis instead)
  - RF-DETR:      ``runs/rfdetr/optuna_relabeled_lb/trial_004/metrics.csv``
  - YOLO v8m:     ``runs/yolo/exp7_yolov8m_ep100_img1024_b4/results.csv``
                  (total loss = box + cls + dfl)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "training_curves"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_fasterrcnn() -> Path:
    csv = ROOT / "runs/fasterrcnn/optuna/trial_002/metrics.csv"
    df = pd.read_csv(csv).groupby("epoch", as_index=False).last()

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(df["epoch"], df["train/loss"], color="C0", label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val/mAP_50_95"], color="C3", label="val mAP@50:95")
    ax2.set_ylabel("val mAP@50:95", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    fig.suptitle("Faster R-CNN — Optuna trial_002 (val_loss not recorded)")
    fig.tight_layout()

    out = OUT_DIR / "loss_fasterrcnn_optuna_trial_002.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_rfdetr() -> Path:
    csv = ROOT / "runs/rfdetr/optuna_relabeled_lb/trial_004/metrics.csv"
    df = pd.read_csv(csv)
    train = df.dropna(subset=["train/loss"])[["epoch", "train/loss"]]
    val = df.dropna(subset=["val/loss"])[["epoch", "val/loss"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(train["epoch"], train["train/loss"], color="C0", label="train loss")
    ax.plot(val["epoch"], val["val/loss"], color="C3", label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("RF-DETR — Optuna trial_004 (relabeled)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "loss_rfdetr_optuna_trial_004.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_yolo() -> Path:
    csv = ROOT / "runs/yolo/exp7_yolov8m_ep100_img1024_b4/results.csv"
    df = pd.read_csv(csv)
    df["train/total_loss"] = df[["train/box_loss", "train/cls_loss", "train/dfl_loss"]].sum(axis=1)
    df["val/total_loss"] = df[["val/box_loss", "val/cls_loss", "val/dfl_loss"]].sum(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["epoch"], df["train/total_loss"], color="C0", label="train total loss")
    ax.plot(df["epoch"], df["val/total_loss"], color="C3", label="val total loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("box + cls + dfl loss")
    ax.set_title("YOLOv8m — exp7 (ep100, img1024, b4)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "loss_yolo_exp7.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for fn in (plot_fasterrcnn, plot_rfdetr, plot_yolo):
        print("wrote", fn())
