import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path: str):
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_training_curves(history: dict, save_dir: str, prefix: str):
    ensure_dir(save_dir)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    if len(epochs) == 0:
        return

    has_single_epoch = len(epochs) == 1

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", linewidth=2, label="train_loss")
    plt.plot(epochs, history["val_loss"], marker="o", linewidth=2, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    if has_single_epoch:
        # One-point runs (smoke test) should still look visible.
        x = epochs[0]
        plt.xlim(x - 0.5, x + 0.5)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(Path(save_dir) / f"{prefix}_loss_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], marker="o", linewidth=2, label="train_acc")
    plt.plot(epochs, history["val_acc"], marker="o", linewidth=2, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    if has_single_epoch:
        x = epochs[0]
        plt.xlim(x - 0.5, x + 0.5)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(Path(save_dir) / f"{prefix}_acc_curve.png"), dpi=150)
    plt.close()
