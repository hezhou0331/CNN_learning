"""
Load a trained checkpoint, pick STL-10 test images, and save Grad-CAM visualizations.

Does not modify train.py. Outputs triptych PNGs (original | heatmap | overlay) under
homework/figures/gradcam/ by default, separate folders for correct vs wrong predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.pyplot as plt

# 避免中文标题在默认字体下缺字警告（Windows 常见）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import torch

from dataset import build_stl10_datasets
from gradcam import GradCAM, denormalize_imagenet_style
from models import build_model


def _args_to_dict(ckpt_args):
    if ckpt_args is None:
        return {}
    if hasattr(ckpt_args, "__dict__"):
        return vars(ckpt_args)
    if isinstance(ckpt_args, dict):
        return ckpt_args
    return dict(ckpt_args)


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = _args_to_dict(ckpt.get("args", {}))
    model_name = ckpt.get("model_name") or ckpt_args.get("model", "baseline")
    use_bn = bool(ckpt_args.get("use_bn", False))
    dropout = float(ckpt_args.get("dropout", 0.0))
    model = build_model(model_name, num_classes=10, use_bn=use_bn, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, {"model_name": model_name, "use_bn": use_bn, "dropout": dropout, "run_tag": ckpt.get("run_tag", "")}


def predict_one(model, x: torch.Tensor, device: torch.device) -> int:
    with torch.no_grad():
        logits = model(x.to(device))
        return int(logits.argmax(dim=1).item())


def collect_indices(
    model,
    test_set,
    device: torch.device,
    n_correct: int,
    n_wrong: int,
    seed: int,
    max_scan: int,
):
    """Scan test set in deterministic order to find first n_correct / n_wrong samples."""
    rng = np.random.RandomState(seed)
    order = np.arange(len(test_set))
    rng.shuffle(order)

    correct = []
    wrong = []
    scanned = 0
    for idx in order:
        idx = int(idx)
        if scanned >= max_scan:
            break
        x, y = test_set[idx]
        y = int(y)
        x_b = x.unsqueeze(0)
        pred = predict_one(model, x_b, device)
        scanned += 1
        if pred == y and len(correct) < n_correct:
            correct.append((idx, int(y), pred))
        elif pred != y and len(wrong) < n_wrong:
            wrong.append((idx, int(y), pred))
        if len(correct) >= n_correct and len(wrong) >= n_wrong:
            break
    return correct, wrong


def save_triptych(
    image_chw: torch.Tensor,
    heatmap_rgb: torch.Tensor,
    overlay_rgb: torch.Tensor,
    out_path: Path,
    title: str,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = denormalize_imagenet_style(image_chw.unsqueeze(0))[0].cpu().numpy()
    img = np.transpose(img, (1, 2, 0))

    h = heatmap_rgb.cpu().numpy()
    o = overlay_rgb.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.clip(img, 0, 1))
    axes[0].set_title("原图")
    axes[0].axis("off")

    axes[1].imshow(np.clip(h, 0, 1))
    axes[1].set_title("Grad-CAM 热力图")
    axes[1].axis("off")

    axes[2].imshow(np.clip(o, 0, 1))
    axes[2].set_title("叠加")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Grad-CAM visualization for STL-10 checkpoints")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="outputs_improved/exp7_improved_longer/checkpoints/exp7_improved_longer_best.pth",
        help="Path to *_best.pth",
    )
    p.add_argument("--data_root", type=str, default="STL10")
    p.add_argument("--output_dir", type=str, default="homework/figures/gradcam")
    p.add_argument("--n_correct", type=int, default=2, help="Number of correctly classified test samples")
    p.add_argument("--n_wrong", type=int, default=2, help="Number of misclassified test samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_scan", type=int, default=8000, help="Stop scanning after this many test images")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")

    model, meta = load_model_from_checkpoint(ckpt_path, device)
    print(f"Loaded model: {meta['model_name']} | use_bn={meta['use_bn']} | dropout={meta['dropout']}")

    train_set, val_set, test_set = build_stl10_datasets(args.data_root, use_augmentation=False)
    class_names = test_set.classes if hasattr(test_set, "classes") else [str(i) for i in range(10)]

    correct, wrong = collect_indices(
        model,
        test_set,
        device,
        n_correct=args.n_correct,
        n_wrong=args.n_wrong,
        seed=args.seed,
        max_scan=args.max_scan,
    )
    print(f"Collected {len(correct)} correct / {len(wrong)} wrong samples (requested {args.n_correct}/{args.n_wrong}).")

    out_root = Path(args.output_dir)
    grad_cam = GradCAM(model)

    def run_one(tag_folder: str, idx: int, y_true: int, pred: int):
        x, _ = test_set[idx]
        x_in = x.unsqueeze(0).to(device).clone().detach().requires_grad_(True)
        _, heat_rgb, over_rgb = grad_cam.compute_cam(x_in, target_class=pred)
        true_name = class_names[y_true]
        pred_name = class_names[pred]
        title = f"{tag_folder} | idx={idx} | y={true_name} | pred={pred_name}"
        fname = f"{tag_folder}_idx{idx}_true{y_true}_pred{pred}.png"
        save_triptych(x, heat_rgb, over_rgb, out_root / tag_folder / fname, title)

    try:
        for i, (idx, y_true, pred) in enumerate(correct):
            run_one("correct", idx, y_true, pred)
        for i, (idx, y_true, pred) in enumerate(wrong):
            run_one("wrong", idx, y_true, pred)

        # Stable filenames for LaTeX / 报告引用
        if correct:
            idx, y_true, pred = correct[0]
            x, _ = test_set[idx]
            x_in = x.unsqueeze(0).to(device).clone().detach().requires_grad_(True)
            _, heat_rgb, over_rgb = grad_cam.compute_cam(x_in, target_class=pred)
            t1 = f"示例（预测正确）| y={class_names[y_true]} | pred={class_names[pred]}"
            t2 = f"预测正确示例 | y={class_names[y_true]} | pred={class_names[pred]}"
            save_triptych(x, heat_rgb, over_rgb, out_root / "gradcam_result.png", t1)
            save_triptych(x, heat_rgb, over_rgb, out_root / "gradcam_correct_example.png", t2)
            print(f"Wrote {out_root / 'gradcam_result.png'} and gradcam_correct_example.png")
        if wrong:
            idx, y_true, pred = wrong[0]
            x, _ = test_set[idx]
            x_in = x.unsqueeze(0).to(device).clone().detach().requires_grad_(True)
            _, heat_rgb, over_rgb = grad_cam.compute_cam(x_in, target_class=pred)
            save_triptych(
                x,
                heat_rgb,
                over_rgb,
                out_root / "gradcam_wrong_example.png",
                f"预测错误示例 | y={class_names[y_true]} | pred={class_names[pred]}",
            )
    finally:
        grad_cam.remove_hooks()

    print(f"[Done] Figures saved under {out_root.resolve()}")


if __name__ == "__main__":
    main()
