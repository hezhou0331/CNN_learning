import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from dataset import build_stl10_datasets, build_dataloaders
from models import build_model
from utils import ensure_dir, save_json

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate STL10 checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best checkpoint")
    parser.add_argument("--data_root", type=str, default="STL10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="task1")
    parser.add_argument("--use_bn", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(9, 7))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    else:
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        ticks = range(len(class_names))
        plt.xticks(ticks, class_names, rotation=45, ha="right")
        plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt.get("model_name", "baseline")
    run_tag = ckpt.get("run_tag", Path(args.ckpt).stem.replace("_best", ""))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loaded model: {model_name} | run_tag: {run_tag}")

    # Evaluation should not use random augmentation.
    train_set, val_set, test_set = build_stl10_datasets(args.data_root, use_augmentation=False)
    _, _, test_loader = build_dataloaders(
        train_set,
        val_set,
        test_set,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    ckpt_args = ckpt.get("args", {})
    use_bn = bool(ckpt_args.get("use_bn", args.use_bn))
    dropout = float(ckpt_args.get("dropout", args.dropout))
    model = build_model(model_name, num_classes=10, use_bn=use_bn, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_true = []
    all_pred = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=torch.cuda.is_available())
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            all_pred.extend(pred)
            all_true.extend(y.numpy().tolist())

    class_names = test_set.classes if hasattr(test_set, "classes") else [str(i) for i in range(10)]
    cm = confusion_matrix(all_true, all_pred)
    report_dict = classification_report(all_true, all_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(all_true, all_pred, target_names=class_names, digits=4)

    accuracy = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro")
    weighted_f1 = f1_score(all_true, all_pred, average="weighted")

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(report_text)

    ckpt_path = Path(args.ckpt).resolve()
    if ckpt_path.parent.name == "checkpoints":
        run_root = ckpt_path.parent.parent
    else:
        run_root = Path("outputs")

    out_report = run_root / "reports"
    out_fig = run_root / "figures"
    ensure_dir(str(out_report))
    ensure_dir(str(out_fig))

    file_prefix = run_tag if args.exp_name == "task1" else args.exp_name

    save_json(
        {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
            "ckpt": args.ckpt,
            "model_name": model_name,
            "use_bn": use_bn,
            "dropout": dropout,
            "optimizer": ckpt_args.get("optimizer"),
            "lr": ckpt_args.get("lr"),
        },
        str(out_report / f"{file_prefix}_{model_name}_test_metrics.json"),
    )
    with open(out_report / f"{file_prefix}_{model_name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    plot_confusion_matrix(cm, class_names, str(out_fig / f"{file_prefix}_{model_name}_confusion_matrix.png"))
    print("[Done] Saved test metrics and confusion matrix.")


if __name__ == "__main__":
    main()
