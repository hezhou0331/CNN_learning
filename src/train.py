import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from dataset import build_stl10_datasets, build_dataloaders
from models import build_model
from utils import ensure_dir, plot_training_curves, save_json, seed_everything


def run_one_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=torch.cuda.is_available())
        y = y.to(device, non_blocking=torch.cuda.is_available())

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_samples += y.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train STL10 classifier")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "improved"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "step", "cosine"])
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--exp_name", type=str, default="task1")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--continue_run", action="store_true")
    return parser.parse_args()


def build_optimizer(name, model, lr, weight_decay):
    n = name.lower()
    if n == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if n == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if n == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(name, optimizer, epochs):
    n = name.lower()
    if n == "none":
        return None
    if n == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    if n == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unsupported scheduler: {name}")


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_tag = args.run_tag or f"{args.exp_name}_{args.model}_e{args.epochs}"

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        # Keep baseline/improved outputs in separate root folders.
        output_root = Path("outputs_baseline") if args.model == "baseline" else Path("outputs_improved")
    run_root = output_root / run_tag
    ckpt_dir = run_root / "checkpoints"
    fig_dir = run_root / "figures"
    report_dir = run_root / "reports"
    ensure_dir(str(ckpt_dir))
    ensure_dir(str(fig_dir))
    ensure_dir(str(report_dir))

    train_set, val_set, test_set = build_stl10_datasets(
        data_root=args.data_root, use_augmentation=args.use_augmentation
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(args.model, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0
    best_epoch = 0
    no_improve = 0

    ckpt_path = ckpt_dir / f"{run_tag}_best.pth"
    history_path = report_dir / f"{run_tag}_history.json"
    summary_path = report_dir / f"{run_tag}_summary.json"
    start_epoch = 1

    if args.continue_run and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_acc = ckpt.get("best_val_acc", best_val_acc)
        best_epoch = ckpt.get("epoch", 0)
        start_epoch = best_epoch + 1

        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        print(f"[Info] Continue run from epoch {best_epoch}.")

    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, device, optimizer=None)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        print(
            f"Epoch {epoch}/{end_epoch} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {lr:.6f}"
        )

        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "best_val_acc": best_val_acc,
                    "run_tag": run_tag,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[Info] Saved best model -> {ckpt_path}")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"[Info] Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    plot_training_curves(history, str(fig_dir), prefix=run_tag)
    save_json(history, str(history_path))
    save_json(
        {
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "ckpt_path": str(ckpt_path),
            "run_tag": run_tag,
            "output_root": str(output_root),
            "run_root": str(run_root),
        },
        str(summary_path),
    )

    # Also save test loader setup for quick evaluation script usage.
    _ = test_loader
    print(f"[Done] Training finished. Run root: {run_root}")


if __name__ == "__main__":
    main()
