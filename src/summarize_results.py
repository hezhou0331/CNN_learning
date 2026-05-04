import argparse
import csv
import json
from pathlib import Path


EXPERIMENT_ORDER = [
    "baseline",
    "exp1_aug",
    "exp2_aug_bn",
    "exp3_aug_dropout",
    "exp4_aug_bn_sgd",
    "exp5_aug_bn_lr",
    "exp6_improved_long",
    "exp7_improved_longer",
    "exp8_improved_longer_12",
]


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_runs(output_root: Path):
    runs = []
    for run_dir in sorted([p for p in output_root.iterdir() if p.is_dir()]):
        report_dir = run_dir / "reports"
        if not report_dir.exists():
            continue
        metric_files = sorted(report_dir.glob("*_test_metrics.json"))
        summary_files = sorted(report_dir.glob("*_summary.json"))
        if not metric_files or not summary_files:
            continue
        metrics = read_json(metric_files[0])
        summary = read_json(summary_files[0])
        row = {
            "run": run_dir.name,
            "model": summary.get("model", metrics.get("model_name", "")),
            "use_augmentation": summary.get("use_augmentation", ""),
            "use_bn": summary.get("use_bn", metrics.get("use_bn", "")),
            "dropout": summary.get("dropout", metrics.get("dropout", "")),
            "optimizer": summary.get("optimizer", metrics.get("optimizer", "")),
            "lr": summary.get("lr", metrics.get("lr", "")),
            "best_epoch": summary.get("best_epoch", ""),
            "best_val_acc": summary.get("best_val_acc", ""),
            "test_accuracy": metrics.get("accuracy", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "weighted_f1": metrics.get("weighted_f1", ""),
        }
        runs.append(row)
    order_index = {name: i for i, name in enumerate(EXPERIMENT_ORDER)}
    runs.sort(key=lambda r: order_index.get(r["run"], 10**6))
    return runs


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "model",
        "use_augmentation",
        "use_bn",
        "dropout",
        "optimizer",
        "lr",
        "best_epoch",
        "best_val_acc",
        "test_accuracy",
        "macro_f1",
        "weighted_f1",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# STL10 Experiment Comparison")
    lines.append("")
    lines.append(
        "| run | model | aug | bn | dropout | optimizer | lr | best_val_acc(%) | test_acc(%) | macro_f1 | weighted_f1 |"
    )
    lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        best_val_acc = f"{float(r['best_val_acc']):.2f}" if r["best_val_acc"] != "" else ""
        test_acc = f"{float(r['test_accuracy']) * 100:.2f}" if r["test_accuracy"] != "" else ""
        macro_f1 = f"{float(r['macro_f1']):.4f}" if r["macro_f1"] != "" else ""
        weighted_f1 = f"{float(r['weighted_f1']):.4f}" if r["weighted_f1"] != "" else ""
        lines.append(
            f"| {r['run']} | {r['model']} | {r['use_augmentation']} | {r['use_bn']} | {r['dropout']} | {r['optimizer']} | {r['lr']} | {best_val_acc} | {test_acc} | {macro_f1} | {weighted_f1} |"
        )
    lines.append("")
    if rows:
        best = max(rows, key=lambda x: float(x["test_accuracy"]))
        lines.append(
            f"- Best by test accuracy: `{best['run']}` ({float(best['test_accuracy']) * 100:.2f}%)."
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Summarize STL10 experiment results")
    parser.add_argument("--output_root", type=str, default="outputs_improved")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    rows = discover_runs(output_root)
    if not rows:
        raise RuntimeError(f"No completed runs found in {output_root}")

    write_csv(rows, output_root / "comparison_summary.csv")
    write_markdown(rows, output_root / "comparison_summary.md")
    print(f"[Done] Wrote {len(rows)} rows to comparison files under {output_root}")


if __name__ == "__main__":
    main()
