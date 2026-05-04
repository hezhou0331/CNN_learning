"""
Compare previous runs under outputs_improved with convergence reruns
(e.g. outputs_improved_convergence) matched by summary.json experiment_id.
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Optional


TARGET_ORDER = [
    "exp1_aug",
    "exp4_aug_bn_sgd",
    "exp5_aug_bn_lr",
    "exp8_improved_longer_12",
]


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(output_root: Path):
    rows = []
    if not output_root.is_dir():
        return rows
    for run_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        report_dir = run_dir / "reports"
        if not report_dir.exists():
            continue
        summaries = sorted(report_dir.glob("*_summary.json"))
        if not summaries:
            continue
        summary = read_json(summaries[0])
        eid = summary.get("experiment_id") or ""
        metrics_files = sorted(report_dir.glob("*_test_metrics.json"))
        test_acc = None
        macro_f1 = None
        if metrics_files:
            m = read_json(metrics_files[0])
            test_acc = m.get("accuracy")
            macro_f1 = m.get("macro_f1")
        rows.append(
            {
                "run_dir": run_dir.name,
                "experiment_id": eid,
                "best_epoch": summary.get("best_epoch", ""),
                "best_val_acc": summary.get("best_val_acc", ""),
                "test_accuracy": test_acc,
                "macro_f1": macro_f1,
            }
        )
    return rows


def pick_run(rows, experiment_id: str, *, exclude_run_substr: Optional[str] = None):
    cand = [r for r in rows if r["experiment_id"] == experiment_id]
    if exclude_run_substr:
        cand = [r for r in cand if exclude_run_substr not in r["run_dir"]]
    if not cand:
        return None
    with_test = [r for r in cand if r["test_accuracy"] is not None]
    pool = with_test if with_test else cand
    return max(pool, key=lambda r: (float(r["test_accuracy"] or -1.0), r["run_dir"]))


def fmt_pct(x) -> str:
    if x is None or x == "":
        return ""
    return f"{float(x) * 100:.2f}"


def fmt_val_acc(x) -> str:
    if x is None or x == "":
        return ""
    return f"{float(x):.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous_root", type=str, default="outputs_improved")
    parser.add_argument("--new_root", type=str, default="outputs_improved_convergence")
    parser.add_argument("--exclude_previous_substr", type=str, default="conv150")
    args = parser.parse_args()

    prev_root = Path(args.previous_root)
    new_root = Path(args.new_root)

    prev_rows = collect_runs(prev_root)
    new_rows = collect_runs(new_root)

    out_md = new_root / "convergence_vs_previous.md"
    out_csv = new_root / "convergence_vs_previous.csv"

    lines = [
        "# Convergence rerun vs previous STL10 runs",
        "",
        f"- Previous root: `{prev_root}`",
        f"- Convergence root: `{new_root}`",
        f"- Reruns used `--epochs 150 --patience 20` (val-acc early stop).",
        "",
        "| experiment_id | prev run (under previous root) | prev best_epoch | prev best_val_acc(%) | prev test_acc(%) | new run (under convergence root) | new best_epoch | new best_val_acc(%) | new test_acc(%) | Δ test_acc (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    csv_rows = []

    for eid in TARGET_ORDER:
        prev = pick_run(
            prev_rows,
            eid,
            exclude_run_substr=args.exclude_previous_substr or None,
        )
        new = pick_run(new_rows, eid)

        if not new:
            lines.append(f"| {eid} | (missing new) | | | | | | | | |")
            continue

        pr = prev or {}
        pte = pr.get("test_accuracy")
        nte = new.get("test_accuracy")
        delta_pp = ""
        if pte is not None and nte is not None:
            delta_pp = f"{(float(nte) - float(pte)) * 100:.2f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    eid,
                    pr.get("run_dir", "(none)"),
                    str(pr.get("best_epoch", "")),
                    fmt_val_acc(pr.get("best_val_acc")),
                    fmt_pct(pte),
                    new.get("run_dir", ""),
                    str(new.get("best_epoch", "")),
                    fmt_val_acc(new.get("best_val_acc")),
                    fmt_pct(nte),
                    delta_pp,
                ]
            )
            + " |"
        )

        csv_rows.append(
            {
                "experiment_id": eid,
                "previous_run": pr.get("run_dir", ""),
                "previous_best_epoch": pr.get("best_epoch", ""),
                "previous_best_val_acc": pr.get("best_val_acc", ""),
                "previous_test_accuracy": pte if pte is None else float(pte),
                "new_run": new.get("run_dir", ""),
                "new_best_epoch": new.get("best_epoch", ""),
                "new_best_val_acc": new.get("best_val_acc", ""),
                "new_test_accuracy": nte if nte is None else float(nte),
                "delta_test_accuracy_pp": delta_pp,
            }
        )

    lines.append("")
    lines.append(
        "Use this table to decide whether to cite convergence reruns in the report; "
        "original `outputs_improved` directories are unchanged."
    )
    lines.append("")

    new_root.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_id",
                "previous_run",
                "previous_best_epoch",
                "previous_best_val_acc",
                "previous_test_accuracy",
                "new_run",
                "new_best_epoch",
                "new_best_val_acc",
                "new_test_accuracy",
                "delta_test_accuracy_pp",
            ],
        )
        w.writeheader()
        w.writerows(csv_rows)

    print(f"[Done] Wrote {out_md} and {out_csv}")


if __name__ == "__main__":
    main()
