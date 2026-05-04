"""
Compare outputs_improved (previous runs) vs outputs_improved_e100 for selected experiment_ids.
Excludes run directories whose names contain e100p20 or conv150 from the previous pool.
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Optional


TARGET_ORDER = [
    "exp2_aug_bn",
    "exp3_aug_dropout",
    "exp6_improved_long_6",
    "exp7_improved_long_9",
    "exp9_aug_bn_15conv",
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
                "stop_reason": summary.get("stop_reason", ""),
                "last_epoch": summary.get("last_epoch", ""),
            }
        )
    return rows


def pick_previous(rows, experiment_id: str) -> Optional[dict]:
    cand = [
        r
        for r in rows
        if r["experiment_id"] == experiment_id
        and "e100p20" not in r["run_dir"]
        and "conv150" not in r["run_dir"]
    ]
    if not cand:
        return None
    with_test = [r for r in cand if r["test_accuracy"] is not None]
    pool = with_test if with_test else cand
    return max(pool, key=lambda r: (float(r["test_accuracy"] or -1.0), r["run_dir"]))


def pick_e100(rows, experiment_id: str) -> Optional[dict]:
    cand = [r for r in rows if r["experiment_id"] == experiment_id and "e100p20" in r["run_dir"]]
    if not cand:
        return None
    return cand[0]


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
    parser.add_argument("--new_root", type=str, default="outputs_improved_e100")
    args = parser.parse_args()

    prev_root = Path(args.previous_root)
    new_root = Path(args.new_root)

    prev_rows = collect_runs(prev_root)
    new_rows = collect_runs(new_root)

    out_md = new_root / "e100_vs_previous.md"
    out_csv = new_root / "e100_vs_previous.csv"

    lines = [
        "# 100-epoch capped runs vs previous `outputs_improved`",
        "",
        f"- Previous pool: `{prev_root}` (excludes run dirs containing `e100p20` or `conv150`).",
        f"- New runs: `{new_root}` (`*_e100p20`, `--epochs 100 --patience 20`, val-acc early stop).",
        "",
        "| experiment_id | prev run | prev test_acc(%) | new run | new last_epoch | stop_reason | new test_acc(%) | Δ test (pp) |",
        "|---|---:|---:|---|---:|---|---:|---:|",
    ]

    csv_rows = []
    for eid in TARGET_ORDER:
        pr = pick_previous(prev_rows, eid)
        nw = pick_e100(new_rows, eid)
        if not nw:
            lines.append(f"| {eid} | (no new) | | | | | | |")
            continue
        pte = pr.get("test_accuracy") if pr else None
        nte = nw.get("test_accuracy")
        delta = ""
        if pte is not None and nte is not None:
            delta = f"{(float(nte) - float(pte)) * 100:.2f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    eid,
                    pr.get("run_dir", "(none)") if pr else "(none)",
                    fmt_pct(pte),
                    nw.get("run_dir", ""),
                    str(nw.get("last_epoch", "")),
                    str(nw.get("stop_reason", "")),
                    fmt_pct(nte),
                    delta,
                ]
            )
            + " |"
        )
        csv_rows.append(
            {
                "experiment_id": eid,
                "previous_run": pr.get("run_dir", "") if pr else "",
                "previous_test_accuracy": pte,
                "new_run": nw.get("run_dir", ""),
                "new_last_epoch": nw.get("last_epoch", ""),
                "new_stop_reason": nw.get("stop_reason", ""),
                "new_test_accuracy": nte,
                "delta_test_pp": delta,
            }
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
                "previous_test_accuracy",
                "new_run",
                "new_last_epoch",
                "new_stop_reason",
                "new_test_accuracy",
                "delta_test_pp",
            ],
        )
        w.writeheader()
        w.writerows(csv_rows)

    print(f"[Done] Wrote {out_md} and {out_csv}")


if __name__ == "__main__":
    main()
