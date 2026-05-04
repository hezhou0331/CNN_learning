"""
Overlay training curves for ablation report figures (from *_history.json).
Each output figure: one comparison, four panels — train_loss, val_loss, train_acc, val_acc.
Dashed = reference run, solid = compared run.

No per-epoch test loss is logged; test accuracy appears only in the summary table.

Usage (from project root):
  python src/plot_comparison_curves.py
  python src/plot_comparison_curves.py --output_dir homework/figures/comparisons
  python src/plot_comparison_curves.py --combined   # optional single mega-figure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_history(root: Path, run_dir: str) -> dict:
    rep = root / run_dir / "reports"
    files = sorted(rep.glob("*_history.json"))
    if not files:
        raise FileNotFoundError(f"No history json under {rep}")
    with open(files[0], encoding="utf-8") as f:
        return json.load(f)


def epochs(h: dict) -> list:
    n = len(h["val_acc"])
    return list(range(1, n + 1))


METRICS = [
    ("train_loss", "Train loss"),
    ("val_loss", "Val loss"),
    ("train_acc", "Train acc (%)"),
    ("val_acc", "Val acc (%)"),
]


def plot_overlay(
    ax,
    ref: dict,
    cmp: dict,
    key: str,
    ref_label: str,
    cmp_label: str,
    *,
    title: str,
    show_xlabel: bool = False,
    show_legend: bool = True,
) -> None:
    er, ec = epochs(ref), epochs(cmp)
    ax.plot(er, ref[key], "--", color="0.42", lw=1.25, label=ref_label)
    ax.plot(ec, cmp[key], "-", color="C0", lw=1.25, label=cmp_label)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.28)
    ax.tick_params(labelsize=7)
    if show_legend:
        ax.legend(fontsize=7, loc="best", framealpha=0.92)
    if show_xlabel:
        ax.set_xlabel("Epoch", fontsize=8)


def plot_one_comparison(
    root: Path,
    ref_run: str,
    cmp_run: str,
    ref_label: str,
    cmp_label: str,
    out_path: Path,
    *,
    suptitle: str | None = None,
) -> None:
    ref = load_history(root, ref_run)
    cmp = load_history(root, cmp_run)
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.35), constrained_layout=True)
    for ax, (key, title) in zip(axes, METRICS, strict=True):
        plot_overlay(
            ax,
            ref,
            cmp,
            key,
            ref_label,
            cmp_label,
            title=title,
            show_xlabel=True,
            show_legend=(key == "val_loss"),
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=8.5, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_in_one(root: Path, out_path: Path) -> None:
    row_labels = [
        "(I) baseline→exp1",
        "(II) exp1→exp2",
        "(III) exp2→exp3",
        "(IV) exp2→exp4",
        "(V) exp2→exp5",
        "(VI-a) exp2→exp6",
        "(VI-b) exp2→exp7",
        "(VI-c) exp2→exp8",
    ]
    pairs = [
        ("baseline", "exp1_aug", "baseline", "exp1"),
        ("exp1_aug", "exp2_aug_bn", "exp1", "exp2"),
        ("exp2_aug_bn", "exp3_aug_dropout", "exp2", "exp3"),
        ("exp2_aug_bn", "exp4_aug_bn_sgd", "exp2", "exp4"),
        ("exp2_aug_bn", "exp5_aug_bn_lr", "exp2", "exp5"),
    ]
    ref2 = load_history(root, "exp2_aug_bn")
    h6 = load_history(root, "exp6_improved_long_6")
    h7 = load_history(root, "exp7_improved_long_9")
    h8 = load_history(root, "exp8_improved_longer_12")
    depth_pairs = [
        (ref2, h6, "exp2", "exp6"),
        (ref2, h7, "exp2", "exp7"),
        (ref2, h8, "exp2", "exp8"),
    ]

    n_data_rows = len(pairs) + len(depth_pairs)
    n_header = 1
    nrows = n_data_rows + n_header

    fig = plt.figure(figsize=(11.2, 20.5), constrained_layout=False)
    gs = GridSpec(
        nrows,
        5,
        figure=fig,
        height_ratios=[0.45] + [1.0] * n_data_rows,
        width_ratios=[0.22, 1.0, 1.0, 1.0, 1.0],
        hspace=0.26,
        wspace=0.28,
        left=0.06,
        right=0.98,
        top=0.97,
        bottom=0.04,
    )

    for j, (_, title) in enumerate(METRICS):
        axh = fig.add_subplot(gs[0, j + 1])
        axh.axis("off")
        axh.set_title(title, fontsize=9, fontweight="bold")
    fig.add_subplot(gs[0, 0]).axis("off")

    def fill_row(ridx: int, ref: dict, cmp: dict, rl: str, cl: str) -> None:
        ax_l = fig.add_subplot(gs[ridx, 0])
        ax_l.axis("off")
        ax_l.text(
            0.5,
            0.5,
            row_labels[ridx - 1],
            transform=ax_l.transAxes,
            va="center",
            ha="center",
            fontsize=8.2,
        )
        last_row = ridx == nrows - 1
        for j, (key, title) in enumerate(METRICS):
            ax = fig.add_subplot(gs[ridx, j + 1])
            plot_overlay(
                ax,
                ref,
                cmp,
                key,
                rl,
                cl,
                title=title,
                show_xlabel=last_row,
                show_legend=(j == 1),
            )

    r = 1
    for ref_run, cmp_run, rl, cl in pairs:
        fill_row(r, load_history(root, ref_run), load_history(root, cmp_run), rl, cl)
        r += 1
    for ref, h, rl, cl in depth_pairs:
        fill_row(r, ref, h, rl, cl)
        r += 1

    fig.suptitle(
        "Training curves (dashed = reference, solid = compared). "
        "No per-epoch test loss in logs.",
        fontsize=8.8,
        y=0.993,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=135)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=Path("outputs_improved"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("homework/figures/comparisons"),
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write cmp_all_training_ablation.png (single figure)",
    )
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = [
        ("cmp_01_baseline_exp1.png", "baseline", "exp1_aug", "baseline", "exp1"),
        ("cmp_02_exp1_exp2.png", "exp1_aug", "exp2_aug_bn", "exp1", "exp2"),
        ("cmp_03_exp2_exp3.png", "exp2_aug_bn", "exp3_aug_dropout", "exp2", "exp3"),
        ("cmp_04_exp2_exp4.png", "exp2_aug_bn", "exp4_aug_bn_sgd", "exp2", "exp4"),
        ("cmp_05_exp2_exp5.png", "exp2_aug_bn", "exp5_aug_bn_lr", "exp2", "exp5"),
        ("cmp_06a_exp2_exp6.png", "exp2_aug_bn", "exp6_improved_long_6", "exp2", "exp6"),
        ("cmp_06b_exp2_exp7.png", "exp2_aug_bn", "exp7_improved_long_9", "exp2", "exp7"),
        ("cmp_06c_exp2_exp8.png", "exp2_aug_bn", "exp8_improved_longer_12", "exp2", "exp8"),
        ("cmp_06d_exp6_exp9.png", "exp6_improved_long_6", "exp9_aug_bn_15conv", "exp6", "exp9"),
    ]
    for fname, rr, cr, rl, cl in groups:
        plot_one_comparison(
            args.output_root,
            rr,
            cr,
            rl,
            cl,
            out_dir / fname,
            suptitle="Dashed = reference, solid = compared | No per-epoch test loss in logs",
        )
        print(f"[OK] {out_dir / fname}")

    if args.combined:
        p = out_dir / "cmp_all_training_ablation.png"
        plot_all_in_one(args.output_root, p)
        print(f"[OK] {p}")


if __name__ == "__main__":
    main()
