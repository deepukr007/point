"""
Publication-quality training / validation curves for RF-DETR runs.

Reads `metrics.csv` (PyTorch Lightning CSV logger) from each RF-DETR output
folder and writes a separate set of figures per run, as vector (PDF) + raster
(PNG, 600 dpi), sized for single- (3.5 in) and double-column (7.16 in)
journal layouts.

Usage:
    python plot_training_curves.py
    python plot_training_curves.py --runs rf_detr_output_2 --out figures

Output goes to <out>/<run folder name>/.

Note: RF-DETR logs no validation loss, so validation is reported via
detection metrics (mAP, precision, recall, F1).
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

SINGLE_COL = 3.5   # inches (IEEE / Elsevier / Springer single column)
DOUBLE_COL = 7.16  # inches

# Okabe-Ito colour-blind-safe palette
BLUE, VERMILION, GREEN, PURPLE, ORANGE = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"


def set_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,  # embed TrueType fonts (required by most publishers)
        "ps.fonttype": 42,
    })


def load_run(run_dir: Path) -> pd.DataFrame:
    """Collapse the multi-row-per-epoch Lightning log into one row per epoch."""
    df = pd.read_csv(run_dir / "metrics.csv")
    df = df.drop(columns=["step"]).groupby("epoch", as_index=False).last()
    df["epoch"] = df["epoch"] + 1  # report 1-based epochs
    return df


def plot_series(ax, df, col, color, label, marker="o", ls="-", scale=1.0):
    s = df[["epoch", col]].dropna()
    if s.empty:
        return
    ax.plot(s["epoch"], s[col] * scale, ls=ls, color=color, marker=marker,
            markevery=max(1, len(s) // 12), markerfacecolor="white",
            markeredgewidth=0.8, label=label)


def mark_best(ax, df, col, scale=100):
    """Annotate the best validation epoch with a vertical line."""
    s = df[["epoch", col]].dropna()
    best = s.loc[s[col].idxmax()]
    ax.axvline(best["epoch"], color="0.5", lw=0.7, ls=(0, (1, 1.5)), zorder=0)
    ax.annotate(f"best: {best[col] * scale:.1f}% (ep. {int(best['epoch'])})",
                xy=(best["epoch"], best[col] * scale), xytext=(-4, -14),
                textcoords="offset points", ha="right", fontsize=6.5, color="0.3")


def finish_axis(ax, ylabel, panel=None, percent=False):
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if percent:
        ax.set_ylim(0, 100)
    if panel:
        ax.text(-0.16, 1.04, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, out_dir: Path, name: str):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(out_dir / f"{name}.{ext}", **kw)
    plt.close(fig)
    print(f"  saved {out_dir / name}.{{pdf,png}}")


# --------------------------------------------------------------------------- #
# Individual panels (each draws one run onto a given axis)
# --------------------------------------------------------------------------- #

def panel_train_loss(ax, df):
    plot_series(ax, df, "train/loss", BLUE, "Total loss")
    finish_axis(ax, "Training loss")
    ax.legend(loc="upper right")


def panel_loss_components(ax, df):
    plot_series(ax, df, "train/loss_ce", BLUE, "Classification (focal)", "o")
    plot_series(ax, df, "train/loss_giou", VERMILION, "GIoU", "s")
    plot_series(ax, df, "train/loss_bbox", GREEN, r"Box $L_1$", "^")
    ax.set_yscale("log")
    finish_axis(ax, "Training loss (log scale)")
    ax.legend(loc="upper right")


def panel_map(ax, df):
    plot_series(ax, df, "val/mAP_50", BLUE, "mAP@50", "o", scale=100)
    plot_series(ax, df, "val/mAP_75", GREEN, "mAP@75", "^", scale=100)
    plot_series(ax, df, "val/mAP_50_95", VERMILION, "mAP@50:95", "s", scale=100)
    mark_best(ax, df, "val/mAP_50_95")
    finish_axis(ax, "Validation mAP (%)", percent=True)
    ax.legend(loc="lower right")


def panel_prf(ax, df):
    plot_series(ax, df, "val/precision", BLUE, "Precision", "o", scale=100)
    plot_series(ax, df, "val/recall", GREEN, "Recall", "^", scale=100)
    plot_series(ax, df, "val/F1", VERMILION, r"$F_1$", "s", scale=100)
    finish_axis(ax, "Validation score (%)", percent=True)
    ax.legend(loc="lower right")


def panel_ema_vs_base(ax, df):
    plot_series(ax, df, "val/mAP_50_95", BLUE, "Base model", "o", scale=100)
    plot_series(ax, df, "val/ema_mAP_50_95", VERMILION, "EMA model", "s", ls="--", scale=100)
    finish_axis(ax, "Validation mAP@50:95 (%)", percent=True)
    ax.legend(loc="lower right")


def panel_loss_vs_map(ax, df):
    """Training loss and validation mAP on twin axes."""
    plot_series(ax, df, "train/loss", BLUE, "Training loss", "o")
    finish_axis(ax, "Training loss")
    ax.yaxis.label.set_color(BLUE)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    plot_series(ax2, df, "val/mAP_50_95", VERMILION, "Validation mAP@50:95", "s", ls="--", scale=100)
    ax2.set_ylabel("Validation mAP@50:95 (%)", color=VERMILION)
    ax2.set_ylim(0, 100)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")


# --------------------------------------------------------------------------- #

def summary_table(df, out_dir: Path):
    val = df.dropna(subset=["val/mAP_50_95"])
    best = val.loc[val["val/mAP_50_95"].idxmax()]
    table = pd.DataFrame([{
        "Best epoch": int(best["epoch"]),
        "mAP@50": 100 * best["val/mAP_50"],
        "mAP@50:95": 100 * best["val/mAP_50_95"],
        "mAP@75": 100 * best["val/mAP_75"],
        "mAR": 100 * best["val/mAR"],
        "Precision": 100 * best["val/precision"],
        "Recall": 100 * best["val/recall"],
        "F1": 100 * best["val/F1"],
    }])
    table.to_csv(out_dir / "summary_best_epoch.csv", index=False, float_format="%.2f")
    (out_dir / "summary_best_epoch.tex").write_text(
        table.to_latex(index=False, float_format="%.1f",
                       caption="Validation performance at the best epoch (by mAP@50:95).",
                       label="tab:rfdetr_results"))
    print(table.to_string(index=False, float_format="%.2f"))


def plot_run(run_dir: Path, out_dir: Path):
    print(f"\n{run_dir}")
    df = load_run(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2x2 overview figure (double column)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.6), constrained_layout=True)
    for ax, fn, panel in [(axes[0, 0], panel_train_loss, "(a)"),
                          (axes[0, 1], panel_loss_components, "(b)"),
                          (axes[1, 0], panel_map, "(c)"),
                          (axes[1, 1], panel_prf, "(d)")]:
        fn(ax, df)
        ax.text(-0.16, 1.04, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="left")
    save(fig, out_dir, "training_overview")

    # Single-column figures, one per panel
    for name, fn in [("train_loss", panel_train_loss),
                     ("train_loss_components", panel_loss_components),
                     ("train_loss_vs_val_map", panel_loss_vs_map),
                     ("val_map", panel_map),
                     ("val_precision_recall_f1", panel_prf),
                     ("val_map_ema_vs_base", panel_ema_vs_base)]:
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4), constrained_layout=True)
        fn(ax, df)
        save(fig, out_dir, name)

    summary_table(df, out_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path,
                   default=[Path("rf_detr_output_1"), Path("rf_detr_output_2")])
    p.add_argument("--out", type=Path, default=Path("figures"))
    args = p.parse_args()

    set_style()
    for run_dir in args.runs:
        plot_run(run_dir, args.out / run_dir.name)


if __name__ == "__main__":
    main()
