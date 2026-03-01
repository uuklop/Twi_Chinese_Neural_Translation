"""
Publication-ready training metric plots.

Outputs (saved to plot_metrics/):
  perplexity.png      — Train/Val perplexity vs epoch
  accuracy.png        — Train/Val token accuracy vs epoch
  bleu.png            — BLEU scores (Twi→Chi, Chi→Twi, Avg) vs epoch
  lr_schedule.png     — Learning-rate schedule vs epoch
  overview.png        — 2×2 combined panel

Usage:
    python plot_pub.py
    python plot_pub.py --metrics results/metrics.jsonl --outdir plot_metrics
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    # Figure
    "figure.facecolor":      "white",
    "figure.dpi":            150,          # screen preview; saved at 300
    # Axes
    "axes.facecolor":        "white",
    "axes.edgecolor":        "#444444",
    "axes.linewidth":        1.2,
    "axes.labelcolor":       "#111111",
    "axes.titlepad":         10,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    # Ticks
    "xtick.color":           "#444444",
    "ytick.color":           "#444444",
    "xtick.major.size":      5,
    "ytick.major.size":      5,
    "xtick.minor.size":      3,
    "ytick.minor.size":      3,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    # Grid
    "axes.grid":             True,
    "grid.color":            "#dddddd",
    "grid.linestyle":        "-",
    "grid.linewidth":        0.8,
    "grid.alpha":            1.0,
    # Legend
    "legend.frameon":        True,
    "legend.framealpha":     0.9,
    "legend.edgecolor":      "#cccccc",
    "legend.fontsize":       10,
    # Font
    "font.family":           "sans-serif",
    "font.sans-serif":       ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":             11,
    "axes.titlesize":        12,
    "axes.labelsize":        11,
    "xtick.labelsize":       10,
    "ytick.labelsize":       10,
    # Lines
    "lines.linewidth":       2.0,
    "lines.antialiased":     True,
    # Math
    "mathtext.fontset":      "dejavusans",
})

# ── Colour palette (colour-blind safe, Okabe–Ito inspired) ───────────────────
C = {
    "train":  "#0072B2",   # blue
    "val":    "#D55E00",   # vermilion
    "fwd":    "#009E73",   # green
    "rev":    "#CC79A7",   # pink-purple
    "avg":    "#E69F00",   # amber
    "lr":     "#56B4E9",   # sky blue
    "annot":  "#111111",
}

SAVE_DPI = 300


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_metrics(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def last_per_epoch(records: list[dict]) -> list[dict]:
    """Return the last logged record for each epoch (sorted by epoch)."""
    by_epoch: dict[int, dict] = {}
    for r in records:
        e = r["epoch"]
        if e not in by_epoch or r["step"] >= by_epoch[e]["step"]:
            by_epoch[e] = r
    return [by_epoch[e] for e in sorted(by_epoch)]


def _annotate_best(ax, x, y, fmt="BLEU {:.2f}", color=C["annot"],
                   x_offset=6, y_offset=8, mode="max"):
    idx = int(np.argmax(y) if mode == "max" else np.argmin(y))
    ax.annotate(
        fmt.format(y[idx]),
        xy=(x[idx], y[idx]),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        fontsize=9,
        color=color,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=1.2, mutation_scale=10),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color,
                  alpha=0.85, lw=0.8),
    )


def _finalize(ax, xlabel="Epoch", ylabel="", title=""):
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_title(title, fontweight="bold")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(which="minor", length=3, color="#888888")


def save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    print(f"  Saved  {path}")
    plt.close(fig)


# ── Plot 1: Perplexity ────────────────────────────────────────────────────────
def plot_perplexity(epochs, train_ppl, val_ppl, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(epochs, train_ppl, color=C["train"], lw=2.2, label="Train PPL",
            marker="o", markersize=3.5, markevery=5)
    ax.plot(epochs, val_ppl, color=C["val"], lw=2.2, linestyle="--",
            label="Val PPL", marker="s", markersize=3.5, markevery=5)

    _annotate_best(ax, epochs, val_ppl,
                   fmt="Val min {:.1f}", color=C["val"],
                   x_offset=-55, y_offset=14, mode="min")

    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    _finalize(ax, ylabel="Perplexity", title="Training & Validation Perplexity")

    fig.tight_layout()
    save(fig, outdir, "perplexity.png")


# ── Plot 2: Token accuracy ────────────────────────────────────────────────────
def plot_accuracy(epochs, train_acc, val_acc, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(epochs, train_acc, color=C["train"], lw=2.2, label="Train Acc",
            marker="o", markersize=3.5, markevery=5)
    ax.plot(epochs, val_acc, color=C["val"], lw=2.2, linestyle="--",
            label="Val Acc", marker="s", markersize=3.5, markevery=5)

    _annotate_best(ax, epochs, val_acc,
                   fmt="Val peak {:.1f}%", color=C["val"],
                   x_offset=-65, y_offset=-18, mode="max")

    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(loc="lower right")
    _finalize(ax, ylabel="Token Accuracy", title="Training & Validation Token Accuracy")

    fig.tight_layout()
    save(fig, outdir, "accuracy.png")


# ── Plot 3: BLEU ──────────────────────────────────────────────────────────────
def plot_bleu(epochs, bleu_fwd, bleu_rev, avg_bleu, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.fill_between(epochs, bleu_fwd, bleu_rev,
                    alpha=0.08, color=C["avg"], label="_fill")
    ax.plot(epochs, bleu_fwd, color=C["fwd"], lw=2.2,
            label="Twi → Chinese", marker="o", markersize=3.5, markevery=5)
    ax.plot(epochs, bleu_rev, color=C["rev"], lw=2.2, linestyle="--",
            label="Chinese → Twi", marker="s", markersize=3.5, markevery=5)
    ax.plot(epochs, avg_bleu, color=C["avg"], lw=2.6, linestyle=":",
            label="Average", marker="D", markersize=4, markevery=5)

    _annotate_best(ax, epochs, avg_bleu,
                   fmt="Best avg {:.2f}", color=C["avg"],
                   x_offset=-65, y_offset=10, mode="max")
    _annotate_best(ax, epochs, bleu_fwd,
                   fmt="Peak {:.2f}", color=C["fwd"],
                   x_offset=6, y_offset=-18, mode="max")

    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    _finalize(ax, ylabel="BLEU Score", title="BLEU Score by Translation Direction")

    fig.tight_layout()
    save(fig, outdir, "bleu.png")


# ── Plot 4: Learning-rate schedule ───────────────────────────────────────────
def plot_lr(epochs, lrs, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ax.plot(epochs, lrs, color=C["lr"], lw=2.2)
    ax.fill_between(epochs, lrs, alpha=0.15, color=C["lr"])

    peak_idx = int(np.argmax(lrs))
    ax.annotate(
        f"peak  {lrs[peak_idx]:.2e}",
        xy=(epochs[peak_idx], lrs[peak_idx]),
        xytext=(8, 8), textcoords="offset points",
        fontsize=9, color=C["annot"], fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=C["annot"],
                        lw=1.2, mutation_scale=10),
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec=C["annot"], alpha=0.85, lw=0.8),
    )

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2e"))
    ax.set_ylim(bottom=0)
    _finalize(ax, ylabel="Learning Rate",
              title="Transformer Warm-up Learning Rate Schedule")

    fig.tight_layout()
    save(fig, outdir, "lr_schedule.png")


# ── Plot 5: Overview 2×2 panel ───────────────────────────────────────────────
def plot_overview(epochs, train_ppl, val_ppl,
                  train_acc, val_acc,
                  bleu_fwd, bleu_rev, avg_bleu,
                  lrs, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Twi ↔ Chinese  Neural Machine Translation — Training Overview",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # 1. Perplexity
    ax = axes[0, 0]
    ax.plot(epochs, train_ppl, color=C["train"], lw=2.0, label="Train PPL",
            marker="o", markersize=3, markevery=5)
    ax.plot(epochs, val_ppl, color=C["val"], lw=2.0, linestyle="--",
            label="Val PPL", marker="s", markersize=3, markevery=5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    _finalize(ax, ylabel="Perplexity", title="(a)  Perplexity")

    # 2. Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, train_acc, color=C["train"], lw=2.0, label="Train Acc",
            marker="o", markersize=3, markevery=5)
    ax.plot(epochs, val_acc, color=C["val"], lw=2.0, linestyle="--",
            label="Val Acc", marker="s", markersize=3, markevery=5)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9, loc="lower right")
    _finalize(ax, ylabel="Token Accuracy", title="(b)  Token Accuracy")

    # 3. BLEU
    ax = axes[1, 0]
    ax.fill_between(epochs, bleu_fwd, bleu_rev,
                    alpha=0.08, color=C["avg"])
    ax.plot(epochs, bleu_fwd, color=C["fwd"], lw=2.0,
            label="Twi→Chi", marker="o", markersize=3, markevery=5)
    ax.plot(epochs, bleu_rev, color=C["rev"], lw=2.0, linestyle="--",
            label="Chi→Twi", marker="s", markersize=3, markevery=5)
    ax.plot(epochs, avg_bleu, color=C["avg"], lw=2.4, linestyle=":",
            label="Average", marker="D", markersize=3.5, markevery=5)
    best_i = int(np.argmax(avg_bleu))
    ax.annotate(f"Best avg\n{avg_bleu[best_i]:.2f}",
                xy=(epochs[best_i], avg_bleu[best_i]),
                xytext=(-55, 8), textcoords="offset points",
                fontsize=8, color=C["avg"], fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=C["avg"], lw=1.0,
                                mutation_scale=9),
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=C["avg"], alpha=0.9, lw=0.7))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    _finalize(ax, ylabel="BLEU Score", title="(c)  BLEU Score")

    # 4. LR
    ax = axes[1, 1]
    ax.plot(epochs, lrs, color=C["lr"], lw=2.0)
    ax.fill_between(epochs, lrs, alpha=0.15, color=C["lr"])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2e"))
    ax.set_ylim(bottom=0)
    _finalize(ax, ylabel="Learning Rate", title="(d)  Learning Rate Schedule")

    fig.tight_layout(pad=2.5)
    save(fig, outdir, "overview.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Generate publication-ready training metric plots."
    )
    ap.add_argument(
        "--metrics", default="results/metrics.jsonl",
        help="Path to metrics.jsonl (default: results/metrics.jsonl)",
    )
    ap.add_argument(
        "--outdir", default="plot_metrics",
        help="Output directory for PNG files (default: plot_metrics)",
    )
    args = ap.parse_args()

    if not os.path.exists(args.metrics):
        raise FileNotFoundError(f"Metrics file not found: {args.metrics}")

    os.makedirs(args.outdir, exist_ok=True)

    records = load_metrics(args.metrics)
    print(f"Loaded {len(records)} records from {args.metrics}")
    records = last_per_epoch(records)
    print(f"Using {len(records)} epochs ({records[0]['epoch']} – {records[-1]['epoch']})")

    epochs    = [r["epoch"]     for r in records]
    train_ppl = [r["train_ppl"] for r in records]
    val_ppl   = [r["val_ppl"]   for r in records]
    train_acc = [r["train_acc"] for r in records]
    val_acc   = [r["val_acc"]   for r in records]
    bleu_fwd  = [r["bleu_fwd"]  for r in records]
    bleu_rev  = [r["bleu_rev"]  for r in records]
    avg_bleu  = [r["avg_bleu"]  for r in records]
    lrs       = [r["lr"]        for r in records]

    print(f"\nGenerating plots → {args.outdir}/")
    plot_perplexity(epochs, train_ppl, val_ppl, args.outdir)
    plot_accuracy(epochs, train_acc, val_acc, args.outdir)
    plot_bleu(epochs, bleu_fwd, bleu_rev, avg_bleu, args.outdir)
    plot_lr(epochs, lrs, args.outdir)
    plot_overview(epochs, train_ppl, val_ppl,
                  train_acc, val_acc,
                  bleu_fwd, bleu_rev, avg_bleu,
                  lrs, args.outdir)

    print(f"\nDone — {len(os.listdir(args.outdir))} files in '{args.outdir}/'")


if __name__ == "__main__":
    main()
