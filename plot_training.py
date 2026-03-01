"""
Plot training metrics from results/metrics.jsonl

Usage:
    python plot_metrics.py                        # show interactive window
    python plot_metrics.py --save metrics.png     # save to file instead
    python plot_metrics.py --metrics results/metrics.jsonl --save out.png
"""
import argparse
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#1e1e2e',
    'axes.facecolor':    '#2a2a3e',
    'axes.edgecolor':    '#3f3f5a',
    'axes.labelcolor':   '#e2e8f0',
    'axes.titlecolor':   '#e2e8f0',
    'xtick.color':       '#94a3b8',
    'ytick.color':       '#94a3b8',
    'grid.color':        '#3f3f5a',
    'grid.linestyle':    '--',
    'grid.alpha':        0.6,
    'text.color':        '#e2e8f0',
    'legend.facecolor':  '#2a2a3e',
    'legend.edgecolor':  '#3f3f5a',
    'legend.labelcolor': '#e2e8f0',
    'font.family':       'sans-serif',
    'font.size':         10,
})

COLORS = {
    'train':   '#7c3aed',
    'val':     '#22d3ee',
    'fwd':     '#22c55e',
    'rev':     '#f97316',
    'avg':     '#f59e0b',
    'lr':      '#ec4899',
}


def load_metrics(path):
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot(records, save_path=None):
    if not records:
        print("No records found in metrics file.")
        sys.exit(1)

    steps    = [r['step']      for r in records]
    epochs   = [r['epoch']     for r in records]

    train_ppl = [r['train_ppl'] for r in records]
    val_ppl   = [r['val_ppl']   for r in records]
    train_acc = [r['train_acc'] for r in records]
    val_acc   = [r['val_acc']   for r in records]
    bleu_fwd  = [r['bleu_fwd']  for r in records]
    bleu_rev  = [r['bleu_rev']  for r in records]
    avg_bleu  = [r['avg_bleu']  for r in records]
    lrs       = [r['lr']        for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Twi ↔ Chinese  —  Training Metrics', fontsize=14, y=1.01)
    fig.tight_layout(pad=3.5)

    x = steps   # use global steps as the x axis

    def _ax(ax, title, ylabel, ylim_bottom=None):
        ax.set_title(title)
        ax.set_xlabel('Global step')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if ylim_bottom is not None:
            ax.set_ylim(bottom=ylim_bottom)
        # secondary x-axis label: epoch
        epoch_ticks = {}
        for s, e in zip(steps, epochs):
            epoch_ticks.setdefault(e, s)   # first step of each epoch
        if len(epoch_ticks) > 1:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ep_vals = sorted(epoch_ticks.keys())
            ep_pos  = [epoch_ticks[e] for e in ep_vals]
            ax2.set_xticks(ep_pos)
            ax2.set_xticklabels([f'ep{e}' for e in ep_vals],
                                fontsize=7, color='#94a3b8', rotation=45)
            ax2.tick_params(length=0)
            for spine in ax2.spines.values():
                spine.set_visible(False)

    # ── 1. Perplexity ──────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x, train_ppl, color=COLORS['train'], lw=1.8, label='Train PPL')
    ax.plot(x, val_ppl,   color=COLORS['val'],   lw=1.8, label='Val PPL',
            linestyle='--')
    ax.legend()
    _ax(ax, 'Perplexity', 'PPL', ylim_bottom=1)

    # ── 2. Accuracy ────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(x, train_acc, color=COLORS['train'], lw=1.8, label='Train Acc')
    ax.plot(x, val_acc,   color=COLORS['val'],   lw=1.8, label='Val Acc',
            linestyle='--')
    ax.legend()
    _ax(ax, 'Token Accuracy (%)', 'Accuracy (%)', ylim_bottom=0)

    # ── 3. BLEU ────────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(x, bleu_fwd, color=COLORS['fwd'], lw=1.8, label='Twi→Chi BLEU')
    ax.plot(x, bleu_rev, color=COLORS['rev'], lw=1.8, label='Chi→Twi BLEU')
    ax.plot(x, avg_bleu, color=COLORS['avg'], lw=2.4, label='Avg BLEU',
            linestyle=':')
    # annotate best avg BLEU
    best_idx = avg_bleu.index(max(avg_bleu))
    ax.annotate(f"best {avg_bleu[best_idx]:.2f}",
                xy=(x[best_idx], avg_bleu[best_idx]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=8, color=COLORS['avg'],
                arrowprops=dict(arrowstyle='->', color=COLORS['avg'], lw=1.2))
    ax.legend()
    _ax(ax, 'BLEU Score', 'BLEU', ylim_bottom=0)

    # ── 4. Learning rate ───────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(x, lrs, color=COLORS['lr'], lw=1.8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    _ax(ax, 'Learning Rate', 'LR')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metrics', default='results/metrics.jsonl',
                   help='Path to metrics.jsonl (default: results/metrics.jsonl)')
    p.add_argument('--save', default=None,
                   help='Save plot to this file instead of showing interactively')
    args = p.parse_args()

    if not os.path.exists(args.metrics):
        print(f"Metrics file not found: {args.metrics}")
        print("Train for at least one eval step first, then re-run this script.")
        sys.exit(1)

    records = load_metrics(args.metrics)
    print(f"Loaded {len(records)} eval steps from {args.metrics}")
    plot(records, save_path=args.save)


if __name__ == '__main__':
    main()
