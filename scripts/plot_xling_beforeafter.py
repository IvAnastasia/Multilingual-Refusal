"""
Paper-style Figure 2 analog: compliance to harmful queries BEFORE vs AFTER ablating a
refusal vector derived from a (non-English) source language, shown per target language.

One panel per source direction. Baseline (before) from output/<alias>/<target>/, ablation
(after) from output/<alias>/xling/<source>_to_<target>/.

Usage:
  python -m scripts.plot_xling_beforeafter -m Qwen2.5-14B-Instruct \
      --sources be ba tg --targets en be ba tg --save figures/xling_beforeafter_14b.png
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt

from scripts.viz_style import palette, style, INK, GRID

LANG_LABELS = {"en": "English", "ba": "Bashkir", "be": "Belarusian", "tg": "Tajik", "yo": "Yoruba"}
METRIC = "wildguard_compliance"


def _read(path):
    return json.load(open(path, encoding="utf-8")).get(METRIC, np.nan) if osp.isfile(path) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_alias", "-m", required=True)
    ap.add_argument("--sources", nargs="+", default=["be", "ba", "tg"])
    ap.add_argument("--targets", nargs="+", default=["en", "be", "ba", "tg"])
    ap.add_argument("--base_dir", default="output")
    ap.add_argument("--save", "-s", default=None)
    args = ap.parse_args()

    P = style()
    base_c = P["baseline"]
    abl_c = P["ablation"]

    # baseline (before) per target — target-only, no steering
    baseline = {t: _read(osp.join(args.base_dir, args.model_alias, t, "completions",
                                   "harmful_baseline_evaluations.json")) for t in args.targets}
    # ablation (after) per (source, target)
    ablation = {s: {t: _read(osp.join(args.base_dir, args.model_alias, "xling", f"{s}_to_{t}",
                                      "completions", "harmful_harm_ablation_evaluations.json"))
                    for t in args.targets} for s in args.sources}

    n = len(args.sources)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 0.5, 4.4), sharey=True)
    if n == 1:
        axes = [axes]
    x = np.arange(len(args.targets))
    w = 0.38
    for ax, s in zip(axes, args.sources):
        before = [baseline[t] for t in args.targets]
        after = [ablation[s][t] for t in args.targets]
        b1 = ax.bar(x - w / 2, before, w * 0.9, color=base_c, label="Before (baseline)", zorder=3)
        b2 = ax.bar(x + w / 2, after, w * 0.9, color=abl_c, label="After ablation", zorder=3)
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}",
                            ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([LANG_LABELS.get(t, t) for t in args.targets], fontsize=9.5, rotation=20, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.arange(0, 1.01, 0.25))
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f"{LANG_LABELS.get(s, s)} direction", fontsize=12, pad=8)
    axes[0].set_ylabel("Compliance to harmful queries", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if args.save:
        os.makedirs(osp.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved: {args.save}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
