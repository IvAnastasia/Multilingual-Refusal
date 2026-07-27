"""
Cross-lingual transfer matrix: apply each SOURCE language's native refusal direction
(ablation) to each TARGET language's harmful test set, and show a WildGuard metric.

Reads output/<alias>/xling/<source>_to_<target>/completions/harmful_harm_ablation_evaluations.json.

Usage:
  python -m scripts.plot_transfer_matrix -m Qwen2.5-14B-Instruct \
      --sources be ba tg --targets en be ba tg --metric wildguard_compliance \
      --save figures/transfer_matrix_14b.png
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scripts.viz_style import palette, style, INK

LANG_LABELS = {"en": "English", "ba": "Bashkir", "be": "Belarusian", "tg": "Tajik", "yo": "Yoruba"}
METRIC_LABELS = {
    "wildguard_compliance": "WildGuard compliance (after ablation)",
    "wildguard_refusal": "WildGuard refusal (after ablation)",
    "wildguard_harmful": "WildGuard harmful / ASR (after ablation)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_alias", "-m", required=True)
    ap.add_argument("--sources", nargs="+", default=["be", "ba", "tg"])
    ap.add_argument("--targets", nargs="+", default=["en", "be", "ba", "tg"])
    ap.add_argument("--metric", default="wildguard_compliance", choices=list(METRIC_LABELS))
    ap.add_argument("--base_dir", default="output")
    ap.add_argument("--save", "-s", default=None)
    args = ap.parse_args()

    P = style()
    S, T = args.sources, args.targets
    M = np.full((len(S), len(T)), np.nan)
    for i, s in enumerate(S):
        for j, t in enumerate(T):
            p = osp.join(args.base_dir, args.model_alias, "xling", f"{s}_to_{t}",
                         "completions", "harmful_harm_ablation_evaluations.json")
            if osp.isfile(p):
                M[i, j] = json.load(open(p, encoding="utf-8")).get(args.metric, np.nan)
            else:
                print(f"  [missing] {s}->{t}")

    # print table
    print(f"\n{args.metric} (rows=direction source, cols=test target):")
    print(f"{'src/tgt':<12}" + "".join(f"{LANG_LABELS.get(t,t):<12}" for t in T))
    for i, s in enumerate(S):
        print(f"{LANG_LABELS.get(s,s):<12}" + "".join(
            (f"{M[i,j]:<12.3f}" if not np.isnan(M[i,j]) else f"{'--':<12}") for j in range(len(T))))

    # sequential ramp: light -> raspberry (darker = higher compliance = more jailbroken)
    cmap = LinearSegmentedColormap.from_list("seq_berry", ["#f5eef1", "#d29aa8", "#a8465f", "#6e2b3c"])
    cmap.set_bad("#e6e4dd")
    fig, ax = plt.subplots(figsize=(1.4 * len(T) + 1.5, 1.2 * len(S) + 1.5))
    im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(len(T))); ax.set_yticks(range(len(S)))
    ax.set_xticklabels([LANG_LABELS.get(t, t) for t in T], fontsize=11)
    ax.set_yticklabels([LANG_LABELS.get(s, s) for s in S], fontsize=11)
    ax.set_xlabel("Ablation target", fontsize=11, labelpad=12)
    ax.set_ylabel("Source of refusal direction", fontsize=11, labelpad=10)
    for i in range(len(S)):
        for j in range(len(T)):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if M[i, j] > 0.55 else INK)

    # mark self-transfer cells (source language == target language)
    for i, s in enumerate(S):
        for j, t in enumerate(T):
            if s == t:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False,
                                           edgecolor=INK, linewidth=2.6, zorder=5))
    fig.text(0.5, -0.02, "Boxed = self-transfer (direction applied to its own language)",
             ha="center", fontsize=9, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if args.save:
        os.makedirs(osp.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"\nSaved: {args.save}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
