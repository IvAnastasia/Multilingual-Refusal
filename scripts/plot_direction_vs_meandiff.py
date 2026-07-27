"""
Figure-5 analog: cross-lingual cosine similarity between a SOURCE language's refusal
direction (a single vector extracted at its own (pos, layer)) and a TARGET language's
difference-in-means vectors across ALL (position, decoder-layer) cells.

Grid = source (rows) x target (cols). Each subplot is a position x layer heatmap;
bright = high cosine similarity (aligned refusal encoding). A consistent bright band at
some layer indicates aligned refusal signals across languages.

All inputs are precomputed artifacts (no GPU):
  source direction:  pipeline/runs/<alias>/<src>/direction_ablation.pt   (en: root)
  target diff-means: pipeline/runs/<alias>/<tgt>/generate_directions/mean_diffs.pt

Usage:
  python -m scripts.plot_direction_vs_meandiff -m Qwen2.5-14B-Instruct \
      --langs en be ba tg --save figures/dir_vs_meandiff_14b.png
"""
import argparse
import os
import os.path as osp

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

from scripts.viz_style import palette, style, INK

LANG_LABELS = {"en": "English", "ba": "Bashkir", "be": "Belarusian", "tg": "Tajik", "yo": "Yoruba"}


def _run_dir(alias, lang):
    base = osp.join("pipeline", "runs", alias)
    return base if lang == "en" else osp.join(base, lang)


def load_source_direction(alias, lang):
    d = torch.load(osp.join(_run_dir(alias, lang), "direction_ablation.pt"), map_location="cpu")
    if isinstance(d, list):
        d = d[0]
    v = d.to(torch.float64).flatten()
    return v / v.norm()


def load_target_meandiffs(alias, lang):
    md = torch.load(osp.join(_run_dir(alias, lang), "generate_directions", "mean_diffs.pt"),
                    map_location="cpu").to(torch.float64)  # [n_pos, n_layers, d_model]
    return md / md.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_alias", "-m", required=True)
    ap.add_argument("--langs", nargs="+", default=["en", "be", "ba", "tg"])
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--save", "-s", default=None)
    args = ap.parse_args()

    style()
    S = args.sources or args.langs
    T = args.targets or args.langs

    src_dirs = {s: load_source_direction(args.model_alias, s) for s in S}
    tgt_md = {t: load_target_meandiffs(args.model_alias, t) for t in T}
    n_pos, n_layers, _ = next(iter(tgt_md.values())).shape

    # cosine maps + global scale
    maps, vmax = {}, 0.0
    for s in S:
        for t in T:
            c = (tgt_md[t] @ src_dirs[s]).numpy()  # [n_pos, n_layers]
            maps[(s, t)] = c
            vmax = max(vmax, float(np.abs(c).max()))

    div = palette()["diverging"]
    cmap = LinearSegmentedColormap.from_list("div", list(div))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    pos_labels = list(range(-n_pos, 0))

    fig, axes = plt.subplots(len(S), len(T), figsize=(2.7 * len(T) + 1, 2.1 * len(S) + 0.6),
                             squeeze=False)
    im = None
    for i, s in enumerate(S):
        for j, t in enumerate(T):
            ax = axes[i][j]
            im = ax.imshow(maps[(s, t)], aspect="auto", cmap=cmap, norm=norm, origin="lower",
                           extent=[0, n_layers, -0.5, n_pos - 0.5])
            if i == 0:
                ax.set_title(LANG_LABELS.get(t, t), fontsize=11)
            if j == 0:
                ax.set_ylabel(LANG_LABELS.get(s, s), fontsize=11)
                ax.set_yticks(range(n_pos)); ax.set_yticklabels(pos_labels, fontsize=7)
            else:
                ax.set_yticks([])
            if i == len(S) - 1:
                ax.set_xlabel("layer", fontsize=9)
            ax.tick_params(labelsize=7)

    fig.supylabel("Source refusal direction", fontsize=12)
    fig.suptitle("Cosine: source refusal direction  vs  target difference-in-means (pos x layer)",
                 fontsize=12.5, y=1.005)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("cosine similarity", fontsize=10)

    if args.save:
        os.makedirs(osp.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved: {args.save}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
