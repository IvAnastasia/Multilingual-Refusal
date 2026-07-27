"""
PCA of multilingual harmful vs harmless representations at the refusal-extraction layer.

For each pair (English, X) it takes the residual-stream activation at (layer, position)
for harmful and harmless prompts in both languages, fits a 2-D PCA on the four groups
jointly, and scatters them. Encoding: color = harm-type (harmful vs harmless),
marker = language (English = circle, X = triangle).

Runs the model (GPU) to extract activations, caches them to results/pca_acts_<lang>.npz,
then plots. Re-runs reuse the cache (CPU-only) unless --refresh.

Usage:
  python -m scripts.plot_pca_representations --model_path <path> \
      --layer 30 --pos -4 --pairs yo ba be tg --save figures/pca_representations_14b.png
"""
import argparse
import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt

from scripts.viz_style import palette, style, INK, GRID

LANG_LABELS = {"en": "English", "ba": "Bashkir", "be": "Belarusian", "tg": "Tajik", "yo": "Yoruba"}


def extract_lang(model_base, lang, n, layer, pos, batch_size):
    from pipeline.submodules.select_direction import get_raw_activations
    from dataset.load_dataset import load_dataset_split
    out = {}
    for ht in ("harmful", "harmless"):
        instr = load_dataset_split(ht, "train", lang, instructions_only=True)[:n]
        acts = get_raw_activations(model_base.model, model_base.tokenizer, instr,
                                   model_base.tokenize_instructions_fn,
                                   model_base.model_block_modules, batch_size=batch_size, positions=[pos])
        out[ht] = acts[:, layer, :].cpu().float().numpy()  # [n, d_model]
    return out


def pca2_fit(X):
    """Return (mean, components[2,d]) fit on X."""
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--layer", type=int, default=30, help="Refusal-extraction layer (default: English's, 30)")
    ap.add_argument("--pos", type=int, default=-4, help="Token position (default: English's, -4)")
    ap.add_argument("--pairs", nargs="+", default=["yo", "ba", "be", "tg"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--refresh", action="store_true", help="Recompute activations even if cached")
    ap.add_argument("--save", "-s", default=None)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    langs = ["en"] + list(args.pairs)
    reps = {}
    model_base = None
    for lang in langs:
        cache = f"results/pca_acts_{lang}_L{args.layer}_p{args.pos}.npz"
        if osp.isfile(cache) and not args.refresh:
            d = np.load(cache)
            reps[lang] = {"harmful": d["harmful"], "harmless": d["harmless"]}
            print(f"[cache] {lang} <- {cache}")
        else:
            if model_base is None:
                from pipeline.model_utils.model_factory import construct_model_base
                model_base = construct_model_base(args.model_path, "en")
            print(f"[extract] {lang} (layer {args.layer}, pos {args.pos})")
            reps[lang] = extract_lang(model_base, lang, args.n, args.layer, args.pos, args.batch_size)
            np.savez(cache, **reps[lang])

    P = style()
    C_HARMFUL, C_HARMLESS = P["ablation"], P["cat"][0]

    npair = len(args.pairs)
    ncol = 2
    nrow = (npair + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.6 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, X in zip(axes, args.pairs):
        groups = [
            ("en", "harmful", C_HARMFUL, "o", f"English harmful"),
            ("en", "harmless", C_HARMLESS, "o", f"English harmless"),
            (X, "harmful", C_HARMFUL, "^", f"{LANG_LABELS.get(X,X)} harmful"),
            (X, "harmless", C_HARMLESS, "^", f"{LANG_LABELS.get(X,X)} harmless"),
        ]
        allX = np.vstack([reps[lg][ht] for lg, ht, *_ in groups])
        mu, comps = pca2_fit(allX)
        for lg, ht, color, marker, label in groups:
            Y = (reps[lg][ht] - mu) @ comps.T
            ax.scatter(Y[:, 0], Y[:, 1], s=22, c=color, marker=marker, alpha=0.7,
                       edgecolors="white", linewidths=0.3, label=label, zorder=3)
        ax.set_title(f"English vs {LANG_LABELS.get(X, X)}", fontsize=12, pad=6)
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, frameon=False, loc="best")

    for ax in axes[npair:]:
        ax.axis("off")

    fig.suptitle(f"Harmful vs harmless representations — layer {args.layer}, pos {args.pos}",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    if args.save:
        os.makedirs(osp.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved: {args.save}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
