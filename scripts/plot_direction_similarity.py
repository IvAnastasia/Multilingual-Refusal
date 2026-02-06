"""
Visualise pairwise cosine similarity of refusal direction vectors across languages.

Produces:
  1. Annotated heatmap of the cosine similarity matrix
  2. (Optional) Bar chart of each language's similarity to English

Usage:
  # From direction files (computes similarity on the fly):
  python scripts/plot_direction_similarity.py --model_alias Qwen2.5-7B-Instruct

  # From a pre-computed JSON (output of direction_similarity.py --out):
  python scripts/plot_direction_similarity.py --json results/direction_similarity.json

  # Customise languages and output path:
  python scripts/plot_direction_similarity.py --model_alias Qwen2.5-7B-Instruct --langs en ba be tg --save figures/similarity.pdf
"""
import argparse
import json
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Reuse helpers from direction_similarity.py
SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from direction_similarity import _direction_path, load_direction_vector, cosine_similarity

LANG_LABELS = {
    "en": "English",
    "ba": "Bashkir",
    "be": "Belarusian",
    "tg": "Tajik",
    "ru": "Russian",
    "zh": "Chinese",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "th": "Thai",
    "yo": "Yoruba",
}

DEFAULT_LANGS = ("en", "ba", "be", "tg")


def compute_matrix(model_alias: str, langs: list, prefer_native: bool = False):
    """Load direction vectors and return (langs_found, similarity_matrix)."""
    vectors = {}
    for lang in langs:
        path = _direction_path(model_alias, lang, prefer_native=prefer_native)
        if path is None:
            print(f"  Skip {lang}: no direction file found")
            continue
        vectors[lang] = load_direction_vector(path)
        print(f"  Loaded {lang}: {path}")

    if len(vectors) < 2:
        raise RuntimeError("Need at least 2 direction files.")

    found = list(vectors.keys())
    n = len(found)
    mat = np.zeros((n, n))
    for i, la in enumerate(found):
        for j, lb in enumerate(found):
            mat[i, j] = cosine_similarity(vectors[la], vectors[lb])
    return found, mat


def load_matrix_from_json(json_path: str):
    """Load a pre-computed similarity JSON (from direction_similarity.py --out)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    langs = data["langs"]
    n = len(langs)
    mat = np.zeros((n, n))
    for i, la in enumerate(langs):
        for j, lb in enumerate(langs):
            mat[i, j] = data["cosine_similarity"][la][lb]
    return langs, mat


def plot_heatmap(langs, matrix, save_path=None):
    """Plot an annotated heatmap of the cosine similarity matrix."""
    labels = [LANG_LABELS.get(l, l) for l in langs]
    n = len(langs)

    fig, ax = plt.subplots(figsize=(max(5, n * 1.2), max(4, n * 1.0)))

    # Color map: blues, with 1.0 = darkest
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")

    # Ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=12, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=12)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val > 0.75 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=11, color=color, fontweight="bold")

    ax.set_title("Refusal Direction Cosine Similarity", fontsize=14, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")
    fig.tight_layout()

    if save_path:
        os.makedirs(osp.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_similarity_to_english(langs, matrix, save_path=None):
    """Bar chart: similarity of each non-English language to English."""
    if "en" not in langs:
        print("No English direction found; skipping bar chart.")
        return

    en_idx = langs.index("en")
    other_langs = [l for l in langs if l != "en"]
    other_indices = [langs.index(l) for l in other_langs]
    sims = [matrix[en_idx, j] for j in other_indices]
    labels = [LANG_LABELS.get(l, l) for l in other_langs]

    fig, ax = plt.subplots(figsize=(max(4, len(other_langs) * 1.5), 4))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(other_langs)))
    bars = ax.bar(labels, sims, color=colors, edgecolor="black", linewidth=0.5)

    for bar, sim in zip(bars, sims):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{sim:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Cosine Similarity to English", fontsize=12)
    ax.set_title("Refusal Direction Similarity to English", fontsize=14)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.tight_layout()

    if save_path:
        bar_path = save_path.rsplit(".", 1)
        bar_path = f"{bar_path[0]}_bars.{bar_path[1]}" if len(bar_path) == 2 else f"{save_path}_bars.png"
        os.makedirs(osp.dirname(bar_path) or ".", exist_ok=True)
        fig.savefig(bar_path, dpi=150, bbox_inches="tight")
        print(f"Bar chart saved to {bar_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualise refusal direction similarity")
    parser.add_argument("--model_alias", "-m", default=None, help="Model alias (e.g. Qwen2.5-7B-Instruct). Computes similarity from direction files.")
    parser.add_argument("--json", "-j", default=None, help="Pre-computed similarity JSON (from direction_similarity.py --out)")
    parser.add_argument("--langs", "-l", nargs="+", default=list(DEFAULT_LANGS), help="Language codes to compare")
    parser.add_argument("--save", "-s", default=None, help="Save figures to this path (e.g. figures/similarity.png or .pdf)")
    parser.add_argument("--native", action="store_true",
                        help="Prefer direction_ablation.pt (language-specific) over direction.pt (copied English)")
    parser.add_argument("--no_bars", action="store_true", help="Skip the bar chart")
    args = parser.parse_args()

    if args.json:
        langs, matrix = load_matrix_from_json(args.json)
    elif args.model_alias:
        langs, matrix = compute_matrix(args.model_alias, args.langs, prefer_native=args.native)
    else:
        parser.error("Provide --model_alias or --json")

    print(f"\nLanguages: {langs}")
    print("Similarity matrix:")
    print(matrix.round(4))

    plot_heatmap(langs, matrix, save_path=args.save)
    if not args.no_bars:
        plot_similarity_to_english(langs, matrix, save_path=args.save)


if __name__ == "__main__":
    main()
