"""
Visualise selected refusal direction layers across languages.

Produces:
  1. Summary table of selected (position, layer) per language
  2. Heatmap of refusal scores across layers and positions (per language)
  3. Bar chart comparing selected layers across languages

Usage:
  python scripts/plot_selected_layers.py --model_alias Qwen2.5-7B-Instruct
  python scripts/plot_selected_layers.py --model_alias Qwen2.5-7B-Instruct --langs en ba be tg de ru zh ja ko th yo
  python scripts/plot_selected_layers.py --model_alias Qwen2.5-7B-Instruct --save figures/selected_layers.pdf
"""
import argparse
import json
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Language display labels (reused across plotting scripts)
# ---------------------------------------------------------------------------
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

DEFAULT_LANGS = ["en", "de", "ru", "zh", "ja", "ko", "th", "yo", "ba", "be", "tg"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _lang_dir(model_alias: str, lang: str) -> str:
    """Return the run directory for a language."""
    if lang == "en":
        return osp.join("pipeline", "runs", model_alias)
    return osp.join("pipeline", "runs", model_alias, lang)


def load_metadata(model_alias: str, lang: str):
    """Load direction_metadata_ablation.json → (position, layer) or None."""
    d = _lang_dir(model_alias, lang)
    path = osp.join(d, "direction_metadata_ablation.json")
    if not osp.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    return meta["pos"][0], meta["layer"][0]


def load_all_scores(model_alias: str, lang: str):
    """Load direction_evaluations_ablation.json → list[dict] or None."""
    d = _lang_dir(model_alias, lang)
    path = osp.join(d, "select_direction", "direction_evaluations_ablation.json")
    if not osp.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_filtered_scores(model_alias: str, lang: str):
    """Load direction_evaluations_filtered_ablation.json → list[dict] or None."""
    d = _lang_dir(model_alias, lang)
    path = osp.join(d, "select_direction", "direction_evaluations_filtered_ablation.json")
    if not osp.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plotting: selected layer bar chart
# ---------------------------------------------------------------------------
def plot_selected_layers_bar(langs, metadata, save_path=None):
    """Bar chart showing the selected layer for each language."""
    present = [(l, metadata[l]) for l in langs if metadata.get(l) is not None]
    if not present:
        print("No metadata found for any language.")
        return

    labels = [LANG_LABELS.get(l, l) for l, _ in present]
    layers = [m[1] for _, m in present]
    positions = [m[0] for _, m in present]

    fig, ax = plt.subplots(figsize=(max(6, len(present) * 0.9), 5))
    x = np.arange(len(present))
    bars = ax.bar(x, layers, color="steelblue", edgecolor="white", width=0.6)

    # Annotate each bar with layer and position
    for i, (bar, layer, pos) in enumerate(zip(bars, layers, positions)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"L{layer}\npos {pos}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Selected Layer")
    ax.set_title("Selected Refusal Direction Layer per Language")
    ax.set_ylim(0, max(layers) + 4)
    fig.tight_layout()

    if save_path:
        os.makedirs(osp.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved bar chart to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plotting: score heatmap per language (refusal score across layer × position)
# ---------------------------------------------------------------------------
def plot_score_heatmaps(langs, model_alias, metric="refusal_score", save_dir=None):
    """
    For each language with data, plot a heatmap of `metric` across (position, layer).
    Also plots a combined figure with one subplot per language.
    """
    lang_data = {}
    for lang in langs:
        scores = load_all_scores(model_alias, lang)
        if scores is None:
            print(f"  Heatmap: skip {lang} (no score data)")
            continue
        lang_data[lang] = scores

    if not lang_data:
        print("No score data found.")
        return
    print(f"  Heatmap: plotting for {list(lang_data.keys())}")

    # Determine grid dimensions from the first language
    sample = list(lang_data.values())[0]
    positions = sorted(set(e["position"] for e in sample))
    layers = sorted(set(e["layer"] for e in sample))
    pos_idx = {p: i for i, p in enumerate(positions)}
    lay_idx = {l: i for i, l in enumerate(layers)}

    n_langs = len(lang_data)
    ncols = min(4, n_langs)
    nrows = (n_langs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, (lang, scores) in enumerate(lang_data.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        mat = np.full((len(positions), len(layers)), np.nan)
        for e in scores:
            pi = pos_idx.get(e["position"])
            li = lay_idx.get(e["layer"])
            if pi is not None and li is not None:
                mat[pi, li] = e[metric]

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
        ax.set_title(f"{LANG_LABELS.get(lang, lang)} ({lang})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Position")
        ax.set_xticks(np.arange(0, len(layers), max(1, len(layers) // 8)))
        ax.set_xticklabels([layers[i] for i in range(0, len(layers), max(1, len(layers) // 8))], fontsize=7)
        ax.set_yticks(np.arange(len(positions)))
        ax.set_yticklabels(positions, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for idx in range(n_langs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    metric_label = metric.replace("_", " ").title()
    fig.suptitle(f"{metric_label} Heatmap (Layer × Position) per Language", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = osp.join(save_dir, f"score_heatmap_{metric}.pdf")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved heatmap to {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plotting: best refusal score per layer (line plot, all languages overlaid)
# ---------------------------------------------------------------------------
def plot_best_score_per_layer(langs, model_alias, metric="refusal_score", save_path=None):
    """
    For each language, compute the best (lowest for ablation) refusal_score at each layer
    across all positions, then overlay them on a single line plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for lang in langs:
        scores = load_all_scores(model_alias, lang)
        if scores is None:
            continue
        layers_set = sorted(set(e["layer"] for e in scores))
        best_per_layer = {}
        for e in scores:
            l = e["layer"]
            v = e[metric]
            if l not in best_per_layer or v < best_per_layer[l]:
                best_per_layer[l] = v
        xs = sorted(best_per_layer.keys())
        ys = [best_per_layer[l] for l in xs]
        label = LANG_LABELS.get(lang, lang)
        ax.plot(xs, ys, marker=".", markersize=3, label=label, alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Best {metric.replace('_', ' ').title()} per Layer (lower = better ablation)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(osp.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved line plot to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualise selected refusal direction layers across languages.")
    parser.add_argument("--model_alias", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--langs", nargs="+", default=None,
                        help="Languages to include (default: auto-detect from pipeline/runs/)")
    parser.add_argument("--metric", type=str, default="refusal_score",
                        choices=["refusal_score", "steering_score", "kl_div_score"],
                        help="Metric for heatmaps and line plot")
    parser.add_argument("--save", type=str, default=None,
                        help="Save bar chart to this path (e.g. figures/layers.pdf)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save heatmaps and line plots")
    args = parser.parse_args()

    # Auto-detect languages if not specified
    if args.langs:
        langs = args.langs
    else:
        base = osp.join("pipeline", "runs", args.model_alias)
        langs = []
        # Check for English (direction in root)
        if osp.isfile(osp.join(base, "direction_metadata_ablation.json")):
            langs.append("en")
        # Check subdirectories
        if osp.isdir(base):
            for entry in sorted(os.listdir(base)):
                subdir = osp.join(base, entry)
                if osp.isdir(subdir) and osp.isfile(osp.join(subdir, "direction_metadata_ablation.json")):
                    langs.append(entry)
        if not langs:
            print(f"No direction metadata found under {base}/")
            sys.exit(1)
        print(f"Auto-detected languages: {langs}")

    print(f"Using languages: {langs}")

    # Load metadata for all languages
    metadata = {}
    for lang in langs:
        m = load_metadata(args.model_alias, lang)
        if m is not None:
            metadata[lang] = m

    # Print summary table
    print(f"\n{'Lang':<8} {'Label':<15} {'Position':<10} {'Layer':<6}")
    print("-" * 42)
    for lang in langs:
        if lang in metadata:
            pos, layer = metadata[lang]
            print(f"{lang:<8} {LANG_LABELS.get(lang, lang):<15} {pos:<10} {layer:<6}")
        else:
            print(f"{lang:<8} {LANG_LABELS.get(lang, lang):<15} {'N/A':<10} {'N/A':<6}")

    # Plot 1: bar chart of selected layers
    plot_selected_layers_bar(langs, metadata, save_path=args.save)

    # Plot 2: score heatmap per language
    plot_score_heatmaps(langs, args.model_alias, metric=args.metric, save_dir=args.save_dir)

    # Plot 3: best score per layer overlay
    line_path = None
    if args.save_dir:
        line_path = osp.join(args.save_dir, f"best_{args.metric}_per_layer.pdf")
    plot_best_score_per_layer(langs, args.model_alias, metric=args.metric, save_path=line_path)


if __name__ == "__main__":
    main()
