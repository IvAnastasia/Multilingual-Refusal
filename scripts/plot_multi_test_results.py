"""
Visualise multi_test.py results: compare baseline vs ablation vs addition across languages.

Reads evaluation JSONs from output/<model_alias>/<lang>/completions/ and produces:
  1. Grouped bar chart: ASR / refusal / compliance per condition per language
  2. Summary table printed to terminal

Usage:
  # Default: look for ba, be, tg under output/Qwen2.5-7B-Instruct/
  python scripts/plot_multi_test_results.py --model_alias Qwen2.5-7B-Instruct

  # Also include multi_inference baseline (from a different output dir):
  python scripts/plot_multi_test_results.py --model_alias Qwen2.5-7B-Instruct --include_inference

  # Custom languages and save:
  python scripts/plot_multi_test_results.py --model_alias Qwen2.5-7B-Instruct --langs ba be tg --save figures/multi_test.png

  # Specify a custom base directory:
  python scripts/plot_multi_test_results.py --model_alias Qwen2.5-7B-Instruct --base_dir output
"""
import argparse
import json
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np

LANG_LABELS = {
    "en": "English", "ba": "Bashkir", "be": "Belarusian", "tg": "Tajik",
    "ru": "Russian", "zh": "Chinese", "de": "German", "ja": "Japanese",
    "ko": "Korean", "th": "Thai", "yo": "Yoruba",
}
DEFAULT_LANGS = ("ba", "be", "tg")
DEFAULT_MODE = "harm_ablation"

# Evaluation JSON files produced by multi_test.py
EVAL_FILES = {
    "baseline":  "harmful_baseline_evaluations.json",
    "ablation":  "harmful_{mode}_evaluations.json",
    "addition":  "harmful_{mode}_addition_evaluations.json",
}

# Metrics to extract (key in JSON → display label)
METRICS = {
    "substring_matching_success_rate": "Substring ASR",
    "wildguard_harmful": "WildGuard ASR",
    "wildguard_refusal": "WildGuard Refusal",
    "wildguard_compliance": "WildGuard Compliance",
}

# For multi_inference baseline (run_wildguard_eval output)
INFERENCE_EVAL_FILE = "evaluation_wildguard.json"


def load_eval(path: str) -> dict:
    """Load evaluation JSON and return metric dict, or None if missing."""
    if not osp.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_results(base_dir: str, model_alias: str, langs: list, mode: str, include_inference: bool):
    """
    Collect metrics for each language and condition.
    Returns: {lang: {condition: {metric: value}}}
    """
    results = {}
    for lang in langs:
        results[lang] = {}

        # multi_test evaluations
        completions_dir = osp.join(base_dir, model_alias, lang, "completions")
        for cond_key, fname_template in EVAL_FILES.items():
            fname = fname_template.format(mode=mode)
            path = osp.join(completions_dir, fname)
            data = load_eval(path)
            if data is None:
                print(f"  [{lang}] {cond_key}: not found ({path})")
                continue
            metrics = {}
            for metric_key, label in METRICS.items():
                if metric_key in data:
                    metrics[label] = data[metric_key]
            results[lang][cond_key] = metrics

        # multi_inference baseline (optional)
        if include_inference:
            inf_path = osp.join("output", "multi_inference", model_alias, lang, "harmful", INFERENCE_EVAL_FILE)
            data = load_eval(inf_path)
            if data is not None:
                metrics = {}
                for metric_key, label in METRICS.items():
                    if metric_key in data:
                        metrics[label] = data[metric_key]
                results[lang]["inference_baseline"] = metrics
            else:
                print(f"  [{lang}] inference_baseline: not found ({inf_path})")

    return results


def print_summary(results: dict):
    """Print a summary table to terminal."""
    print("\n" + "=" * 80)
    print("MULTI-TEST RESULTS SUMMARY")
    print("=" * 80)
    for lang, conditions in results.items():
        label = LANG_LABELS.get(lang, lang)
        print(f"\n--- {label} ({lang}) ---")
        for cond, metrics in conditions.items():
            print(f"  {cond}:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value:.4f}")
    print("=" * 80)


def plot_grouped_bars(results: dict, save_path=None):
    """
    Grouped bar chart: for each metric, group by language, bars = conditions.
    Produces one figure per metric.
    """
    # Collect all conditions and metrics present
    all_conditions = set()
    all_metrics = set()
    langs = list(results.keys())
    for lang in langs:
        for cond, metrics in results[lang].items():
            all_conditions.add(cond)
            all_metrics.update(metrics.keys())

    conditions = sorted(all_conditions)
    cond_colors = {
        "baseline": "#4c72b0",
        "ablation": "#dd8452",
        "addition": "#55a868",
        "inference_baseline": "#8172b3",
    }

    for metric_name in sorted(all_metrics):
        fig, ax = plt.subplots(figsize=(max(6, len(langs) * 2.5), 5))

        x = np.arange(len(langs))
        width = 0.8 / max(len(conditions), 1)

        for ci, cond in enumerate(conditions):
            values = []
            for lang in langs:
                val = results[lang].get(cond, {}).get(metric_name, None)
                values.append(val if val is not None else 0)

            offset = (ci - len(conditions) / 2 + 0.5) * width
            color = cond_colors.get(cond, f"C{ci}")
            bars = ax.bar(x + offset, values, width * 0.9, label=cond.replace("_", " ").title(),
                          color=color, edgecolor="black", linewidth=0.5)

            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        lang_labels = [LANG_LABELS.get(l, l) for l in langs]
        ax.set_xticks(x)
        ax.set_xticklabels(lang_labels, fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f"{metric_name} — Baseline vs Ablation vs Addition", fontsize=13)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10, loc="upper right")
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)
        fig.tight_layout()

        if save_path:
            base, ext = osp.splitext(save_path)
            metric_slug = metric_name.lower().replace(" ", "_")
            fig_path = f"{base}_{metric_slug}{ext}"
            os.makedirs(osp.dirname(fig_path) or ".", exist_ok=True)
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {fig_path}")
        else:
            plt.show()
        plt.close(fig)


def plot_combined(results: dict, save_path=None):
    """
    Single figure with subplots: one per metric, grouped bars by language.
    """
    all_conditions = set()
    all_metrics = set()
    langs = list(results.keys())
    for lang in langs:
        for cond, metrics in results[lang].items():
            all_conditions.add(cond)
            all_metrics.update(metrics.keys())

    conditions = sorted(all_conditions)
    metrics_list = sorted(all_metrics)
    n_metrics = len(metrics_list)

    if n_metrics == 0:
        print("No metrics found to plot.")
        return

    cond_colors = {
        "baseline": "#4c72b0",
        "ablation": "#dd8452",
        "addition": "#55a868",
        "inference_baseline": "#8172b3",
    }

    fig, axes = plt.subplots(1, n_metrics, figsize=(max(5, n_metrics * 4), 5), sharey=True)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metrics_list):
        x = np.arange(len(langs))
        width = 0.8 / max(len(conditions), 1)

        for ci, cond in enumerate(conditions):
            values = []
            for lang in langs:
                val = results[lang].get(cond, {}).get(metric_name, None)
                values.append(val if val is not None else 0)
            offset = (ci - len(conditions) / 2 + 0.5) * width
            color = cond_colors.get(cond, f"C{ci}")
            bars = ax.bar(x + offset, values, width * 0.9, label=cond.replace("_", " ").title(),
                          color=color, edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        lang_labels = [LANG_LABELS.get(l, l) for l in langs]
        ax.set_xticks(x)
        ax.set_xticklabels(lang_labels, fontsize=10, rotation=30, ha="right")
        ax.set_title(metric_name, fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)

    axes[0].set_ylabel("Score", fontsize=12)
    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(conditions), fontsize=10,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Multi-Test Results: Baseline vs Ablation vs Addition", fontsize=14, y=1.08)
    fig.tight_layout()

    if save_path:
        os.makedirs(osp.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Combined figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualise multi_test results across languages")
    parser.add_argument("--model_alias", "-m", required=True, help="Model alias (e.g. Qwen2.5-7B-Instruct)")
    parser.add_argument("--langs", "-l", nargs="+", default=list(DEFAULT_LANGS), help="Language codes")
    parser.add_argument("--base_dir", default="output", help="Base output dir (default: output)")
    parser.add_argument("--mode", default=DEFAULT_MODE, help="Intervention mode label (default: harm_ablation)")
    parser.add_argument("--include_inference", action="store_true", help="Also include multi_inference baseline (from output/multi_inference/)")
    parser.add_argument("--save", "-s", default=None, help="Save figures (e.g. figures/multi_test.png)")
    parser.add_argument("--separate", action="store_true", help="Save one figure per metric instead of combined")
    args = parser.parse_args()

    results = collect_results(args.base_dir, args.model_alias, args.langs, args.mode, args.include_inference)
    print_summary(results)

    if args.separate:
        plot_grouped_bars(results, save_path=args.save)
    else:
        plot_combined(results, save_path=args.save)


if __name__ == "__main__":
    main()
