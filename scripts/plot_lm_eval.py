"""
Visualise LM-Eval-Harness benchmark results across languages.

Reads jsonpickle-encoded result files produced by run_pipeline.py's eval_harness step.
Expected files:
  pipeline/runs/<model_alias>/<lang>/lm_eval_results/harm_ablation_mmlu.json   (ablation)
  pipeline/runs/<model_alias>/<lang>/lm_eval_results/harm_ablation.json        (other tasks)
  pipeline/runs/<model_alias>/<lang>/lm_eval_results/baseline_mmlu.json        (baseline, if exists)
  pipeline/runs/<model_alias>/<lang>/lm_eval_results/baseline.json

Produces:
  1. Summary table of benchmark scores per language and condition
  2. Grouped bar chart comparing scores across languages
  3. Optional: baseline vs ablation delta chart

Usage:
  python scripts/plot_lm_eval.py --model_alias Qwen2.5-7B-Instruct
  python scripts/plot_lm_eval.py --model_alias Qwen2.5-7B-Instruct --langs en ba be tg --save figures/lm_eval.pdf
"""
import argparse
import json
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import jsonpickle
except ImportError:
    jsonpickle = None

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

# Standard metric keys produced by lm-eval-harness (in order of preference)
METRIC_KEYS = ["acc,none", "acc_norm,none", "acc", "acc_norm", "exact_match,none"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _lang_dir(model_alias: str, lang: str) -> str:
    if lang == "en":
        return osp.join("pipeline", "runs", model_alias)
    return osp.join("pipeline", "runs", model_alias, lang)


def _load_lm_eval_json(path: str) -> dict | None:
    """Load a jsonpickle-encoded lm-eval result file, returning the raw dict."""
    if not osp.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    # Try jsonpickle first, then plain json
    if jsonpickle is not None:
        try:
            return jsonpickle.decode(raw)
        except Exception:
            pass
    try:
        return json.loads(raw)
    except Exception:
        return None


def _extract_scores(result_dict: dict) -> dict:
    """
    From an lm-eval result dict, extract {task_name: accuracy} for all tasks.
    Handles the standard lm-eval-harness output format.
    """
    if result_dict is None:
        return {}
    results = result_dict.get("results", result_dict)
    if not isinstance(results, dict):
        return {}

    scores = {}
    for task_name, task_results in results.items():
        if not isinstance(task_results, dict):
            continue
        # Skip alias or meta keys
        if task_name in ("all", "config", "versions", "n-samples", "higher_is_better"):
            continue
        # Find the best metric key available
        for key in METRIC_KEYS:
            if key in task_results:
                val = task_results[key]
                if isinstance(val, (int, float)):
                    scores[task_name] = round(float(val), 4)
                    break
    return scores


def load_lm_eval_for_lang(model_alias: str, lang: str) -> dict:
    """
    Load all lm-eval results for a language.
    Returns: {condition: {task: score}} where condition is 'harm_ablation' or 'baseline'.
    """
    d = osp.join(_lang_dir(model_alias, lang), "lm_eval_results")
    if not osp.isdir(d):
        return {}

    results = {}
    for fname in os.listdir(d):
        if not fname.endswith(".json"):
            continue
        path = osp.join(d, fname)
        data = _load_lm_eval_json(path)
        if data is None:
            continue
        scores = _extract_scores(data)
        if not scores:
            continue

        # Determine condition from filename
        base = fname.replace(".json", "")
        if "baseline" in base:
            condition = "baseline"
        elif "harm_ablation" in base:
            condition = "ablation"
        else:
            condition = base

        if condition not in results:
            results[condition] = {}
        results[condition].update(scores)

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_summary_table(all_results: dict, langs: list):
    """Print a formatted table of all results."""
    # Collect all (condition, task) pairs
    all_tasks = set()
    all_conditions = set()
    for lang_results in all_results.values():
        for cond, scores in lang_results.items():
            all_conditions.add(cond)
            all_tasks.update(scores.keys())

    conditions = sorted(all_conditions)
    tasks = sorted(all_tasks)

    if not tasks:
        print("No LM-eval results found.")
        return

    # Header
    col_headers = []
    for cond in conditions:
        for task in tasks:
            col_headers.append(f"{cond}/{task}")

    header = f"{'Lang':<10}"
    for ch in col_headers:
        header += f" {ch:>20}"
    print(header)
    print("-" * len(header))

    for lang in langs:
        row = f"{LANG_LABELS.get(lang, lang):<10}"
        lang_res = all_results.get(lang, {})
        for cond in conditions:
            cond_scores = lang_res.get(cond, {})
            for task in tasks:
                val = cond_scores.get(task)
                if val is not None:
                    row += f" {val:>20.4f}"
                else:
                    row += f" {'—':>20}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_scores_grouped(all_results: dict, langs: list, save_path=None):
    """
    Grouped bar chart: one group per language, one bar per (condition, task).
    """
    # Collect unique conditions and tasks
    all_conditions = set()
    all_tasks = set()
    for lang_res in all_results.values():
        for cond, scores in lang_res.items():
            all_conditions.add(cond)
            all_tasks.update(scores.keys())

    conditions = sorted(all_conditions)
    tasks = sorted(all_tasks)
    bar_labels = [f"{c} — {t}" for c in conditions for t in tasks]
    n_bars = len(bar_labels)

    if n_bars == 0:
        print("No data to plot.")
        return

    present_langs = [l for l in langs if l in all_results and all_results[l]]
    if not present_langs:
        print("No languages with data.")
        return

    x = np.arange(len(present_langs))
    width = 0.8 / max(n_bars, 1)
    colors = plt.cm.Set2(np.linspace(0, 1, n_bars))

    fig, ax = plt.subplots(figsize=(max(8, len(present_langs) * 1.5), 6))

    for bar_idx, (cond, task) in enumerate([(c, t) for c in conditions for t in tasks]):
        vals = []
        for lang in present_langs:
            lang_res = all_results.get(lang, {})
            val = lang_res.get(cond, {}).get(task, 0)
            vals.append(val)

        offset = (bar_idx - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=f"{cond} — {task}",
                       color=colors[bar_idx], edgecolor="white")

        # Add value labels on bars
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABELS.get(l, l) for l in present_langs], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("LM-Eval Benchmark Scores per Language")
    ax.legend(fontsize=7, loc="upper right", ncol=max(1, n_bars // 3))
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(osp.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_ablation_delta(all_results: dict, langs: list, save_path=None):
    """
    Bar chart showing the delta (ablation − baseline) per task per language.
    Only plotted if both baseline and ablation exist for at least one language.
    """
    # Check if any language has both conditions
    has_both = any(
        "baseline" in res and "ablation" in res
        for res in all_results.values()
    )
    if not has_both:
        print("Skipping delta plot: need both 'baseline' and 'ablation' conditions.")
        return

    # Collect tasks that appear in both conditions
    tasks = set()
    for lang_res in all_results.values():
        bl = lang_res.get("baseline", {})
        ab = lang_res.get("ablation", {})
        tasks.update(bl.keys() & ab.keys())
    tasks = sorted(tasks)
    if not tasks:
        return

    present_langs = [
        l for l in langs
        if "baseline" in all_results.get(l, {}) and "ablation" in all_results.get(l, {})
    ]

    x = np.arange(len(present_langs))
    n_tasks = len(tasks)
    width = 0.8 / max(n_tasks, 1)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_tasks))

    fig, ax = plt.subplots(figsize=(max(8, len(present_langs) * 1.2), 5))

    for t_idx, task in enumerate(tasks):
        deltas = []
        for lang in present_langs:
            bl = all_results[lang].get("baseline", {}).get(task, 0)
            ab = all_results[lang].get("ablation", {}).get(task, 0)
            deltas.append(ab - bl)

        offset = (t_idx - n_tasks / 2 + 0.5) * width
        bars = ax.bar(x + offset, deltas, width, label=task,
                       color=colors[t_idx], edgecolor="white")

        for bar, d in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.002 if d >= 0 else -0.012),
                    f"{d:+.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABELS.get(l, l) for l in present_langs], rotation=45, ha="right")
    ax.set_ylabel("Δ Score (ablation − baseline)")
    ax.set_title("Impact of Refusal Ablation on General Benchmarks")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        base, ext = osp.splitext(save_path)
        delta_path = f"{base}_delta{ext}"
        os.makedirs(osp.dirname(delta_path) or ".", exist_ok=True)
        fig.savefig(delta_path, dpi=200, bbox_inches="tight")
        print(f"Saved delta plot to {delta_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualise LM-Eval benchmark results across languages.")
    parser.add_argument("--model_alias", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--langs", nargs="+", default=None,
                        help="Languages to include (default: auto-detect)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save main chart to this path (e.g. figures/lm_eval.pdf)")
    args = parser.parse_args()

    # Auto-detect languages
    if args.langs:
        langs = args.langs
    else:
        base = osp.join("pipeline", "runs", args.model_alias)
        langs = []
        # English (root level)
        if osp.isdir(osp.join(base, "lm_eval_results")):
            langs.append("en")
        # Sub-directories
        if osp.isdir(base):
            for entry in sorted(os.listdir(base)):
                subdir = osp.join(base, entry)
                if osp.isdir(subdir) and osp.isdir(osp.join(subdir, "lm_eval_results")):
                    langs.append(entry)
        if not langs:
            print(f"No lm_eval_results found under {base}/")
            print("Make sure run_pipeline.py completed the eval_harness step.")
            sys.exit(1)
        print(f"Auto-detected languages with lm_eval_results: {langs}")

    # Load results
    all_results = {}
    for lang in langs:
        res = load_lm_eval_for_lang(args.model_alias, lang)
        if res:
            all_results[lang] = res
        else:
            print(f"  No lm_eval_results for {lang}")

    if not all_results:
        print("No results found. Nothing to plot.")
        sys.exit(1)

    # Print table
    print_summary_table(all_results, langs)

    # Plot
    plot_scores_grouped(all_results, langs, save_path=args.save)
    plot_ablation_delta(all_results, langs, save_path=args.save)


if __name__ == "__main__":
    main()
