"""
Visualise LM-Eval-Harness capability results across languages (refusal-ablated model).

Reads jsonpickle-encoded result files written by run_pipeline.py's eval_harness step:
  pipeline/runs/<alias>/<lang>/lm_eval_results/<cond>.json        (arc_challenge, truthfulqa_gen, wikitext)
  pipeline/runs/<alias>/<lang>/lm_eval_results/<cond>_mmlu.json    (mmlu; use the top-level average)

The previous version exploded MMLU's ~57 sub-tasks into one series each (a 10k-px legend)
and put perplexity on the same 0-1 axis as accuracy. This version:
  * aggregates MMLU to its single overall accuracy,
  * splits metrics by SCALE into separate panels — accuracy (0-1, higher better) and
    WikiText perplexity (lower better) never share an axis.

Usage:
  python -m scripts.plot_lm_eval --model_alias Qwen2.5-14B-Instruct --langs en ba be tg \
      --save figures/lm_eval_14b.png
"""
from __future__ import annotations

import argparse
import os
import os.path as osp

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np

LANG_LABELS = {"en": "English", "ba": "Bashkir", "be": "Belarusian", "tg": "Tajik",
               "ru": "Russian", "zh": "Chinese", "de": "German", "ja": "Japanese",
               "ko": "Korean", "th": "Thai", "yo": "Yoruba"}
DEFAULT_LANGS = ("en", "ba", "be", "tg")

# (task key, display label, metric key) — accuracy-type, all on a 0-1 scale, higher is better
ACC_TASKS = [
    ("mmlu",           "MMLU (5-shot)",  "acc,none"),
    ("arc_challenge",  "ARC-Challenge",  "acc_norm,none"),
    ("truthfulqa_gen", "TruthfulQA",     "rouge1_acc,none"),
]
# perplexity — different scale, lower is better
PPL_TASK = ("wikitext", "WikiText", "word_perplexity,none")

# shared design system
from scripts.viz_style import palette, style as _style, INK, MUTED, GRID
_P = palette()
TASK_COLORS = _P["cat"]
PPL_COLOR = _P["ppl"]


def _lang_dir(base, alias, lang):
    return osp.join(base, alias) if lang == "en" else osp.join(base, alias, lang)


def _load(path):
    if not osp.isfile(path):
        return None
    obj = jsonpickle.decode(open(path, encoding="utf-8").read())
    return obj.get("results", obj) if isinstance(obj, dict) else obj


def collect(base, alias, langs, cond):
    """Return acc[task_label][lang] and ppl[lang]."""
    acc = {label: {} for _, label, _ in ACC_TASKS}
    ppl = {}
    for lang in langs:
        d = osp.join(_lang_dir(base, alias, lang), "lm_eval_results")
        main = _load(osp.join(d, f"{cond}.json")) or {}
        mmlu = _load(osp.join(d, f"{cond}_mmlu.json")) or {}
        for task, label, metric in ACC_TASKS:
            src = mmlu if task == "mmlu" else main
            if task in src and metric in src[task]:
                acc[label][lang] = float(src[task][metric])
        if PPL_TASK[0] in main and PPL_TASK[2] in main[PPL_TASK[0]]:
            ppl[lang] = float(main[PPL_TASK[0]][PPL_TASK[2]])
    return acc, ppl


def _acc_panel(ax, langs, acc):
    x = np.arange(len(langs))
    labels = [lbl for _, lbl, _ in ACC_TASKS]
    w = 0.8 / max(len(labels), 1)
    for ti, label in enumerate(labels):
        vals = [acc[label].get(l, np.nan) for l in langs]
        off = (ti - (len(labels) - 1) / 2) * w
        bars = ax.bar(x + off, vals, w * 0.9, color=TASK_COLORS[ti % len(TASK_COLORS)],
                      label=label, zorder=3)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABELS.get(l, l) for l in langs], fontsize=10.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Task accuracy  (higher is better)", fontsize=12, pad=8)


def _ppl_panel(ax, langs, ppl):
    x = np.arange(len(langs))
    vals = [ppl.get(l, np.nan) for l in langs]
    bars = ax.bar(x, vals, 0.55, color=PPL_COLOR, zorder=3)
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABELS.get(l, l) for l in langs], fontsize=10.5)
    top = (max([v for v in vals if not np.isnan(v)] or [1])) * 1.18
    ax.set_ylim(0, top)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("WikiText word perplexity", fontsize=11)
    ax.set_title("Fluency  (lower is better)", fontsize=12, pad=8)


def main():
    ap = argparse.ArgumentParser(description="LM-Eval capability plots (split by scale)")
    ap.add_argument("--model_alias", "-m", required=True)
    ap.add_argument("--langs", "-l", nargs="+", default=list(DEFAULT_LANGS))
    ap.add_argument("--base_dir", default=osp.join("pipeline", "runs"))
    ap.add_argument("--cond", default="harm_ablation", help="Condition prefix (default: harm_ablation)")
    ap.add_argument("--no_ppl", action="store_true", help="Drop the WikiText perplexity (fluency) panel; show accuracy only")
    ap.add_argument("--save", "-s", default=None)
    args = ap.parse_args()

    _style()
    acc, ppl = collect(args.base_dir, args.model_alias, args.langs, args.cond)

    print(f"\nAccuracy ({args.cond}):")
    print(f"{'task':<16}" + "".join(f"{LANG_LABELS.get(l, l):<12}" for l in args.langs))
    for _, label, _ in ACC_TASKS:
        print(f"{label:<16}" + "".join(f"{acc[label].get(l, float('nan')):<12.3f}" for l in args.langs))
    print(f"{'WikiText ppl':<16}" + "".join(f"{ppl.get(l, float('nan')):<12.2f}" for l in args.langs))

    if args.no_ppl:
        fig, acc_ax = plt.subplots(1, 1, figsize=(7, 4.6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.7, 1]})
        acc_ax = axes[0]
        _ppl_panel(axes[1], args.langs, ppl)
    _acc_panel(acc_ax, args.langs, acc)
    handles, labels = acc_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Capability retention under refusal ablation — {args.model_alias}",
                 fontsize=13, y=1.09)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if args.save:
        os.makedirs(osp.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"\nSaved: {args.save}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
