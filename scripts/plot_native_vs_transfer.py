"""
Native vs Transfer comparison: for each language, how does ablating/adding the
*English* refusal direction (transfer) compare to the language's *own* direction (native)?

Reads multi_test.py evaluation JSONs:
  transfer: output/<alias>/<lang>/completions/harmful_{baseline,harm_ablation,harm_ablation_addition}_evaluations.json
  native:   output/<alias>/<lang>_native/completions/...

Produces a two-panel figure (Transfer | Native), grouped bars per language,
conditions baseline / ablation / addition, for one WildGuard metric (default: refusal).

Palette (validated, harmonic): baseline = neutral gray, ablation = warm orange,
addition = cool blue — the two interventions read as a warm/cool pair around a neutral control.

Usage:
  python -m scripts.plot_native_vs_transfer --model_alias Qwen2.5-14B-Instruct \
      --langs ba be tg --metric wildguard_refusal --save figures/native_vs_transfer_14b.png
"""
import argparse
import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

LANG_LABELS = {"en": "English", "ba": "Bashkir", "be": "Belarusian", "tg": "Tajik",
               "ru": "Russian", "zh": "Chinese", "de": "German", "ja": "Japanese",
               "ko": "Korean", "th": "Thai", "yo": "Yoruba"}
DEFAULT_LANGS = ("ba", "be", "tg")

CONDITIONS = ["baseline", "ablation", "addition"]
EVAL_FILES = {
    "baseline": "harmful_baseline_evaluations.json",
    "ablation": "harmful_{mode}_evaluations.json",
    "addition": "harmful_{mode}_addition_evaluations.json",
}
METRIC_LABELS = {
    "wildguard_refusal": "WildGuard refusal rate",
    "wildguard_harmful": "WildGuard harmful (ASR)",
    "wildguard_compliance": "WildGuard compliance",
    "substring_matching_success_rate": "Substring-match ASR",
}

# --- shared design system --------------------------------------------------------
from scripts.viz_style import palette, style as _style, INK, MUTED, GRID
_P = palette()
COND_COLOR = {"baseline": _P["baseline"], "ablation": _P["ablation"], "addition": _P["addition"]}


def load_metric(base_dir, alias, lang, suffix, mode, metric):
    """Return {condition: value} for one language/arm, or {} if nothing found."""
    d = osp.join(base_dir, alias, f"{lang}{suffix}", "completions")
    out = {}
    for cond, tmpl in EVAL_FILES.items():
        p = osp.join(d, tmpl.format(mode=mode))
        if osp.isfile(p):
            data = json.load(open(p, encoding="utf-8"))
            if metric in data:
                out[cond] = data[metric]
    return out


def _panel(ax, langs, arm_results, metric, title, show_ylabel):
    x = np.arange(len(langs))
    w = 0.26
    for ci, cond in enumerate(CONDITIONS):
        vals = [arm_results[l].get(cond, np.nan) for l in langs]
        off = (ci - 1) * w
        bars = ax.bar(x + off, vals, w * 0.9, color=COND_COLOR[cond],
                      label=cond.capitalize(), zorder=3)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABELS.get(l, l) for l in langs], fontsize=10.5)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=12, color=INK, pad=8)
    if show_ylabel:
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)


def main():
    ap = argparse.ArgumentParser(description="Native vs Transfer comparison figure")
    ap.add_argument("--model_alias", "-m", required=True)
    ap.add_argument("--langs", "-l", nargs="+", default=list(DEFAULT_LANGS))
    ap.add_argument("--base_dir", default="output")
    ap.add_argument("--mode", default="harm_ablation")
    ap.add_argument("--metric", default="wildguard_refusal", choices=list(METRIC_LABELS))
    ap.add_argument("--save", "-s", default=None)
    args = ap.parse_args()

    _style()
    transfer = {l: load_metric(args.base_dir, args.model_alias, l, "", args.mode, args.metric) for l in args.langs}
    native = {l: load_metric(args.base_dir, args.model_alias, l, "_native", args.mode, args.metric) for l in args.langs}

    # summary table
    print(f"\n{args.metric}:")
    print(f"{'lang':<10}{'arm':<10}" + "".join(f"{c:<11}" for c in CONDITIONS))
    for l in args.langs:
        for arm, res in (("transfer", transfer), ("native", native)):
            row = "".join(f"{res[l].get(c, float('nan')):<11.3f}" for c in CONDITIONS)
            print(f"{LANG_LABELS.get(l, l):<10}{arm:<10}{row}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    _panel(axes[0], args.langs, transfer, args.metric, "Transfer  (English direction)", True)
    _panel(axes[1], args.langs, native, args.metric, "Native  (own-language direction)", False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{METRIC_LABELS.get(args.metric, args.metric)} — steering by refusal direction",
                 fontsize=13, y=1.10)
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
