"""
Cross-lingual safety profile: WildGuard Response-Harmful (ASR) / Refusal / Compliance
per language, for one condition (default: baseline = no steering).

Reads output/<alias>/<lang>/completions/harmful_<cond>_evaluations.json. Languages with
no evaluation file are skipped with a warning (so you can add en/yo once they're run).

Usage:
  python -m scripts.plot_safety_profile -m Qwen2.5-14B-Instruct \
      --langs en ba be tg yo --cond baseline --save figures/safety_profile_14b.png
"""
import argparse
import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

from scripts.viz_style import palette, style, INK, GRID

LANG_LABELS = {"en": "English", "ba": "Bashkir", "be": "Belarusian", "tg": "Tajik",
               "ru": "Russian", "zh": "Chinese", "de": "German", "ja": "Japanese",
               "ko": "Korean", "th": "Thai", "yo": "Yoruba"}
# (metric key, display label)
METRICS = [
    ("wildguard_harmful", "Response Harmful"),
    ("wildguard_refusal", "Refusal Rate"),
    ("wildguard_compliance", "Compliance Rate"),
]


def load_row(base, alias, lang, cond):
    p = osp.join(base, alias, lang, "completions", f"harmful_{cond}_evaluations.json")
    if not osp.isfile(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    return {k: d.get(k) for k, _ in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_alias", "-m", required=True)
    ap.add_argument("--langs", "-l", nargs="+", required=True)
    ap.add_argument("--base_dir", default="output")
    ap.add_argument("--cond", default="baseline")
    ap.add_argument("--save", "-s", default=None)
    args = ap.parse_args()

    _P = style()
    colors = _P["cat"][:3]

    rows, present = {}, []
    for l in args.langs:
        r = load_row(args.base_dir, args.model_alias, l, args.cond)
        if r is None:
            print(f"  [skip] {l}: no {args.cond} evaluation found")
            continue
        rows[l] = r
        present.append(l)

    # table
    print(f"\n{args.cond} WildGuard safety profile:")
    print(f"{'Language':<12}{'Resp.Harmful':<14}{'Refusal':<10}{'Compliance':<12}")
    for l in present:
        r = rows[l]
        print(f"{LANG_LABELS.get(l, l):<12}"
              f"{r['wildguard_harmful']:<14.3f}{r['wildguard_refusal']:<10.3f}{r['wildguard_compliance']:<12.3f}")

    if not present:
        print("Nothing to plot.")
        return

    x = np.arange(len(present))
    w = 0.8 / len(METRICS)
    fig, ax = plt.subplots(figsize=(max(6, len(present) * 1.5), 4.6))
    for mi, (key, label) in enumerate(METRICS):
        vals = [rows[l][key] for l in present]
        off = (mi - (len(METRICS) - 1) / 2) * w
        bars = ax.bar(x + off, vals, w * 0.9, color=colors[mi], label=label, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABELS.get(l, l) for l in present], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("WildGuard rate", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Harmful-prompt response profile ({args.cond}) — {args.model_alias}",
                 fontsize=12.5, y=1.10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.save:
        os.makedirs(osp.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"\nSaved: {args.save}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
