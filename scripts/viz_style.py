"""
Shared visual style for all figures — one place to control palette + chrome so every
plot reads as one system. Switch the whole look by changing ACTIVE (or the VIZ_PALETTE
env var), and every figure updates.

Roles:
  baseline / ablation / addition  -> the three steering conditions (baseline is the neutral control)
  cat[0..]                        -> categorical series (e.g. lm-eval tasks)
  ppl                             -> single-series accent (perplexity)
  diverging (neg, mid, pos)       -> similarity heatmap poles + neutral midpoint
  ink / muted / grid              -> text and chart chrome (shared across variants)
"""
import os

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

PALETTES = {
    # 1) vivid — the validated default (high-contrast, saturated)
    "vivid": {
        "baseline": "#898781", "ablation": "#eb6834", "addition": "#2a78d6",
        "cat": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "ppl": "#4a3aa7",
        "diverging": ("#c0392b", "#f0efec", "#2a78d6"),
    },
    # 2) muted — desaturated, print/paper friendly, gentler contrast (terracotta / steel)
    "muted": {
        "baseline": "#9a9a93", "ablation": "#c96a4a", "addition": "#3f6fa3",
        "cat": ["#3f6fa3", "#c96a4a", "#5b9279", "#c79a3e"],
        "ppl": "#6a5a8c",
        "diverging": ("#b5563f", "#eeece6", "#3f6fa3"),
    },
    # 2a) muted_teal — coral vs teal
    "muted_teal": {
        "baseline": "#9a9a93", "ablation": "#d08770", "addition": "#4c8c8c",
        "cat": ["#4c8c8c", "#d08770", "#7a9c5f", "#c9a24a"],
        "ppl": "#6a5a8c",
        "diverging": ("#c0674a", "#eeeae3", "#3f7f7f"),
    },
    # 2b) muted_amber — amber vs indigo
    "muted_amber": {
        "baseline": "#9a9a93", "ablation": "#cc9544", "addition": "#4b5b96",
        "cat": ["#4b5b96", "#cc9544", "#6a8f6a", "#b5654a"],
        "ppl": "#7a6a95",
        "diverging": ("#c0894a", "#eeeae3", "#4b5b96"),
    },
    # 2c) muted_rose — dusty rose vs slate blue
    "muted_rose": {
        "baseline": "#9c9a94", "ablation": "#bd6f7a", "addition": "#5170a3",
        "cat": ["#5170a3", "#bd6f7a", "#6a9c8a", "#c9a24a"],
        "ppl": "#8a6a8c",
        "diverging": ("#b56776", "#eeece6", "#5170a3"),
    },
    # 2d) vivid_rose — saturated rose vs blue (vivid contrast, rose warm pole)
    "vivid_rose": {
        "baseline": "#898781", "ablation": "#d6446e", "addition": "#2a78d6",
        "cat": ["#2a78d6", "#d6446e", "#1baf7a", "#eda100"],
        "ppl": "#7a4ba7",
        "diverging": ("#d6446e", "#f0efec", "#2a78d6"),
    },
    # --- more muted candidates (distinct hue pairs) ---
    # sienna vs denim blue
    "muted_sienna": {
        "baseline": "#9a9a93", "ablation": "#b0673f", "addition": "#45688f",
        "cat": ["#45688f", "#b0673f", "#5f8f6a", "#bf9a4a"],
        "ppl": "#6f5f8a", "diverging": ("#b0673f", "#eeeae3", "#45688f"),
    },
    # berry/raspberry vs pine green (magenta-green pairing)
    "muted_berry": {
        "baseline": "#9a9a93", "ablation": "#a8465f", "addition": "#3f7f6a",
        "cat": ["#3f7f6a", "#a8465f", "#4a6d99", "#bf9a4a"],
        "ppl": "#7a5a85", "diverging": ("#a8465f", "#eeeae3", "#3f7f6a"),
    },
    # soft pumpkin vs petrol teal-blue
    "muted_pumpkin": {
        "baseline": "#9a9a93", "ablation": "#cf8a4a", "addition": "#2f6d80",
        "cat": ["#2f6d80", "#cf8a4a", "#6a9c6a", "#b5654a"],
        "ppl": "#6f5f8a", "diverging": ("#cf8a4a", "#eeeae3", "#2f6d80"),
    },
    # muted plum-pink vs slate blue
    "muted_slateplum": {
        "baseline": "#9a9a93", "ablation": "#9c5a72", "addition": "#556b8c",
        "cat": ["#556b8c", "#9c5a72", "#6a9c8a", "#bf9a4a"],
        "ppl": "#7a6a95", "diverging": ("#9c5a72", "#eeece6", "#556b8c"),
    },
    # 3) okabe_ito — the scientific colorblind-safe standard (Okabe & Ito 2008)
    "okabe_ito": {
        "baseline": "#999999", "ablation": "#d55e00", "addition": "#0072b2",
        "cat": ["#0072b2", "#d55e00", "#009e73", "#e69f00"],
        "ppl": "#cc79a7",
        "diverging": ("#d55e00", "#f0f0f0", "#0072b2"),
    },
}

ACTIVE = os.environ.get("VIZ_PALETTE", "muted_berry")


def palette(name: str = None) -> dict:
    return PALETTES[name or ACTIVE]


def style(name: str = None):
    """Apply shared matplotlib rcParams (chrome is palette-independent)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.labelcolor": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "figure.dpi": 110,
    })
    return palette(name)


def _preview(save="figures/palette_variants.png", names=None):
    """Render a side-by-side comparison of variants: condition swatches,
    a mini grouped-bar sample, and the diverging ramp."""
    import os as _os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    style()
    names = names or list(PALETTES)
    fig, axes = plt.subplots(len(names), 3, figsize=(12, 2.5 * len(names)),
                             gridspec_kw={"width_ratios": [1.1, 1.4, 1.1]})
    rng = [0.75, 0.05, 0.95]  # fake baseline/ablation/addition sample
    for r, nm in enumerate(names):
        p = PALETTES[nm]
        # col 0: condition swatches
        ax = axes[r][0]
        for i, (role, c) in enumerate([("baseline", p["baseline"]), ("ablation", p["ablation"]), ("addition", p["addition"])]):
            ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, color=c))
            ax.text(i + 0.45, -0.25, role, ha="center", va="top", fontsize=9, color=INK)
        ax.set_xlim(-0.1, 3); ax.set_ylim(-0.6, 1.1); ax.axis("off")
        ax.set_title(f"{nm}", fontsize=13, loc="left", color=INK, fontweight="bold")
        # col 1: mini grouped bars (3 conditions x 2 groups)
        ax = axes[r][1]
        x = np.arange(2)
        for i, (role, c) in enumerate([("baseline", p["baseline"]), ("ablation", p["ablation"]), ("addition", p["addition"])]):
            ax.bar(x + (i - 1) * 0.27, [rng[i], 1 - rng[i]], 0.25, color=c, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(["lang A", "lang B"], fontsize=9)
        ax.set_ylim(0, 1.1); ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.set_title("sample bars", fontsize=10)
        # col 2: diverging ramp
        ax = axes[r][2]
        cmap = LinearSegmentedColormap.from_list("d", list(p["diverging"]))
        grad = np.linspace(-1, 1, 256).reshape(1, -1)
        ax.imshow(grad, aspect="auto", cmap=cmap, extent=[-1, 1, 0, 1])
        ax.set_yticks([]); ax.set_xticks([-1, 0, 1]); ax.set_xticklabels(["-1", "0", "+1"], fontsize=9)
        ax.set_title("diverging (heatmap)", fontsize=10)
    fig.suptitle("Palette variants — pick one for all figures", fontsize=14, y=1.01)
    fig.tight_layout()
    _os.makedirs(_os.path.dirname(save) or ".", exist_ok=True)
    fig.savefig(save, dpi=150, bbox_inches="tight")
    print(f"Saved: {save}")


if __name__ == "__main__":
    _preview()
