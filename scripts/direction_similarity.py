"""
Compute pairwise cosine similarity between refusal/steering direction vectors across languages.

Expects direction files under pipeline/runs/<model_alias>/<lang>/ (or pipeline/runs/<model_alias>/ for en).
Run after you have direction_ablation.pt (or direction.pt) for each language, e.g. from run_pipeline.py
with --source_lang ba/be/tg/en.

Usage:
  python scripts/direction_similarity.py --model_alias Qwen2.5-7B-Instruct
  python scripts/direction_similarity.py --model_alias Qwen2.5-7B-Instruct --langs en ba be tg --out similarity.json
"""
import argparse
import json
import os.path as osp
import torch

RUNS_BASE = "pipeline/runs"
DEFAULT_LANGS = ("en", "ba", "be", "tg")


def _direction_path(model_alias: str, lang: str, prefer_native: bool = False):
    """Return path to direction file for a language, or None if missing.

    prefer_native=False (default): tries direction.pt first, then direction_ablation.pt.
        Good for transfer experiments (direction.pt is the copied English vector).
    prefer_native=True: tries direction_ablation.pt first, then direction.pt.
        Good for similarity comparison (direction_ablation.pt is computed from the language's own data).
    """
    if lang == "en":
        base = osp.join(RUNS_BASE, model_alias)
    else:
        base = osp.join(RUNS_BASE, model_alias, lang)
    if prefer_native:
        order = ("direction_ablation.pt", "direction.pt")
    else:
        order = ("direction.pt", "direction_ablation.pt")
    for name in order:
        p = osp.join(base, name)
        if osp.isfile(p):
            return p
    return None


def load_direction_vector(path: str) -> torch.Tensor:
    """Load direction file and return a 1D float tensor (L2-normalized)."""
    d = torch.load(path, map_location="cpu")
    if isinstance(d, list):
        d = d[0]
    v = d.float().flatten()
    v = v / (v.norm() + 1e-8)
    return v


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1D tensors (assumed already unit norm)."""
    return (a * b).sum().item()


def main():
    parser = argparse.ArgumentParser(description="Refusal direction similarity across languages")
    parser.add_argument("--model_alias", "-m", required=True, help="Model alias, e.g. Qwen2.5-7B-Instruct")
    parser.add_argument("--langs", "-l", nargs="+", default=list(DEFAULT_LANGS), help="Language codes to compare")
    parser.add_argument("--native", action="store_true",
                        help="Prefer direction_ablation.pt (language-specific) over direction.pt (copied English). "
                             "Use this to compare truly language-specific directions.")
    parser.add_argument("--out", "-o", default=None, help="Optional JSON path to save similarity matrix")
    args = parser.parse_args()

    # Load vectors only for langs that have a direction file
    vectors = {}
    for lang in args.langs:
        path = _direction_path(args.model_alias, lang, prefer_native=args.native)
        if path is None:
            print(f"  Skip {lang}: no direction.pt / direction_ablation.pt found")
            continue
        vectors[lang] = load_direction_vector(path)
        print(f"  Loaded {lang}: {path} (dim={vectors[lang].shape[0]})")

    if len(vectors) < 2:
        print("Need at least 2 directions. Run pipeline with --source_lang for each language.")
        return

    langs = list(vectors.keys())
    n = len(langs)
    matrix = [[0.0] * n for _ in range(n)]
    for i, la in enumerate(langs):
        for j, lb in enumerate(langs):
            matrix[i][j] = round(cosine_similarity(vectors[la], vectors[lb]), 4)

    print("\nCosine similarity (refusal directions):")
    print("     " + " ".join(f"{lb:>6}" for lb in langs))
    for i, la in enumerate(langs):
        print(f" {la:>3} " + " ".join(f"{matrix[i][j]:>6.4f}" for j in range(n)))

    result = {"langs": langs, "cosine_similarity": {la: {lb: matrix[i][j] for j, lb in enumerate(langs)} for i, la in enumerate(langs)}}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.out}")

    # LaTeX table
    LANG_NAMES = {
        "en": "English", "ba": "Bashkir", "be": "Belarusian", "tg": "Tajik",
        "ru": "Russian", "zh": "Chinese", "de": "German", "ja": "Japanese",
        "ko": "Korean", "th": "Thai", "yo": "Yoruba",
    }
    print("\n% ---- LaTeX table ----")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Pairwise cosine similarity of refusal direction vectors.}")
    print(r"\label{tab:direction_similarity}")
    col_fmt = "l" + "c" * n
    print(r"\begin{tabular}{" + col_fmt + "}")
    print(r"\toprule")
    header = " & ".join(LANG_NAMES.get(l, l) for l in langs)
    print(r" & " + header + r" \\")
    print(r"\midrule")
    for i, la in enumerate(langs):
        row_vals = []
        for j in range(n):
            val = matrix[i][j]
            cell = f"{val:.4f}"
            if i == j:
                cell = r"\textbf{" + cell + "}"
            row_vals.append(cell)
        print(LANG_NAMES.get(la, la) + " & " + " & ".join(row_vals) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("% ---- end LaTeX ----")


if __name__ == "__main__":
    main()
