"""
Copy the English steering direction to ba, be, tg (or specified langs) so multi_test.py
can use the same direction for all three languages.

Prerequisites:
  1. Get the English direction by running the pipeline with source_lang=en:
       python pipeline/run_pipeline.py --config_path configs/cfg.yaml --model_path Qwen/Qwen2.5-7B-Instruct
     with config containing: source_lang: en
     This creates pipeline/runs/<model_alias>/direction_ablation.pt and direction_metadata_ablation.json.

  2. Then run this script to copy them to ba, be, tg:
       python scripts/copy_english_direction_to_langs.py --model_alias Qwen2.5-7B-Instruct
     Or for specific langs:
       python scripts/copy_english_direction_to_langs.py --model_alias Qwen2.5-7B-Instruct --langs ba be
"""
import argparse
import json
import os.path as osp
import shutil

DEFAULT_LANGS = ("ba", "be", "tg")
RUNS_BASE = "pipeline/runs"


def main():
    parser = argparse.ArgumentParser(description="Copy English direction to ba, be, tg (or specified langs)")
    parser.add_argument("--model_alias", "-m", required=True, help="Model alias, e.g. Qwen2.5-7B-Instruct")
    parser.add_argument("--langs", "-l", nargs="+", default=list(DEFAULT_LANGS), help=f"Target langs (default: {DEFAULT_LANGS})")
    parser.add_argument("--dry_run", action="store_true", help="Only print what would be copied")
    args = parser.parse_args()

    base = osp.join(RUNS_BASE, args.model_alias)
    meta_name = "direction_metadata_ablation.json"
    meta_src = osp.join(base, meta_name)
    if not osp.isfile(meta_src):
        raise FileNotFoundError(
            f"English metadata not found: {meta_src}\n"
            "Run pipeline with source_lang=en first:\n"
            "  python pipeline/run_pipeline.py --config_path configs/cfg.yaml --model_path Qwen/Qwen2.5-7B-Instruct\n"
            "  (ensure config has source_lang: en)"
        )

    for fname in ("direction.pt", "direction_ablation.pt"):
        dir_src = osp.join(base, fname)
        if osp.isfile(dir_src):
            break
    else:
        raise FileNotFoundError(
            f"No direction.pt or direction_ablation.pt in {base}\n"
            "Run pipeline with source_lang=en first (see above)."
        )

    for lang in args.langs:
        dest_dir = osp.join(base, lang)
        dir_dest = osp.join(dest_dir, "direction.pt")
        meta_dest = osp.join(dest_dir, meta_name)
        if args.dry_run:
            print(f"Would copy: {dir_src} -> {dir_dest}")
            print(f"Would copy: {meta_src} -> {meta_dest}")
            continue
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(dir_src, dir_dest)
        shutil.copy2(meta_src, meta_dest)
        print(f"Copied English direction -> {dest_dir}/ (direction.pt, {meta_name})")

    if not args.dry_run:
        print(f"Done. You can run: python scripts/multi_test.py --run_three_langs")


if __name__ == "__main__":
    main()
