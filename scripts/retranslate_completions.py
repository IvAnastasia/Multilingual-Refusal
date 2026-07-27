"""
Re-fill `response_translated` for completion files whose back-translation failed
(e.g. produced "Translation Error" because the Google key was invalid at generation time).

Regenerating is unnecessary — the model `response` is intact; only the translation is broken.
This re-runs back-translation (Google Translate) in place. No GPU needed.

Usage:
  python -m scripts.retranslate_completions --completions <path.json> --lang ba
  # or many:
  for l in ba be tg; do
    python -m scripts.retranslate_completions \
      --completions output/multi_inference/Qwen2.5-14B-Instruct/$l/harmful/completions_baseline_harmful.json \
      --lang $l
  done
"""
import argparse
import json

from scripts.multi_test import back_translate_completions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--completions", "-c", required=True)
    ap.add_argument("--lang", "-l", required=True)
    ap.add_argument("--only_failed", action="store_true",
                    help="Only re-translate entries whose response_translated is missing/'Translation Error'")
    args = ap.parse_args()

    with open(args.completions, encoding="utf-8") as f:
        completions = json.load(f)

    if args.only_failed:
        todo = [c for c in completions if c.get("response_translated", "") in ("", "Translation Error")]
    else:
        todo = completions
    print(f"[retranslate] {args.completions}: {len(todo)}/{len(completions)} to translate (lang={args.lang})")

    if todo:
        back_translate_completions(todo, args.lang)

    errs = sum(1 for c in completions if c.get("response_translated", "") == "Translation Error")
    print(f"[retranslate] remaining 'Translation Error': {errs}/{len(completions)}")

    with open(args.completions, "w", encoding="utf-8") as f:
        json.dump(completions, f, indent=4, ensure_ascii=False)
    print(f"[retranslate] saved {args.completions}")


if __name__ == "__main__":
    main()
