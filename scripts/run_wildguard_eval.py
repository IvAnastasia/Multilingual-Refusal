"""
Run WildGuard evaluation on completions produced by multi_inference.py.

Usage (after multi_inference.py has finished):
  python scripts/run_wildguard_eval.py \\
    --completions output/multi_inference/Qwen2.5-7B-Instruct/ba/harmful/completions_baseline_harmful.json \\
    --lang ba

  # Or for all three languages:
  for lang in ba be tg; do
    python scripts/run_wildguard_eval.py \\
      --completions output/multi_inference/Qwen2.5-7B-Instruct/$lang/harmful/completions_baseline_harmful.json \\
      --lang $lang
  done
"""
import argparse
import json
import logging
import os.path as osp

import mmengine

# Add project root so imports work
PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
import sys
sys.path.insert(0, PROJECT_ROOT)

from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak


def main():
    parser = argparse.ArgumentParser(description="Run WildGuard on multi_inference completions")
    parser.add_argument("--completions", "-c", required=True, help="Path to completions_baseline_*.json from multi_inference")
    parser.add_argument("--lang", "-l", default="en", help="Language code (ba, be, tg, en, ...). Non-en uses response_translated.")
    parser.add_argument("--config", default=osp.join(PROJECT_ROOT, "configs", "cfg.yaml"), help="Config for data_loader etc.")
    parser.add_argument("--output", "-o", default=None, help="Evaluation JSON path (default: same dir as completions, evaluation_wildguard.json)")
    args = parser.parse_args()

    if not osp.isfile(args.completions):
        raise FileNotFoundError(f"Completions file not found: {args.completions}")

    cfg = mmengine.Config.fromfile(args.config)
    cfg.lang = args.lang

    evaluation_path = args.output
    if evaluation_path is None:
        dirname = osp.dirname(args.completions)
        evaluation_path = osp.join(dirname, "evaluation_wildguard.json")

    logger = logging.getLogger("wildguard_eval")
    logger.setLevel(logging.INFO)

    evaluation = evaluate_jailbreak(
        completions_path=args.completions,
        methodologies=["wildguard"],
        evaluation_path=evaluation_path,
        translation=(args.lang != "en"),
        cfg=cfg,
        logger=logger,
    )

    print("\nWildGuard summary:")
    print(f"  response_harmful (ASR): {evaluation.get('wildguard_harmful', 'N/A')}")
    print(f"  refusal rate:          {evaluation.get('wildguard_refusal', 'N/A')}")
    print(f"  compliance:             {evaluation.get('wildguard_compliance', 'N/A')}")
    print(f"  Saved: {evaluation_path}")


if __name__ == "__main__":
    main()
