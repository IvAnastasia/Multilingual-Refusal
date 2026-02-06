"""
Run inference + back-translation for multilingual refusal (ba, be, tg).
After it finishes, run: scripts/run_wildguard_eval.py --completions <path> --lang <ba|be|tg>
to get WildGuard scores (response_harmful, refusal, compliance).
"""
import torch
import json
import os
import os.path as osp
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(override=True)
import argparse
import requests
from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.model_utils.model_factory import construct_model_base
import mmengine
import sys
from tqdm import tqdm

# Google Cloud Translation API (back-translate to English)
GOOGLE_TRANSLATE_API_KEY = os.environ.get("GOOGLE_TRANSLATE_API_KEY")
TRANSLATE_API_URL = "https://translation.googleapis.com/language/translate/v2"

# Default: run for these three languages with Qwen2.5-7B
DEFAULT_LANGS = ("ba", "be", "tg")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Batch size for generation: larger = faster (use 8–16 if GPU allows; reduce if OOM)
DEFAULT_BATCH_SIZE = 16
# Shorter max tokens = faster; 256 often enough for refusal-style answers
DEFAULT_MAX_NEW_TOKENS = 256
# Use only this fraction of the test set per language (1/4 = faster runs)
DATA_FRACTION = 0.25


def translate_to_english(text: str, source_lang: str, api_key: str = None) -> str:
    """Translate text to English using Google Cloud Translation API."""
    key = api_key or GOOGLE_TRANSLATE_API_KEY
    if not key:
        raise ValueError(
            "GOOGLE_TRANSLATE_API_KEY not set. Add it to your .env for back-translation."
        )
    source = "zh-CN" if source_lang == "zh" else source_lang
    payload = {"q": [text], "target": "en", "source": source}
    resp = requests.post(
        TRANSLATE_API_URL,
        params={"key": key},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["data"]["translations"][0]["translatedText"]


def run_for_lang(cfg, data_type, model_base):
    """Run inference + back-translate + save for one language (cfg.lang)."""
    time_stamp = datetime.now().strftime("%y%m%d_%H%M")
    model_alias = os.path.basename(cfg.model_path)
    artifact_path = os.path.join("output", "multi_inference", model_alias, cfg.lang, data_type)
    os.makedirs(artifact_path, exist_ok=True)

    # Don't redirect stdout/stderr so tqdm progress bars update in the terminal
    data_test = load_dataset_split(data_type, split='test', lang=cfg.lang)
    frac = getattr(cfg, "data_fraction", DATA_FRACTION)
    if frac < 1.0:
        n_full = len(data_test)
        n_use = max(1, int(n_full * frac))
        data_test = data_test[:n_use]
        print(f"  Using {n_use}/{n_full} items ({frac:.0%} of test set)")
    n_items = len(data_test)
    batch_size = getattr(cfg, "batch_size", DEFAULT_BATCH_SIZE)
    if batch_size < 4:
        batch_size = min(DEFAULT_BATCH_SIZE, n_items)
        print(f"  (batch_size was < 4; using {batch_size} for speed)")
    batch_size = min(batch_size, n_items)
    max_new_tokens = getattr(cfg, "max_new_tokens_inference", getattr(cfg, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    n_batches = (n_items + batch_size - 1) // batch_size
    print(f"  Generating completions for {n_items} items in {n_batches} batch(es) (batch_size={batch_size}, max_new_tokens={max_new_tokens})...")
    completions_baseline = model_base.generate_completions(
        data_test,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        system=None,
        translation=True if cfg.lang != 'en' else False,
    )

    if cfg.lang != 'en':
        print(f"  Back-translating {len(completions_baseline)} model responses to English...")
        for completion in tqdm(completions_baseline, desc=f"Back-translating {cfg.lang}"):
            text = completion["response"]  # model output in target lang; we translate to English
            if len(text) >= 5000:
                text = text[:4999]
            try:
                completion["response_translated"] = translate_to_english(text, source_lang=cfg.lang)
            except Exception as e:
                print(f"Translation failed: {e}")
                completion["response_translated"] = "Translation Error"

    out_path = osp.join(artifact_path, f"completions_baseline_{data_type}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(completions_baseline, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(completions_baseline)} completions to {out_path}")


def main(config_path, data_type, model_name=None, run_three_langs=False, batch_size=None, max_new_tokens=None, data_fraction=None):
    cfg = mmengine.Config.fromfile(config_path)
    if model_name is not None:
        cfg.model_path = model_name

    if run_three_langs:
        # Run from scratch for ba, be, tg with given model (default Qwen2.5-7B)
        cfg.model_path = cfg.model_path or DEFAULT_MODEL
        # Force batched generation so 3 langs finish in a few hours (config often has batch_size=1)
        cfg.batch_size = batch_size if batch_size is not None else getattr(cfg, "batch_size", DEFAULT_BATCH_SIZE)
        if cfg.batch_size < 4:
            cfg.batch_size = DEFAULT_BATCH_SIZE
        if max_new_tokens is not None:
            cfg.max_new_tokens_inference = max_new_tokens
        else:
            cfg.max_new_tokens_inference = getattr(cfg, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        cfg.data_fraction = data_fraction if data_fraction is not None else DATA_FRACTION
        print(f"Using batch_size={cfg.batch_size}, max_new_tokens={cfg.max_new_tokens_inference}, data_fraction={cfg.data_fraction}")
        model_base = construct_model_base(cfg.model_path)
        for lang in DEFAULT_LANGS:
            cfg.lang = lang
            print(f"\n--- Running for lang={lang} ---")
            run_for_lang(cfg, data_type, model_base)
        return

    # Single run from config (original behavior)
    model_base = construct_model_base(cfg.model_path)
    run_for_lang(cfg, data_type, model_base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/cfg.yaml')
    parser.add_argument('--type', '-t', type=str, default='harmful')
    parser.add_argument('--model_name', '-m', type=str, default=None, help='Model path (e.g. Qwen/Qwen2.5-7B-Instruct)')
    parser.add_argument('--run_three_langs', action='store_true', help='Run for ba, be, tg in one go with Qwen2.5-7B (or --model_name)')
    parser.add_argument('--batch_size', '-b', type=int, default=None, help='Generation batch size (default 16 for --run_three_langs; use 8 or 4 if OOM)')
    parser.add_argument('--max_new_tokens', type=int, default=None, help='Max new tokens per reply (default 256 for three_langs; 512 from config otherwise)')
    parser.add_argument('--frac', type=float, default=None, help='Use only this fraction of test set per language (default 0.25 = 1/4); use 1.0 for full')
    args = parser.parse_args()
    main(args.config, args.type, model_name=args.model_name, run_three_langs=args.run_three_langs, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, data_fraction=args.frac)
