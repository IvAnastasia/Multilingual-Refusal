"""
Run baseline + steering/ablation for harmful test set, back-translate responses, then evaluate.
Use --run_three_langs to run only for ba, be, tg with Google Cloud Translation API for back-translation.
"""
import torch
import json
import os
import os.path as osp
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(override=True)
import argparse
import sys
import requests
from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
import mmengine
from tqdm import tqdm
from utils.utils import LoggerWriter

# Google Cloud Translation API (back-translate model responses to English)
GOOGLE_TRANSLATE_API_KEY = os.environ.get("GOOGLE_TRANSLATE_API_KEY")
TRANSLATE_API_URL = "https://translation.googleapis.com/language/translate/v2"
DEFAULT_LANGS = ("ba", "be", "tg")


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


def back_translate_completions(completions_list, lang: str):
    """Fill response_translated for each completion using Google Cloud Translation API."""
    for completion in tqdm(completions_list, desc=f"Back-translating {lang}"):
        text = completion["response"]
        if len(text) >= 5000:
            text = text[:4999]
        try:
            completion["response_translated"] = translate_to_english(text, source_lang=lang)
        except Exception as e:
            print(f"Translation failed: {e}")
            completion["response_translated"] = "Translation Error"


def _direction_paths(model_alias: str, lang: str):
    """Return (direction.pt path, metadata path) for a language. Tries direction.pt then direction_ablation.pt."""
    if lang == "en":
        base = osp.join("pipeline", "runs", model_alias)
    else:
        base = osp.join("pipeline", "runs", model_alias, lang)
    meta_path = osp.join(base, "direction_metadata_ablation.json")
    for name in ("direction.pt", "direction_ablation.pt"):
        dir_path = osp.join(base, name)
        if osp.isfile(dir_path):
            return dir_path, meta_path
    return None, meta_path


def _load_direction_and_layer(cfg, model_alias: str):
    """Load direction tensor and layer index. Fall back to English direction if lang-specific path missing."""
    lang = getattr(cfg, "source_lang", cfg.lang)
    dir_path, meta_path = _direction_paths(model_alias, lang)
    if dir_path is None and lang != "en":
        dir_path, meta_path = _direction_paths(model_alias, "en")
        if dir_path is not None:
            print(f"  No direction for {lang}, using English direction")
    if dir_path is None:
        raise FileNotFoundError(
            f"No direction found. Run pipeline with source_lang=en to create "
            f"pipeline/runs/{model_alias}/direction_ablation.pt (or direction.pt), "
            f"then copy to ba/be/tg with: python scripts/copy_english_direction_to_langs.py --model_alias {model_alias}"
        )
    direction_ablation = torch.load(dir_path, map_location="cpu")
    with open(meta_path, encoding="utf-8") as f:
        layer = json.load(f)["layer"][0]
    if isinstance(direction_ablation, list):
        direction_ablation = direction_ablation[0]
    return direction_ablation, layer


def run_for_lang(cfg, model_base, logger):
    """Run generation (baseline + ablation + addition), back-translate if non-en, save and evaluate for one language."""
    model_alias = cfg.model_alias
    direction_ablation, layer = _load_direction_and_layer(cfg, model_alias)

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    harm_actadd_fwd_pre_hooks, harm_actadd_fwd_hooks = [], []
    or_ablation_fwd_pre_hooks, or_ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction_ablation, 0)
    harm_actadd_fwd_pre_hooks.append((model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction_ablation, coeff=+cfg.addact_coeff)))
    or_ablation_harm_actadd_fwd_pre_hooks = or_ablation_fwd_pre_hooks + harm_actadd_fwd_pre_hooks
    or_ablation_harm_actadd_fwd_hooks = or_ablation_fwd_hooks + harm_actadd_fwd_hooks

    data_test = load_dataset_split("harmful", split="test", lang=cfg.lang)
    dataset_name = "harmful"
    intervention_label = cfg.mode

    completions = model_base.generate_completions(
        data_test, fwd_pre_hooks=or_ablation_harm_actadd_fwd_pre_hooks, fwd_hooks=or_ablation_harm_actadd_fwd_hooks,
        max_new_tokens=512, batch_size=cfg.batch_size, system=None, translation=(cfg.lang != "en"))
    completions_baseline = model_base.generate_completions(
        data_test, fwd_pre_hooks=baseline_fwd_pre_hooks, fwd_hooks=baseline_fwd_hooks,
        max_new_tokens=512, batch_size=cfg.batch_size, system=None, translation=(cfg.lang != "en"))
    completions_addition = model_base.generate_completions(
        data_test, fwd_pre_hooks=harm_actadd_fwd_pre_hooks, fwd_hooks=harm_actadd_fwd_hooks,
        max_new_tokens=512, batch_size=cfg.batch_size, system=None, translation=(cfg.lang != "en"))

    if cfg.lang != "en":
        back_translate_completions(completions, cfg.lang)
        back_translate_completions(completions_baseline, cfg.lang)
        back_translate_completions(completions_addition, cfg.lang)

    os.makedirs(osp.join(cfg.artifact_path, "completions"), exist_ok=True)
    for name, data in [
        (f"{dataset_name}_{intervention_label}_completions", completions),
        (f"{dataset_name}_baseline_completions", completions_baseline),
        (f"{dataset_name}_{intervention_label}_addition_completions", completions_addition),
    ]:
        with open(osp.join(cfg.artifact_path, "completions", f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    torch.cuda.empty_cache()

    for completions_data, eval_name in [
        (completions, f"{dataset_name}_{intervention_label}_evaluations"),
        (completions_baseline, f"{dataset_name}_baseline_evaluations"),
        (completions_addition, f"{dataset_name}_{intervention_label}_addition_evaluations"),
    ]:
        evaluate_jailbreak(
            completions=completions_data,
            methodologies=cfg.jailbreak_eval_methodologies,
            evaluation_path=osp.join(cfg.artifact_path, "completions", f"{eval_name}.json"),
            translation=(cfg.lang != "en"),
            cfg=cfg,
            logger=logger,
        )
    print(f"  Saved completions and evaluations to {cfg.artifact_path}/completions/")


def main(config_path, run_three_langs=False):
    cfg = mmengine.Config.fromfile(config_path)
    time_stamp = datetime.now().strftime("%y%m%d_%H%M")
    model_alias = os.path.basename(cfg.model_path)
    cfg.model_alias = model_alias

    if run_three_langs:
        # Run only for ba, be, tg with Google API back-translation; don't redirect stdout so tqdm works
        model_base = construct_model_base(cfg.model_path)
        for lang in DEFAULT_LANGS:
            cfg.lang = lang
            cfg.source_lang = lang
            cfg.artifact_path = osp.join("output", model_alias, lang)
            os.makedirs(cfg.artifact_path, exist_ok=True)
            log_file = osp.join(cfg.artifact_path, f"{time_stamp}.log")
            logger = mmengine.MMLogger.get_instance(
                name=f"dissect_{lang}",
                logger_name=f"dissect_{lang}",
                log_file=log_file,
            )
            print(f"\n--- Running for lang={lang} ---")
            run_for_lang(cfg, model_base, logger)
        return

    # Single run from config (original behavior)
    if "artifact_path" not in cfg:
        cfg.artifact_path = osp.join("output", model_alias, cfg.lang)
    cfg.source_lang = getattr(cfg, "source_lang", cfg.lang)
    logger = mmengine.MMLogger.get_instance(
        name="dissect",
        logger_name="dissect",
        log_file=osp.join(cfg.artifact_path, f"{time_stamp}.log"),
    )
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.error)

    model_base = construct_model_base(cfg.model_path)
    run_for_lang(cfg, model_base, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="configs/cfg.yaml")
    parser.add_argument("--run_three_langs", action="store_true", help="Run only for ba, be, tg with Google API back-translation")
    args = parser.parse_args()
    main(args.config, run_three_langs=args.run_three_langs)