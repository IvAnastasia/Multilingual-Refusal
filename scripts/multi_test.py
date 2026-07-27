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


def _direction_paths(model_alias: str, lang: str, prefer_native: bool = False):
    """Return (direction_path, metadata_path) for a language.

    Transfer (default): direction.pt (copied English vector) + direction_metadata_ablation.json
        (which the copy set to English's layer). Falls back to direction_ablation.pt.
    Native (prefer_native=True): direction_ablation.pt (the language's own vector) +
        direction_metadata_native.json (the language's own layer, since the copy overwrote
        direction_metadata_ablation.json with English's layer). Falls back to direction.pt / ablation meta.
    """
    if lang == "en":
        base = osp.join("pipeline", "runs", model_alias)
    else:
        base = osp.join("pipeline", "runs", model_alias, lang)
    if prefer_native:
        pt_names = ("direction_ablation.pt", "direction.pt")
        meta_names = ("direction_metadata_native.json", "direction_metadata_ablation.json")
    else:
        pt_names = ("direction.pt", "direction_ablation.pt")
        meta_names = ("direction_metadata_ablation.json", "direction_metadata_native.json")
    meta_path = next((osp.join(base, m) for m in meta_names if osp.isfile(osp.join(base, m))),
                     osp.join(base, meta_names[-1]))
    for name in pt_names:
        dir_path = osp.join(base, name)
        if osp.isfile(dir_path):
            return dir_path, meta_path
    return None, meta_path


def _load_direction_and_layer(cfg, model_alias: str):
    """Load direction tensor and layer index. Fall back to English direction if lang-specific path missing."""
    lang = getattr(cfg, "source_lang", cfg.lang)
    prefer_native = getattr(cfg, "prefer_native", False)
    dir_path, meta_path = _direction_paths(model_alias, lang, prefer_native=prefer_native)
    if dir_path is None and lang != "en":
        dir_path, meta_path = _direction_paths(model_alias, "en", prefer_native=prefer_native)
        if dir_path is not None:
            print(f"  No direction for {lang}, using English direction")
    print(f"  [{'native' if prefer_native else 'transfer'}] direction={dir_path}  meta={meta_path}")
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

    # baseline-only fast path: no steering direction needed (just the model's default behavior)
    if getattr(cfg, "baseline_only", False):
        os.makedirs(osp.join(cfg.artifact_path, "completions"), exist_ok=True)
        data_test = load_dataset_split("harmful", split="test", lang=cfg.lang)
        comp = model_base.generate_completions(
            data_test, fwd_pre_hooks=[], fwd_hooks=[], max_new_tokens=512,
            batch_size=cfg.batch_size, system=None, translation=(cfg.lang != "en"))
        if cfg.lang != "en":
            back_translate_completions(comp, cfg.lang)
        with open(osp.join(cfg.artifact_path, "completions", "harmful_baseline_completions.json"), "w", encoding="utf-8") as f:
            json.dump(comp, f, indent=4, ensure_ascii=False)
        evaluate_jailbreak(
            completions=comp, methodologies=cfg.jailbreak_eval_methodologies,
            evaluation_path=osp.join(cfg.artifact_path, "completions", "harmful_baseline_evaluations.json"),
            translation=(cfg.lang != "en"), cfg=cfg, logger=logger)
        print(f"  [baseline_only] saved to {cfg.artifact_path}/completions/")
        return

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


def run_transfer_pair(cfg, model_base, logger):
    """Apply cfg.source_lang's (native) refusal direction to cfg.lang's harmful test set,
    via ABLATION, and WildGuard-evaluate. Baseline is target-only (already computed elsewhere),
    so it is not recomputed here."""
    direction, layer = _load_direction_and_layer(cfg, cfg.model_alias)  # uses cfg.source_lang + prefer_native
    abl_pre, abl_hooks = get_all_direction_ablation_hooks(model_base, direction, 0)
    data_test = load_dataset_split("harmful", split="test", lang=cfg.lang)
    tr = (cfg.lang != "en")
    comp = model_base.generate_completions(
        data_test, fwd_pre_hooks=abl_pre, fwd_hooks=abl_hooks,
        max_new_tokens=512, batch_size=cfg.batch_size, system=None, translation=tr)
    if tr:
        back_translate_completions(comp, cfg.lang)
    os.makedirs(osp.join(cfg.artifact_path, "completions"), exist_ok=True)
    with open(osp.join(cfg.artifact_path, "completions", "harmful_harm_ablation_completions.json"), "w", encoding="utf-8") as f:
        json.dump(comp, f, indent=4, ensure_ascii=False)
    evaluate_jailbreak(
        completions=comp, methodologies=cfg.jailbreak_eval_methodologies,
        evaluation_path=osp.join(cfg.artifact_path, "completions", "harmful_harm_ablation_evaluations.json"),
        translation=tr, cfg=cfg, logger=logger)
    print(f"  Saved {cfg.source_lang}->{cfg.lang} to {cfg.artifact_path}/completions/")


def main(config_path, run_three_langs=False, native=False, langs=None, baseline_only=False,
         transfer_matrix=False, sources=None, targets=None):
    cfg = mmengine.Config.fromfile(config_path)
    time_stamp = datetime.now().strftime("%y%m%d_%H%M")
    model_alias = os.path.basename(cfg.model_path)
    cfg.model_alias = model_alias
    cfg.prefer_native = native
    cfg.baseline_only = baseline_only
    suffix = "_native" if native else ""
    loop_langs = tuple(langs) if langs else DEFAULT_LANGS

    if transfer_matrix:
        # cross-lingual transfer: each source's native direction applied to each target's test set
        srcs = tuple(sources) if sources else ("be", "ba", "tg")
        tgts = tuple(targets) if targets else ("en", "be", "ba", "tg")
        cfg.prefer_native = True
        model_base = construct_model_base(cfg.model_path)
        for s in srcs:
            for t in tgts:
                cfg.source_lang = s
                cfg.lang = t
                cfg.artifact_path = osp.join("output", model_alias, "xling", f"{s}_to_{t}")
                print(f"\n--- direction from {s} -> test on {t} ---")
                run_transfer_pair(cfg, model_base, logger=None)
        return

    if run_three_langs:
        # Run for the requested languages with Google API back-translation (non-en);
        # don't redirect stdout so tqdm works
        model_base = construct_model_base(cfg.model_path)
        for lang in loop_langs:
            cfg.lang = lang
            cfg.source_lang = lang
            cfg.artifact_path = osp.join("output", model_alias, f"{lang}{suffix}")
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
    parser.add_argument("--native", action="store_true",
                        help="Use each language's own (native) direction instead of the transferred English one; "
                             "writes to output/<alias>/<lang>_native/")
    parser.add_argument("--langs", nargs="+", default=None,
                        help="Override the default ba/be/tg loop (e.g. --langs en yo)")
    parser.add_argument("--baseline_only", action="store_true",
                        help="Only run the no-steering baseline + WildGuard (safety-alignment check)")
    parser.add_argument("--transfer_matrix", action="store_true",
                        help="Cross-lingual: apply each --sources native direction to each --targets test set (ablation)")
    parser.add_argument("--sources", nargs="+", default=None, help="Direction source langs (default: be ba tg)")
    parser.add_argument("--targets", nargs="+", default=None, help="Test target langs (default: en be ba tg)")
    args = parser.parse_args()
    main(args.config, run_three_langs=args.run_three_langs, native=args.native,
         langs=args.langs, baseline_only=args.baseline_only,
         transfer_matrix=args.transfer_matrix, sources=args.sources, targets=args.targets)