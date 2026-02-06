import torch
import json
import os
import os.path as osp
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(override=True)
import argparse
import random
import requests
from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
import mmengine
from pipeline.utils.hook_utils import add_hooks
import sys
from tqdm import tqdm
from utils.utils import LoggerWriter

# Google Cloud Translation API (back-translate to English)
GOOGLE_TRANSLATE_API_KEY = os.environ.get("GOOGLE_TRANSLATE_API_KEY")
TRANSLATE_API_URL = "https://translation.googleapis.com/language/translate/v2"


def translate_to_english(text: str, source_lang: str, api_key: str = None) -> str:
    """Translate text to English using Google Cloud Translation API. Uses GOOGLE_TRANSLATE_API_KEY from .env."""
    key = api_key or GOOGLE_TRANSLATE_API_KEY
    if not key:
        raise ValueError(
            "GOOGLE_TRANSLATE_API_KEY not set. Add it to your .env for back-translation."
        )
    # Google API uses zh-CN for Chinese; keep other codes as-is
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

def main(config_path, data_type):
    
    cfg = mmengine.Config.fromfile(config_path)
    time_stamp = datetime.now().strftime("%y%m%d_%H%M")
    
    
    model_alias = os.path.basename(cfg.model_path)
    cfg.model_alias = model_alias
    
    
    artifact_path = os.path.join("output", "multi_inference" , cfg.model_alias, cfg.lang, data_type)
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path)
    
    logger = mmengine.MMLogger.get_instance(
        name="dissect",
        logger_name="dissect",
        log_file=osp.join(artifact_path, f"{time_stamp}.log"),
    )
    
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.error)

    model_base = construct_model_base(cfg.model_path)
    
    data_test = load_dataset_split(data_type, split='test', lang=cfg.lang)
    
    
    completions_baseline = model_base.generate_completions(data_test, max_new_tokens=512, batch_size=cfg.batch_size, system=None, translation=True if cfg.lang != 'en' else False)

    
    if cfg.lang != 'en':
        for response in tqdm(completions_baseline, desc="Back-translating to English"):
            text = response["response"]
            if len(text) >= 5000:
                text = text[:4999]
            try:
                response["response_translated"] = translate_to_english(text, source_lang=cfg.lang)
            except Exception as e:
                print(f"Translation failed: {e}")
                response["response_translated"] = "Translation Error"
    
    with open (f"{artifact_path}/completions_baseline_{data_type}.json", "w") as f:
        json.dump(completions_baseline, f, indent=4)
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/cfg.yaml')
    parser.add_argument('--type', '-t', type=str, default='harmful')
    args = parser.parse_args()
    config_path = args.config
    data_type = args.type
    main(config_path, data_type)
    

