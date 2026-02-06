from dotenv import load_dotenv
load_dotenv(override=True)
import os
import sys
from pathlib import Path

# Project root: for imports and for output paths (works regardless of cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.load_dataset import load_dataset_split
import json
import requests
from tqdm import tqdm

# API key from .env (GOOGLE_TRANSLATE_API_KEY)
GOOGLE_TRANSLATE_API_KEY = os.environ.get("GOOGLE_TRANSLATE_API_KEY")

TRANSLATE_API_URL = "https://translation.googleapis.com/language/translate/v2"

# Set to a small number (e.g. 3) to run with a few examples for testing; None = process full dataset
MAX_EXAMPLES = None

data_type = 'harmful'
split = 'train'


def translate_text(text: str, target_lang: str, source_lang: str = 'en', api_key: str = None) -> str:
    """Translate a single text with Google Cloud Translation API (API key from .env)."""
    key = api_key or GOOGLE_TRANSLATE_API_KEY
    payload = {
        "q": [text],
        "target": target_lang,
        "source": source_lang,
    }
    resp = requests.post(
        TRANSLATE_API_URL,
        params={"key": key},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["data"]["translations"][0]["translatedText"]


# translate the harmful_test set to the target language
def translate_harmful_to_target_language(target_lang, split):
    if data_type == 'harmless':
        if split == 'test':
            split_ = split + '_500_sampled'
        else:
            split_ = split + '_200_sampled'
    else:
        split_ = split
    harmful_test_set = load_dataset_split(data_type, split=split_)
    if MAX_EXAMPLES is not None:
        harmful_test_set = harmful_test_set[:MAX_EXAMPLES]
        print(f"  (using first {MAX_EXAMPLES} examples only)")
    harmful_test_set_translated = []

    if split == 'test':
        for item in tqdm(harmful_test_set):
            try:
                translated_item = translate_text(item['instruction'], target_lang)
                to_add = {
                    'instruction': item['instruction'],
                    'instruction_translated': translated_item,
                    'category': item['category']
                }
                harmful_test_set_translated.append(to_add)
            except Exception as e:
                print(f"Could not translate: {item['instruction']}")
                print(f"Error: {e}")
                continue
    else:
        for item in tqdm(harmful_test_set):
            try:
                translated_item = translate_text(item['instruction'], target_lang)
                to_add = {
                    'instruction': translated_item,
                    'category': item['category']
                }
                harmful_test_set_translated.append(to_add)
            except Exception as e:
                print(f"Could not translate: {item['instruction']}")
                print(f"Error: {e}")
                continue
    return harmful_test_set_translated


# 'en', 'de','es', 'fr','it', 'nl', 'pl',  'ar','th',  'yo', 'ru',
target_langs = ['be', 'tg', 'ba']
#Belarusian, Tajik, Bashkir
# [ 'zh-CN', 'ja','ko',]
#  'th',  'yo', 'ru', 'zh', 'ja','ko',
# ['de', 'es', 'fr', 'it', 'nl', 'ja', 'pl', 'ru', 'zh', 'ko', 'ar']
# target_langs = []


for target_lang in target_langs:
    for data_type in ["jailbreakbench"]:
        for split in ['test']:
            print(f"Translating {data_type}_{split} to {target_lang}")
            harmful_test_set_translated = translate_harmful_to_target_language(target_lang, split)
            out_name = f'{data_type}_{split}_translated_{target_lang}'
            if MAX_EXAMPLES is not None:
                out_name += f'_sample{MAX_EXAMPLES}'
            out_path = PROJECT_ROOT / 'dataset' / 'splits_multi' / f'{out_name}.json'
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(harmful_test_set_translated, f, indent=4, ensure_ascii=False)
            print(f"  -> {out_path} ({len(harmful_test_set_translated)} items)")
    for data_type in ["harmful", "harmless"]:
        for split in ['train', 'val', 'test']:
            print(f"Translating {data_type}_{split} to {target_lang}")
            harmful_test_set_translated = translate_harmful_to_target_language(target_lang, split)
            out_name = f'{data_type}_{split}_translated_{target_lang}'
            if MAX_EXAMPLES is not None:
                out_name += f'_sample{MAX_EXAMPLES}'
            out_path = PROJECT_ROOT / 'dataset' / 'splits_multi' / f'{out_name}.json'
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(harmful_test_set_translated, f, indent=4, ensure_ascii=False)
            print(f"  -> {out_path} ({len(harmful_test_set_translated)} items)")