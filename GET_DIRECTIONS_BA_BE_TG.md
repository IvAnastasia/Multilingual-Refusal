# Getting harmful/harmless data and directions for ba, be, tg

## 1. Get harmful and harmless data (train / val / test) for ba, be, tg

The pipeline needs translated splits:

- **Harmful:** `harmful_train`, `harmful_val`, `harmful_test`
- **Harmless:** `harmless_train`, `harmless_val`, `harmless_test`

These are produced by **translate_data.py** using the Google Cloud Translation API.

**Steps:**

1. Set `GOOGLE_TRANSLATE_API_KEY` in `.env`.
2. In `scripts/translate_data.py`, ensure `target_langs = ['be', 'tg', 'ba']` (already set).
3. Run:

   ```bash
   cd /path/to/Multilingual-Refusal
   python scripts/translate_data.py
   ```

This writes into `dataset/splits_multi/` for each language:

- `harmful_{train,val,test}_translated_{lang}.json`
- `harmless_{train,val,test}_translated_{lang}.json`

Harmless uses sampled English splits (`harmless_train_200_sampled`, `harmless_val_200_sampled`, `harmless_test_500_sampled`); harmful uses the full train/val/test splits.

---

## 2. Get the direction for each language (ba, be, tg)

Directions are computed by **run_pipeline.py** with `source_lang` set to the target language. Each run loads that language’s harmful/harmless train and val, computes a steering direction, and saves it under `pipeline/runs/<model_alias>/<source_lang>/`.

**Per-language runs:**

```bash
# Bashkir (ba)
python pipeline/run_pipeline.py --config_path configs/cfg.yaml --model_path Qwen/Qwen2.5-7B-Instruct --source_lang ba

# Belarusian (be)
python pipeline/run_pipeline.py --config_path configs/cfg.yaml --model_path Qwen/Qwen2.5-7B-Instruct --source_lang be

# Tajik (tg)
python pipeline/run_pipeline.py --config_path configs/cfg.yaml --model_path Qwen/Qwen2.5-7B-Instruct --source_lang tg
```

Or set `source_lang` in the config and run without `--source_lang`.

**Outputs (per language):**

- `pipeline/runs/Qwen2.5-7B-Instruct/ba/direction_ablation.pt`
- `pipeline/runs/Qwen2.5-7B-Instruct/ba/direction_metadata_ablation.json`
- Same for `be/` and `tg/`.

**Optional:** use smaller data for a quick run:

```bash
python pipeline/run_pipeline.py --config_path configs/cfg.yaml --model_path Qwen/Qwen2.5-7B-Instruct --source_lang ba --n_train 16 --n_val 8 --n_test 16
```

---

## 3. Summary

| Step | What | Command / output |
|------|------|------------------|
| **Data** | Harmful + harmless train/val/test for ba, be, tg | `python scripts/translate_data.py` → `dataset/splits_multi/*_translated_{ba,be,tg}.json` |
| **Direction** | One direction per language | `python pipeline/run_pipeline.py ... --source_lang {ba,be,tg}` → `pipeline/runs/<model_alias>/{ba,be,tg}/direction_ablation.pt` + metadata |

After that, **multi_test.py --run_three_langs** will use these per-language directions (or fall back to the English direction if a lang folder is missing).
