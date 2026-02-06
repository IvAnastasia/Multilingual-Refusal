# Evaluation order for paper (ba, be, tg)

Run steps in this order. Prerequisites: `.env` with `GOOGLE_TRANSLATE_API_KEY`; config `configs/cfg.yaml` with `model_path` (e.g. Qwen/Qwen2.5-7B-Instruct). Use the same `model_alias` everywhere (e.g. `Qwen2.5-7B-Instruct`).

---

## Phase 1 – Data and English direction (one-time)

### Step 1 – Translated data for ba, be, tg

Needed for directions and for test sets.

```bash
python scripts/translate_data.py
```

→ `dataset/splits_multi/`: harmful & harmless train/val/test for ba, be, tg.

---

### Step 2 – English refusal direction

Compute the direction on English harmful vs harmless (train/val), save under `pipeline/runs/<model_alias>/`.

```bash
python pipeline/run_pipeline.py --config_path configs/cfg.yaml --model_path Qwen/Qwen2.5-7B-Instruct --source_lang en
```

→ `pipeline/runs/Qwen2.5-7B-Instruct/direction_ablation.pt` and `direction_metadata_ablation.json`.

**Note:** For `source_lang en` there is **no** `en` subfolder. The English direction is saved directly in `pipeline/runs/<model_alias>/`. Do **not** interrupt the run (e.g. with Ctrl+C) during model download or direction computation, or the direction files will not be created.

---

### Step 3 – Transfer English direction to ba, be, tg

Copy the same English direction into each language folder so “transfer” runs use it.

```bash
python scripts/copy_english_direction_to_langs.py --model_alias Qwen2.5-7B-Instruct
```

→ `pipeline/runs/Qwen2.5-7B-Instruct/{ba,be,tg}/direction.pt` and metadata (same vector as en).

---

## Phase 2 – Baselines (no steering) + WildGuard

### Step 4 – Baseline completions for ba, be, tg

Model with **no** steering on harmful test.

```bash
python scripts/multi_inference.py --run_three_langs --type harmful
```

→ `output/multi_inference/Qwen2.5-7B-Instruct/{ba,be,tg}/harmful/completions_baseline_harmful.json`.

---

### Step 5 – WildGuard on baseline (your “first step” metrics)

Evaluate baseline completions with WildGuard → **baseline refusal / harmful ASR** per language.

```bash
for lang in ba be tg; do
  python scripts/run_wildguard_eval.py \
    --completions output/multi_inference/Qwen2.5-7B-Instruct/$lang/harmful/completions_baseline_harmful.json \
    --lang $lang
done
```

→ `output/multi_inference/.../harmful/evaluation_wildguard.json` per lang. Record: `wildguard_refusal`, `wildguard_harmful`, `wildguard_compliance` as **baseline** for the paper.

---

## Phase 3 – Transfer (English vector) + WildGuard

### Step 6 – Completions with transferred English vector

Same model, but with **English** refusal direction (ablation + addition) on ba, be, tg. Uses the direction you copied in Step 3.

```bash
python scripts/multi_test.py --run_three_langs
```

→ `output/Qwen2.5-7B-Instruct/{ba,be,tg}/completions/`:  
`harmful_baseline_completions.json`, `harmful_<mode>_completions.json`, `harmful_<mode>_addition_completions.json`, plus **evaluation JSONs** (WildGuard is run inside multi_test).

So you get **transfer** WildGuard metrics from:  
`output/Qwen2.5-7B-Instruct/{ba,be,tg}/completions/*_evaluations.json`.

---

### Step 7 – Compare baseline vs transfer

- **Baseline**: from Step 5 (WildGuard on multi_inference baseline completions).
- **Transfer**: from Step 6 (WildGuard on multi_test runs that use the English direction).

Compare refusal rate / harmful ASR / compliance **baseline vs transfer** per language (table or plot in the paper).

---

## Phase 4 – Per-language directions and similarity (optional but useful)

### Step 8 – Per-language refusal directions (ba, be, tg)

Compute a **local** direction for each language (for similarity and possibly “local steering” comparison).

```bash
for lang in ba be tg; do
  python pipeline/run_pipeline.py --config_path configs/cfg.yaml --model_path Qwen/Qwen2.5-7B-Instruct --source_lang $lang
done
```

→ `pipeline/runs/Qwen2.5-7B-Instruct/{ba,be,tg}/direction_ablation.pt` (these **overwrite** the copied English direction in those folders; run Step 8 **after** Step 6 if you want to keep transfer runs unchanged, or use a copy of the en direction elsewhere before overwriting).

**Recommendation:** Run Step 8 **after** Step 6 so transfer experiments use the copied English vector. Then you get per-lang directions only for similarity (and optionally for a “local vector” comparison later).

---

### Step 9 – Refusal vector similarity

Compare direction vectors across en, ba, be, tg (need English + per-language directions from Step 2 and Step 8).

```bash
python scripts/direction_similarity.py --model_alias Qwen2.5-7B-Instruct --langs en ba be tg --out results/direction_similarity.json
```

→ Pairwise cosine similarity matrix (e.g. en–ba, en–be, en–tg, ba–be, …). Use in the paper to discuss “how similar” refusal vectors are across languages.

---

## Order summary (what to run when)

| Order | What | Script / command |
|-------|------|-------------------|
| 1 | Data for ba, be, tg | `translate_data.py` |
| 2 | English direction | `run_pipeline.py --source_lang en` |
| 3 | Transfer en → ba, be, tg | `copy_english_direction_to_langs.py` |
| 4 | Baseline completions (no steering) | `multi_inference.py --run_three_langs` |
| 5 | **Baseline** WildGuard metrics | `run_wildguard_eval.py` per lang on baseline completions |
| 6 | **Transfer** completions + WildGuard | `multi_test.py --run_three_langs` |
| 7 | Compare baseline vs transfer | Use outputs of 5 and 6 |
| 8 | Per-lang directions (for similarity) | `run_pipeline.py --source_lang {ba,be,tg}` (after Step 6) |
| 9 | Refusal vector similarity | `direction_similarity.py --langs en ba be tg` |

**Short flow:** 1 → 2 → 3 → 4 → 5 (baseline done) → 6 (transfer done) → 7 (comparison) → 8 → 9 (similarity).
