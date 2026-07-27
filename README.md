# Refusal Directions Across Safety-Aligned and Low-Resource Languages

*Abstract:* Refusal directions have been shown to transfer cross-lingually with near-perfect effectiveness across safety-aligned languages (Wang et al., 2025). This universality is reported to hold regardless of resource level, but the only truly low-resourced language evaluated is Yoruba, which is safety-misaligned and where transfer is limited. It thus remains unclear how refusal directions behave for other low-resourced languages. **We extend PolyRefuse to three additional low-resourced languages (Belarusian, Bashkir, and Tajik) and replicate the pipeline of (Wang et al., 2025) on Qwen2.5-14B-Instruct.** We find that universality is asymmetric: directions from safety-aligned languages transfer broadly, while those from strongly misaligned languages transfer only to other misaligned languages. We further show that geometric alignment and behavioral transfer are partially dissociated. These findings complicate the universality claim and highlight that users of the least-resourced languages may face weaker safety guarantees.


This repository studies how the **refusal direction** — a single linear direction in a
language model's residual stream that mediates refusal of harmful requests — behaves
**across languages**, with a focus on the contrast between higher-resource,
safety-aligned languages and low-resource ones.

The analysis here targets **Qwen2.5-14B-Instruct** and five languages:
**English (en), Belarusian (be), Bashkir (ba), Tajik (tg), Yoruba (yo)**.

---
## Key findings (Qwen2.5-14B-Instruct)

- **Cross-lingual safety gradient** (baseline refusal rate): en ≈ 0.95 › be ≈ 0.75 ›
  yo ≈ 0.38 › ba ≈ 0.12 › tg ≈ 0.08. Safety alignment degrades sharply for low-resource
  languages.
- **A refusal direction is only cleanly extractable where harmful/harmless representations
  separate.** Silhouette scores (harmful vs harmless at the extraction layer):
  en 0.36, be 0.17, yo 0.09, tg 0.07, ba 0.05. Low-separation languages yield
  **degenerate directions** (selection collapses to a trivial layer).
- **Universality is bounded.** Ablating the **English** or **Belarusian** direction drives
  harmful-query compliance to ≈ 0.9–1.0 across *all* languages (universal), while the
  **Bashkir / Tajik** directions barely affect English (compliance ≈ 0.06 / 0.33) — they are
  language-local.
- **Shared refusal subspace.** For en/be, the refusal direction aligns with the
  difference-in-means vectors of every language at a consistent depth (~layer 30 of 48).
- **Refusal-token language matters.** The model expresses refusals in different scripts per
  language; using mismatched refusal tokens (e.g. Cyrillic tokens for a language the model
  refuses in English) makes the refusal signal invisible and breaks extraction. Tokens are
  therefore **calibrated per language** (see below).

---

## Repository layout

```
configs/            experiment config (cfg.yaml)
dataset/            multilingual harmful/harmless splits + loaders
pipeline/           direction extraction + evaluation pipeline
  run_pipeline.py     extract a refusal direction for one language, then evaluate
  model_utils/        per-model-family wrappers (refusal tokens, hooks, chat template)
  submodules/         generate_directions, select_direction, evaluate_jailbreak
  evaluator/          lm-eval harness wrapper
  runs/<alias>/<lang>/ per-language artifacts (directions, metadata, completions)
evaluators/         WildGuard safety classifier wrapper
scripts/            calibration, cross-lingual evaluation, plotting
output/<alias>/     evaluation outputs (baseline / ablation / addition / transfer matrix)
figures/            generated figures
results/            cached activations, similarity/silhouette JSONs
```


---
Below is the README of the paper we build on:

# Refusal Direction is Universal Across Safety-Aligned Languages

This repository contains the code and dataset for the paper "Refusal Direction is Universal Across Safety-Aligned Languages".

## PolyRefuse Dataset

The **PolyRefuse** dataset is a multilingual safety evaluation dataset covering 14 languages: ar, de, en, es, fr, it, ja, ko, nl, pl, ru, th, zh, yo.

You can find the dataset in the [`PolyRefuse/`](PolyRefuse) directory, which contains:
- Harmful prompts (train/val/test splits) translated to all languages
- Harmless prompts (train/val/test splits) translated to all languages
- Back-translated versions for analysis

## Setup

### Installation

```bash
source setup.sh
```
Install the evaluation harness from source

```bash
cd lm-evaluation-harness
pip install -e .
``` 
## Usage

### Running Experiments

#### Refusal Vector Ablation

```bash
# Configure your experiment settings in configs/cfg.yaml
python -m pipeline.run_pipeline --config configs/cfg.yaml
```
#### for example, we run the experiment on Qwen2.5-7B-Instruct model in Japanese with the following settings:

```bash
python -m pipeline.run_pipeline --config runs/Qwen2.5-7B-Instruct/ja/ja.yaml
```

#### Evaluating the model on multiple languages

```bash
# For running multiple language evaluation configurations
python -m scripts.multi_test --config configs/cfg.yaml
```

#### for example, we evaluate the Qwen2.5-7B-Instruct model (ablated the refusal direction extracted in Japanese) in Korean language with the following settings:

```bash
python -m scripts.multi_test --config output/ja_vector_sweep/Qwen/Qwen2.5-7B-Instruct/ko/20250519-232436/1/ko.yaml 
```



## Repository Structure

```
.
├── PolyRefuse/              # Multilingual safety dataset
├── configs/                 # Configuration files
├── dataset/                 # Dataset loading and processing
├── evaluators/              # Safety evaluators
├── pipeline/                # Main experimental pipeline
│   ├── model_utils/        # Model implementations
│   ├── submodules/         # Pipeline components
│   └── run_pipeline.py     # Main pipeline runner
├── scripts/                 # Utility scripts and experiments
├── utils/                   # Helper utilities
└── requirements.txt        # Python dependencies
```

## Citation

If you use this code or dataset, please cite our paper:

```bibtex
@inproceedings{
wang2025refusal,
title={Refusal Direction is Universal Across Safety-Aligned Languages},
author={Xinpeng Wang and Mingyang Wang and Yihong Liu and Hinrich Schuetze and Barbara Plank},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=eWxKpdAdXH}
}
```

## License

See [LICENSE](LICENSE) for details.

## Baseline vs English Refusal Vector Ablation

![Baseline vs English Refusal Vector Ablation](images/baseline_vs_harm_ablation-1.png)
