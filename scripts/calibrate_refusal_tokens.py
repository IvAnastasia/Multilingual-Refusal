"""
Derive refusal tokens for a language the way the paper's Table 4 does:
the most frequent *sentence-initial* tokens that appear distinctively when the
model REFUSES harmful requests, compared to its responses to HARMLESS prompts.

Method:
  1. Generate baseline (no-steering) responses to harmful and harmless prompts
     in the target language (same prompts the direction pipeline uses).
  2. Use WildGuard to label which harmful responses are genuine refusals.
  3. Count the first generated token of (a) refusals and (b) harmless responses.
  4. Refusal tokens = first tokens frequent in refusals but NOT in harmless
     responses (contrastive), i.e. high freq_refusal - freq_harmless.

Output: token ids + decoded text + frequencies, and a ready-to-paste
REFUSAL_TOKENS_LANG entry. Run this, paste the ids into qwen2_model.py, then
re-run the pipeline for that language.

Usage:
  python -m scripts.calibrate_refusal_tokens --model_path <path> --lang tg
"""
import argparse
from collections import Counter
from types import SimpleNamespace

from dataset.load_dataset import load_dataset_split
from pipeline.model_utils.model_factory import construct_model_base


def build_dataset(tgt_prompts, en_prompts=None):
    """generate_completions expects dicts with 'instruction' (fed to the model)
    and 'instruction_en' (English prompt, used by WildGuard)."""
    if en_prompts is None:
        en_prompts = tgt_prompts
    return [{"instruction": t, "instruction_en": e} for t, e in zip(tgt_prompts, en_prompts)]


def first_tokens(completions, only_refusals=False):
    toks = []
    for c in completions:
        if only_refusals and c.get("wildguard", {}).get("refusal", 0) != 1:
            continue
        gt = c.get("generation_tokens", "").split()
        if gt:
            toks.append(int(gt[0]))
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--split", default="train", help="which split to draw prompts from")
    ap.add_argument("--n_harmful", type=int, default=260)
    ap.add_argument("--n_harmless", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--top_k", type=int, default=3, help="how many refusal tokens to propose")
    ap.add_argument("--min_refusal_freq", type=float, default=0.05,
                    help="token must open at least this fraction of refusals")
    args = ap.parse_args()

    mb = construct_model_base(args.model_path, args.lang)
    tok = mb.tokenizer

    # target-language prompts (what the model sees); English prompts for WildGuard's request field
    tgt_harmful = load_dataset_split("harmful", args.split, args.lang, instructions_only=True)[: args.n_harmful]
    en_harmful = load_dataset_split("harmful", args.split, "en", instructions_only=True)[: args.n_harmful]
    tgt_harmless = load_dataset_split("harmless", args.split, args.lang, instructions_only=True)[: args.n_harmless]

    print(f"[calibrate] lang={args.lang}  harmful={len(tgt_harmful)}  harmless={len(tgt_harmless)}")

    print("[calibrate] generating harmful responses...")
    harmful_c = mb.generate_completions(build_dataset(tgt_harmful, en_harmful),
                                        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
    print("[calibrate] generating harmless responses...")
    harmless_c = mb.generate_completions(build_dataset(tgt_harmless),
                                         max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)

    # WildGuard is English-centric, so it must read an English response.
    #  - en / ba / tg: the model already refuses in English -> use raw response (lang='en').
    #  - be (and any lang that refuses in its own script): back-translate first (Google API),
    #    then tell WildGuard lang=<lang> so it reads 'response_translated'.
    wg_lang = "en"
    if args.lang != "en":
        # Detect whether responses are already English (ba/tg) or need translation (be).
        from scripts.multi_test import back_translate_completions
        sample = " ".join((c.get("response", "") or "")[:60] for c in harmful_c[:20])
        needs_translation = sum(ch.isascii() for ch in sample) < 0.6 * max(len(sample), 1)
        if needs_translation:
            print(f"[calibrate] responses look non-English -> back-translating for WildGuard...")
            back_translate_completions(harmful_c, args.lang)
            wg_lang = args.lang  # evaluate_all will read 'response_translated'
        else:
            print("[calibrate] responses already English -> WildGuard reads raw response.")

    print("[calibrate] loading WildGuard + labelling refusals...")
    from evaluators.wildguard import WildGuardEvaluator
    wg = WildGuardEvaluator()
    harmful_c = wg.evaluate_all(harmful_c, logger=None, cfg=SimpleNamespace(lang=wg_lang))

    n_ref = sum(c.get("wildguard", {}).get("refusal", 0) == 1 for c in harmful_c)
    print(f"[calibrate] WildGuard refusals: {n_ref}/{len(harmful_c)}")

    ref_toks = first_tokens(harmful_c, only_refusals=True)
    harm_toks = first_tokens(harmless_c)
    ref_ct, harm_ct = Counter(ref_toks), Counter(harm_toks)
    n_ref_t, n_harm_t = max(len(ref_toks), 1), max(len(harm_toks), 1)

    # contrastive score: frequent as a refusal opener, rare as a harmless opener
    scored = []
    for t, c in ref_ct.items():
        fr = c / n_ref_t
        fh = harm_ct.get(t, 0) / n_harm_t
        if fr >= args.min_refusal_freq and fr > fh:
            scored.append((fr - fh, fr, fh, t))
    scored.sort(reverse=True)

    print("\n=== distinctive refusal openers (freq_refusal - freq_harmless) ===")
    print(f"{'delta':>6} {'f_ref':>6} {'f_harm':>6}  id       text")
    for delta, fr, fh, t in scored[:12]:
        print(f"{delta:6.2f} {fr:6.2f} {fh:6.2f}  {t:<8} {tok.decode([t])!r}")

    chosen = [t for _, _, _, t in scored[: args.top_k]]
    print("\n=== proposed REFUSAL_TOKENS_LANG entry ===")
    decoded = [tok.decode([t]) for t in chosen]
    print(f"    '{args.lang}': {chosen},  # {decoded}")


if __name__ == "__main__":
    main()
