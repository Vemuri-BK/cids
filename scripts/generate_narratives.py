"""Generate Type-N narrative prompts with an open LLM (run on Kaggle GPU).

Two runs are needed:
  1) --target train_style : model A (default Qwen2.5-7B-Instruct), ALL 1506 cases
                            -> columns narrative_1..3
  2) --target heldout     : a DIFFERENT model (default Phi-3.5-mini), TEST cases only
                            -> columns narrative_heldout_1..3 (never used in training)

Resumable: every finished case is appended to <out_dir>/<target>_progress.jsonl,
so a Kaggle session that times out can simply be re-run.

Example (Kaggle):
    python scripts/generate_narratives.py --prompts /kaggle/input/cids-assets/cids_prompts.csv \
        --target train_style --out_dir /kaggle/working/narratives --limit 20   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cids.prompts.narrative import (SYSTEM_PROMPT, FilterConfig,  # noqa: E402
                                    build_user_message, score_candidates, select_top_k)

FIELD_COLS = ["age_bin", "menopause", "ethnicity", "subtype", "field_strength", "manufacturer",
              "view", "laterality", "fat_suppressed", "implant", "num_phases"]
DEFAULT_MODELS = {"train_style": "Qwen/Qwen2.5-7B-Instruct",
                  "heldout": "microsoft/Phi-3.5-mini-instruct"}


# ---------------------------------------------------------------------------
class LLM:
    def __init__(self, name: str, quant: str = "4bit"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(device_map="auto", torch_dtype=torch.float16)
        if quant == "4bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
        self.torch = torch

    def generate(self, user_msgs, m: int, temperature: float, max_new_tokens: int):
        chats = [self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True) for u in user_msgs]
        enc = self.tok(chats, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                                      num_return_sequences=m, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(gen, skip_special_tokens=True)
        return [texts[i * m:(i + 1) * m] for i in range(len(user_msgs))]


class Similarity:
    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)

    def __call__(self, ref, texts):
        e = self.model.encode([ref] + list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return (e[1:] @ e[0]).tolist()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="cids_prompts.csv")
    ap.add_argument("--target", choices=["train_style", "heldout"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--quant", choices=["4bit", "fp16"], default="4bit")
    ap.add_argument("--sim_model", default="pritamdeka/S-PubMedBert-MS-MARCO")
    ap.add_argument("--out_dir", default="narratives")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--retries", type=int, default=2, help="extra rounds if < k candidates pass")
    ap.add_argument("--limit", type=int, default=0, help="only first N cases (smoke test)")
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.90)
    a = ap.parse_args()

    cfg = FilterConfig(m=a.m, k=a.k, tau=a.tau)
    df = pd.read_csv(a.prompts, keep_default_na=False)
    if a.target == "heldout":
        df = df[df["split"] == "test"]
    if a.limit:
        df = df.head(a.limit)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / f"{a.target}_progress.jsonl"
    done = set()
    if progress.exists():
        done = {json.loads(l)["patient_id"] for l in progress.open(encoding="utf-8")}
    todo = df[~df["patient_id"].isin(done)]
    print(f"{a.target}: {len(df)} cases, {len(done)} already done, {len(todo)} to go")

    model_name = a.model or DEFAULT_MODELS[a.target]
    llm, sim = LLM(model_name, a.quant), Similarity(a.sim_model)
    t0 = time.time()
    rows = todo.to_dict("records")
    counts = {"strict": 0, "relaxed": 0, "insufficient": 0}
    n_cand = n_fact_ok = 0
    bar = tqdm(total=len(df), initial=len(done), desc=f"{a.target}", unit="case",
               file=sys.stdout, dynamic_ncols=True, mininterval=5, smoothing=0.1)
    for b in range(0, len(rows), a.batch_size):
        batch = rows[b:b + a.batch_size]
        facts = [{c: r[c] for c in FIELD_COLS} for r in batch]
        pending = list(range(len(batch)))
        pool = {i: [] for i in pending}
        chosen = {}
        for attempt in range(1 + a.retries):
            msgs = [build_user_message(batch[i]["prompt_S"], facts[i]) for i in pending]
            gens = llm.generate(msgs, cfg.m, a.temperature, a.max_new_tokens)
            still = []
            for i, g in zip(pending, gens):
                pool[i] += score_candidates(batch[i]["prompt_S"], g, facts[i], sim, cfg)
                sel, status = select_top_k(pool[i], cfg)
                chosen[i] = (sel, status, attempt)
                if not status.startswith(("strict", "relaxed")):
                    still.append(i)
            pending = still
            if not pending:
                break
        with progress.open("a", encoding="utf-8") as fh:
            for i, r in enumerate(batch):
                sel, status, attempt = chosen[i]
                fh.write(json.dumps({
                    "patient_id": r["patient_id"], "model": model_name, "status": status,
                    "attempts": attempt + 1,
                    "chosen": [c.text for c in sel],
                    "candidates": [dict(text=c.text, D=round(c.diversity, 3), S=round(c.similarity, 3),
                                        score=round(c.score, 3), strict=c.valid_strict,
                                        relaxed=c.valid_relaxed, problems=c.fact_problems)
                                   for c in pool[i]],
                }, ensure_ascii=False) + "\n")
        for i in range(len(batch)):
            counts[chosen[i][1].split("(")[0]] += 1
            n_cand += len(pool[i])
            n_fact_ok += sum(not c.fact_problems for c in pool[i])
        bar.update(len(batch))
        bar.set_postfix(strict=counts["strict"], relaxed=counts["relaxed"],
                        short=counts["insufficient"],
                        fact_ok=f"{n_fact_ok / max(n_cand, 1):.0%}")
    bar.close()
    print(f"finished {len(rows)} cases in {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- merge back into the prompt table ----
    recs = [json.loads(l) for l in progress.open(encoding="utf-8")]
    prefix = "narrative_" if a.target == "train_style" else "narrative_heldout_"
    full = pd.read_csv(a.prompts, keep_default_na=False)
    by_id = {r["patient_id"]: r for r in recs}
    for j in range(1, cfg.k + 1):
        full[f"{prefix}{j}"] = [
            (by_id[p]["chosen"][j - 1] if p in by_id and len(by_id[p]["chosen"]) >= j else full.at[i, f"{prefix}{j}"])
            for i, p in enumerate(full["patient_id"])]
    merged = out_dir / f"cids_prompts_with_{a.target}.csv"
    full.to_csv(merged, index=False, encoding="utf-8")

    stats = pd.Series([r["status"].split("(")[0] for r in recs]).value_counts()
    cands = [c for r in recs for c in r["candidates"]]
    print("\nstatus per case:\n", stats.to_string())
    print(f"candidates: {len(cands)} | fact-check pass: {sum(not c['problems'] for c in cands)/len(cands):.1%}"
          f" | strict-valid: {sum(c['strict'] for c in cands)/len(cands):.1%}")
    print(f"wrote {merged}")


if __name__ == "__main__":
    main()
