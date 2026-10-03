# CIDS — Project Log

Daily record of what was done, decided and measured. Newest day at the top.
Times are IST. Commit ids refer to `github.com/Vemuri-BK/cids`.

---

## Status board (update every day)

| Stage | Status | Output / where |
|---|---|---|
| Idea + method design (CIDS-1) | ✅ done | this log, Day 1 |
| Repo scaffold (VS Code → GitHub → Kaggle) | ✅ done | `Vemuri-BK/cids` (public) |
| Structured prompts (Type-S) + counterfactuals | ✅ done | `data/prompts/cids_prompts.xlsx/.csv` |
| Narrative prompts (Type-N, Qwen2.5-7B) | ✅ done, re-filtered | 1506 × 3, 0 failing |
| Held-out narratives (Phi-3.5-mini) | ✅ done, re-filtered | 306 × 3, 0 failing |
| S0: SAM-Med3D embeddings + zero-shot baseline | ✅ run done | `data/sam_emb/` unzipped + verified (1200 train, 306 test, sizes match zip); Kaggle dataset `cids-sam-emb` pending |
| Fixed split train 1080 / val 120 / test 306 (+ LOCO columns) | ✅ done | `configs/splits.csv` |
| Upload final `cids_prompts.csv` to Kaggle `cids-assets` | ⏳ todo | new dataset version |
| S1: SegResNet + text (council member B) | 🛠 code ready, tested | `notebooks/02_s1_segresnet.ipynb` (Exp 1 img / Exp 2 txt) |
| Baselines (SegResNet, nnU-Net ref., SAM-Med3D FT) | ⏳ | — |
| S2: FIPG + SAM decoder LoRA (member A) | ⏳ | needs `cids-sam-emb` |
| S3: council + dual supervision | ⏳ | — |
| Experiments E0–E5, fairness, ablations | ⏳ | — |
| Paper (IEEE JBHI / TMI) | ⏳ | — |

---

## Day 2–3 — 2026-10-02/03

### Decisions
- **Validation comes out of the official train set.** Official MAMA-MIA split = 1200 train / 306 **test** (file columns `train_split`, `test_split`; the FADC cache only *named* the folder `val/`). CIDS split: **train 1080 / val 120 (10% per collection, seed 42) / test 306** — test is used only for final numbers. `scripts/make_splits.py` → `configs/splits.csv` (deterministic; also `loco_<collection>` columns for leave-one-hospital-out).
  - Val per collection: DUKE 20, ISPY1 10, ISPY2 85, NACT 5.
  - Note for FADC: if its best epoch was chosen on the 306, those numbers are optimistic.
- SAM-Med3D adaptation = frozen encoder (92.9M) + LoRA r=8 on the 14 q/v attention projections of the mask decoder (~70k params; decoder 7.6M). Ablation planned: LoRA vs full-decoder fine-tune; optional full fine-tune (encoder LoRA) on local A5000s.
- Council member B confirmed: **SegResNet** (MONAI), text via FiLM + cross-attention, 3 input channels (pre, post1, subtraction), 1 mm.

### Analysis
- Zero-shot SAM-Med3D Dice by tumour volume quartile (1 click): 0.36 (<5.3 ml) · 0.51 · 0.57 · 0.62 (>27 ml); 5 clicks 0.50 → 0.67. Only 3.9% of cases < 0.1 Dice, 8.8% > 0.8 → SAM finds the tumour but boundaries/small tumours are poor (32³ decoder output ≈ 6 mm voxels).

### S1 code (2026-10-03)
- `TextSegResNet` (MONAI SegResNet 4.70M; +FiLM 4.86M) — FiLM at bottleneck + each decoder stage, zero-init (text model == image model at step 0; verified: identical epoch-1 loss).
- Input 3 ch [pre, post1, post1−pre] at 1 mm; 96³ patches ×4/patient (⅔ tumour-centred), flips + intensity jitter; DiceCE; AdamW 2e-4, cosine, 60 epochs, AMP; val Dice every 5 epochs (sliding window overlap 0.25) → best.pt; test once (overlap 0.5) → Dice/HD95/NSD@2mm per case + saved masks. Resumable, 10 h time budget.
- Text: BioClinicalBERT mean-pooled (frozen), `prompt_S` with 30% field dropout at train time; full prompt at val/test. `--prompt_mode N/SN` ready for E2/E3.
- Tested: 23 unit tests; end-to-end on 3 real cases (CPU) for img and txt paths, resume, outputs.

### Next
- ✅ Unzipped `data/sam_emb.zip` (Explorer *Extract all*, <1 min); file sets match `configs/splits.csv`, all 1506 sizes match the zip, sampled arrays finite. Upload final `cids_prompts.csv` to `cids-assets`; create `cids-sam-emb`; `git push`.
- Write notebook 02 (S1): SegResNet without text (Exp 1) and with text (Exp 2), run in parallel on two Kaggle accounts.

---

## Day 1 — 2026-09-30 (Wed), ~18:00 → 02:10

### Decisions
- **Project = CIDS: Clinical Instruction Dual Supervision** (chosen from 5 CIDS variants). Required parts: SAM-Med3D, a second model, frequency-converted image-feature prompts, 2-model council incl. SAM-Med3D; data MAMA-MIA 3D DCE-MRI; target IEEE journal.
- Method: SegResNet (text-conditioned) finds the tumour → SAM-Med3D-turbo (frozen encoder) refines with a Frequency-Instructed Prompt Generator (3D Haar DWT bands gated by the clinical instruction, + box) → jury fuses masks. Dual supervision = ground truth (+ band-wise boundary loss) and reliability-weighted peer supervision.
- HHH band is mostly noise → instruction-conditioned soft-threshold shrinkage + boundary-shell masking (ablation rows A–E planned).
- Two prompt types (FairVLM-inspired): Type-S structured vs Type-N narrative; experiments E0–E5; metrics Dice/HD95/NSD, ES-Dice, Δgap, worst-group Dice, prompt sensitivity, counterfactual invariance, gate sensitivity.
- Workflow: local VS Code → push to public GitHub `Vemuri-BK/cids` → Kaggle notebook clones repo, pins `EXPECTED_GIT_COMMIT`. Data never goes to GitHub (`data/` git-ignored); it reaches Kaggle as datasets.
- Reuse the FADC 2-channel cache (ch0 pre, ch1 post-contrast-1, 1 mm iso, RAS) and its default split 1200 train / 306 test (= official MAMA-MIA split).

### Data facts found
- Clinical sheet: `breast_density` 96% missing → dropped from prompts. Excluded as leakage/unusable: pCR, treatment, surgery, follow-up, multifocal_cancer, oncotype, nottingham grade, bilateral_breast_cancer.
- 30 patients have bilateral cancer, but **expert masks always label only the primary tumour in one breast** (checked 26 bilateral-scan cases). Keep these patients; flag them at evaluation (contralateral cancer counts as FP).
- Cache pre/post were percentile-normalised separately → `post1 − pre` is only an approximate subtraction.
- Local `Desktop/mama_mia_cache` is an old **1-channel** cache; the Kaggle 2-channel cache is `kaggle_upload_2ch` (train.zip / val.zip).

### Prompts
- `build_prompts.py` → 1506 rows: `prompt_S`, `prompt_S_hints`, `prompt_S_nosubtype`, `prompt_S_noethnicity`, counterfactuals `cf_age/cf_ethnicity/cf_field_strength` (test only). Excel has a `prompt_guide` sheet explaining every column.
- Narrative runs (Kaggle): Run 1 Qwen2.5-7B-Instruct 4-bit, all 1506 cases, ~2 h; Run 2 Phi-3.5-mini-instruct, 306 test cases (held-out wording).
- Filters: fact check (ours) + Jaccard diversity 0.3–0.5 (fallback 0.2–0.7) + PubMedBERT similarity ≥ 0.90, top-3 by 0.4·D + 0.6·S.
- Bugs found & fixed: bilateral *scan* turned into bilateral *tumour* (3c1b62a); spelled-out phase counts ("four") and "under/over the age of" wrongly rejected (4b68d9a). Fixed offline with `scripts/refilter_narratives.py` (no GPU rerun).
- Final after re-filter:
  - Run 1: fact-pass 77.3% of 8875 candidates; strict 1036 / relaxed 470 / short 0; 0 failing chosen.
  - Run 2: fact-pass 74.6% of 2700 candidates; strict 105 / relaxed 201 / short 0; 0 failing chosen.
  - Logs: `data/narratives/refiltered_train_style_log.csv`, `refiltered_final_log.csv`.

### S0 — SAM-Med3D-turbo embeddings (Kaggle, 2nd account `kumarvemuri1`)
- Model: `medim` package (official) + `blueyo0/SAM-Med3D/sam_med3d_turbo.pth`, strict key check. Input 128³ at 1.5 mm, z-norm over voxels > 0; embedding 384×8³; low-res mask 32³.
- Crops: train 1 centred + 3 jittered (±16 vox), test 1 centred. All tumours fully inside the 192 mm crop. 1506 cases, 36.9 min, 0 errors, 3.5 GB.
- **Zero-shot baseline (test crops, oracle GT clicks), mean Dice:**

  | clicks | post1 | subtraction |
  |---|---|---|
  | 1 | 0.5145 | 0.2885 |
  | 3 | 0.5765 | 0.4180 |
  | 5 | 0.6136 | 0.4942 |

  1 click per collection (post1 / subtraction): DUKE 0.475/0.204 · ISPY1 0.541/0.414 · ISPY2 0.529/0.271 · NACT 0.514/0.385.
- **Decision: SAM input = post1.** SegResNet keeps both channels.

### Commits
b60616c scaffold · 2fd4d94 notebook without token + commit pin · e7ed042 prompt_guide sheet · 74b4129 tqdm bar · 9166366 GPU check · 3c1b62a bilateral-tumour check · 6e3004e exclusion note · 0a7a0bd S0 · 4b68d9a fact-check fixes + refilter script

### Open items for Day 2
1. Unzip `data/sam_emb.zip` on Windows (`tar -xf sam_emb.zip`) → verify 1200/306 files.
2. Upload final `data/prompts/cids_prompts.csv` as a new version of Kaggle dataset `cids-assets`.
3. Create Kaggle dataset `cids-sam-emb` from S0 Version #2 output (check its table matches Day 1 numbers).
4. `git push` (local is ahead of GitHub).
5. Build S1: SegResNet + BioClinicalBERT (FiLM + cross-attention), 3-channel input (pre, post1, subtraction), 5-fold/LOCO splits, baseline without text.
