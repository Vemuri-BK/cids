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
| S0: SAM-Med3D embeddings + zero-shot baseline | ✅ done | `data/sam_emb/`; Kaggle dataset `kumarvemuri1/cids-sam-emb` (from sam_emb.zip) |
| Fixed split train 1080 / val 120 / test 306 (+ LOCO columns) | ✅ done | `configs/splits.csv` |
| Upload final `cids_prompts.csv` to Kaggle `cids-assets` | ⏳ todo | new dataset version |
| S1: SegResNet img (Exp 1) / + text FiLM (Exp 2) | ✅ done | test Dice after rel20: img 0.655, txt 0.650 (p=0.94); Day 4 |
| S1 post-processing (connected components, rule chosen on val) | ✅ done (rel20) | `postproc_*` in s1_img / s1_txt outputs |
| S2a: SAM-Med3D-turbo + decoder LoRA, image only (Exp 3) | ✅ done (oracle prompts) | test Dice box 0.708, click1 0.650, click5 0.717; zero-shot click1 0.520 |
| Breast-region depth filter on S1 | 🛠 cell ready | not run yet |
| SegResNet → SAM (automatic box, first council step) | ⏳ next | needs re-encoding crops at SegResNet boxes |
| Text refiner diagnostic (4 arms, field tokens) | 💡 proposed | after SegResNet→SAM |
| Baselines (SegResNet, nnU-Net ref., SAM-Med3D FT) | ⏳ | — |
| S2: FIPG + SAM decoder LoRA (member A) | ⏳ | needs `cids-sam-emb` |
| S3: council + dual supervision | ⏳ | — |
| Experiments E0–E5, fairness, ablations | ⏳ | — |
| Paper (IEEE JBHI / TMI) | ⏳ | — |

---

## Day 4 — 2026-10-04 (to 2026-10-05 morning)

### S1 results (test 306, official split; seed 42; best epoch by val Dice)
| Model | Dice raw | Dice rel20 | Dice median | HD95 median | NSD@2mm | missed <0.1 |
|---|---|---|---|---|---|---|
| SegResNet image (Exp 1), best val 0.693 @60 | 0.640 | **0.655 ± 0.250** | 0.740 | 16.0 mm (raw 67.6) | 0.681 | 15 |
| SegResNet + text FiLM (Exp 2), best val 0.693 @60 | 0.634 | 0.650 ± 0.253 | 0.730 | 17.0 mm (raw 61.2) | 0.674 | 16 |
- Text vs image: no overall difference (Dice p=0.94, NSD p=0.44, paired Wilcoxon). Per site (rel20 Dice img/txt): DUKE 0.490/0.512 (p=0.048), ISPY1 0.718/0.723, ISPY2 0.728/0.696 (p=0.011, only one surviving Bonferroni over 4 sites), NACT 0.727/0.741.
- Why text via FiLM does not help (discussed): prompt has no spatial/boundary info, mostly visible in image, template sentences -> near-identical mean-pooled BERT vectors, FiLM is global, zero-init lets model ignore it.
- Power cut during txt run: resumed from last.pt; epoch-45 val recomputed in a separate cell (0.634 < best 0.683, best.pt unchanged).
- DUKE is the failure: Dice 0.49, HD95 ~150 mm, 48/91 cases still >2x over-segmented after post-processing (large connected false regions: heart, chest wall). Small tumours (<5 ml) Dice 0.47. Error types seen in figures: distant FPs (DUKE), under-segmented non-mass tumours (ISPY1), leakage into attached enhancing tissue (ISPY2).

### Post-processing (connected-component analysis, 26-connectivity)
- Rules raw / largest / rel20 (components >= 20% of largest) / min0.1ml; chosen on val (both models picked rel20); applied once to test. img: Dice 0.640->0.655, HD95 median 67.6->16.0 mm, p~1e-22; 7 new misses (small tumour beside larger false blob).
- Breast-region filter (depth behind most anterior body point per axial slice; per-collection threshold = deepest training tumour + 15 mm): cell written and tested on duke_019; not run yet.
- Visualisation cells: best/median/worst per collection; one-per-collection figure with legend (image | GT | pred | error slice | error all slices).

### S2a — SAM-Med3D-turbo + decoder LoRA (71,424 params), image only (Exp 3)
- Bug found by smoke test and fixed (6d8a3d5): first decoder step must get an all-zero mask prompt (as official/S0), not no_mask_embed -> step-1 Dice was 0.01. Stale last.pt from the buggy run was deleted before the clean run.
- Training: 60 epochs x ~20 s; val score 0.683 @5 -> 0.719 @60 (click1 0.675, click5 0.745, box 0.739).
- Test 306, full resolution 1 mm (ORACLE prompts: GT-centred crop, GT clicks/box):
| | Dice | median | HD95 med | NSD | missed |
|---|---|---|---|---|---|
| zero-shot click1 | 0.520 | 0.561 | 9.0 | 0.439 | 12 |
| zero-shot click5 | 0.618 | 0.653 | 6.0 | 0.576 | 6 |
| fine-tuned click1 | 0.650 | 0.690 | 5.9 | 0.644 | 0 |
| fine-tuned box | **0.708** | 0.741 | **4.4** | 0.729 | 0 |
| fine-tuned click5 | 0.717 | 0.745 | 4.6 | 0.738 | 0 |
| GT crop ceiling | 0.932 | | 1.0 | | |
- Per site, box: DUKE 0.678 (SegResNet 0.490), ISPY1 0.715, ISPY2 0.720, NACT 0.747. Small tumours <5 ml: 0.651 vs SegResNet 0.474; >27 ml SegResNet better (0.782 vs 0.741).
- Insight: DUKE/small-tumour failure is localisation, not segmentation -> council (SegResNet box -> SAM refine). Oracle numbers are an upper bound only.

### Reviewed / parked
- FSSM-Net (Inf. Sci. 2027) parked. External suggestion: frozen SegResNet + zero-init text cross-attention residual refiner -> keep as diagnostic with 4 arms (none / no-text / correct / shuffled text), 11 field tokens, hard negatives, judged after rel20 vs 0.655.

### Next
1. SegResNet -> SAM automatic pipeline (crop + box from SegResNet rel20 mask, re-encode, fine-tuned SAM box) vs 0.655.
2. Breast-region filter run. 3. Text refiner diagnostic. 4. S2 FIPG (frequency + text prompts). 5. Council + dual supervision. 6. Final: 3 seeds, longer schedule on A5000s, LOCO, fairness.

---

## Day 3 (evening) — 2026-10-03

### S1 progress (Kaggle, two accounts, Draft Sessions — keep tabs open, Quick Save with output at the end)
- img: ~7.7 min/epoch, val Dice 0.397 @5, 0.528 @10. txt: ~8.7 min/epoch, val 0.385 @5, 0.529 @10. Seed 42.

### S2a — SAM-Med3D-turbo fine-tuned on images only (Exp 3 baseline, variant A)
- Purpose: "just adapt SAM to MAMA-MIA" baseline that the full CIDS SAM branch (FIPG prompts) must beat.
- Frozen encoder → S0 cached post1 embeddings; mask decoder LoRA r=8 α=16 on 14 q/v projections = **71,424 trainable params** (incl. 768 box params). `--tune decoder_full` = 7.58M.
- **Found:** turbo has no usable box prompt (prompt encoder has only 2 point embeddings, box path is 2D-only and would crash) → added 2 learnable 3D box-corner tokens (init = positive-point embedding). So zero-shot has no box row.
- Train: 1080 × 4 cached crops (1 random per epoch), batch 8, AdamW 5e-4, cosine, 2 warm-up epochs, 60 epochs, AMP. Prompt per batch: jittered GT box (±5 vox, p=0.5) or random first click, then 0–4 corrective clicks (SAM-Med3D error-region rule), previous low-res mask fed back; loss BCE + soft Dice on 128³ upsampled logits, averaged over steps.
- Val (120, centred crop): mean(Dice@1 click, @5 clicks, @box). Test (306): zero-shot and fine-tuned with the same seeded clicks; crop Dice (1.5 mm) steps 1–5 for click and box(+clicks); full-res Dice/HD95/NSD at 1 mm for click1, click5, box1.
- Full-res check: GT crop pasted back to 1 mm gives Dice 0.941 / HD95 1.0 mm on duke_019 → resampling ceiling, reported as row `gt_crop_ceiling`.
- Tested: 26 unit tests; end-to-end CPU run on real embeddings (duke_001/002/019) with random weights, resume, decoder_full. Turbo weights not downloadable from this sandbox → first real numbers come from the Kaggle smoke test (zero-shot click1 should be ~0.4–0.6).
- Variant B (encoder LoRA, on-the-fly encoding) planned later, possibly on local A5000s.

### Parked
- FSSM-Net (Zhao et al., Inf. Sci. 2027): low-frequency-guided subband gating, learnable anti-aliasing before Haar, local-only HH path. Parked by vbk until CIDS architecture/results are fixed; possible later ablation (LLL+text gates). Must be cited as closest image-only frequency-gating work.

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
