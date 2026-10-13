# CIDS — Project Log

Daily record of what was done, decided and measured. Newest day at the top.
Times are IST. Commit ids refer to `github.com/Vemuri-BK/cids`.

---

## Status board (update every day)

| Stage | Status | Output / where |
|---|---|---|
| Idea + method design (CIDS-1) | ✅ done, revised Day 7 | this log |
| Repo scaffold (VS Code → GitHub → Kaggle) | ✅ done | `Vemuri-BK/cids` (public) |
| Structured / narrative prompts (text track) | ✅ done, **parked** | text FiLM gave no gain (Day 4); metadata now acquisition-only |
| S0: SAM-Med3D embeddings + zero-shot baseline | ✅ done | Kaggle `kumarvemuri1/cids-sam-emb` |
| Fixed split train 1080 / val 120 / test 306 (+ LOCO columns) | ✅ done | `configs/splits.csv` |
| S1: SegResNet image, 60 → **100 epochs** (warm restart) | ✅ done | test rel20 **0.684** (60 ep: 0.655); `segresnet_img_e100_best.pt` |
| S2a: SAM decoder LoRA, clean GT boxes | ✅ done | test box 0.708 (oracle prompts) |
| S2b: SAM + messy / negative boxes | ✅ done | test box 0.698, messy 0.653, rejects 96.7 % synthetic negatives; `sam_s2b_best.pt` |
| S3a: SegResNet-60 → SAM-S2a, rule-based jury | ✅ done | test 0.675 |
| S3b: SegResNet-100 → SAM-S2b, rule `segjury_0.7` | ✅ done | test **0.699** |
| S2c: SAM trained on SegResNet's real false candidates | ✅ done — **negative result** | rejection ↑ but real tumours lost; AUC not improved |
| **S3d: learned jury (gradient boosting on candidate clues)** | ✅ done — **current best** | **test 0.735**, HD95 3.6, NSD 0.783, p = 0.0008 vs S3b |
| Frequency council code (S4 adapter, S5 maps, S6 council) | ✅ built, tested | `cids/fcouncil/`, `scripts/s4–s6`, notebooks 07, 08 |
| S4: SAM frequency adapter (60 epochs) | 🏃 running (notebook 07) | decides whether 08 uses the adapter |
| S5 + S6: frequency council, dual supervision (main method) | ⏳ next (notebook 08) | must beat 0.735 |
| One-file pipeline export (S3d as `.pt`) | 🛠 code ready, uncommitted | `cids/pipeline.py`, `scripts/s3d_export_pipeline.py` |
| Acquisition-metadata token (geometry → band weights) | ⏳ after S6 | `cond_dim` slot already in adapter / gate |
| Ablations, LOCO, 3 seeds, fairness, faster full-res metrics | ⏳ | — |
| Paper (IEEE JBHI / TMI) | ⏳ | — |

### Results so far (test 306 = official test set; val 120)
| Method (fully automatic unless noted) | Val Dice | Test Dice | HD95 med | NSD | missed |
|---|---|---|---|---|---|
| SegResNet-60, rel20 | 0.693 (best val) | 0.655 | 16.0 | 0.681 | 15 |
| SegResNet-100, rel20 | 0.724 | 0.684 | 7.3 | 0.716 | 14 |
| S3a: SegResNet-60 + SAM-S2a, rules | – | 0.675 | – | – | – |
| S3b: SegResNet-100 + SAM-S2b, `segjury_0.7` | 0.747 | 0.699 | 4.1 | 0.744 | 26 |
| **S3d: SegResNet-100 + SAM-S2b + learned jury** | **0.766** | **0.735** | **3.6** | **0.783** | **17** |
| perfect candidate choice (upper bound, all touching candidates) | 0.812 | 0.787 | – | – | – |
| SAM-S2a / S2b with GT box (oracle prompt, not automatic) | – / 0.729 crop | 0.708 / 0.698 | 4.4 / – | 0.729 / – | – |

---

## Day 10 — 2026-10-13

### S2c-balanced (hard negatives, fixed) — negative result confirmed
- Retrained from S2b with equal batches (20 % real false candidates → empty mask, 20 % real true candidates), lr 5e-5, 30 epochs, val every 2, new selection rule: an epoch can only be "best" if it keeps ≥ 95 % of real tumours on val (`--min_real_keep 0.95`). New metric: AUC of SAM's soft volume (true vs false candidates).
- Epoch 0 = S2b: val AUC 0.801, hard_reject 0.006, real_keep 1.0. Training only moved the operating point (e.g. ep 4: hard_reject 0.669 / real_keep 0.823), **AUC never improved (0.80 → 0.77)** and box Dice fell (0.729 → 0.64). No epoch passed the 95 % rule → best = S2b; test AUC 0.822 = S2b.
- Conclusion: fine-tuning the mask decoder on the finder's false positives trades sensitivity for specificity without improving discrimination. The information is already in S2b's output → use it in a better decision rule.

### S3d — learned jury (commit 9c54ae1, notebook 10) — **new best, test 0.735**
- Same pipeline as S3b (SegResNet-100 → candidates prob > 0.5, ≥ 0.1 ml, top 5 → SAM-S2b crop + box each). Only the decision step changes.
- 11 clues per candidate (no ground truth): log size, size relative to largest, is-largest, rank, number of candidates, SegResNet mean / max prob, agreement Dice(SAM, SegResNet), SAM hard and **soft** volume, SAM / SegResNet volume ratio. Label for training: candidate touches the tumour.
- Train 1080 → 2561 candidates (1246 true / 1315 false); fit logistic regression + gradient boosting. Val 120: choose model + cut-off by Dice → **gboost, t = 0.60** (val 0.766; logreg 0.747). Test applied once. If no candidate ≥ t: keep the most likely one. Final mask = SegResNet mask of kept candidates.
- Dice for any selection computed exactly from per-candidate tables (components disjoint: Dice = 2Σinter / (Σsize + |GT|)); unit test checks it against mask Dice.

| Test 306 | Dice | HD95 | NSD | missed |
|---|---|---|---|---|
| seg_rel20 | 0.684 | 7.3 | 0.716 | 14 |
| segjury_0.7 (S3b) | 0.699 | 4.1 | 0.744 | 26 |
| **learned (gboost, t 0.60)** | **0.735** | **3.6** | **0.783** | **17** |
| oracle | 0.787 | | | |
- Wilcoxon learned vs segjury p = 0.0008, vs rel20 p = 7e-8. Per case: better than S3b in 39, worse in 14 (|Δ| > 0.05); vs rel20 75 / 12.
- Per collection (rel20 / S3b / learned): DUKE 0.544 / 0.587 / **0.674**, ISPY1 0.729 / 0.743 / 0.733, ISPY2 0.753 / 0.752 / **0.793**, NACT 0.720 / 0.719 / 0.626.
- AUC true vs false candidates: jury test 0.916 (val 0.903); single clues val: agree 0.844, SegResNet mean prob 0.866, SAM soft volume 0.801.
- Checks (reproduced locally from the feature CSVs, exact 0.7352): best possible test cut-off 0.75 → 0.741, so the val-chosen 0.60 loses only 0.006 (not over-tuned). NACT drop = 3 of 17 cases (nact_55, 17, 60) where the jury chose the wrong blob (nact_55: true largest candidate got P = 0.05). 17 misses: 15 had a true candidate available, 2 had none. Test has 27 cases (9 %) with no candidate ≥ 0.60 vs 5 in val. Tried fallback "keep largest" → val 0.759 (worse) → rejected.
- Possible later improvements: within-case relative clues (e.g. P relative to the case's best candidate), keeping all fragments of a split tumour.

### S5 update — checker map weighted by the jury (commit c626987)
- `s5_export_maps.py --jury s3d_train_features.csv`: each candidate's SAM map × jury P(tumour), so false blobs enter the council weak.
- **Cross-fitting**: training cases get P from a 5-fold jury that never saw them (in-sample train AUC 0.971 is too optimistic; cross-fitted 0.93 ≈ test 0.916). Full model reproduces test 0.7352.
- Notebook 08: `JURY = {CKPT_DIR}/s3d_train_features.csv`, `MAPS_DIR = maps_jury`, `ADAPTER_CKPT = ''` until S4 is judged.

### One-file export of S3d (code ready, not committed yet)
- `cids/pipeline.py` (`CIDSPipeline.load(...).predict(img) → mask + candidate table`) and `scripts/s3d_export_pipeline.py` → `cids_s3d_pipeline.pt` (~403 MB: SegResNet + full SAM incl. turbo encoder + jury + settings). Save → reload gives identical predictions (tested on CPU, random weights). Kaggle export + check on 5 val cases: later.

### S4 frequency adapter — running (notebook 07, 60 epochs, account 2)
- Smoke OK: epoch 0 = S2b exactly (box 0.699, messy 0.706, neg_reject 1.0) → zero-init works; band weights start at 1.0.
- Decision rule for notebook 08: adapter used only if messy1 clearly better (Wilcoxon p < 0.05) and HD95 / NSD not worse.

### Lab-meeting framing (Day 10)
- "Finder–checker candidate verification with a learned jury" = strong baseline (0.735). "Frequency council" = proposed method (S6). Keep the two names separate.

### Next
1. Read S4 result → set `ADAPTER_CKPT` → notebook 08 once (S5 ~2–3 h + S6 ~4 h). Bar: 0.735.
2. Ablation `USE_TEACHERS = False` (reuse S5 maps).
3. Acquisition metadata: CPU check (does geometry predict band energy?) → geometry token into band weights.
4. Commit + run the S3d one-file export. 5. Ablations, LOCO, 3 seeds, faster full-res metrics, fairness.

---

## Day 9 — 2026-10-12

### S2b — SAM decoder on realistic boxes (commit 86a317e, notebook 05)
- Box prompts like SegResNet's: clean (GT ± 5 vox), messy (0.7–1.8× size, shifted ≤ 40 %), negative (tumour-free tissue → empty target). Val score = mean(box1, messy1, neg_reject); 100 epochs (best val score 0.803).
- Test (full res, oracle box): box 0.698, messy 0.653, rejects 96.7 % of synthetic negative boxes.

### S3b — SegResNet-100 + S2b (commit 85dcc32, notebook 06)
- New rules segkeep / samkeep (drop candidates SAM rejects); per-candidate feature tables (`val/test_candidates.csv`).
- Test: segjury_0.7 **0.699** (vs SegResNet-100 0.684, p = 0.11), HD95 7.3 → 4.1, NSD 0.744, DUKE 0.544 → 0.587, missed 14 → 26; best-candidate oracle 0.741.
- Key finding: S2b rejected only **0.6 %** of SegResNet's real false candidates (vs 97 % synthetic). But SAM volume still discriminates (median 2.2 vs 8.9 ml) and a logistic regression on candidate features gave AUC ≈ 0.90 → motivated S2c and later the learned jury.

### Frequency council built without metadata (commit ba35859)
- vbk's correction: the council = **independent frequency-band experts producing the final mask** (first build that put frequency inside SegResNet was discarded → `outputs/_obsolete_s4_v1/`).
- `cids/fcouncil/`: `bands.py` (3D Haar, groups E1 LLL shape / E2 LLH,LHL,HLL edges / E3 LHH,HLH,HHL corners / E4 HHH texture), `adapter.py` (SamFreqAdapter: Haar → 8 learned band weights → CNN → zero-init residual on SAM's 384×8³ embedding, 333 k params, `cond` slot), `experts.py` (BandExpert small 3D U-Net, ~0.78 M each), `council.py` (gate, uniform at init), `model.py` (FreqCouncil, input 5 ch = image 3 + finder map + checker map; final, experts, weights, uncertainty), `losses.py` (dual supervision: GT on final + every expert, agreement-gated distillation to mean(finder, checker), gate balance), `data.py` (map augmentation).
- Scripts: `s4_sam_adapter.py`, `s5_export_maps.py`, `s6_freq_council.py`; notebooks 07, 08.

### S2c — hard negatives (commits 897f519, dcf8596, notebook 09)
- Mined SegResNet's real candidates with the same crop + box as the pipeline (`s3a.candidate_crops`), frozen-encoder embeddings, crop origins saved: train (prob > 0.3, top 8) 3274 = 1241 true + **2033 false** (62 % of cases have ≥ 1 false); val 307 (166 false); test 836 = 349 true + 487 false (66 % of cases). False blobs small (median ~0.9 ml) vs true ~9 ml. 85 min on T4, 1.6 GB.
- Run 1 (25 % hard / 10 % real-positive batches, lr 2e-4, 100 epochs): test hard_reject 0.844 but **real_keep 0.650**, real Dice 0.42 (S2b 0.595); val box 0.729 → ~0.6. Over-rejection from batch imbalance → Day 10 balanced run.
- Lesson on tooling: files copied to the device under an existing name can arrive as an older cached copy → always copy under a new name and verify md5.

---

## Day 5–8 — 2026-10-05 … 10-11

### S3a — first automatic SegResNet → SAM pipeline (commits 04e22ba, 0be680b, notebook 04)
- Candidates = connected components of SegResNet mask (≥ 0.1 ml, top 5) → one SAM crop + box each → rules (sam / avg / jury / segjury / avgjury). Test 0.675 with SegResNet-60 + S2a.

### SegResNet extended 60 → 100 epochs (commit 50d2e0c, notebook 02)
- Warm-restart LR cycle (`--restart_at 60 --restart_lr 1e-4`), LR reset on resume. Test rel20 0.655 → **0.684**, HD95 16.0 → 7.3, DUKE 0.490 → 0.544.

### Architecture and research question finalised
- Research question: *Can acquisition geometry guide the selection of local frequency features in a 3D foundation model and, combined with a finder–checker council, improve fully automatic 3D breast tumour segmentation on unseen MRI collections?*
- Components: finder SegResNet, checker SAM-Med3D-turbo (frozen encoder + LoRA decoder + 3D box tokens), 3D Haar bands, frequency council of 4 independent band experts, dual supervision (GT + agreement-gated finder/checker teachers), acquisition-geometry metadata (later).
- Metadata: **acquisition fields only** (pixel spacing, slice thickness, view, field strength, manufacturer/model, TE, TR, post-contrast timing, bilateral); patient fields (age, menopause, subtype) only as ablation; ethnicity only for fairness evaluation. Build the architecture **without metadata first**.
- Out of scope for now: agent, diffusion models. "Fully automatic" = no human prompt at test time.

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
