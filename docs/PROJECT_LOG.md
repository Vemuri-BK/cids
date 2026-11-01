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
| S3d: learned jury (gradient boosting on 11 candidate clues) | ✅ done | test 0.735, HD95 3.6, NSD 0.783, p = 0.0008 vs S3b |
| **S3e/S3f: jury + memory of known candidates (13 clues)** | ✅ done — **final pipeline** | **test 0.7495**, HD95 med 3.32, NSD 0.799, missed 10; +0.066 vs SegResNet (p = 5e-13); notebooks 11, 12 |
| Frequency council code (S4 adapter, S5 maps, S6 council) | ✅ built, tested | `cids/fcouncil/`, `scripts/s4–s6`, notebooks 07, 08 |
| S4: SAM frequency adapter (60 epochs) | ✅ done — not used | messy +0.013, clean −0.008 vs S2b |
| S5 + S6: frequency council, dual supervision | ⛔ stopped — no gain | val council 0.7236 ≈ finder 0.7238 (Day 11–16) |
| Code review fixes (vbk review 2026-10-14) | ✅ done | 10 fixes, 45 tests (Day 11–16); 73 tests now |
| S7: frequency boundary refiner of the jury mask (v1, arms A/B, learning test, run C) | ✅ done — no gain | val ≈ M0 0.7742; learns on fitted cases, not on unseen ones (Days 17–33) |
| Lab server (~/CIDS, `user@iiser`, 2× A5000; runs on **GPU 0 only**) | ✅ done | reproduces Kaggle SegResNet exactly (test raw 0.671) |
| Out-of-fold SegResNet maps (3 folds + full) | ✅ done | `outputs/seg_oof/{fold0,fold1,fold2,full}`; held-out Dice ≈ 0.70 |
| s3g: honest (out-of-fold) jury masks for training | ✅ done | train M0 Dice 0.808 (in-sample) → **0.742** (OOF); `outputs/s3g_oof_masks` |
| S7 refiner v1-new / v2 (OOF masks, ± SegResNet probability) | ✅ done — not significant | val 0.7752 / 0.7761 vs M0 0.7742; probability channel no help (p = 0.48) |
| Frequency ablation of the refiner (spatial / LLL / all / minus HHH) | ⏳ next (optional) | `spatial`, `freq_lll` ready; minus-HHH needs code |
| Calibration study of the refiner checkpoint | 💡 proposed | reliability/ECE, temperature, confident edits, case-level safety rule |
| nnU-Net baseline on our split (guide's SOTA question) | ⏳ after guide meeting | ~1–2 days on GPU 0 |
| One-file pipeline export (S3d as `.pt`) | 🛠 code ready, uncommitted | `cids/pipeline.py`, `scripts/s3d_export_pipeline.py` |
| Acquisition-metadata token (geometry → band weights) | ⏳ later | `cond_dim` slot already in adapter / gate |
| Ablations, LOCO, 3 seeds, fairness, faster full-res metrics | ⏳ | — |
| Paper (IEEE JBHI / TMI) | ⏳ | — |

### Results so far (test 306 = official test set; val 120)
| Method (fully automatic unless noted) | Val Dice | Test Dice | HD95 med | NSD | missed |
|---|---|---|---|---|---|
| SegResNet-60, rel20 | 0.693 (best val) | 0.655 | 16.0 | 0.681 | 15 |
| SegResNet-100, rel20 | 0.724 | 0.684 | 7.3 | 0.716 | 14 |
| S3a: SegResNet-60 + SAM-S2a, rules | – | 0.675 | – | – | – |
| S3b: SegResNet-100 + SAM-S2b, `segjury_0.7` | 0.747 | 0.699 | 4.1 | 0.744 | 26 |
| S3d: SegResNet-100 + SAM-S2b + learned jury (11 clues) | 0.766 | 0.735 | 3.6 | 0.783 | 17 |
| **S3e/S3f: + memory + probe clues (13 clues) = final** | **0.774** | **0.7495** | **3.32** | **0.799** | **10** |
| S7 refiner v1-new on S3f mask (val only, best epoch) | 0.7752 | not run | 3.00 (val) | 0.8225 (val) | – |
| S7 refiner v2 (+ SegResNet probability, val only) | 0.7761 | not run | 2.91 (val) | 0.8244 (val) | – |
| best candidate subset (oracle, upper bound of any jury) | 0.812 | 0.790 | – | – | – |
| keep all touching candidates (old 'oracle') | 0.812 | 0.787 | – | – | – |
| SAM-S2a / S2b with GT box (oracle prompt, not automatic) | – / 0.729 crop | 0.708 / 0.698 | 4.4 / – | 0.729 / – | – |

---

## Days 17–33 — 2026-10-20 → 2026-11-01: frequency boundary refiner, out-of-fold maps, lab server

### Summary
- Final pipeline stays **SegResNet-100 → SAM-S2b → 13-clue memory jury (S3f)**: test Dice 0.7495, HD95 median 3.32 mm, NSD 0.799, missed 10, +0.066 vs SegResNet alone (p = 5e-13).
- The frequency boundary refiner (S7) was tested in 7 runs. With honest training masks it learns on training cases, but on new patients the gain is not significant (≤ +0.002 Dice). **The jury mask remains the final output.**
- Project moved to the lab server; out-of-fold SegResNet maps and honest training masks built there.

### S7 refiner design (`cids/refine/`, `scripts/s7_refine.py`)
- Edits only the jury mask M0, within **8 mm** of its boundary. Output z = z0 + allow · sigmoid(gate) · delta, with z0 = ±log 3 and zero-init delta/gate → epoch 0 = M0 exactly (checked: per-case parity with notebook 12 `val_metrics.csv`, max |diff| 0).
- Input: MRI (3 ch) + M0 + allowed band (+ optional SegResNet probability = v2).
- Modes: `freq` (3D Haar, all 8 bands, learn per band, inverse), `freq_lll` (detail bands zeroed), `spatial` (same network, no wavelet split). Optional output gate (`--no_gate`). Losses: masked dice_bce or `balanced` (helpful vs harmful edit groups).
- Training: ROI crops, error-aware patch sampling (25/25/50), 8 patches/case, 60 epochs; val every 3 epochs with edit diagnostics (added/removed × helpful/harmful voxels) and a train-check on 120 unaugmented training cases.
- S7 oracle (commit e1d859b): upper bound of an editable band of 4 / 8 / 12 mm → 8 mm chosen.

### S7 experiments and results (val 120; M0 = 0.7742)
| Run | Training masks | Inputs | Key result | Verdict |
|---|---|---|---|---|
| v1 arms A / B (Kaggle) | in-sample (S3f refit) | MRI + M0 | val ≈ M0 | no gain |
| Learning test (16 fitted + 16 unseen cases) | in-sample | MRI + M0, ± gate, dice_bce / balanced | fitted ROI 0.837 → **0.887**; unseen 0.796 → **0.761**; gate no effect; balanced loss over-edits at w_keep = 1 | can learn, does not transfer |
| Run C (60 ep, 8 patches, error sampling) | in-sample | MRI + M0 | best val 0.7747; train-check +0.002 | no gain |
| **v1-new** (60 ep, server) | **OOF (s3g)** | MRI + M0 | best val **0.7752** (ep 48); train-check 0.7165 → 0.7322 (+0.016); NSD 0.8204 → 0.8225; HD95 med 3.00 → 3.00 | tiny, not significant |
| **v2** (60 ep, server) | **OOF (s3g)** | MRI + M0 + SegResNet prob | best val **0.7761** (ep 48); train-check 0.7165 → 0.7291 (+0.013); NSD 0.8244; HD95 med 2.91 | not significant |

Per-case statistics at the best epoch (bootstrap 10 000, Wilcoxon signed-rank):

| | Mean gain | 95 % CI | Median | Wilcoxon p | improved / worsened (any) | > 0.01 better / worse |
|---|---|---|---|---|---|---|
| v1-new | +0.0011 | [−0.0021, +0.0039] | +0.0010 | 0.002 | 77 / 40 | 12 / 10 |
| v2 | +0.0019 | [−0.0021, +0.0058] | +0.0003 | 0.24 | 64 / 53 | 15 / 16 |
| v2 − v1-new | +0.0008 | [−0.0022, +0.0042] | 0.000 | 0.48 | 57 / 60 | 12 / 15 |

- Counterfactual on v2 (exact from voxel counts): removals only 0.7753, additions only 0.7750, both 0.7761 → no edit-type filter rescues it.
- Same largest losses in both runs (gains correlate, r = 0.61): ispy2_775346 (−0.09/−0.10), ispy2_788149 (−0.08/−0.06), duke_132 (−0.07, v1-new) → the refiner removes real tumour where tumour and parenchyma enhance alike.
- Late epochs: ~1 600–2 000 changed voxels per case, better ≈ worse; val returns to ≈ M0.

### Conclusions
1. **Honest (out-of-fold) training masks are what make the refiner learn** (train-check +0.016 vs +0.002 with in-sample masks). The SegResNet probability channel adds nothing (v2 − v1-new p = 0.48).
2. On unseen patients the refiner only smooths the border slightly (median +0.001); a few large wrong removals cancel it → mean gain not significant.
3. **Little headroom at the border**: on val, the 14 cases with M0 Dice < 0.5 carry **37 %** of all remaining Dice error (the 56 cases ≥ 0.85 only 19 %). These are region errors (wrong candidate, missed tumour, large FP) that a border refiner cannot fix; fixing them to 0.85 would give val 0.841.
4. Not yet tested: whether the Haar split itself helps (all runs used `freq`). The matched `spatial` / `freq_lll` / minus-HHH comparison is the open frequency ablation.

### Out-of-fold SegResNet maps (`scripts/make_folds.py`, notebook 20, server)
- `configs/splits.csv`: `fold3_0/1/2` (3 folds of train, stratified by collection, seed 42; values train / heldout / val / test).
- `s1_train_segresnet.py`: `--split_col`, `--eval_split` (list), `--save_prob` (uint8 probability maps).
| Fold | Held-out cases | Held-out Dice |
|---|---|---|
| fold 0 | 361 | 0.705 |
| fold 1 | 360 | 0.703 |
| fold 2 | 359 | ≈ 0.70 |
| full (existing SegResNet-100, val / test raw) | 120 / 306 | 0.709 / **0.671** (= Kaggle; 0.684 after rel20) |
- Kaggle lessons: 500-output-file limit (fold 0 first lost 47 maps → notebook 20 zips and verifies outputs, recovery mode from best.pt); notebook cells don't update from GitHub (re-import the .ipynb); all notebooks print one line per epoch (`sh()`).

### s3g — honest jury masks for training (`scripts/s3g_oof_train_masks.py`)
- Training M0 rebuilt from OOF SegResNet candidates → SAM-S2b → S3f cross-fit memory jury (never the patient's own fold). Val/test masks copied unchanged (426).
- Train M0 Dice **0.8077 (in-sample) → 0.7417 (OOF)** — now matches val (0.774) / test (0.750) difficulty. SegResNet rel20 OOF 0.711, oracle 0.789, jury AUC 0.923.
- 989 / 1080 patients keep the same candidates as notebook 12 (scikit-learn 1.9.1 on the server vs Kaggle) → parity check uses `--parity_tol 0.01`.

### Lab server (from 2026-10-28)
- Linux, `user@iiser` (10.67.0.12), 2× RTX A5000 24 GB, 125 GB RAM; VS Code Remote-SSH from the laptop.
- Project `~/CIDS`, env `~/venvs/cids`, MRI cache `~/data` (train/, val/), long runs in tmux, logs via plain `| tee` (awk filter buffered output).
- **All training on GPU 0**; GPU 1 drives the display and is left free for others.
- Missing packages installed for medim: `requests einops appdirs huggingface_hub` (to add to requirements.txt).
- 2026-10-30: v1-new crashed at epoch 24 with `CUDA error: unspecified launch failure`; dmesg **Xid 31 (MMU fault) + Xid 32**, no thermal/ECC Xid → one-off driver/AMP fault; persistence mode enabled (`nvidia-smi -pm 1`); rerun completed without error.

### Next
1. (Optional, ~2 h each on GPU 0) frequency ablation of the refiner: `spatial`, then `freq_lll`, minus-HHH — answers "does the Haar split matter?" for the paper.
2. (Proposed) calibration study on the v1-new checkpoint: reliability/ECE in the 8 mm band, temperature scaling, confident-edits-only τ ∈ {0.6, 0.7, 0.8, 0.9}, case-level max-change rule; choose on val, test once.
3. nnU-Net baseline on our split (guide's state-of-the-art question; also as a stronger finder for the 14 region-level failures).
4. Ablation table, LOCO, 3 seeds, MAMA_MIA_COMPLETE transfer, paper.

---

## Days 11–16 — 2026-10-14 → 2026-10-19

### S4 frequency adapter (notebook 07, 60 epochs) — small, mixed effect
- Messy-box Dice +0.013 over S2b, clean-box Dice −0.008. Did not meet the bar ("messy clearly better and HD95/NSD not worse") → notebook 08 ran **without** the adapter (`ADAPTER_CKPT = ''`).

### S5 + S6 frequency council (notebook 08) — stopped, no gain
- S5 maps took ~3.3 h; S6 then trained for its full 10 h budget (run > 14 h in total).
- Val council ≈ finder (0.7236 vs 0.7238). Possible explanation (not proven by similar mean Dice): the council relied on the finder map instead of improving on it.
- Likely causes: (1) training-case finder maps are in-sample (SegResNet saw those cases) → maps look cleaner than on new cases (stacking leak); (2) patches are tumour-centred; (3) distillation targets were the *augmented* maps (fixed Day 16).
- Decision (vbk, 2026-10-14): drop this run; the frequency idea moves to **boundary refinement of the jury mask** (see Next).

### Code review by vbk (2026-10-14) — `docs/CODEBASE_REVIEW_2026-10-14.md`, `docs/review_2026-10-14/`
- Checked against our numbers:
  - "Oracle" (keep all touching candidates) is not the true maximum → best candidate subset (all ≤ 31 subsets per case): **test 0.7901**, val 0.812 (touching: 0.7873 / 0.8119). Report **0.790**.
  - S3's `seg_rel20` (rel20 among top-5 candidates) ≠ shared rel20 — but identical on our data (test 0.6835 both). Results stand.
  - S2c selection bug is real, but the actual run selected ≈ S2b, so its conclusion stands.

### Second review (vbk, `docs/CODE_REVIEW_2026-10-19.txt`) — provenance fixes (52 tests pass)
- `cids/provenance.py`: content hash (sha256) of every checkpoint; cases digest (ids + cache file sizes + preprocessing version); unique `*.stale-<time>` archives (2nd invalidation failed on Windows); atomic `.npz` writes; required-case checks.
- Cached S3d/S3e tables: manifest now uses content hashes + cases digest.
- S5 maps: `_maps_manifest.json` (models by hash, settings); a folder with other or unknown maps is refused; a split is marked complete only when all its cases exist; S6 refuses incomplete / foreign map folders.
- S6 resume: `best.pt` hash stored in `last.pt` and checked on every resume (same or new folder); the maps identity must match; a recorded best is never replaced by current weights.
- All scripts: missing cases → error with a list (no silent smaller evaluation); summaries record case counts and `--limit`.
- S3e preflight: ≥ 2 patients, both classes, k ≥ 1.
- Notebooks 08, 11: shell steps via `sh()` (checked subprocess) → Run All stops on any failed step (pip, pytest, smoke, runs).
- Overlap: S6 validation 0.25 / test 0.5 kept for speed; finder and council always compared at the same overlap (documented).
- Open (next step): S3e bundle + integration into S5 / pipeline; full S3e metrics (HD95/NSD).
- S3e run files kept in `outputs/s3e_run_2026-10-19/` (summary, feature tables, train descriptors) for re-checking 0.7495.

### Bug fixes from the review (2026-10-19, 45 tests pass)
| Fix | Scripts |
|---|---|
| S2c: epoch 0 (= S2b) saved as fallback best; an epoch keeping < `--min_real_keep` of real tumours can never be best | s2a |
| S4: epoch 0 (= S2b, zero adapter) saved as best; the adapter must beat it | s4 |
| S6: resuming into a new folder copies the old `best.pt` (error if missing) | s6 |
| Cached feature tables carry a manifest (checkpoints, cases, overlap, version); mismatch → old files renamed `*.stale`, split recomputed | s3d, s3e |
| `seg_rel20` = shared rel20 everywhere; old rule kept as `cand_rel20` | s3a, s3d, s3e |
| `oracle` = best candidate subset; old one kept as `oracle_touching` | s3d, s3e |
| LOCO full-res metrics read the cache folder of the *official* split | s2a |
| Council distillation targets = clean maps (inputs stay augmented) | fcouncil/data.py, s6 |
| S5 time budget stops all splits and exits non-zero (S6 can't start with missing maps) | s5 |
| scikit-learn, tqdm in requirements | requirements.txt |

### S3e — memory clue for the learned jury (notebook 11) — **new best, test 0.7495**
- Idea (vbk, from cache-augmented generation): compare each candidate with a memory of KNOWN training candidates.
- Descriptor per candidate: SAM's frozen embedding (384×8³), mean over the box tokens ‖ mean over the crop → 768-d, L2-normalised (`s3a.embedding_descriptor`).
- Memory = 2561 training candidates (1246 real / 1315 false).
- Two new clues:
  - `mem_knn`: similarity-weighted share of real tumours among the 10 nearest memories (cosine, softmax T 0.05).
  - `mem_probe`: logistic regression on the descriptor (C 0.1).
- Both are cross-fitted by patient on train (5 folds, never the candidate's own patient; unit test). Val/test use the full memory.
- Same gradient-boosting jury and val-chosen cut-off as S3d.

| Variant (test 306) | Val Dice | Test Dice | Δ vs jury11 | better / worse (>0.05) | Wilcoxon p | test AUC |
|---|---|---|---|---|---|---|
| jury11 (= S3d, reproduced) | 0.766 | 0.7352 | — | — | — | 0.916 |
| jury11 + memory | 0.773 | 0.7404 | +0.005 | 14 / 9 | 0.39 | 0.931 |
| **jury11 + memory + probe (13 clues)** | **0.774** | **0.7495** | **+0.014** | **17 / 4** | **0.053** | **0.938** |

- Robustness (reproduced locally from the CSVs):
  - Better than jury11 at every cut-off 0.3–0.9 (0.741–0.751 vs 0.729–0.741).
  - 95 % bootstrap CI of the gain [+0.004, +0.026].
- Per collection (jury11 → 13 clues): DUKE 0.674 → 0.701, ISPY1 0.733 → 0.741, ISPY2 0.793 → 0.798, NACT 0.626 → 0.668.
- Single-clue AUC (test): mem_knn 0.795, mem_probe 0.762, sam_soft_ml 0.822, agree 0.834. Weak alone, but complementary.
- Biggest losses: ispy2_894673 −0.76, duke_560 −0.36, duke_567 −0.22, duke_099 −0.15.
- Gap to the oracle (0.790) closed from 0.055 to 0.041.
- Run: 113 min (T4). Memory version B (richer descriptor) skipped: little headroom left in candidate selection.

### Next
1. **Frequency boundary refinement** of the 13-clue jury mask (vbk): frequency bands refine the kept mask's boundary before the final output — no new masks. The only way past the 0.790 selection ceiling. Read `docs/review_2026-10-14/04_FREQUENCY_MASK_REFINEMENT_RESEARCH.txt` first.
2. Full metrics (HD95/NSD, missed) for the 13-clue jury; S5/pipeline use the 13-clue jury.
3. Later: one-file pipeline export, LOCO, 3 seeds, paper update (oracle 0.790, S3e).

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
