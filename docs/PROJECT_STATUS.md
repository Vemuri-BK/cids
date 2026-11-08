# CIDS project status (as of 2026-11-09)

Project: fully automatic 3D breast tumour segmentation on breast DCE-MRI (MAMA-MIA; split train 1080 / val 120 / test 306), plus a QC-guided one-click review step. Target: MIDL 2027 (abstract Nov 30, paper Dec 4). Lab server is the official record (`~/CIDS/outputs`); a results mirror is on the laptop (`cids project/outputs/`, git-ignored).

---------------------------------------------------------------------------------------------------
## 0. Summary for the paper
- **Automatic pipeline (frozen, tested once):** SegResNet-100 finder -> radiomics jury B0_rad. Test Dice **0.7479 [0.7218, 0.7727]**, +0.0247 over the jury without radiomics (CI above 0).
- **QC-guided one-click review (S3v, pre-registered, train OOF):** a click on the 10% of scans flagged by the QC score removes **17** Dice<0.1 failures (random: 3.5); diff +13.5 [5.9, 21.2].
- **SAM's role (S3v R4, out-of-fold decoders):** in the review step, when the click lands on tumour SegResNet did not segment well, SAM is the better tool: chosen for 30 patients, **+0.204 Dice [0.075, 0.327]** over the nearest SegResNet blob (22/30 better); +0.005 [0.002, 0.009] mean Dice over all patients at the 10% budget.
- **SAM in the automatic pipeline:** 7 pre-registered roles, none passed (S3l, S3m, S3n, S3p, S3q, S3r, S3s); one-click biomarker tool (S3u): SAM adds nothing over the nearest blob on average.

---------------------------------------------------------------------------------------------------
## 1. Final automatic method (frozen, tested once, 2026-11-07)

SegResNet-100 finder (3 channels: pre, post1, post1 - pre) -> candidates (p > 0.5, >= 0.1 ml, top 5) -> **binary radiomics jury B0_rad** (24 clues: 7 SegResNet + 15 radiomics + rel_seg_prob + rel_ring_cnr; HistGradientBoosting; threshold 0.40 chosen in inner folds), fitted on the 1,080 OUT-OF-FOLD training candidates -> output = union of the kept SegResNet blobs. Fully automatic. **No SAM.**

- Frozen: `scripts/s3o_final_test.py --step freeze` (val reproduced S3k exactly: B0_rad 0.7727, B0 0.7369). Model: `outputs/s3o_final/frozen/b0_rad.pkl` (+ manifest with md5). Finder: `checkpoints/segresnet_img_e100_best.pt`.
- Tested once: `--step test` (refuses a second run or a changed model). Outputs `outputs/s3o_final/test/`.

### Test (306 patients; benchmark seen by earlier pipeline versions, stated as a caveat)
| Policy | Dice [95% CI] | Median | HD95 med (n) | NSD@2mm | Dice<0.1 | Dice<0.5 |
|---|---|---|---|---|---|---|
| **B0_rad (final)** | **0.7479 [0.7218, 0.7727]** | 0.823 | 3.46 mm (305) | 0.796 | 14 | 39 |
| B0 (7 finder clues) | 0.7232 [0.6923, 0.7521] | 0.819 | 3.61 | 0.769 | 25 | 46 |
| SegResNet + rel20 | 0.6835 | 0.767 | 7.28 | 0.716 | 14 | 61 |
| keep-all candidates | 0.6736 | 0.733 | 32.3 | 0.698 | 8 | 61 |
| perfect selection (ceiling) | 0.7901 | - | - | - | - | - |
- Ablation B0_rad - B0: **+0.0247 [0.0102, 0.0416]**; 34 better / 28 worse / 243 identical; rescued below 0.1: 12 (harmed 1).
- Per hospital: DUKE 0.693 (n=91; 9 of the 14 Dice<0.1), ISPY1 0.742 (67), ISPY2 0.793 (131), NACT 0.717 (17).
- Excluding bilateral cancer (295): 0.7548; bilateral cancer only (11): 0.563.
- Development: train OOF B0_rad 0.7532 (Dice<0.1: 60, <0.5: 137); val 0.7727 (4 / 13).

---------------------------------------------------------------------------------------------------
## 2. Decision-layer experiments (train OOF, matched, frozen rules, identical folds/seeds)
Frozen two-route rule: route 1 = CI lower bound of paired dDice > 0 and #Dice<0.1 not higher; route 2 = #Dice<0.1 <= 0.75 x reference and mean Dice >= reference - 0.005; both: #Dice<0.5 <= 1.10 x reference.

| Step | Comparison | Train OOF Dice (Dice<0.1 / <0.5) | Paired dDice [95% CI] | Candidate AUC | Verdict |
|---|---|---|---|---|---|
| E0 | recall-first ceiling | tumour among candidates 98.7% | - | - | recall-first dropped |
| S3i | subset-selection pilot | U0 0.8119 / U1 0.8116 / B0 0.8105 (in-sample) | within noise | - | no SAM gate |
| S3j | U0 vs binary juries | U0 0.7475 (46/145), B0 0.7398 (69/152) | U0-B0 +0.0077 [0.0009, 0.0143] | - | binary B0 main |
| S3k | B0 -> + radiomics (B0_rad) | 0.7398 -> **0.7532** (60/137) | **+0.0134 [0.0053, 0.0214]** | 0.900 -> **0.958** | **radiomics qualify** |
| S3l | B0_rad -> + 4 SAM box features | 0.7508 (60/142) | -0.0024 [-0.0073, 0.0025] | 0.959 | no |
| S3l check | B0 -> + 4 SAM features | 0.7386 (75/153) | -0.0012 [-0.0057, 0.0030] | 0.899 | no |
| **S3r** | B0_rad (A) vs + non-SAM 2nd anchor (B) vs + SAM memory + SAM anchor (C) | A 0.7532 (60/137), B 0.7561 (60/135), C 0.7565 (56/130); val 0.7727 / 0.7821 / 0.7726 | C-A +0.0033 [-0.0018, 0.0086]; C-B +0.0004 [-0.0055, 0.0063] | 0.958 / 0.962 / 0.961 | SAM memory does not qualify |

S3r pre-check (train OOF): finder's top blob wrong in 74 patients; B0_rad wrong picks 46 (36 with a wrong finder anchor); SAM memory (mem_knn) top candidate hits the tumour in 32/46 vs 22 for the best non-SAM pick, but points elsewhere in 173 of 1,020 correct patients -> net only 11 rescued / 7 new failures in the jury.

---------------------------------------------------------------------------------------------------
## 3. SAM-Med3D-turbo (S2b LoRA decoder) -- all roles tested (development data, pre-registered)
| # | Role | Study | Frozen result | Verdict |
|---|---|---|---|---|
| 1 | Jury clues (box decoder) | S3l | no net benefit | no |
| 2 | Automatic contour of kept blobs | S3m | D vs A dNSD -0.142 [-0.149, -0.135] | no |
| 3 | QC review flag | S3n | 65 vs 65 failures caught at 10% | no |
| 4 | Click repair editor | S3p v2 | TS vs brush dNSD -0.084 [-0.137, -0.035] | no |
| 5 | Whole-breast outline (feasibility) | S3q | 6 rescuable < 10 | stop |
| 6 | Memory + second anchor in the jury | S3r | C-A +0.0033 n.s.; C-B +0.0004 | no |
| 7 | Multi-prompt stability as a clue | S3s | turbo AUC 0.49, 9/46 hits (chance 17.7) | no |
| 8 | One-click biomarker tool | S3u | equal to nearest SegResNet blob (CCC 0.899 vs 0.901) | no |
| **9** | **Backup tool in QC-guided one-click review** | **S3v R4** | **+0.204 Dice [0.075, 0.327] on 30 chosen patients; +0.005 [0.002, 0.009] overall at 10%** | **PASS** |
Paper wording: "SAM did not improve the automatic pipeline in any of seven pre-registered roles, nor a one-click biomarker tool, but in a QC-guided one-click review it is the better tool when the click lands on tumour the specialist model did not segment (out-of-fold decoders)."

### S3m -- SAM as automatic boundary refiner (train OOF)
| Arm | Dice | NSD@2mm | HD95 med | Dice<0.5 |
|---|---|---|---|---|
| A SegResNet blob | **0.7532** | **0.805** | **3.0 mm** | **137** |
| B SAM box | 0.6530 | 0.667 | 5.4 | 194 |
| C box + geometric point | 0.6594 | 0.665 | 5.2 | 191 |
| D box + enhancement point | 0.6582 | 0.663 | 5.4 | 191 |
| E region growing | 0.3873 | 0.478 | 13.3 | 708 |

### S3n -- SAM disagreement as QC flag
10% budget: baseline 65 / +SAM 65 of 137 failures (AUROC 0.877 vs 0.875). The cheap baseline QC score (7 features) catches 47% of failures in 10% of scans -> reused in S3v.

### S3p -- simulated click repair (108-patient review queue)
v2: TS vs TB dNSD -0.084 [-0.137, -0.035], harmed 29 vs 7; TS_hybrid +0.026 [-0.001, 0.055]. SAM's first click recovers missed tumour fastest (wrong target 0.250 vs 0.174 after 1 click).

### S3q -- location audit
36 wrong picks; 6 rescuable by a breast outline (< 10) -> stop; max gain ~+0.004 Dice. Most wrong picks are choices between two in-breast enhancing regions.

### S3s -- multi-prompt stability (idea from multi-text joint prompts SAM, JBHI 2026)
Pre-check on 2,709 train OOF candidates (22 min). Gate needed >= 27/46 hits and <= 100 misleads:
| Score | Wrong-pick hits (46) | Misleads (1,020 correct) | AUC |
|---|---|---|---|
| finder anchor (seg_prob_mean) | 10 | 38 | 0.861 |
| peak SegResNet prob | 22 | 53 | 0.810 |
| SAM memory | 32 | 173 | 0.799 |
| turbo stability (primary) | 9 | 380 | 0.490 |
| S2b stability | 14 | 193 | 0.703 |
Turbo returns the same broad mask for any click (stability 0.96 on tumours and false blobs alike) -> FAIL.

### S3u -- one-click masks for radiomic biomarkers (val 120)
Click = deepest point of the expert tumour (simulated). 17 features (volume, diameter, enhancement, shape, contrast, texture).
| Mask | Mean Dice | Dice<0.1 | Median CCC vs expert (all 120) |
|---|---|---|---|
| B automatic B0_rad | 0.773 | 4 | 0.821 |
| C click inside a SegResNet blob | 0.767 | 8 | - |
| Cn click -> nearest SegResNet blob (post-hoc) | **0.808** | **0** | **0.901** |
| S SAM alone (one click) | 0.617 | 1 | 0.519 |
| H blob under click, else SAM | 0.806 | 0 | 0.899 |
- Cn - B median CCC +0.080 [0.007, 0.125] (post-hoc); H - Cn -0.002 [-0.071, 0.020].
- 8 patients where the click missed every blob: nearest blob 0.616 vs SAM 0.589 (blob better 5, SAM better 3; SAM wins where the nearest blob is poor).
- pCR (exploratory, 119 patients, 40 responders): AUC 0.39-0.59, not interpretable.

### S3v -- cost-aware review agent (pre-registered; train OOF 1,080 + val 120)
Workflow: automatic B0_rad -> QC risk (S3n) -> one simulated click on flagged scans -> nearest SegResNet blob or SAM (two-stage agent: stage 1 when to click from QC features; stage 2 which tool from post-click features). Formulation inspired by MedSAM-Agent (arXiv 2602.03320), implemented as a lightweight policy (no LLM).
| Train OOF, click -> nearest blob | Clicks | Mean Dice | Dice<0.1 | Dice<0.5 | Vol. error (median) |
|---|---|---|---|---|---|
| no review (B0_rad) | 0 | 0.7532 | 60 | 137 | 0.148 |
| QC triage 5% | 54 | 0.7603 | 50 | 125 | 0.149 |
| **QC triage 10%** | 108 | **0.7664** | **43** | 118 | 0.144 |
| random 10% | 108 | 0.7565 | 56.4 | 132.6 | 0.147 |
| oracle 10% | 108 | 0.7887 | 22 | 90 | 0.133 |
| QC triage 20% | 216 | 0.7705 | 39 | 111 | 0.143 |
| click everyone | 1080 | 0.7856 | 25 | 92 | 0.136 |
Val: no review 0.7727 (4 / 13) -> QC 10% 0.7818 (3 / 12) -> QC 20% 0.7966 (1 / 9) -> everyone 0.8076 (0 / 8).
- **R1 (primary) PASS:** QC 10% removes 17 failures vs 3.5 random; +13.5 [5.9, 21.2]; val same direction (+0.6).
- R2 FAIL: learned stage-1 agent vs QC ranking -0.0025 [-0.0065, 0.0014] -> use the simple QC rule.
- Click precision: 3-5 mm offset clicks give identical results.
- **R4 SAM role PASS (out-of-fold decoders):** 5 S2b decoders retrained without each S3k outer fold (`scripts/make_s2b_folds.py`, 864 train patients each, 60 epochs, ~12 min per fold; best val score 0.760-0.768 vs original S2b). Stage 2 chose SAM for 30 train patients: SAM 0.441 vs nearest blob 0.237, +0.204 [0.075, 0.327], 22/30 better; at the QC 10% budget +0.005 [0.002, 0.009] over always-blob. Leaky (original S2b) run: 35 chosen, 26/35 better, 0.48 vs 0.21 -> result not driven by memorisation. Val: 1 patient chosen, SAM 0.794 vs 0.684.
- Outputs: `outputs/s3v_review_agent` (original S2b) and `outputs/s3v_review_agent_oof` (clean, use this).
- Caveats: simulated clicks; development data only; overall gain small (targeted at ~3% of patients).

---------------------------------------------------------------------------------------------------
## 4. Earlier work (closed / negative)
- SegResNet: 60 -> 100 epochs (test rel20 0.655 -> 0.684); text FiLM conditioning no gain (p = 0.94).
- SAM S2a/S2b fine-tuning (box 0.698 test); S2c hard negatives (no AUC gain); S4 frequency adapter not used.
- Frequency council (S6), boundary refiners v1/v2 (+0.001, n.s.).
- Earlier juries: S3b 0.699, S3d 0.735, S3f memory jury 0.7495 (val-chosen), S3h jury13_rad_rel 0.7431.
- Security / adversarial track (S9): dropped.
- Ideas reviewed and not pursued: privacy audit of SAM memory banks (same-scan linkage, not re-identification); 4D cross-phase SAM (cache has no late phase; kinetic radiomics is the fair control); MASG-SAM modules (2D few-shot, does not fit).
- Note: other AI tools used `outputs/s3o_final/test` for development analyses; test results must not be used to choose methods.

## 5. Next
1. Guide: approve (a) one confirmatory run of the frozen S3v protocol (R1 + R4) on the 306 test patients, (b) nnU-Net plan, (c) a radiologist for a ~20-case pilot reader study.
2. **nnU-Net baseline** (guide 2026-11-09: train it to fit the project, need not copy the challenge setup): same split (train 1080, val 120 as validation, test 306 predicted once), 3 channels (pre, post1, post1 - pre) from the cache, 3d_fullres, single model, 1000 (or 250) epochs, evaluated with our metrics; optional: B0_rad jury on nnU-Net candidates. Waiting on disk check (`du -sh ~/data`, `df -h ~`) before writing the converter.
3. Realistic-click stress tests (CPU): 5-10 mm offsets, edge clicks, wrong-lesion clicks at 5-10%.
4. Failure analysis + figures (effort-accuracy curve; S3q Stage B pictures), MIDL writing, citation/novelty check.
5. Later: SAM label-efficiency study (N = 10-100 patients; ~20-40 GPU h), response tracking through chemotherapy (I-SPY follow-up scans), reader study, Duke side-cue audit, external cohort, kinetic radiomics.

## 6. Scripts (repo `scripts/`, tests in `tests/`)
s3i_subset_jury, s3j_selection_tradeoff, s3k_radiomics_matched, s3l_sam_matched, s3m_contour_study, s3n_qc_review, s3o_final_test, s3p_click_repair, s3q_location_audit, **s3r_sam_memory_matched**, **s3s_prompt_stability** (precheck / gate), **s3u_click_biomarkers** (masks / analyse / posthoc), **s3v_review_agent** (outcomes / policy; `--sam_fold_ckpt` for out-of-fold SAM), **make_s2b_folds** (5-fold S2b splits; training with `s2a_sam_finetune.py --split_col s2b_f<k> --box_aug realistic --skip_test`), e0_recall_ceiling.
