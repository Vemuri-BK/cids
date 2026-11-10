# CIDS project status (as of 2026-11-10)

Project: fully automatic 3D breast tumour segmentation on breast DCE-MRI (MAMA-MIA; split train 1080 / val 120 / test 306), plus a QC-guided one-click review step. Target: MIDL 2027 (abstract Nov 30, paper Dec 4). Lab server is the official record (`~/CIDS/outputs`); a results mirror is on the laptop (`cids project/outputs/`, git-ignored).

---------------------------------------------------------------------------------------------------
## 0. Summary for the paper (method FROZEN 2026-11-10)
- **Automatic pipeline (frozen, tested once, S3o):** SegResNet-100 finder -> radiomics jury B0_rad. Test Dice **0.7479 [0.7218, 0.7727]**, +0.0247 over the jury without radiomics (CI above 0); 14 Dice<0.1.
- **QC-guided one-click review (S3v development, S3t test confirmation):** QC flags the 10% riskiest scans -> one (simulated) click -> agent picks the nearest SegResNet blob or **F_max = SAM u SegResNet** (component at the click).
  - Test (306, once, frozen): **0.7479 -> 0.7621 with 31 clicks; Dice<0.1 failures 14 -> 7**; everyone clicked 0.7880 (1 failure).
  - **R1 PASS on test:** QC triage removes 5 failures vs 1.1 random, +3.9 [0.3, 7.9] (train OOF +13.5 [5.9, 21.2]).
  - **R9 PASS (val + test pooled):** F_max beats SAM alone on the patients the agent sends to SAM: 0.627 vs 0.501, **+0.125 [0.045, 0.213]**, 5 better / 0 worse (n = 7: 1 val + 6 test).
  - R4 (SAM alone vs blob) NOT confirmed on test: 6 patients, +0.094 [-0.18, +0.42], 2/6 better (train OOF +0.204 [0.075, 0.327]) -> underpowered.
- **SAM's role:** complement, not replace. 13 pre-registered roles tested; SAM helps only as an independent, click-driven rescue, best when fused with SegResNet's confident voxels.
- **Agent:** QC triage + learned tool choice; the choice reaches ~77% of the oracle gain (always-blob 0.7664 -> agent 0.7725 -> oracle 0.7743, train OOF); a learned verify-and-fallback step (S3x) does not help.
- Paper title candidates: "Automatic First, One Click When Needed: Radiomics-Guided Candidate Selection and SAM-SegResNet Fusion for Breast Tumour Segmentation in DCE-MRI"; "Pick Right, Click Once: ..."; "Complement, Not Replace: ...".
- Disclosure sentence (agreed): methods and rules were developed on train OOF + val and frozen; the test split was used only for frozen confirmatory runs (S3o, S3t), although earlier, discarded pipeline versions (e.g. SegResNet 60 vs 100 epochs, earlier juries) had been scored on it.

---------------------------------------------------------------------------------------------------
## 1. Final automatic method (frozen, tested once, 2026-11-07)

SegResNet-100 finder (3 channels: pre, post1, post1 - pre) -> candidates (p > 0.5, >= 0.1 ml, top 5) -> **binary radiomics jury B0_rad** (24 clues: 7 SegResNet + 15 radiomics + rel_seg_prob + rel_ring_cnr; HistGradientBoosting; threshold 0.40 chosen in inner folds), fitted on the 1,080 OUT-OF-FOLD training candidates -> output = union of the kept SegResNet blobs. Fully automatic. **No SAM.**

- Frozen: `scripts/s3o_final_test.py --step freeze` (val reproduced S3k exactly: B0_rad 0.7727, B0 0.7369). Model: `outputs/s3o_final/frozen/b0_rad.pkl` (+ manifest with md5). Finder: `checkpoints/segresnet_img_e100_best.pt` (best checkpoint chosen on val).
- Tested once: `--step test` (refuses a second run or a changed model). Outputs `outputs/s3o_final/test/`.

### Test (306 patients)
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
## 2. Final review step (frozen; development S3v/S3z, test S3t)

Flow: automatic B0_rad -> QC risk (S3n baseline logistic score, 7 features; empty masks first) -> top 10% get one click (deepest interior point of the expert tumour, simulated clinician; 3-5 mm offset click = sensitivity) -> stage-2 agent (S3v gradient boosting on post-click features, fitted on train OOF) -> nearest SegResNet blob (Cn) or F_max = max(P_SAM, P_SegResNet) > 0.5, 26-connected component at the click. SAM = SAM-Med3D-turbo + S2b LoRA decoder (out-of-fold decoders for train, `sam_s2b_best.pt` for val/test).

### Same protocol on all three datasets
| Rule | Train OOF (1,080) | Val (120) | Test (306) |
|---|---|---|---|
| R1 QC triage removes failures (10%) | 17 vs 3.5 random, +13.5 [5.9, 21.2] PASS | 1 vs 0.4 (too few) | 5 vs 1.1, +3.9 [0.3, 7.9] **PASS** |
| R4 SAM alone vs blob (agent picks SAM) | 30, +0.204 [0.075, 0.327] PASS | 1: 0.794 vs 0.684 | 6, +0.094 [-0.18, 0.42] FAIL (underpowered) |
| R9 F_max vs SAM alone | 30, +0.097 [0.036, 0.171], 15/1 (hypothesis source) | 1: +0.038 | pooled val+test 7: +0.125 [0.045, 0.213], 5/0 **PASS** |

| Mean Dice (Dice<0.1) | Train OOF | Val | Test |
|---|---|---|---|
| automatic only | 0.7532 (60) | 0.7727 (4) | 0.7479 (14) |
| 10% clicks -> blob | 0.7664 (43) | 0.7818 (3) | 0.7580 (9) |
| 10% clicks -> agent (blob / SAM) | 0.7714 (32) | 0.7818 (3) | 0.7617 (7) |
| **10% clicks -> agent (blob / F_max)** | ~0.7724 | 0.7818 (3) | **0.7621 (7)** |
| 20% clicks -> agent (blob / F_max) | - | - | 0.7669 (7) |
| everyone clicked -> agent (blob / F_max) | 0.7939 | 0.8089 | 0.7880 (1) |
- Test per hospital (automatic -> 10% agent): DUKE 0.693 -> 0.714, ISPY1 0.742 -> 0.754, ISPY2 0.793 -> 0.804, NACT 0.717 -> 0.719.
- Simple no-learning variant "always F_max" at 10%: train 0.7694 (Dice<0.1 30, <0.5 113) vs agent ~0.7724; val 0.7788 vs blob 0.7818 -> report as variant (useful on a new backbone, e.g. nnU-Net).
- Caveats: simulated clicks; R9 on 7 patients (pre-registered, never worse); R4 underpowered on test.

### S3t -- test confirmation (2026-11-10, guide-approved, run once)
`scripts/s3t_review_test.py`: `--step freeze` (no test read) checked that the rule code reproduces S3v, the stage-2 agent reproduces S3v's val decisions, the QC model reproduces S3n's val risk and the frozen jury reproduces S3n's val scores (all true), then wrote `test_plan.json` with md5 of every input; `--step test` ran once (done.json; B0_rad and SAM parity per patient). Outputs `outputs/s3t_test/` (outcomes.csv, curves_*.csv, per_hospital.csv, r9_patients.csv, results.json).

---------------------------------------------------------------------------------------------------
## 3. Decision-layer experiments (train OOF, matched, frozen rules, identical folds/seeds)
Frozen two-route rule: route 1 = CI lower bound of paired dDice > 0 and #Dice<0.1 not higher; route 2 = #Dice<0.1 <= 0.75 x reference and mean Dice >= reference - 0.005; both: #Dice<0.5 <= 1.10 x reference.

| Step | Comparison | Train OOF Dice (Dice<0.1 / <0.5) | Paired dDice [95% CI] | Candidate AUC | Verdict |
|---|---|---|---|---|---|
| E0 | recall-first ceiling | tumour among candidates 98.7% | - | - | recall-first dropped |
| S3i | subset-selection pilot | U0 0.8119 / U1 0.8116 / B0 0.8105 (in-sample) | within noise | - | no SAM gate |
| S3j | U0 vs binary juries | U0 0.7475 (46/145), B0 0.7398 (69/152) | U0-B0 +0.0077 [0.0009, 0.0143] | - | binary B0 main |
| S3k | B0 -> + radiomics (B0_rad) | 0.7398 -> **0.7532** (60/137) | **+0.0134 [0.0053, 0.0214]** | 0.900 -> **0.958** | **radiomics qualify** |
| S3l | B0_rad -> + 4 SAM box features | 0.7508 (60/142) | -0.0024 [-0.0073, 0.0025] | 0.959 | no |
| S3l check | B0 -> + 4 SAM features | 0.7386 (75/153) | -0.0012 [-0.0057, 0.0030] | 0.899 | no |
| S3r | B0_rad (A) vs + non-SAM 2nd anchor (B) vs + SAM memory + SAM anchor (C) | A 0.7532 (60/137), B 0.7561 (60/135), C 0.7565 (56/130); val 0.7727 / 0.7821 / 0.7726 | C-A +0.0033 [-0.0018, 0.0086]; C-B +0.0004 [-0.0055, 0.0063] | 0.958 / 0.962 / 0.961 | SAM memory does not qualify |

S3r pre-check (train OOF): finder's top blob wrong in 74 patients; B0_rad wrong picks 46 (36 with a wrong finder anchor); SAM memory (mem_knn) top candidate hits the tumour in 32/46 vs 22 for the best non-SAM pick, but points elsewhere in 173 of 1,020 correct patients -> net only 11 rescued / 7 new failures in the jury.

---------------------------------------------------------------------------------------------------
## 4. SAM-Med3D-turbo (S2b LoRA decoder) -- all roles tested (pre-registered)
| # | Role | Study | Frozen result | Verdict |
|---|---|---|---|---|
| 1 | Jury clues (box decoder) | S3l | no net benefit | no |
| 2 | Automatic contour of kept blobs (box / point prompts from SegResNet) | S3m | D vs A dNSD -0.142 [-0.149, -0.135] | no |
| 3 | QC review flag | S3n | 65 vs 65 failures caught at 10% | no |
| 4 | Click repair editor (multi-click) | S3p v2 | TS vs brush dNSD -0.084 [-0.137, -0.035] | no |
| 5 | Whole-breast outline (feasibility) | S3q | 6 rescuable < 10 | stop |
| 6 | Memory + second anchor in the jury | S3r | C-A +0.0033 n.s.; C-B +0.0004 | no |
| 7 | Multi-prompt stability as a clue | S3s | turbo AUC 0.49, 9/46 hits (chance 17.7) | no |
| 8 | One-click biomarker tool | S3u | equal to nearest SegResNet blob (CCC 0.899 vs 0.901) | no |
| 9 | Backup tool in QC-guided one-click review | S3v R4 | train OOF +0.204 [0.075, 0.327] on 30; test 6 patients n.s. | PASS dev / underpowered test |
| 10 | Composite SegResNet prompts (points + negatives + mask) | S3w | R5a SC vs S1 -0.017 [-0.091, 0.044] on the 30 | no |
| 11 | Decoder specialised for the backup role | S3y | R6a -0.008 [-0.017, 0.001] | no |
| 12 | Fusion with SegResNet (agent with 4 tools) | S3z R7 | +0.0005 [-0.0013, 0.0025] | no |
| **13** | **F_max = SAM u SegResNet as the rescue mask** | **S3z-b / S3t R9** | **+0.125 [0.045, 0.213] (val + test), 5/0** | **PASS (test)** |
Paper wording: "SAM did not improve the automatic pipeline in any pre-registered automatic role; in a QC-guided one-click review it rescues tumours the specialist missed, and fusing it with the specialist's confident voxels improves the rescue (confirmed on held-out test)."

### S3m -- SAM as automatic boundary refiner (train OOF)
| Arm | Dice | NSD@2mm | HD95 med | Dice<0.5 |
|---|---|---|---|---|
| A SegResNet blob | **0.7532** | **0.805** | **3.0 mm** | **137** |
| B SAM box | 0.6530 | 0.667 | 5.4 | 194 |
| C box + geometric point | 0.6594 | 0.665 | 5.2 | 191 |
| D box + enhancement point | 0.6582 | 0.663 | 5.4 | 191 |
| E region growing | 0.3873 | 0.478 | 13.3 | 708 |

### S3n -- SAM disagreement as QC flag
10% budget: baseline 65 / +SAM 65 of 137 failures (AUROC 0.877 vs 0.875). The cheap baseline QC score (7 features) catches 47% of failures in 10% of scans -> reused in S3v / S3t.

### S3p -- simulated click repair (108-patient review queue)
v2: TS vs TB dNSD -0.084 [-0.137, -0.035], harmed 29 vs 7; TS_hybrid +0.026 [-0.001, 0.055]. SAM's first click recovers missed tumour fastest (wrong target 0.250 vs 0.174 after 1 click).

### S3q -- location audit
36 wrong picks; 6 rescuable by a breast outline (< 10) -> stop; max gain ~+0.004 Dice.

### S3s -- multi-prompt stability
Turbo returns the same broad mask for any click (stability 0.96 on tumours and false blobs alike); 9/46 hits, 380 misleads, AUC 0.49 -> FAIL.

### S3u -- one-click masks for radiomic biomarkers (val 120)
| Mask | Mean Dice | Dice<0.1 | Median CCC vs expert |
|---|---|---|---|
| B automatic B0_rad | 0.773 | 4 | 0.821 |
| C click inside a SegResNet blob | 0.767 | 8 | - |
| Cn click -> nearest SegResNet blob (post-hoc) | **0.808** | **0** | **0.901** |
| S SAM alone (one click) | 0.617 | 1 | 0.519 |
| H blob under click, else SAM | 0.806 | 0 | 0.899 |
pCR exploratory: AUC 0.39-0.59, not interpretable.

### S3v -- cost-aware review agent (train OOF 1,080 + val 120)
| Train OOF, click -> nearest blob | Clicks | Mean Dice | Dice<0.1 | Dice<0.5 |
|---|---|---|---|---|
| no review (B0_rad) | 0 | 0.7532 | 60 | 137 |
| QC triage 5% | 54 | 0.7603 | 50 | 125 |
| **QC triage 10%** | 108 | **0.7664** | **43** | 118 |
| random 10% | 108 | 0.7565 | 56.4 | 132.6 |
| oracle 10% | 108 | 0.7887 | 22 | 90 |
| QC triage 20% | 216 | 0.7705 | 39 | 111 |
| click everyone | 1080 | 0.7856 | 25 | 92 |
- R1 PASS; R2 FAIL (learned stage-1 vs QC ranking -0.0025 n.s. -> use QC); offset clicks identical; R4 PASS with out-of-fold S2b decoders (`make_s2b_folds.py`, ~12 min per fold). Outputs `outputs/s3v_review_agent_oof` (use this).
- 10% is the pre-registered operating point (most of the gain: 10 -> 20% doubles clicks for 4 more failures fixed); curves report 0-100%.

### S3w -- composite SegResNet prompts (idea from RADAR, arXiv 2610.14944)
S1 one click / SP + 2 extra positives + up to 3 negatives / SC + SegResNet mask prompt. All patients: S1 0.611 -> SC 0.632 (train), val 0.617 -> 0.648; but on the 30 SAM-chosen patients S1 0.441, SP 0.431, SC 0.424 (R5a FAIL, 10 better / 20 worse); R5b R4 fails with SC. SegResNet hints mislead where SAM is needed.

### S3y -- SAM decoder specialised for the backup role
Hard patients = nearest blob or B0_rad Dice < 0.5 (139 of 1,080); train = hard of other folds + equal random others; init fold S2b decoder, clicks only, 30 epochs, lr 2e-4, best by val Dice@1 (start-model fallback was not applied: logged deviation). Best val click1 0.667-0.687. **R6a FAIL:** 0.434 vs 0.441, -0.008 [-0.017, 0.001], 11 better / 19 worse; all train -0.007 (CI below 0); val 0.6171 = 0.6171. Keep the original S2b decoders. Outputs `outputs/s3y_backup`, `outputs/s3v_review_agent_s3y`.

### S3z -- fusing SAM and SegResNet at the click
Masks: S (P_sam > 0.5), F_avg ((P_sam + P_seg)/2 > 0.5), F_max (max > 0.5); parity with S3v exact (0 of 2,400 differ). All patients deep click: train S 0.611 / F_avg 0.704 / F_max 0.756 (blob 0.786); val 0.617 / 0.722 / 0.785 (0.808).
- R7 FAIL: 4-tool agent vs 2-tool agent +0.0005 [-0.0013, 0.0025] (A4 picked F_max 41 times); R7a FAIL: F_avg alone vs agent -0.076 on the 108 clicked. Feature control A2_ext +0.0007 n.s.
- Observed (post-hoc): on the 30 SAM-chosen patients F_max 0.538 vs SAM 0.441 vs blob 0.237 -> S3z-b.

### S3z-b -- frozen rule R9 (F_max as the SAM rescue mask)
Frozen 2026-11-10 15:28 UTC (`outputs/s3z_b/decision_rule.json`) before val was read. Development (hypothesis): 30 patients, +0.097 [0.036, 0.171], 15 better / 1 worse; val 1 patient +0.038; decided on val + test in S3t: **PASS**.

### S3x -- verify-and-fallback agent (train OOF + val; test not used)
Verifier = gradient boosting classifier (out-of-fold) predicting delivered Dice < 0.5 from post-click, fusion, QC and delivered-mask features. **R8 FAIL:** AUROC 0.61 [0.50, 0.72] on the 108 clicked (needed > 0.6); **R8b FAIL:** switching the top 20% to the other tool -0.003 [-0.006, -0.001] (7 better / 15 worse); hand-back 10 of 38 remaining failures vs 7.6 random (n.s.). The verifier works across all patients (AUROC 0.90 train, 0.82 val) but not among the flagged hard cases; oracle fallback only 0.7743 vs agent 0.7725 -> the agent already reaches ~77% of the oracle gain. Agent kept without verification. Outputs `outputs/s3x_verify`.

---------------------------------------------------------------------------------------------------
## 5. Earlier work (closed / negative)
- SegResNet: 60 -> 100 epochs (test rel20 0.655 -> 0.684; checkpoints chosen on val); text FiLM conditioning no gain (p = 0.94).
- SAM S2a/S2b fine-tuning (box 0.698 test); S2c hard negatives (no AUC gain); S4 frequency adapter not used.
- Frequency council (S6), boundary refiners v1/v2 (+0.001, n.s.).
- Earlier juries: S3b 0.699, S3d 0.735, S3f memory jury 0.7495 (val-chosen), S3h jury13_rad_rel 0.7431.
- Security / adversarial track (S9): dropped.
- Ideas reviewed and not pursued: privacy audit of SAM memory banks; 4D cross-phase SAM; MASG-SAM modules; metadata text prompts for SAM; automatic SegResNet-to-SAM box/point prompting (already negative in S3m / S3w).
- Note: other AI tools used `outputs/s3o_final/test` for development analyses; test results must not be used to choose methods.

## 6. Next
1. **nnU-Net baseline** (prepared: Dataset501_MAMAMIA3ch, 3 channels, our 1080/120 split in splits_final.json, plans + preprocessing done): `nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_250epochs` (GPU 0), then predict test once and `nnunet_baseline.py --step evaluate`. If nnU-Net beats the automatic pipeline, apply the review layer on top (QC + click + always F_max as the no-retraining variant; ask the guide before using test).
2. Day after tomorrow: learned prompt generator (small network turning SegResNet output + click into SAM prompt tokens) at the review step, as a journal-extension idea; would need a third test use.
3. Figures: effort-accuracy curves (train / val / test), rescue examples (blob vs SAM vs F_max), SAM-roles table; MIDL writing (abstract Nov 30).
4. Reader pilot package ready (`outputs/reader_pilot_package.zip`, 20 val cases); guide arranging a radiologist and ethics; optional 2 extra sheet questions (trust for measurement; usefulness 1-5).
5. Later: SAM label-efficiency study, response tracking (I-SPY follow-up), external cohort, kinetic radiomics.

## 7. Scripts (repo `scripts/`, tests in `tests/`)
s3i_subset_jury, s3j_selection_tradeoff, s3k_radiomics_matched, s3l_sam_matched, s3m_contour_study, s3n_qc_review, s3o_final_test, s3p_click_repair, s3q_location_audit, s3r_sam_memory_matched, s3s_prompt_stability, s3u_click_biomarkers, s3v_review_agent (`--sam_fold_ckpt`), make_s2b_folds, s3w_composite_prompts, s3y_backup_sam (+ `s2a_sam_finetune.py --select click1`), **s3z_fusion** (masks / analyse), **s3z_fusion_b** (freeze / val), **s3t_review_test** (freeze / test, run once), **s3x_verify_agent**, nnunet_baseline (convert / evaluate), reader_package, e0_recall_ceiling.
