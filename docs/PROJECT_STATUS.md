# CIDS project status (as of 2026-11-08)

Project: fully automatic 3D breast tumour segmentation on breast DCE-MRI (MAMA-MIA; split train 1080 / val 120 / test 306). Target: MIDL 2027 (abstract Nov 30, paper Dec 4). Lab server is the official record (`~/CIDS/outputs`); a results mirror is on the laptop (`cids project/outputs/`, git-ignored).

---------------------------------------------------------------------------------------------------
## 1. Final method (frozen, tested once, 2026-11-07)

SegResNet-100 finder (3 channels: pre, post1, post1 - pre) -> candidates (p > 0.5, >= 0.1 ml, top 5) -> **binary radiomics jury B0_rad** (24 clues: 7 SegResNet + 15 radiomics + rel_seg_prob + rel_ring_cnr; HistGradientBoosting; threshold 0.40 chosen in inner folds), fitted on the 1,080 OUT-OF-FOLD training candidates -> output = union of the kept SegResNet blobs. Fully automatic. **No SAM.**

- Frozen: `scripts/s3o_final_test.py --step freeze` (val reproduced S3k exactly: B0_rad 0.7727, B0 0.7369). Model: `outputs/s3o_final/frozen/b0_rad.pkl` (+ manifest with md5, threshold, features, git commit). Finder: `checkpoints/segresnet_img_e100_best.pt`.
- Tested once: `--step test` (guards: refuses a second run or a changed model). Outputs `outputs/s3o_final/test/` (per_patient, summary, ablation, 306 masks).

### Test (306 patients; benchmark seen by earlier pipeline versions, stated as a caveat)
| Policy | Dice [95% CI] | Median | HD95 med (n) | NSD@2mm | Dice<0.1 | Dice<0.5 |
|---|---|---|---|---|---|---|
| **B0_rad (final)** | **0.7479 [0.7218, 0.7727]** | 0.823 | 3.46 mm (305) | 0.796 | 14 | 39 |
| B0 (7 finder clues) | 0.7232 [0.6923, 0.7521] | 0.819 | 3.61 | 0.769 | 25 | 46 |
| SegResNet + rel20 among candidates | 0.6835 | 0.767 | 7.28 | 0.716 | 14 | 61 |
| keep-all candidates | 0.6736 | 0.733 | 32.3 | 0.698 | 8 | 61 |
| perfect selection (ceiling) | 0.7901 | - | - | - | - | - |
- Ablation B0_rad - B0: **+0.0247 [0.0102, 0.0416]**; 34 better / 28 worse / 243 identical; rescued below 0.1: 12 (harmed 1); below 0.5: 8 (harmed 1).
- Per hospital (B0_rad): DUKE 0.693 (n=91; 9 of the 14 Dice<0.1; B0 0.618), ISPY1 0.742 (67), ISPY2 0.793 (131), NACT 0.717 (17).
- Excluding bilateral cancer (295): 0.7548; bilateral cancer only (11): 0.563 (expert masks label only the primary tumour).
- Earlier locked jury13_rad_rel (33 clues incl. SAM, in-sample candidates) on the same test: 0.7431 -- descriptive only.

Development: train OOF (1,080) B0_rad 0.7532 (Dice<0.1: 60, <0.5: 137); val (120) 0.7727 (4 / 13).

---------------------------------------------------------------------------------------------------
## 2. Decision-layer experiments (train OOF, matched, frozen rules, identical folds/seeds)
Frozen two-route rule (S3k/S3l): route 1 = CI lower bound of paired dDice > 0 and #Dice<0.1 not higher; route 2 = #Dice<0.1 <= 0.75 x reference and mean Dice >= reference - 0.005; both: #Dice<0.5 <= 1.10 x reference.

| Step | Comparison | Train OOF Dice (Dice<0.1 / <0.5) | Paired dDice [95% CI] | Candidate AUC | Verdict |
|---|---|---|---|---|---|
| E0 | recall-first ceiling (lower finder thresholds) | tumour among candidates 98.7% (14/1080 missed); lower thresholds lower the ceiling | - | - | recall-first dropped |
| S3i | subset-selection pilot (in-sample tables) | U0 0.8119 / U1 0.8116 / B0 0.8105 (train-OOF of in-sample) | within noise | - | failures halved; no SAM gate |
| S3j | U0 (best combination) vs binary juries (OOF) | U0 0.7475 (46/145), B0 0.7398 (69/152), keep_all 0.7004, oracle 0.7889 | U0-B0 +0.0077 [0.0009, 0.0143] | - | U0 does not advance; binary B0 main |
| S3k | B0 -> + radiomics (B0_rad) | 0.7398 -> **0.7532** (60/137); val 0.7369 -> 0.7727 | **+0.0134 [0.0053, 0.0214]** | 0.900 -> **0.958** | **radiomics qualify** |
| S3l | B0_rad -> + 4 SAM features | 0.7532 -> 0.7508 (60/142) | -0.0024 [-0.0073, 0.0025] | 0.958 -> 0.959 | SAM features do not qualify |
| S3l check | B0 -> + 4 SAM features | 0.7398 -> 0.7386 (75/153) | -0.0012 [-0.0057, 0.0030] | 0.900 -> 0.899 | SAM features do not qualify |

---------------------------------------------------------------------------------------------------
## 3. SAM-Med3D-turbo (S2b checkpoint) -- five roles tested, none passed (development data, pre-registered)
| Role | Study | Frozen result |
|---|---|---|
| Selection clues | S3l | no net benefit (above) |
| Automatic contour of selected blobs | S3m | D vs A dNSD -0.142 [-0.149, -0.135], dDice -0.095 |
| QC review flag (10% budget) | S3n | 65 vs 65 failures caught |
| Simulated interactive click repair | S3p v2 | TS vs toggle+brush dNSD -0.084 [-0.137, -0.035]; harmed 29 vs 7 |
| Whole-breast outline for the jury (feasibility) | S3q audit | 6 rescuable patients < 10 -> STOP |

Paper wording: "Across pre-registered roles, the adapted SAM-Med3D-turbo checkpoint did not improve on simpler alternatives within our automatic pipeline; its main measurable benefit was rapid recovery of missed tumour with a single positive click."

### S3m -- SAM as automatic boundary refiner
Selection frozen at B0_rad; same kept candidates; patient mask = union. Step 0 parity exact (max |diff| 0, 1,200 patients). Primary endpoint NSD@2mm; frozen test D vs A only (B, C vs A exploratory).

| Arm (train OOF, fallback) | Dice | NSD@2mm | HD95 med | Dice<0.5 |
|---|---|---|---|---|
| A SegResNet blob | **0.7532** | **0.805** | **3.0 mm** | **137** |
| B SAM box | 0.6530 | 0.667 | 5.4 | 194 |
| C SAM box + geometric point | 0.6594 | 0.665 | 5.2 | 191 |
| D SAM box + enhancement point | 0.6582 | 0.663 | 5.4 | 191 |
| E region growing (same seed, tau 0.5) | 0.3873 | 0.478 | 13.3 | 708 |
- D vs A failed all five conditions: 105 patients lose >= 0.20 Dice (limit 10), Dice<0.5 191 (limit 150), 63 newly < 0.5 vs 9 rescued. Val the same (dNSD -0.165).
- Shift check (30 cases): the tested translations did not close the gap (D 0.758 -> 0.779 best shift vs A 0.848).

### S3n (Q1) -- SAM disagreement as a QC review flag
Failure = A Dice < 0.5 (137/1,080); budget 10% = 108 reviews (empty masks first). Baseline review model (SegResNet confidence, jury margins, counts, volume; fold-correct jury scores reproduced S3k exactly) vs + 1 - Dice(A, box-only SAM).

| Budget | Baseline caught | +SAM caught |
|---|---|---|
| 5% | 41 | 41 |
| **10%** | **65** | **65** (gain 0, CI [-4, 4]; needed >= +10) |
| 20% | 90 | 89 |

AUROC 0.877 vs 0.875. Side finding: the cheap baseline QC score catches 47% of failures in 10% of scans (~14 at random).

### S3p -- simulated click repair (108-patient S3n review queue = development cohort)
- Arms from the same B0_rad mask, same 5-click budget/click policy/blob rules: T (blob toggle), TB (toggle + 5 mm brush), TS (toggle + SAM), exploratory TS_hist, TS_hybrid. Simulated reader uses the expert mask; editors never do (verified).
- v1 (tag `s3p-v1`) smoke exposed simulator flaws (repeated no-op clicks, unbounded edits, expert-assisted toggle) and SAM ignoring negative clicks. Mask-prompt calibration on 20 separate patients: tau 0.25, L 1 (round-trip / compliance only). Replay: negative click alone p 0.01 vs with history 0.92 -> removing point history restored compliance in 15/18 always and 6/6 oracle attempts. v2: no unchanged-mask retries, 20 mm edit bound, explicit reader_delete_blob, PRIMARY TS = current click + current mask.
- v2 full (frozen, always-accept, 3 clicks, TS vs TB): **dNSD -0.084 [-0.137, -0.035]**, below-start 29 vs 7 -> do not advance. Excluding the 12 smoke patients -0.082 [-0.137, -0.031]. Exploratory: TS_hist -0.010 [-0.051, 0.028] (26 vs 7); TS_hybrid +0.026 [-0.001, 0.055] (16 vs 7).
- By start type (NSD after 3 clicks): ok masks (43) TS 0.666 vs TB 0.812 (TS 25 below start, 10 newly < 0.5); poor boundary (30) 0.552 vs 0.565; wrong target (32) 0.391 vs 0.474 (hybrid 0.531). SAM's first click recovers missed tumour fastest (wrong target 0.250 vs 0.174 after 1 click).
- Negative-click compliance TS 0.66 vs TS_hist 0.49; positive ~0.90; ~0.5 s per edit.

### S3q -- location audit: could a whole-breast outline help the jury?
- Population: 36 train-OOF wrong picks (B0_rad Dice < 0.1, tumour among candidates, kept blob misses it; ISPY2 15, DUKE 14, ISPY1 5, NACT 2). Ceiling: 34 reach Dice >= 0.1 and 22 >= 0.5 with a perfect pick; fixing all wrong picks: severe failures 60 -> 26, mean Dice +0.020.
- Two-stage review: Stage A blind (candidates only), Stage B reveal. 132 candidates labelled blind by an AI annotator (`outputs/s3q_location_audit/annotation_filled_ai.csv`): inside 88, boundary_uncertain 30, outside 14.
- Kept blob location per patient: inside breast 17, boundary 12, clearly outside 7. **Rescuable (clear outside + inside alternative Dice >= 0.1): 6** (>= 0.5: 5); uncertain with an alternative: 10 (counted separately). Threshold 10 -> **STOP**. Max gain from the 6: ~+0.004 mean Dice.
- Conclusion: most remaining wrong picks are choices between two in-breast enhancing regions (often the other breast in bilateral DUKE scans) -> anatomy cannot fix them; the side-cue / clinical-location idea is more relevant.
- Released whole-breast masks (MAMA-MIAAI, PhysioNet, open access, CC BY-NC(-SA) 4.0, 104.5 GB zip; 1,506 automatic + 94 manual) noted but not downloaded.

---------------------------------------------------------------------------------------------------
## 4. Earlier work (closed / negative)
- SegResNet: 60 -> 100 epochs (test rel20 0.655 -> 0.684); text FiLM conditioning no gain (p = 0.94).
- SAM S2a/S2b fine-tuning (box 0.698 test, rejects 96.7% synthetic negatives but 0.6% of real false candidates); S2c hard negatives (AUC never improved); S4 frequency adapter not used.
- Frequency council (S6), boundary refiners v1/v2 (+0.001, n.s.).
- Earlier juries: S3b segjury 0.699, S3d learned jury 0.735, S3f memory jury 0.7495 (val-chosen), S3h jury13_rad_rel 0.7431 (consistent protocol); LOHO, volume agreement, disagreement-regime analyses.
- Security / adversarial track (S9): explored and dropped (code archived).
- Methodology + results draft (IEEEtran, laptop `outputs/paper_methodology/`, not in git; pre-dates the final B0_rad result).

## 5. Next
1. Meet guide: final method = SegResNet + radiomics jury (test 0.748); SAM reported as a rigorous negative finding in five roles (or dropped from the pipeline).
2. Failure analysis + figures on the test result (14 Dice<0.1 / 39 Dice<0.5; DUKE 9 of 14; wrong in-breast lesion vs boundary); S3q Stage B pictures usable as figures.
3. MIDL writing (abstract Nov 30); literature/novelty check of all citations; update the methodology draft to the final pipeline.
4. Optional / later: Duke side-cue audit (needs Duke `Clinical_and_Other_Features.xlsx` from TCIA; exclude MAMA-MIA `primary_lesion.breast_coordinates`), external cohort (e.g. public 100-case Yunnan DCE-MRI), radiomic-reproducibility + pCR paper (expert vs B0_rad vs SAM masks), nnU-Net baseline, kinetic / acquisition-time features, calibration, fairness, reader study only if a new interactive approach shows promise.

## 6. Scripts (repo `scripts/`, tests in `tests/`)
s3i_subset_jury, s3j_selection_tradeoff, s3k_radiomics_matched, s3l_sam_matched (+ `--base finder`), s3m_contour_study (parity / arms / shiftcheck / overlap), s3n_qc_review (scores / evaluate), s3o_final_test (freeze / test), s3p_click_repair (parity / calibrate / replay / smoke / full; `--version v1|v2`), s3q_location_audit (images / tally), e0_recall_ceiling.
