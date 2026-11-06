# S3i: conditional CPU pilot of patient-level subset selection

Development data only (train out-of-fold + reused val). Test files were not read.

## Scope
- Inputs: the S3f candidate tables (`train_features.csv`, `val_features.csv`) and the saved jury scores (`dev_scores.csv`).
- The candidate pool, the upstream SegResNet-100 / SAM-S2b predictions and the candidate masks are unchanged.
- **This is a conditional decision-layer pilot.** The upstream models saw the training patients, so it does not establish fully out-of-fold performance.
- No new networks, prompts, memory or radiomics.
- **Hypothesis:** patient-level subset scoring may select better combinations from imperfect evidence than independent candidate classification.

## Arms (identical outer folds; complexity and thresholds from inner patient folds)
| Arm | Model | Clues |
|---|---|---|
| B0 | binary classifier + threshold (highest-score fallback) | 7 finder |
| B1 | singleton-quality regressor, target 2·I_k/(V_k+G), + threshold | 7 finder |
| U0 | joint subset utility regressor (permutation-invariant summaries) | 7 finder |
| B11 | binary classifier + threshold | 7 finder + 4 SAM |
| U1 | joint subset utility regressor | 7 finder + 4 SAM |
| HIST_* | saved scores at documented thresholds | reproduction only, not matched |
| ORACLE | best nonempty subset | non-deployable |

Further details:
- Model: HistGradientBoosting with `early_stopping=False`; 4-config grid; 5 outer and 3 inner folds, grouped by patient; seed 42.
- Subset rows are weighted so that each patient's rows sum to 1.
- K = 1 keeps the sole candidate. K = 0 stays in the evaluation with an empty mask (none occurred).
- Ties in U0/U1 prefer the fold-consistent B0 subset; otherwise the first subset in enumeration order.
- **Harmful removal:** the decision excluded at least one tumour-touching candidate, and adding the excluded touching candidates back would have increased that patient's Dice.

## Run
```bash
python -m pytest -q tests/test_s3i_subset_jury.py          # 9 tests
python scripts/s3i_subset_jury.py --train data/train_features.csv --val data/val_features.csv \
    --dev_scores data/dev_scores.csv --config configs/s3i_subset_jury.json --out results
```
It takes about 11 minutes on 2 CPU cores. On the lab server, the inputs are
`~/CIDS/outputs/s3f_jury_masks/s3f_jury/{train,val}_features.csv` and
`~/CIDS/outputs/s3h_radiomics/dev_scores.csv` (the manifests are read from next to the tables).

## Outputs (`results/`)
| File | Contents |
|---|---|
| `arms_summary.csv` | Dice, regret, failures, harmful removals and per-collection Dice per arm; all patients and K≥2 |
| `paired_comparisons.csv` | Patient-bootstrap CIs; better/worse counts; new and rescued failures |
| `routing_summary.csv`, `routing_budget.csv` | U0-vs-U1 complementarity; oracle and random routing by encoding budget |
| `per_patient.csv` | Every arm's Dice and selected subset; outer fold; status |
| `folds.csv`, `selected_hyperparameters.json`, `provenance.json` | Fold assignments, chosen configs/thresholds per fold, input hashes |

## Results (all patients)
| Arm | Train OOF Dice | Val Dice | Train Dice<0.1 | Val Dice<0.1 | Train harmful removals |
|---|---|---|---|---|---|
| B0 | 0.8105 | 0.7529 | 35 | 6 | 118 |
| B1 | 0.8092 | 0.7593 | 36 | 6 | 122 |
| **U0** | **0.8129** | **0.7693** | **17** | **2** | **86** |
| B11 | 0.8089 | 0.7602 | 35 | 4 | 120 |
| U1 | 0.8116 | 0.7601 | 17 | 3 | 81 |
| HIST finder_only | 0.8111 | 0.7592 | 32 | 5 | 112 |
| HIST final (33 clues) | 0.8184 | 0.7870 | 24 | 2 | 78 |
| ORACLE | 0.8448 | 0.8120 | 5 | 0 | 0 |

The fresh B0 reproduces the historical finder_only (0.8105 vs 0.8111), which confirms the pipeline.

**Paired comparisons (train OOF / val):**
- U0 − B0: **+0.0024** [−0.0035, +0.0084] / **+0.0164** [−0.0075, +0.0439].
  Severe failures (Dice < 0.1): 24 rescued, 6 new (train).
  Patients better/worse by > 0.01: 50/98 (train).
- U1 − U0: −0.0013 [−0.0039, +0.0011] / −0.0092 [−0.0247, +0.0004].
- B11 − B0: −0.0016 / +0.0074.

**Routing, U0 vs U1:**
- Identical subsets: 85% of patients (75–78% of K ≥ 2).
- Oracle routing gain over U0: **+0.0023** (train OOF), **+0.0008** (val).
- Random consultation lowers Dice at every budget.

## Conclusion
1. **Does joint selection help?** Its mean gain is small and uncertain: the CIs include 0 on both sets. But it **halves near-total failures**: Dice < 0.1 falls from 35 to 17 on train OOF and from 6 to 2 on val, and harmful removals fall by about 27%. The price is small losses on more patients (98 worse vs 50 better). This is a failure-reduction effect, not a large average gain. It's worth continuing, with that framing.
2. **Does SAM add useful information?** Not in this pilot. U1 ≤ U0 and B11 ≈ B0, on both sets.
3. **Is a consultation gate justified?** No. Oracle routing headroom is +0.002 / +0.001 Dice.
4. **Caveats:**
   - The historical 33-clue jury is still the best train-OOF/val result, but its threshold was chosen globally (not nested), and it uses memory and radiomics.
   - Validation is a reused development set.
   - The upstream models are not out-of-fold.
