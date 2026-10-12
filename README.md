# CIDS — Clinical Instruction Dual Supervision

Frequency-prompted SAM-Med3D council for 3D breast DCE-MRI tumor segmentation (MAMA-MIA).

- **Member A:** SAM-Med3D (frozen encoder) + Frequency-Instructed Prompt Generator (3D Haar DWT bands gated by the clinical instruction) + LoRA mask decoder
- **Member B:** SegResNet conditioned on the instruction (FiLM + cross-attention); proposes the tumor box
- **Council:** jury head fuses both masks; **dual supervision** = expert ground truth (+ band-wise boundary loss) and reliability-weighted peer supervision
- **Prompt study (FairVLM-style):** Type-S structured vs Type-N narrative instructions, counterfactual invariance, fairness metrics

## Workflow

```
VS Code (this folder)  --git push-->  GitHub (private repo)  --git clone-->  Kaggle notebook (GPU)
data never goes to GitHub; it reaches Kaggle as Kaggle datasets
```

Kaggle inputs:
| Kaggle dataset | Contents |
|---|---|
| existing 2-channel MAMA-MIA cache (from FADC) | `train/*.npz`, `val/*.npz` — ch0 pre-contrast, ch1 post-contrast phase 1, 1 mm iso, labels |
| `cids-assets` (new) | `cids_prompts.csv`, `clinical_and_imaging_info.xlsx`, `train_test_splits.csv` |

## Layout

```
cids/prompts/fields.py       metadata normalization + list of EXCLUDED (leaky) fields
cids/prompts/structured.py   Type-S prompts, knowledge hints, counterfactuals
cids/prompts/narrative.py    Type-N filters: fact check, Jaccard diversity, similarity, top-k
cids/models/haar.py          3D Haar DWT / inverse, instruction-conditioned shrinkage
cids/models/sam_med3d.py     SAM-Med3D-turbo loading (strict key check), click inference
cids/data/cache.py           FADC cache reader, 1.5 mm resampling, SAM crops, z-norm
scripts/build_prompts.py     metadata -> data/prompts/cids_prompts.xlsx (+ .csv)
scripts/generate_narratives.py   LLM narratives on Kaggle (resumable)
scripts/update_excel.py      put Kaggle narratives back into the Excel sheet
notebooks/00_generate_narratives.ipynb   Kaggle notebook for the step above
scripts/s0_sam_embeddings.py   S0 embedding cache + zero-shot baseline
notebooks/01_s0_sam_embeddings.ipynb   Kaggle notebook for S0
cids/models/segresnet_text.py SegResNet + FiLM text conditioning (zero-init)
cids/text/encoder.py         frozen BioClinicalBERT sentence encoder
cids/eval/metrics.py         3D Dice / HD95 / NSD
cids/data/volumes.py         1 mm volumes, patch sampling, augmentation
cids/prompts/sampling.py     prompt choice + field dropout at train time
scripts/make_splits.py       train/val/test + LOCO split -> configs/splits.csv
scripts/s1_train_segresnet.py  S1 training / validation / test
notebooks/02_s1_segresnet.ipynb  Kaggle notebook for S1
cids/models/sam_lora.py      decoder LoRA, 3D box-corner tokens, batched click simulation
scripts/s2a_sam_finetune.py  S2a: SAM-Med3D fine-tuned on images only (Exp 3 baseline)
notebooks/03_s2a_sam_decoder.ipynb  Kaggle notebook for S2a
scripts/s3a_seg2sam.py       S3a: SegResNet -> SAM, fully automatic (candidates, box prompts, jury)
notebooks/04_s3a_seg2sam.ipynb  Kaggle notebook for S3a
notebooks/05_s2b_sam_realistic_boxes.ipynb  Kaggle notebook for S2b (realistic + negative boxes)
notebooks/06_s3b_council_s2b.ipynb  Kaggle notebook for S3b (council with SegResNet-100 + S2b)
tests/                       pytest
```

## Step 1 — prompts (done locally)

```bash
python scripts/build_prompts.py \
  --clinical "C:/Users/bhara/Desktop/MAMA_MIA_COMPLETE/clinical_and_imaging_info.xlsx" \
  --splits   "C:/Users/bhara/Desktop/MAMA_MIA_COMPLETE/train_test_splits.csv" \
  --out data/prompts/cids_prompts.xlsx
```

One row per patient (1506). Columns:
- `prompt_S` main structured instruction · `prompt_S_hints` + knowledge hints · `prompt_S_nosubtype`, `prompt_S_noethnicity` ablations
- `narrative_1..3` Type-N (LLM A, all cases) · `narrative_heldout_1..3` (LLM B, test cases only, never trained on)
- `cf_age`, `cf_ethnicity`, `cf_field_strength` counterfactuals (test cases only)

Field dropout (fields → "unknown") is applied at training time, not stored in the sheet.

## Step 2 — narratives (Kaggle GPU)

Upload `data/prompts/cids_prompts.csv` in the `cids-assets` dataset, open
`notebooks/00_generate_narratives.ipynb` on Kaggle (GPU T4, Internet ON), run all.
Download `cids_prompts_final.csv`, then:

```bash
python scripts/update_excel.py --csv cids_prompts_final.csv --xlsx data/prompts/cids_prompts.xlsx
```

## Step 3 — S0: SAM-Med3D embeddings (Kaggle GPU, can run in parallel with step 2)

`notebooks/01_s0_sam_embeddings.ipynb` (input: the 2-channel cache; Internet ON).
For every case: 1 mm -> 1.5 mm, 128^3 crops (train 1 centred + 3 jittered, test 1 centred),
two SAM inputs (`subtraction` = post1 - pre, `post1`), frozen SAM-Med3D-turbo encoder -> 384x8^3 fp16.
Writes `sam_emb/{train,test}/<cache_id>.npz`, `index.csv`, and `zero_shot_clicks.csv`
(SAM-Med3D zero-shot, 1-5 GT-sampled clicks, official protocol) -> save output as Kaggle dataset `cids-sam-emb`.

## Step 4 — S1: SegResNet, council member B (Kaggle GPU)

`notebooks/02_s1_segresnet.ipynb` — set `EXPERIMENT='img'` (Exp 1, image only) or `'txt'`
(Exp 2, + clinical instruction via BioClinicalBERT → FiLM, 30% field dropout); run both in parallel
on two Kaggle accounts. Inputs: 2-channel cache + `cids-assets`. Split `configs/splits.csv`
(train 1080 / val 120 / test 306). 96³ patches ×4 per patient, DiceCE, AdamW + cosine, AMP; best epoch on
val Dice; test once → `test_metrics.csv` (Dice/HD95/NSD per case), `test_pred/` masks. Resumable; time budget.

## Step 5 — S2a: SAM-Med3D-turbo fine-tuned on images only (Exp 3 baseline, Kaggle GPU)

`notebooks/03_s2a_sam_decoder.ipynb` — inputs `cids-sam-emb` + 2-channel cache; Internet ON.
Frozen encoder (cached post1 embeddings), mask decoder adapted with LoRA r=8 on q/v (`TUNE='lora'`, ~71k params)
or fully (`'decoder_full'`, 7.6M) + 2 learnable 3D box-corner tokens. Train prompts: jittered GT box (50%) or a
random click, then up to 4 corrective clicks. Val score = mean(Dice@1 click, @5 clicks, @box) → best.pt.
Test: zero-shot vs fine-tuned with identical seeded clicks → `test_crop_metrics.csv` (crop Dice, 1.5 mm, steps 1–5),
`test_full_metrics.csv` (Dice/HD95/NSD at 1 mm; includes `gt_crop_ceiling` = resampling upper bound), `test_summary.json`.

## Step 6 — S3a: SegResNet → SAM, fully automatic (Kaggle GPU)

`notebooks/04_s3a_seg2sam.ipynb` — inputs: 2-channel cache + `cids-checkpoints` (`segresnet_img_best.pt` = S1 image-only best.pt,
`sam_lora_best.pt` = S2a best.pt); Internet ON. SegResNet candidates (components >= 0.1 ml, top 5) -> SAM crop + box per
candidate -> rules seg_rel20 / sam_rel20 / avg_rel20 / jury_t (agreement = Dice(SAM, SegResNet candidate)); rule chosen on val,
test once -> `test_metrics.csv`, `test_cases.csv`, `summary.json`, `test_pred/`. No ground truth is used for prompting.


## Step 7 — S2b: SAM trained on realistic + negative boxes (Kaggle GPU)

`notebooks/05_s2b_sam_realistic_boxes.ipynb` — same script as S2a with `--box_aug realistic`: box prompts are
clean (GT ± 5 vox, 40%), messy (0.7–1.8× size, shifted ≤ 40%, 40%) or negative (tumour-free tissue, empty target, 20%),
so SAM learns to handle SegResNet-like boxes and to answer "no tumour". Val score = mean(clean-box Dice,
messy-box Dice, negative rejection rate; rejection = output < 0.1 ml). 100 epochs from the original turbo weights.
Test adds `messy` and `neg` prompts and evaluates the S2a checkpoint on the same boxes (`--compare_ckpt`, model `compare`).
`RUN='s2a100'` re-trains S2a for 100 epochs with clean boxes (fair baseline).


## Step 8 — S3b: council with SegResNet-100 + S2b (Kaggle GPU)

`notebooks/06_s3b_council_s2b.ipynb` — same script as S3a (`scripts/s3a_seg2sam.py`) with the new checkpoints
(`segresnet_img_e100_best.pt`, `sam_s2b_best.pt`; `RUN='s2a'` uses `sam_lora_best.pt` for the comparison).
New rules `segkeep`/`samkeep` keep only candidates SAM did not reject (SAM output ≥ 0.1 ml). Also writes
`val_candidates.csv` / `test_candidates.csv` (one row per candidate: size, SegResNet confidence, agreement, SAM volume,
position, on-tumour) for the learned council, and reports how many false/true candidates SAM rejected.

## Tests

```bash
pip install -r requirements.txt
pytest -q
```
