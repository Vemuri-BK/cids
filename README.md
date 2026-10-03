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

## Tests

```bash
pip install -r requirements.txt
pytest -q
```
