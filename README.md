# CIDS — Clinical Instruction Dual Supervision

Frequency-prompted SAM-Med3D council for 3D breast DCE-MRI tumor segmentation (MAMA-MIA).

- **Member A:** SAM-Med3D (frozen encoder) + Frequency-Instructed Prompt Generator (3D Haar DWT bands gated by the clinical instruction) + LoRA mask decoder
- **Member B:** SegResNet conditioned on the instruction (FiLM + cross-attention); proposes the tumor box
- **Council:** jury head fuses both masks; **dual supervision** = expert ground truth (+ band-wise boundary loss) and reliability-weighted peer supervision
- **Prompt study (FairVLM-style):** Type-S structured vs Type-N narrative instructions, counterfactual invariance, fairness metrics

## Workflow

```
VS Code (this folder)  --git push-->  GitHub Vemuri-BK/cids (public)  --git clone-->  Kaggle notebook (GPU)
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
scripts/build_prompts.py     metadata -> data/prompts/cids_prompts.xlsx (+ .csv)
scripts/generate_narratives.py   LLM narratives on Kaggle (resumable)
scripts/update_excel.py      put Kaggle narratives back into the Excel sheet
notebooks/00_generate_narratives.ipynb   Kaggle notebook for the step above
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
`notebooks/00_generate_narratives.ipynb` on Kaggle (GPU T4, Internet ON), optionally pin EXPECTED_GIT_COMMIT, run all.
Download `cids_prompts_final.csv`, then:

```bash
python scripts/update_excel.py --csv cids_prompts_final.csv --xlsx data/prompts/cids_prompts.xlsx
```

## Tests

```bash
pip install -r requirements.txt
pytest -q
```
