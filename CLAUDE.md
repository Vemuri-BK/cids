# CIDS — context for Claude sessions

Read `docs/PROJECT_LOG.md` first (status board + daily log). At the end of every working
session, add/extend that day's entry and update the status board, then commit.

## Project
CIDS = Clinical Instruction Dual Supervision: SAM-Med3D-turbo (frozen encoder) + text-conditioned
SegResNet council with Haar-wavelet frequency prompts gated by clinical instructions, for 3D breast
DCE-MRI tumour segmentation on MAMA-MIA. Target: IEEE JBHI / TMI.

## Workflow
- Code: this folder (VS Code) → `git push` → public GitHub `Vemuri-BK/cids` → Kaggle notebook clones it
  and pins `EXPECTED_GIT_COMMIT`. Commit author: Vemuri-BK.
- Data never goes to GitHub (`data/` is git-ignored). Kaggle datasets:
  - 2-channel MAMA-MIA cache `mama-mia-preprocessed-cache-2ch` (auto-detected under /kaggle/input)
  - `cids-assets` (cids_prompts.csv), `cids-sam-emb` (S0 embeddings)
- Two Kaggle accounts may run notebooks in parallel.

## Conventions / facts
- Cache npz: `image` (2,X,Y,Z) float16 [pre, post1], 1 mm, RAS; `label` (1,X,Y,Z). Split: train/ = 1200, val/ = 306 (= official test).
- `cache_id` = lower-case patient_id (duke_001). Clinical sheet uses upper case.
- SAM-Med3D input: post1, 1.5 mm, 128³, z-norm over voxels > 0 (subtraction was clearly worse zero-shot).
- Never put pCR/treatment/follow-up/multifocal/bilateral_breast_cancer into prompts (leakage).
- Expert masks label only the primary tumour (one breast); 30 bilateral-cancer patients are flagged at evaluation.
- Narrative filters live in `cids/prompts/narrative.py`; if they change, re-run `scripts/refilter_narratives.py` (no GPU).
- Run `pytest -q` before committing.

## Git from Claude sessions
- Claude may commit (author Vemuri-BK) but must first get delete permission for this folder,
  so git can remove its own lock files (.git/index.lock, HEAD.lock). If locks are left behind,
  remove them before the next git command. vbk also commits/pushes from VS Code; pushing is done by vbk.
