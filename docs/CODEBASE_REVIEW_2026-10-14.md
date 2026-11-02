# CIDS codebase review — 2026-10-14

## Scope and validation

Reviewed the active Python package, stage scripts, seven test files, eleven notebook workflows, configuration, README, project log, and implementation-related claims in the paper draft. Parsed all 52 active Python files (5,655 lines) successfully. Archived and incoming copies under ignored `outputs/` are not treated as the active implementation. This is a code review, not independent replication of the reported research results or verification of the paper's external references.

Local checks:

- `python -m pytest -q -rs`: **11 passed, 7 skipped**. Six active test modules and one incoming duplicate module were skipped because PyTorch is absent. MONAI, medim, and transformers are also absent. The passing tests cover prompt processing, not model execution.
- Parsed every active Python file with `ast.parse` and inspected notebook source. No Python source syntax errors found.
- Checked `configs/splits.csv`: train 1,080 / validation 120 / test 306; no duplicate cache IDs. LOCO columns are present.
- Executed unchanged pure candidate-selection functions in isolation using NumPy/Pandas/SciPy to demonstrate the baseline and oracle issues below. These checks do not exercise the neural networks.
- Demonstrated that the narrative fact checker accepts an invented implant for metadata specifying no implant.

No training code was changed. Existing untracked work at review start: `cids/pipeline.py`, `scripts/s3d_export_pipeline.py`, and `docs/paper/`.

## Architecture and data flow

CIDS is a staged, Kaggle-oriented research repository for breast DCE-MRI tumour segmentation. It depends on an externally generated FADC cache; raw MRI ingestion and preprocessing are outside this repository.

| Component | Responsibility | Key contract |
|---|---|---|
| `cids/data/cache.py` | Load volumes, resample, crop/pad, normalize SAM inputs | Image `(2,X,Y,Z)`, pre/post1, 1 mm RAS; SAM crops 128 cubed at nominal 1.5 mm |
| `cids/data/volumes.py` | Three-channel input and patch sampling | Pre, post1, post1 minus pre; default 96-cubed patches |
| `cids/prompts/` | Normalize metadata; structured/narrative prompts; counterfactuals | Explicit selection of permitted source fields |
| `cids/text/encoder.py` | Frozen BioClinicalBERT, mean pooling, embedding cache | Sentence vectors used by FiLM |
| `cids/models/segresnet_text.py` | Image finder, optional text conditioning | SegResNet plus zero-initialized FiLM; no implemented cross-attention |
| `cids/models/sam_med3d.py`, `sam_lora.py` | SAM loading, decoder adaptation, box/click prompts | Frozen encoder; q/v LoRA; two learned 3D corner tokens |
| `scripts/s3a_seg2sam.py` | Automatic finder-to-checker pipeline | Connected components → crops/boxes → SAM → selection rules |
| `scripts/s3d_learned_jury.py` | Learned candidate selection | 11 inference-available features; train classifier, select model/threshold on validation |
| `cids/pipeline.py` | Bundle loading and image-to-mask inference | Includes neural weights, pickled jury, settings; still imports stage scripts |
| `cids/fcouncil/` | Frequency adapter, independent band experts, voxel gate, losses | Four experts receive separate Haar groups plus common finder/checker maps |
| `cids/eval/` | Dice, surface distances, NSD, connected-component filtering | Binary arrays on a common grid; default 1 mm spacing |

The main automatic path is:

```text
2-channel cache → 3-channel SegResNet → probability map
  → top connected-component candidates → SAM crop + box per candidate
  → SAM response features → learned jury → union of accepted SegResNet components
```

S3d therefore removes or keeps entire finder components. SAM informs the decision; its refined boundary is not the final S3d mask. This limits S3d's ability to repair finder under-segmentation or attached false tissue.

The experimental extension is:

```text
S4: Haar(SAM crop) → weighted bands → CNN residual → SAM embedding
S5: export finder and checker probability maps, optionally jury-weighted
S6: Haar(image) → 4 independent U-Nets + shared map inputs
    → voxel-wise softmax gate → weighted expert logits → final mask
```

S6 supervises the final prediction and every expert against ground truth, adds agreement-gated teacher distillation, and regularizes average gate usage. `--no_teachers` disables distillation, but retains finder/checker maps as model inputs. Its interpretation is therefore “no distillation,” not “no teacher information.”

The project log reports S3d as the best completed method: test Dice 0.735, versus 0.699 for the rule jury and 0.684 for the finder. These are recorded project results, not measurements reproduced in this review. The text track is parked; acquisition conditioning has reserved module interfaces but is not wired through training datasets/scripts. S4/S6 results remain unresolved in the supplied status log.

## High-priority findings

### 1. S2c does not enforce the minimum true-candidate retention constraint

**Location:** `scripts/s2a_sam_finetune.py:540`, `:563`, `:579`.

`best` starts at -1. Epoch-zero S2b validation is logged, but neither saved as the initial best checkpoint nor assigned to `best`. Later selection subtracts 1 from the score when `real_keep < min_real_keep`, then saves whenever this penalized score exceeds `best`.

A failing epoch with score 0.70 receives -0.30, which exceeds -1 and is saved. If every epoch fails the constraint, the best failing epoch remains selected. This contradicts the documented guarantee that an epoch can only be best if retention is at least 95%, and does not implement the logged fallback to S2b.

**Fix:** save the initial checkpoint and its validation score; explicitly reject ineligible epochs. Persist eligibility and selection scores. Audit the historical S2c artifact against its run source before claiming that the selected checkpoint equals S2b. The present code discrepancy alone does not establish what happened in a previously modified Kaggle run.

### 2. S6 can lose the historical best model when resuming into a new output directory

**Location:** `scripts/s6_freq_council.py:170–179`, `:232–237`.

Resume restores the latest model and historical best score but does not copy/load the old `best.pt`. If no subsequent epoch exceeds the historical score, the final fallback saves the latest weights as `best.pt`. The test then evaluates these weights while reporting the old best validation score.

**Fix:** restore both last and best checkpoints, or include the best weights in the resume artifact. Require the selected artifact to match the reported validation record.

### 3. Cache reuse is not tied to the experiment that produced the cache

**Locations:** S0 `:89`; S2c mining `:102`; S3d `:75`; S5 `:88`.

Existing files are skipped based on filename alone. Changing checkpoints, thresholds, top-k, overlap, split definitions, adapter, jury, preprocessing, or a smoke-test limit can silently reuse incompatible data. S3d can reuse old train/validation feature CSVs while recomputing test features with new models. S5 can mix old and new maps in one directory.

**Fix:** store and validate a manifest with source hashes, preprocessing settings, expected case IDs, and schema version. Write artifacts atomically and mark completion only after validation.

### 4. S3's `seg_rel20` is not the shared full-mask rel20 baseline

**Locations:** `scripts/s3a_seg2sam.py:83`, `:161`; `scripts/s3d_learned_jury.py:116`, `:245`; `cids/eval/postproc.py`.

The shared rel20 keeps every component at least 20% of the largest. S3 first truncates candidates to five and applies a 100-voxel filter, then computes rel20 over those candidates. Thus it can report a different baseline under the same name.

**Reproduction:** six separate 125-voxel components give 750 voxels under shared rel20, but 625 under S3's baseline. Minimum-size filtering introduces another discrepancy. This establishes a semantic difference, not its frequency in the actual dataset.

**Fix:** compute baseline rel20 directly on the original finder mask. Keep candidate restrictions for the candidate-based methods. Recompute paired comparisons if actual baseline masks differ.

### 5. The all-touching “oracle” is not a maximum achievable Dice bound

**Location:** `scripts/s3d_learned_jury.py:123–125`; project log and paper upper-bound language.

Keeping every candidate with any ground-truth intersection can add a large mostly-false region and reduce Dice. The table arithmetic is correct, but the selection policy is not Dice-optimal.

**Reproduction:** a 100-voxel component with 100 true voxels plus a 1,000-voxel component with one true voxel, against 101 GT voxels: all-touching Dice is 0.1682; keeping only the first gives 0.9950.

**Fix:** rename this “all-overlapping candidates,” or enumerate all subsets to obtain the true per-case best Dice. With at most five candidates, only 32 subsets are needed. An oracle prompt experiment is also an upper reference, not a mathematical upper bound on every other method.

### 6. LOCO full-resolution SAM evaluation uses the wrong cache folder

**Location:** `scripts/s2a_sam_finetune.py:411`, `:415`.

`--split_col` supports LOCO, whose test patients include both official train and official test cases. Full-resolution evaluation nevertheless loads every case from `cache_root/val`. Held-out-collection cases originally in official train live under `train`, causing missing-file failures.

**Fix:** carry original cache paths or the official split through preloading and use them for metrics. The S4 path handling already uses the official split. Most downstream S3–S6 scripts also hardcode the ordinary `split` column, so end-to-end LOCO is not yet supported consistently.

## Other correctness and operational findings

| Priority | Finding | Consequence and action |
|---|---|---|
| Medium | Missing case files are silently filtered in most stage scripts | Runs can evaluate incomplete cohorts while printing fixed labels such as “306.” Validate expected IDs, nonempty splits, and actual counts; make partial mode explicit. |
| Medium | S0 saves each NPZ immediately but writes index/metric rows only every 50 cases (`s0_sam_embeddings.py:125–138`) | Interruption loses pending table rows; resume skips the already-written NPZs. Reconstruct tables or checkpoint metadata per completed case. |
| Medium | `CIDSPipeline` changes global `s3.DEV`, and inference helpers use that global (`pipeline.py:46,59`) | Loading a second pipeline on another device can break the first. Pass device explicitly or infer it from model parameters. Normalize `cuda:0` and `torch.device` inputs; current AMP selection tests exact string equality with `cuda`. |
| Medium | `check_facts` has no negative implant rule (`narrative.py:101,135`) | A paragraph saying “Breast implant present” passes for `implant=no`; reproduced locally. Presence checks also do not generally understand negation. Add contradiction tests and explicit value-aware matching. |
| Medium | Single-case zero-shot `sample_click` assumes nonempty GT when there is no error (`sam_med3d.py:46–49`) | Empty labels, including lesions lost in resampling, produce an empty random-index range. Match the defensive handling in batched `sample_clicks`. |
| Medium | S2 training can stop before its first validation and then load nonexistent `best.pt` | Save an initial/evaluated fallback or require validation before exit. Validate batch size against available training cases to avoid zero-batch runs. |
| Medium | S4 has no resume checkpoint or time-budget handling | Interrupted Kaggle training loses optimizer state and progress. Add `last.pt`, RNG state, configuration validation, and resume support. Epoch-zero validation is not included in best selection. |
| Medium | S3d's all-empty candidate table lacks feature columns | A split with no candidates can fail in `rule_tables` or classifier fitting rather than return valid empty-mask metrics or a useful error. Use a fixed schema and check class availability. |
| Low | S5 time-budget `break` exits only the current split loop | It may begin later splits after the budget has expired. Use an outer stop flag as in S2c mining. |
| Low | Export `--check` prints reloaded-model Dice but does not compare with pre-save predictions or S3d reference predictions | It is a smoke check, not an equivalence test. Compare masks, probabilities, selected IDs, and settings before/after save. |
| Low | Export loading ignores its version tag and hardcodes model architecture/LoRA choices | Validate bundle version and record architecture, library versions, and preprocessing contract. Reject unsupported checkpoint types clearly. |

## Experimental interpretation

1. **Test-driven iteration needs transparent reporting.** The log explicitly uses earlier test failures to motivate later designs and includes hindsight test-threshold analysis. Threshold selection for S3d is correctly performed on validation in code, but the project-wide process is not an untouched-test design. Preserve a final external or otherwise untouched evaluation for confirmatory claims; label exploratory analyses honestly.
2. **Cross-fitting covers the jury, not the finder/checker training pipeline.** `CrossFitJury` correctly groups by patient. However, finder maps and candidate features for training patients come from a finder trained on those patients. Map augmentation and jury cross-fitting do not make the upstream features out-of-fold. Treat this as a stacking distribution mismatch and test out-of-fold upstream maps where feasible.
3. **Teacher corruption also corrupts distillation targets.** Council map dropout/gamma/noise modifies input channels 3/4; those same modified maps are passed as teacher targets in S6. Jointly dropped or near-zero maps can become agreeing background targets even on GT-positive voxels. This may be intentional regularization, but should be separated experimentally from distilling clean teachers. Keep clean target maps and an availability mask if the intended supervision is from actual teacher predictions.
4. **The teacher agreement threshold is permissive and not configurable through S6.** The loss uses absolute probability difference below 0.5, allowing pairs on opposite sides of the segmentation threshold to agree. Validate this definition and expose the threshold if it is an experimental parameter.
5. **Metric conventions must be explicit.** HD95 pools both directions and returns NaN for one-empty masks; mean/median aggregation consequently excludes those failures. NSD counts boundary voxels equally. Document the implementation and valid HD95 case count; do not imply automatic equivalence with differently defined surface metrics. The existing miss count helps but does not show the HD95 denominator explicitly in all stages.
6. **The S3d fallback assumes every examination contains tumour.** If candidates exist, at least one survives even with uniformly low jury probabilities. This is appropriate only for the stated tumour-positive cohort contract; negative-case performance has not been established by that policy.
7. **S6 uncertainty is expert disagreement, not calibrated confidence.** Shared maps, supervision, and gate training can make experts agree while wrong. The existing uncertainty-versus-Dice analysis is useful, but does not establish probabilistic calibration.
8. **Band weights alone do not prove band importance.** Downstream convolutions can compensate for input scaling. Use controlled band ablations or perturbations to support interpretability claims.

## Reproducibility, security, performance, and maintenance

- `requirements.txt` does not explicitly list scikit-learn, although S3d and jury-weighted S5 require it; tqdm is also directly imported but undeclared. Transitive installs may supply these packages, but the direct dependency contract is incomplete. Most versions, model revisions, and notebook commit pins are unset. There is no project packaging metadata or CI configuration in the active tree.
- Notebook `!pytest` and `!python` calls generally do not enforce fail-fast execution. Notebook 04/06 setup also deletes previous output directories automatically. Prefer checked subprocess calls and unique run directories.
- `pytest -q` discovers an incoming duplicate under ignored `outputs/`. `.gitignore` does not control test discovery. Set `testpaths=tests` and separate lightweight versus required model test jobs. Skipping all model tests must not be accepted as model validation.
- Resume artifacts do not preserve all RNG states. In S1 the prompt RNG restarts; S2 click generators restart. Resumed optimization is therefore not guaranteed to match uninterrupted training even where case order and patch sampling are epoch-seeded.
- Model loading uses `torch.load(..., weights_only=False)` and jury loading uses pickle. These formats have an executable-deserialization trust boundary; distribution requires trusted artifacts. The current bundle also lacks hashes and exact environment metadata. This is a local artifact-loading concern, not evidence of a network-facing exploit.
- `cids/pipeline.py` imports orchestration scripts by editing `sys.path`, coupling the library to repository layout and global state. Move reusable inference, feature extraction, and model-construction functions into package modules, leaving scripts as CLI wrappers.
- `configs/cids.yaml` is not consumed by the active Python training scripts. It contains stale settings such as subtraction SAM input and planned loss parameters. README describes cross-attention and instruction-gated frequency prompts beyond the active implementation. Align architecture descriptions and configuration with the current S3d/S4/S6 separation.
- `sam_refine` retains a full-resolution float probability volume per candidate. S3d needs only agreement and volume features, so this work is avoidable for its path. Add a feature-only mode and batch/chunk SAM encodings to control memory.
- SAM fine-tuning preloads several gigabytes of embeddings; S4 additionally holds crop arrays. Full-volume sliding-window outputs and S6's ten output channels can consume substantial device memory. Record peak memory in actual Kaggle smoke runs rather than infer safety from unit tensor shapes.
- HD95 and NSD recompute the same surface-distance transforms separately. Compute them once in `all_metrics` and share the distances.
- `TextEncoder` retains an unbounded GPU embedding cache, including numerous field-dropout variants. A bounded CPU cache would make longer text runs more predictable.
- Validate image shape, finite values, channel count, normalization, and geometry at the pipeline boundary. The bundle currently relies on callers honoring its docstring contract; it does not perform raw-image preprocessing.

## Strengths and recommended order of work

The separation of data utilities, prompts, model components, and metrics is understandable. Candidate cropping is shared between mining and inference, and tests verify that labels do not alter the crop/box prompts. Zero-initialized FiLM, LoRA output, adapter residuals, and gate weights are sensible controlled starting points with targeted tests. Patient-grouped jury cross-fitting, fixed split generation, and explicit oracle-versus-automatic distinctions are valuable foundations.

Prioritize work in this order:

1. Fix S2c selection and S6 resume; add regression tests that assert the exact selected checkpoint.
2. Correct baseline/oracle definitions and LOCO paths; audit any affected reported results.
3. Enforce dataset completeness and cache manifests; prevent mixed-experiment artifacts.
4. Run the complete model suite in a pinned PyTorch/MONAI/medim environment, followed by real-checkpoint smoke and bundle-equivalence checks.
5. Resolve clean versus augmented teacher targets, upstream out-of-fold inputs, and metric reporting before interpreting S6 gains.
6. Extract a stable inference package, validate bundle/input contracts, and update README/configuration and manuscript claims.

The repository is a useful research prototype with a coherent candidate-verification approach. Its main readiness gaps are experiment integrity, interrupted-run behavior, and reproducibility; reported segmentation quality alone does not resolve those gaps.
