# CADASIL supervision-density SFT mixtures

> GitHub 代码快照仅包含本目录的脚本、说明及依赖文件，不包含本地完整数据包的 frozen inputs、images、JSONL 或 manifests。以下“离线重建”命令针对持有这些本地冻结文件的环境；缺少 inputs 时不能直接重建原来的 100 条三组队列。主流程从源重跑后形成新的输入版本，也不等同于原始冻结队列。


This package builds real-source **image-only binary diagnosis** experiments. It never trains a model, calls a generative API, alters source images or modifies existing SFT code.

中文摘要：`audit_summary_zh.md`。

Read `audit_report.md` and `manifests/validation_results.json` first. These are research exports with source-reported labels, not independently adjudicated clinical gold data. In particular, verified independent patient IDs are unavailable. Existing CADASIL evaluation samples were used in earlier research and are not an untouched prospective cohort.

## Transfer to the other training machine

Copy this entire directory, including `images/`, the three dataset directories and `scripts/`. The full package also contains frozen `inputs/` for reproducible reconstruction. Keep `inputs/` out of training datasets: it contains raw diagnoses, original text, source image names and heldout provenance.

Paths in each JSONL are relative to its own dataset directory, e.g. `../images/<SHA256>.jpg`. The `image` and `image_path` fields are aliases; do not load the same image twice. Each record is one original full figure, potentially containing several panels, not one patient or necessarily one slice. Do not perform panel splitting independently between arms.

Test portable loading, without training:

```powershell
python scripts/load_export.py dataset_A_20pct/train.jsonl
python scripts/load_export.py dataset_A_20pct/val.jsonl
python scripts/load_export.py dataset_A_20pct/test.jsonl
```

Dependencies for reconstruction: Python 3.10+, Pillow, numpy, matplotlib. OCR is separate: its completed frozen results can be reused without installing its runtime. For a fresh pixel audit, install rapidocr_onnxruntime==1.4.4 and onnxruntime==1.20.1 in an isolated environment (Python 3.12, numpy==1.26.4, opencv-python==4.11.0.86 were used here). The fresh-audit command must finish before building mixtures. Diagnostic-term flags quarantine images; source bytes are unchanged.

## Rebuild

From this package directory:

```powershell
python scripts/build_mixture_datasets.py
```

This deterministically rebuilds generated mixture files/reports from frozen inputs with seed 42; it does not overwrite original RareMed/MedPix files. Existing frozen inputs with changed source hashes are rejected when re-snapshotting. Save a copy/version of this output before intentionally changing the experiment. Sampling is without replacement, using nested positive and negative prefixes and case round-robin ranking. Dataset ratio is measured in **image records**, not patients. Refer to the report for the actual maximum common N and exact case counts.

For initial collection only, with the original RareMed-Pairs project as parent:

```powershell
python scripts/build_mixture_datasets.py --source-root .. --download-medpix --prepare-only
python scripts/audit_pixel_text.py
python scripts/build_mixture_datasets.py
python scripts/validate_exports.py
```

MedPix image downloads use the authors' GitHub repository pinned to the revision in `inputs/medpix_github_tree.json`, with Git-blob SHA-1 verification and SHA-256 manifests. Public source: https://github.com/CHILab1/MedPix-2.0/tree/main/MedPix-2-0 . Original MedPix author train/dev/test IDs are retained in `inputs/medpix_author_split_uid_list.json` and audited; inherited author split seed is not assumed to be 42.

## Model inputs and targets

The existing local export `data/datasets/diagnosis-sft-v0.2` uses a user message with an image placeholder and fixed question, followed by an assistant text answer. These exports retain that layout. The actual training-machine QLoRA loader, processor revision, label masking and chat template are not available here, so runtime compatibility must be checked there before training.

Use **only the decoded RGB image and `messages`** in the model processor. `scripts/load_export.py` demonstrates that boundary. User message is identical across all records; assistant answer is exactly `CADASIL` or `non-CADASIL`. Final diagnosis, history, captions, case IDs and filenames are not prompt content. Source image bytes are unchanged and exported image filenames contain only hashes. Keep processor/preprocessing, training budget, prompt and output scoring identical across arms.

`case_history` is ancillary provenance, unused in this image-only experiment. RareMed history is blank because article-level context was not safely assigned to the individual image/case. MedPix history is a sanitized copy of the source History, dropping sentences with diagnosis/pathology terms. Raw history remains in frozen source cases. These two forms are recorded, but neither is an approved history-augmented training input. Do not silently add history to one arm.

## What the manifests mean

- `sample_manifest.csv`: eligible source pool; rows have original IDs, source case/article IDs, original and final splits, hashes and evidence levels. Eligibility is not the same as selection into every arm.
- `sample_membership.csv`: exact selected sample IDs for each arm/split.
- `dataset_comparison.csv`: actual ratios, record counts, provisional/source case counts and negative strata.
- `split_leakage_check.csv`: exact/case/article/conservative-component split checks.
- `raremed_record_audit.csv`, `medpix_record_audit.csv`, `upstream_figure_audit.csv`: inclusion/exclusion decisions and source label statuses.
- `negative_availability.csv`: real negative diagnoses and case/image availability, including scarce hard negatives.
- `label_reconciliation.csv`, if present: differences between old coarse diagnosis-family labels and source-diagnosis-based binary labels.
- `build_artifact_hashes.json`: generated table/JSONL/script hashes.

An image count, a training-record count, a source-case count and a patient count are different quantities. Source case IDs and article-local model case labels do not establish globally unique patients. Multiple images from a case always remain in one split.

## Evaluation rules

All three arms share byte-identical validation and test JSONLs. Use validation for checkpoint/ratio/threshold selection. Do not repeatedly evaluate test to pick settings. Freeze a separate MedQA evaluation if studying general capability retention; this package does not contain MedQA and cannot guarantee its performance is retained. Report CADASIL sensitivity with its actual small case denominator and false-positive rate by negative disease/difficulty, rather than interpreting all image rows as independent patients.

Original source licensing/attribution is retained in manifests; this package does not grant blanket image redistribution rights or assert that privacy/clinical review is complete. No model performance has been measured by this build.

An independent technical validation can be rerun with `python scripts/validate_exports.py`. See `manifests/pixel_answer_leakage.csv` for OCR findings. OCR screening cannot rule out all missed text or pixel PHI.

Source/domain confounding remains: positive training examples come from RareMed-Pairs and negatives from MedPix. A change in performance cannot be attributed purely to disease concepts without further cross-source controls.

For a fair comparison, score the unchanged base MedGemma and all three new SFT arms on the same frozen question, targets and heldouts. The earlier 239-example free-diagnosis SFT model differs in total data size and output task; treat comparisons to that model as exploratory rather than a controlled class-ratio experiment.

Reconstruction dependencies can be installed in a separate data-tool environment using requirements.txt; do not downgrade or replace the training-machine environment just for rebuilding these files. A portable ZIP can be rebuilt with `python scripts/package_exports.py`.
