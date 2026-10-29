# RareMed-Pairs / MedGemma experiments

Public-literature image–case–source diagnosis curation and local MedGemma QLoRA experiments.
The two directories must remain siblings. No patient-level clinical validation is claimed.

## Layout

- `RareMed-Pairs/`: original curation pipeline, frozen source locks, prompts, schemas and aggregate extraction results.
- `MedGemma/`: local model training/evaluation and versioned collection adapters.

No original medical images, article/case text, JSONL datasets, API payloads, predictions, tokens or model weights are included.
The original curation environment and model environment have different dependency pins. Use separate Python 3.11 environments.
Do not install both requirements files into one environment.

## Local paths and credentials

Set `RAREMED_LOCAL_ROOT` locally for model/data/cache storage before model commands.
Release copies replace legacy G-drive literals with that variable or a home-directory default.
Some historical experiment commands still require local manifests/checkpoints not distributed here.
Do not interpret the presence of a script as a turnkey reproduction of unpublished runs.
Hugging Face acceptance/login is local. DeepSeek reads `DEEPSEEK_API_KEY` from the environment and requires explicit API opt-in.
Never commit `.env`, credentials or the generated working folders.

## Recommended collection order

In `MedGemma/`, use `run_other_disease_search_1000.py` for GBM/astrocytoma first-page searches,
`run_tumor_fulltext.py` for original-rule BioC screening, and `run_tumor_images.py` for official PMC Cloud images.
Each supports a check/plan phase. See `MedGemma/docs/TUMOR_CLOUD_ACQUISITION.zh-CN.md`.
`collect_independent_test.py` supplies a history helper but its expanded-query CLI is disabled.
The old `run_other_disease_search.py` is retained for dependency compatibility only. Do not use its multi-page CLI.
Historical CADASIL v2 pagination is separate from the single-page tumor protocol.

## Scientific boundaries

Source-reported diagnosis is not independently adjudicated clinical ground truth.
Article, figure, case group and patient are different units. Keep source/case groups isolated and review cross-article duplicates.
Use train only for fitting, validation for selection, and a frozen independent test for final evaluation.
The current collection adapters do not automatically create a clinically reviewed test set.
Direct-answer MedQA differs from the official reasoning protocol.

## Checks

In `MedGemma/`: install Pillow, then `python -m unittest test_pmc_cloud_images.py test_pipeline_runtime_repair.py -v`.
One optional local-run integration test skips when its unpublished fixture is absent.
In a separate environment for `RareMed-Pairs/`: install its requirements and run `python tools/run_tests.py`.

See `RELEASE_SCOPE.md` for limitations and rights. No open-source license has been selected.
