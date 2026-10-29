# Frozen aggregate extraction results

These are source-text extraction/association results for GBM, astrocytoma and CADASIL. They are not MedGemma diagnostic performance or SFT outcome measurements.

- `metrics.csv`, `matrices.json`: frozen task-level confusion matrices and metrics.
- `disease_statistics.csv`: disease-specific Precision, Recall and F1 with 95% source-cluster bootstrap intervals.
- `cluster_confusions.json`: anonymous aggregate sufficient statistics to recompute those intervals. No clinical narratives, captions, source case identifiers or images are included.
- `coverage.json`: planned/validated coverage and task limitations.
- `figures/`: numerical confusion-matrix plots, without medical images.

Run `python tools/replay_public_results.py` to verify estimates and confidence limits offline. The ledger aggregates existing predictions; it does not independently establish ground-truth validity or regenerate model outputs.

MultiCaRe disease strata are mentions, not confirmed diagnosis labels. Its anatomy label is HEAD, broader than brain. MedPix source-description recovery is adapter/import integrity, not independent semantic alignment. No positive CADASIL MedPix references means CADASIL diagnostic sensitivity cannot be estimated from that source; unscorable results remain explicit.
