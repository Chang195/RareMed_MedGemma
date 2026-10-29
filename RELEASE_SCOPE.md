# Release scope and rights

This package includes explicitly selected source code, configurations, documentation, and the original
RareMed-Pairs pipeline's already allowlisted aggregate extraction metrics/plots. It excludes raw medical
images, patient/case text, source-response caches, benchmark question files, per-item predictions,
models/adapters/checkpoints, local virtual environments, and unpublished slide decks.

No code license has been selected. Publishing to GitHub does not automatically grant an open-source license.
Third-party source licenses and model terms remain applicable. This package does not relicense medical materials.

The release builder preserves frozen local experiment source files. Only release copies replace machine-specific
G-drive path literals with RAREMED_LOCAL_ROOT / a home-directory default. Full before/after hashes stay in
the local build audit. Original RareMed-Pairs source files remain byte-identical.

Historical replay and end-to-end evaluation require separately obtained data, accepted model terms,
and local run manifests/checkpoints. CPU-only checks do not validate GPU model performance.
Release path adaptations change source hashes. Use the original local source to resume a frozen historical
experiment. Do not replace its source with this portable release or bypass its protocol checks.
Secret scanning is a bounded pattern scan, not a guarantee that every possible sensitive value is detected.
