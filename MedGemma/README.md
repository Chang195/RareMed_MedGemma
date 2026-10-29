# MedGemma model experiments

Model: google/medgemma-1.5-4b-it, project-pinned revision 91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b.
The proven local recipe uses NF4 double quantization, FP32 compute, SDPA and language q_proj/v_proj LoRA rank 4.
This is an experimental 6 GB GPU setup, not the official full-precision evaluation configuration.

Install a compatible PyTorch build separately, then requirements.txt in a dedicated model environment.
The data-pipeline requirements in ../RareMed-Pairs are separate.
For non-model collection commands, avoid loading GPU weights or changing the proven model environment.

Use the repository-level README for current entry points. docs/ contains historical method notes,
some of which reference local artifacts that are intentionally excluded. Their existence is not evidence
that those datasets, results or model checkpoints are available in this code-only release.
No medical reference answers have been generated or relabeled by this release builder.
