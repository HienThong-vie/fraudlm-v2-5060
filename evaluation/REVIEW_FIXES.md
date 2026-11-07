# Grounding and runtime verification

## Implemented

- Shared answer checks apply to case and general chat. Case-specific contradiction checks remain an additional layer. Negative review phrases no longer trigger positive-flag contradictions.
- Regression checks cover protected-attribute reasons, invented scores and IDs, accusations, unsupported safety guarantees, general questions and correct negative flags.
- Runtime registry facts replace empty-input model-fact targets. Current case stores are checked against model identity, threshold and checkpoint hash before loading weights.
- Refreshed data use at most 500 sampled cases per source, preserve original instruction training rows and add modest deterministic wording variation. This remains template-assisted data, not a diverse human-written benchmark.
- Original CSV positions and prediction-file hashes are recorded. BAF templates distinguish model attributions from observed inputs.
- Historical BAF application exposure is screened from original validation/test rows. The current original-test audit excludes 90 of 500 rows and retains 410; screened rows remain private.
- Initialized training can resume without forcing a checksum mismatch. Deterministic settings are locked on resume; an opt-in CUDA deterministic mode is available but not validated on the PC.
- CPU tied embedding/output weights use owned storage, resolving the reproduced small-model generation crash. Kaggle bundles were regenerated.
- Public README wording matches the code repository's visibility. The continued model Release is an unpublished, recoverable draft; licensing clearance remains outstanding.

## Verification on the Mac

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m pytest -q --tb=short`

Final result after device profiles and classifier-identity checks: **436 passed, 1 skipped**, 171.20 seconds. The skip concerns the optional Transformers reference package. Real small-model chat/training and CPU resume checks ran after the public weights were cached. The earlier native crash is not hidden as a skip. The RTX profile's command routing/configuration is tested here; a fresh CUDA execution on the PC remains a separate check.

The refreshed training split contains 15,621 rows; tokenisation at max length 384 dropped/truncated none. Validation has 613 rows and test has 598 after historical exposure screening. The generated rows are local and are not committed. `refreshed_data_manifest.json` provides aggregate counts and checksums.

## Evaluation boundaries

The completed adapter has not been retrained on the refreshed data. The 0.0589 loss describes a mostly templated validation set, not improved generalisation.

The 32-context and 8-context generation results predate these guard changes. `guard_recheck.json` records separate replay of their saved raw answers under the strengthened guards. It is a known-answer regression check, not blind evaluation or evidence of learned reasoning.

The matched original-validation comparison is complete for both adapters. Run 1 / run 2 loss is 0.327504 / 0.336809 on all 500 rows and 0.379331 / 0.390314 on the 422 overlap-screened rows. Lower is better: continued training did not improve this diagnostic. See `original_validation_comparison.json` for full precision, hashes and protocol. Diagnostic loss does not replace task-wise semantic evaluation. Full original-skill retention, global entity independence and strict CUDA repeatability remain unverified.
