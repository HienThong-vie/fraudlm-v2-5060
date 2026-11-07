# Final model grounding

The prediction registry now points to one frozen, validation-selected classifier per source. The headline review threshold targets 5% FPR on calibration data; actual test FPR can differ. Test results are exploratory because the periods were inspected repeatedly.

Local `dataset_final_v2/` contains 4,000 sampled prediction contexts from eight sources, capped at 500 per source, plus the original instruction data in its SFT splits. Model-fact targets are omitted. The earlier 40,000-case `dataset_final/` is preserved for reproducibility. Inputs exclude target labels. Available input signals are not causal explanations or model attributions. Elliptic results are offline AML results, not a validated live banking service.

The rebuilt splits contain 15,621 training, 613 validation and 598 test rows. Original contexts exposed during the completed run are excluded from validation (78 rows) and test (90 rows). These are conservative application/context checks, not proof of globally independent entities. The training split has no dropped or truncated examples at the configured 384-token cap.

The existing adapter has **not** been retrained on these refreshed contexts. The previous registry is preserved as `configs/prediction_sources_legacy.json` for reproducibility. Generated data and private answer logs are intentionally not tracked.

## Run

Use the existing environment and the arguments documented by `python chat_multisource.py --help`. Its default prediction store is now `dataset_final_v2/prediction_store.jsonl`. Supply that local directory separately when cloning; it is not on GitHub. `facts <dataset_id>` reads current model settings directly from the registry.

For a separate future training run, use `configs/rtx5060_final_models.json` with the existing training entry point. This starts from the completed adapter but writes a new checkpoint directory. It does not overwrite the old run. Check the training script's `--help` before launching on the PC.

## Evaluation scope and limitations

`evaluation/continued_adapter/summary.json` records actual generation on 32 fixed, outcome-balanced examples from the original run's instruction holdout. All 32 repeated the score, threshold and flag correctly under automatic string checks. This is not population accuracy, semantic advice quality, or proof of improvement over the previous adapter; underlying entity overlap is not fully ruled out.

`evaluation/refreshed_context_probe/summary.json` records eight refreshed-context smoke cases: score and threshold repetition were 6/8; flag correctness was 7/8. These historical results used the earlier guard. Manual review found an invented Elliptic case number, an unsupported BankSim explanation and an unsafe ULB safety claim. The strengthened guard now checks shared answer rules, explicit case IDs, contradictory flags and unsupported safety guarantees. `evaluation/guard_recheck.json` separately records replay of saved raw answers; it is a known-answer regression check, not a new blind evaluation. Do not use the adapter autonomously for financial decisions.

`evaluation/original_validation_comparison.json` is complete for both adapters. Response-token-weighted cross-entropy, lower is better:

| Adapter | Original 500 validation rows | Overlap-screened 422 rows |
|---|---:|---:|
| Original `xl_r16_v1` | 0.327504 | 0.379331 |
| Continued `xl_r16_multidata_5060_v1` | 0.336809 | 0.390314 |

Both use FP32 merged adapters, the same data and masking, micro-batch one, max length 1024 and EOS targets. Run 2 is slightly worse on this diagnostic; the 0.0589 templated-set loss does not establish improved original-task ability. `evaluation/original_holdout_audit.json` records screening of the original test split; private retained rows are under `.private/evaluation/clean_original/`. Broad semantic skill retention and independent entity-level evaluation remain unfinished.

The public summaries contain no raw case input records. Raw generated answers remain private under `.private/evaluation/`. The existing CUDA strict-repeat reproducibility discrepancy remains unresolved; these checks used CPU inference on the Mac.

## Verification

Focused code, guard, dataset and resume checks passed on the Mac. A native CPU crash in single-token generation was resolved by allocating owned storage for the tied embedding/output matrix rather than retaining mapped-file storage. All 33 chat tests subsequently passed, including real small-model integration checks. Strict CUDA resume repeatability remains a separate unresolved limitation.

Final suite: 436 passed, one optional Transformers reference test skipped. Device routing/configuration is covered; fresh RTX 5060 execution was not performed on this Mac.
