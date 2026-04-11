# Fixes Notepad

Working list of follow-up fixes and investigations for the research pipeline.

## Open Items

- [ ] ElasticNet convergence on the real Stage 05 run
  - Observed in `scripts/05_tune_models.py` during the `elasticnet` study on the real dataset.
  - Current implementation uses `MultiTaskElasticNet` with feature scaling, but some sampled parameter regions still emit real convergence warnings.
  - Follow-up:
    - expose solver controls such as `tol`,
    - consider increasing `max_iter`,
    - tighten the low-regularization search space so obviously unstable trials are not sampled.

- [ ] LightGBM Stage 05 runtime is structurally expensive
  - Current implementation trains one `LGBMRegressor` per target cell.
  - With the committed `9 x 9` surface grid, that means `81` separate booster fits per split per trial.
  - Follow-up:
    - review whether the benchmark design should stay per-cell,
    - profile whether a lower-dimensional target representation would preserve benchmark value at lower runtime cost.

- [ ] Investigate repeated LightGBM GPU warnings during Stage 05
  - During the real `lightgbm` tuning run, the console repeatedly prints `1 warning generated.`
  - The tuning stage completed successfully, so this is not currently a hard failure condition.
  - Follow-up:
    - capture the exact originating warning text,
    - determine whether it comes from the GPU backend, OpenCL compilation, or LightGBM wrapper behavior,
    - fix the underlying cause if it indicates a real runtime or correctness issue.

- [ ] Improve Stage 05 progress visibility for slow model families
  - Current progress advances once per completed split, not per inner target fit.
  - That is acceptable for ridge and HAR, but it makes `lightgbm` look stalled because one visible step contains many booster fits.
  - Follow-up:
    - decide whether to keep the current coarse progress model,
    - or add model-specific progress detail for long-running per-target estimators.

- [ ] Strengthen Stage 01 and Stage 02 reproducibility granularity
  - Earlier audit finding: Stage 01 and Stage 02 manifests hash summary artifacts, but do not individually record every produced bronze/silver parquet artifact in the same granular way later stages do.
  - Follow-up:
    - review whether per-artifact hashing should be added for all generated parquet outputs in those stages.

## Resolved Recently

- [x] Fail-fast schema drift now accepts audited vendor extras while still rejecting unknown unexpected columns.
- [x] Atomic artifact writes are hardened against transient Windows rename locks.
- [x] Stage 08 hedging now derives one daily SPX spot state from the median valid `active_underlying_price_1545` instead of assuming one identical value across all option rows for a date.
- [x] README workflow ordering and `sync-dev` requirement were aligned with the actual `Makefile`.

## Notes

- This file is intended as a running engineering notepad, not a formal audit report.
- Keep items here until they are either fixed, rejected with reason, or moved into a more formal issue tracker.
