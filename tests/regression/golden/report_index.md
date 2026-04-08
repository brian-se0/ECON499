# Report Artifacts

- Benchmark model: `no_change`
- Primary loss metric: `mean_observed_wrmse_total_variance`
- Best full-sample loss model: `no_change` (0.004209)
- Best hedging revaluation model: `no_change` (0.888332)
- MCS included models: neural_surface, ridge
- Interpolation sensitivity summary: mean RMSE diff 0.004684, max abs diff 0.009312

## Strongest Slice Gains

| slice_family | slice_label | evaluation_scope | best_model_name | best_metric_value | benchmark_model | benchmark_metric_value | improvement_vs_benchmark_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| maturity | 30d | full | neural_surface | 0.005514 | no_change | 0.005639 | 2.206527 |
| maturity | 60d | full | neural_surface | 0.003357 | no_change | 0.003372 | 0.453887 |
| maturity | 90d | full | no_change | 0.004944 | no_change | 0.004944 | 0.000000 |
| maturity | 30d | observed | neural_surface | 0.006096 | no_change | 0.006211 | 1.843531 |
| maturity | 60d | observed | neural_surface | 0.003357 | no_change | 0.003372 | 0.453887 |
