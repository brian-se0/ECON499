# Round 002F Batch 1 Synthesis Response

Captured: 2026-04-27

BATCH_1_LITERATURE_STANDARD

15:45 availability must be explicit and field-level.
ECON499 may use only quotes, underlier values, forwards, rates, dividends, filters, and derived fields available at or before 15:45 America/New_York. The close-based SPX papers support daily IVS forecasting but do not justify same-day EOD leakage; 2406-11520v3 supports intraday S&P 500 option-surface construction but not ECON499’s exact 15:45 next-surface forecast.
Support: ebsco-fulltext-04-08-2026, dynamics-of-implied-volatility-surfaces, s11147-023-09195-5, s11147-020-09166-0 direct SPX/S&P 500 daily evidence; 2406-11520v3 direct S&P 500 intraday evidence.

The option universe and every filter/drop rule must be reproducible and logged.
Filters for DTE, moneyness, price, duplicate records, no-arbitrage violations, IV inversion failures, liquidity/volume, OTM call/put selection, and option type must be parameterized, counted, and testable.
Support: ebsco-fulltext-04-08-2026, dynamics-of-implied-volatility-surfaces, s11147-023-09195-5, s11147-020-09166-0, 2406-11520v3; direct SPX/S&P 500 evidence plus broader IVS methodology.

Surface coordinates must be declared in artifact metadata, not inferred.
ECON499 must state whether it uses spot moneyness, forward moneyness K/F, log-forward moneyness, normalized log moneyness, delta, strike, or transformed coordinates such as sqrt(τ) and k/sqrt(τ). Forward, spot, ATM-IV, rate, and dividend inputs must be timestamp-safe.
Support: ebsco-fulltext-04-08-2026, dynamics-of-implied-volatility-surfaces, s11147-023-09195-5, s11147-020-09166-0, 2406-11520v3; direct SPX/S&P 500 evidence and broader IVS methodology.

Maturity and time-to-expiry conventions must be versioned.
DTE bounds, year-fraction convention, trading-day versus calendar-day treatment, near-expiry handling, and maturity interpolation must be explicit. Near-expiry and long-expiry behavior should not be silently extrapolated.
Support: ebsco-fulltext-04-08-2026, dynamics-of-implied-volatility-surfaces, s11147-023-09195-5, s11147-020-09166-0; direct SPX/S&P 500 evidence plus broader IVS methodology.

Observed option data and completed-grid data are different objects.
Raw observed quotes/options and observed grid cells must remain distinguishable from fitted, smoothed, interpolated, extrapolated, or neural-generated values. Completed-grid values are model outputs, not raw market observations.
Support: all five Batch 1 papers; strongest direct evidence from dynamics-of-implied-volatility-surfaces, s11147-023-09195-5, s11147-020-09166-0, and 2406-11520v3.

Observed-cell masks are mandatory.
Every surface artifact used for modeling or evaluation must preserve an observed-cell mask, and ideally richer provenance flags: observed, interpolated, extrapolated, artificial support, generated-only, or missing.
Support: direct inference from irregular/sparse SPX evidence in dynamics-of-implied-volatility-surfaces, s11147-023-09195-5, s11147-020-09166-0, and 2406-11520v3.

Interpolation and smoothing must be treated as model choices.
Kernel, spline, SVI, neural-operator, or other construction methods require serialized method names, hyperparameters, bandwidths, minimum-observation rules, domain bounds, extrapolation rules, and version IDs.
Support: s11147-023-09195-5, s11147-020-09166-0, 2406-11520v3; direct S&P 500 evidence and broader IVS methodology.

Surface-construction validation must use observed data, not only fitted grids.
Cross-sectional holdout, leave-one-out, dropped-point tests, or similar observed-option validation should evaluate interpolation and extrapolation quality. This validation does not replace blocked time-series validation for forecasting models.
Support: s11147-023-09195-5, s11147-020-09166-0, 2406-11520v3; direct S&P 500 evidence and broader IVS methodology.

Observed-cell and completed-grid forecast metrics must be reported separately.
Aggregate completed-grid RMSE/MAE can hide sparse-region and interpolation artifacts. ECON499 should separately report observed-cell loss, completed-grid loss, interpolated-cell loss, extrapolated-cell loss, and coverage counts.
Support: direct inference from s11147-023-09195-5, s11147-020-09166-0, 2406-11520v3; supported by traded-option evaluation in ebsco-fulltext-04-08-2026.

Moneyness/maturity slice diagnostics are mandatory.
Reports should include coverage, errors, and construction diagnostics by moneyness and maturity buckets, especially short maturities and tail moneyness regions.
Support: ebsco-fulltext-04-08-2026, dynamics-of-implied-volatility-surfaces, s11147-023-09195-5, s11147-020-09166-0, 2406-11520v3; direct SPX/S&P 500 evidence.

Completed-surface robustness is required before strong research claims.
Because downstream option-implied measures can change materially across surface-construction methods, ECON499 should report interpolation/smoothing sensitivity or at least parameter/domain robustness for completed-grid claims.
Support: strongest in s11147-020-09166-0; reinforced by s11147-023-09195-5 and 2406-11520v3; direct S&P 500 evidence plus broader IVS methodology.

No-change and factor benchmarks must be coordinate-explicit.
A no-change benchmark must specify sticky moneyness, sticky strike, fixed-grid no-change, or another coordinate convention. Factor/PCA/HAR-style benchmarks are academically grounded because SPX IVS dynamics exhibit low-dimensional level, skew/slope, and curvature variation.
Support: ebsco-fulltext-04-08-2026 and dynamics-of-implied-volatility-surfaces; direct SPX/S&P 500 evidence.

Factor construction must be split-safe.
PCA, coefficient surfaces, scalers, smoothing choices used as features, and dynamic benchmark parameters must be fit only inside the training window for each split.
Support: inference from ebsco-fulltext-04-08-2026 and dynamics-of-implied-volatility-surfaces; direct SPX/S&P 500 methodology, with leakage discipline imposed by ECON499.

Arbitrage-aware language is the Batch 1 ceiling unless hard constraints are proven later.
Batch 1 supports arbitrage diagnostics and soft penalties, but does not establish that ECON499’s neural model is arbitrage-free. Soft butterfly/calendar penalties justify “arbitrage-aware,” not “arbitrage-free.”
Support: ebsco-fulltext-04-08-2026 explicitly notes its unrestricted VAR does not exploit no-arbitrage restrictions; 2406-11520v3 uses soft arbitrage penalties; broader IVS methodology.

Intraday smoothing evidence is not the same as next-day forecasting evidence.
2406-11520v3 supports intraday/nowcasting construction from current quotes and irregular observed support, but ECON499 still needs separate time-series forecasting validation for tomorrow or future-surface targets.
Support: 2406-11520v3; direct S&P 500 intraday evidence, indirectly relevant to ECON499 forecasting.

BATCH_1_CODE_AUDIT_CHECKLIST

Confirm the official sample window is encoded as 2004-01-02 through 2021-04-09.

Confirm the prediction timestamp is encoded as 15:45 America/New_York and propagated into ingestion, cleaning, feature, surface, and evaluation stages.

Verify every input field used for a same-day forecast has an availability timestamp less than or equal to 15:45.

Verify same-day EOD settlements, closing quotes, closing underlier values, or post-15:45 fields cannot enter features for the 15:45 forecast.

Verify quote timestamps survive raw ingestion through Parquet/Arrow artifacts.

Verify raw data are immutable and that no cleaning stage overwrites raw artifacts.

Verify option filters are explicit, parameterized, and tested: DTE, price, bid/ask validity, spread, moneyness, option type, liquidity/volume, duplicates, no-arbitrage violations, and IV inversion failures.

Verify each filter emits row counts by date and reason code.

Verify there are no silent row drops, silent date coercions, silent type coercions, or unlogged parsing failures.

Verify call/put handling is explicit: OTM-only, both calls and puts, parity-based selection, or aggregation.

Verify bid, ask, mid, and trade/quote price construction are explicit and timestamped.

Verify forward-price construction is documented and timestamp-safe.

Verify risk-free-rate and dividend inputs are documented, versioned, and timestamp-safe.

Verify put-call parity matching, if used, enforces timestamp/date consistency and logs unmatched pairs.

Verify IV inversion method is specified and failures are logged.

Verify surface coordinate metadata states the coordinate system and all transformations.

Verify DTE and year-fraction conventions are documented and tested.

Verify near-expiry and long-expiry handling is explicit.

Verify grid coordinates are versioned and reproducible.

Verify raw observed options remain recoverable after surface construction.

Verify observed grid cells are persisted separately from fitted/completed values.

Verify every completed-grid cell has an observed-cell mask.

Verify richer provenance flags exist or can be derived: observed, interpolated, extrapolated, missing, generated-only.

Verify interpolation/smoothing method, bandwidths, hyperparameters, domain bounds, minimum-observation rules, and extrapolation rules are serialized.

Verify sparse dates or sparse slices cannot silently produce unstable surfaces without warnings or failure modes.

Verify interpolation does not extrapolate outside supported moneyness/maturity regions unless explicitly flagged.

Verify surface-construction validation, if present, uses held-out observed options/cells.

Verify cross-sectional holdout validation is not used as a substitute for blocked time-series validation of forecasts.

Verify interpolation/smoothing sensitivity reports compare methods, bandwidths, or domain choices, or at minimum flag that robustness has not been run.

Verify observed-cell metrics and completed-grid metrics are computed separately.

Verify interpolation-only and extrapolation-cell metrics are distinguishable where possible.

Verify evaluation reports coverage counts by moneyness and maturity slice.

Verify evaluation reports forecast errors by moneyness and maturity slice.

Verify aggregate metrics cannot be produced without accompanying observed-mask coverage metadata.

Verify no-change benchmark implementation states its coordinate convention.

Verify factor/PCA/HAR benchmarks state whether they use IV levels, log-IV levels, IV changes, log-IV changes, total variance, or total-variance changes.

Verify PCA/factor bases, scalers, imputers, bandwidth choices used as features, and model preprocessing are fit inside each training window only.

Verify split manifests serialize train, validation, test, and forecast date ranges.

Verify no random CV is used for HPO in forecasting models.

Verify any random validation used for smoothing-only tasks is isolated from forecasting HPO and documented as cross-sectional construction validation.

Verify forecast targets are aligned to the correct horizon and same grid/cell identity.

Verify completed-grid forecast targets are not described as raw observed market truth.

Verify neural surface documentation says “arbitrage-aware” unless hard no-arbitrage constraints are proven elsewhere.

Verify arbitrage penalties and diagnostics, if used, are stored with weights, formulas, grid/domain, and before/after violation summaries.

Verify report text does not cite daily EOD literature as evidence that same-day EOD features are available at 15:45.

MINIMAL_REPO_FILES_FOR_BATCH_1_CODE_AUDIT
File path	Why needed
src/ivsurf/config.py	Verify official sample window, 15:45 timestamp, grid defaults, artifact roots, and global project constants.
src/ivsurf/schemas.py	Verify raw/clean/surface schema strictness, timestamp fields, mask fields, and prevention of silent coercion.
src/ivsurf/calendar.py	Verify trading-session logic, DTE/year-fraction convention, early-close handling, and target-date alignment.
src/ivsurf/io/ingest_cboe.py	Audit raw Cboe ingestion, quote timestamp handling, raw immutability, type parsing, and row-drop behavior.
src/ivsurf/io/parquet.py	Verify Arrow/Parquet schema preservation, partitioning, and lazy dataset reads/writes.
src/ivsurf/qc/timing_checks.py	Verify field availability checks and same-day EOD leakage prevention.
src/ivsurf/qc/schema_checks.py	Verify no silent type/date coercions and required-column enforcement.
src/ivsurf/qc/sample_window.py	Verify enforcement of the official 2004-01-02 to 2021-04-09 window.
src/ivsurf/cleaning/option_filters.py	Audit explicit option filter/drop rules, logging, and testability.
src/ivsurf/cleaning/derived_fields.py	Audit moneyness, forward, rate/dividend, DTE, IV, total variance, and timestamp-safe derived fields.
src/ivsurf/surfaces/grid.py	Audit grid design, coordinate metadata, maturity/moneyness axes, and versioning.
src/ivsurf/surfaces/aggregation.py	Audit observed quote-to-cell aggregation and key/cardinality assumptions.
src/ivsurf/surfaces/interpolation.py	Audit completed-grid construction, smoothing/interpolation parameters, extrapolation flags, and fit scope.
src/ivsurf/surfaces/masks.py	Verify observed-cell mask creation, persistence, and semantics.
src/ivsurf/surfaces/arbitrage_diagnostics.py	Batch 1 only needs language/diagnostic sanity; full static-arbitrage audit is deferred to Batch 2.
src/ivsurf/features/lagged_surface.py	Verify lagged surfaces do not use same-day post-15:45 or future completed-grid information.
src/ivsurf/features/factors.py	Verify PCA/factor construction scope, coordinate basis, and training-window-only fitting.
src/ivsurf/features/tabular_dataset.py	Audit feature/target alignment, mask propagation, and prevention of completed-grid leakage.
src/ivsurf/models/naive.py	Verify no-change benchmark and coordinate convention.
src/ivsurf/models/har_factor.py	Verify factor/HAR benchmark target, input basis, and split-safe fitting.
src/ivsurf/models/neural_surface.py	Check only Batch 1-relevant claims: target representation, masks, output grid, and “arbitrage-aware” wording.
src/ivsurf/models/losses.py	Verify observed-mask-aware losses and separation of IV/variance/price losses.
src/ivsurf/models/penalties.py	Check whether penalties are soft versus hard constraints; full theory deferred to Batch 2.
src/ivsurf/splits/walkforward.py	Verify chronological split generation needed for factor/smoothing/forecast leakage checks.
src/ivsurf/splits/manifests.py	Verify serialized split manifests include dates, versions, and preprocessing scope.
src/ivsurf/evaluation/alignment.py	Verify forecast/target/cell/date alignment.
src/ivsurf/evaluation/metrics.py	Verify separate observed-cell and completed-grid metrics.
src/ivsurf/evaluation/interpolation_sensitivity.py	Verify construction-method or parameter robustness support.
src/ivsurf/evaluation/forecast_store.py	Verify forecast artifact schema includes model ID, split ID, grid version, target version, and mask/provenance metadata.
scripts/01_ingest_cboe.py	Verify orchestration starts with timestamp-safe raw ingestion and does not bypass checks.
scripts/02_qc_raw.py or nearest equivalent	Verify raw/timing/schema/sample-window checks are actually run.
scripts/03_build_surfaces.py or nearest equivalent	Verify surface construction uses the audited grid, masks, and interpolation stages.
scripts/04_build_features.py or nearest equivalent	Verify feature generation uses lagged, timestamp-safe, mask-aware surfaces.
scripts/07_evaluate.py or nearest equivalent	Verify observed-vs-completed and slice metrics are produced.
Relevant tests under tests/ for the files above	Verify the standards are enforced by automated unit/integration/regression tests, not only by comments.
Relevant docs/configs under configs/, docs/, or research/consults/gpt55-pro-audit/	Verify documented coordinate conventions, data availability assumptions, surface versions, and audit decisions match code.
OPEN_LITERATURE_GAPS

Formal static-arbitrage conditions and hard guarantees.
Batch 1 supports caution and soft-penalty language, but does not settle full no-arbitrage diagnostics, SVI/eSSVI conditions, call-price convexity, calendar monotonicity, or when “arbitrage-free” is justified. This is Batch 2.

Total variance as the neural target.
Batch 1 mentions total-volatility/variance conditions through 2406-11520v3, but the project’s requirement that the neural model predicts total variance needs Batch 2 grounding.

Forecast-model benchmark completeness.
Batch 1 supports no-change and factor-style benchmarks, but does not fully settle ridge, elastic net, HAR/factor, neural, LightGBM/random forest, or multistep benchmark design. This is Batch 3.

Blocked walk-forward validation and HPO.
Batch 1 implies temporal discipline but does not provide the full standard for blocked time-series validation, HPO, embargoes, nested validation, or split-manifest design. This is Batch 4.

Statistical forecast comparison.
Batch 1 supports aligned loss panels and DM/Newey-West awareness through ebsco-fulltext-04-08-2026, but does not settle MCS, SPA, bootstrap design, or proxy-target reliability. This is Batch 5.

Hedging and economic-value evaluation.
Batch 1 includes some trading/economic-value material from ebsco-fulltext-04-08-2026, but exact hedging PnL standards, revaluation timing, transaction costs, and hedge instrument availability remain deferred to the hedging literature batch.

Exact 15:45 market-microstructure standard.
2406-11520v3 provides intraday S&P 500 option evidence, but Batch 1 does not settle all Cboe 15:45-specific rules such as quote consolidation, NBBO selection, early close behavior, stale quote filtering, or auction/settlement distinctions. These may need repository artifacts first; additional literature should be searched only if the code claims a microstructure-specific method not covered by the local corpus.

FORMAL_FINDINGS_IF_ANY

none

Extended Pro
ChatGPT is AI and can make mistakes. Check important info.
