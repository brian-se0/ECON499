# Round 002C Pro Response: Batch 1 Paper 3

Captured: 2026-04-27

## READING_COVERAGE

* Paper: `s11147-023-09195-5` — “Implied volatility surfaces: a comprehensive analysis using half a billion option prices.”
* Sections/equations/tables/figures read:

  * Abstract and Introduction.
  * Section 2, “Constructing single stock volatility surfaces.”
  * Section 2.1, one-dimensional kernel smoother, including equations (1), (3), (4), (5), and (6).
  * Section 2.2, OptionMetrics three-dimensional kernel smoother, including equations (7) and (8).
  * Section 2.3, spline method, including equations (9), (10), and (11).
  * Section 3, Validation, including leave-one-out procedure and RMSE/MAE definitions in equations (12) and (13).
  * Table 1, data summary by S&P 500 versus single-stock options, maturity, and moneyness.
  * Section 4.1, option prices.
  * Section 4.2, risk-free rate, dividend yields, and implied volatility.
  * Section 5, Findings.
  * Section 5.1, S&P 500 results.
  * Section 5.2, single-firm results.
  * Section 5.4, lessons from the tails.
  * Section 5.5, implications for option-implied information.
  * Tables 2-4 and Figures 1-8 at interpretation level.
  * Section 6, Conclusion.
* Extracted text vs PDF:

  * Extracted `paper.md` was sufficient for prose, equations, method descriptions, and conclusions.
  * PDF pages were consulted earlier for layout-sensitive method/table interpretation.
* Extraction caveats:

  * Table extraction is noisy for large numeric tables, especially Tables 3-4. The qualitative ranking and key reported errors were readable, but exact numeric cells should be rechecked against the PDF before any numeric reproduction.
  * Figure-dependent distributional details were read from captions/prose; exact visual boxplot interpretation should be treated as a caveat.
* Important section deferred:

  * No major Batch 1 section deferred. Exact table-cell reproduction is deferred because this round sets audit standards rather than reproducing the paper’s numerical tables.

## PAPER_SPECIFIC_EVIDENCE

* `s11147-023-09195-5`, Abstract and Introduction: the paper evaluates volatility-surface construction methods using daily option-price observations for 499 S&P 500 constituents and the S&P 500 index.
  Evidence type: broader US-option evidence plus direct S&P 500 index evidence.
  ECON499 implication: surface construction is itself a model choice and must be audited, not treated as a neutral preprocessing detail.

* `s11147-023-09195-5`, Introduction and Section 6: the one-dimensional kernel smoother is reported as more accurate and less noisy than the OptionMetrics three-dimensional kernel smoother and Figlewski-style spline across tested moneyness, maturity, and liquidity categories.
  Evidence type: broad US-option evidence, with direct S&P 500 results in Section 5.1.
  ECON499 implication: interpolation/smoothing method choice must be justified and sensitivity-tested.

* `s11147-023-09195-5`, Section 2.1, equation (1): moneyness is defined as `m = K / F`, where `K` is strike and `F` is the forward price.
  Evidence type: general method applied to S&P 500 and single-stock options.
  ECON499 implication: ECON499 must explicitly document whether it uses forward moneyness, spot moneyness, log-forward moneyness, delta, or another coordinate.

* `s11147-023-09195-5`, Section 2.1, equation (3): normalized moneyness is defined using `ln(m)` scaled by maturity and ATM implied volatility.
  Evidence type: general IVS methodology.
  ECON499 implication: if normalized/log-forward coordinates are used, ATM-IV, maturity, and forward inputs must have timestamp-safe availability.

* `s11147-023-09195-5`, Section 2.1, equations (4)-(6): the one-dimensional kernel smoother is fitted separately by maturity and uses a maturity-specific bandwidth.
  Evidence type: general method with direct S&P 500 evaluation.
  ECON499 implication: smoothing parameters and per-maturity fitting choices must be serialized in surface metadata.

* `s11147-023-09195-5`, Section 2.2: the OptionMetrics-style three-dimensional kernel uses delta, maturity, put/call flag, vega weighting, and input filters such as vega at least 0.5.
  Evidence type: general methodology.
  ECON499 implication: if the repo uses delta-space or OptionMetrics-style smoothing, it must record delta, vega, option-type, and maturity definitions and filters.

* `s11147-023-09195-5`, Section 2.3: the spline method is fitted to non-in-the-money observations and uses adaptive polynomial order depending on available observations.
  Evidence type: general methodology.
  ECON499 implication: spline/grid construction must expose minimum-observation rules and must not silently fit unstable surfaces in sparse cells.

* `s11147-023-09195-5`, Section 3, Validation: the paper uses leave-one-out cross-validation within a day’s option panel, predicting a left-out implied volatility from the remaining same-day observations.
  Evidence type: surface-construction methodology, not time-series forecasting validation.
  ECON499 implication: same-day cross-sectional holdout can validate interpolation quality, but it cannot replace blocked time-series validation for forecast models.

* `s11147-023-09195-5`, Section 3: the paper separately evaluates “interpolation and extrapolation” versus “interpolation only” by excluding leftmost and rightmost points.
  Evidence type: general methodology with direct S&P 500 results.
  ECON499 implication: ECON499 should flag and report extrapolated cells separately from interpolated or observed cells.

* `s11147-023-09195-5`, equations (12)-(13): accuracy is measured with RMSE and MAE in annualized implied-volatility units over time, underlyings, maturities, and observations.
  Evidence type: general evaluation design.
  ECON499 implication: evaluation metrics must state target units and aggregation axes.

* `s11147-023-09195-5`, Table 1 and Section 4.1: the S&P 500 index sample contains 14,267,068 option prices from January 2004 to July 2019; the broader single-stock sample contains 373,825,605 prices.
  Evidence type: direct S&P 500 plus broad US-option evidence.
  ECON499 implication: ECON499’s 2004-2021 SPX window is in the same broad era but must document its own sample coverage, missing days, and quote counts.

* `s11147-023-09195-5`, Table 1 and Section 4.1: maturities are bucketed as short 1-29 days, medium 30-365 days, and long over 365 days; moneyness buckets use `m = K/F` with left below 0.95, center 0.95-1.05, and right above 1.05.
  Evidence type: direct S&P 500 plus broad US-option evidence.
  ECON499 implication: reporting by maturity and moneyness slices is necessary, not optional.

* `s11147-023-09195-5`, Section 4.1: data are daily quote data and mid prices; duplicates, basic no-arbitrage violations, and non-convergent implied volatilities are removed.
  Evidence type: direct S&P 500 plus broad US-option evidence.
  ECON499 implication: ECON499 must log duplicate removal, no-arbitrage filters, IV inversion failures, and bid/mid/ask price choice.

* `s11147-023-09195-5`, Section 4.2: risk-free rates are extracted from index options using put-call parity, dividend yields are estimated from ATM call-put pairs, and implied volatilities are computed using a pricing model for early-exercise features.
  Evidence type: general data-construction methodology.
  ECON499 implication: forward/rate/dividend/IV-inversion assumptions must be documented and timestamp-safe.

* `s11147-023-09195-5`, Section 5.1 and Table 2: for S&P 500 index options, the one-dimensional kernel smoother produces small errors across moneyness/maturity buckets, while the three-dimensional kernel and spline show large errors in short-term tail regions.
  Evidence type: direct S&P 500 evidence.
  ECON499 implication: completed-grid errors should be reported by region, especially short maturity and OTM/tail buckets.

* `s11147-023-09195-5`, Section 5.2 and Figure 2: lower trading volume is associated with larger surface-construction error for single-stock options.
  Evidence type: broader US-option evidence.
  ECON499 implication: ECON499 should report coverage/liquidity diagnostics even if the target is SPX, because sparse support still affects surface quality.

* `s11147-023-09195-5`, Section 5.4 and Table 4: tail error statistics such as 95% absolute-error quantiles and maximum errors reveal large differences across smoothing methods.
  Evidence type: broad US-option evidence, with S&P 500 evidence in related tables.
  ECON499 implication: mean RMSE/MAE is insufficient; robustness reports should include tail-error diagnostics.

* `s11147-023-09195-5`, Section 5.5 and Section 6: noisy surface construction can distort option-implied moments and downstream economic inference.
  Evidence type: broader US-option evidence.
  ECON499 implication: forecast-performance claims based on completed surfaces must acknowledge interpolation/smoothing error as part of the target-construction uncertainty.

## AUDIT_STANDARD_CONTRIBUTION

* Option filtering and liquidity:

  * Supports explicit duplicate removal, no-arbitrage filtering, IV-convergence filtering, mid-price choice, and liquidity/volume breakdowns.
  * ECON499 standard: every filter must be explicit, logged by date and reason, and testable.
  * Support type: direct S&P 500 plus broader US-option evidence.

* Moneyness/maturity coordinate choices:

  * Supports forward moneyness `K/F`, normalized log-forward moneyness, and maturity/moneyness bucket reporting.
  * ECON499 standard: coordinate system, forward construction, ATM-IV use, and maturity convention must be serialized in metadata.
  * Support type: general IVS methodology with direct S&P 500 evaluation.

* Observed surface construction:

  * Supports starting from observed daily quote/mid-price implied volatilities and constructing surfaces from irregular option panels.
  * ECON499 standard: raw observed options and observed grid cells must remain distinct from fitted, smoothed, or extrapolated values.
  * Support type: direct S&P 500 plus broader US-option evidence.

* Completed-grid interpolation or smoothing:

  * Supports comparing alternative smoothing methods and treating construction quality as empirically testable.
  * ECON499 standard: interpolation method, hyperparameters, bandwidths, minimum-observation rules, and extrapolation rules must be versioned and reproducible.
  * Support type: direct S&P 500 plus broader US-option evidence.

* Observed-vs-completed evaluation:

  * The paper validates fitted values by holding out observed options; it does not define ECON499-style observed-cell masks.
  * ECON499 standard: observed-cell loss and completed-grid loss must be reported separately, because completed values are model-implied outputs.
  * Support type: paper directly supports holdout-observed evaluation; mask separation is ECON499 inference.

* Interpolation sensitivity and robustness:

  * Strongly supports sensitivity analysis across construction methods and interpolation versus extrapolation.
  * ECON499 standard: completed-grid results should include interpolation-method robustness and flags for extrapolated regions.
  * Support type: direct S&P 500 plus broader US-option evidence.

* Sparse-region diagnostics:

  * Supports moneyness, maturity, volume, tail-error, and worst-case/quantile diagnostics.
  * ECON499 standard: reports must include coverage counts and errors by moneyness/maturity slice, especially short maturity and tail cells.
  * Support type: direct S&P 500 plus broader US-option evidence.

* Limitations for 15:45 ECON499 forecasting:

  * The paper uses daily quote/end-of-day-style data and does not establish intraday 15:45 availability.
  * ECON499 standard: this paper cannot justify same-day EOD fields in a 15:45 forecast; all forward, rate, dividend, price, quote, and IV inputs need availability timestamps.
  * Support type: limitation; 15:45 constraint is ECON499-specific.

## REPO_CHECKS_DERIVED_FROM_THIS_PAPER

* Verify option-price source fields specify bid, ask, mid, trade, or quote-derived price.

* Verify mid-price construction, if used, is explicit and timestamped.

* Verify duplicate-removal rules are explicit, counted, logged, and tested.

* Verify no-arbitrage filter rules are explicit, counted, logged, and tested.

* Verify IV non-convergence cases are logged with reason codes, not silently dropped.

* Verify option universe includes documented call/put handling: OTM-only, both sides, parity-based, or aggregated.

* Verify moneyness is explicitly defined, including whether it is `K/F`, `K/S`, log moneyness, normalized moneyness, delta, or strike.

* Verify forward-price inputs, rate inputs, dividend inputs, ATM-IV inputs, and spot/index inputs have availability timestamps no later than 15:45 when used for forecasting.

* Verify maturity buckets and DTE/year-fraction conventions are explicit and reproducible.

* Verify surface metadata stores grid coordinates, smoothing method, bandwidths, minimum-observation rules, extrapolation rules, and version identifiers.

* Verify interpolation/smoothing does not silently extrapolate beyond observed support; extrapolated cells should be flagged.

* Verify completed-grid artifacts preserve observed-cell masks and distinguish observed, interpolated, and extrapolated cells.

* Verify cross-sectional interpolation validation, if present, holds out observed cells/contracts rather than validating on fitted values used in construction.

* Verify cross-sectional leave-one-out validation is not used as a substitute for blocked time-series model validation or HPO.

* Verify forecast evaluation reports observed-cell metrics separately from completed-grid metrics.

* Verify reports include moneyness-by-maturity slice metrics and coverage counts.

* Verify reports include sparse-region diagnostics, especially short-dated and tail moneyness cells.

* Verify interpolation sensitivity compares at least one alternative construction method or reports robustness to smoothing parameters.

* Verify tail-error diagnostics such as 95% absolute error quantiles or maximum errors are available for surface-construction validation.

* Verify downstream model comparisons do not treat completed-grid targets as if they were raw observed market data.

* Verify documentation does not cite this paper as support for 15:45 availability or same-day EOD leakage tolerance.

## FORMAL_FINDINGS_IF_ANY

none
