# Round 002D Pro Response: Batch 1 Paper 4

Captured: 2026-04-27

## READING_COVERAGE

* Paper: `s11147-020-09166-0` — “Option-implied information: What’s the vol surface got to do with it?”
* Sections/tables/figures/equations read:

  * Abstract and introduction.
  * Surface-construction motivation and why option-implied information depends on surface construction.
  * Section 3.1, one-dimensional kernel regression with tail extrapolation, including equations for kernel regression, moneyness `m = K/F`, log-maturity interpolation, one-dimensional maturity-specific smoothing, and tail extrapolation.
  * Section 4, “On the uniqueness of option-implied information.”
  * Section 5, “Assessing the accuracy of a volatility surface,” including leave-one-out cross-validation, RMSE, MAE, and smoothness definitions.
  * Section 6.1, end-of-day findings.
  * Section 6.2, intraday/data-poor findings.
  * Section 7, conclusion.
  * Appendix A, dividend-yield construction by put-call parity.
  * Appendix B, alternative volatility-surface methods: Gram–Charlier, spline, and OptionMetrics-style three-dimensional kernel.
  * Appendix C, variance-risk-premium construction.
  * Appendix D, MSE/noise/bias/smoothness discussion.
  * Tables 1–10 and Figures 1–4 at interpretation level.
* Extracted text vs PDF:

  * Extracted `paper.md` was sufficient for the main prose, formulas, method descriptions, and conclusions.
  * PDF/layout text was consulted earlier for title metadata, table/equation placement, and ambiguous table extraction.
* Extraction caveats:

  * Table numbering in the extracted text is noisy because the extraction inventory’s table labels do not always match the paper’s printed table captions.
  * Large numeric tables are readable for qualitative conclusions, but exact numeric reproduction should be checked against the original PDF if later used in a report.
  * Figure-dependent visual details were read from captions/prose; no further figure-level visual interpretation is used here.
* Important section deferred:

  * Exact cell-by-cell verification of all cross-validation and option-implied-measure tables is deferred because this round establishes audit standards, not numerical replication.

## PAPER_SPECIFIC_EVIDENCE

* `s11147-020-09166-0`, Abstract: option-implied variance, skewness, and variance risk premium are sensitive to volatility-surface construction; risk-neutral variance differs by more than 10% on average and VRP estimates differ by about 60% on average across methods.
  Evidence type: direct S&P 500 plus Euro Stoxx 50 evidence.
  ECON499 implication: completed-surface construction is a first-order research design choice, not neutral preprocessing.

* `s11147-020-09166-0`, Abstract and Section 7: the paper compares state-of-the-art parametric, semi-parametric, and non-parametric surfaces using 14 years of end-of-day and intraday S&P 500 and Euro Stoxx 50 option data.
  Evidence type: direct S&P 500 evidence.
  ECON499 implication: ECON499 should benchmark or sensitivity-test its interpolation method rather than relying on one unvalidated completed grid.

* `s11147-020-09166-0`, Table 1 / data summary: the S&P 500 end-of-day sample contains over 9 million price observations, while the intraday S&P 500 sample contains over 43 million trade observations; EOD and intraday are treated as distinct data environments.
  Evidence type: direct S&P 500 evidence.
  ECON499 implication: 15:45 forecasting must distinguish intraday-available information from end-of-day information.

* `s11147-020-09166-0`, Table 1 / data summary: maturity buckets are short `<30 days`, medium `30–365 days`, and long `>365 days`; moneyness uses `m = K/F` with left tail `<0.95`, ATM `0.95–1.05`, and right tail `>1.05`.
  Evidence type: direct S&P 500 plus Euro Stoxx 50 evidence.
  ECON499 implication: evaluation should report surface quality and forecast error by maturity and moneyness region.

* `s11147-020-09166-0`, Section 3.1, equations (1)–(4): the proposed method reduces the Aït-Sahalia–Lo kernel regression to a one-dimensional moneyness smoother by defining `m = K/F` and fitting separately by maturity.
  Evidence type: broader option-methodology evidence with direct S&P 500 evaluation.
  ECON499 implication: moneyness definition, forward construction, maturity-specific fitting, and bandwidth rules must be serialized and testable.

* `s11147-020-09166-0`, Section 3.1: the paper recommends linear interpolation along log maturity when a target maturity is not directly observed.
  Evidence type: broader option-methodology evidence.
  ECON499 implication: any maturity interpolation rule must be explicit, versioned, and flagged separately from observed maturities.

* `s11147-020-09166-0`, Section 3.1 and Figure 1: tail extrapolation creates artificial implied-volatility points down to moneyness `0.4` and up to `1.6` before kernel smoothing.
  Evidence type: broader option-methodology evidence.
  ECON499 implication: ECON499 must distinguish observed cells from artificial extrapolation support and completed-grid values.

* `s11147-020-09166-0`, Section 3.1 and Table 2: dropping the maturity dimension and adding tail extrapolation substantially improve cross-validation accuracy; no-arbitrage enforcement does not materially worsen accuracy in the reported refinement table.
  Evidence type: direct S&P 500 plus Euro Stoxx 50 evidence.
  ECON499 implication: interpolation design choices should be validated empirically, and arbitrage post-processing should be checked for accuracy impact.

* `s11147-020-09166-0`, Section 4 and Table 3: VRP estimates differ materially across surface-construction methods even though the same input data and integration scheme are used.
  Evidence type: direct S&P 500 plus Euro Stoxx 50 evidence.
  ECON499 implication: downstream claims based on completed surfaces must include construction-method robustness.

* `s11147-020-09166-0`, Section 4 and Table 4: risk-neutral skewness estimates are statistically different across volatility surfaces and can even imply different changes across subperiods.
  Evidence type: direct S&P 500 plus Euro Stoxx 50 evidence.
  ECON499 implication: surface construction can change qualitative conclusions, so completed-grid forecast targets require caution.

* `s11147-020-09166-0`, Section 5: leave-one-out validation evaluates each method on option observations not used to construct the fitted surface.
  Evidence type: broader option-methodology evidence.
  ECON499 implication: surface construction should be validated against held-out observed quotes/cells, not only against the completed grid it generated.

* `s11147-020-09166-0`, Section 5, equations (5)–(7): the paper uses RMSE, MAE, and a smoothness measure based on second derivatives along moneyness.
  Evidence type: broader option-methodology evidence.
  ECON499 implication: ECON499 should report both accuracy and smoothness/interpolation diagnostics for completed surfaces.

* `s11147-020-09166-0`, Section 6.1 and Table 5/Table 9: in the end-of-day S&P 500 setting, the one-dimensional kernel surface is generally most accurate; short-maturity and left-tail regions are especially difficult.
  Evidence type: direct S&P 500 evidence.
  ECON499 implication: sparse/tail/short-dated regions need separate coverage and error diagnostics.

* `s11147-020-09166-0`, Section 6.2 and Tables 7–8/Table 10: in the intraday data-poor setting, spline interpolation can overfit with few observations, while the one-dimensional kernel remains strong overall.
  Evidence type: direct S&P 500 plus Euro Stoxx 50 evidence.
  ECON499 implication: a 15:45 intraday grid must test robustness to sparse same-time support and avoid unstable interpolation in thin regions.

* `s11147-020-09166-0`, Appendix A: dividend yields are estimated from put-call parity, with intraday matching requiring same-day put/call pairs and a small underlying-price-change constraint.
  Evidence type: broader option-methodology evidence with direct index-option use.
  ECON499 implication: forward, dividend, rate, and put-call matching inputs must be timestamp-safe and explicitly validated.

* `s11147-020-09166-0`, Appendix B.3: OptionMetrics-style three-dimensional kernel smoothing uses delta, log maturity, put/call flag, vega weights, and a delta range; the paper notes the OptionMetrics surface is not necessarily arbitrage-free before post-processing.
  Evidence type: broader option-methodology evidence.
  ECON499 implication: documentation must not equate common vendor-style smoothing with arbitrage-free construction unless post-processing or constraints prove it.

## AUDIT_STANDARD_CONTRIBUTION

* Surface-construction sensitivity:

  * Strong contribution. The paper directly shows that option-implied variance, skewness, and VRP can change materially across surface-construction methods.
  * ECON499 standard: completed-grid construction must be treated as a model component with versioned method choice, parameters, validation, and robustness checks.

* Completed-grid target reliability:

  * Strong contribution. The paper shows completed/interpolated surfaces can induce economically meaningful bias in downstream measures.
  * ECON499 standard: completed-grid targets must not be treated as raw market truth; reports must disclose construction method, observed support, interpolation/extrapolation status, and sensitivity.

* Observed-vs-completed evaluation:

  * Strong contribution through leave-one-out validation on observed option prices.
  * ECON499 standard: observed-cell or held-out-observed evaluation must be reported separately from completed-grid evaluation.

* Risk-neutral moment or downstream-measure robustness:

  * Strong contribution. The paper directly studies model-free implied variance, skewness, and VRP sensitivity to volatility-surface construction.
  * ECON499 standard: any downstream feature, target, or diagnostic derived from an integrated/completed surface requires robustness checks across construction assumptions.

* Interpolation/extrapolation caution:

  * Strong contribution. Tail extrapolation and sparse intraday interpolation materially affect results; spline overfitting appears in data-poor intraday settings.
  * ECON499 standard: interpolated and extrapolated cells must be flagged separately; sparse-region and tail diagnostics are mandatory.

* Documentation standards:

  * Strong contribution. The paper’s method descriptions show that moneyness, maturity, bandwidth, tail extrapolation, no-arbitrage post-processing, and input-price definitions affect results.
  * ECON499 standard: surface artifacts must store coordinate system, forward/rate/dividend assumptions, smoothing parameters, extrapolation rules, no-arbitrage processing, and construction timestamp.

* Limitations for 15:45 ECON499 forecasting:

  * The paper contains intraday evidence, but it does not define ECON499’s exact 15:45 prediction problem.
  * ECON499 standard: this paper supports intraday/data-poor caution, but it cannot justify same-day EOD leakage or any feature unavailable at 15:45.

## REPO_CHECKS_DERIVED_FROM_THIS_PAPER

* Verify completed-grid construction is represented as a versioned model/artifact, not as raw observed data.

* Verify every completed-grid cell has a provenance flag: observed, interpolated, extrapolated, artificial-support-driven, or missing.

* Verify observed-cell masks are persisted and used in forecast evaluation.

* Verify observed-cell metrics and completed-grid metrics are reported separately.

* Verify interpolation/extrapolation rules are serialized in surface metadata.

* Verify moneyness is explicitly defined, especially whether it is `K/F`, `K/S`, log moneyness, normalized moneyness, delta, or strike.

* Verify forward-price construction is documented and timestamp-safe.

* Verify dividend-yield estimation, if used, is documented, reproducible, and timestamp-safe.

* Verify risk-free-rate inputs are documented and timestamp-safe.

* Verify put-call matching logic, if used, enforces same timestamp or documented intraday tolerance.

* Verify maturity interpolation rules are explicit, especially any interpolation along log maturity.

* Verify tail extrapolation is explicitly flagged and bounded.

* Verify surface construction does not silently extrapolate beyond observed support.

* Verify bandwidths or smoothing hyperparameters are stored with each surface artifact.

* Verify any smoothing hyperparameter selection uses only permitted data for the relevant task.

* Verify no-arbitrage post-processing, if applied, records before/after diagnostics and does not silently overwrite the original fitted surface.

* Verify documentation does not call a surface arbitrage-free merely because it uses OptionMetrics-style or kernel smoothing.

* Verify surface-construction validation uses held-out observed options/cells where possible.

* Verify cross-sectional leave-one-out validation is not substituted for blocked time-series validation of forecasts.

* Verify reports include error and coverage by moneyness/maturity buckets, including short-dated and tail regions.

* Verify sparse intraday regions have diagnostics for number of strikes, number of maturities, and interpolation stability.

* Verify spline or high-flexibility interpolation cannot fit with too few observations without an explicit minimum-observation rule.

* Verify downstream measures derived from surfaces, including variance, skewness, or variance-risk-premium-like features, identify the surface version used.

* Verify forecast claims based on completed-grid targets include sensitivity to construction method or smoothing parameters.

* Verify 15:45 forecasting inputs do not use end-of-day settlements, closing quotes, or post-15:45 information unless explicitly excluded from predictors and used only for post-hoc target construction.

## FORMAL_FINDINGS_IF_ANY

none
