# Round 002B Pro Response: Batch 1 Paper 2

Captured: 2026-04-27

## READING_COVERAGE

* Paper read: `dynamics-of-implied-volatility-surfaces` — “Dynamics of implied volatility surfaces.”
* Sections/equations/tables/figures read:

  * Abstract and Introduction.
  * Section 2.1, “Definitions and notations,” including equations defining implied volatility surface and moneyness.
  * Section 2.4, deterministic/sticky-moneyness and sticky-strike rules.
  * Section 3.1, “Data sets.”
  * Section 3.2, “Construction of smooth volatility surfaces,” including Nadaraya–Watson smoothing equation.
  * Sections 3.3–3.4, Karhunen–Loève/PCA construction and numerical implementation.
  * Section 4, SP500 empirical results.
  * Figures 1–8, 10–14 at interpretation level; figure-dependent exact shape details remain an extraction caveat.
  * Table 1, SP500 principal-component summary statistics.
  * Section 6.1, empirical observations summary.
  * Section 6.2, mean-reverting factor model, including equations for factor representation and AR(1)/OU dynamics.
  * Section 7, applications/limitations at a high level.
* Extracted text vs PDF:

  * Extracted `paper.md` was sufficient for most prose, definitions, equations, and section structure.
  * PDF/layout images were consulted for first pages, figures/tables, and to verify the paper is daily end-of-day surface/factor evidence rather than real-time 15:45 evidence.
* Extraction caveats:

  * Figure extraction is incomplete/noisy; figure-dependent factor-shape claims are based on prose plus figure captions, not full independent visual validation of every figure.
  * Table extraction is noisy around figure/table placement. The exact numeric variance share of the first SP500 factor appears inconsistent between caption/prose/table extraction; the robust conclusion is that a small number of factors dominate daily SP500 IVS variation.
* Important section deferred:

  * Detailed hedging/scenario-generation implications in Section 7 are deferred because this round concerns surface construction and forecasting-audit standards, not hedging evaluation.

## PAPER_SPECIFIC_EVIDENCE

* `dynamics-of-implied-volatility-surfaces`, Abstract and Introduction: the paper studies SP500 and FTSE index-option implied-volatility surfaces and models their daily deformation with a small number of orthogonal random factors.
  Direct SPX/SP500 evidence: yes.
  ECON499 implication: factor/PCA/HAR-style benchmarks are academically relevant for SPX IVS forecasting.

* `dynamics-of-implied-volatility-surfaces`, Section 2.1, Definitions and notations: the implied-volatility surface is represented as a function of moneyness and time to maturity, with moneyness defined as `m = K / S(t)` and surface notation `I_t(m, τ)`.
  Direct SPX/SP500 evidence: general IVS definition applied later to SP500.
  ECON499 implication: surface metadata must explicitly state whether coordinates are spot moneyness, forward moneyness, log moneyness, or another convention.

* `dynamics-of-implied-volatility-surfaces`, Section 2.4: sticky moneyness assumes the surface is unchanged in relative coordinates from day to day, while sticky strike assumes unchanged implied volatility in absolute strike/maturity coordinates.
  Direct SPX/SP500 evidence: general methodology with SP500 motivation.
  ECON499 implication: no-change/sticky-surface benchmarks should be coordinate-explicit; “no change” is not uniquely defined without a coordinate system.

* `dynamics-of-implied-volatility-surfaces`, Section 3.1, Data sets: the SP500 sample uses end-of-day prices of European-style calls and puts on the SP500 index, traded on CBOE, from 2 March 2000 to 2 February 2001.
  Direct SPX/SP500 evidence: yes.
  ECON499 implication: this paper supports daily end-of-day IVS dynamics, but not same-day EOD data use for a 15:45 forecasting problem.

* `dynamics-of-implied-volatility-surfaces`, Section 3.1: the paper observes daily implied volatilities for traded options, typically around 100 options per day.
  Direct SPX/SP500 evidence: yes.
  ECON499 implication: raw observed option support is sparse/irregular and should not be treated as a naturally complete grid.

* `dynamics-of-implied-volatility-surfaces`, Section 3.1 and Figure 3: observations are irregular and time-varying across moneyness and maturity; traded strikes decline away from the money and as maturity increases.
  Direct SPX/SP500 evidence: yes.
  ECON499 implication: observed-cell masks and moneyness/maturity slice reporting are necessary to prevent dense-grid metrics from hiding sparse-region behavior.

* `dynamics-of-implied-volatility-surfaces`, Section 3.1: maturity ranges from about one month to one year; moneyness outside `[0.5, 1.5]` is filtered; very-short maturities are avoided.
  Direct SPX/SP500 evidence: yes.
  ECON499 implication: ECON499 must document DTE/maturity and moneyness inclusion ranges and justify extrapolation or exclusion near expiry.

* `dynamics-of-implied-volatility-surfaces`, Section 3.1: the paper uses OTM calls for `m > 1` and OTM puts for `m < 1`.
  Direct SPX/SP500 evidence: yes.
  ECON499 implication: the option universe must specify call/put selection rules and whether put-call parity, OTM-only selection, or both-side aggregation is used.

* `dynamics-of-implied-volatility-surfaces`, Section 3.2 and equation (7): smooth daily IV surfaces are constructed from observed irregular data using a nonparametric Nadaraya–Watson Gaussian-kernel estimator on a fixed grid.
  Direct SPX/SP500 evidence: yes.
  ECON499 implication: completed-grid values are fitted/smoothed outputs and must be separated from observed market data.

* `dynamics-of-implied-volatility-surfaces`, Section 3.2: bandwidth choice controls overfitting vs oversmoothing and can be chosen by cross-validation or adaptive bandwidth methods.
  Direct SPX/SP500 evidence: methodology applied to SP500.
  ECON499 implication: interpolation/smoothing hyperparameters must be selected without future leakage and sensitivity-tested.

* `dynamics-of-implied-volatility-surfaces`, Section 3.2: the paper does not extrapolate to `τ = 0` and states very-near-expiry behavior can be irregular and unsuitable for risk-management use.
  Direct SPX/SP500 evidence: methodology applied to SP500.
  ECON499 implication: ECON499 should either exclude near-expiry cells or explicitly test/document near-expiry interpolation behavior.

* `dynamics-of-implied-volatility-surfaces`, Sections 3.3–3.4: the paper applies a Karhunen–Loève decomposition to daily variations of log implied volatility, `log I_t(m,τ) − log I_{t−1}(m,τ)`.
  Direct SPX/SP500 evidence: yes.
  ECON499 implication: PCA/factor features should be trained on training-window surfaces only and should clearly define whether levels, changes, IV, log IV, or total variance are decomposed.

* `dynamics-of-implied-volatility-surfaces`, Section 4, Figures 4–7: SP500 average surfaces show skew and term-structure structure; daily standard deviations are non-negligible; eigenvalues decay quickly.
  Direct SPX/SP500 evidence: yes.
  ECON499 implication: surface dynamics are structured enough for low-dimensional benchmarks, but surface losses should still be reported across the full grid/slices.

* `dynamics-of-implied-volatility-surfaces`, Section 4, Figures 8–14 and Table 1: the first factor is a level effect, the second affects skew/slope, the third affects curvature/butterfly-like shape; principal-component processes are autocorrelated and mean-reverting.
  Direct SPX/SP500 evidence: yes.
  ECON499 implication: a defensible factor benchmark should include level, slope/skew, and curvature dynamics, not only ATM volatility.

* `dynamics-of-implied-volatility-surfaces`, Table 1: SP500 factor processes show mean-reversion times around days-to-weeks, with the first factor negatively correlated with underlying returns and other factors less tied to the underlying.
  Direct SPX/SP500 evidence: yes.
  ECON499 implication: forecast features may include lagged factor dynamics and underlying-return interactions, but must preserve timestamp availability.

* `dynamics-of-implied-volatility-surfaces`, Section 6.1: the paper summarizes that daily log-IV variations can be explained by two or three principal components, with AR(1)/Ornstein–Uhlenbeck-like autocorrelation.
  Direct SPX/SP500 evidence: yes, jointly with FTSE.
  ECON499 implication: HAR/factor/AR benchmarks are not optional if the project claims research-grade IVS forecasting.

* `dynamics-of-implied-volatility-surfaces`, Section 6.2: the proposed factor model represents log-IV surfaces as an initial surface plus factor loadings with mean-reverting component processes.
  Direct SPX/SP500 evidence: methodology derived from SP500/FTSE.
  ECON499 implication: dynamic benchmark design should distinguish today’s calibrated/smoothed surface from the time-series model fitted to past factor scores.

## AUDIT_STANDARD_CONTRIBUTION

* Surface coordinate system:

  * Supports spot moneyness `m = K / S(t)` and time to maturity `τ = T − t` as a valid IVS coordinate system.
  * Supports making sticky/no-change benchmarks coordinate-explicit.
  * Does not directly support forward-moneyness or 15:45-specific coordinate availability.
  * Evidence type: direct SP500 methodology, but not real-time 15:45 evidence.

* Observed surface construction:

  * Supports starting from observed traded option implied volatilities on an irregular, changing grid.
  * Supports filtering by moneyness, maturity, and OTM call/put selection before constructing smooth surfaces.
  * Supports documenting sparse regions and avoiding near-expiry extrapolation.
  * Evidence type: direct SP500 evidence.

* Factor/PCA/HAR benchmark design:

  * Strongly supports low-dimensional factor benchmarks using daily log-IV surface changes.
  * Supports level, skew/slope, and curvature/butterfly-style factors.
  * Supports AR(1)/mean-reverting dynamics for factor scores; HAR-style extensions are ECON499 inference, not directly in this paper.
  * Evidence type: direct SP500 evidence.

* Forecast target and dynamics:

  * Supports modeling daily surface dynamics, especially log-IV changes on a smoothed fixed grid.
  * Does not directly define a modern supervised-learning forecast target or multi-horizon ML protocol.
  * Evidence type: direct SP500 daily EOD dynamics.

* Observed-vs-completed evaluation:

  * The paper constructs smooth completed surfaces from observed irregular data but does not define observed-cell masks or separately evaluate observed vs completed cells.
  * ECON499 standard: observed cells and completed-grid cells must be distinguished because this paper’s method itself shows observed support is irregular and smoothing-dependent.
  * Evidence type: direct SP500 motivation plus ECON499 inference.

* Leakage risks:

  * Supports temporal modeling of daily changes and today-to-tomorrow dynamics.
  * Does not discuss train/test leakage, HPO, or preprocessing fit scope in modern ML terms.
  * ECON499 inference: factor loadings, smoothing bandwidths, scalers, and any PCA basis used as features/benchmarks must be fit inside training windows only.

* Limitations for 15:45 ECON499 forecasting:

  * The paper uses daily end-of-day prices, so it cannot justify using same-day EOD fields in a 15:45 forecast.
  * It studies smoothed historical surfaces before factor analysis, not a real-time, intraday, availability-timestamped grid.
  * It does not establish arbitrage-free construction or static-arbitrage guarantees.

## REPO_CHECKS_DERIVED_FROM_THIS_PAPER

* Verify the surface coordinate metadata states the exact coordinate system: spot moneyness, forward moneyness, log moneyness, normalized log-forward moneyness, or strike.

* Verify the no-change benchmark is coordinate-explicit: sticky moneyness, sticky strike, or fixed-grid no-change.

* Verify spot/index level used for moneyness has an availability timestamp no later than 15:45.

* Verify the option universe states whether it uses OTM calls/puts, both calls and puts, put-call parity selection, or aggregation across option types.

* Verify moneyness filters are explicit, parameterized, logged, and tested.

* Verify DTE/maturity filters are explicit, parameterized, logged, and tested.

* Verify very-near-expiry options are either excluded by rule or handled by a documented special procedure.

* Verify raw observed option records are preserved separately from smoothed/interpolated/completed surfaces.

* Verify completed-grid artifacts include an observed-cell mask and do not label smoothed values as observed market data.

* Verify the grid is fixed only after explicit construction and that the original irregular observed support is still recoverable.

* Verify smoothing/interpolation parameters are recorded in surface metadata.

* Verify interpolation/smoothing parameter selection is performed without using validation/test/future dates when the smoothed surface is used as a feature or model input.

* Verify any offline target-construction smoothing is clearly separated from feature construction and cannot leak future information into predictors.

* Verify surface construction does not extrapolate outside the supported moneyness/maturity domain without an explicit extrapolation flag and diagnostics.

* Verify slice diagnostics report coverage/counts by moneyness and maturity, not only forecast error.

* Verify factor/PCA basis construction is fit on training-window data only for every split.

* Verify factor-score time series are aligned so that only factors observable by the prediction timestamp enter forecasts.

* Verify factor benchmark documentation states whether it decomposes IV levels, log-IV levels, IV changes, log-IV changes, or total variance.

* Verify factor benchmarks include at least level and shape dynamics; a benchmark using only ATM IV should not be treated as a full-surface benchmark.

* Verify AR/HAR/factor model lag selection is done inside training/validation windows, not on the full sample.

* Verify reports distinguish direct observed-cell performance from completed-grid/fitted-surface performance.

* Verify documentation does not cite this paper as support for 15:45 availability, same-day EOD feature use, static-arbitrage guarantees, or arbitrage-free claims.

## FORMAL_FINDINGS_IF_ANY

none
