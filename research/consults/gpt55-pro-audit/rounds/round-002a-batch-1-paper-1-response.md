# Round 002A Pro Response: Batch 1 Paper 1

Captured: 2026-04-27

## READING_COVERAGE

* `ebsco-fulltext-04-08-2026` — “Predictable Dynamics in the S&P 500 Index Options Implied Volatility Surface”
* Sections/tables/figures/equations read:

  * Abstract/introduction: read for motivation, SPX IVS predictability claim, and two-step modeling setup.
  * Section II.A, “The Data”: read line-by-line for sample, daily closing timestamp, dividends, risk-free rate, filters, DTE, moneyness, and option counts.
  * Table 1: read for moneyness/maturity bins, sample composition, and observation imbalance.
  * Section II.B, “Fitting the Implied Volatility Surface”: read line-by-line for transformed moneyness, `DTE/252`, log-IV regression, and daily cross-sectional fit.
  * Equation (1): read and PDF-checked for the cross-sectional log-implied-volatility model.
  * Table 2 and Figures 1–3: read for fit quality, daily coefficient instability, and coefficient autocorrelation/cross-correlation.
  * Section III, “Modeling the Dynamics of the Implied Volatility Surface”: read for VAR dynamics, Dumas et al. strawman, NGARCH benchmark, and BIC lag selection.
  * Equation (2): read and PDF-checked for VAR coefficient dynamics.
  * Table 3 and Figure 4: read for dynamic-model fit and fitted-surface behavior.
  * Section IV, “Statistical Measures of Predictability”: read line-by-line for expanding-window out-of-sample design, one-step-ahead target, benchmarks, metrics, DM/Newey-West testing, and forecast horizon notes.
  * Tables 4–5: read for benchmark/loss design and moneyness/maturity-sliced errors.
  * Section V and Tables 6–8: read for economic-value setup and transaction-cost sensitivity.
  * Section VI, Appendix B, Tables 9–11: read for measurement-error and bid-ask-spread robustness; table structure was checked, but not every numeric cell was independently revalidated.
  * Conclusion: read for limitations, especially the admission that the model does not impose no-arbitrage restrictions.
* Whether extracted text was sufficient or PDF was consulted:

  * Extracted `paper.md` was sufficient for most prose.
  * PDF-rendered pages were consulted for layout-sensitive abstract, equations (1)–(2), table interpretation, and conclusion/no-arbitrage wording.
* Extraction caveats:

  * Table extraction is noisy in places, especially Tables 6–11 and appendix equations. The substantive table notes and model/evaluation definitions were readable, but exact numeric trading-table cells should be rechecked against the PDF before any hedging/economic-value audit.
  * Some mathematical symbols in equations are imperfectly extracted; equations (1), (2), and Appendix B were PDF-checked at the level needed for this round.
* Important section deferred:

  * Detailed hedging-return algebra in Section V/App. A and exact numeric verification of Tables 9–11 are deferred because Round 002A is about surface construction and forecasting standards, not hedging-module audit.

## PAPER_SPECIFIC_EVIDENCE

1. `ebsco-fulltext-04-08-2026`, Section II.A “The Data”
   Direct SPX evidence: yes.
   Evidence: the paper uses daily closing prices for S&P 500 index options from CBOE, January 3, 1992–June 28, 1996.
   Implication for ECON499: this paper supports a daily SPX IVS forecasting design, but it is close-based evidence and does not directly justify using same-day EOD fields for a 15:45 forecast.

2. `ebsco-fulltext-04-08-2026`, Section II.A “The Data”
   Direct SPX evidence: yes.
   Evidence: the paper computes DTE in trading days and sets time to maturity as `τ = DTE / 252`.
   Implication for ECON499: the repo must define and consistently test its DTE and year-fraction convention; if it uses calendar-day ACT/365 instead, that difference must be documented.

3. `ebsco-fulltext-04-08-2026`, Section II.A and Table 1 note
   Direct SPX evidence: yes.
   Evidence: moneyness is defined relative to the forward, `m = strike / forward price − 1`, with forward price based on `exp(rτ)S`.
   Implication for ECON499: all moneyness coordinates must state whether they use spot, forward, log-forward, or normalized log-forward moneyness, and all inputs must be available by 15:45.

4. `ebsco-fulltext-04-08-2026`, Section II.A “The Data”
   Direct SPX evidence: yes.
   Evidence: the paper applies explicit filters: thin trading below 100 contracts/day, basic no-arbitrage violations, DTE fewer than six trading days, DTE greater than one year, absolute moneyness above 10%, and price below $3/8.
   Implication for ECON499: every option filter must be explicit, logged, testable, and not silently drop rows.

5. `ebsco-fulltext-04-08-2026`, Table 1
   Direct SPX evidence: yes.
   Evidence: the filtered sample is unevenly distributed across moneyness/maturity buckets; short- and medium-term OTM/ATM contracts dominate, while DITM and long-term contracts are underrepresented.
   Implication for ECON499: evaluation must report performance by moneyness/maturity slice, not only an aggregate surface loss.

6. `ebsco-fulltext-04-08-2026`, Section II.B and Equation (1)
   Direct SPX evidence: yes.
   Evidence: the paper fits each day’s cross section using log implied volatility as a function of normalized log-forward moneyness, squared moneyness, maturity, and a moneyness-maturity interaction.
   Implication for ECON499: surface construction should distinguish observed raw option data from fitted or smoothed values; fitted values are model-implied completions, not observed quotes.

7. `ebsco-fulltext-04-08-2026`, Section II.B, Table 2, Figures 1–3
   Direct SPX evidence: yes.
   Evidence: daily cross-sectional coefficients are unstable and autocorrelated, motivating a second-stage dynamic model.
   Implication for ECON499: HAR/factor/coefficient-dynamic benchmarks are academically relevant, and preprocessing/factor fitting must be restricted to training windows.

8. `ebsco-fulltext-04-08-2026`, Section III and Equation (2)
   Direct SPX evidence: yes.
   Evidence: the paper models daily coefficient dynamics with VAR-type models and selects lag order by BIC up to a maximum of 12.
   Implication for ECON499: any dynamic benchmark should use time-series validation and training-window-only model selection, not random cross-validation.

9. `ebsco-fulltext-04-08-2026`, Section IV and Table 4
   Direct SPX evidence: yes.
   Evidence: the out-of-sample design uses expanding estimation windows followed by nonoverlapping six-month prediction windows, with one-step-ahead daily forecasts.
   Implication for ECON499: split manifests should serialize expanding or blocked walk-forward windows and prove that forecasts use only information available through forecast time.

10. `ebsco-fulltext-04-08-2026`, Section IV and Table 4
    Direct SPX evidence: yes.
    Evidence: benchmarks include a Dumas et al. ad hoc strawman, NGARCH(1,1), and a random-walk/no-change IV benchmark.
    Implication for ECON499: a no-change surface benchmark is mandatory, and model improvements should be judged against strong dynamic and practitioner-style baselines.

11. `ebsco-fulltext-04-08-2026`, Section IV and Table 4
    Direct SPX evidence: yes.
    Evidence: errors are computed on traded options using IV and option-price losses, with Diebold-Mariano tests using Newey-West covariance estimates.
    Implication for ECON499: evaluation should include aligned loss panels, observed-option or observed-cell losses, and statistically defensible forecast-comparison tests.

12. `ebsco-fulltext-04-08-2026`, Section IV and Table 5
    Direct SPX evidence: yes.
    Evidence: the paper reports forecast errors by moneyness and maturity classes and shows performance varies materially by region of the surface.
    Implication for ECON499: aggregate RMSE/MAE is insufficient; slice reports by maturity and moneyness are required.

13. `ebsco-fulltext-04-08-2026`, Section IV, note on pricing metrics
    Direct SPX evidence: yes.
    Evidence: the paper warns that estimating in IV loss but evaluating option-price loss can create loss-function mismatch concerns.
    Implication for ECON499: the repo must document whether models are trained/evaluated in IV, total variance, price, or mixed losses, and must not overstate comparability across loss scales.

14. `ebsco-fulltext-04-08-2026`, Section VI.A and Appendix B
    Direct SPX evidence: yes.
    Evidence: the paper treats measurement error in option prices and the underlying index as a source of heteroskedasticity/correlation in implied-volatility errors and uses feasible GLS as a robustness check.
    Implication for ECON499: cleaning and surface fitting should preserve quality diagnostics and support robustness checks for noisy quotes, illiquidity, and sparse regions.

15. `ebsco-fulltext-04-08-2026`, Conclusion
    Direct SPX evidence: yes.
    Evidence: the authors state their unrestricted VAR approach does not exploit no-arbitrage restrictions.
    Implication for ECON499: this paper cannot justify calling a learned surface “arbitrage-free”; at most it supports forecasting IVS dynamics unless hard no-arbitrage constraints are separately imposed.

## AUDIT_STANDARD_CONTRIBUTION

* Timestamp and data availability:

  * Contribution: supports strict use of information available through the forecasting date and nonoverlapping out-of-sample windows, but the paper is based on daily closing prices.
  * Standard for ECON499: because ECON499 forecasts at 15:45 America/New_York, same-day EOD/close fields are forbidden unless the artifact proves they are available by 15:45.
  * Support: direct SPX evidence for daily forecasting; 15:45 timestamp discipline is an ECON499 inference, not a direct paper claim.

* Option filtering/universe:

  * Contribution: supports explicit, documented filters for volume/liquidity, no-arbitrage violations, DTE bounds, moneyness bounds, and minimum price.
  * Standard for ECON499: every filter/drop rule must be parameterized, logged, counted, testable, and reproducible.
  * Support: direct SPX evidence.

* Moneyness/maturity coordinate choices:

  * Contribution: supports forward-based moneyness and `τ = DTE/252`; also supports normalized log-forward moneyness for cross-sectional fitting.
  * Standard for ECON499: moneyness coordinate, forward/spot choice, interest-rate/dividend treatment, and year-fraction convention must be explicit and timestamp-safe.
  * Support: direct SPX evidence.

* Observed surface construction:

  * Contribution: supports constructing each daily cross section from actually traded options and then fitting a cross-sectional model.
  * Standard for ECON499: raw observed option records/cells must remain distinguishable from fitted or completed surface values.
  * Support: direct SPX evidence plus project inference.

* Completed-grid or fitted-surface treatment:

  * Contribution: the paper fits a smooth parametric surface, but does not define a fixed completed grid in the modern ML sense.
  * Standard for ECON499: any completed grid must be labeled as model/interpolation output, not observed data.
  * Support: direct SPX evidence for fitted surfaces; completed-grid requirement is an ECON499 inference.

* Forecast target and benchmark design:

  * Contribution: supports one-step-ahead daily IVS forecasting, no-change/random-walk benchmark, practitioner ad hoc benchmark, structural benchmark, coefficient/factor dynamics, expanding-window evaluation, and DM tests.
  * Standard for ECON499: no-change surface benchmark and dynamic factor/HAR-style benchmark are mandatory; tuning and benchmark selection must be confined to training/validation windows.
  * Support: direct SPX evidence.

* Observed-vs-completed evaluation:

  * Contribution: the paper evaluates forecast errors against traded options and reports moneyness/maturity slices; it does not explicitly discuss observed-cell masks.
  * Standard for ECON499: observed-cell evaluation must be primary or separately reported; completed-grid evaluation must not obscure observed-cell performance.
  * Support: direct SPX evidence for traded-option evaluation; observed-mask standard is project inference.

* Leakage risks:

  * Contribution: the paper emphasizes using past dynamic information, nonoverlapping estimation/prediction windows, and current values when future primitive inputs are unknown.
  * Standard for ECON499: forecasts at 15:45 must not use same-day EOD fields, future coefficients, future interpolation fits, future scaling objects, or future target information.
  * Support: direct SPX evidence for temporal separation; 15:45 leakage rule is ECON499-specific inference.

## REPO_CHECKS_DERIVED_FROM_THIS_PAPER

1. Verify the official forecast timestamp is represented in code/artifacts and that no field with availability after 15:45 is used for same-day prediction.

2. Verify all raw-to-clean option filters are explicit, parameterized, logged, and tested, including DTE bounds, price bounds, moneyness bounds, no-arbitrage filters, liquidity filters, and option-type filters.

3. Verify every filter produces row counts by date and reason, with no silent row drops.

4. Verify DTE is computed from the trading calendar or documented convention, and that the time-to-expiry denominator is explicit.

5. Verify moneyness is consistently defined as spot, forward, log-forward, or normalized log-forward moneyness, and that the chosen definition is documented in surface metadata.

6. Verify forward, rate, dividend, and spot/index inputs used for moneyness have availability timestamps no later than the prediction timestamp.

7. Verify observed raw option records or observed grid cells are persisted separately from fitted/interpolated/completed values.

8. Verify completed-grid artifacts include a mask identifying which cells were observed versus filled.

9. Verify surface-fitting/interpolation is fit using only data available up to the relevant construction/forecast timestamp and does not pool future dates unless explicitly in an offline target-construction step that is never used as a feature.

10. Verify all preprocessing objects used for features, factors, scaling, PCA, or coefficient dynamics are fit inside the training window only.

11. Verify walk-forward or expanding-window split manifests serialize train/validation/test dates and prove nonoverlap.

12. Verify no random cross-validation is used for model selection or HPO in time-series forecasting modules.

13. Verify the no-change/random-walk surface benchmark is implemented and included in evaluation outputs.

14. Verify benchmark/evaluation reports include moneyness-by-maturity slice metrics, not only aggregate RMSE/MAE.

15. Verify metrics are computed separately for observed cells and completed-grid cells.

16. Verify forecast/target alignment requires the same option cell or grid coordinate and correct forecast horizon.

17. Verify any direction-of-change metric uses only cells/contracts observed on the required consecutive dates.

18. Verify forecast-comparison tests operate on aligned loss panels and use time-series-appropriate covariance/bootstrap handling.

19. Verify documentation does not describe fitted/interpolated surfaces as observed market data.

20. Verify documentation does not claim “arbitrage-free” based on this paper; any such claim must come from separate hard constraints or static-arbitrage literature.

## FORMAL_FINDINGS_IF_ANY

none
