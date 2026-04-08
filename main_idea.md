# COMPREHENSIVE_IDEA_BREAKDOWN

Project title
Arbitrage-aware next-day forecasting of the SPX 15:45 implied volatility surface from raw Cboe option data

Core thesis
Build a research-grade end-to-end pipeline that forecasts the next observed trading day's SPX volatility surface using only information available at 15:45 ET on day t, then test whether those forecasts improve next-day revaluation and hedge decisions relative to strong simple baselines.

Motivation
Implied volatility surfaces are operational inputs for marking option books, estimating risk, comparing exposures across strikes and maturities, and selecting hedge adjustments. A better next-day surface forecast matters only if it improves one of those tasks. This project is not about proving a tradable volatility alpha. It is about whether better surface forecasts improve real decision quality under uncertainty.

Exact research question
Given raw SPX option quotes and 15:45 calc fields from Cboe/Livevol, can an arbitrage-aware joint model forecast the next observed trading day’s SPX surface more accurately than:
1. a no-change surface benchmark,
2. regularized linear/factor/HAR-style models,
3. standard tree-based models,
and does any statistical improvement translate into lower next-day revaluation error or hedge error for a standardized SPX option book?

Why the question matters
This is a good thesis question because:
- it is economically meaningful,
- it is narrow enough to answer cleanly,
- it forces serious treatment of timing, leakage, and data engineering,
- it sits directly inside established literature on IV-surface dynamics, no-arbitrage geometry, and option hedging.

Data scope
Underlying: SPX only
Sample: 2004-01-02 to 2021-04-09
Decision time: 15:45 ET on day t
Information set: only fields operationally available by 15:45 ET on day t
Target date: next observed trading day t+1
Frequency: daily
Source: Cboe/Livevol Option EOD Summary product with calcs included

Key data fields
Required:
- quote_date
- expiration
- strike
- option_type
- bid_1545, ask_1545, bid_size_1545, ask_size_1545
- active_underlying_price_1545
- implied_volatility_1545
- delta_1545, gamma_1545, theta_1545, vega_1545, rho_1545

Optional, only after availability is audited:
- open_interest

Forbidden in the core 15:45 forecasting problem:
- same-day EOD quote fields
- same-day OHLC fields
- same-day VWAP
- any feature that is only known after the 15:45 decision time

Target construction
Primary modeled target:
- total implied variance surface w_{t+1}(m, tau) on a fixed moneyness x maturity grid

Reported target:
- implied volatility surface sigma_{t+1}(m, tau), obtained by transforming the predicted total variance surface

Why total variance is the primary target
- it behaves better across maturities,
- it aligns better with maturity monotonicity constraints,
- it is more appropriate for static-geometry checks than raw IV,
- it avoids pretending that raw-IV interpolation has good economic geometry.

Surface construction approach
1. Clean option-level observations.
   - remove nonpositive bid/ask quotes
   - require ask >= bid
   - remove options with invalid or zero 15:45 IV
   - remove clearly unusable tails by pre-registered rules
   - handle early-close days explicitly
2. Compute time to maturity tau in years using the trading calendar.
3. Compute log-moneyness consistently for the whole project. Use active_underlying_price_1545 as the default reference price unless the thesis explicitly implements a forward-based alternative.
4. Map each option to a fixed grid in log-moneyness x maturity.
5. Within each grid cell, aggregate option-level total variance using vega weights.
6. Store the observed-cell mask and coverage statistics for every day.
7. Complete missing cells in total variance using a shape-preserving interpolator.
   - Because PCHIP is one-dimensional, define the 2D completion explicitly as sequential axis-wise interpolation.
   - Keep the interpolation order fixed.
   - Run a sensitivity check to show that results are not driven by interpolation order.
8. Convert completed total variance back to IV only after completion.
9. Persist both:
   - observed-cell surface
   - completed full-grid surface

Important honesty constraint
This construction is pragmatic and economically informed. It is not a theorem-backed globally arbitrage-free surface construction. Call it arbitrage-aware, not arbitrage-free.

Arbitrage-aware modeling rationale
The flagship model should predict the whole next-day surface jointly and include soft penalties that discourage:
- decreases in total variance across maturity,
- non-convexity across strike or moneyness,
- optionally, excessive roughness that creates obviously implausible local artifacts.

The objective is not to force a closed-form hard-constrained surface class. The objective is to bias a flexible predictive model toward economically coherent outputs while preserving enough flexibility to learn market dynamics.

Baseline models
Mandatory baselines
1. No-change / random-walk surface:
   tomorrow’s forecast = today’s 15:45 completed surface
2. Ridge regression
3. Elastic net
4. HAR-style or factor-autoregressive benchmark on surface factors
5. LightGBM
6. Random forest

Interpretation of baselines
- No-change is the minimum bar.
- Ridge and elastic net are the most important “simple but serious” competitors.
- HAR/factor models are the literature-friendly classical benchmark family.
- LightGBM is the strongest non-neural tabular baseline.
- Random forest is kept as a weak benchmark, not an expected winner.

Flagship model
- compact fully connected neural network or surface MLP
- predicts the full next-day grid jointly
- outputs total variance, not raw IV
- includes soft calendar and convexity penalties
- uses observed/imputed masks in the loss so the model is not rewarded equally for directly observed cells and interpolated cells

Feature design
Keep the core feature set narrow and defensible:
- lagged completed surface levels
- lagged surface changes
- lagged observed-cell masks and coverage ratios
- vega-weighted liquidity and spread features by cell or maturity bucket
- lagged underlying return / move proxies from the same dataset
- optional exogenous regime variables only as secondary ablations, not as a dependency of the core thesis

Anti-leakage design
Non-negotiable rules:
- all preprocessing respects time order
- no random train/validation/test splits
- every scaler, imputer, interpolation hyperparameter, and model tuner is fit inside the training window only
- grid design is fixed ex ante and not chosen using future quantiles
- no hidden use of tomorrow’s observed-cell pattern
- hyperparameters are tuned only on pre-OOS blocked validation windows and then frozen for the main OOS run, or re-tuned on a pre-registered schedule
- no same-day EOD information in features

Evaluation design
Primary statistical evaluation
- vega-weighted RMSE on total variance
- vega-weighted MAE on total variance
- vega-weighted RMSE on IV
- vega-weighted MAE on IV
- MSE on IV change

Report metrics on:
a) observed cells only
b) full completed grid
c) maturity slices
d) moneyness slices
e) stressed subperiods

Secondary statistical evaluation
- QLIKE only on strictly positive total-variance targets with an explicit floor
- arbitrage-violation diagnostics:
  - calendar monotonicity violations
  - convexity violations
  - magnitude-weighted violation summaries

Formal forecast comparison
- Diebold-Mariano for pairwise tests
- SPA for multiple-model / data-snooping-aware superiority testing
- MCS for a superior-model set rather than winner-take-all storytelling

Hedging-utility justification
Economic value should not be framed as speculative volatility alpha.
Use a standardized SPX option book that contains:
- ATM straddle exposure for surface level
- risk-reversal exposure for skew
- calendar-spread exposure for term structure

Run two downstream tests.

1. Revaluation test
- mark the book at 15:45 on day t
- forecast the t+1 surface with each model
- convert each forecast to option values
- compare predicted t+1 book value with actual t+1 marked value
- report absolute and squared revaluation error

2. Hedge-error test
- set a delta-vega hedge at 15:45 on day t using each model’s t+1 surface forecast
- carry the hedge one trading day
- compare realized next-day hedge P&L variance / absolute error against:
  - no-change surface hedge
  - simple benchmark model hedge

Likely risks and limitations
- the no-change surface may be very hard to beat
- ridge / elastic net may beat the neural model
- interpolation may create artificial smoothness and exaggerate predictability
- soft penalties reduce arbitrage but do not guarantee its absence
- vendor IV/Greeks include model assumptions and invalid zero states
- same-day open interest may prove operationally ambiguous
- SPX-only scope limits generalization
- sample ends in 2021, so the conclusion is historical, not universal

What would count as success
A successful thesis does NOT require the neural model to dominate everything.

Success means:
1. the pipeline is clean, reproducible, and leak-free,
2. the surface-construction stage is explicit and audited,
3. the no-change benchmark is beaten in at least part of the surface or by at least one serious model family,
4. formal tests show whether gains are statistically meaningful,
5. the study gives a clear answer on whether arbitrage-aware modeling improves next-day revaluation or hedge quality,
6. negative findings are documented honestly if simpler models win.

Best final contribution claim
This thesis contributes a disciplined, anti-leakage, arbitrage-aware SPX IV-surface forecasting benchmark built from raw 15:45 option data, together with a concrete next-day revaluation and hedging-use test.