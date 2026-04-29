ROUND_008B_VERIFICATION_DECISION: all_fixed

B6-CODE-001: fixed. The penalty layer now fails fast before computing empty second-difference tensors: roughness_penalty requires at least 3x3, convexity_penalty requires at least 3 moneyness points, and calendar_monotonicity_penalty requires at least 2 maturity points. NeuralSurfaceRegressor.__post_init__ also rejects enabled penalties whose configured grid is mathematically invalid. The official-smoke surface config is now 3x3, and the synthetic official-smoke raw generator emits matching 3-maturity x 3-moneyness rows.

Test assessment: adequate for this targeted finding. The added tests would have failed against the original bug: the direct 2x2 roughness test now expects ValueError, the neural regressor 2x2 enabled-roughness test now expects fail-fast construction, the non-finite aggregate-loss test rejects silent NaN loss propagation, and the repository contract now asserts the official-smoke grid is valid and finite for enabled neural penalties. The submitted evidence reports targeted ruff/pytest passing; I also confirmed targeted syntax compilation and isolated penalty behavior in the provided context.

Directly related regressions: none identified.

NEXT_REQUIRED_STEP: Codex should prepare a refreshed full-project code context and request another fresh full-project closure audit before any terminal closure statement.
