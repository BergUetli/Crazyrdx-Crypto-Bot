
## [2026-09-02] Weekday-hour interaction feature — NEEDS HUMAN REVIEW

The current grammar encodes hour_of_day_sin/cos and day_of_week independently but not their interaction. A precomputed is_asia_open, is_us_open, is_eu_open boolean (hour buckets per UTC) would let the engine test session-specific momentum without needing new data sources, only a feature-engineering pass on existing timestamps.

Rationale: Champion strategies already lean on day_of_week and weekend effects; intraday session structure is the logical next layer since funding resets and liquidation cascades cluster by timezone overlap.

## [2026-09-02] Drawdown-conditioned position sizing — NEEDS HUMAN REVIEW

Add a sizing_method option drawdown_scaled that reduces sizing_base proportionally when the strategy's rolling 7-day equity drawdown exceeds a configurable threshold. Requires the engine to track per-strategy equity curves and pass a drawdown_pct context variable into the sizing module at entry time.

Rationale: Volatility-scaled sizing adjusts for market vol but not for strategy-specific losing streaks; drawdown conditioning is a well-documented risk control that would reduce ruin risk during regime changes without requiring new market data.
