"""Build past-only features: lags, rolling means, month, season, market category.

Every feature must be computable from information available strictly before the
target week, so that no future value can leak into training. Filled in during Phase 2.
"""
