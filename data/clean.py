"""Cleaning (P4): gaps, outliers, delisted names, stablecoin exclusion.
Log every survivorship choice (which names were dropped and why)."""


def clean_prices(prices, max_gap_days=5, outlier_z=10.0):
    """Return (clean DataFrame, log of what was removed)."""
    raise NotImplementedError("P4")
