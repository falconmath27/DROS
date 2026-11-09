"""Proportional research cost scenarios; presets are assumptions, not quoted tariffs."""
from dataclasses import asdict, dataclass, replace
import numpy as np


@dataclass(frozen=True)
class CostModel:
    buy_bps: float = 0.0
    sell_bps: float = 0.0
    half_spread_bps: float = 0.0
    slippage_bps: float = 0.0
    funding_bps_per_day: float = 0.0
    discount_rate: float = 0.0
    label: str = "custom_assumption"

    def __post_init__(self):
        for name in ("buy_bps", "sell_bps", "half_spread_bps", "slippage_bps"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.sell_bps + self.half_spread_bps + self.slippage_bps >= 10000:
            raise ValueError("Exit friction must be below 100 percent")
        if not np.isfinite(self.funding_bps_per_day) or not np.isfinite(self.discount_rate):
            raise ValueError("Funding/discount must be finite")

    def net_return(self, relative_price, elapsed_days):
        buy = (self.buy_bps + self.half_spread_bps + self.slippage_bps) / 10000
        sell = (self.sell_bps + self.half_spread_bps + self.slippage_bps) / 10000
        value = np.asarray(relative_price) * (1 - sell) / (1 + buy) - 1
        value = value - self.funding_bps_per_day / 10000 * np.asarray(elapsed_days)
        return value * np.exp(-self.discount_rate * np.asarray(elapsed_days) / 365.25)

    def scaled(self, multiplier):
        if multiplier < 0 or not np.isfinite(multiplier):
            raise ValueError("Cost multiplier must be finite/nonnegative")
        return replace(self, **{k: getattr(self, k) * multiplier for k in
            ("buy_bps", "sell_bps", "half_spread_bps", "slippage_bps", "funding_bps_per_day")})

    def to_dict(self):
        return asdict(self)


def venue_costs(venue, multiplier=1.0):
    presets = {
        "nse": CostModel(12, 12, 2, 2, label="NSE_assumed_12bp_fees_each_side"),
        "us": CostModel(1, 1, 1, 1, label="US_assumed_1bp_fees_each_side"),
        "crypto": CostModel(10, 10, 2, 2, label="spot_assumed_10bp_taker_each_side"),
        "perpetual": CostModel(5, 5, 2, 2, 1, label="perpetual_simplified_funding_scenario"),
        "zero": CostModel(label="frictionless"),
    }
    if venue not in presets:
        raise ValueError(f"Unknown venue {venue}")
    return presets[venue].scaled(multiplier)
