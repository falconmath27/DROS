"""Causal daily-observation exits, sharing the learned methods' fit/stop interface."""
import numpy as np
from common.problem import first_true


class HoldToEnd:
    def fit(self, X, R, X_val=None, R_val=None):
        return self

    def stop(self, X, R):
        return np.full(len(X), X.shape[1] - 1, dtype=int)


class TakeProfitStopLoss(HoldToEnd):
    def __init__(self, take_profit=0.05, stop_loss=0.03):
        if take_profit <= 0 or not 0 < stop_loss < 1:
            raise ValueError("Positive profit and stop_loss in (0,1) required")
        self.take_profit, self.stop_loss = take_profit, stop_loss

    def stop(self, X, R):
        # Feature 0 = gross return since entry. No future reward input is inspected.
        mask = (X[:, :, 0] >= self.take_profit) | (X[:, :, 0] <= -self.stop_loss)
        mask[:, 0] = False
        return first_true(mask, X.shape[1] - 1)


class TrailingStop(HoldToEnd):
    def __init__(self, drawdown=0.05):
        if not 0 < drawdown < 1:
            raise ValueError("drawdown must lie in (0,1)")
        self.drawdown = drawdown

    def stop(self, X, R):
        mask = X[:, :, 1] <= -self.drawdown
        mask[:, 0] = False
        return first_true(mask, X.shape[1] - 1)
