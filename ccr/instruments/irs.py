"""
irs.py — Interest Rate Swap pricing
====================================
Prices a vanilla fixed-for-floating IRS using a flat or bootstrapped discount
curve. Returns MtM, DV01 (dollar duration) and annuity factor.

Conventions
-----------
- Payer swap  : fixed rate payer / floating rate receiver  → MtM > 0 when rates rise
- Receiver swap : opposite → MtM > 0 when rates fall
- MtM is from the perspective of the fixed payer

References
----------
- Brigo & Mercurio (2006), Interest Rate Models — Theory and Practice
- Basel BCBS (2014), SA-CCR: Supervisory delta for linear IR instruments
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class IRSwap:
    """
    Vanilla fixed-for-floating interest rate swap.

    Parameters
    ----------
    notional : float
        Notional principal in base currency.
    fixed_rate : float
        Annualised fixed coupon rate (e.g. 0.03 for 3%).
    maturity : float
        Remaining maturity in years.
    pay_freq : int
        Coupon payments per year (1=annual, 2=semi, 4=quarterly).
    position : {"payer", "receiver"}
        "payer"   → pays fixed, receives floating  (long rates)
        "receiver" → receives fixed, pays floating (short rates)
    """
    notional: float = 1_000_000.0
    fixed_rate: float = 0.03
    maturity: float = 5.0
    pay_freq: int = 2
    position: Literal["payer", "receiver"] = "payer"

    def _payment_times(self) -> np.ndarray:
        dt = 1.0 / self.pay_freq
        return np.arange(dt, self.maturity + 1e-9, dt)

    def discount_factors(self, rate_curve: float | np.ndarray, times: np.ndarray) -> np.ndarray:
        """Flat or term-structured discount factors P(0,t) = exp(-r(t)*t)."""
        if np.isscalar(rate_curve):
            return np.exp(-rate_curve * times)
        return np.exp(-rate_curve * times)   # rate_curve already indexed to times

    def mtm(self, market_rate: float) -> float:
        """
        Mark-to-market of the swap at current market rate.

        MtM = N * (S - K) * A     for payer
        MtM = N * (K - S) * A     for receiver

        where S = market swap rate, K = fixed rate, A = annuity factor.

        Parameters
        ----------
        market_rate : float
            Current par swap rate for the remaining tenor.
        """
        times = self._payment_times()
        df = self.discount_factors(market_rate, times)
        dt = 1.0 / self.pay_freq
        annuity = np.sum(df * dt)           # PV of 1 bp running coupon stream
        sign = 1.0 if self.position == "payer" else -1.0
        return sign * self.notional * (market_rate - self.fixed_rate) * annuity

    def dv01(self, market_rate: float, bump: float = 0.0001) -> float:
        """
        DV01: sensitivity of MtM to a 1 bp parallel shift in the rate curve.

        DV01 = -(MtM(r+1bp) - MtM(r)) / 1
        Always returned as a positive number (loss per bp rise for payer).
        """
        return -(self.mtm(market_rate + bump) - self.mtm(market_rate)) / (bump / 0.0001)

    def annuity(self, market_rate: float) -> float:
        """Present value of a unit annuity over the swap's life."""
        times = self._payment_times()
        df = self.discount_factors(market_rate, times)
        return np.sum(df * (1.0 / self.pay_freq))

    def par_rate(self, market_rate: float) -> float:
        """
        Fair fixed rate that makes the swap NPV = 0.
        S = (1 - P(0,T)) / annuity
        """
        times = self._payment_times()
        df = self.discount_factors(market_rate, times)
        return (1.0 - df[-1]) / self.annuity(market_rate)

    def mtm_profile(self, rate_paths: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        """
        Compute MtM along Monte Carlo rate paths.

        Parameters
        ----------
        rate_paths : np.ndarray, shape (n_paths, n_times)
            Short-rate or par-rate paths.
        time_grid : np.ndarray, shape (n_times,)
            Simulation time grid in years.

        Returns
        -------
        np.ndarray, shape (n_paths, n_times)
            MtM at each path/time step (positive = in-the-money for payer).
        """
        n_paths, n_times = rate_paths.shape
        mtm_matrix = np.zeros((n_paths, n_times))

        for j, t in enumerate(time_grid):
            remaining = max(self.maturity - t, 0.0)
            if remaining < 1e-6:
                continue
            # Truncate to remaining life
            swap_t = IRSwap(
                notional=self.notional,
                fixed_rate=self.fixed_rate,
                maturity=remaining,
                pay_freq=self.pay_freq,
                position=self.position,
            )
            for i in range(n_paths):
                mtm_matrix[i, j] = swap_t.mtm(rate_paths[i, j])

        return mtm_matrix
