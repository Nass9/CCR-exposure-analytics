"""
cds.py — Credit Default Swap pricing
======================================
Prices a vanilla single-name CDS using reduced-form (intensity) approach.
The hazard rate λ is assumed constant or piecewise constant.

MtM from protection buyer's perspective:
  MtM = Protection Leg PV  -  Premium Leg PV

Protection leg: PV of receiving (1 – R) on default before maturity
Premium leg:    PV of paying spread s on surviving notional

References
----------
- O'Kane & Turnbull (2003), Lehman Brothers Fixed Income Quantitative Research
- Hull & White (2000), Valuing credit default swaps I: No counterparty default risk
- ISDA CDS Standard Model (2009)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class CDS:
    """
    Vanilla single-name Credit Default Swap.

    Parameters
    ----------
    notional : float
        Reference notional.
    spread : float
        Running spread (annual, in decimal, e.g. 0.01 = 100 bps).
    maturity : float
        Remaining maturity in years.
    recovery : float
        Recovery rate R (typically 0.40 for senior unsecured).
    pay_freq : int
        Premium payment frequency per year (typically 4 = quarterly).
    position : {"protection_buyer", "protection_seller"}
        "protection_buyer" → pays spread, receives (1-R) on default
        "protection_seller" → receives spread, pays (1-R) on default
    """
    notional: float = 1_000_000.0
    spread: float = 0.01         # 100 bps
    maturity: float = 5.0
    recovery: float = 0.40
    pay_freq: int = 4
    position: Literal["protection_buyer", "protection_seller"] = "protection_buyer"

    def _hazard_rate_from_spread(self, market_spread: float) -> float:
        """
        Approximate hazard rate: λ ≈ s / (1 - R)
        (exact under flat intensity and flat discount curve).
        """
        return market_spread / (1.0 - self.recovery)

    def survival_prob(self, t: float, market_spread: float) -> float:
        """Survival probability Q(τ > t) = exp(-λ * t)."""
        lam = self._hazard_rate_from_spread(market_spread)
        return np.exp(-lam * t)

    def _premium_leg_pv(self, market_rate: float, market_spread: float) -> float:
        """
        PV of premium leg: sum over payment dates of
            s/freq * N * P(0,ti) * Q(τ > ti)
        with accrual approximation.
        """
        dt = 1.0 / self.pay_freq
        times = np.arange(dt, self.maturity + 1e-9, dt)
        lam = self._hazard_rate_from_spread(market_spread)
        pv = 0.0
        for t in times:
            df = np.exp(-market_rate * t)
            sp = np.exp(-lam * t)
            pv += self.spread / self.pay_freq * df * sp
        return self.notional * pv

    def _protection_leg_pv(self, market_rate: float, market_spread: float) -> float:
        """
        PV of protection leg (continuous approximation):
            (1-R) * N * integral_0^T P(0,t) * lambda * exp(-lambda*t) dt
        """
        lam = self._hazard_rate_from_spread(market_spread)
        # Numerical integration
        n_steps = 252
        dt = self.maturity / n_steps
        times = np.linspace(dt, self.maturity, n_steps)
        df = np.exp(-market_rate * times)
        surv_density = lam * np.exp(-lam * times)
        pv = np.sum(df * surv_density) * dt
        return self.notional * (1.0 - self.recovery) * pv

    def mtm(self, market_rate: float, market_spread: float) -> float:
        """
        Mark-to-market from protection buyer's perspective.
        MtM = Protection Leg - Premium Leg  (for buyer)
        MtM > 0 when market spread > contracted spread (buyer wins).

        Parameters
        ----------
        market_rate   : float   Current risk-free rate.
        market_spread : float   Current market CDS spread for this name/tenor.
        """
        prot = self._protection_leg_pv(market_rate, market_spread)
        prem = self._premium_leg_pv(market_rate, market_spread)
        # At-market CDS: replace our spread with market spread to get breakeven
        prem_contracted = self._premium_leg_pv_contracted(market_rate, market_spread)
        mtm_buyer = prot - prem_contracted
        sign = 1.0 if self.position == "protection_buyer" else -1.0
        return sign * mtm_buyer

    def _premium_leg_pv_contracted(self, market_rate: float, market_spread: float) -> float:
        """PV of contractual premium leg (uses self.spread, not market_spread)."""
        dt = 1.0 / self.pay_freq
        times = np.arange(dt, self.maturity + 1e-9, dt)
        lam = self._hazard_rate_from_spread(market_spread)
        pv = 0.0
        for t in times:
            df = np.exp(-market_rate * t)
            sp = np.exp(-lam * t)
            pv += self.spread / self.pay_freq * df * sp
        return self.notional * pv

    def par_spread(self, market_rate: float, market_spread: float) -> float:
        """
        Fair spread that makes MtM = 0:
        s* = Protection Leg PV / Risky Annuity
        """
        prot = self._protection_leg_pv(market_rate, market_spread)
        dt = 1.0 / self.pay_freq
        times = np.arange(dt, self.maturity + 1e-9, dt)
        lam = self._hazard_rate_from_spread(market_spread)
        risky_annuity = np.sum([
            np.exp(-market_rate * t) * np.exp(-lam * t) / self.pay_freq
            for t in times
        ])
        return prot / (self.notional * risky_annuity) if risky_annuity > 0 else 0.0

    def cs01(self, market_rate: float, market_spread: float, bump: float = 0.0001) -> float:
        """
        CS01: sensitivity of MtM to a 1 bp parallel shift in credit spread.
        Returned as absolute value (loss for protection buyer when spreads tighten).
        """
        mtm_up   = self.mtm(market_rate, market_spread + bump)
        mtm_down = self.mtm(market_rate, market_spread)
        return (mtm_up - mtm_down) / (bump / 0.0001)

    def mtm_profile(
        self,
        spread_paths: np.ndarray,
        rate_paths: np.ndarray,
        time_grid: np.ndarray,
    ) -> np.ndarray:
        """
        MtM profile along Monte Carlo spread/rate paths.

        Parameters
        ----------
        spread_paths : np.ndarray, shape (n_paths, n_times)
        rate_paths   : np.ndarray, shape (n_paths, n_times)
        time_grid    : np.ndarray, shape (n_times,)

        Returns
        -------
        np.ndarray, shape (n_paths, n_times)
        """
        n_paths, n_times = spread_paths.shape
        mtm_matrix = np.zeros((n_paths, n_times))
        for j, t in enumerate(time_grid):
            rem = max(self.maturity - t, 0.0)
            if rem < 1e-9:
                continue
            cds_t = CDS(
                notional=self.notional,
                spread=self.spread,
                maturity=rem,
                recovery=self.recovery,
                pay_freq=self.pay_freq,
                position=self.position,
            )
            for i in range(n_paths):
                mtm_matrix[i, j] = cds_t.mtm(rate_paths[i, j], spread_paths[i, j])
        return mtm_matrix
