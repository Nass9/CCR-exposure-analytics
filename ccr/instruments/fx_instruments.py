"""
fx_instruments.py — FX Forward and FX Option pricing
======================================================
Prices FX forwards (outright) and vanilla FX options using the
Garman-Kohlhagen model (1983) — extension of Black-Scholes to FX.

Convention
----------
- S  = spot FX rate (units of domestic per unit of foreign, e.g. EUR/USD = 1.10)
- rd = domestic risk-free rate (continuously compounded)
- rf = foreign risk-free rate (continuously compounded)
- Forward: F(0,T) = S * exp((rd - rf) * T)

References
----------
- Garman & Kohlhagen (1983), JFM — Foreign currency option values
- Reiswich & Wystup (2010) — FX volatility smile and conventions
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# FX Forward
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FXForward:
    """
    Vanilla FX forward contract.

    Parameters
    ----------
    notional_foreign : float
        Notional in the foreign (base) currency.
    forward_rate_agreed : float
        Contractual forward rate agreed at inception (K).
    maturity : float
        Remaining time to delivery in years.
    position : {"long", "short"}
        "long"  → buy foreign / sell domestic at K
        "short" → sell foreign / buy domestic at K
    """
    notional_foreign: float = 1_000_000.0
    forward_rate_agreed: float = 1.10          # e.g. EUR/USD at inception
    maturity: float = 1.0
    position: Literal["long", "short"] = "long"

    def mtm(self, spot: float, rd: float, rf: float) -> float:
        """
        MtM of the FX forward at current spot and rates.

        MtM = N_foreign * (F_market - K) * exp(-rd * T)    [long]
        F_market = spot * exp((rd - rf) * T)

        Parameters
        ----------
        spot : float      Current spot FX rate.
        rd   : float      Domestic risk-free rate.
        rf   : float      Foreign risk-free rate.
        """
        T = self.maturity
        F_market = spot * np.exp((rd - rf) * T)
        pv_diff = (F_market - self.forward_rate_agreed) * np.exp(-rd * T)
        sign = 1.0 if self.position == "long" else -1.0
        return sign * self.notional_foreign * pv_diff

    def mtm_profile(
        self,
        spot_paths: np.ndarray,
        rd: float,
        rf: float,
        time_grid: np.ndarray,
    ) -> np.ndarray:
        """
        MtM along Monte Carlo spot paths.

        Parameters
        ----------
        spot_paths : np.ndarray, shape (n_paths, n_times)
        time_grid  : np.ndarray, shape (n_times,)  — simulation times in years

        Returns
        -------
        np.ndarray, shape (n_paths, n_times)
        """
        n_paths, n_times = spot_paths.shape
        mtm_matrix = np.zeros((n_paths, n_times))
        for j, t in enumerate(time_grid):
            rem = max(self.maturity - t, 0.0)
            if rem < 1e-9:
                continue
            fwd_t = FXForward(
                notional_foreign=self.notional_foreign,
                forward_rate_agreed=self.forward_rate_agreed,
                maturity=rem,
                position=self.position,
            )
            for i in range(n_paths):
                mtm_matrix[i, j] = fwd_t.mtm(spot_paths[i, j], rd, rf)
        return mtm_matrix


# ─────────────────────────────────────────────────────────────────────────────
# FX Option — Garman-Kohlhagen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FXOption:
    """
    Vanilla European FX option (Garman-Kohlhagen, 1983).

    Parameters
    ----------
    notional_foreign : float
        Notional in the foreign currency.
    strike : float
        Strike price K (domestic per foreign).
    maturity : float
        Time to expiry in years.
    option_type : {"call", "put"}
        "call" → right to buy foreign (payer of domestic)
        "put"  → right to sell foreign
    position : {"long", "short"}
    sigma : float
        Implied volatility of the FX rate.
    """
    notional_foreign: float = 1_000_000.0
    strike: float = 1.10
    maturity: float = 1.0
    option_type: Literal["call", "put"] = "call"
    position: Literal["long", "short"] = "long"
    sigma: float = 0.10

    def _d1_d2(self, S: float, rd: float, rf: float, T: float):
        d1 = (np.log(S / self.strike) + (rd - rf + 0.5 * self.sigma**2) * T) / (
            self.sigma * np.sqrt(T)
        )
        d2 = d1 - self.sigma * np.sqrt(T)
        return d1, d2

    def price(self, S: float, rd: float, rf: float) -> float:
        """
        Garman-Kohlhagen option price.

        Call = S*exp(-rf*T)*N(d1) - K*exp(-rd*T)*N(d2)
        Put  = K*exp(-rd*T)*N(-d2) - S*exp(-rf*T)*N(-d1)
        """
        T = self.maturity
        if T < 1e-9:
            if self.option_type == "call":
                intrinsic = max(S - self.strike, 0.0)
            else:
                intrinsic = max(self.strike - S, 0.0)
            sign = 1.0 if self.position == "long" else -1.0
            return sign * self.notional_foreign * intrinsic

        d1, d2 = self._d1_d2(S, rd, rf, T)
        if self.option_type == "call":
            pv = S * np.exp(-rf * T) * norm.cdf(d1) - self.strike * np.exp(-rd * T) * norm.cdf(d2)
        else:
            pv = self.strike * np.exp(-rd * T) * norm.cdf(-d2) - S * np.exp(-rf * T) * norm.cdf(-d1)

        sign = 1.0 if self.position == "long" else -1.0
        return sign * self.notional_foreign * pv

    def delta(self, S: float, rd: float, rf: float) -> float:
        """Delta: ∂V/∂S * (1/notional). Positive for long call, negative for long put."""
        T = self.maturity
        if T < 1e-9:
            return 0.0
        d1, _ = self._d1_d2(S, rd, rf, T)
        sign = 1.0 if self.position == "long" else -1.0
        if self.option_type == "call":
            return sign * np.exp(-rf * T) * norm.cdf(d1)
        else:
            return sign * np.exp(-rf * T) * (norm.cdf(d1) - 1.0)

    def vega(self, S: float, rd: float, rf: float) -> float:
        """Vega: ∂V/∂σ (per 1% move in vol)."""
        T = self.maturity
        if T < 1e-9:
            return 0.0
        d1, _ = self._d1_d2(S, rd, rf, T)
        sign = 1.0 if self.position == "long" else -1.0
        return sign * self.notional_foreign * S * np.exp(-rf * T) * norm.pdf(d1) * np.sqrt(T) * 0.01

    def mtm_profile(
        self,
        spot_paths: np.ndarray,
        rd: float,
        rf: float,
        time_grid: np.ndarray,
        vol_paths: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        MtM along Monte Carlo spot paths (stochastic vol supported via vol_paths).

        Parameters
        ----------
        spot_paths : np.ndarray, shape (n_paths, n_times)
        vol_paths  : np.ndarray or None, shape (n_paths, n_times)
                     If None, use constant self.sigma.
        """
        n_paths, n_times = spot_paths.shape
        mtm_matrix = np.zeros((n_paths, n_times))
        for j, t in enumerate(time_grid):
            rem = max(self.maturity - t, 0.0)
            if rem < 1e-9:
                # At expiry: intrinsic only
                for i in range(n_paths):
                    S = spot_paths[i, j]
                    if self.option_type == "call":
                        val = max(S - self.strike, 0.0)
                    else:
                        val = max(self.strike - S, 0.0)
                    sign = 1.0 if self.position == "long" else -1.0
                    mtm_matrix[i, j] = sign * self.notional_foreign * val
                continue
            for i in range(n_paths):
                sigma_i = self.sigma if vol_paths is None else vol_paths[i, j]
                opt_t = FXOption(
                    notional_foreign=self.notional_foreign,
                    strike=self.strike,
                    maturity=rem,
                    option_type=self.option_type,
                    position=self.position,
                    sigma=sigma_i,
                )
                mtm_matrix[i, j] = opt_t.price(spot_paths[i, j], rd, rf)
        return mtm_matrix
