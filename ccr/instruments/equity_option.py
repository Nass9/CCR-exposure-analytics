"""
equity_option.py — European/American equity option pricing
============================================================
Black-Scholes model for European options with full Greeks suite.
American options priced via binomial tree (Cox-Ross-Rubinstein).

References
----------
- Black & Scholes (1973), JPE — The Pricing of Options and Corporate Liabilities
- Cox, Ross & Rubinstein (1979), JFE — Option Pricing: A Simplified Approach
- Hull (2022), Options, Futures, and Other Derivatives — Ch.15, 21
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal
from scipy.stats import norm


@dataclass
class EquityOption:
    """
    Vanilla equity option (European or American).

    Parameters
    ----------
    notional : float
        Number of contracts × contract size.
    spot : float
        Current underlying price S₀.
    strike : float
        Strike price K.
    maturity : float
        Time to expiry T in years.
    sigma : float
        Implied (or historical) volatility.
    rate : float
        Continuously compounded risk-free rate.
    dividend_yield : float
        Continuous dividend yield q.
    option_type : {"call", "put"}
    style : {"european", "american"}
    position : {"long", "short"}
    """
    notional: float = 100.0
    spot: float = 100.0
    strike: float = 100.0
    maturity: float = 1.0
    sigma: float = 0.20
    rate: float = 0.03
    dividend_yield: float = 0.0
    option_type: Literal["call", "put"] = "call"
    style: Literal["european", "american"] = "european"
    position: Literal["long", "short"] = "long"

    def _d1_d2(self, S: float | None = None, T: float | None = None):
        S = S or self.spot
        T = T or self.maturity
        d1 = (np.log(S / self.strike) + (self.rate - self.dividend_yield + 0.5 * self.sigma**2) * T) / (
            self.sigma * np.sqrt(T)
        )
        d2 = d1 - self.sigma * np.sqrt(T)
        return d1, d2

    def bs_price(self, S: float | None = None, T: float | None = None) -> float:
        """Black-Scholes price for European option."""
        S = S if S is not None else self.spot
        T = T if T is not None else self.maturity
        if T < 1e-9:
            if self.option_type == "call":
                return max(S - self.strike, 0.0)
            return max(self.strike - S, 0.0)
        d1, d2 = self._d1_d2(S, T)
        if self.option_type == "call":
            pv = (S * np.exp(-self.dividend_yield * T) * norm.cdf(d1)
                  - self.strike * np.exp(-self.rate * T) * norm.cdf(d2))
        else:
            pv = (self.strike * np.exp(-self.rate * T) * norm.cdf(-d2)
                  - S * np.exp(-self.dividend_yield * T) * norm.cdf(-d1))
        return pv

    def american_price(self, S: float | None = None, T: float | None = None, n_steps: int = 200) -> float:
        """
        CRR binomial tree for American options.
        Supports early exercise for puts (and calls with dividends).
        """
        S = S if S is not None else self.spot
        T = T if T is not None else self.maturity
        if T < 1e-9:
            if self.option_type == "call":
                return max(S - self.strike, 0.0)
            return max(self.strike - S, 0.0)

        dt = T / n_steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1.0 / u
        disc = np.exp(-self.rate * dt)
        p = (np.exp((self.rate - self.dividend_yield) * dt) - d) / (u - d)

        # Terminal nodes
        S_T = S * (u ** np.arange(n_steps, -1, -1)) * (d ** np.arange(0, n_steps + 1))
        if self.option_type == "call":
            V = np.maximum(S_T - self.strike, 0.0)
        else:
            V = np.maximum(self.strike - S_T, 0.0)

        # Backward induction
        for _ in range(n_steps):
            V = disc * (p * V[:-1] + (1 - p) * V[1:])
            S_T = S_T[:-1] / d
            if self.option_type == "call":
                intrinsic = np.maximum(S_T - self.strike, 0.0)
            else:
                intrinsic = np.maximum(self.strike - S_T, 0.0)
            V = np.maximum(V, intrinsic)

        return V[0]

    def price(self, S: float | None = None, T: float | None = None) -> float:
        """Price using BS (European) or binomial (American)."""
        if self.style == "european":
            raw = self.bs_price(S, T)
        else:
            raw = self.american_price(S, T)
        sign = 1.0 if self.position == "long" else -1.0
        return sign * self.notional * raw

    # ── Greeks (Black-Scholes, European) ────────────────────────────────────

    def delta(self) -> float:
        """∂V/∂S — hedge ratio. Long call: (0,1]; long put: [-1,0)."""
        d1, _ = self._d1_d2()
        T = self.maturity
        sign = 1.0 if self.position == "long" else -1.0
        if self.option_type == "call":
            return sign * np.exp(-self.dividend_yield * T) * norm.cdf(d1)
        else:
            return sign * np.exp(-self.dividend_yield * T) * (norm.cdf(d1) - 1.0)

    def gamma(self) -> float:
        """∂²V/∂S² — curvature. Always positive for long options."""
        d1, _ = self._d1_d2()
        T = self.maturity
        sign = 1.0 if self.position == "long" else -1.0
        return sign * (np.exp(-self.dividend_yield * T) * norm.pdf(d1)) / (
            self.spot * self.sigma * np.sqrt(T)
        )

    def vega(self) -> float:
        """∂V/∂σ per 1% move in implied vol."""
        d1, _ = self._d1_d2()
        T = self.maturity
        sign = 1.0 if self.position == "long" else -1.0
        return sign * self.notional * self.spot * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * np.sqrt(T) * 0.01

    def theta(self) -> float:
        """∂V/∂t per calendar day (time decay). Negative for long options."""
        d1, d2 = self._d1_d2()
        S, K, T = self.spot, self.strike, self.maturity
        r, q, sig = self.rate, self.dividend_yield, self.sigma
        sign = 1.0 if self.position == "long" else -1.0
        if self.option_type == "call":
            th = (-S * np.exp(-q * T) * norm.pdf(d1) * sig / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2)
                  + q * S * np.exp(-q * T) * norm.cdf(d1))
        else:
            th = (-S * np.exp(-q * T) * norm.pdf(d1) * sig / (2 * np.sqrt(T))
                  + r * K * np.exp(-r * T) * norm.cdf(-d2)
                  - q * S * np.exp(-q * T) * norm.cdf(-d1))
        return sign * self.notional * th / 365.0   # per calendar day

    def rho(self) -> float:
        """∂V/∂r per 1% move in interest rate."""
        _, d2 = self._d1_d2()
        T, K = self.maturity, self.strike
        sign = 1.0 if self.position == "long" else -1.0
        if self.option_type == "call":
            return sign * self.notional * K * T * np.exp(-self.rate * T) * norm.cdf(d2) * 0.01
        else:
            return -sign * self.notional * K * T * np.exp(-self.rate * T) * norm.cdf(-d2) * 0.01

    def greeks(self) -> dict:
        """Return all Greeks as a dictionary."""
        return {
            "price":  self.bs_price(),
            "delta":  self.delta(),
            "gamma":  self.gamma(),
            "vega":   self.vega(),
            "theta":  self.theta(),
            "rho":    self.rho(),
        }

    def mtm_profile(
        self,
        spot_paths: np.ndarray,
        time_grid: np.ndarray,
        vol_paths: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        MtM along Monte Carlo paths.

        Parameters
        ----------
        spot_paths : np.ndarray, shape (n_paths, n_times)
        vol_paths  : np.ndarray or None — stochastic vol (shape (n_paths, n_times))
        """
        n_paths, n_times = spot_paths.shape
        mtm_matrix = np.zeros((n_paths, n_times))
        for j, t in enumerate(time_grid):
            rem = max(self.maturity - t, 0.0)
            for i in range(n_paths):
                sigma_i = self.sigma if vol_paths is None else max(vol_paths[i, j], 0.01)
                opt_t = EquityOption(
                    notional=self.notional,
                    spot=spot_paths[i, j],
                    strike=self.strike,
                    maturity=rem,
                    sigma=sigma_i,
                    rate=self.rate,
                    dividend_yield=self.dividend_yield,
                    option_type=self.option_type,
                    style=self.style,
                    position=self.position,
                )
                mtm_matrix[i, j] = opt_t.price()
        return mtm_matrix
