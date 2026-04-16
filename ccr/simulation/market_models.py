"""
market_models.py — Monte Carlo risk factor simulation
=======================================================
Implements three industry-standard stochastic processes for CCR exposure:

1. Hull-White 1-factor (HW1F)  — mean-reverting short rate (IRS)
2. Geometric Brownian Motion (GBM) — equity and FX spot (log-normal)
3. Cox-Ingersoll-Ross (CIR)    — credit spreads (non-negative, mean-reverting)

Correlated multi-factor simulation supported via Cholesky decomposition.

References
----------
- Hull & White (1990), JFQA — Pricing Interest Rate Derivative Securities
- Cox, Ingersoll & Ross (1985), Econometrica — A Theory of the Term Structure
- Glasserman (2003), Monte Carlo Methods in Financial Engineering — Ch.3
- Basel BCBS (2005), IMM guidelines — Annex 4
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# Simulation result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation output."""
    paths: np.ndarray          # shape (n_paths, n_times)
    time_grid: np.ndarray      # shape (n_times,)
    model: str
    params: dict = field(default_factory=dict)

    @property
    def n_paths(self) -> int:
        return self.paths.shape[0]

    @property
    def n_times(self) -> int:
        return self.paths.shape[1]

    def percentile(self, q: float) -> np.ndarray:
        """Percentile across paths at each time step."""
        return np.percentile(self.paths, q * 100, axis=0)

    def mean(self) -> np.ndarray:
        return self.paths.mean(axis=0)

    def std(self) -> np.ndarray:
        return self.paths.std(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hull-White 1-Factor (for interest rates)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HullWhite1F:
    """
    Hull-White one-factor model for the short rate.

    dr(t) = [θ(t) - a·r(t)] dt + σ dW(t)

    Under a flat initial yield curve: θ = a·r₀ (constant drift).

    Parameters
    ----------
    r0 : float
        Initial short rate.
    a : float
        Mean-reversion speed (typically 0.01–0.10 for rates).
    sigma : float
        Instantaneous volatility of the short rate.
    theta : float or None
        Long-run mean level (= a * r0 for flat curve calibration).

    The exact discretisation is used to avoid discretisation bias:
        r(t+dt) = r(t)*exp(-a*dt) + (θ/a)*(1-exp(-a*dt))
                  + σ*sqrt((1-exp(-2a*dt))/(2a)) * Z
    """
    r0: float = 0.03
    a: float   = 0.05
    sigma: float = 0.010
    theta: float | None = None   # if None, calibrated to flat curve

    def _theta(self) -> float:
        return self.theta if self.theta is not None else self.a * self.r0

    def simulate(
        self,
        n_paths: int = 5000,
        n_steps: int = 100,
        horizon: float = 10.0,
        seed: int | None = 42,
        antithetic: bool = True,
    ) -> SimulationResult:
        """
        Simulate short-rate paths using exact discretisation.

        Parameters
        ----------
        n_paths    : int    Number of Monte Carlo paths.
        n_steps    : int    Number of time steps.
        horizon    : float  Simulation horizon in years.
        antithetic : bool   Use antithetic variates for variance reduction.

        Returns
        -------
        SimulationResult
            paths shape (n_paths, n_steps+1), time_grid shape (n_steps+1,)
        """
        rng = np.random.default_rng(seed)
        dt = horizon / n_steps
        time_grid = np.linspace(0, horizon, n_steps + 1)

        theta = self._theta()
        e_adt = np.exp(-self.a * dt)
        mean_reversion_level = theta / self.a if self.a > 0 else self.r0
        vol_dt = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * dt)) / (2 * self.a))

        n_sim = n_paths // 2 if antithetic else n_paths
        Z = rng.standard_normal((n_sim, n_steps))
        if antithetic:
            Z = np.vstack([Z, -Z])

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.r0

        for t in range(n_steps):
            paths[:, t + 1] = (paths[:, t] * e_adt
                               + mean_reversion_level * (1 - e_adt)
                               + vol_dt * Z[:, t])

        return SimulationResult(
            paths=paths,
            time_grid=time_grid,
            model="HullWhite1F",
            params={"r0": self.r0, "a": self.a, "sigma": self.sigma,
                    "theta": theta, "n_paths": n_paths, "horizon": horizon},
        )

    def bond_price(self, r: float, t: float, T: float) -> float:
        """
        Zero-coupon bond price under HW1F:
        P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
        """
        tau = T - t
        if tau <= 0:
            return 1.0
        B = (1 - np.exp(-self.a * tau)) / self.a
        theta = self._theta()
        log_A = (theta / self.a - self.sigma**2 / (2 * self.a**2)) * (B - tau) - (
            self.sigma**2 * B**2 / (4 * self.a)
        )
        return np.exp(log_A - B * r)

    def forward_rate(self, r: float, t: float, T: float) -> float:
        """
        Par (swap) rate approximation from bond prices.
        Uses the zero rate implied by P(t,T).
        """
        P = self.bond_price(r, t, T)
        tau = T - t
        if tau < 1e-9 or P <= 0:
            return r
        return -np.log(P) / tau


# ─────────────────────────────────────────────────────────────────────────────
# 2. Geometric Brownian Motion (equity / FX)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GBM:
    """
    Geometric Brownian Motion for equity and FX simulation.

    dS(t) = μ·S(t)·dt + σ·S(t)·dW(t)

    Exact log-normal discretisation (Euler on log S):
        log S(t+dt) = log S(t) + (μ - σ²/2)·dt + σ·√dt·Z

    Parameters
    ----------
    S0    : float   Initial asset/spot price.
    mu    : float   Drift (rd - rf for FX, r - q for equity).
    sigma : float   Volatility.
    """
    S0: float    = 100.0
    mu: float    = 0.03
    sigma: float = 0.20

    def simulate(
        self,
        n_paths: int = 5000,
        n_steps: int = 100,
        horizon: float = 5.0,
        seed: int | None = 42,
        antithetic: bool = True,
    ) -> SimulationResult:
        """
        Simulate GBM paths with exact log-normal discretisation.
        """
        rng = np.random.default_rng(seed)
        dt = horizon / n_steps
        time_grid = np.linspace(0, horizon, n_steps + 1)

        n_sim = n_paths // 2 if antithetic else n_paths
        Z = rng.standard_normal((n_sim, n_steps))
        if antithetic:
            Z = np.vstack([Z, -Z])

        log_drift = (self.mu - 0.5 * self.sigma**2) * dt
        log_vol   = self.sigma * np.sqrt(dt)

        log_paths = np.zeros((n_paths, n_steps + 1))
        log_paths[:, 0] = np.log(self.S0)
        for t in range(n_steps):
            log_paths[:, t + 1] = log_paths[:, t] + log_drift + log_vol * Z[:, t]

        return SimulationResult(
            paths=np.exp(log_paths),
            time_grid=time_grid,
            model="GBM",
            params={"S0": self.S0, "mu": self.mu, "sigma": self.sigma,
                    "n_paths": n_paths, "horizon": horizon},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cox-Ingersoll-Ross (credit spreads)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CIR:
    """
    Cox-Ingersoll-Ross process for credit spreads (or rates).

    dX(t) = κ·(θ - X(t))·dt + σ·√X(t)·dW(t)

    Feller condition: 2κθ ≥ σ² → process stays strictly positive.
    Discretised with the Milstein scheme for better positivity control.

    Parameters
    ----------
    x0    : float   Initial spread level.
    kappa : float   Mean-reversion speed.
    theta : float   Long-run mean spread.
    sigma : float   Volatility coefficient.
    """
    x0: float    = 0.01        # initial spread (100 bps)
    kappa: float = 0.30        # mean-reversion speed
    theta: float = 0.01        # long-run mean
    sigma: float = 0.05        # vol

    @property
    def feller_satisfied(self) -> bool:
        return 2 * self.kappa * self.theta >= self.sigma**2

    def simulate(
        self,
        n_paths: int = 5000,
        n_steps: int = 100,
        horizon: float = 5.0,
        seed: int | None = 42,
        antithetic: bool = True,
    ) -> SimulationResult:
        """
        CIR simulation using the truncated Euler (full truncation) scheme.
        Guarantees positivity even when Feller condition is not met.
        """
        rng = np.random.default_rng(seed)
        dt = horizon / n_steps
        time_grid = np.linspace(0, horizon, n_steps + 1)

        n_sim = n_paths // 2 if antithetic else n_paths
        Z = rng.standard_normal((n_sim, n_steps))
        if antithetic:
            Z = np.vstack([Z, -Z])

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.x0

        for t in range(n_steps):
            x = np.maximum(paths[:, t], 0.0)    # full truncation
            drift = self.kappa * (self.theta - x) * dt
            diffusion = self.sigma * np.sqrt(x * dt) * Z[:, t]
            paths[:, t + 1] = np.maximum(x + drift + diffusion, 0.0)

        return SimulationResult(
            paths=paths,
            time_grid=time_grid,
            model="CIR",
            params={"x0": self.x0, "kappa": self.kappa, "theta": self.theta,
                    "sigma": self.sigma, "feller": self.feller_satisfied},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Correlated multi-factor engine
# ─────────────────────────────────────────────────────────────────────────────

def simulate_correlated(
    models: list,
    correlation_matrix: np.ndarray,
    n_paths: int = 5000,
    n_steps: int = 100,
    horizon: float = 5.0,
    seed: int | None = 42,
) -> list[SimulationResult]:
    """
    Simulate multiple correlated risk factor paths.

    Applies a Cholesky decomposition to the correlation matrix to generate
    correlated Brownian motions across all factors simultaneously.

    Parameters
    ----------
    models : list
        List of model instances (HullWhite1F, GBM, or CIR).
        Each must have a .sigma attribute and support path building.
    correlation_matrix : np.ndarray
        (n_factors × n_factors) correlation matrix, must be positive semi-definite.
    n_paths, n_steps, horizon : standard simulation parameters.

    Returns
    -------
    list[SimulationResult]
        One SimulationResult per model, with correlated paths.

    Notes
    -----
    Correlation is introduced at the Brownian motion level.
    Each model's specific dynamics (drift, mean-reversion) are preserved.
    """
    n_factors = len(models)
    assert correlation_matrix.shape == (n_factors, n_factors), "Correlation matrix shape mismatch"

    rng = np.random.default_rng(seed)
    dt = horizon / n_steps
    time_grid = np.linspace(0, horizon, n_steps + 1)

    # Cholesky decomposition: C = L @ L.T
    try:
        L = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # Regularise if not PD
        eps = 1e-8
        L = np.linalg.cholesky(correlation_matrix + eps * np.eye(n_factors))

    # Generate independent standard normals: shape (n_paths, n_steps, n_factors)
    Z_indep = rng.standard_normal((n_paths, n_steps, n_factors))
    # Correlate: Z_corr[i,t,:] = L @ Z_indep[i,t,:]
    Z_corr = Z_indep @ L.T     # (n_paths, n_steps, n_factors)

    # Build paths for each model using its correlated Brownian driver
    results = []
    for k, model in enumerate(models):
        paths = np.zeros((n_paths, n_steps + 1))
        Z_k = Z_corr[:, :, k]   # (n_paths, n_steps)

        if isinstance(model, HullWhite1F):
            e_adt = np.exp(-model.a * dt)
            mr_level = model._theta() / model.a if model.a > 0 else model.r0
            vol_dt = model.sigma * np.sqrt((1 - np.exp(-2 * model.a * dt)) / (2 * model.a))
            paths[:, 0] = model.r0
            for t in range(n_steps):
                paths[:, t + 1] = (paths[:, t] * e_adt
                                   + mr_level * (1 - e_adt)
                                   + vol_dt * Z_k[:, t])

        elif isinstance(model, GBM):
            log_drift = (model.mu - 0.5 * model.sigma**2) * dt
            log_vol   = model.sigma * np.sqrt(dt)
            log_paths = np.zeros((n_paths, n_steps + 1))
            log_paths[:, 0] = np.log(model.S0)
            for t in range(n_steps):
                log_paths[:, t + 1] = log_paths[:, t] + log_drift + log_vol * Z_k[:, t]
            paths = np.exp(log_paths)

        elif isinstance(model, CIR):
            paths[:, 0] = model.x0
            for t in range(n_steps):
                x = np.maximum(paths[:, t], 0.0)
                drift     = model.kappa * (model.theta - x) * dt
                diffusion = model.sigma * np.sqrt(x * dt) * Z_k[:, t]
                paths[:, t + 1] = np.maximum(x + drift + diffusion, 0.0)

        else:
            raise TypeError(f"Unsupported model type: {type(model)}")

        results.append(SimulationResult(
            paths=paths,
            time_grid=time_grid,
            model=type(model).__name__,
            params={"factor_index": k},
        ))

    return results
