"""
exposure_metrics.py — CCR Exposure Metrics
===========================================
Computes the standard CCR exposure profile from a matrix of MtM values
(Monte Carlo paths × time steps).

Metrics implemented
-------------------
EE(t)   — Expected (Positive) Exposure: E[max(V(t), 0)]
ENE(t)  — Expected Negative Exposure:   E[max(-V(t), 0)]  (used for DVA)
EPE     — Expected Positive Exposure (time-averaged EE over [0,T])
EEPE    — Effective EPE (capped maximum of EPE profile, IMM capital metric)
PFE(t)  — Potential Future Exposure at quantile α
MaxPFE  — Maximum PFE over the simulation horizon
CESM    — Collateral-adjusted exposure (with VM and IM)

Basel IMM definitions (BCBS 2005 / CRR3)
-----------------------------------------
EEPE = (1/year) * integral_0^1year  max_{s≤t} EPE(s)  dt

The EEPE is the key input to the IMM capital calculation:
  EAD = α * EEPE   (α = 1.4 regulatory multiplier)

References
----------
- BCBS (2005), Annex 4 — Treatment of CCR
- BCBS (2014), SA-CCR
- Gregory (2015), The xVA Challenge — Ch.6, 7
- Pykhtin & Zhu (2007), A Guide to Modelling Counterparty Credit Risk
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# Netting set parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NettingSet:
    """
    A netting set groups trades under a single legal agreement (ISDA Master
    Agreement) such that in the event of default, the net MtM is the exposure —
    not the sum of positive MtMs.

    Under netting:
        V_net(t) = sum_i V_i(t)   (signed sum across all trades)
    Without netting:
        Exposure = sum_i max(V_i(t), 0)   (gross positive exposure)

    Parameters
    ----------
    name : str
    trades : list of MtM matrices, each shape (n_paths, n_times)
    apply_netting : bool
        If True, exposure = max(sum(V_i), 0).
        If False, exposure = sum(max(V_i, 0)).
    """
    name: str = "NettingSet_001"
    trades: list = field(default_factory=list)   # list of np.ndarray (n_paths, n_times)
    apply_netting: bool = True

    def net_mtm(self) -> np.ndarray:
        """
        Net MtM matrix (signed sum if netting, else gross).
        Returns shape (n_paths, n_times).
        """
        if not self.trades:
            raise ValueError("No trades in netting set")
        mtm_stack = np.stack(self.trades, axis=0)   # (n_trades, n_paths, n_times)
        if self.apply_netting:
            return mtm_stack.sum(axis=0)             # (n_paths, n_times)
        else:
            return mtm_stack.sum(axis=0)             # same: netting = sum, we apply max later

    def netting_benefit(self) -> np.ndarray:
        """
        Netting benefit at each (path, time):
        = gross exposure - net exposure
        Always ≥ 0.
        """
        mtm_stack = np.stack(self.trades, axis=0)
        gross = np.maximum(mtm_stack, 0.0).sum(axis=0)
        net   = np.maximum(mtm_stack.sum(axis=0), 0.0)
        return gross - net


# ─────────────────────────────────────────────────────────────────────────────
# Exposure profile computation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExposureProfile:
    """
    Exposure profile computed from a MtM matrix.

    Parameters
    ----------
    mtm_matrix : np.ndarray, shape (n_paths, n_times)
        Mark-to-market values. Positive = in-the-money (counterparty owes us).
    time_grid  : np.ndarray, shape (n_times,)
        Simulation time points in years.
    pfe_quantile : float
        Quantile for PFE computation (typically 0.95 or 0.99).
    """
    mtm_matrix: np.ndarray
    time_grid: np.ndarray
    pfe_quantile: float = 0.95

    def __post_init__(self):
        assert self.mtm_matrix.shape[1] == len(self.time_grid), (
            f"MtM shape {self.mtm_matrix.shape} inconsistent with time_grid len {len(self.time_grid)}"
        )

    @property
    def n_paths(self) -> int:
        return self.mtm_matrix.shape[0]

    @property
    def n_times(self) -> int:
        return self.mtm_matrix.shape[1]

    # ── Core metrics ─────────────────────────────────────────────────────────

    def ee(self) -> np.ndarray:
        """
        EE(t) = E[max(V(t), 0)]
        Average positive exposure across paths at each time step.
        """
        return np.maximum(self.mtm_matrix, 0.0).mean(axis=0)

    def ene(self) -> np.ndarray:
        """
        ENE(t) = E[max(-V(t), 0)]
        Expected negative exposure — used in DVA calculation.
        """
        return np.maximum(-self.mtm_matrix, 0.0).mean(axis=0)

    def pfe(self, quantile: float | None = None) -> np.ndarray:
        """
        PFE(t) = quantile_α of max(V(t), 0)
        The α-quantile of the positive exposure distribution at each t.
        Represents the worst-case exposure with probability α.
        """
        q = quantile or self.pfe_quantile
        return np.percentile(np.maximum(self.mtm_matrix, 0.0), q * 100, axis=0)

    def epe(self, horizon: float | None = None) -> float:
        """
        EPE = (1/T) * integral_0^T EE(t) dt
        Time-averaged EE over [0, T]. Trapezoidal integration.

        Parameters
        ----------
        horizon : float or None
            Integration horizon. If None, uses full time_grid.
        """
        ee_profile = self.ee()
        t = self.time_grid
        if horizon is not None:
            mask = t <= horizon + 1e-9
            ee_profile = ee_profile[mask]
            t = t[mask]
        T = t[-1] - t[0]
        if T < 1e-9:
            return float(ee_profile[0])
        return float(np.trapz(ee_profile, t) / T)

    def eepe(self, capital_horizon: float = 1.0) -> float:
        """
        EEPE — Effective Expected Positive Exposure (Basel IMM capital metric).

        EEPE = (1/T*) * integral_0^T* EEPEE(t) dt
        where EEPEE(t) = max_{s ≤ t, s ≤ T*} EPE(s)   (running maximum of EPE)

        More precisely, under BCBS Annex 4 (discrete approximation):
        EEPE = sum_k  EEE(t_k) * Δt_k / T*

        where EEE(t_k) = max(EEE(t_{k-1}), EE(t_k))  — effective EE (non-decreasing)
        and T* = capital_horizon (typically 1 year).

        Parameters
        ----------
        capital_horizon : float
            Typically 1 year per Basel. Can be shorter for short-dated portfolios.
        """
        ee_profile = self.ee()
        t = self.time_grid
        T_star = min(capital_horizon, t[-1])
        mask = t <= T_star + 1e-9
        ee_sub = ee_profile[mask]
        t_sub  = t[mask]

        # Effective EE: running maximum (EEPE must be non-decreasing)
        eee = np.maximum.accumulate(ee_sub)

        # Discrete integration
        dt = np.diff(t_sub)
        eepe_val = np.sum(eee[1:] * dt) / T_star if T_star > 0 else eee[0]
        return float(eepe_val)

    def ead_imm(self, alpha: float = 1.4) -> float:
        """
        IMM Exposure at Default (EAD):
        EAD = α * EEPE
        α = 1.4 regulatory multiplier (Basel III default, may be reduced by supervisor).
        """
        return alpha * self.eepe()

    def max_pfe(self, quantile: float | None = None) -> float:
        """Maximum PFE over the simulation horizon."""
        return float(self.pfe(quantile).max())

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame with EE, ENE, PFE at each time step.
        """
        return pd.DataFrame({
            "time":  self.time_grid,
            "EE":    self.ee(),
            "ENE":   self.ene(),
            f"PFE_{self.pfe_quantile:.0%}": self.pfe(),
        }).set_index("time")

    def scalar_metrics(self, capital_horizon: float = 1.0) -> dict:
        """
        Key scalar metrics for regulatory reporting.
        """
        return {
            "EPE":     self.epe(capital_horizon),
            "EEPE":    self.eepe(capital_horizon),
            "EAD_IMM": self.ead_imm(),
            "MaxPFE":  self.max_pfe(),
            "MaxEE":   float(self.ee().max()),
            "n_paths": self.n_paths,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collateral-adjusted exposure
# ─────────────────────────────────────────────────────────────────────────────

def collateral_adjusted_exposure(
    mtm_matrix: np.ndarray,
    collateral_matrix: np.ndarray,
) -> np.ndarray:
    """
    Collateral-adjusted MtM for exposure computation.

    Credit exposure at time t after collateral:
    E_adj(t) = max(V(t) - C(t), 0)

    where C(t) = total collateral held (VM + IM from counterparty).
    Negative C = we have posted collateral to counterparty.

    Parameters
    ----------
    mtm_matrix        : np.ndarray, shape (n_paths, n_times) — trade MtM
    collateral_matrix : np.ndarray, shape (n_paths, n_times) — collateral held

    Returns
    -------
    np.ndarray, shape (n_paths, n_times) — adjusted MtM
    """
    return mtm_matrix - collateral_matrix
