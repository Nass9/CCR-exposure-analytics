"""
collateral_manager.py — Collateral Management: VM, IM, MPOR
=============================================================
Simulates the collateral dynamics of a collateralised portfolio under a CSA
(Credit Support Annex). Implements:

  - Variation Margin (VM): daily mark-to-market exchange
  - Initial Margin (IM):   protection against future MtM changes during MPOR
  - MPOR (Margin Period of Risk): exposure window between last margin call and
    close-out / replacement of the portfolio
  - Threshold (TH): uncollateralised exposure allowed before margin call
  - Minimum Transfer Amount (MTA): minimum margin call size (operational)
  - Independent Amount (IA): upfront IM-like collateral

Framework
---------
Without collateral:
    Exposure(t) = max(V(t), 0)

With CSA (VM only):
    Exposure(t) = max(V(t) - C_VM(t), 0)
    ≈ max(V(t) - V(t - MPOR), 0)  for zero threshold/MTA

With CSA (VM + IM):
    Exposure(t) ≤ max(ΔV_MPOR - IM, 0)    (best case — IM absorbs the move)

MPOR values (Basel CRR3 / BCBS 2014 SA-CCR)
--------------------------------------------
| Agreement type              | MPOR (min) |
|-----------------------------|------------|
| OTC bilateral, liquid       | 10 days    |
| OTC bilateral, illiquid     | 20 days    |
| Cleared (CCP)               |  5 days    |
| Non-CSA (unsecured)         | Trade life |
| Disputed / SLA breach ≥20d  | 2 × normal |

References
----------
- Basel BCBS (2013), CRE52 — Margin Period of Risk
- Gregory (2015), The xVA Challenge — Ch.7: Collateral
- ISDA (2021), SIMM Methodology v2.5
- EBA (2014), RTS on risk mitigation techniques for OTC derivatives
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# CSA parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CSAParameters:
    """
    Credit Support Annex (CSA) parameters governing collateral exchange.

    Parameters
    ----------
    threshold_we : float
        Threshold for the bank (we must receive collateral above this MtM).
        If MtM > threshold_we: counterparty must post.
    threshold_cp : float
        Threshold for the counterparty (they receive collateral above this MtM).
        If MtM < -threshold_cp: bank must post.
    mta_we : float
        Minimum Transfer Amount (bank receives). Below this, no call is made.
    mta_cp : float
        Minimum Transfer Amount (bank posts). Below this, no post is made.
    independent_amount : float
        Upfront Independent Amount posted by counterparty (always held).
        Can represent a floor to the collateral held.
    rounding : float
        Rounding unit for collateral calls (e.g. 10,000 means round to nearest 10k).
    rehypothecation : bool
        Whether posted collateral can be re-used (affects funding cost — not in scope here).
    mpor_days : int
        Margin Period of Risk in business days.
    collateral_type : {"cash", "securities"}
        "cash" → immediate settlement, no haircut.
        "securities" → subject to haircut.
    haircut : float
        Haircut applied to securities collateral (0 = no haircut for cash).
    """
    threshold_we: float = 0.0           # zero threshold = full collateralisation
    threshold_cp: float = 0.0
    mta_we: float = 0.0                 # zero MTA = every margin call made
    mta_cp: float = 0.0
    independent_amount: float = 0.0
    rounding: float = 0.0
    rehypothecation: bool = True
    mpor_days: int = 10                 # Basel standard: 10 business days bilateral OTC
    collateral_type: Literal["cash", "securities"] = "cash"
    haircut: float = 0.0

    @property
    def mpor_years(self) -> float:
        return self.mpor_days / 252.0

    @classmethod
    def zero_threshold_bilateral(cls, mpor_days: int = 10) -> "CSAParameters":
        """Standard bilateral OTC: zero threshold, zero MTA, 10-day MPOR."""
        return cls(threshold_we=0.0, threshold_cp=0.0, mta_we=0.0, mta_cp=0.0,
                   mpor_days=mpor_days)

    @classmethod
    def cleared_ccp(cls) -> "CSAParameters":
        """CCP-cleared: zero threshold, 5-day MPOR (EMIR/Dodd-Frank)."""
        return cls(threshold_we=0.0, threshold_cp=0.0, mta_we=0.0, mta_cp=0.0,
                   mpor_days=5)

    @classmethod
    def unsecured(cls, trade_maturity_years: float = 5.0) -> "CSAParameters":
        """No CSA: threshold = infinity, MPOR = trade life."""
        return cls(threshold_we=1e18, threshold_cp=1e18, mpor_days=int(trade_maturity_years * 252))

    @classmethod
    def institutional_bilateral(
        cls,
        threshold: float = 500_000.0,
        mta: float = 100_000.0,
        ia: float = 0.0,
        mpor_days: int = 10,
    ) -> "CSAParameters":
        """Typical institutional CSA with non-zero threshold and MTA."""
        return cls(
            threshold_we=threshold, threshold_cp=threshold,
            mta_we=mta, mta_cp=mta,
            independent_amount=ia,
            mpor_days=mpor_days,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Variation Margin dynamics
# ─────────────────────────────────────────────────────────────────────────────

class VMEngine:
    """
    Simulates Variation Margin (VM) dynamics along MtM paths.

    VM represents the daily (or periodic) settlement of MtM changes.
    Under a fully collateralised CSA (zero threshold, zero MTA):
        C_VM(t) ≈ V(t - MPOR)
    i.e. the collateral reflects the portfolio value from MPOR days ago.

    Under a CSA with threshold and MTA:
        margin_call(t) = max(V(t) - TH - C_held(t), MTA) if V(t) > TH + C_held
        The call is rounded to the nearest rounding unit if applicable.
    """

    def __init__(self, csa: CSAParameters):
        self.csa = csa

    def _apply_mta_rounding(self, call_amount: float) -> float:
        """Apply MTA and rounding conventions to a raw margin call amount."""
        if abs(call_amount) < self.csa.mta_we:
            return 0.0
        if self.csa.rounding > 0:
            call_amount = round(call_amount / self.csa.rounding) * self.csa.rounding
        return call_amount

    def compute_vm_matrix(
        self,
        mtm_matrix: np.ndarray,
        time_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate VM holdings along all paths.

        The VM held at time t reflects the margin call made at t - MPOR.
        Under zero threshold/MTA: C_VM(t) = V(t - MPOR).
        Under threshold/MTA: C_VM(t) = max(V(t-MPOR) - TH, 0) rounded.

        Parameters
        ----------
        mtm_matrix : np.ndarray, shape (n_paths, n_times)
        time_grid  : np.ndarray, shape (n_times,)

        Returns
        -------
        np.ndarray, shape (n_paths, n_times) — VM collateral held by bank
        """
        n_paths, n_times = mtm_matrix.shape
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0 / 252
        mpor_steps = max(1, int(self.csa.mpor_days / (dt * 252)))

        vm = np.zeros((n_paths, n_times))

        for t_idx in range(mpor_steps, n_times):
            # Last known MtM: MPOR steps ago (before dispute / close-out)
            v_last_known = mtm_matrix[:, t_idx - mpor_steps]

            # Collateral owed to bank = max(V - TH, 0) - what CP already posted
            call_raw = v_last_known - self.csa.threshold_we
            # With MTA: only call if call_raw >= MTA
            call_net = np.where(
                call_raw >= self.csa.mta_we,
                call_raw,
                0.0,
            )
            # Independent Amount is always held (floor)
            vm[:, t_idx] = np.maximum(call_net + self.csa.independent_amount, 0.0)

        return vm

    def collateral_adjusted_exposure(
        self,
        mtm_matrix: np.ndarray,
        time_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Compute exposure after VM collateral.
        E(t) = max(V(t) - C_VM(t), 0)

        The residual exposure reflects the gap between current MtM and
        stale collateral from MPOR days ago.
        """
        vm = self.compute_vm_matrix(mtm_matrix, time_grid)
        return np.maximum(mtm_matrix - vm, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Initial Margin
# ─────────────────────────────────────────────────────────────────────────────

class IMEngine:
    """
    Initial Margin (IM) computation and dynamic simulation.

    IM protects against the change in portfolio MtM during the MPOR —
    i.e. the gap between last collateral exchange and close-out.

    Two IM methodologies are implemented:

    1. ISDA-SIMM (simplified) — sensitivity-based, regulatory IM
       IM_SIMM ≈ σ_portfolio × √MPOR × z_α  (simplified 1-factor)

    2. VaR-based IM — internal model approach
       IM_VaR(t) = VaR_α(ΔV over MPOR | information at t)

    Under UMR (Uncleared Margin Rules, 2016–2022):
    - Phase 1-6 entities must exchange IM for non-cleared derivatives
    - IM must be held in segregated accounts (no rehypothecation)
    - IM reduces exposure by reducing the MPOR gap to max(ΔV - IM, 0)
    """

    def __init__(self, csa: CSAParameters, confidence_level: float = 0.99):
        self.csa = csa
        self.cl = confidence_level

    def simm_im_simplified(
        self,
        portfolio_dv01: float,
        rate_vol: float = 0.005,
    ) -> float:
        """
        Simplified ISDA-SIMM IM for a linear rates portfolio.

        IM ≈ DV01 × σ_rates × √MPOR × z_α

        where σ_rates is the daily rate volatility and z_α is the quantile
        at the given confidence level.

        Parameters
        ----------
        portfolio_dv01 : float   Dollar duration (DV01) of the portfolio.
        rate_vol       : float   Daily volatility of rates (e.g. 50 bps / √252 ≈ 0.003).
        """
        from scipy.stats import norm
        z = norm.ppf(self.cl)
        mpor_factor = np.sqrt(self.csa.mpor_days)
        return abs(portfolio_dv01) * rate_vol * mpor_factor * z

    def var_based_im(
        self,
        mtm_matrix: np.ndarray,
        time_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Rolling VaR-based IM using the empirical distribution of MPOR P&L.

        IM(t) = VaR_α(V(t + MPOR) - V(t))
        Computed from the cross-sectional distribution across paths.

        Parameters
        ----------
        mtm_matrix : np.ndarray, shape (n_paths, n_times)

        Returns
        -------
        np.ndarray, shape (n_paths, n_times)
        """
        n_paths, n_times = mtm_matrix.shape
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0 / 252
        mpor_steps = max(1, int(self.csa.mpor_days / (dt * 252)))
        alpha = 1.0 - self.cl
        im_matrix = np.zeros((n_paths, n_times))

        for t_idx in range(n_times - mpor_steps):
            # MPOR P&L: V(t+MPOR) - V(t), cross-sectional
            delta_v = mtm_matrix[:, t_idx + mpor_steps] - mtm_matrix[:, t_idx]
            # VaR at confidence level (loss tail)
            var_level = -np.percentile(delta_v, alpha * 100)
            im_matrix[:, t_idx] = np.maximum(var_level, 0.0)

        return im_matrix

    def exposure_with_im(
        self,
        mtm_matrix: np.ndarray,
        time_grid: np.ndarray,
        vm_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Exposure after VM and IM.

        E(t) = max(V(t) - C_VM(t) - IM_posted_by_CP(t), 0)

        With IM, residual exposure ≈ max(ΔV_MPOR - IM, 0)
        → for a well-sized IM, exposure is near zero even during MPOR.

        Parameters
        ----------
        vm_matrix : np.ndarray or None
            VM collateral held. If None, uses VMEngine with CSA parameters.
        """
        if vm_matrix is None:
            vm_engine = VMEngine(self.csa)
            vm_matrix = vm_engine.compute_vm_matrix(mtm_matrix, time_grid)

        im_matrix = self.var_based_im(mtm_matrix, time_grid)
        total_collateral = vm_matrix + im_matrix
        return np.maximum(mtm_matrix - total_collateral, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# MPOR scenario analysis
# ─────────────────────────────────────────────────────────────────────────────

class MPORAnalyser:
    """
    Analyses the impact of MPOR on exposure and the cost of collateral disputes.

    The MPOR gap is the core driver of collateralised CCR exposure:
    - Under zero MPOR: exposure ≈ 0 (perfect collateralisation)
    - Under MPOR > 0:  exposure = max(V(t) - V(t - MPOR), 0) ≈ ΔV over MPOR

    Basel prescribes MPOR floors to capture this gap:
    - 10 days: standard bilateral OTC
    - 20 days: illiquid portfolios / large netting sets (>5000 trades)
    - 5 days: cleared derivatives

    Dispute clause (CRR3 Article 285(4)):
    If margin disputes exceed 20 business days in the past two quarters,
    the MPOR is doubled.
    """

    @staticmethod
    def mpor_exposure_profile(
        mtm_matrix: np.ndarray,
        time_grid: np.ndarray,
        mpor_days: int,
    ) -> np.ndarray:
        """
        Compute the MPOR gap exposure at each time step.

        MPOR_exposure(t) = max(V(t) - V(t - MPOR), 0)

        This is the residual exposure after full VM but before IM.
        Intuitively: the loss that could occur if counterparty defaults
        during the margin period.

        Parameters
        ----------
        mtm_matrix : np.ndarray, shape (n_paths, n_times)
        mpor_days  : int   Margin Period of Risk in business days.
        """
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0 / 252
        mpor_steps = max(1, int(mpor_days / (dt * 252)))
        n_paths, n_times = mtm_matrix.shape
        exposure = np.zeros((n_paths, n_times))

        for t_idx in range(mpor_steps, n_times):
            delta_v = mtm_matrix[:, t_idx] - mtm_matrix[:, t_idx - mpor_steps]
            exposure[:, t_idx] = np.maximum(delta_v, 0.0)

        return exposure

    @staticmethod
    def compare_mpor_scenarios(
        mtm_matrix: np.ndarray,
        time_grid: np.ndarray,
        mpor_scenarios: list[int] = [0, 5, 10, 20],
    ) -> pd.DataFrame:
        """
        Compare EE and EEPE across different MPOR scenarios.

        Useful for:
        - Quantifying the benefit of central clearing (MPOR 5 vs 10 days)
        - Impact of dispute clause (MPOR doubling)
        - Sensitivity of IMM capital to MPOR assumption

        Returns
        -------
        pd.DataFrame with columns: MPOR_days, EE_mean, EE_max, EEPE
        """
        from ..exposure.exposure_metrics import ExposureProfile
        rows = []
        for mpor in mpor_scenarios:
            exp_mtm = MPORAnalyser.mpor_exposure_profile(mtm_matrix, time_grid, mpor)
            profile = ExposureProfile(exp_mtm, time_grid)
            ee = profile.ee()
            rows.append({
                "MPOR_days":  mpor,
                "EE_mean":    float(ee.mean()),
                "EE_max":     float(ee.max()),
                "EEPE":       profile.eepe(),
                "EAD_IMM":    profile.ead_imm(),
                "MaxPFE_95%": float(profile.pfe(0.95).max()),
            })
        return pd.DataFrame(rows).set_index("MPOR_days")

    @staticmethod
    def dispute_scenario(
        mtm_matrix: np.ndarray,
        time_grid: np.ndarray,
        base_mpor: int = 10,
    ) -> dict:
        """
        Compute exposure under normal and dispute (doubled MPOR) scenarios.

        Returns comparison of EE, EEPE, EAD under both scenarios.
        """
        from ..exposure.exposure_metrics import ExposureProfile
        results = {}
        for label, mpor in [("Normal", base_mpor), ("Dispute (2×MPOR)", 2 * base_mpor)]:
            exp_mtm = MPORAnalyser.mpor_exposure_profile(mtm_matrix, time_grid, mpor)
            profile = ExposureProfile(exp_mtm, time_grid)
            results[label] = profile.scalar_metrics()
            results[label]["MPOR_days"] = mpor
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Full collateral manager
# ─────────────────────────────────────────────────────────────────────────────

class CollateralManager:
    """
    Main entry point for collateral simulation.

    Combines VM, IM and MPOR to compute the full collateral-adjusted exposure
    profile, and provides a comprehensive diagnostic report.

    Usage
    -----
    >>> csa = CSAParameters.zero_threshold_bilateral(mpor_days=10)
    >>> cm = CollateralManager(csa, mtm_matrix, time_grid)
    >>> report = cm.full_report()
    """

    def __init__(
        self,
        csa: CSAParameters,
        mtm_matrix: np.ndarray,
        time_grid: np.ndarray,
        include_im: bool = True,
        im_confidence: float = 0.99,
    ):
        self.csa        = csa
        self.mtm        = mtm_matrix
        self.time_grid  = time_grid
        self.include_im = include_im
        self.vm_engine  = VMEngine(csa)
        self.im_engine  = IMEngine(csa, im_confidence) if include_im else None

    def gross_exposure(self) -> np.ndarray:
        """Uncollateralised positive exposure: max(V(t), 0)."""
        return np.maximum(self.mtm, 0.0)

    def vm_collateral(self) -> np.ndarray:
        """VM held at each path/time."""
        return self.vm_engine.compute_vm_matrix(self.mtm, self.time_grid)

    def im_collateral(self) -> np.ndarray | None:
        """IM held at each path/time (None if not applicable)."""
        if self.im_engine is None:
            return None
        return self.im_engine.var_based_im(self.mtm, self.time_grid)

    def net_exposure_vm_only(self) -> np.ndarray:
        """Exposure after VM only (no IM)."""
        return np.maximum(self.mtm - self.vm_collateral(), 0.0)

    def net_exposure_vm_im(self) -> np.ndarray:
        """Exposure after VM + IM."""
        if self.im_engine is None:
            return self.net_exposure_vm_only()
        return self.im_engine.exposure_with_im(
            self.mtm, self.time_grid, self.vm_collateral()
        )

    def mpor_analysis(self) -> pd.DataFrame:
        """Compare exposure across MPOR scenarios (0, 5, 10, 20 days)."""
        return MPORAnalyser.compare_mpor_scenarios(
            self.mtm, self.time_grid,
            mpor_scenarios=[0, 5, 10, 15, 20],
        )

    def full_report(self) -> dict:
        """
        Full collateral impact report.
        Returns EE, EEPE, EAD for: gross, VM-only, VM+IM.
        """
        from ..exposure.exposure_metrics import ExposureProfile

        def metrics(exp_matrix):
            p = ExposureProfile(exp_matrix, self.time_grid)
            return p.scalar_metrics()

        report = {
            "Gross (no collateral)":  metrics(self.gross_exposure()),
            "VM only":                metrics(self.net_exposure_vm_only()),
        }
        if self.include_im:
            report["VM + IM"] = metrics(self.net_exposure_vm_im())

        report["CSA_params"] = {
            "threshold_we":   self.csa.threshold_we,
            "threshold_cp":   self.csa.threshold_cp,
            "mta_we":         self.csa.mta_we,
            "mpor_days":      self.csa.mpor_days,
            "include_im":     self.include_im,
        }
        return report
