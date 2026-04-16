"""
sa_ccr.py — SA-CCR: Standardised Approach for Counterparty Credit Risk
========================================================================
Full implementation of the Basel SA-CCR framework (BCBS, March 2014)
as incorporated into CRR2/CRR3 (EU) and the revised Basel III framework.

SA-CCR replaces CEM (Current Exposure Method) and SM (Standardised Method).
It is the fall-back approach for institutions not approved for IMM.

EAD formula
-----------
EAD = α × (RC + PFE_addon)

where:
  α    = 1.4   (regulatory multiplier)
  RC   = Replacement Cost
  PFE  = Potential Future Exposure = multiplier × AddOn_aggregate

Replacement Cost
----------------
Without margining (unmargined):
  RC = max(V - C, 0)   where V = current MtM, C = collateral held

With margining (margined):
  RC = max(V - C, TH + MTA - NICA, 0)
  NICA = Net Independent Collateral Amount (IA held - IA posted)

PFE Add-on
----------
Aggregated across 5 asset classes:
  1. Interest Rates (IR)
  2. Foreign Exchange (FX)
  3. Credit (CDS)
  4. Equity
  5. Commodities

Each trade contributes an add-on computed as:
  AddOn_trade = Supervisory_delta × Adjusted_notional × Supervisory_factor × Maturity_factor

Aggregation uses correlation structures within each asset class.

Supervisory parameters (BCBS 2014, Annex A)
--------------------------------------------
Asset class  | Supervisory factor | Correlation (ρ)
-------------|-------------------|----------------
IR           | 0.50%             | 100% intra-bucket / 0% cross-bucket
FX           | 4.00%             | N/A (each pair is standalone)
Credit IG    | 0.38%             | 50%
Credit HY    | 1.06%             | 50%
Equity large | 32%               | 50%
Equity small | 20%               | N/A
Commodity    | 18-40%            | 40%

References
----------
- BCBS (2014), SA-CCR: Standardised Approach for Measuring CCR Exposures
- EBA (2020), Guidelines on SA-CCR
- CRR2 Article 274-279h (EU implementation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# Supervisory parameters (BCBS 2014, Annex A — Table 1)
# ─────────────────────────────────────────────────────────────────────────────

SUPERVISORY_FACTORS = {
    "ir":              0.0050,   # 0.50%
    "fx":              0.0400,   # 4.00%
    "credit_ig":       0.0038,   # 0.38%
    "credit_hy":       0.0106,   # 1.06%
    "equity_large":    0.3200,   # 32%
    "equity_small":    0.2000,   # 20%
    "commodity_energy":0.1800,   # 18%
    "commodity_metal": 0.1800,   # 18%
    "commodity_agri":  0.1800,   # 18%
    "commodity_other": 0.1800,   # 18%
}

SUPERVISORY_CORRELATIONS = {
    "ir":              0.0,      # correlation between IR hedging sets
    "fx":              0.0,      # each FX pair is its own hedging set
    "credit_ig":       0.50,
    "credit_hy":       0.50,
    "equity_large":    0.50,
    "equity_small":    0.50,
    "commodity_energy":0.40,
    "commodity_other": 0.40,
}

# Maturity factor supervisory options (BCBS Table 1)
SUPERVISORY_DURATIONS = {
    "ir_lt1y":   0.0,
    "ir_1y_5y":  0.25,
    "ir_gt5y":   1.0,
}

ALPHA = 1.4   # regulatory multiplier


# ─────────────────────────────────────────────────────────────────────────────
# Trade data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SACCRTrade:
    """
    A single derivative trade for SA-CCR calculation.

    Parameters
    ----------
    trade_id : str
    asset_class : {"ir", "fx", "credit", "equity", "commodity"}
    notional : float
        Adjusted notional (for IR: notional × supervisory duration).
        For FX/Equity: notional in domestic currency.
        For IR swaps: notional × SD_i (supervisory duration).
    current_mtm : float
        Current MtM in domestic currency (positive = in-the-money).
    maturity : float
        Remaining maturity in years.
    start_date : float
        Start date in years from today (0 for spot-starting).
    option_type : {"call", "put", "none"} — for options only
    is_bought : bool
        True if long (bought protection / call / payer).
    sub_type : str
        More specific type (e.g. "ig" for investment grade CDS).
    collateral_held : float
        Collateral already held against this trade (for RC).
    """
    trade_id: str = "T001"
    asset_class: Literal["ir", "fx", "credit", "equity", "commodity"] = "ir"
    notional: float = 1_000_000.0
    current_mtm: float = 0.0
    maturity: float = 5.0
    start_date: float = 0.0
    option_type: Literal["call", "put", "none"] = "none"
    is_bought: bool = True
    sub_type: str = "ig"          # e.g. "hy", "large_cap"
    collateral_held: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SA-CCR Calculator
# ─────────────────────────────────────────────────────────────────────────────

class SACCR:
    """
    SA-CCR EAD calculator for a netting set.

    Computes:
    - Replacement Cost (RC)
    - PFE Add-on by asset class and aggregate
    - EAD = α × (RC + PFE)
    """

    def __init__(
        self,
        trades: list[SACCRTrade],
        is_margined: bool = True,
        threshold: float = 0.0,
        mta: float = 0.0,
        nica: float = 0.0,     # Net Independent Collateral Amount
        vm_held: float = 0.0,  # VM currently held
    ):
        self.trades      = trades
        self.is_margined = is_margined
        self.threshold   = threshold
        self.mta         = mta
        self.nica        = nica
        self.vm_held     = vm_held

    # ── Replacement Cost ──────────────────────────────────────────────────────

    def replacement_cost(self) -> float:
        """
        RC = max(V - C, 0)  [unmargined]
        RC = max(V - C, TH + MTA - NICA, 0)  [margined]

        V = net MtM of the netting set
        C = net collateral held (VM - IM posted by us)
        """
        V = sum(t.current_mtm for t in self.trades)
        C = self.vm_held + self.nica

        if not self.is_margined:
            return max(V - C, 0.0)
        else:
            floor = self.threshold + self.mta - self.nica
            return max(V - C, floor, 0.0)

    # ── Maturity Factor ───────────────────────────────────────────────────────

    def maturity_factor(self, trade: SACCRTrade) -> float:
        """
        MF_margined   = √(min(MPOR, 1year) / 1year)   [Basel: MPOR=10d → MF=0.21]
        MF_unmargined = √(min(M, 1year) / 1year)       [M = remaining maturity]

        The maturity factor scales the supervisory add-on to reflect the
        effective exposure horizon.
        """
        if self.is_margined:
            mpor_years = 10.0 / 252.0    # 10-day MPOR (standard bilateral)
            return np.sqrt(min(mpor_years, 1.0) / 1.0)
        else:
            M = max(trade.maturity, 10.0 / 252.0)
            return np.sqrt(min(M, 1.0) / 1.0)

    # ── Adjusted Notional ─────────────────────────────────────────────────────

    def adjusted_notional(self, trade: SACCRTrade) -> float:
        """
        IR: Adjusted notional = Notional × Supervisory Duration (SD)
        FX: Adjusted notional = Notional (in domestic currency)
        Credit/Equity: Adjusted notional = Notional (reference)

        Supervisory Duration (IR):
        SD = [exp(-0.05 * S) - exp(-0.05 * E)] / 0.05
        where S = start date, E = end date (both in years).
        """
        if trade.asset_class == "ir":
            S = trade.start_date
            E = trade.start_date + trade.maturity
            sd = (np.exp(-0.05 * S) - np.exp(-0.05 * E)) / 0.05
            return trade.notional * sd
        return trade.notional

    # ── Supervisory Delta ─────────────────────────────────────────────────────

    def supervisory_delta(self, trade: SACCRTrade) -> float:
        """
        The supervisory delta captures the direction and optionality of the trade.

        Linear trades (non-options):
            δ = +1  (long / payer / buyer)
            δ = -1  (short / receiver / seller)

        Options (Black's formula for implied probability):
            Call: δ = +Φ(d₁)   [long] or -Φ(d₁) [short]
            Put:  δ = -Φ(-d₁)  [long] or +Φ(-d₁) [short]

        where d₁ = [ln(P/K) + σ²T/2] / (σ√T)
        and P = underlying price, K = strike, σ = supervisory vol.
        """
        if trade.option_type == "none":
            return 1.0 if trade.is_bought else -1.0

        # Supervisory volatility by asset class (BCBS 2014, Annex)
        sv = {"ir": 0.50, "fx": 0.15, "credit": 1.00, "equity": 0.80, "commodity": 0.70}
        sigma = sv.get(trade.asset_class, 0.50)

        # Use current MtM as proxy for P/K ratio (moneyness)
        moneyness = 1.0 + trade.current_mtm / max(abs(trade.notional), 1.0)
        moneyness = max(moneyness, 0.001)
        T = max(trade.maturity, 1.0 / 252)
        d1 = (np.log(moneyness) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))

        if trade.option_type == "call":
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)

        return delta if trade.is_bought else -delta

    # ── Asset class add-ons ───────────────────────────────────────────────────

    def _addon_trade(self, trade: SACCRTrade) -> float:
        """
        Individual trade add-on:
        AddOn_i = δ_i × d_i × SF_i × MF_i
        """
        delta = self.supervisory_delta(trade)
        adj_n = self.adjusted_notional(trade)
        sf    = self._supervisory_factor(trade)
        mf    = self.maturity_factor(trade)
        return delta * adj_n * sf * mf

    def _supervisory_factor(self, trade: SACCRTrade) -> float:
        """Look up the supervisory factor for a trade."""
        mapping = {
            "ir": "ir",
            "fx": "fx",
            "credit": f"credit_{trade.sub_type}" if trade.sub_type in ("ig", "hy") else "credit_ig",
            "equity": f"equity_{trade.sub_type}" if trade.sub_type in ("large", "small") else "equity_large",
            "commodity": f"commodity_{trade.sub_type}" if trade.sub_type in ("energy", "metal", "agri") else "commodity_other",
        }
        key = mapping.get(trade.asset_class, "ir")
        return SUPERVISORY_FACTORS.get(key, 0.005)

    def addon_ir(self) -> float:
        """
        IR add-on: aggregated across maturity buckets within each currency.

        Buckets:
          1: [0, 1 year]  — short term
          2: [1, 5 years] — medium term
          3: [5+  years]  — long term

        Aggregation (same currency, different buckets): sum (ρ=100% within bucket)
        Cross-currency: sum of absolute values (correlation = 0 across currencies).
        """
        ir_trades = [t for t in self.trades if t.asset_class == "ir"]
        if not ir_trades:
            return 0.0

        # Group by currency (simplified: all same currency here)
        buckets = {1: 0.0, 2: 0.0, 3: 0.0}
        for t in ir_trades:
            addon = self._addon_trade(t)
            if t.maturity <= 1.0:
                buckets[1] += addon
            elif t.maturity <= 5.0:
                buckets[2] += addon
            else:
                buckets[3] += addon

        # Intra-currency aggregation with bucket correlations ρ12=ρ23=0.7, ρ13=0.3
        rho = {(1, 2): 0.70, (1, 3): 0.30, (2, 3): 0.70}
        b = list(buckets.values())
        addon_sq = (b[0]**2 + b[1]**2 + b[2]**2
                    + 2 * rho[(1, 2)] * b[0] * b[1]
                    + 2 * rho[(1, 3)] * b[0] * b[2]
                    + 2 * rho[(2, 3)] * b[1] * b[2])
        return np.sqrt(max(addon_sq, 0.0))

    def addon_fx(self) -> float:
        """FX add-on: each currency pair is a standalone hedging set (sum of abs)."""
        fx_trades = [t for t in self.trades if t.asset_class == "fx"]
        return abs(sum(self._addon_trade(t) for t in fx_trades))

    def addon_credit(self) -> float:
        """
        Credit add-on: correlation within single-name hedging sets.

        AddOn_credit = sqrt( (ρ * ΣAddOn_i)² + (1-ρ²) * Σ(AddOn_i²) )
        ρ = 0.50 for all credit
        """
        cr_trades = [t for t in self.trades if t.asset_class == "credit"]
        if not cr_trades:
            return 0.0
        addons = [self._addon_trade(t) for t in cr_trades]
        rho = 0.50
        sys_part = (rho * sum(addons)) ** 2
        idio_part = (1 - rho**2) * sum(a**2 for a in addons)
        return np.sqrt(sys_part + idio_part)

    def addon_equity(self) -> float:
        """
        Equity add-on: same structure as credit.
        ρ = 0.50 for large-cap, 0.20 for small-cap (simplified to 0.50).
        """
        eq_trades = [t for t in self.trades if t.asset_class == "equity"]
        if not eq_trades:
            return 0.0
        addons = [self._addon_trade(t) for t in eq_trades]
        rho = 0.50
        sys_part = (rho * sum(addons)) ** 2
        idio_part = (1 - rho**2) * sum(a**2 for a in addons)
        return np.sqrt(sys_part + idio_part)

    def addon_commodity(self) -> float:
        """Commodity add-on: correlation ρ=0.40 within sub-type hedging set."""
        cm_trades = [t for t in self.trades if t.asset_class == "commodity"]
        if not cm_trades:
            return 0.0
        addons = [self._addon_trade(t) for t in cm_trades]
        rho = 0.40
        sys_part = (rho * sum(addons)) ** 2
        idio_part = (1 - rho**2) * sum(a**2 for a in addons)
        return np.sqrt(sys_part + idio_part)

    # ── PFE multiplier ────────────────────────────────────────────────────────

    def pfe_multiplier(self) -> float:
        """
        PFE multiplier accounts for collateralisation level.

        multiplier = min(1, 0.05 + 0.95 * exp(V - C) / (2 × 0.95 × AddOn_agg))

        where V - C = net MtM minus collateral held.
        Multiplier = 1 if no excess collateral; < 1 if over-collateralised.
        Floored at 0.05 (conservative minimum).
        """
        V = sum(t.current_mtm for t in self.trades)
        C = self.vm_held + self.nica
        addon_agg = self.addon_ir() + self.addon_fx() + self.addon_credit() + self.addon_equity() + self.addon_commodity()

        if addon_agg < 1e-9:
            return 1.0

        exponent = (V - C) / (2 * 0.95 * addon_agg)
        return min(1.0, 0.05 + 0.95 * np.exp(exponent))

    # ── EAD ──────────────────────────────────────────────────────────────────

    def ead(self) -> float:
        """
        EAD = α × (RC + PFE)
        PFE = multiplier × AddOn_aggregate
        α   = 1.4
        """
        rc = self.replacement_cost()
        addon_agg = (self.addon_ir() + self.addon_fx() + self.addon_credit()
                     + self.addon_equity() + self.addon_commodity())
        mult = self.pfe_multiplier()
        pfe = mult * addon_agg
        return ALPHA * (rc + pfe)

    # ── Full report ───────────────────────────────────────────────────────────

    def report(self) -> pd.DataFrame:
        """Detailed SA-CCR decomposition report."""
        rc = self.replacement_cost()
        addon_ir   = self.addon_ir()
        addon_fx   = self.addon_fx()
        addon_cr   = self.addon_credit()
        addon_eq   = self.addon_equity()
        addon_cm   = self.addon_commodity()
        addon_agg  = addon_ir + addon_fx + addon_cr + addon_eq + addon_cm
        mult       = self.pfe_multiplier()
        pfe        = mult * addon_agg
        ead_val    = ALPHA * (rc + pfe)

        rows = [
            ("Replacement Cost (RC)",    rc,        ""),
            ("AddOn — Interest Rates",   addon_ir,  f"{addon_ir/max(addon_agg,1):.1%}"),
            ("AddOn — FX",               addon_fx,  f"{addon_fx/max(addon_agg,1):.1%}"),
            ("AddOn — Credit",           addon_cr,  f"{addon_cr/max(addon_agg,1):.1%}"),
            ("AddOn — Equity",           addon_eq,  f"{addon_eq/max(addon_agg,1):.1%}"),
            ("AddOn — Commodity",        addon_cm,  f"{addon_cm/max(addon_agg,1):.1%}"),
            ("AddOn aggregate",          addon_agg, "100%"),
            ("PFE multiplier",           mult,      ""),
            ("PFE = mult × AddOn_agg",   pfe,       ""),
            ("EAD = 1.4 × (RC + PFE)",   ead_val,   ""),
        ]
        return pd.DataFrame(rows, columns=["Component", "Value (€)", "% of AddOn"])
