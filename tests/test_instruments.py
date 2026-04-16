"""Tests for instrument pricing modules."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from ccr.instruments import IRSwap, FXForward, FXOption, CDS, EquityOption


# ─── IRS ──────────────────────────────────────────────────────────────────────

class TestIRSwap:
    def test_at_market_mtm_is_zero(self):
        """A swap struck at the current market rate has MtM = 0."""
        swap = IRSwap(notional=1e6, fixed_rate=0.03, maturity=5.0)
        assert abs(swap.mtm(market_rate=0.03)) < 1.0  # < 1 EUR

    def test_payer_gains_when_rates_rise(self):
        """Payer swap benefits from rising rates."""
        swap = IRSwap(notional=1e6, fixed_rate=0.03, maturity=5.0, position="payer")
        assert swap.mtm(0.05) > 0.0
        assert swap.mtm(0.01) < 0.0

    def test_receiver_is_opposite_payer(self):
        """Receiver MtM = -Payer MtM."""
        payer    = IRSwap(notional=1e6, fixed_rate=0.03, maturity=5.0, position="payer")
        receiver = IRSwap(notional=1e6, fixed_rate=0.03, maturity=5.0, position="receiver")
        rate = 0.04
        assert abs(payer.mtm(rate) + receiver.mtm(rate)) < 1.0

    def test_dv01_positive_for_payer(self):
        """DV01 > 0 for payer (loses when rates fall)."""
        swap = IRSwap(notional=1e6, fixed_rate=0.03, maturity=5.0, position="payer")
        assert swap.dv01(0.03) > 0.0

    def test_dv01_increases_with_maturity(self):
        """Longer maturity → higher DV01."""
        s5  = IRSwap(maturity=5.0)
        s10 = IRSwap(maturity=10.0)
        assert s10.dv01(0.03) > s5.dv01(0.03)

    def test_mtm_profile_shape(self):
        """MtM profile returns correct shape."""
        swap = IRSwap(notional=1e6, fixed_rate=0.03, maturity=5.0)
        rate_paths = np.random.default_rng(42).normal(0.03, 0.005, (100, 20))
        time_grid  = np.linspace(0, 4.0, 20)
        profile = swap.mtm_profile(rate_paths, time_grid)
        assert profile.shape == (100, 20)

    def test_annuity_positive(self):
        swap = IRSwap()
        assert swap.annuity(0.03) > 0.0


# ─── FX Forward ───────────────────────────────────────────────────────────────

class TestFXForward:
    def test_at_market_forward_mtm_zero(self):
        """Forward struck at current forward rate has MtM ≈ 0."""
        S, rd, rf, T = 1.10, 0.03, 0.01, 1.0
        F = S * np.exp((rd - rf) * T)
        fwd = FXForward(notional_foreign=1e6, forward_rate_agreed=F, maturity=T)
        assert abs(fwd.mtm(S, rd, rf)) < 10.0

    def test_long_gains_when_spot_rises(self):
        fwd = FXForward(notional_foreign=1e6, forward_rate_agreed=1.10, maturity=1.0, position="long")
        assert fwd.mtm(1.20, 0.03, 0.01) > 0.0

    def test_short_is_opposite_long(self):
        long  = FXForward(notional_foreign=1e6, forward_rate_agreed=1.10, maturity=1.0, position="long")
        short = FXForward(notional_foreign=1e6, forward_rate_agreed=1.10, maturity=1.0, position="short")
        mtm_long  = long.mtm(1.15, 0.03, 0.01)
        mtm_short = short.mtm(1.15, 0.03, 0.01)
        assert abs(mtm_long + mtm_short) < 1.0

    def test_profile_shape(self):
        fwd = FXForward(notional_foreign=1e6, forward_rate_agreed=1.10, maturity=2.0)
        paths = np.ones((50, 15)) * 1.10
        tg = np.linspace(0, 1.5, 15)
        assert fwd.mtm_profile(paths, 0.03, 0.01, tg).shape == (50, 15)


# ─── FX Option ────────────────────────────────────────────────────────────────

class TestFXOption:
    def test_call_price_positive(self):
        opt = FXOption(notional_foreign=1e6, strike=1.10, maturity=1.0, option_type="call", sigma=0.10)
        assert opt.price(1.10, 0.03, 0.01) > 0.0

    def test_put_call_parity(self):
        """Call - Put = PV(F - K) — Garman-Kohlhagen put-call parity."""
        S, K, rd, rf, T, sig = 1.10, 1.10, 0.03, 0.01, 1.0, 0.10
        call = FXOption(notional_foreign=1.0, strike=K, maturity=T, option_type="call", sigma=sig)
        put  = FXOption(notional_foreign=1.0, strike=K, maturity=T, option_type="put",  sigma=sig)
        F = S * np.exp((rd - rf) * T)
        parity = call.price(S, rd, rf) - put.price(S, rd, rf)
        theoretical = (F - K) * np.exp(-rd * T)
        assert abs(parity - theoretical) < 1e-6

    def test_deep_itm_call_approaches_forward(self):
        """Deep ITM call ≈ PV of forward."""
        opt = FXOption(notional_foreign=1.0, strike=0.01, maturity=1.0, option_type="call", sigma=0.10)
        S, rd, rf = 1.10, 0.03, 0.01
        F_pv = (S * np.exp((rd - rf) * 1.0) - 0.01) * np.exp(-rd * 1.0)
        assert abs(opt.price(S, rd, rf) - F_pv) < 0.01

    def test_long_short_sum_zero(self):
        long  = FXOption(notional_foreign=1e6, strike=1.10, maturity=1.0, sigma=0.10, position="long")
        short = FXOption(notional_foreign=1e6, strike=1.10, maturity=1.0, sigma=0.10, position="short")
        assert abs(long.price(1.10, 0.03, 0.01) + short.price(1.10, 0.03, 0.01)) < 1.0


# ─── CDS ──────────────────────────────────────────────────────────────────────

class TestCDS:
    def test_protection_buyer_gains_on_spread_widening(self):
        cds = CDS(notional=1e6, spread=0.01, maturity=5.0, position="protection_buyer")
        mtm_tight = cds.mtm(0.03, 0.005)
        mtm_wide  = cds.mtm(0.03, 0.020)
        assert mtm_wide > mtm_tight

    def test_protection_seller_is_opposite(self):
        buyer  = CDS(notional=1e6, spread=0.01, maturity=5.0, position="protection_buyer")
        seller = CDS(notional=1e6, spread=0.01, maturity=5.0, position="protection_seller")
        assert abs(buyer.mtm(0.03, 0.015) + seller.mtm(0.03, 0.015)) < 100.0

    def test_survival_prob_decreasing(self):
        cds = CDS(spread=0.01, recovery=0.40)
        p1 = cds.survival_prob(1.0, 0.01)
        p5 = cds.survival_prob(5.0, 0.01)
        assert p1 > p5 > 0.0

    def test_cs01_positive_for_buyer(self):
        """CS01 > 0: buyer benefits from spread widening."""
        cds = CDS(notional=1e6, spread=0.01, maturity=5.0, position="protection_buyer")
        assert cds.cs01(0.03, 0.01) > 0.0


# ─── Equity Option ────────────────────────────────────────────────────────────

class TestEquityOption:
    def test_bs_price_positive(self):
        opt = EquityOption(spot=100, strike=100, maturity=1.0, sigma=0.20, rate=0.03)
        assert opt.bs_price() > 0.0

    def test_put_call_parity(self):
        S, K, r, q, T, sig = 100.0, 100.0, 0.03, 0.0, 1.0, 0.20
        call = EquityOption(spot=S, strike=K, maturity=T, sigma=sig, rate=r, dividend_yield=q, option_type="call", notional=1.0)
        put  = EquityOption(spot=S, strike=K, maturity=T, sigma=sig, rate=r, dividend_yield=q, option_type="put",  notional=1.0)
        parity = call.bs_price() - put.bs_price()
        theoretical = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(parity - theoretical) < 1e-6

    def test_delta_call_between_0_and_1(self):
        opt = EquityOption(spot=100, strike=100, maturity=1.0, sigma=0.20, rate=0.03, option_type="call")
        assert 0.0 < opt.delta() < 1.0

    def test_delta_put_between_minus1_and_0(self):
        opt = EquityOption(spot=100, strike=100, maturity=1.0, sigma=0.20, rate=0.03, option_type="put")
        assert -1.0 < opt.delta() < 0.0

    def test_gamma_positive_for_long(self):
        opt = EquityOption(spot=100, strike=100, maturity=1.0, sigma=0.20, rate=0.03)
        assert opt.gamma() > 0.0

    def test_vega_positive_for_long(self):
        opt = EquityOption(spot=100, strike=100, maturity=1.0, sigma=0.20, rate=0.03)
        assert opt.vega() > 0.0

    def test_theta_negative_for_long(self):
        """Time decay: long option loses value as time passes."""
        opt = EquityOption(spot=100, strike=100, maturity=1.0, sigma=0.20, rate=0.03)
        assert opt.theta() < 0.0

    def test_american_put_ge_european(self):
        """American put ≥ European put (early exercise premium)."""
        eu = EquityOption(spot=100, strike=110, maturity=1.0, sigma=0.20, rate=0.05, option_type="put", style="european")
        am = EquityOption(spot=100, strike=110, maturity=1.0, sigma=0.20, rate=0.05, option_type="put", style="american")
        assert am.price() >= eu.price() - 0.01   # small tolerance for numerical methods

    def test_price_decreases_with_strike_for_call(self):
        """Higher strike → cheaper call (all else equal)."""
        c1 = EquityOption(spot=100, strike=90,  maturity=1.0, sigma=0.20, rate=0.03, option_type="call")
        c2 = EquityOption(spot=100, strike=110, maturity=1.0, sigma=0.20, rate=0.03, option_type="call")
        assert c1.bs_price() > c2.bs_price()
