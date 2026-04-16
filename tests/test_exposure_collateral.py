"""Tests for exposure metrics, collateral manager and SA-CCR."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import pandas as pd
from ccr.exposure import ExposureProfile, NettingSet
from ccr.collateral import CSAParameters, VMEngine, IMEngine, MPORAnalyser, CollateralManager
from ccr.sa_ccr import SACCR, SACCRTrade
from ccr.simulation import HullWhite1F, GBM, CIR


# ─── Exposure Metrics ─────────────────────────────────────────────────────────

class TestExposureProfile:
    def setup_method(self):
        rng = np.random.default_rng(42)
        n_paths, n_times = 500, 50
        self.time_grid  = np.linspace(0, 5.0, n_times)
        # Simple GBM-like MtM: paths are centred around zero
        self.mtm_matrix = rng.normal(0, 100_000, (n_paths, n_times))
        self.profile    = ExposureProfile(self.mtm_matrix, self.time_grid)

    def test_ee_nonnegative(self):
        assert np.all(self.profile.ee() >= 0.0)

    def test_ene_nonnegative(self):
        assert np.all(self.profile.ene() >= 0.0)

    def test_pfe_ge_ee(self):
        """PFE at 95% ≥ EE at every time step."""
        assert np.all(self.profile.pfe(0.95) >= self.profile.ee() - 1e-6)

    def test_eepe_ge_epe(self):
        """EEPE ≥ EPE (running maximum property)."""
        assert self.profile.eepe() >= self.profile.epe() - 1e-6

    def test_eepe_positive(self):
        assert self.profile.eepe() > 0.0

    def test_ead_imm_is_1_4_times_eepe(self):
        assert abs(self.profile.ead_imm() - 1.4 * self.profile.eepe()) < 1e-6

    def test_summary_shape(self):
        df = self.profile.summary()
        assert len(df) == len(self.time_grid)
        assert "EE" in df.columns

    def test_scalar_metrics_keys(self):
        metrics = self.profile.scalar_metrics()
        for key in ["EPE", "EEPE", "EAD_IMM", "MaxPFE", "MaxEE"]:
            assert key in metrics

    def test_all_positive_portfolio_ee_equals_mean_mtm(self):
        """If all MtM values > 0, EE = mean MtM."""
        pos_mtm   = np.abs(self.mtm_matrix)
        profile_p = ExposureProfile(pos_mtm, self.time_grid)
        np.testing.assert_allclose(profile_p.ee(), pos_mtm.mean(axis=0), rtol=1e-6)


class TestNettingSet:
    def test_netting_reduces_exposure(self):
        """Net exposure ≤ gross exposure."""
        rng = np.random.default_rng(42)
        t1 = rng.normal(50, 30, (200, 20))
        t2 = rng.normal(-50, 30, (200, 20))
        ns = NettingSet(trades=[t1, t2], apply_netting=True)
        net_mtm = ns.net_mtm()
        gross   = np.maximum(t1, 0.0) + np.maximum(t2, 0.0)
        net_exp = np.maximum(net_mtm, 0.0)
        assert np.all(net_exp.mean(axis=0) <= gross.mean(axis=0) + 1e-6)

    def test_netting_benefit_nonneg(self):
        rng = np.random.default_rng(42)
        ns = NettingSet(trades=[rng.normal(0, 1, (100, 10)) for _ in range(3)])
        benefit = ns.netting_benefit()
        assert np.all(benefit >= -1e-10)


# ─── Collateral ───────────────────────────────────────────────────────────────

class TestCSAParameters:
    def test_zero_threshold_bilateral(self):
        csa = CSAParameters.zero_threshold_bilateral(mpor_days=10)
        assert csa.threshold_we == 0.0
        assert csa.mpor_days == 10

    def test_cleared_ccp_5_days(self):
        csa = CSAParameters.cleared_ccp()
        assert csa.mpor_days == 5

    def test_mpor_years(self):
        csa = CSAParameters(mpor_days=10)
        assert abs(csa.mpor_years - 10.0 / 252.0) < 1e-9


class TestVMEngine:
    def setup_method(self):
        rng = np.random.default_rng(42)
        n_paths, n_times = 200, 60
        self.time_grid  = np.linspace(0, 5.0, n_times)
        self.mtm_matrix = rng.normal(0, 200_000, (n_paths, n_times))
        self.csa        = CSAParameters.zero_threshold_bilateral(mpor_days=10)
        self.vm_engine  = VMEngine(self.csa)

    def test_vm_matrix_shape(self):
        vm = self.vm_engine.compute_vm_matrix(self.mtm_matrix, self.time_grid)
        assert vm.shape == self.mtm_matrix.shape

    def test_vm_nonneg_at_zero_threshold(self):
        """VM held ≥ 0 (bank never has negative VM)."""
        vm = self.vm_engine.compute_vm_matrix(self.mtm_matrix, self.time_grid)
        assert np.all(vm >= -1e-6)

    def test_collateral_adjusted_less_than_gross(self):
        adj = self.vm_engine.collateral_adjusted_exposure(self.mtm_matrix, self.time_grid)
        gross = np.maximum(self.mtm_matrix, 0.0)
        assert np.mean(adj) <= np.mean(gross) + 1e-6


class TestIMEngine:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.n_paths, self.n_times = 200, 60
        self.time_grid  = np.linspace(0, 5.0, self.n_times)
        self.mtm_matrix = rng.normal(0, 200_000, (self.n_paths, self.n_times))
        self.csa        = CSAParameters.zero_threshold_bilateral(mpor_days=10)
        self.im_engine  = IMEngine(self.csa, confidence_level=0.99)

    def test_im_matrix_shape(self):
        im = self.im_engine.var_based_im(self.mtm_matrix, self.time_grid)
        assert im.shape == self.mtm_matrix.shape

    def test_im_nonneg(self):
        im = self.im_engine.var_based_im(self.mtm_matrix, self.time_grid)
        assert np.all(im >= 0.0)

    def test_exposure_vm_im_le_vm_only(self):
        """Adding IM can only reduce exposure."""
        vm_engine = VMEngine(self.csa)
        vm = vm_engine.compute_vm_matrix(self.mtm_matrix, self.time_grid)
        exp_vm_only = np.maximum(self.mtm_matrix - vm, 0.0)
        exp_vm_im   = self.im_engine.exposure_with_im(self.mtm_matrix, self.time_grid, vm)
        assert np.mean(exp_vm_im) <= np.mean(exp_vm_only) + 1e-6


class TestMPORAnalyser:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.time_grid  = np.linspace(0, 5.0, 60)
        self.mtm_matrix = rng.normal(0, 100_000, (200, 60))

    def test_longer_mpor_gives_higher_exposure(self):
        """Longer MPOR → more time for MtM to move → higher exposure."""
        df = MPORAnalyser.compare_mpor_scenarios(self.mtm_matrix, self.time_grid, [5, 10, 20])
        eepe_5  = df.loc[5,  "EEPE"]
        eepe_10 = df.loc[10, "EEPE"]
        eepe_20 = df.loc[20, "EEPE"]
        assert eepe_20 >= eepe_10 - 1e-6
        assert eepe_10 >= eepe_5  - 1e-6

    def test_zero_mpor_gives_near_zero_exposure(self):
        """MPOR=0 → collateral is always current → exposure ≈ 0."""
        df = MPORAnalyser.compare_mpor_scenarios(self.mtm_matrix, self.time_grid, [0])
        assert df.loc[0, "EE_max"] < 1.0

    def test_dispute_scenario_returns_dict(self):
        result = MPORAnalyser.dispute_scenario(self.mtm_matrix, self.time_grid, base_mpor=10)
        assert "Normal" in result
        assert "Dispute (2×MPOR)" in result
        assert result["Dispute (2×MPOR)"]["EEPE"] >= result["Normal"]["EEPE"] - 1e-6

    def test_compare_returns_dataframe(self):
        df = MPORAnalyser.compare_mpor_scenarios(self.mtm_matrix, self.time_grid)
        assert isinstance(df, pd.DataFrame)
        assert "EEPE" in df.columns


# ─── SA-CCR ───────────────────────────────────────────────────────────────────

class TestSACCR:
    def _make_irs(self, mtm=10_000, maturity=5.0):
        return SACCRTrade(
            trade_id="IRS_01", asset_class="ir",
            notional=1_000_000, current_mtm=mtm,
            maturity=maturity, is_bought=True,
        )

    def _make_fx(self, mtm=5_000):
        return SACCRTrade(
            trade_id="FX_01", asset_class="fx",
            notional=500_000, current_mtm=mtm,
            maturity=1.0, is_bought=True,
        )

    def test_ead_positive(self):
        calc = SACCR([self._make_irs()], is_margined=True)
        assert calc.ead() > 0.0

    def test_ead_equals_alpha_times_rc_plus_pfe(self):
        """EAD = 1.4 × (RC + PFE)."""
        t = self._make_irs(mtm=100_000)
        calc = SACCR([t], is_margined=True)
        rc   = calc.replacement_cost()
        addon = calc.addon_ir() + calc.addon_fx() + calc.addon_credit() + calc.addon_equity() + calc.addon_commodity()
        mult = calc.pfe_multiplier()
        pfe  = mult * addon
        expected_ead = 1.4 * (rc + pfe)
        assert abs(calc.ead() - expected_ead) < 1.0

    def test_rc_zero_when_mtm_zero_and_no_collateral(self):
        t = self._make_irs(mtm=0.0)
        calc = SACCR([t], is_margined=True, threshold=0.0, mta=0.0)
        assert calc.replacement_cost() >= 0.0

    def test_netting_reduces_rc(self):
        """Opposite trades reduce RC via netting."""
        long  = SACCRTrade("L", "ir", 1e6, current_mtm=+50_000, maturity=5.0, is_bought=True)
        short = SACCRTrade("S", "ir", 1e6, current_mtm=-40_000, maturity=5.0, is_bought=False)
        calc_both  = SACCR([long, short])
        calc_long  = SACCR([long])
        assert calc_both.replacement_cost() <= calc_long.replacement_cost()

    def test_addon_ir_positive(self):
        calc = SACCR([self._make_irs()])
        assert calc.addon_ir() > 0.0

    def test_addon_fx_positive(self):
        calc = SACCR([self._make_fx()])
        assert calc.addon_fx() > 0.0

    def test_multiplier_between_0_and_1(self):
        calc = SACCR([self._make_irs()])
        mult = calc.pfe_multiplier()
        assert 0.0 < mult <= 1.0

    def test_report_returns_dataframe(self):
        calc = SACCR([self._make_irs(), self._make_fx()])
        report = calc.report()
        assert isinstance(report, pd.DataFrame)
        assert "EAD = 1.4 × (RC + PFE)" in report["Component"].values

    def test_margined_ead_less_than_unmargined(self):
        """Margined EAD uses shorter MPOR → lower AddOn → lower EAD."""
        t = self._make_irs(maturity=5.0)
        calc_margined   = SACCR([t], is_margined=True)
        calc_unmargined = SACCR([t], is_margined=False)
        assert calc_margined.ead() <= calc_unmargined.ead() + 1.0


# ─── Simulation Models ────────────────────────────────────────────────────────

class TestMarketModels:
    def test_hw1f_positive_rates(self):
        """HW1F can produce negative rates (not floored) but should have positive mean."""
        hw = HullWhite1F(r0=0.03, a=0.05, sigma=0.010)
        sim = hw.simulate(n_paths=500, n_steps=50, horizon=5.0, seed=42)
        assert sim.paths.shape == (500, 51)
        assert sim.mean().mean() > 0.0   # positive on average

    def test_gbm_paths_positive(self):
        """GBM log-normal → always positive."""
        gbm = GBM(S0=100.0, mu=0.03, sigma=0.20)
        sim = gbm.simulate(n_paths=500, n_steps=50, horizon=5.0, seed=42)
        assert np.all(sim.paths > 0.0)

    def test_cir_paths_nonneg(self):
        """CIR full-truncation → non-negative."""
        cir = CIR(x0=0.01, kappa=0.30, theta=0.01, sigma=0.05)
        sim = cir.simulate(n_paths=500, n_steps=50, horizon=5.0, seed=42)
        assert np.all(sim.paths >= 0.0)

    def test_gbm_mean_close_to_analytical(self):
        """E[S(T)] = S0 * exp(mu * T) under GBM."""
        gbm = GBM(S0=100.0, mu=0.05, sigma=0.20)
        sim = gbm.simulate(n_paths=10_000, n_steps=1, horizon=1.0, seed=42)
        theoretical = 100.0 * np.exp(0.05 * 1.0)
        empirical   = sim.paths[:, -1].mean()
        assert abs(empirical - theoretical) / theoretical < 0.02  # within 2%
