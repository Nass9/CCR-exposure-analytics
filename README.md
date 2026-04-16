# ccr-exposure-analytics

Python toolkit for **Counterparty Credit Risk (CCR)** exposure simulation and regulatory capital computation.

## Features

| Module | Description |
|---|---|
| `ccr/instruments/` | IRS, FX Forward, FX Option (Garman-Kohlhagen), CDS, Equity Option (BS + American) |
| `ccr/simulation/` | Hull-White 1F (rates), GBM (equity/FX), CIR (credit spreads), correlated multi-factor |
| `ccr/exposure/` | EE, ENE, EPE, **EEPE** (IMM), PFE, EAD, netting sets |
| `ccr/collateral/` | VM dynamics, IM (VaR-based), **MPOR** scenarios, CSA parameters, full collateral manager |
| `ccr/sa_ccr.py` | **SA-CCR** EAD: RC, PFE add-ons (IR/FX/Credit/Equity), supervisory delta, PFE multiplier |

## Regulatory framework

- Basel III / CRR2: IMM (EEPE) and SA-CCR
- MPOR: 5 days (CCP), 10 days (bilateral OTC), 20 days (illiquid)
- FRTB / CRR3: SA-CVA, IM (ISDA-SIMM simplified)

## Notebooks

| Notebook | Content |
|---|---|
| `01_instruments_pricing.ipynb` | IRS DV01, FX option Greeks, CDS CS01, equity option surfaces |
| `02_exposure_simulation.ipynb` | Monte Carlo EE/PFE profiles, netting benefit, EEPE vs EPE |
| `03_collateral_mpor.ipynb` | VM/IM dynamics, MPOR sensitivity, clearing vs bilateral |
| `04_sa_ccr.ipynb` | SA-CCR EAD decomposition, add-ons by asset class, multiplier |

## Quick start

```bash
git clone https://github.com/Nass9/ccr-exposure-analytics
cd ccr-exposure-analytics
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
jupyter notebook notebooks/
```
