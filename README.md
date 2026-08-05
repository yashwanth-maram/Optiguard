# OptiGuard AI

Photon-budget measurement assurance for spectroscopic and optical inspection.

## Build order

0. Repo + env                       <- you are here
1. physics/detector.py              ADU <-> photons, gain, read noise, dark
2. physics/lineshapes.py            Lorentzian, pseudo-Voigt, instrument function
3. physics/thinning.py              binomial photon thinning
4. data/simulator.py                Tier A synthetic maps  <- THE dataset
5. estimation/fit.py                vectorised Poisson-weighted peak fitting
6. physics/crlb.py                  numerical Fisher information   <- KEYSTONE
7. baselines + eval harness         Savitzky-Golay / PCA / NMF
8. models/ + training/              restoration network (Colab)
9. assurance/pooling.py             Jacobian probe -> N_eff
10. assurance/gate.py               T1 / T2 / T3
11+ risk fusion, planner, OOD, API/UI, Tier B evidence

Steps 0-7 need no GPU and no downloaded dataset.

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest -m "not slow"     # fast tests
    pytest                   # includes statistical tests

Tests in tests/test_physics.py are the specification. Never weaken an
assertion to make it pass.
