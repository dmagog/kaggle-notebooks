# S6E8: why gaming_hours helps but adds nothing new — measurement scripts

The notebook lives on Kaggle:
[why gaming_hours helps but adds nothing new](https://www.kaggle.com/code/georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new).
Every number its prose types in by hand, rather than computes in a cell, is reproduced by one of
the scripts here, run against the public downloads they name in their docstrings.

| script | reproduces |
|---|---|
| `scripts/precedent_boards.py` | the live-board readings sections 11 and 12 type in: top-ten gaps, packing rulers, the cut sweep, the S6E2 slide, the 2,058 hairline pair, the fork-cluster series, today's top-ten tier profile |
| `scripts/pool_arms.py` | section 10's pool table: 86 → 111 → 158 members, placebo control, the CVs behind the 18 August submission pair |
| `scripts/stacker_arms.py` | the LightGBM-vs-logistic combiner comparison (the negative result) |
| `scripts/formula_backtest.py` | the seven-board back-test behind the discussion-734005 reply: band correlations, the noise-only null, survival by public band |
| `scripts/gapcurve.py` | section 11's seven resampled pairs: sd of the gap as a function of pair correlation |
| `scripts/ext_gate.py`, `scripts/ext_arms.py` | the external-library screens and paired-fold arms that fed the pool |
| `scripts/check_names.py` | the pre-push checker for notebook cells: names loaded before anything binds them, and helpers clobbered by later cells |

Data inputs: the competition files, [the S6 leaderboards dataset](https://www.kaggle.com/datasets/georgymamarin/playground-series-s6-leaderboards),
the public OOF libraries credited in the notebook, and Meta Kaggle.
