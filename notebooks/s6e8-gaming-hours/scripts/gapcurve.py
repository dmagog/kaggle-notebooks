"""How the resolvable difference depends on correlation, measured rather than derived.

The critic's point: sd(gap) = sigma * sqrt(2(1-rho)) means my 0.00002 and Dariush
Afshar's 0.00015 are one formula at two values of rho, not a disagreement. So the
useful thing to publish is the CURVE, measured on real pairs spanning a range of
rho, plus the check that the formula reproduces it.

Two correlations are NOT the same object and the reply must not conflate them:
  rho_pred = correlation of the two prediction vectors
  rho_auc  = correlation of the two AUC ESTIMATES across row draws  <- the one in the formula
"""
import glob
import os

import numpy as np
import pandas as pd

from blendgap import fast_auc


def ranks(v):
    n = len(v)
    order = np.argsort(v, kind="stable")
    r = np.empty(n, dtype=np.float32)
    r[order] = np.linspace(0, 1, n, dtype=np.float32)
    return r


tr = pd.read_csv("data/train.csv", engine="pyarrow")
y = tr["addicted_label"].to_numpy()
test = pd.read_csv("data/test.csv", engine="pyarrow")

oof_f = sorted(glob.glob("oof_lib/oof/oof_*.npy"))
names = [os.path.basename(f)[4:-4] for f in oof_f]
P = np.empty((len(names), len(y)), dtype=np.float32)
for i, fo in enumerate(oof_f):
    P[i] = ranks(np.load(fo))

solo = np.array([fast_auc(y, P[i]) for i in range(len(names))])
order = np.argsort(-solo)
n_pub = int(len(test) * 0.20)
n_priv = len(test) - n_pub

# Pairs chosen to span a range of similarity, from "same stack truncated twice"
# (the best case, which is what I measured before) down to genuinely different blends.
PAIRS = [
    ("top-8 vs top-10 (one is a subset of the other)", P[order[:8]].mean(0),  P[order[:10]].mean(0)),
    ("top-10 vs top-20",                               P[order[:10]].mean(0), P[order[:20]].mean(0)),
    ("top-10 vs top-40",                               P[order[:10]].mean(0), P[order[:40]].mean(0)),
    ("top-10 vs all 74",                               P[order[:10]].mean(0), P.mean(0)),
    ("evens vs odds of the top 20 (disjoint members)", P[order[:20][::2]].mean(0), P[order[:20][1::2]].mean(0)),
    ("best single vs 2nd best single",                 P[order[0]],           P[order[1]]),
    ("best single vs 20th best single",                P[order[0]],           P[order[19]]),
]

rgen = np.random.default_rng(1)
draws = []
for _ in range(200):
    pub = rgen.choice(len(y), n_pub, replace=False)
    priv = rgen.choice(np.setdiff1d(np.arange(len(y)), pub), n_priv, replace=False)
    draws.append((pub, priv))

print(f"{'pair':<48} {'rho_pred':>9} {'rho_auc':>8} {'sd_move':>9} {'sd_gap':>9} "
      f"{'ratio':>6} {'1.96sd':>9} {'formula':>9}")
rows = []
for label, A, B in PAIRS:
    rho_pred = np.corrcoef(A, B)[0, 1]
    a1s, a2s, b1s, b2s = [], [], [], []
    for pub, priv in draws:
        a1s.append(fast_auc(y[pub], A[pub]));  a2s.append(fast_auc(y[priv], A[priv]))
        b1s.append(fast_auc(y[pub], B[pub]));  b2s.append(fast_auc(y[priv], B[priv]))
    a1s, a2s, b1s, b2s = map(np.array, (a1s, a2s, b1s, b2s))
    moves = a2s - a1s
    gaps = (a2s - b2s) - (a1s - b1s)
    rho_auc = np.corrcoef(a1s, b1s)[0, 1]
    sd_move, sd_gap = np.std(moves), np.std(gaps)
    formula = sd_move * np.sqrt(2 * (1 - rho_auc))
    rows.append((label, rho_pred, rho_auc, sd_move, sd_gap, 1.96 * sd_gap))
    print(f"{label:<48} {rho_pred:9.5f} {rho_auc:8.5f} {sd_move:9.6f} {sd_gap:9.6f} "
          f"{sd_move/sd_gap:6.1f} {1.96*sd_gap:9.6f} {formula:9.6f}")

print()
print("Dariush Afshar quotes 0.00015 for 'highly correlated' and 0.0013-0.0018 for weak.")
for label, rp, ra, sm, sg, thr in rows:
    verdict = "finer than his 0.00015" if thr < 0.00015 else "COARSER than his 0.00015"
    print(f"  {label:<48} -> {thr:.5f}  {verdict}")
