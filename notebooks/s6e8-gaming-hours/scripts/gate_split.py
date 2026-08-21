"""Which half of the `+slack` arm carries the +0.00071?

gate.py bundles two columns into one arm: tamerlanomralinov's `slack` (daily minus the sum of the
observed components) and `n_comp_obs`, a count of how many components are observed. That bundle
scores +0.00071 on the strong reference, 5 folds of 5, the largest feature gain in our table.

`n_comp_obs` is Muhammad Faheem's `missing_count` with the sign flipped. He measured it alone and
got +0.00009 out of fold, then read the leaderboard's silence as the gain evaporating. We were
about to tell him it carries nothing, which our own gate cannot support: we never ran the two
columns apart, so we cannot say whose the +0.00071 is.

This separates them. Same folds, same seed, same reference, same target-encoding construction as
gate.py, so the numbers drop straight into that table.

  base                    the strong reference
  +slack                  tamerlanomralinov's column alone
  +n_comp_obs             the observed-component count alone  <- Faheem's quantity
  +slack+n_comp_obs       the published bundle, as a reproduction check against +0.00071
  +placebo                random values on the same NaN pattern, the floor

Run:  python3 gate_split.py
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from gate import SEED, features
from reference import (CAT, RAW, STRONG, STRONG_ROUNDS, TARGET, FOLD_SEED, N_SPLITS, te_columns)

ARMS = {
    "base":               [],
    "+slack":             ["slack"],
    "+n_comp_obs":        ["n_comp_obs"],
    "+slack+n_comp_obs":  ["slack", "n_comp_obs"],
    "+placebo":           ["placebo_ub"],
}


def main() -> None:
    tr = pd.read_csv("data/train.csv", engine="pyarrow")
    y = tr[TARGET].to_numpy()
    feats = features(tr)

    base_x = tr[RAW].copy()
    for c in CAT:
        base_x[c] = base_x[c].astype("category")

    print(f"rows {len(tr):,}   folds {N_SPLITS} @ random_state={FOLD_SEED}   reference: strong_te")
    print(f"n_comp_obs takes values {sorted(set(feats['n_comp_obs']))}, "
          f"mean {feats['n_comp_obs'].mean():.3f} of 3 components observed\n", flush=True)

    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=FOLD_SEED)
    splits = list(skf.split(base_x, y))
    res = {a: [] for a in ARMS}

    t0 = time.time()
    for k, (a, b) in enumerate(splits):
        xa, xb = base_x.iloc[a].reset_index(drop=True), base_x.iloc[b].reset_index(drop=True)
        ta, tb = te_columns(tr.iloc[a], tr.iloc[b], y[a], RAW, SEED)
        xa = pd.concat([xa, ta.reset_index(drop=True)], axis=1)
        xb = pd.concat([xb, tb.reset_index(drop=True)], axis=1)
        for arm, cols in ARMS.items():
            xa2, xb2 = xa.copy(), xb.copy()
            for c in cols:
                xa2[c] = feats[c][a]
                xb2[c] = feats[c][b]
            m = lgb.train({**STRONG, "seed": SEED},
                          lgb.Dataset(xa2, y[a]), num_boost_round=STRONG_ROUNDS)
            res[arm].append(roc_auc_score(y[b], m.predict(xb2)))
        print(f"  fold {k}: " + "  ".join(f"{a_}={res[a_][-1]:.5f}" for a_ in ARMS), flush=True)
    print(f"  ({(time.time() - t0) / 60:.1f} min)\n")

    base = np.array(res["base"])
    out = {}
    print(f"{'arm':<20}{'mean OOF':>10}{'vs base':>11}{'fold signs':>12}")
    for arm in ARMS:
        v = np.array(res[arm])
        d = v - base
        signs = int((d > 0).sum())
        out[arm] = {"mean": float(v.mean()), "delta": float(d.mean()),
                    "signs": signs, "per_fold": [round(x, 6) for x in d]}
        print(f"{arm:<20}{v.mean():>10.5f}{d.mean():>+11.5f}{f'{signs}/{N_SPLITS}':>12}")

    s, n = out["+slack"]["delta"], out["+n_comp_obs"]["delta"]
    both = out["+slack+n_comp_obs"]["delta"]
    print(f"\nthe bundle is {both:+.5f}; its parts alone are {s:+.5f} and {n:+.5f}, "
          f"summing to {s + n:+.5f}")
    print(f"so the two columns {'overlap' if s + n > both else 'add up or better'}: "
          f"bundle minus sum = {both - (s + n):+.5f}")
    json.dump(out, open("gate_split.json", "w"), indent=1)
    print("\nwrote gate_split.json")


if __name__ == "__main__":
    main()
