"""Measure the same features against a weak and a strong reference, on identical folds.

The question this answers: how much of a published delta is the feature, and how much is
the baseline being too weak to have found the same structure by itself?

Two authors have measured that inflation on two axes already and published it:
  - fold count      tamerlanomralinov, log-ratio budget features +0.00036 at 3 folds, +0.00001 at 10
  - model capacity  our own earlier run
This measures the third: feature-set richness. Same folds, same seed, one thing changed.

Arms are paired per fold, so the difference is a paired statistic and the fold-to-fold
noise cancels. A placebo column with the same missingness pattern and random values runs
alongside: whatever it returns is the floor, and any feature that does not clear its own
placebo has not been shown to do anything.

Credit for the features under test:
  other_screen   Dariush Afshar, discussion 732223, on ryota517's budget constraint
  slack/n_comp   tamerlanomralinov, who prices it at +0.00001 on his own reference
  social_ub      ours, the one-sided bound the constraint gives when social is missing
"""

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from reference import (CAT, DST, GAM, RAW, SM, SMOOTH, STRONG, STRONG_ROUNDS, TARGET,
                       WEAK, WEAK_ROUNDS, WK, WRK, FOLD_SEED, N_SPLITS, te_columns)

SEED = 0
COMP = [SM, GAM, WRK]


def features(tr):
    """Every candidate, plus the placebo. Returned as plain float arrays."""
    d, s, g, w = tr[DST], tr[SM], tr[GAM], tr[WRK]
    rng = np.random.default_rng(7)

    other = d - (s + g + w)                       # exists only when all four do
    slack = d - tr[COMP].fillna(0).sum(axis=1)    # tamerlan's: defined whenever daily is
    n_comp = tr[COMP].notna().sum(axis=1).astype(float)

    # When social is missing, the budget still bounds it from above: social <= daily - gaming - work.
    # Masked to NaN where social is observed, so the column only speaks where it knows something.
    ub = d - (g.fillna(0) + w.fillna(0))
    social_ub = ub.where(s.isna())

    # Placebo: same NaN pattern as social_ub, values carry nothing.
    placebo = pd.Series(rng.normal(size=len(tr)), index=tr.index).where(s.isna())

    return {
        "other_screen": other.to_numpy(dtype=float),
        "slack": slack.to_numpy(dtype=float),
        "n_comp_obs": n_comp.to_numpy(dtype=float),
        "social_ub": social_ub.to_numpy(dtype=float),
        "placebo_ub": placebo.to_numpy(dtype=float),
    }


ARMS = {
    "base":         [],
    "+other_screen": ["other_screen"],
    "+slack":        ["slack", "n_comp_obs"],
    "+social_ub":    ["social_ub"],
    "+placebo":      ["placebo_ub"],
}

REFS = {
    "weak_raw":  dict(params=WEAK,   rounds=WEAK_ROUNDS,   te=False),
    "strong_te": dict(params=STRONG, rounds=STRONG_ROUNDS, te=True),
}


def main() -> None:
    tr = pd.read_csv("data/train.csv", engine="pyarrow")
    y = tr[TARGET].to_numpy()
    feats = features(tr)

    base_x = tr[RAW].copy()
    for c in CAT:
        base_x[c] = base_x[c].astype("category")

    print(f"rows {len(tr):,}   folds {N_SPLITS} @ random_state={FOLD_SEED}")
    print(f"social_ub defined on {np.isfinite(feats['social_ub']).sum():,} rows "
          f"(social missing); other_screen on {np.isfinite(feats['other_screen']).sum():,}\n")

    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=FOLD_SEED)
    splits = list(skf.split(base_x, y))
    results = {r: {a: [] for a in ARMS} for r in REFS}

    for ref_name, cfg in REFS.items():
        t0 = time.time()
        print(f"=== reference: {ref_name} ===", flush=True)
        for k, (a, b) in enumerate(splits):
            xa, xb = base_x.iloc[a].reset_index(drop=True), base_x.iloc[b].reset_index(drop=True)
            if cfg["te"]:
                ta, tb = te_columns(tr.iloc[a], tr.iloc[b], y[a], RAW, SEED)
                xa = pd.concat([xa, ta.reset_index(drop=True)], axis=1)
                xb = pd.concat([xb, tb.reset_index(drop=True)], axis=1)
            for arm, cols in ARMS.items():
                xa2, xb2 = xa.copy(), xb.copy()
                for c in cols:
                    xa2[c] = feats[c][a]
                    xb2[c] = feats[c][b]
                m = lgb.train({**cfg["params"], "seed": SEED},
                              lgb.Dataset(xa2, y[a]), num_boost_round=cfg["rounds"])
                s = roc_auc_score(y[b], m.predict(xb2))
                results[ref_name][arm].append(s)
            print(f"  fold {k}: " + "  ".join(
                f"{a_}={results[ref_name][a_][-1]:.5f}" for a_ in ARMS), flush=True)
        print(f"  ({(time.time()-t0)/60:.1f} min)\n", flush=True)

    print(f"{'reference':<12}{'arm':<16}{'mean OOF':>10}{'vs base':>10}{'fold signs':>12}")
    summary = {}
    for ref_name in REFS:
        base = np.array(results[ref_name]["base"])
        for arm in ARMS:
            v = np.array(results[ref_name][arm])
            d = v - base
            pos = int((d > 0).sum())
            summary[f"{ref_name}|{arm}"] = dict(mean=float(v.mean()), delta=float(d.mean()),
                                                signs=pos, per_fold=d.round(6).tolist())
            print(f"{ref_name:<12}{arm:<16}{v.mean():>10.5f}{d.mean():>+10.5f}"
                  f"{(str(pos)+'/'+str(N_SPLITS)):>12}")

    print("\ninflation factor, weak delta divided by strong delta:")
    for arm in ARMS:
        if arm == "base":
            continue
        dw = summary[f"weak_raw|{arm}"]["delta"]
        ds = summary[f"strong_te|{arm}"]["delta"]
        ratio = (dw / ds) if abs(ds) > 1e-6 else float("inf")
        print(f"  {arm:<16} weak {dw:+.5f}   strong {ds:+.5f}   ratio {ratio:>7.1f}x")

    json.dump(summary, open("gate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
