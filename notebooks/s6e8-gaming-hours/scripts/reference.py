"""A common reference model for S6E8, and a paired gate for measuring features on it.

Why this exists. Every delta published in this competition is measured against a
different baseline: tamerlanomralinov against a 3-fold raw LightGBM at 0.96432,
tomasa2 against his own tuned pipeline, raykkretzschmar against a 74-member stack,
and us against a twelve-column model at 0.96216. None of those numbers are comparable,
and two authors have now shown independently that a weak reference inflates them:

  - tamerlanomralinov: log-ratio budget features, +0.00036 at 3 folds, +0.00001 at 10.
    "With 90% of the data instead of 67%, the trees learn those relationships from
    splits by themselves."
  - our own earlier run: the same collapse on the capacity axis.

The third axis, feature-set richness, is what this file measures. The reference uses
the fold split the public OOF library is frozen on, StratifiedKFold(5, shuffle=True,
random_state=42), so anything measured here is comparable with the field's own numbers.

Target encoding is the lever the field found: encoding a column's exact VALUE, not its
magnitude, picks up the generator's quantisation lattice. Credit: tomasa2 measured it at
+0.0023 CV, tamerlanomralinov ships it as "value-level target encoding". Implemented here
with inner-fold out-of-fold statistics so the training rows never see their own label.

Usage:
  python3 reference.py --quick          one fold, sanity check the pipeline
  python3 reference.py --arms base,te   full 5-fold OOF for the named arms
"""

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "addicted_label"
DST, SM, WK = "daily_screen_time_hours", "social_media_hours", "weekend_screen_time"
GAM, WRK = "gaming_hours", "work_study_hours"
NUM = ["age", DST, SM, GAM, WRK, "sleep_hours", "notifications_per_day",
       "app_opens_per_day", WK]
CAT = ["gender", "stress_level", "academic_work_impact"]
RAW = NUM + CAT
RESID = "other_screen"

# The public OOF library is frozen on this split. Matching it is the whole point.
FOLD_SEED = 42
N_SPLITS = 5
SMOOTH = 10.0
INNER = 5

STRONG = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=255,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbosity=-1, num_threads=8)
WEAK = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
            min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.9,
            bagging_freq=1, verbosity=-1, num_threads=8)
STRONG_ROUNDS, WEAK_ROUNDS = 1500, 400


def te_columns(tr_df, va_df, y_tr, cols, seed):
    """Out-of-fold target encoding of the exact value of each column.

    Validation rows use statistics from the whole training part. Training rows use
    statistics from an inner split that excludes them, so no row is encoded with its
    own label. Unseen values fall back to the training prior.
    """
    prior = y_tr.mean()
    out_tr = pd.DataFrame(index=tr_df.index)
    out_va = pd.DataFrame(index=va_df.index)
    inner = StratifiedKFold(INNER, shuffle=True, random_state=seed)
    inner_splits = list(inner.split(tr_df, y_tr))

    for c in cols:
        key_tr = tr_df[c].astype("object")
        key_va = va_df[c].astype("object")

        # validation side: full training part
        g = pd.DataFrame({"k": key_tr, "y": y_tr}).groupby("k")["y"].agg(["sum", "count"])
        enc_full = (g["sum"] + prior * SMOOTH) / (g["count"] + SMOOTH)
        out_va["te_" + c] = key_va.map(enc_full).astype(float).fillna(prior).to_numpy()

        # training side: inner out-of-fold
        col = np.full(len(tr_df), np.nan)
        for a, b in inner_splits:
            gi = pd.DataFrame({"k": key_tr.iloc[a], "y": y_tr[a]}).groupby("k")["y"].agg(["sum", "count"])
            enc_i = (gi["sum"] + prior * SMOOTH) / (gi["count"] + SMOOTH)
            col[b] = key_tr.iloc[b].map(enc_i).astype(float).fillna(prior).to_numpy()
        out_tr["te_" + c] = col
    return out_tr, out_va


def build(tr):
    x = tr[RAW].copy()
    for c in CAT:
        x[c] = x[c].astype("category")
    x[RESID] = tr[DST] - (tr[SM] + tr[GAM] + tr[WRK])
    return x


ARMS = {
    # name          -> (params, rounds, use target encoding)
    "weak_raw":      (WEAK, WEAK_ROUNDS, False),
    "strong_raw":    (STRONG, STRONG_ROUNDS, False),
    "strong_te":     (STRONG, STRONG_ROUNDS, True),
}


def run_arm(tr, y, name, seed=0, quick=False, extra=None):
    params, rounds, use_te = ARMS[name]
    x = build(tr)
    if extra is not None:
        for k, v in extra.items():
            x[k] = v
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=FOLD_SEED)
    oof = np.full(len(y), np.nan)
    per_fold = []
    for k, (a, b) in enumerate(skf.split(x, y)):
        if quick and k > 0:
            break
        xa, xb = x.iloc[a], x.iloc[b]
        if use_te:
            ta, tb = te_columns(tr.iloc[a], tr.iloc[b], y[a], RAW, seed)
            xa = pd.concat([xa.reset_index(drop=True), ta.reset_index(drop=True)], axis=1)
            xb = pd.concat([xb.reset_index(drop=True), tb.reset_index(drop=True)], axis=1)
        m = lgb.train({**params, "seed": seed}, lgb.Dataset(xa, y[a]), num_boost_round=rounds)
        p = m.predict(xb)
        oof[b] = p
        per_fold.append(roc_auc_score(y[b], p))
        print(f"    fold {k}  {per_fold[-1]:.5f}", flush=True)
    score = float(np.mean(per_fold)) if quick else roc_auc_score(y, oof)
    return score, per_fold, oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--arms", default="weak_raw,strong_raw,strong_te")
    args = ap.parse_args()

    tr = pd.read_csv("data/train.csv", engine="pyarrow")
    y = tr[TARGET].to_numpy()
    print(f"rows {len(tr):,}  folds {N_SPLITS} @ random_state={FOLD_SEED} "
          f"(the public OOF library's split)\n")

    out = {}
    for name in args.arms.split(","):
        t0 = time.time()
        print(f"  {name}:", flush=True)
        score, folds, oof = run_arm(tr, y, name, quick=args.quick)
        out[name] = {"score": score, "folds": folds, "minutes": (time.time() - t0) / 60}
        print(f"  {name:<12} OOF {score:.5f}   {out[name]['minutes']:.1f} min\n", flush=True)
        if not args.quick:
            np.save(f"oof_{name}.npy", oof)

    json.dump(out, open("reference_quick.json" if args.quick else "reference.json", "w"), indent=1)
    for k, v in out.items():
        print(f"{k:<12}{v['score']:.5f}")


if __name__ == "__main__":
    main()
