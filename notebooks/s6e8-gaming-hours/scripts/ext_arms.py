"""Does adding the two compatible libraries move the stack, or did the screen already answer it?

ext_gate.py measured the proxy: none of the twelve usable newcomers is more decorrelated from the
74 than the library's own most decorrelated member (0.9468). That is a screen and not a verdict.
The five factorization machines sit at 0.9847 to 0.9886, which is exactly where `lookup` sits
(0.9869), and `lookup` was worth +0.000109 to szymonkapiski's blend. A proxy that ranks a member
below the best in the library says nothing about whether it is above zero.

So: run the same nested logistic stack from section 10, on the same frozen folds, over four arms.

    base        the 74
    +fm         plus the five aligned factorization machines
    +golem      plus dariushafshar's seven
    +both       all 86

Everything is paired: identical folds, identical seed, the same rank transform, and the deltas are
reported per fold rather than as a single difference of means, because one split can and does give
a confident answer in the wrong direction.

The control that can fail: `+placebo` adds five columns of pure noise, matched in shape and density
to the factorization machines. If the stack "gains" from those too, the apparatus is measuring its
own extra capacity and not the members.

Run:  python3 ext_arms.py
"""
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
FOLD_SEED, N_SPLITS = 42, 5
N_TRAIN = 691369


def ranks(v):
    o = np.argsort(v, kind="stable")
    r = np.empty(len(v), dtype=np.float32)
    r[o] = np.linspace(0, 1, len(v), dtype=np.float32)
    return r


def load_ext(folder, names):
    out = {}
    for n in names:
        o = np.load(os.path.join(folder, f"oof_{n}.npy"))
        if len(o) == N_TRAIN:
            out[n] = ranks(o)
    return out


def main() -> None:
    y = pd.read_parquet("oof_lib/train_keys.parquet")["addicted_label"].to_numpy()
    base_f = sorted(glob.glob("oof_lib/oof/oof_*.npy"))
    P = np.stack([ranks(np.load(f)) for f in base_f])
    print(f"base {P.shape[0]} members, {P.shape[1]:,} rows")

    fm = load_ext("ext/fm", ["fmpure", "fmnum", "fmwide", "fmplr", "fmdeep"])
    gl = load_ext("ext/golem", list("abcdefg"))
    print(f"aligned newcomers: fm {len(fm)}, golem {len(gl)}")

    rng = np.random.default_rng(20260817)
    placebo = np.stack([ranks(rng.normal(size=N_TRAIN)) for _ in range(5)])

    ARMS = {
        "base": P,
        "+fm": np.vstack([P, np.stack(list(fm.values()))]),
        "+golem": np.vstack([P, np.stack(list(gl.values()))]),
        "+both": np.vstack([P, np.stack(list(fm.values())), np.stack(list(gl.values()))]),
        "+placebo": np.vstack([P, placebo]),
    }

    folds = np.empty(len(y), int)
    for k, (_, b) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True,
                                               random_state=FOLD_SEED).split(P.T, y)):
        folds[b] = k
    idx = np.arange(len(y))

    per_fold, overall = {}, {}
    for name, M in ARMS.items():
        scores, oof = [], np.zeros(len(y))
        for k in range(N_SPLITS):
            ti, vi = idx[folds != k], idx[folds == k]
            lr = LogisticRegression(max_iter=1000).fit(M[:, ti].T, y[ti])
            oof[vi] = lr.decision_function(M[:, vi].T)
            scores.append(roc_auc_score(y[vi], oof[vi]))
        per_fold[name] = np.array(scores)
        overall[name] = roc_auc_score(y, oof)
        print(f"  {name:<10} {M.shape[0]:>3} members   pooled OOF {overall[name]:.6f}")

    print(f"\n{'arm':<10}{'pooled':>10}{'vs base':>11}{'folds up':>10}{'paired SE':>11}{'95% CI':>22}")
    out = []
    for name in ARMS:
        if name == "base":
            print(f"{name:<10}{overall[name]:10.6f}{'--':>11}{'--':>10}{'--':>11}{'--':>22}")
            continue
        d = per_fold[name] - per_fold["base"]
        se = d.std(ddof=1) / np.sqrt(N_SPLITS)
        lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
        print(f"{name:<10}{overall[name]:10.6f}{overall[name] - overall['base']:+11.6f}"
              f"{int((d > 0).sum()):>7} of 5{se:11.6f}   [{lo:+.6f}, {hi:+.6f}]")
        out.append({"arm": name, "pooled": overall[name],
                    "delta": overall[name] - overall["base"],
                    "folds_up": int((d > 0).sum()), "per_fold": d.tolist(),
                    "se": float(se), "ci": [float(lo), float(hi)]})

    print("\nsection 11 puts the resolvable difference between two blends of this library at "
          "about 0.00002,")
    print("so read anything under that as unmeasured rather than as zero.")
    json.dump({"overall": overall, "arms": out,
               "per_fold": {k: v.tolist() for k, v in per_fold.items()}},
              open("ext_arms.json", "w"), indent=1)
    print("wrote ext_arms.json")


if __name__ == "__main__":
    main()
