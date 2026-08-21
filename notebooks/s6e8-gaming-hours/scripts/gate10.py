"""Does `slack` survive at ten folds, the fold count its author used?

Our five-fold gate prices tamerlanomralinov's constraint-slack feature at +0.00071 on a
strong target-encoded reference, 5/5 fold signs. He prices the same feature at +0.00001.
Seventy times apart, so one of the two measurements is answering a different question.

The most likely explanation is his, and it is one he published himself: his log-ratio
budget features went +0.00036 at three folds to +0.00001 at ten, because "with 90% of the
data instead of 67%, the trees learn those relationships from splits by themselves". If
slack does the same between five and ten, he is right and we were measuring a crutch.

Same reference, same features, same seed. Only the fold count changes.
"""
import json, time
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from reference import CAT, RAW, STRONG, STRONG_ROUNDS, TARGET, FOLD_SEED, te_columns
from gate import features

SEED, N = 0, 10
tr = pd.read_csv("data/train.csv", engine="pyarrow")
y = tr[TARGET].to_numpy()
feats = features(tr)
base_x = tr[RAW].copy()
for c in CAT:
    base_x[c] = base_x[c].astype("category")

ARMS = {"base": [], "+slack": ["slack", "n_comp_obs"], "+other_screen": ["other_screen"],
        "+placebo": ["placebo_ub"]}
res = {a: [] for a in ARMS}
skf = StratifiedKFold(N, shuffle=True, random_state=FOLD_SEED)
t0 = time.time()
for k, (a, b) in enumerate(skf.split(base_x, y)):
    xa = base_x.iloc[a].reset_index(drop=True)
    xb = base_x.iloc[b].reset_index(drop=True)
    ta, tb = te_columns(tr.iloc[a], tr.iloc[b], y[a], RAW, SEED)
    xa = pd.concat([xa, ta.reset_index(drop=True)], axis=1)
    xb = pd.concat([xb, tb.reset_index(drop=True)], axis=1)
    for arm, cols in ARMS.items():
        x1, x2 = xa.copy(), xb.copy()
        for c in cols:
            x1[c] = feats[c][a]; x2[c] = feats[c][b]
        m = lgb.train({**STRONG, "seed": SEED}, lgb.Dataset(x1, y[a]),
                      num_boost_round=STRONG_ROUNDS)
        res[arm].append(roc_auc_score(y[b], m.predict(x2)))
    print(f"  fold {k}: " + "  ".join(f"{a_}={res[a_][-1]:.5f}" for a_ in ARMS), flush=True)

print(f"\n({(time.time()-t0)/60:.1f} min)   ten folds, strong target-encoded reference\n")
base = np.array(res["base"])
out = {}
for arm in ARMS:
    v = np.array(res[arm]); d = v - base
    out[arm] = dict(mean=float(v.mean()), delta=float(d.mean()), signs=int((d > 0).sum()))
    print(f"{arm:<16}{v.mean():.5f}   {d.mean():+.5f}   signs {int((d>0).sum())}/{N}")
json.dump(out, open("gate10.json", "w"), indent=1)
