"""Does a nonlinear stacker beat the logistic one on the same frozen folds, honestly?

The nested logistic stack is our best measured combiner (pooled OOF 0.969673 on 86 members).
A logistic stack cannot use interactions between members; a GBDT stacker can. It can also
overfit the OOF matrix far more eagerly, so the placebo arm matters more here, not less.

Arms, all on identical frozen folds (StratifiedKFold 5, shuffle, seed 42), all paired:

    logit86        the production combiner (baseline)
    lgbm86         LightGBM over the same 86 rank-transformed OOF columns
    lgbm86+plac    same, plus 5 pure-noise columns: the overfit detector
    lgbm+logit     mean of the two combiners' per-fold rank predictions: the usual free lunch

Read: an arm is real if delta > 0 on 5/5 folds AND the placebo arm shows the noise columns
did not help. Section 11's floor: differences under ~0.00002 are unmeasured, not zero.

Run:  python3 stacker_arms.py
"""
import glob, json, os, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")
FOLD_SEED, N_SPLITS, N_TRAIN = 42, 5, 691369

def ranks(v):
    o = np.argsort(v, kind="stable"); r = np.empty(len(v), dtype=np.float32)
    r[o] = np.linspace(0, 1, len(v), dtype=np.float32); return r

y = pd.read_parquet("oof_lib/train_keys.parquet")["addicted_label"].to_numpy()
base_f = sorted(glob.glob("oof_lib/oof/oof_*.npy"))
cols = [ranks(np.load(f)) for f in base_f]
for folder, ms in (("ext/fm", ["fmpure","fmnum","fmwide","fmplr","fmdeep"]),
                   ("ext/golem", list("abcdefg"))):
    for m in ms:
        o = np.load(f"{folder}/oof_{m}.npy"); assert len(o) == N_TRAIN
        cols.append(ranks(o))
P = np.stack(cols).T.astype(np.float32)            # rows x 86
rng = np.random.default_rng(20260817)
PLAC = np.stack([ranks(rng.normal(size=N_TRAIN)) for _ in range(5)]).T.astype(np.float32)
print(f"members {P.shape[1]}, rows {P.shape[0]:,}")

folds = np.empty(N_TRAIN, int)
for k, (_, b) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True,
                                           random_state=FOLD_SEED).split(P, y)):
    folds[b] = k
idx = np.arange(N_TRAIN)

LPAR = dict(n_estimators=700, learning_rate=0.05, num_leaves=63, colsample_bytree=0.8,
            subsample=0.8, subsample_freq=1, min_child_samples=200, reg_lambda=1.0,
            random_state=0, n_jobs=8, verbose=-1)

def run(name, fit_predict):
    scores, oof = [], np.zeros(N_TRAIN)
    for k in range(N_SPLITS):
        ti, vi = idx[folds != k], idx[folds == k]
        oof[vi] = fit_predict(ti, vi)
        scores.append(roc_auc_score(y[vi], oof[vi]))
    pooled = roc_auc_score(y, oof)
    print(f"  {name:<14} pooled {pooled:.6f}   folds {[f'{s:.5f}' for s in scores]}")
    return np.array(scores), pooled, oof

def logit_fp(M):
    def fp(ti, vi):
        lr = LogisticRegression(max_iter=1000).fit(M[ti], y[ti])
        return lr.decision_function(M[vi])
    return fp

def lgbm_fp(M):
    def fp(ti, vi):
        # inner 10% of the training part as early-stopping watch, never the validation fold
        cut = int(len(ti) * 0.9)
        tr, wa = ti[:cut], ti[cut:]
        mdl = lgb.LGBMClassifier(**LPAR)
        mdl.fit(M[tr], y[tr], eval_set=[(M[wa], y[wa])],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        return mdl.predict_proba(M[vi], num_iteration=mdl.best_iteration_)[:, 1]
    return fp

res = {}
res["logit86"] = run("logit86", logit_fp(P))
res["lgbm86"] = run("lgbm86", lgbm_fp(P))
res["lgbm86+plac"] = run("lgbm86+plac", lgbm_fp(np.hstack([P, PLAC])))

# rank-mean of the two combiners, per fold (each fold's ranks computed within the fold)
sc, oof = [], np.zeros(N_TRAIN)
for k in range(N_SPLITS):
    vi = idx[folds == k]
    oof[vi] = ranks(res["logit86"][2][vi]) + ranks(res["lgbm86"][2][vi])
    sc.append(roc_auc_score(y[vi], oof[vi]))
res["lgbm+logit"] = (np.array(sc), roc_auc_score(y, oof), oof)
print(f"  {'lgbm+logit':<14} pooled {res['lgbm+logit'][1]:.6f}")

base = res["logit86"][0]
print(f"\n{'arm':<14}{'pooled':>10}{'vs logit86':>12}{'folds up':>10}{'95% CI':>26}")
out = {}
for name, (s, pooled, _) in res.items():
    if name == "logit86":
        print(f"{name:<14}{pooled:10.6f}{'--':>12}{'--':>10}{'--':>26}")
        continue
    d = s - base; se = d.std(ddof=1) / np.sqrt(N_SPLITS)
    lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
    print(f"{name:<14}{pooled:10.6f}{pooled - res['logit86'][1]:+12.6f}"
          f"{int((d > 0).sum()):>7}/5{'':>3}[{lo:+.6f}, {hi:+.6f}]")
    out[name] = {"pooled": pooled, "delta": pooled - res["logit86"][1],
                 "folds_up": int((d > 0).sum()), "ci": [float(lo), float(hi)]}
print("\nplacebo read: lgbm86+plac minus lgbm86 should be <= 0; if it is positive the GBDT is "
      "mining noise and NOTHING from this table ships.")
json.dump(out, open("stacker_arms.json", "w"), indent=1)
np.save("oof_stack_lgbm.npy", res["lgbm86"][2]); np.save("oof_stack_logit.npy", res["logit86"][2])
print("wrote stacker_arms.json + oof arrays")
