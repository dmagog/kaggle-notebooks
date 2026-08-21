"""The full public pool, gated: does it deliver the +0.0004 the recon priced, and honestly?

Sources and their fold status:
  base86    our production pool (szymonkapiski 74 + fm 5 + golem 7)      frozen, measured
  adarsh22  adarsh1077/s6e8-adarsh-oof-library                            frozen, README verbatim,
                                                                          spot-verified
  mohan3    mohankrishnathalla xgb/cat/lgb v3                             xgb PROVEN per-fold,
                                                                          cat/lgb declared
  bolt47    boltuzamaki/s6e8-oof-prediction-library                       UNPROVEN ("five or ten
                                                                          fold" in his own words)
bolt stays its own arm: if its CV gain is leak, the LB probe will land short of the predicted
offset (+0.00108, stable to 3e-5 across our two submissions), and that shortfall is decisive.

Hygiene first (adarsh's checklist, measured on his pool): byte-level dedupe, KS drift screen on
rank scale (drop > 0.05), lbfgs convergence asserted. Then two measured dimensions:
  step 1: combiner space on base86 - ranks vs rank-gauss, C grid {1, 0.1, 0.03}
  step 2: pool arms in the winning config, paired folds, placebo control

Run:  python3 pool_arms.py
"""
import glob, hashlib, json, os, warnings
import numpy as np, pandas as pd
from scipy.stats import norm, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
FOLD_SEED, N_SPLITS, N_TRAIN, N_TEST = 42, 5, 691369, 296302

def ranks(v):
    o = np.argsort(v, kind="stable"); r = np.empty(len(v), dtype=np.float64)
    r[o] = np.linspace(0, 1, len(v), dtype=np.float64); return r

def load_all():
    """name -> (oof, test, source). float64 throughout; adarsh measured float32 flipping ranks."""
    out = {}
    def add(name, o, t, src):
        assert len(o) == N_TRAIN and len(t) == N_TEST, (name, len(o), len(t))
        out[name] = (o.astype(np.float64), t.astype(np.float64), src)
    for f in sorted(glob.glob("oof_lib/oof/oof_*.npy")):
        n = os.path.basename(f)[4:-4]
        add(f"lib:{n}", np.load(f), np.load(f"oof_lib/oof/test_{n}.npy"), "lib")
    for m in ["fmpure","fmnum","fmwide","fmplr","fmdeep"]:
        add(f"fm:{m}", np.load(f"ext/fm/oof_{m}.npy"), np.load(f"ext/fm/test_{m}.npy"), "fm")
    for m in list("abcdefg"):
        add(f"golem:{m}", np.load(f"ext/golem/oof_{m}.npy"),
            np.load(f"ext/golem/test_{m}.npy"), "golem")
    for f in sorted(glob.glob("ext/adarsh/oof_*.npy")):
        n = os.path.basename(f)[4:-4]
        add(f"adarsh:{n}", np.load(f), np.load(f"ext/adarsh/test_{n}.npy"), "adarsh")
    for m in ["xgb_v3","cat_v3","lgb_v3"]:
        add(f"mohan:{m}", np.load(f"ext/mohan/oof_{m}.npy"),
            np.load(f"ext/mohan/test_{m}.npy"), "mohan")
    ob = pd.read_parquet("ext/bolt/oof_predictions.parquet")
    tb = pd.read_parquet("ext/bolt/test_predictions.parquet")
    sc = [c for c in ob.columns if c in tb.columns and pd.api.types.is_numeric_dtype(ob[c])]
    for c in sc:
        add(f"bolt:{c}", ob[c].to_numpy(), tb[c].to_numpy(), "bolt")
    return out

pool = load_all()
print(f"loaded {len(pool)} members")

# -------- hygiene 1: byte-level dedupe (keep first in source priority order)
prio = {"lib":0, "fm":1, "golem":2, "adarsh":3, "mohan":4, "bolt":5}
seen, drop = {}, []
for name in sorted(pool, key=lambda n: (prio[pool[n][2]], n)):
    h = hashlib.md5(pool[name][0].tobytes()).hexdigest()
    if h in seen: drop.append((name, seen[h]))
    else: seen[h] = name
for name, twin in drop:
    del pool[name]
print(f"dedupe: dropped {len(drop)} byte-identical: {drop}")

# -------- hygiene 2: KS drift screen on rank scale, OOF vs test, per member
ksdrop = []
for name, (o, t, src) in list(pool.items()):
    ks = ks_2samp(ranks(o), ranks(t)).statistic
    if ks > 0.05:
        ksdrop.append((name, round(float(ks), 3))); del pool[name]
print(f"KS>0.05 dropped {len(ksdrop)}: {ksdrop}")

y = pd.read_parquet("oof_lib/train_keys.parquet")["addicted_label"].to_numpy()
names = sorted(pool, key=lambda n: (prio[pool[n][2]], n))
SRC = {n: pool[n][2] for n in names}
R_OOF = {n: ranks(pool[n][0]) for n in names}
print({s: sum(1 for n in names if SRC[n]==s) for s in prio})

folds = np.empty(N_TRAIN, int)
dummy = np.zeros((N_TRAIN, 1))
for k, (_, b) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True,
                                           random_state=FOLD_SEED).split(dummy, y)):
    folds[b] = k
idx = np.arange(N_TRAIN)

def gauss(M):    # rank-gauss: norm.ppf of percentile ranks, adarsh's meta space
    return norm.ppf(np.clip(M, 1e-6, 1 - 1e-6))

def nested(M, C):
    scores, oof = [], np.zeros(N_TRAIN)
    for k in range(N_SPLITS):
        ti, vi = idx[folds != k], idx[folds == k]
        lr = LogisticRegression(C=C, max_iter=3000).fit(M[ti], y[ti])
        assert lr.n_iter_.max() < 3000, "lbfgs did not converge; a non-converged fit reads HIGH"
        oof[vi] = lr.decision_function(M[vi])
        scores.append(roc_auc_score(y[vi], oof[vi]))
    return np.array(scores), roc_auc_score(y, oof), oof

def member_matrix(subset):
    return np.stack([R_OOF[n] for n in subset]).T

BASE86 = [n for n in names if SRC[n] in ("lib","fm","golem")]
M86 = member_matrix(BASE86)

print("\n=== step 1: combiner space on base86 ===")
step1 = {}
for label, M, C in [("rank C=1", M86, 1.0), ("rank C=0.03", M86, 0.03),
                    ("gauss C=1", gauss(M86), 1.0), ("gauss C=0.1", gauss(M86), 0.1),
                    ("gauss C=0.03", gauss(M86), 0.03)]:
    s, p, _ = nested(M, C)
    step1[label] = {"pooled": p, "folds": s.tolist()}
    print(f"  {label:<14} pooled {p:.6f}")
best1 = max(step1, key=lambda k: step1[k]["pooled"])
print(f"best combiner space: {best1}")
use_gauss = best1.startswith("gauss")
bestC = float(best1.split("C=")[1])

print(f"\n=== step 2: pool arms in [{best1}] ===")
rng = np.random.default_rng(20260818)
plac = np.stack([ranks(rng.normal(size=N_TRAIN)) for _ in range(5)]).T
ARMS = {
    "base86": BASE86,
    "+adarsh": BASE86 + [n for n in names if SRC[n]=="adarsh"],
    "+mohan": BASE86 + [n for n in names if SRC[n]=="mohan"],
    "+verified": BASE86 + [n for n in names if SRC[n] in ("adarsh","mohan")],
    "+bolt(UNPROVEN)": BASE86 + [n for n in names if SRC[n]=="bolt"],
    "+all": names,
}
res = {}
for arm, subset in ARMS.items():
    M = member_matrix(subset)
    if use_gauss: M = gauss(M)
    res[arm] = nested(M, bestC)
    print(f"  {arm:<16} {len(subset):>3} members   pooled {res[arm][1]:.6f}")
Mp = member_matrix(BASE86)
Mp = np.hstack([Mp, plac])
if use_gauss: Mp = gauss(Mp)
res["+placebo"] = nested(Mp, bestC)
print(f"  {'+placebo':<16} {len(BASE86)+5:>3} members   pooled {res['+placebo'][1]:.6f}")

base = res["base86"][0]
print(f"\n{'arm':<17}{'pooled':>10}{'vs base86':>12}{'folds up':>9}{'95% CI':>26}")
out = {}
for arm, (s, pooled, _) in res.items():
    if arm == "base86":
        print(f"{arm:<17}{pooled:10.6f}{'--':>12}{'--':>9}{'--':>26}"); continue
    d = s - base; se = d.std(ddof=1)/np.sqrt(N_SPLITS)
    lo, hi = d.mean()-1.96*se, d.mean()+1.96*se
    print(f"{arm:<17}{pooled:10.6f}{pooled-res['base86'][1]:+12.6f}"
          f"{int((d>0).sum()):>6}/5{'':>2}[{lo:+.6f}, {hi:+.6f}]")
    out[arm] = {"pooled": pooled, "delta": pooled-res["base86"][1],
                "folds_up": int((d>0).sum()), "ci": [float(lo), float(hi)]}

json.dump({"step1": step1, "best1": best1, "arms": out,
           "dropped_dupes": drop, "dropped_ks": ksdrop,
           "n_members": {a: len(s) for a, s in ARMS.items()}},
          open("pool_arms.json", "w"), indent=1)
print("\nwrote pool_arms.json")
