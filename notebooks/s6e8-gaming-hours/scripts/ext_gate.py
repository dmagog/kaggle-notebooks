"""Two more published OOF libraries: are they new information, or more of the same?

szymonkapiski's library is 74 models on StratifiedKFold(5, shuffle=True, random_state=42). Two
other authors publish on the SAME declared partition, so their arrays can be stacked against it
without leaking:

    dariushafshar/s6e8-golem-oof-library          7 models, a..g
    raykkretzschmar/s6e8-fm-lattice-blend-members 7 factorization machines

Two others could not be used. adarsh1077's library ships no description at all, so its fold scheme
is unknown; najiama's says "5-Fold K-Fold" without stratification or a seed. Kretzschmar's own card
gives the reason to care: most published S6E8 OOF arrays use a different fold count or average over
seeds, and mixing partitions "will quietly inflate a blend built on the 5-fold split."

WHAT DECIDES THIS. Not the new members' own AUC. szymonkapiski's own evidence is that the payer was
decorrelation rather than strength: `lookup` scores 0.96853, below his strongest, but its maximum
correlation against every other member is 0.9869 where the rest of the pack sits between 0.987 and
0.999, and adding it alone was worth +0.000109. Meanwhile `rmlp_lat3` gained +0.00015 solo and
exactly 0.000000 in the blend, because it correlates 0.9994 with its own parent.

So the number to look at first is max correlation against the existing 74. Factorization machines
are a different family from the GBDT and neural pack, which is why they are the interesting half.

This script only measures and gates. It does not build a submission.

Run:  python3 ext_gate.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

N_TRAIN, N_TEST = 691369, 296302
EXT = {"golem": "ext/golem", "fm": "ext/fm"}


def ranks(v):
    """Rank-transform to [0,1]: AUC is rank-only and the members are not calibrated alike."""
    o = np.argsort(v, kind="stable")
    r = np.empty(len(v), dtype=np.float32)
    r[o] = np.linspace(0, 1, len(v), dtype=np.float32)
    return r


def auc(y, s):
    """AUC via the rank identity, cheaper than sklearn on 691k rows repeated 90 times."""
    r = pd.Series(s).rank().to_numpy()
    npos, nneg = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def collect(folder):
    """Return {name: (oof_path, test_path)} for every complete pair, whatever the prefix."""
    out = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.npy"))):
        b = os.path.basename(f)[:-4]
        for opre, tpre in (("oof_", "test_"), ("bandoof_", "bandtest_")):
            if b.startswith(opre):
                t = os.path.join(folder, tpre + b[len(opre):] + ".npy")
                if os.path.exists(t):
                    out[b[len(opre):]] = (f, t)
    return out


def main() -> None:
    y = pd.read_parquet("oof_lib/train_keys.parquet")["addicted_label"].to_numpy()
    assert len(y) == N_TRAIN

    base_f = sorted(glob.glob("oof_lib/oof/oof_*.npy"))
    base = [os.path.basename(f)[4:-4] for f in base_f]
    P = np.stack([ranks(np.load(f)) for f in base_f])
    print(f"base library: {len(base)} members, {P.shape[1]:,} rows")

    rows = []
    for lib, folder in EXT.items():
        pairs = collect(folder)
        print(f"\n{lib}: {len(pairs)} complete oof/test pairs -> {sorted(pairs)}")
        for name, (fo, ft) in sorted(pairs.items()):
            o, t = np.load(fo), np.load(ft)
            if len(o) != N_TRAIN or len(t) != N_TEST:
                print(f"  SKIP {name}: shapes {o.shape} / {t.shape}, not aligned to this split")
                continue
            r = ranks(o)
            c = np.abs(np.corrcoef(np.vstack([r, P]))[0, 1:])   # against every base member
            rows.append({"library": lib, "member": name, "oof auc": auc(y, o),
                         "max corr vs the 74": c.max(),
                         "closest member": base[int(np.argmax(c))],
                         "mean corr": c.mean()})

    # The same statistic for the base library itself, so the new members have something to be
    # compared against rather than a bare number. This is the control: if the newcomers sit
    # inside the pack's own range, they carry nothing the pack does not already have.
    inner = []
    for i in range(len(base)):
        c = np.abs(np.corrcoef(np.vstack([P[i], np.delete(P, i, axis=0)]))[0, 1:])
        inner.append(c.max())
    inner = np.array(inner)

    df = pd.DataFrame(rows).sort_values("max corr vs the 74")
    pd.set_option("display.width", 200)
    print("\n" + "=" * 96)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("=" * 96)
    print(f"\nfor scale, each of the 74 against the other 73: "
          f"min {inner.min():.4f}   median {np.median(inner):.4f}   max {inner.max():.4f}")
    print(f"the most decorrelated member the library already has sits at {inner.min():.4f} "
          f"(szymonkapiski's card names `lookup` at 0.9869)")
    below = df[df["max corr vs the 74"] < inner.min()]
    print(f"\nnew members MORE decorrelated than anything already in the library: {len(below)}")
    if len(below):
        print(below.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    else:
        print("  none. On this evidence the two libraries are inside the pack, and adding them")
        print("  should be worth about what rmlp_lat3 was worth, which is nothing.")
    json.dump({"members": rows, "base_max_corr": {"min": float(inner.min()),
               "median": float(np.median(inner)), "max": float(inner.max())}},
              open("ext_gate.json", "w"), indent=1)
    print("\nwrote ext_gate.json")


if __name__ == "__main__":
    main()
