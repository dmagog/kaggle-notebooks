"""The resolution formula, checked against seven boards that have already finished.

Two results are circulating in this competition and both are correct arithmetic. szymonkapiski
published the resolution floor: single-score sd 0.000272 at test size, paired difference 0.000072,
multiplier sqrt(2(1-rho)), so "a change smaller than roughly 0.00014 cannot be resolved."
broccoli beef published the transfer probability, P(z_priv > 0 | z_pub) = Phi(z_pub / S) with
S^2 = tau^2 (1/n_pub + 1/n_priv).

Both carry the same stated exclusion, in szymonkapiski's own words: the bootstrap "measures
sampling noise only, not the public/private split." Tilii asked the next question in thread 734005
-- is the public board indicative of private placement -- and nobody has answered it with data.
Every number so far is derived from one live board where the answer is not yet observable.

Seven Season 6 boards have finished and carry both halves. This checks the model against them.

WHAT THE MODEL PREDICTS, AND HOW IT CAN FAIL. Write public = skill + e_pub and
private = skill + e_priv, with the two errors independent given skill. That is exactly the model
both derivations assume. It has a consequence that survives without knowing tau, without knowing
rho, and without the metric being AUC:

    inside ANY set selected on public score, the correlation between public and private
    cannot be negative.

Selection on public score attenuates the correlation toward zero -- pick a narrow enough band and
public ordering becomes pure noise -- but the covariance keeps the sign of Cov(public, skill),
which truncation from below leaves positive. Sampling noise scrambles an ordering. It cannot
invert one. So a negative correlation inside a top band is not a large noise level; it is a
different mechanism, and it rejects the model rather than stretching it.

The simulation below makes the same point with numbers instead of algebra: it takes each finished
board's private scores AS the skill, which is generous to the null since it removes private noise
entirely, adds independent public noise wide enough to reproduce the churn actually observed, and
reports what correlation the null can reach in the same top band.

TIES. Balanced accuracy produces genuine ties on the private side, 6 to 30 per cent of pairs here.
A tie is not a surviving ordering and not a broken one, so it is scored as half and the tie rate
is printed rather than hidden.

Run:  python3 formula_backtest.py
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA = "../s6_boards/dataset"
BANDS = [30, 100, 300, 1000]
MAXPAIRS = 40000
NSIM = 200
RNG = np.random.default_rng(20260817)


def sampled_pairs(n):
    idx = list(itertools.combinations(range(n), 2))
    if len(idx) > MAXPAIRS:
        idx = idx[:: len(idx) // MAXPAIRS + 1]
    return (np.fromiter((a for a, _ in idx), int),
            np.fromiter((b for _, b in idx), int))


def survival(pub, pri, i, j):
    """Share of public orderings that survive to private, ties counted as half."""
    gp, gr = pub[i] - pub[j], pri[i] - pri[j]
    m = gp != 0
    gp, gr = gp[m], gr[m]
    return float(np.mean(np.where(gr == 0, 0.5, (np.sign(gp) == np.sign(gr)).astype(float)))), \
        float(np.mean(gr == 0))


def null_band_correlation(skill, band, churn):
    """Highest band correlation the noise-only model reaches, given the churn it must reproduce.

    `skill` is the private board, used as truth: the null is handed a private side with no noise
    of its own, which can only help it. Public noise sd is set so the simulated board reproduces
    the observed full-board Spearman, i.e. the null is calibrated to the real amount of churn.
    """
    lo, hi = 0.0, float(np.std(skill)) * 4 + 1e-12
    for _ in range(40):                      # bisect the noise sd onto the observed full-board rho
        sd = (lo + hi) / 2
        r = np.mean([spearmanr(skill + RNG.normal(0, sd, len(skill)), skill).statistic
                     for _ in range(12)])
        lo, hi = (sd, hi) if r > churn else (lo, sd)
    sd = (lo + hi) / 2
    out = []
    for _ in range(NSIM):
        pub = skill + RNG.normal(0, sd, len(skill))
        top = np.argsort(-pub)[:band]
        out.append(spearmanr(pub[top], skill[top]).statistic)
    return float(np.mean(out)), float(np.quantile(out, 0.01)), float(np.std(out)), sd


def main() -> None:
    lb = pd.read_csv(os.path.join(DATA, "s6_leaderboards.csv"))
    eps = pd.read_csv(os.path.join(DATA, "s6_episodes.csv")).set_index("episode")
    lb = lb[~lb.is_host_baseline.astype(bool)]

    rows, sims = [], []
    for e in sorted(lb.episode.unique()):
        g = lb[lb.episode == e].dropna(subset=["private_score"]).copy()
        if not len(g):
            continue
        sgn = 1 if bool(eps.loc[e, "higher_is_better"]) else -1
        g["pub"], g["pri"] = g.public_score * sgn, g.private_score * sgn
        g = g.sort_values("public_rank")

        row = {"episode": e, "metric": eps.loc[e, "metric"].replace(" Score", ""),
               "teams": len(g),
               "kept of top ten": int(((g.public_rank <= 10) & (g.private_rank <= 10)).sum())}
        for b in BANDS:
            sub = g[g.public_rank <= b]
            row[f"rho top {b}"] = (spearmanr(sub.pub, sub.pri).statistic
                                   if len(sub) > 10 else np.nan)
        row["rho whole board"] = spearmanr(g.pub, g.pri).statistic
        i, j = sampled_pairs(len(g[g.public_rank <= 300]))
        s300 = g[g.public_rank <= 300]
        row["survived top 300"], row["private ties"] = survival(
            s300.pub.to_numpy(), s300.pri.to_numpy(), i, j)
        rows.append(row)

        mean, p01, spread, sd = null_band_correlation(g.pri.to_numpy(), 300,
                                                      row["rho whole board"])
        sims.append({"episode": e, "observed": row["rho top 300"], "null mean": mean,
                     "null 1st pct": p01,
                     "sd from null mean": (row["rho top 300"] - mean) / spread,
                     "rejects null": bool(row["rho top 300"] < p01),
                     "noise sd fitted": sd})

    out, sim = pd.DataFrame(rows), pd.DataFrame(sims)
    pd.set_option("display.width", 240)
    f = lambda v: f"{v:.3f}"
    print("Public-to-private rank correlation, by how deep into the public board you look\n")
    print(out.to_string(index=False, float_format=f))

    # The same question in the form a competitor actually asks it: standing HERE publicly,
    # what were my odds of finishing in the private top ten per cent?
    print("\nShare of a public band that finished in the private top ten per cent\n")
    eps_list = [r["episode"] for r in rows]
    print(f"{'public band':<16}" + "".join(f"{e:>8}" for e in eps_list))
    fwd = {}
    for lo, hi in [(1, 10), (11, 50), (51, 100), (101, 300), (301, 1000)]:
        line, cells = f"{f'{lo}-{hi}':<16}", {}
        for e in eps_list:
            g = lb[lb.episode == e].dropna(subset=["private_score"])
            b = g[(g.public_rank >= lo) & (g.public_rank <= hi)]
            cells[e] = float(b.private_rank.le(len(g) * 0.10).mean())
            line += f"{cells[e] * 100:7.0f}%"
        fwd[f"{lo}-{hi}"] = cells
        print(line)
    print("a team picked at random would be there 10 per cent of the time")

    print("\nWhere the private top ten stood on the public board\n")
    origins = {}
    for e in eps_list:
        g = lb[lb.episode == e].dropna(subset=["private_score"])
        r = g[g.private_rank <= 10].sort_values("private_rank").public_rank.astype(int).tolist()
        origins[e] = r
        print(f"  {e}  {str(r):<48} median {int(np.median(r)):>4}")
    allr = [x for r in origins.values() for x in r]
    print(f"\n  of these {len(allr)}, {sum(x <= 10 for x in allr)} stood in the public top ten, "
          f"{sum(x <= 100 for x in allr)} in the top hundred")
    print("\nThe noise-only model, calibrated on each board's own churn, simulated 200 times.")
    print("It is handed the private board as truth, so it never pays for private-side noise.\n")
    print(sim.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    n = int(sim["rejects null"].sum())
    print(f"\n{n} of {len(sim)} boards fall below the null's 1st percentile.")
    json.dump({"bands": out.to_dict("records"), "null": sim.to_dict("records"),
               "forward": fwd, "origins": origins},
              open("formula_backtest.json", "w"), indent=1)
    print("wrote formula_backtest.json")


if __name__ == "__main__":
    main()
