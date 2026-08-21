"""The snapshot numbers section 12 of the S6E8 notebook types in, and where they come from.

Everything about the seven FINISHED boards is computed inside the notebook itself, from the
attached dataset georgymamarin/playground-series-s6-leaderboards. This script covers the part
that cannot be: readings off LIVE public boards. The notebook has no internet, a running board
is not attachable, and the dataset deliberately holds finished episodes only, so those few
numbers are typed into the notebook as dated snapshots. This is the arithmetic behind them.

Two rulers, both "how tightly is the very top packed against the field below it":
  top 1% / top 10%   bands as shares of the field, so the field-size control bites
  top 10 / top 300   bands as team counts, so the control cannot move it and depth is read instead
At 2,058 teams both separate the seven finished boards, and they disagree about the live one.

USE THE POST-CLOSE PUBLIC FILE for anything about a finished episode. An earlier version of
this script compared S6E7's 30 July snapshot, a day before the deadline, against the final
private board; zero-of-ten survives either way, but the private top ten came from public ranks
307-391 on the pre-close file and 63-450 on the final one.

Run:  python3 precedent_boards.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd

# Public leaderboard downloads, newest last. S6E8 is live, so its readings are the snapshots;
# S6E7 is the only episode with enough snapshots to show how a ruler drifts as a board fills.
SNAPSHOT_GLOBS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "playground-series-s6e8-publicleaderboard-*.csv"),
    # Later downloads land in _live/ rather than beside this script; without this glob the
    # script silently reproduces an older board than the notebook is quoting.
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "_live*", "playground-series-s6e8-publicleaderboard-*.csv"),
    "../s6_boards/S6E8_public.csv",
    "../s6e6_lb_work/playground-series-s6e7-publicleaderboard-*.csv",
]
DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
# One download carries its date in the dataset build rather than in its filename.
UNDATED = {"S6E8_public.csv": "2026-08-08"}


def load(path):
    """Rank 0 is a host baseline row, not a team; leaving it in shifts every rank by one."""
    df = pd.read_csv(path)
    return df[df["Rank"] > 0]


def packing(df, a, b, as_share):
    """Span of the top band over the span of the wider one. Higher means the leaders are
    spread out relative to the field; lower means they are packed into a hair's breadth."""
    n = len(df)
    ka, kb = (max(2, int(n * a)), int(n * b)) if as_share else (a, b)
    hi, lo = df[df.Rank <= ka].Score, df[df.Rank <= kb].Score
    return (hi.max() - hi.min()) / (lo.max() - lo.min())


rows = []
for pattern in SNAPSHOT_GLOBS:
    for path in sorted(glob.glob(pattern)):
        df = load(path)
        stamp = DATE.search(path)
        rows.append({
            "board": "S6E8" if "s6e8" in path.lower() or "S6E8" in path else "S6E7",
            "read on": stamp.group(1) if stamp else UNDATED[os.path.basename(path)],
            "teams": len(df),
            "leader": df.Score.max(),
            "top ten span": df[df.Rank <= 10].Score.max() - df[df.Rank <= 10].Score.min(),
            "top 1% / top 10%": packing(df, .01, .10, True),
            "top 10 / top 300": packing(df, 10, 300, False),
        })

out = pd.DataFrame(rows).sort_values(["board", "teams"])
pd.set_option("display.width", 200)
print(out.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
print()
print("What section 12 types in, from the newest S6E8 row above:")
live = out[out.board == "S6E8"].iloc[-1]
print(f'  N_LIVE = {live["teams"]}')
print(f'  LIVE = {{"top 1% / top 10%": {live["top 1% / top 10%"]:.3f}, '
      f'"top 10 / top 300": {live["top 10 / top 300"]:.3f}}}')
print("  MATURITY = " + ", ".join(f'"{r.board}, {r["read on"]}": ({r.teams}, '
                                  f'{r["top 10 / top 300"]:.3f})' for _, r in out.iterrows()))
print()
print("What section 11 types in, the nine adjacent gaps in the public top ten, from the same file:")
# Sort by the DATE in the filename, not by the path: _live/ sorts before the script's own
# directory, so a path sort silently picks an older download than the newest one on disk.
_s6e8 = [p for pat in SNAPSHOT_GLOBS for p in glob.glob(pat) if "s6e8" in p.lower()]
top = load(max(_s6e8, key=lambda p: (DATE.search(p).group(1) if DATE.search(p) else "")))
gaps = -np.diff(top[top.Rank <= 10].sort_values("Rank").Score.to_numpy())
# Ties between teams give a gap of exactly zero; float subtraction can sign it -0.0.
print("  TOP10_GAPS = np.array([" +
      ", ".join(f"{g if g > 0 else 0.0:.5f}" for g in gaps) + "])")
print()
# Section 12's prose types in four more things that no notebook cell prints: the share ruler's
# own reversal on this board, S6E2's slide under the cut, and the cut values where the seven
# separate. A number the reader cannot recompute is one they must take on trust, so emit them.
print()
print("What section 12's prose types in beyond the snapshots above:")

_share = []
for _f in sorted(_s6e8, key=lambda p: (DATE.search(p).group(1) if DATE.search(p) else "")):
    _m = DATE.search(_f)
    if _m is None:            # one download carries its date in the build, not the filename
        continue
    _b = load(_f)
    _share.append((_m.group(1), len(_b), packing(_b, .01, .10, True)))
print("  the share ruler on this board over time: " +
      ", ".join(f"{d} {n} teams {v:.3f}" for d, n, v in _share))
if len(_share) > 1:
    _vals = [v for *_, v in _share]
    print(f"  its widest swing: {max(_vals) - min(_vals):.3f}")

# The seven finished boards, from the attached dataset rather than from a live download.
_LB = "../s6_boards/dataset/s6_leaderboards.csv"
if os.path.exists(_LB):
    _lb = pd.read_csv(_LB)
    _lb = _lb[~_lb.is_host_baseline.astype(bool)]
    _eps = pd.read_csv("../s6_boards/dataset/s6_episodes.csv").set_index("episode")
    _fin, _kept = {}, {}
    for _e, _g in _lb.groupby("episode"):
        _sgn = 1 if bool(_eps.loc[_e, "higher_is_better"]) else -1
        _fin[_e] = pd.DataFrame({"Rank": _g.public_rank.to_numpy(),
                                 "Score": _sgn * _g.public_score.to_numpy()})
        _kept[_e] = int(((_g.public_rank <= 10) & (_g.private_rank <= 10)).sum())
    _wiped = [e for e in _fin if _kept[e] == 0]
    _held = [e for e in _fin if _kept[e] > 0]

    def _cut_read(ep, cut):
        g = _fin[ep]
        return packing(g[g.Rank <= cut].assign(Rank=lambda d: d.Rank), .01, .10, True)

    print(f"  S6E2 under the cut: {_cut_read('S6E2', 1161):.3f} at 1,161 teams, "
          f"{_cut_read('S6E2', 2058):.3f} at 2,058, {_cut_read('S6E2', 2455):.3f} at 2,455")
    _c2058 = {e: _cut_read(e, 2058) for e in _fin}
    _wmax = max((v, e) for e, v in _c2058.items() if e in _wiped)
    _hmin = min((v, e) for e, v in _c2058.items() if e in _held)
    print(f"  the 2,058 hairline pair: wiped max {_wmax[1]} {_wmax[0]:.3f} vs "
          f"held min {_hmin[1]} {_hmin[0]:.3f}")
    _runs = []
    for _c in range(1200, 3001, 25):
        _w = [_cut_read(e, _c) for e in _wiped]
        _h = [_cut_read(e, _c) for e in _held]
        _ok = max(_w) < min(_h) or max(_h) < min(_w)
        if _runs and _runs[-1][2] == _ok:
            _runs[-1][1] = _c
        else:
            _runs.append([_c, _c, _ok])
    print("  where the cut separates the seven: " +
          ", ".join(f"{a:,}-{b:,} {'separates' if ok else 'overlaps'}" for a, b, ok in _runs))
else:
    print(f"  ({_LB} not found, so the cut sweep did not run)")
print()
# Section 12 also types the live board's fork-cluster series and today's top-ten tier profile.
print()
print("What the thread-signals block types in:")
_snaps = sorted((p for p in _s6e8 if DATE.search(p)),
                key=lambda p: DATE.search(p).group(1))
_ser = []
for _f in _snaps:
    _b = load(_f)
    _t = _b[_b.Rank <= 100]
    _ser.append((DATE.search(_f).group(1), int(_t.Score.value_counts().max())))
print("  largest identical-score cluster in the public top 100, by snapshot: " +
      ", ".join(f"{d} -> {v}" for d, v in _ser))
_sp = []
for _f in _snaps:
    _b = load(_f)
    _h = _b[_b.Rank <= 300].Score
    _sp.append((DATE.search(_f).group(1), _h.max() - _h.min()))
print("  span of ranks 1-300, by snapshot: " +
      ", ".join(f"{d} -> {v:.5f}" for d, v in _sp))

_users = glob.glob("ext/mk/Users.csv") or glob.glob("../s6e8/ext/mk/Users.csv")
if _users:
    _b = load(_snaps[-1])
    _top = _b[_b.Rank <= 10]
    _names = set()
    for _s_ in _top.TeamMemberUserNames.astype(str):
        _names.update(x.strip() for x in re.split(r"[,\s]+", _s_) if x.strip())
    _u = pd.read_csv(_users[0], usecols=["UserName", "PerformanceTier"])
    _u = dict(zip(_u.UserName, _u.PerformanceTier))
    _best = []
    for _, _r in _top.iterrows():
        _mem = [x.strip() for x in re.split(r"[,\s]+", str(_r.TeamMemberUserNames)) if x.strip()]
        _best.append(max((_u.get(m, -1) for m in _mem), default=-1))
    _best = [b for b in _best if b >= 0]
    print(f"  today's top ten: master-or-above {sum(b >= 3 for b in _best)} of {len(_best)}, "
          f"mean best tier {sum(_best) / len(_best):.1f}")
else:
    print("  (Meta Kaggle Users.csv not found locally; the tier line did not run)")
print()
print("Both rulers separate the seven finished boards at this field size, and what they say about")
print("the live one keeps changing: at 2,058 teams (17 Aug) the share ruler put this board below")
print("all seven and the depth-read count ruler put it among the holds; at 2,455 (21 Aug) the share")
print("ruler puts it inside the wipeout range and the count ruler leaves it between the two groups.")
print("The separation is also not monotone in the cut: sweep it and the seven come apart between")
print("1,575 and 1,675, close again to 1,975, and reopen above 2,000. The count ruler is unchanged")
print("by the cut because trimming the tail leaves ranks 1-300 alone, which is why the notebook")
print("reads it at a depth instead.")
