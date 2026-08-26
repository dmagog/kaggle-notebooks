"""Does a cushion inside the public top-5% line survive the private split? Seven boards answer.

Written for a comment thread under the S6E8 notebook: a competitor at public rank 132 of 2,911
(13 ranks inside the top-5% line) priced his survival with a Poisson-binomial over all opponents
at his blend family's bootstrap sd and got P(stay) = 0.9976. This prices the same question on the
seven finished Season 6 boards instead, where both halves are known.

Convention: the line is ceil(0.05 * n) after dropping the host baseline; a team "stays" when its
private rank is at or inside the same line. The dataset's known caveat applies: the two halves
score different submissions for 29-55% of every top 300, so the churn here bundles split noise
with submission selection. The single-submission and three-or-fewer cuts below are the control
for that: those teams had little or no selection surface, and they fall at the same rate.

Data: s6_leaderboards.csv from kaggle.com/datasets/georgymamarin/playground-series-s6-leaderboards

Run:  python3 cushion_survival.py
"""
import math
import statistics

import pandas as pd

lb = pd.read_csv("s6_leaderboards.csv")
lb = lb[~lb.is_host_baseline]

BANDS = (("1-5", 1, 5), ("6-15", 6, 15), ("16-40", 16, 40))

pooled, narrow, falls = {}, [0, 0], []
print(f"{'ep':<6}{'teams':>6}{'line':>6}   stayed, by cushion (ranks inside the line)")
for e, g in lb.groupby("episode"):
    n = len(g)
    line = math.ceil(0.05 * n)
    parts = []
    for name, lo, hi in BANDS:
        b = g[(g.public_rank >= line - hi) & (g.public_rank <= line - lo)]
        stay = int((b.private_rank <= line).sum())
        a, c = pooled.get(name, (0, 0))
        pooled[name] = (a + stay, c + len(b))
        parts.append(f"{name}: {stay}/{len(b)}")
    # the commenter's own band, 10-16 ranks inside
    b = g[(g.public_rank >= line - 16) & (g.public_rank <= line - 10)]
    narrow[0] += int((b.private_rank <= line).sum()); narrow[1] += len(b)
    # how far past the line the 6-15 band's non-survivors land
    b = g[(g.public_rank >= line - 15) & (g.public_rank <= line - 6)]
    falls += list((b[b.private_rank > line].private_rank - line).values)
    print(f"{e:<6}{n:>6}{line:>6}   " + "   ".join(parts))

print("\npooled across the seven boards:")
for name, (s, m) in pooled.items():
    print(f"  cushion {name:>5}: stayed {s}/{m} = {s/m:.0%}")
print(f"  cushion 10-16: stayed {narrow[0]}/{narrow[1]} = {narrow[0]/narrow[1]:.0%}")
print(f"\nnon-survivors of the 6-15 band: median lands {statistics.median(falls):.0f} ranks "
      f"past the line (max {max(falls)})")

sel = [0, 0, 0, 0]
for e, g in lb.groupby("episode"):
    n = len(g); line = math.ceil(0.05 * n)
    b = g[(g.public_rank >= line - 15) & (g.public_rank <= line - 6)]
    one, few = b[b.submissions == 1], b[b.submissions <= 3]
    sel[0] += int((one.private_rank <= line).sum()); sel[1] += len(one)
    sel[2] += int((few.private_rank <= line).sum()); sel[3] += len(few)
print(f"\nselection control, same 6-15 band: submissions == 1 stayed {sel[0]}/{sel[1]}, "
      f"submissions <= 3 stayed {sel[2]}/{sel[3]}")
