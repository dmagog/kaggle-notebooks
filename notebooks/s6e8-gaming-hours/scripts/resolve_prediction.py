"""Score section 12b's bet the hour the private board lands, not the evening after.

12b, written before the split: "At most two of the public top ten will be in the private top
ten, or at least five. Three or four and I am wrong." This resolves that against the real
board, plus the three things the notebook will be asked about the moment results appear:
where the private top ten came from, whether the seven-board cushion base rate held here,
and how this episode's kept count sits against its seven predecessors.

Two modes, so the logic is exercised before it is needed:

  python3 resolve_prediction.py --test S6E7     replay a finished episode from the dataset
  python3 resolve_prediction.py --live pub.csv priv.csv     the real thing on 1 September

The test mode is the point: every number below is computed by the same function on a board
whose answer is already known and published in section 12.
"""
import argparse
import math
import sys

import pandas as pd

BOARDS = "../s6_boards/dataset/s6_leaderboards.csv"
KNOWN = {"S6E1": 8, "S6E2": 0, "S6E3": 6, "S6E4": 5, "S6E5": 8, "S6E6": 0, "S6E7": 0}
WATCH = ["Georgy Mamarin", "Chris Deotte", "Dariush Afshar"]


def resolve(df, label):
    """df: one row per team with public_rank, private_rank, teams implied by len(df)."""
    n = len(df)
    line = math.ceil(0.05 * n)
    top = df[df.public_rank <= 10]
    kept = int((top.private_rank <= 10).sum())
    came = df.public_rank[df.private_rank <= 10]

    print(f"\n=== {label}: {n:,} teams, top-5% line at rank {line} ===")
    print(f"kept of the public top ten: {kept}")
    verdict = "RIGHT" if kept <= 2 or kept >= 5 else "WRONG"
    print(f"12b bet (at most two, or at least five): {verdict}"
          + ("" if verdict == "RIGHT" else "   <- three or four, exactly the losing band"))
    print(f"the private top ten came from public {int(came.min())}-{int(came.max())}"
          f"   ({int((came > 100).sum())} of 10 from outside the public top hundred)")

    # the cushion base rate the comment thread quoted: 6-15 ranks inside the line
    band = df[(df.public_rank >= line - 15) & (df.public_rank <= line - 6)]
    stayed = int((band.private_rank <= line).sum())
    print(f"cushion 6-15 inside the line: {stayed} of {len(band)} stayed inside"
          f"   (seven-board base rate was 20 of 70, 29%)")

    for name in WATCH:
        row = df[df.team_name.astype(str).str.fullmatch(name, case=False, na=False)]
        if len(row):
            r = row.iloc[0]
            print(f"  {name}: public {int(r.public_rank)} -> private {int(r.private_rank)}"
                  f"   ({int(r.public_rank) - int(r.private_rank):+d})")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", metavar="EPISODE")
    ap.add_argument("--live", nargs=2, metavar=("PUBLIC_CSV", "PRIVATE_CSV"))
    a = ap.parse_args()

    if a.test:
        lb = pd.read_csv(BOARDS)
        lb = lb[(~lb.is_host_baseline) & (lb.episode == a.test)]
        if not len(lb):
            sys.exit(f"no rows for {a.test}")
        kept = resolve(lb, f"{a.test} (replay)")
        want = KNOWN.get(a.test)
        print(f"\ncheck against section 12's published count: {want} -> "
              f"{'MATCH' if kept == want else 'MISMATCH, the resolver is wrong'}")
        sys.exit(0 if kept == want else 1)

    if a.live:
        pub = pd.read_csv(a.live[0]); priv = pd.read_csv(a.live[1])
        pub.columns = [c.strip().lower() for c in pub.columns]
        priv.columns = [c.strip().lower() for c in priv.columns]
        key = "teamid" if "teamid" in pub.columns else "teamname"
        pub = pub.sort_values("score", ascending=False).reset_index(drop=True)
        priv = priv.sort_values("score", ascending=False).reset_index(drop=True)
        pub["public_rank"] = pub.index + 1
        priv["private_rank"] = priv.index + 1
        df = pub.merge(priv[[key, "private_rank"]], on=key, how="inner")
        df["team_name"] = df["teamname"]
        resolve(df, "S6E8 (live)")
        sys.exit(0)

    ap.print_help()


if __name__ == "__main__":
    main()
