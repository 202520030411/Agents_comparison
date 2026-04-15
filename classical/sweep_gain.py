"""Sweep K_P_HEADING values with adaptive lookahead on, seeds 0-9."""
import numpy as np
import classical.controller as C
from classical.run import run


GAINS = [2.2, 2.8, 3.4, 4.0]
SEEDS = list(range(10))


def main():
    table = {}
    for g in GAINS:
        C.K_P_HEADING = g
        rets = []
        for s in SEEDS:
            r = run(s, render=False)
            rets.append(r)
            print(f"  gain={g}  seed={s}  return={r:.1f}")
        table[g] = rets
        print(f"gain={g}  mean={np.mean(rets):.1f}")
    print()
    print("summary:")
    for g, rets in table.items():
        print(f"  gain={g}  mean={np.mean(rets):7.1f}  "
              f"success(>900)={sum(r > 900 for r in rets)}  "
              f"good(>700)={sum(r > 700 for r in rets)}")


if __name__ == "__main__":
    main()
