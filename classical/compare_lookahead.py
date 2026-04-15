"""Compare fixed lookahead vs adaptive lookahead, both using the fixed controller.

- Fixed: lookahead = 35 always, gain 2.2 (the original tuned values).
- Adaptive: lookahead in [15, 55] based on centerline column spread, gain 3.4.

Both run the SAME controller (with the stall-trap fix).
"""
import numpy as np

from env.car_racing_env import make_env, preprocess
from classical.perception import track_mask, centerline
import classical.planner as P
import classical.controller as C
from classical.controller import Controller


def run_fixed(seed: int, lookahead: int = 35) -> float:
    C.K_P_HEADING = 2.2
    env = make_env(seed=seed)
    obs, _ = env.reset(seed=seed)
    ctrl = Controller()
    total = 0.0
    for _ in range(1100):
        frame = preprocess(obs)
        mask = track_mask(frame)
        pts = centerline(mask)
        if pts:
            target_row = P.CAR_ROW - lookahead
            pt = min(pts, key=lambda rc: abs(rc[0] - target_row))
            r, c = pt
            dx = float(c - P.CAR_COL)
            dy = P.CAR_ROW - r
            hdg = float(np.arctan2(dx, max(dy, 1)))
            action = ctrl.step(dx, hdg)
        else:
            action = ctrl.fallback()
        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        if term or trunc:
            break
    env.close()
    return total


def run_adaptive(seed: int) -> float:
    C.K_P_HEADING = 3.4
    env = make_env(seed=seed)
    obs, _ = env.reset(seed=seed)
    planner = P.Planner()
    ctrl = Controller()
    total = 0.0
    for _ in range(1100):
        frame = preprocess(obs)
        mask = track_mask(frame)
        pts = centerline(mask)
        tgt = planner.target(pts)
        if tgt is not None:
            dx, hdg, _ = tgt
            action = ctrl.step(dx, hdg)
        else:
            action = ctrl.fallback()
        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        if term or trunc:
            break
    env.close()
    return total


if __name__ == "__main__":
    seeds = list(range(10))
    fixed = [run_fixed(s) for s in seeds]
    adapt = [run_adaptive(s) for s in seeds]

    print(f"{'seed':>4} {'fixed':>10} {'adaptive':>10} {'delta':>10}")
    for s, f, a in zip(seeds, fixed, adapt):
        print(f"{s:>4} {f:>10.1f} {a:>10.1f} {a-f:>+10.1f}")
    print()
    print(f"{'mean':>4} {np.mean(fixed):>10.1f} {np.mean(adapt):>10.1f} "
          f"{np.mean(adapt)-np.mean(fixed):>+10.1f}")
    print(f"{'good(>700)':>10}   fixed={sum(r>700 for r in fixed)}  "
          f"adaptive={sum(r>700 for r in adapt)}")
    print(f"{'laps(>900)':>10}   fixed={sum(r>900 for r in fixed)}  "
          f"adaptive={sum(r>900 for r in adapt)}")
    print(f"{'std':>4} {np.std(fixed):>10.1f} {np.std(adapt):>10.1f}")
