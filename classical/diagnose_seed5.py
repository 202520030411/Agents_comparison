"""Run seed 5 under both classical configurations and log per-step info.

Config A = fixed lookahead 35, gain 2.2   (old, mean 502, seed 5 = 738)
Config B = adaptive lookahead 15-55, gain 3.4  (new, mean 602, seed 5 = 207)

For each config, write a CSV with one row per frame and dump a frame image
every 40 steps so we can see where the car actually is.
"""
import os
import csv
import cv2
import numpy as np

from env.car_racing_env import make_env, preprocess
from classical.perception import track_mask, centerline
import classical.planner as P
import classical.controller as C
from classical.controller import Controller

SEED = 5
DUMP_EVERY = 40
OUT_DIR = "videos/seed5_diag"
os.makedirs(OUT_DIR, exist_ok=True)


def run_fixed():
    """Config A: fixed lookahead 35, gain 2.2."""
    C.K_P_HEADING = 2.2
    env = make_env(seed=SEED)
    obs, _ = env.reset(seed=SEED)
    ctrl = Controller()
    rows = []
    total = 0.0
    for t in range(1100):
        frame = preprocess(obs)
        mask = track_mask(frame)
        pts = centerline(mask)

        if pts:
            # replicate old fixed-lookahead pursuit inline
            target_row = P.CAR_ROW - 35
            pt = min(pts, key=lambda rc: abs(rc[0] - target_row))
            r, c = pt
            dx = c - P.CAR_COL
            dy = P.CAR_ROW - r
            hdg = float(np.arctan2(dx, max(dy, 1)))
            lookahead = 35
            cols = np.fromiter((c for _, c in pts), dtype=np.float32)
            curvy = float(cols.std())
            action = ctrl.step(float(dx), hdg)
        else:
            dx = hdg = curvy = float("nan")
            lookahead = 35
            action = ctrl.fallback()

        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        rows.append({
            "t": t, "reward": reward, "total": total,
            "n_pts": len(pts), "curvy": curvy, "lookahead": lookahead,
            "dx": dx, "hdg": hdg,
            "steer": float(action[0]), "gas": float(action[1]), "brake": float(action[2]),
        })
        if t % DUMP_EVERY == 0:
            cv2.imwrite(f"{OUT_DIR}/A_t{t:04d}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if term or trunc:
            break
    env.close()
    return rows, total


def run_adaptive():
    """Config B: adaptive lookahead 15-55, gain 3.4."""
    C.K_P_HEADING = 3.4
    env = make_env(seed=SEED)
    obs, _ = env.reset(seed=SEED)
    planner = P.Planner()
    ctrl = Controller()
    rows = []
    total = 0.0
    for t in range(1100):
        frame = preprocess(obs)
        mask = track_mask(frame)
        pts = centerline(mask)

        tgt = planner.target(pts)
        if tgt is not None:
            dx, hdg, lookahead = tgt
            cols = np.fromiter((c for _, c in pts), dtype=np.float32)
            curvy = float(cols.std())
            action = ctrl.step(dx, hdg)
        else:
            dx = hdg = curvy = float("nan")
            lookahead = -1
            action = ctrl.fallback()

        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        rows.append({
            "t": t, "reward": reward, "total": total,
            "n_pts": len(pts), "curvy": curvy, "lookahead": lookahead,
            "dx": dx, "hdg": hdg,
            "steer": float(action[0]), "gas": float(action[1]), "brake": float(action[2]),
        })
        if t % DUMP_EVERY == 0:
            cv2.imwrite(f"{OUT_DIR}/B_t{t:04d}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if term or trunc:
            break
    env.close()
    return rows, total


def save(rows, name):
    path = f"{OUT_DIR}/{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


if __name__ == "__main__":
    print("running config A (fixed lookahead, gain 2.2)...")
    A, totA = run_fixed()
    save(A, "A_fixed")
    print(f"  total return = {totA:.1f}  frames = {len(A)}")

    print("running config B (adaptive lookahead, gain 3.4)...")
    B, totB = run_adaptive()
    save(B, "B_adaptive")
    print(f"  total return = {totB:.1f}  frames = {len(B)}")

    # Find the first frame where their cumulative reward diverges by > 50
    print("\ndivergence trace (|totalA - totalB| > 50):")
    for i in range(min(len(A), len(B))):
        if abs(A[i]["total"] - B[i]["total"]) > 50:
            print(f"  first diverge at t={i}  A.total={A[i]['total']:.1f}  B.total={B[i]['total']:.1f}")
            break

    print("\nconfig B rollout at 20-frame intervals (where things went wrong):")
    print(f"{'t':>4} {'reward':>7} {'total':>8} {'n_pts':>6} {'curvy':>6} "
          f"{'look':>5} {'dx':>7} {'hdg':>7} {'steer':>7} {'gas':>5} {'brake':>6}")
    for r in B[::20]:
        def f(k, w, p=1):
            v = r[k]
            if isinstance(v, float) and np.isnan(v):
                return f"{'nan':>{w}}"
            return f"{v:>{w}.{p}f}"
        print(f"{r['t']:>4} {f('reward',7,2)} {f('total',8)} {r['n_pts']:>6} "
              f"{f('curvy',6)} {r['lookahead']:>5} {f('dx',7)} {f('hdg',7,2)} "
              f"{f('steer',7,2)} {f('gas',5,2)} {f('brake',6,2)}")
