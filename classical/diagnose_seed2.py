"""Per-frame log of seed 2 with the current best classical config (adaptive + gain 3.4).

Dump a frame every 20 steps so we can see exactly where the car goes and when.
"""
import os
import cv2
import numpy as np

from env.car_racing_env import make_env, preprocess
from classical.perception import track_mask, centerline
import classical.planner as P
import classical.controller as C
from classical.controller import Controller

SEED = 2
DUMP_EVERY = 20
OUT_DIR = "videos/seed2_diag"
os.makedirs(OUT_DIR, exist_ok=True)


def run():
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
            dx, hdg, look = tgt
            action = ctrl.step(dx, hdg)
            lat = dx
        else:
            lat = hdg = float("nan")
            look = -1
            action = ctrl.fallback()
        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        rows.append({
            "t": t, "r": reward, "total": total,
            "n_pts": len(pts), "look": look, "dx": lat, "hdg": hdg,
            "steer": float(action[0]), "gas": float(action[1]), "brake": float(action[2]),
        })
        if t % DUMP_EVERY == 0:
            cv2.imwrite(f"{OUT_DIR}/t{t:04d}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{OUT_DIR}/t{t:04d}_mask.png", mask)
        if term or trunc:
            break
    env.close()

    print(f"seed {SEED} total={total:.1f} frames={len(rows)}")
    print(f"{'t':>4} {'r':>6} {'total':>7} {'pts':>4} {'look':>4} {'dx':>6} "
          f"{'hdg':>6} {'steer':>6} {'gas':>5} {'brake':>6}")
    for r in rows[::20]:
        def f(k, w, p=1):
            v = r[k]
            if isinstance(v, float) and np.isnan(v):
                return f"{'nan':>{w}}"
            return f"{v:>{w}.{p}f}"
        print(f"{r['t']:>4} {f('r',6,2)} {f('total',7)} {r['n_pts']:>4} "
              f"{r['look']:>4} {f('dx',6)} {f('hdg',6,2)} "
              f"{f('steer',6,2)} {f('gas',5,2)} {f('brake',6,2)}")

    # Also find no-track stretches (perception lost the road)
    lost = [r for r in rows if r["n_pts"] == 0]
    if lost:
        print(f"\nno-track frames: {len(lost)}  first={lost[0]['t']}  last={lost[-1]['t']}")


if __name__ == "__main__":
    run()
