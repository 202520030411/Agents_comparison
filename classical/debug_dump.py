"""Dump frames + masks at regular intervals to inspect perception."""
import os
import cv2
import numpy as np
from env.car_racing_env import make_env, preprocess
from classical.perception import track_mask, centerline
from classical.planner import CAR_ROW, CAR_COL, LOOKAHEAD_ROWS, pursuit_target


OUT = "videos/debug"
os.makedirs(OUT, exist_ok=True)


def run(seed: int = 0, n_frames: int = 200, save_every: int = 20):
    env = make_env(seed=seed)
    obs, _ = env.reset(seed=seed)
    for t in range(n_frames):
        frame = preprocess(obs)
        mask = track_mask(frame)
        pts = centerline(mask)

        if t % save_every == 0:
            # overlay centerline + target on frame
            vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
            for r, c in pts:
                cv2.circle(vis, (c, r), 1, (0, 255, 255), -1)
            cv2.circle(vis, (CAR_COL, CAR_ROW), 2, (0, 0, 255), -1)
            target = pursuit_target(pts)
            if target is not None:
                # redraw the chosen target point
                target_row = CAR_ROW - LOOKAHEAD_ROWS
                pt = min(pts, key=lambda rc: abs(rc[0] - target_row))
                cv2.circle(vis, (pt[1], pt[0]), 3, (0, 255, 0), -1)
            cv2.imwrite(f"{OUT}/t{t:04d}_frame.png", vis)
            cv2.imwrite(f"{OUT}/t{t:04d}_mask.png", mask)
            print(f"t={t} n_centerline={len(pts)} target={target}")

        # drive straight-ish while we dump
        action = np.array([0.0, 0.3, 0.0], dtype=np.float32)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    env.close()


if __name__ == "__main__":
    run()
