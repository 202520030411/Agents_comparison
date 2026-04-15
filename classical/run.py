"""Run the classical agent on one seed. Usage: python -m classical.run --seed 0 --render"""
import argparse
from env.car_racing_env import make_env, preprocess
from classical.perception import track_mask, centerline
from classical.planner import Planner
from classical.controller import Controller


def run(seed: int, render: bool) -> float:
    env = make_env(seed=seed, render_mode="human" if render else None)
    obs, _ = env.reset(seed=seed)
    planner = Planner()
    ctrl = Controller()
    total_reward = 0.0
    done = False
    while not done:
        frame = preprocess(obs)
        mask = track_mask(frame)
        pts = centerline(mask)
        target = planner.target(pts)
        if target is not None:
            lat, hdg, _look = target
            action = ctrl.step(lat, hdg)
        else:
            action = ctrl.fallback()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    env.close()
    return total_reward


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    r = run(args.seed, args.render)
    print(f"seed={args.seed} return={r:.1f}")
