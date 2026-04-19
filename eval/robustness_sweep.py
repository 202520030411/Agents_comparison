"""Robustness sweep: run both agents under pixel noise and hue shift.

Tests how each agent degrades when the input image is perturbed.
Classical's HSV thresholds should break under hue shift (the road
changes color and the mask goes empty). DL's CNN may or may not
generalize — that's what we're measuring.

Usage:
    python -m eval.robustness_sweep
"""
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

from env.car_racing_env import preprocess
from env.perturbations import NoisyObsWrapper, HueShiftWrapper
from classical.perception import track_mask, centerline
from classical.planner import Planner
import classical.controller as C
from classical.controller import Controller


# ---- seeds to evaluate (subset for speed) ----
SEEDS = [0, 3, 5, 7, 9]

# ---- perturbation levels ----
NOISE_SIGMAS = [0, 10, 25, 50]          # stddev of Gaussian pixel noise
HUE_SHIFTS = [0, 15, 30, 60]            # degrees of hue rotation (OpenCV 0-179 scale)


# ---- classical agent runner ----
def run_classical(seed, noise_sigma=0, hue_shift=0):
    C.K_P_HEADING = 3.4
    env = gym.make("CarRacing-v3", continuous=True)
    if noise_sigma > 0:
        env = NoisyObsWrapper(env, noise_sigma)
    if hue_shift != 0:
        env = HueShiftWrapper(env, hue_shift)
    obs, _ = env.reset(seed=seed)
    planner = Planner()
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
    return float(total)


# ---- DL agent runner ----
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = PPO.load("dl/ppo_carracing", device="auto")
    return _model


def run_dl(seed, noise_sigma=0, hue_shift=0):
    model = _get_model()
    def _make():
        env = gym.make("CarRacing-v3", continuous=True)
        if noise_sigma > 0:
            env = NoisyObsWrapper(env, noise_sigma)
        if hue_shift != 0:
            env = HueShiftWrapper(env, hue_shift)
        env.reset(seed=seed)
        return env
    vec = DummyVecEnv([_make])
    vec = VecTransposeImage(vec)
    vec = VecFrameStack(vec, n_stack=4)
    obs = vec.reset()
    total = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec.step(action)
        total += reward[0]
        done = dones[0]
    vec.close()
    return float(total)


# ---- sweep ----
def sweep(param_name, values, make_kwargs):
    """Run both agents across all seeds for each perturbation level."""
    print(f"\n{'='*60}")
    print(f"  SWEEP: {param_name}")
    print(f"{'='*60}")

    all_results = []
    for val in values:
        kwargs = make_kwargs(val)
        cls_rets, dl_rets = [], []
        for s in SEEDS:
            c = run_classical(s, **kwargs)
            d = run_dl(s, **kwargs)
            cls_rets.append(c)
            dl_rets.append(d)
        cls_mean = np.mean(cls_rets)
        dl_mean = np.mean(dl_rets)
        all_results.append((val, cls_mean, dl_mean, cls_rets, dl_rets))
        print(f"  {param_name}={val:>3}  classical={cls_mean:7.1f}  dl={dl_mean:7.1f}")

    return all_results


def main():
    # Noise sweep
    noise_results = sweep(
        "noise_sigma",
        NOISE_SIGMAS,
        lambda sigma: {"noise_sigma": sigma, "hue_shift": 0},
    )

    # Hue sweep
    hue_results = sweep(
        "hue_shift",
        HUE_SHIFTS,
        lambda shift: {"noise_sigma": 0, "hue_shift": shift},
    )

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    print("\nGaussian noise sweep:")
    print(f"  {'sigma':>8}  {'classical':>12}  {'dl':>12}  {'winner':>10}")
    for val, cm, dm, _, _ in noise_results:
        winner = "classical" if cm > dm else "dl"
        print(f"  {val:>8}  {cm:>12.1f}  {dm:>12.1f}  {winner:>10}")

    print("\nHue shift sweep:")
    print(f"  {'shift':>8}  {'classical':>12}  {'dl':>12}  {'winner':>10}")
    for val, cm, dm, _, _ in hue_results:
        winner = "classical" if cm > dm else "dl"
        print(f"  {val:>8}  {cm:>12.1f}  {dm:>12.1f}  {winner:>10}")


if __name__ == "__main__":
    main()
