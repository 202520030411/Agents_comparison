"""Load a trained PPO checkpoint and run one episode on a given seed.

Same interface as classical.run.run: takes (seed, render) and returns
the total reward. This is what the eval harness calls.
"""
import argparse

from stable_baselines3 import PPO

from env.car_racing_env import make_env


CHECKPOINT = "dl/ppo_carracing"


# Cache the loaded model so the eval sweep doesn't reload from disk per seed.
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = PPO.load(CHECKPOINT, device="auto")
    return _model


def run(seed: int, render: bool = False) -> float:
    model = _get_model()
    env = make_env(seed=seed, render_mode="human" if render else None)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    done = False
    while not done:
        # deterministic=True -> take the mean action from the policy
        # distribution instead of sampling. Gives reproducible eval.
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        done = term or trunc
    env.close()
    return float(total)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    r = run(args.seed, args.render)
    print(f"seed={args.seed} return={r:.1f}")
