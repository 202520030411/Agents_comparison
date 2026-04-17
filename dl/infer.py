"""Load a trained PPO checkpoint and run one episode on a given seed.

Same interface as classical.run.run: takes (seed, render) and returns
the total reward. This is what the eval harness calls.

The model was trained with frame stacking (4 frames concatenated along
the channel axis), so inference must use the same wrapping.
"""
import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack


CHECKPOINT = "dl/ppo_carracing"
N_STACK = 4


# Cache the loaded model so the eval sweep doesn't reload from disk per seed.
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = PPO.load(CHECKPOINT, device="auto")
    return _model


def _make_eval_env(seed: int):
    """Wrap a single CarRacing env with the same frame-stack + transpose
    that the model was trained with."""
    def _make():
        env = gym.make("CarRacing-v3", continuous=True)
        env.reset(seed=seed)
        return env
    vec = DummyVecEnv([_make])
    vec = VecTransposeImage(vec)
    vec = VecFrameStack(vec, n_stack=N_STACK)
    return vec


def run(seed: int, render: bool = False) -> float:
    model = _get_model()
    vec_env = _make_eval_env(seed)
    obs = vec_env.reset()
    total = 0.0
    done = False
    while not done:
        # deterministic=True -> take the mean action from the policy
        # distribution instead of sampling. Gives reproducible eval.
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        total += reward[0]
        done = dones[0]
    vec_env.close()
    return float(total)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    r = run(args.seed, args.render)
    print(f"seed={args.seed} return={r:.1f}")
