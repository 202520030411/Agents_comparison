"""Train a PPO agent on CarRacing-v3 using Stable-Baselines3.

The trained model takes the same 96x96 RGB frame the classical agent sees
and outputs a (steer, gas, brake) vector. Unlike the classical agent,
perception + planning + control are all one neural net, trained from
scratch by trial and error.

Usage:
    python -m dl.train_ppo                       # default 500k steps
    python -m dl.train_ppo --steps 1_000_000     # longer training
    python -m dl.train_ppo --steps 100_000       # quick sanity check

Outputs:
    dl/ppo_carracing.zip                         # SB3 model checkpoint
    dl/tb_logs/                                  # tensorboard logs
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage


SAVE_PATH = "dl/ppo_carracing"


def main(total_steps: int, n_envs: int):
    # 8 parallel envs so PPO collects 8 episodes' worth of experience per
    # rollout instead of 1. Huge wall-clock speedup.
    vec_env = make_vec_env(
        "CarRacing-v3",
        n_envs=n_envs,
        env_kwargs={"continuous": True},
    )
    # SB3's CnnPolicy expects channels-first images (C, H, W). CarRacing
    # returns (H, W, C), so we wrap in VecTransposeImage to swap axes.
    vec_env = VecTransposeImage(vec_env)

    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        # Hyperparams tuned for CarRacing in the SB3 zoo. These are the
        # values that get ~800-900 return at ~1M steps in the literature.
        learning_rate=1e-4,
        n_steps=512,         # steps per env per rollout (x8 envs = 4096)
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="auto",       # uses MPS on Apple Silicon, CUDA on Linux+GPU, else CPU
    )

    print(f"training for {total_steps:,} steps on {n_envs} parallel envs")
    model.learn(total_timesteps=total_steps, progress_bar=False)
    model.save(SAVE_PATH)
    print(f"saved to {SAVE_PATH}.zip")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=8)
    args = p.parse_args()
    main(args.steps, args.n_envs)
