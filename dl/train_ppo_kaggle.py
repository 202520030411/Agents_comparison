"""Self-contained Kaggle training script for PPO on CarRacing-v3.

Paste this into a Kaggle notebook cell (or save as .py and %run it).
No external files needed — it installs its own deps, trains, and writes
`/kaggle/working/ppo_carracing.zip` which you can download afterwards.

How to use on Kaggle:
    1. Create a new Kaggle notebook.
    2. Settings -> Accelerator -> GPU T4 x1 (or P100, whichever is free).
    3. Paste the whole contents of this file into a single cell.
    4. Run the cell.
    5. When it finishes, download `ppo_carracing.zip` from the right-side
       "Output" panel (path: /kaggle/working/ppo_carracing.zip).
    6. Place it locally at `dl/ppo_carracing.zip` in this repo.

Expected training time on Kaggle T4: ~8-15 min for 500k steps.
Expected final ep_rew_mean: ~500-800 (PPO hits plateau near 900 at ~1M steps).
"""

# ---- install deps (Kaggle base images usually have torch; these are the
#      rest). Comment these out if already installed in your Kaggle env.
import subprocess, sys

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip("gymnasium[box2d]==1.2.3", "stable-baselines3==2.8.0", "swig")

# ---- the actual training
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))

TOTAL_STEPS = 500_000   # raise to 1_000_000 if you want a stronger agent
N_ENVS = 8

vec_env = make_vec_env(
    "CarRacing-v3",
    n_envs=N_ENVS,
    env_kwargs={"continuous": True},
)
vec_env = VecTransposeImage(vec_env)

# Same hyperparams as dl/train_ppo.py in the local repo, so a model trained
# on Kaggle is a drop-in replacement for a locally trained one.
model = PPO(
    "CnnPolicy",
    vec_env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=512,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

print(f"training for {TOTAL_STEPS:,} steps on {N_ENVS} parallel envs")
model.learn(total_timesteps=TOTAL_STEPS, progress_bar=False)

SAVE_PATH = "/kaggle/working/ppo_carracing"
model.save(SAVE_PATH)
print(f"saved to {SAVE_PATH}.zip")

# Quick in-notebook sanity eval on seed 0.
import gymnasium as gym
env = gym.make("CarRacing-v3", continuous=True)
obs, _ = env.reset(seed=0)
total = 0.0
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, _ = env.step(action)
    total += reward
    done = term or trunc
env.close()
print(f"sanity eval seed=0 return={total:.1f}")
