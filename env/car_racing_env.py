"""Thin wrapper around gymnasium CarRacing so both agents share the same setup."""
import gymnasium as gym
import numpy as np


def make_env(seed: int = 0, render_mode: str | None = None):
    env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
    env.reset(seed=seed)
    return env


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Crop off the bottom status bar (last 12 rows) and return HxWx3 uint8."""
    return frame[:84, :, :]
