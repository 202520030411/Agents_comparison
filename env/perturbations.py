"""Observation perturbations for the robustness sweep.

Wraps a CarRacing-v3 env so that every frame returned to the agent is
perturbed, while the underlying physics/rewards are unchanged. Both
agents run on the same perturbed frame so the comparison is fair.

Two perturbation types:

- Gaussian pixel noise: add N(0, sigma) noise to every pixel. Models
  camera sensor noise / low light / video compression artifacts.

- Hue jitter: rotate the hue channel of every frame by a fixed amount
  in HSV space. Models a recolored track (e.g. night / desert / snow).
  This is where we expect classical HSV thresholding to break hard,
  because our color ranges are hand-picked for the default track hue.
"""
import cv2
import gymnasium as gym
import numpy as np


class NoisyObsWrapper(gym.ObservationWrapper):
    """Add Gaussian pixel noise with stddev `sigma` (in 0-255 scale)."""

    def __init__(self, env, sigma: float):
        super().__init__(env)
        self.sigma = float(sigma)

    def observation(self, obs):
        if self.sigma <= 0:
            return obs
        noise = np.random.normal(0, self.sigma, obs.shape).astype(np.float32)
        noisy = obs.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


class HueShiftWrapper(gym.ObservationWrapper):
    """Rotate the hue channel of every frame by `shift` (0-179, OpenCV scale)."""

    def __init__(self, env, shift: int):
        super().__init__(env)
        self.shift = int(shift) % 180

    def observation(self, obs):
        if self.shift == 0:
            return obs
        hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int32) + self.shift) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def make_perturbed_env(seed: int, noise_sigma: float = 0.0, hue_shift: int = 0,
                       render_mode: str | None = None):
    env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
    if noise_sigma > 0:
        env = NoisyObsWrapper(env, noise_sigma)
    if hue_shift != 0:
        env = HueShiftWrapper(env, hue_shift)
    env.reset(seed=seed)
    return env
