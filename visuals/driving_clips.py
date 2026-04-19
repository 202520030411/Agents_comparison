"""Visual aid 6: Side-by-side driving clips — classical vs DL on the same seed.

Records both agents on seed 3 (a track where both score well), then
stitches the frames side by side into a GIF and an AVI.

Outputs:
  visuals/driving_side_by_side.gif   — for slides / embedding
  visuals/driving_side_by_side.avi   — higher quality for video editing
"""
import cv2
import numpy as np
import gymnasium as gym
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

from env.car_racing_env import preprocess
from classical.perception import track_mask, centerline
from classical.planner import Planner
import classical.controller as C
from classical.controller import Controller

SEED = 3
MAX_STEPS = 500    # ~17 seconds at 30fps
FPS = 30
FRAME_H, FRAME_W = 96, 96
SCALE = 3          # upscale for visibility (96 → 288 px)


def add_label(frame, text, position="top"):
    """Put a text label on the frame (works on scaled-up frames)."""
    labeled = frame.copy()
    font_scale = 0.5 * SCALE / 2
    thickness = max(1, SCALE // 2)
    y = 18 * (SCALE // 2) if position == "top" else frame.shape[0] - 8 * (SCALE // 2)
    cv2.putText(labeled, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    # Dark shadow for readability
    cv2.putText(labeled, text, (5, y - 1), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(labeled, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return labeled


def record_classical(seed):
    """Run classical agent, return list of RGB frames (96×96)."""
    C.K_P_HEADING = 3.4
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    planner = Planner()
    ctrl = Controller()

    frames = []
    total = 0.0
    for step in range(MAX_STEPS):
        frames.append(obs[:FRAME_H].copy())
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
    print(f"  Classical: {len(frames)} frames, return={total:.1f}")
    return frames, total


def record_dl(seed):
    """Run DL agent, return list of RGB frames (96×96)."""
    model = PPO.load("dl/ppo_carracing", device="auto")

    # Use render_mode="rgb_array" on the inner env so we can grab frames
    def _make():
        env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
        env.reset(seed=seed)
        return env
    vec = DummyVecEnv([_make])
    vec = VecTransposeImage(vec)
    vec = VecFrameStack(vec, n_stack=4)
    obs = vec.reset()

    frames = []
    total = 0.0
    for step in range(MAX_STEPS):
        # obs shape (1, 12, 96, 96) — last 3 channels = newest RGB frame in CHW.
        # Transpose to HWC so it matches the classical 96x96 frames.
        latest = np.transpose(obs[0, -3:], (1, 2, 0))
        frames.append(latest[:FRAME_H].copy())

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec.step(action)
        total += reward[0]
        if dones[0]:
            break

    vec.close()
    print(f"  DL:        {len(frames)} frames, return={total:.1f}")
    return frames, total


def stitch(cls_frames, dl_frames, cls_ret, dl_ret):
    """Combine frame lists side by side, return list of RGB arrays."""
    n = min(len(cls_frames), len(dl_frames))
    gap = 4 * SCALE
    combined_frames = []

    for i in range(n):
        # Scale up for visibility
        left = cv2.resize(cls_frames[i], (FRAME_W * SCALE, FRAME_H * SCALE),
                          interpolation=cv2.INTER_NEAREST)
        right = cv2.resize(dl_frames[i], (FRAME_W * SCALE, FRAME_H * SCALE),
                           interpolation=cv2.INTER_NEAREST)

        left = add_label(left, f"Classical ({cls_ret:.0f})")
        right = add_label(right, f"DL / PPO ({dl_ret:.0f})")

        divider = np.full((FRAME_H * SCALE, gap, 3), 30, dtype=np.uint8)
        combined = np.concatenate([left, divider, right], axis=1)
        combined_frames.append(combined)

    return combined_frames


def save_gif(frames, path, fps=15):
    """Save frames as an animated GIF (subsample for smaller file)."""
    # Take every other frame for reasonable GIF size
    step = max(1, FPS // fps)
    pil_frames = [Image.fromarray(f) for f in frames[::step]]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000 // fps,
        loop=0,
    )
    print(f"  Saved GIF: {path} ({len(pil_frames)} frames)")


def save_avi(frames, path, fps=30):
    """Save frames as AVI with MJPG codec."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"  Saved AVI: {path} ({len(frames)} frames)")


def main():
    print(f"Recording seed {SEED} ({MAX_STEPS} steps max)...")
    cls_frames, cls_ret = record_classical(SEED)
    dl_frames, dl_ret = record_dl(SEED)

    print("Stitching...")
    combined = stitch(cls_frames, dl_frames, cls_ret, dl_ret)

    save_gif(combined, "visuals/driving_side_by_side.gif", fps=15)
    save_avi(combined, "visuals/driving_side_by_side.avi", fps=30)
    print("Done!")


if __name__ == "__main__":
    main()
