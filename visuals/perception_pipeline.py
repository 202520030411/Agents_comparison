"""Visual aid 1: Perception pipeline — raw frame → HSV mask → centerline overlay.

Saves a 3-panel figure showing each stage of the classical agent's perception.
"""
import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from env.car_racing_env import preprocess
from classical.perception import track_mask, centerline


def main():
    # Get a frame with visible road (skip the initial zoom-in animation)
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    obs, _ = env.reset(seed=3)
    for _ in range(65):
        obs, *_ = env.step(np.array([0.0, 0.1, 0.0]))
    env.close()

    frame = preprocess(obs)
    mask = track_mask(frame)
    pts = centerline(mask)

    # --- build the 3-panel figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: raw frame
    axes[0].imshow(frame)
    axes[0].set_title("1. Raw Frame (96×84 RGB)", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: binary mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("2. Track Mask (HSV threshold)", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: centerline overlaid on frame
    overlay = frame.copy()
    for r, c in pts:
        cv2.circle(overlay, (c, r), 1, (255, 0, 0), -1)

    # Draw the target point the planner would pick
    from classical.planner import Planner
    planner = Planner()
    tgt = planner.target(pts)
    if tgt is not None:
        dx, hdg, la = tgt
        target_col = 48 + int(dx)
        target_row = 70 - la
        cv2.circle(overlay, (target_col, target_row), 4, (0, 255, 0), -1)
        # Draw line from car to target
        cv2.line(overlay, (48, 70), (target_col, target_row), (0, 255, 0), 1)

    axes[2].imshow(overlay)
    axes[2].set_title("3. Centerline + Target Point", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    # Add legend-like annotations
    axes[2].plot([], [], "ro", markersize=4, label="Centerline points")
    axes[2].plot([], [], "go", markersize=8, label="Planner target")
    axes[2].legend(loc="lower right", fontsize=9, framealpha=0.8)

    fig.suptitle("Classical Agent Perception Pipeline", fontsize=18,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("visuals/perception_pipeline.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("Saved visuals/perception_pipeline.png")


if __name__ == "__main__":
    main()
