"""Visual aid 2: Frame stacking diagram.

Shows 4 consecutive frames side by side with an arrow pointing to
the stacked 12-channel tensor, illustrating how the CNN gets velocity
information from pixel differences between frames.
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")


def main():
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    obs, _ = env.reset(seed=3)

    # Drive forward a bit so the car is moving and frames differ
    for _ in range(65):
        obs, *_ = env.step(np.array([0.0, 0.3, 0.0]))

    # Capture 4 consecutive frames with gentle steering so they look different
    frames = []
    for i in range(4):
        obs, *_ = env.step(np.array([0.05, 0.3, 0.0]))
        frames.append(obs[:84].copy())  # crop status bar
    env.close()

    # --- Build the figure ---
    fig = plt.figure(figsize=(16, 5))

    # Layout: 4 frames on the left, arrow in the middle, stacked viz on the right
    # Use gridspec for precise control
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 0.4, 1.3],
                          wspace=0.08)

    # Show 4 individual frames
    labels = ["t−3", "t−2", "t−1", "t"]
    colors = ["#FF6B6B", "#FFA07A", "#87CEEB", "#4ECDC4"]
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frames[i])
        ax.set_title(f"Frame {labels[i]}", fontsize=12, fontweight="bold",
                     color=colors[i])
        ax.axis("off")
        # Add colored border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(colors[i])
            spine.set_linewidth(3)

    # Arrow in the middle
    ax_arrow = fig.add_subplot(gs[0, 4])
    ax_arrow.axis("off")
    ax_arrow.annotate(
        "", xy=(0.95, 0.5), xytext=(0.05, 0.5),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=3),
    )
    ax_arrow.text(0.5, 0.65, "stack", ha="center", va="center",
                  fontsize=11, fontweight="bold", transform=ax_arrow.transAxes)

    # Stacked visualization on the right
    ax_stack = fig.add_subplot(gs[0, 5])
    ax_stack.axis("off")

    # Show the most recent frame but annotate the channel count
    ax_stack.imshow(frames[-1])
    ax_stack.set_title("CNN Input\n12 channels (4×RGB)", fontsize=12,
                       fontweight="bold")

    # Draw 4 stacked rectangles offset to show depth
    for i, color in enumerate(reversed(colors)):
        rect = mpatches.FancyBboxPatch(
            (-8 + i * 3, -8 + i * 3), 96, 84,
            boxstyle="round,pad=1",
            linewidth=2, edgecolor=color, facecolor="none",
            transform=ax_stack.transData,
        )
        ax_stack.add_patch(rect)

    # Add channel breakdown text below
    channel_text = "R G B  R G B  R G B  R G B"
    ax_stack.text(48, 92, "3ch + 3ch + 3ch + 3ch = 12ch",
                  ha="center", va="top", fontsize=9, fontweight="bold",
                  color="#333333")

    fig.suptitle(
        "Frame Stacking: giving the CNN velocity information",
        fontsize=16, fontweight="bold", y=1.02
    )

    # Add explanation text at the bottom
    fig.text(0.5, -0.04,
             "A single frame is a photograph — it shows where the road is but not how fast the car moves.\n"
             "By stacking 4 frames, the CNN can see pixel differences between frames → it can infer speed and turning direction.",
             ha="center", va="top", fontsize=10, style="italic", color="#555555")

    plt.savefig("visuals/frame_stacking.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("Saved visuals/frame_stacking.png")


if __name__ == "__main__":
    main()
