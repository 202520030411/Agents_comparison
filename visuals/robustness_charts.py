"""Visual aid 4: Robustness sweep — noise and hue shift line charts.

Two side-by-side plots showing how each agent degrades under perturbation.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Data from docs/robustness_results.md
NOISE_SIGMAS = [0, 10, 25, 50]
NOISE_CLS    = [813.8, 774.4, 826.5, -44.5]
NOISE_DL     = [846.1, 814.0, 878.7, 867.2]

HUE_SHIFTS   = [0, 15, 30, 60]
HUE_CLS      = [813.8, 775.8, 743.3, 817.7]
HUE_DL       = [846.1, -82.8, -93.1, -93.1]


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: Gaussian noise ---
    ax1.plot(NOISE_SIGMAS, NOISE_CLS, "o-", color="#4A90D9", linewidth=2.5,
             markersize=8, label="Classical", zorder=3)
    ax1.plot(NOISE_SIGMAS, NOISE_DL, "s-", color="#E85D75", linewidth=2.5,
             markersize=8, label="DL / PPO", zorder=3)

    ax1.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.fill_between(NOISE_SIGMAS, NOISE_CLS, alpha=0.1, color="#4A90D9")
    ax1.fill_between(NOISE_SIGMAS, NOISE_DL, alpha=0.1, color="#E85D75")

    # Annotate the collapse
    ax1.annotate("Classical\ncollapses\n(−44.5)", xy=(50, -44.5),
                 xytext=(38, -180), fontsize=9, fontweight="bold",
                 color="#4A90D9",
                 arrowprops=dict(arrowstyle="->", color="#4A90D9"))

    ax1.set_xlabel("Noise σ (standard deviation of pixel noise)", fontsize=11)
    ax1.set_ylabel("Mean Episode Return", fontsize=11)
    ax1.set_title("Gaussian Pixel Noise", fontsize=14, fontweight="bold")
    ax1.set_xticks(NOISE_SIGMAS)
    ax1.set_xticklabels(["0\n(clean)", "10\n(mild)", "25\n(moderate)", "50\n(heavy)"])
    ax1.set_ylim(-250, 1000)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Winner banner
    ax1.text(25, 950, "DL wins all levels", ha="center", fontsize=10,
             fontweight="bold", color="#E85D75",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#E85D75",
                       alpha=0.1, edgecolor="#E85D75"))

    # --- Right: Hue shift ---
    ax2.plot(HUE_SHIFTS, HUE_CLS, "o-", color="#4A90D9", linewidth=2.5,
             markersize=8, label="Classical", zorder=3)
    ax2.plot(HUE_SHIFTS, HUE_DL, "s-", color="#E85D75", linewidth=2.5,
             markersize=8, label="DL / PPO", zorder=3)

    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.fill_between(HUE_SHIFTS, HUE_CLS, alpha=0.1, color="#4A90D9")
    ax2.fill_between(HUE_SHIFTS, HUE_DL, alpha=0.1, color="#E85D75")

    # Annotate the collapse
    ax2.annotate("DL collapses\n(−93.1)", xy=(30, -93.1),
                 xytext=(42, -180), fontsize=9, fontweight="bold",
                 color="#E85D75",
                 arrowprops=dict(arrowstyle="->", color="#E85D75"))

    ax2.set_xlabel("Hue Shift (degrees of color rotation)", fontsize=11)
    ax2.set_ylabel("Mean Episode Return", fontsize=11)
    ax2.set_title("Hue Shift (Recolored Track)", fontsize=14, fontweight="bold")
    ax2.set_xticks(HUE_SHIFTS)
    ax2.set_xticklabels(["0\n(default)", "15\n(subtle)", "30\n(strong)", "60\n(extreme)"])
    ax2.set_ylim(-250, 1000)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    # Winner banner
    ax2.text(30, 950, "Classical wins all nonzero", ha="center", fontsize=10,
             fontweight="bold", color="#4A90D9",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#4A90D9",
                       alpha=0.1, edgecolor="#4A90D9"))

    fig.suptitle("Robustness Evaluation: How Each Agent Breaks",
                 fontsize=16, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig("visuals/robustness_charts.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("Saved visuals/robustness_charts.png")

    # --- Individual charts for slides ---
    save_noise_chart()
    save_hue_chart()


def save_noise_chart():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(NOISE_SIGMAS, NOISE_CLS, "o-", color="#4A90D9", linewidth=2.5,
            markersize=8, label="Classical", zorder=3)
    ax.plot(NOISE_SIGMAS, NOISE_DL, "s-", color="#E85D75", linewidth=2.5,
            markersize=8, label="DL / PPO", zorder=3)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.fill_between(NOISE_SIGMAS, NOISE_CLS, alpha=0.1, color="#4A90D9")
    ax.fill_between(NOISE_SIGMAS, NOISE_DL, alpha=0.1, color="#E85D75")
    ax.annotate("Classical\ncollapses\n(−44.5)", xy=(50, -44.5),
                xytext=(38, -180), fontsize=10, fontweight="bold",
                color="#4A90D9",
                arrowprops=dict(arrowstyle="->", color="#4A90D9"))
    ax.set_xlabel("Noise σ (standard deviation of pixel noise)", fontsize=12)
    ax.set_ylabel("Mean Episode Return", fontsize=12)
    ax.set_title("Gaussian Pixel Noise", fontsize=15, fontweight="bold")
    ax.set_xticks(NOISE_SIGMAS)
    ax.set_xticklabels(["0\n(clean)", "10\n(mild)", "25\n(moderate)", "50\n(heavy)"])
    ax.set_ylim(-250, 1000)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.text(25, 950, "DL wins all levels", ha="center", fontsize=11,
            fontweight="bold", color="#E85D75",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E85D75",
                      alpha=0.1, edgecolor="#E85D75"))
    plt.tight_layout()
    plt.savefig("visuals/robustness_noise.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("Saved visuals/robustness_noise.png")


def save_hue_chart():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(HUE_SHIFTS, HUE_CLS, "o-", color="#4A90D9", linewidth=2.5,
            markersize=8, label="Classical", zorder=3)
    ax.plot(HUE_SHIFTS, HUE_DL, "s-", color="#E85D75", linewidth=2.5,
            markersize=8, label="DL / PPO", zorder=3)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.fill_between(HUE_SHIFTS, HUE_CLS, alpha=0.1, color="#4A90D9")
    ax.fill_between(HUE_SHIFTS, HUE_DL, alpha=0.1, color="#E85D75")
    ax.annotate("DL collapses\n(−93.1)", xy=(30, -93.1),
                xytext=(42, -180), fontsize=10, fontweight="bold",
                color="#E85D75",
                arrowprops=dict(arrowstyle="->", color="#E85D75"))
    ax.set_xlabel("Hue Shift (degrees of color rotation)", fontsize=12)
    ax.set_ylabel("Mean Episode Return", fontsize=12)
    ax.set_title("Hue Shift (Recolored Track)", fontsize=15, fontweight="bold")
    ax.set_xticks(HUE_SHIFTS)
    ax.set_xticklabels(["0\n(default)", "15\n(subtle)", "30\n(strong)", "60\n(extreme)"])
    ax.set_ylim(-250, 1000)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.text(30, 950, "Classical wins all nonzero", ha="center", fontsize=11,
            fontweight="bold", color="#4A90D9",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#4A90D9",
                      alpha=0.1, edgecolor="#4A90D9"))
    plt.tight_layout()
    plt.savefig("visuals/robustness_hue.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("Saved visuals/robustness_hue.png")


if __name__ == "__main__":
    main()
