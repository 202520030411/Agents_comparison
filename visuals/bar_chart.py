"""Visual aid 3: Per-seed bar chart — classical vs DL on seeds 0–9."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Data from docs/dl_journey.md
SEEDS = list(range(10))
CLASSICAL = [784.0, 874.5, 470.1, 892.6, 867.3, 741.9, 773.2, 802.8, 808.4, 847.4]
DL        = [930.8, 804.0, 893.0, 893.4, 871.7, 896.3, 928.1, 935.7, 830.5, 574.4]

CLS_MEAN = np.mean(CLASSICAL)  # 786.2
DL_MEAN  = np.mean(DL)         # 855.8


def main():
    x = np.arange(len(SEEDS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.5))

    bars_cls = ax.bar(x - width/2, CLASSICAL, width, label=f"Classical (mean {CLS_MEAN:.1f})",
                      color="#4A90D9", edgecolor="white", linewidth=0.5)
    bars_dl  = ax.bar(x + width/2, DL, width, label=f"DL / PPO (mean {DL_MEAN:.1f})",
                      color="#E85D75", edgecolor="white", linewidth=0.5)

    # Mean lines
    ax.axhline(CLS_MEAN, color="#4A90D9", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.axhline(DL_MEAN, color="#E85D75", linestyle="--", linewidth=1.2, alpha=0.6)

    # 900 line for "full lap" threshold
    ax.axhline(900, color="#2ECC71", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(9.6, 905, "full lap", fontsize=8, color="#2ECC71", va="bottom")

    # Highlight the big-delta seeds
    for i in range(len(SEEDS)):
        delta = DL[i] - CLASSICAL[i]
        if abs(delta) > 100:
            winner_color = "#E85D75" if delta > 0 else "#4A90D9"
            ax.annotate(f"{delta:+.0f}", xy=(i, max(CLASSICAL[i], DL[i]) + 15),
                        ha="center", fontsize=8, fontweight="bold", color=winner_color)

    ax.set_xlabel("Random Seed", fontsize=12)
    ax.set_ylabel("Episode Return", fontsize=12)
    ax.set_title("Head-to-Head: Classical vs DL Agent (10 random tracks)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SEEDS)
    ax.set_ylim(0, 1050)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("visuals/bar_chart.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("Saved visuals/bar_chart.png")


if __name__ == "__main__":
    main()
