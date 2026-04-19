"""Visual aid 5: 2×2 robustness punchline grid.

A simple, memorable summary: classical vs DL × noise vs color.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")


def main():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    # Table data
    cell_text = [
        ["Survives\n(867 at σ=50)", "Collapses\n(−93 at shift 30)"],
        ["Collapses\n(−44 at σ=50)", "Survives\n(743 at shift 30)"],
    ]
    row_labels = ["DL / PPO", "Classical"]
    col_labels = ["Pixel Noise", "Color Shift"]

    # Colors: green for survives, red for collapses
    cell_colors = [
        ["#C8E6C9", "#FFCDD2"],  # DL: green, red
        ["#FFCDD2", "#C8E6C9"],  # Classical: red, green
    ]

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        rowColours=["#E8EAF6", "#E8EAF6"],
        colColours=["#E8EAF6", "#E8EAF6"],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.5, 2.5)

    # Bold the header cells
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(fontweight="bold", fontsize=14)
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(1.5)

    ax.set_title(
        "Neither agent is strictly better —\nthey have complementary failure modes",
        fontsize=15, fontweight="bold", pad=20,
    )

    # Bottom annotation
    fig.text(0.5, 0.02,
             "Green = robust (score holds)    Red = fragile (score collapses to negative)",
             ha="center", fontsize=10, color="#666666")

    plt.savefig("visuals/punchline_grid.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("Saved visuals/punchline_grid.png")


if __name__ == "__main__":
    main()
