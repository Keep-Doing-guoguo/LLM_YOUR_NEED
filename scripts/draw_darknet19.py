from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


LAYERS = [
    ("Input", "224x224x3", "input"),
    ("Conv", "3x3, 32\n224x224", "conv"),
    ("MaxPool", "2x2 /2\n112x112", "pool"),
    ("Conv", "3x3, 64\n112x112", "conv"),
    ("MaxPool", "2x2 /2\n56x56", "pool"),
    ("Conv", "3x3, 128\n56x56", "conv"),
    ("Conv", "1x1, 64\n56x56", "conv1"),
    ("Conv", "3x3, 128\n56x56", "conv"),
    ("MaxPool", "2x2 /2\n28x28", "pool"),
    ("Conv", "3x3, 256\n28x28", "conv"),
    ("Conv", "1x1, 128\n28x28", "conv1"),
    ("Conv", "3x3, 256\n28x28", "conv"),
    ("MaxPool", "2x2 /2\n14x14", "pool"),
    ("Conv", "3x3, 512\n14x14", "conv"),
    ("Conv", "1x1, 256\n14x14", "conv1"),
    ("Conv", "3x3, 512\n14x14", "conv"),
    ("Conv", "1x1, 256\n14x14", "conv1"),
    ("Conv", "3x3, 512\n14x14", "conv"),
    ("MaxPool", "2x2 /2\n7x7", "pool"),
    ("Conv", "3x3, 1024\n7x7", "conv"),
    ("Conv", "1x1, 512\n7x7", "conv1"),
    ("Conv", "3x3, 1024\n7x7", "conv"),
    ("Conv", "1x1, 512\n7x7", "conv1"),
    ("Conv", "3x3, 1024\n7x7", "conv"),
    ("Conv", "1x1, 1000\n7x7", "head"),
    ("Global AvgPool", "7x7 -> 1x1", "gap"),
    ("Softmax", "1000 classes", "output"),
]

COLORS = {
    "input": "#E8F1F2",
    "conv": "#2F80ED",
    "conv1": "#56CCF2",
    "pool": "#F2994A",
    "head": "#9B51E0",
    "gap": "#27AE60",
    "output": "#EB5757",
}


def add_box(ax, x, y, w, h, title, subtitle, kind):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5,
        edgecolor="#1F2933",
        facecolor=COLORS[kind],
    )
    ax.add_patch(patch)
    text_color = "#FFFFFF" if kind not in {"input", "conv1"} else "#111827"
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=text_color)
    ax.text(x + w / 2, y + h * 0.34, subtitle, ha="center", va="center",
            fontsize=8.5, color=text_color, linespacing=1.1)


def draw():
    cols = 7
    w, h = 1.75, 0.9
    x_gap, y_gap = 0.45, 0.85
    fig_w, fig_h = 17, 8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    fig.patch.set_facecolor("#FAFBFC")
    ax.set_facecolor("#FAFBFC")

    positions = []
    for i, layer in enumerate(LAYERS):
        row = i // cols
        col = i % cols
        if row % 2 == 0:
            x = col * (w + x_gap)
        else:
            x = (cols - 1 - col) * (w + x_gap)
        y = -(row * (h + y_gap))
        positions.append((x, y))
        add_box(ax, x, y, w, h, *layer)

    for i in range(len(positions) - 1):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        if abs(y1 - y2) < 0.1:
            start = (x1 + w, y1 + h / 2) if x2 > x1 else (x1, y1 + h / 2)
            end = (x2, y2 + h / 2) if x2 > x1 else (x2 + w, y2 + h / 2)
        else:
            start = (x1 + w / 2, y1)
            end = (x2 + w / 2, y2 + h)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", color="#364152", lw=1.4,
                            shrinkA=4, shrinkB=4),
        )

    ax.text(0, 1.45, "DarkNet-19 Network Architecture", fontsize=22,
            fontweight="bold", color="#111827", ha="left")
    ax.text(
        0,
        1.08,
        "YOLOv2 backbone: 19 convolutional layers + 5 max-pooling layers; each Conv uses BatchNorm and Leaky ReLU except the classification softmax.",
        fontsize=10.5,
        color="#4B5563",
        ha="left",
    )

    legend_items = [
        ("3x3 Conv", "conv"),
        ("1x1 Conv", "conv1"),
        ("MaxPool", "pool"),
        ("Classifier Head", "head"),
        ("Global AvgPool", "gap"),
    ]
    legend_y = -4.95
    for j, (label, kind) in enumerate(legend_items):
        lx = j * 2.45
        ax.add_patch(FancyBboxPatch((lx, legend_y), 0.32, 0.22,
                                    boxstyle="round,pad=0.02,rounding_size=0.03",
                                    linewidth=0, facecolor=COLORS[kind]))
        ax.text(lx + 0.42, legend_y + 0.11, label, va="center", fontsize=9.5,
                color="#374151")

    ax.set_xlim(-0.15, cols * (w + x_gap) - x_gap + 0.15)
    ax.set_ylim(-5.45, 1.75)
    ax.axis("off")

    out = Path("assets/darknet19_architecture.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(out.resolve())


if __name__ == "__main__":
    draw()
