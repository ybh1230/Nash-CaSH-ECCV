from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "ink": "#1f2937",
    "muted": "#64748b",
    "line": "#94a3b8",
    "panel": "#f8fafc",
    "context": "#dbeafe",
    "context_edge": "#2563eb",
    "native": "#dcfce7",
    "native_edge": "#16a34a",
    "equilibrium": "#fff7ed",
    "equilibrium_edge": "#ea580c",
    "violet": "#f3e8ff",
    "violet_edge": "#7c3aed",
    "output": "#e0f2fe",
    "output_edge": "#0284c7",
}


def add_box(ax, xy, wh, title, lines=None, fc=None, ec=None, lw=1.6,
            title_size=11.5, line_size=8.5, radius=0.025):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec or COLORS["line"],
        facecolor=fc or COLORS["panel"],
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.04 * w, y + h - 0.22 * h, title,
        fontsize=title_size, fontweight="bold",
        color=COLORS["ink"], va="top", ha="left", zorder=3,
    )
    if lines:
        ax.text(
            x + 0.04 * w, y + h - 0.47 * h, "\n".join(lines),
            fontsize=line_size, color=COLORS["muted"],
            va="top", ha="left", linespacing=1.28, zorder=3,
        )
    return box


def add_arrow(ax, start, end, color=None, lw=1.6, rad=0.0, z=1):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color or COLORS["line"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
        zorder=z,
    )
    ax.add_patch(arrow)
    return arrow


def add_mini_video(ax, x, y, w, h, edge, fill):
    offsets = [(0.020, 0.020), (0.010, 0.010), (0.000, 0.000)]
    for dx, dy in offsets:
        rect = Rectangle(
            (x + dx, y + dy), w, h,
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.1,
            zorder=3,
        )
        ax.add_patch(rect)
    for k in range(3):
        ax.plot(
            [x + 0.035 + k * 0.035, x + 0.035 + k * 0.035],
            [y + 0.02, y + h - 0.02],
            color=edge,
            linewidth=0.7,
            alpha=0.5,
            zorder=4,
        )


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(14.8, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.025, 0.955,
        "Equilibrium-Guided Semantic Coordination",
        fontsize=14.0, fontweight="bold", color=COLORS["ink"],
        ha="left", va="top",
    )
    ax.text(
        0.025, 0.905,
        "Training-free 1080P refinement with adaptive semantic authority",
        fontsize=9.8, color=COLORS["muted"], ha="left", va="top",
    )

    add_box(
        ax, (0.035, 0.36), (0.145, 0.31),
        "Input",
        ["prompt + base latent", "target 1080P grid"],
        fc="#f8fafc", ec="#cbd5e1",
    )
    add_mini_video(ax, 0.105, 0.392, 0.050, 0.066, "#64748b", "#e2e8f0")

    add_box(
        ax, (0.230, 0.33), (0.165, 0.39),
        "Cross-attention\nlayer",
        ["visual queries Q", "text keys / values", "high-resolution tokens"],
        fc="#f8fafc", ec="#94a3b8",
    )

    add_box(
        ax, (0.445, 0.61), (0.195, 0.235),
        "Contextual Evidence",
        ["scene layout", "identity and count", "cache refresh every P steps"],
        fc=COLORS["context"], ec=COLORS["context_edge"],
    )
    add_box(
        ax, (0.445, 0.245), (0.195, 0.235),
        "Native-grid Evidence",
        ["local detail", "boundary and texture", "current native-scale response"],
        fc=COLORS["native"], ec=COLORS["native_edge"],
    )

    add_box(
        ax, (0.685, 0.535), (0.195, 0.260),
        "Reliability Observation",
        [
            r"$A_x=\max(0,\cos(O_x,Q_x))$",
            r"$C_x=E_x/(E_c+E_n)$",
            r"$r_x=\pi_x(A_x+C_x+\epsilon)$",
        ],
        fc=COLORS["violet"], ec=COLORS["violet_edge"],
        line_size=9.2,
    )

    add_box(
        ax, (0.685, 0.145), (0.195, 0.285),
        "Nash Coordination",
        [
            r"$\max\;\sum_x r_x\log(a_x-d_x)$",
            r"$a_x^*=d_x+S\frac{r_x}{\sum_j r_j}$",
        ],
        fc=COLORS["equilibrium"], ec=COLORS["equilibrium_edge"],
        line_size=8.2,
    )

    add_box(
        ax, (0.912, 0.355), (0.082, 0.305),
        "Output",
        ["refined", "1080P video"],
        fc=COLORS["output"], ec=COLORS["output_edge"],
        title_size=10.2,
        line_size=7.8,
    )
    add_mini_video(ax, 0.952, 0.393, 0.022, 0.048, "#0284c7", "#bae6fd")

    add_arrow(ax, (0.180, 0.515), (0.230, 0.515))
    add_arrow(ax, (0.395, 0.555), (0.445, 0.725), color=COLORS["context_edge"], rad=0.08)
    add_arrow(ax, (0.395, 0.485), (0.445, 0.360), color=COLORS["native_edge"], rad=-0.08)
    add_arrow(ax, (0.640, 0.725), (0.685, 0.665), color=COLORS["context_edge"], rad=-0.05)
    add_arrow(ax, (0.640, 0.360), (0.685, 0.620), color=COLORS["native_edge"], rad=0.12)
    add_arrow(ax, (0.782, 0.535), (0.782, 0.420), color=COLORS["violet_edge"], rad=0.0)
    add_arrow(ax, (0.880, 0.292), (0.912, 0.505), color=COLORS["equilibrium_edge"], rad=0.0)

    ax.text(
        0.542, 0.545,
        r"$O_c$",
        fontsize=12.0, color=COLORS["context_edge"], fontweight="bold",
        ha="center", va="center",
    )
    ax.text(
        0.542, 0.185,
        r"$O_n$",
        fontsize=12.0, color=COLORS["native_edge"], fontweight="bold",
        ha="center", va="center",
    )
    ax.text(
        0.902, 0.300,
        r"$O=aO_c+(1-a)O_n$",
        fontsize=9.6, color=COLORS["ink"], ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cbd5e1", linewidth=0.9),
    )

    badge_specs = [
        (0.685, 0.080, "training-free"),
        (0.795, 0.080, "weight-preserving"),
        (0.915, 0.080, "cache-compatible"),
    ]
    for x, y, text in badge_specs:
        ax.text(
            x, y, text,
            fontsize=8.6, color=COLORS["muted"], ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.26",
                facecolor="#ffffff",
                edgecolor="#cbd5e1",
                linewidth=0.9,
            ),
        )

    fig.savefig(OUT_DIR / "sem_eq_method.png", dpi=350, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_DIR / "sem_eq_method.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
