"""
Graphical abstract for the GMCC robustness paper (Computational Biology and Chemistry).

Reads the same per-seed Kendall tau_b file as make_figures.py
(data/processed/raw_kendall_tau_per_seed.npz, keys "dataset|method|rate"),
so the curves in the middle panel are the Figure 1 means. The findings panel
summarises the main text, including the detection-dependence analyses
(Sections 3.3-3.6).

Output (figures/): graphical_abstract.{pdf,tiff,png}
Canvas: 13.28 x 5.31 in (1328 x 531 px at 100 dpi, the journal minimum,
w x h); raster files are saved at 300 dpi (3984 x 1593 px), keeping 2.5:1.

Usage:
    python src/figures/make_graphical_abstract.py [path/to/raw_kendall_tau_per_seed.npz]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.path.join(REPO, "data", "processed")
DEFAULT_TAU_NPZ = os.path.join(DATA_DIR, "raw_kendall_tau_per_seed.npz")
FIG_DIR = os.path.join(REPO, "figures")

MASKING_LEVELS = [0.10, 0.30, 0.50, 0.70]
MASKING_PCT = np.array([int(r * 100) for r in MASKING_LEVELS])
DATASETS = ["PBMC3K", "Paul15"]
DATASET_LABEL = {"PBMC3K": "PBMC 3K", "Paul15": "Paul15"}
ORDER = ["GMCC", "Cosine", "Spearman", "Pearson"]

BLUE = "#1F4E9C"      # GMCC and main elements
FILL = "#8FA8CF"      # detected (non-zero) entries
MASK = "#D9822B"      # entries masked to zero; detection threshold
ZERO = "#F3D9BF"      # shared zeros (findings panel)
INK = "#222222"
NOTE = "0.35"
DARK, MID, LIGHT = "0.35", "0.55", "0.72"   # Cosine, Spearman, Pearson
STYLE = {  # comparators keep the Figure 1 markers and line styles
    "GMCC":     dict(marker="o", linestyle="-",  color=BLUE,  lw=2.6, ms=6.0, zorder=5),
    "Cosine":   dict(marker="s", linestyle="--", color=DARK,  lw=1.6, ms=4.5, zorder=4),
    "Spearman": dict(marker="v", linestyle=":",  color=MID,   lw=1.6, ms=4.5, zorder=3),
    "Pearson":  dict(marker="D", linestyle=(0, (3, 1, 1, 1)), color=LIGHT,
                     lw=1.6, ms=4.0, zorder=3),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.linewidth": 0.8,
})

# Panel boundaries in figure coordinates
X_LEFT, X_MID, X_RIGHT = 0.0, 0.285, 0.600


def load_means(npz_path):
    if not os.path.exists(npz_path):
        sys.exit(f"ERROR: {npz_path} not found. Run paired_difference_ci.py first "
                 "or pass the path as the first argument.")
    data = np.load(npz_path)
    vals = {}
    for key in data.files:
        ds, method, rate = key.split("|")
        vals[(ds, method, float(rate))] = float(np.mean(data[key]))
    missing = [(ds, m, r) for ds in DATASETS for m in ORDER for r in MASKING_LEVELS
               if (ds, m, r) not in vals]
    if missing:
        sys.exit(f"ERROR: conditions missing from {npz_path}: {missing[:4]} ...")
    return vals


def panel_title(fig, x, text):
    fig.text(x, 0.965, text, fontsize=14.5, fontweight="bold", color=INK,
             ha="left", va="top")


def grid(ax, x0, y0, cell, filled, masked=None, ec="0.80"):
    nr, nc = filled.shape
    for i in range(nr):
        for j in range(nc):
            x, y = x0 + j * cell, y0 + (nr - 1 - i) * cell
            if masked is not None and masked[i, j]:
                ax.add_patch(Rectangle((x, y), cell, cell, fc="white", ec=MASK,
                                       lw=1.5, zorder=3))
            else:
                ax.add_patch(Rectangle((x, y), cell, cell,
                                       fc=FILL if filled[i, j] else "white",
                                       ec=ec, lw=0.6, zorder=2))


def arrow(ax, p0, p1, lw=1.6, color=INK, ms=14):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color))


def text_run(ax, x, y, parts, fontsize):
    """Draw consecutive text pieces [(text, kwargs), ...] on one line."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    for txt, kw in parts:
        t = ax.text(x, y, txt, fontsize=fontsize, va="center", ha="left", **kw)
        bb = t.get_window_extent(renderer=renderer)
        x = ax.transData.inverted().transform((bb.x1, bb.y0))[0]


# ── Panel 1: controlled masking ──────────────────────────────────────
def panel_design(fig):
    panel_title(fig, X_LEFT + 0.012, "Controlled masking")
    ax = fig.add_axes([X_LEFT, 0.0, X_MID - X_LEFT, 0.90])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    ax.text(0.4, 9.1, "PBMC 3K (human)  +  Paul15 (mouse)", fontsize=11.5,
            fontweight="bold", va="center")

    rng = np.random.default_rng(7)
    filled = rng.random((6, 7)) < 0.42
    det = np.argwhere(filled)
    masked = np.zeros_like(filled)
    sel = det[rng.choice(len(det), size=int(round(0.5 * len(det))), replace=False)]
    masked[sel[:, 0], sel[:, 1]] = True

    cell, y0 = 0.46, 4.85
    grid(ax, 0.4, y0, cell, filled)
    grid(ax, 5.25, y0, cell, filled, masked)
    ax.text(0.4 + 3.5 * cell, y0 + 6 * cell + 0.35, "Observed matrix",
            ha="center", fontsize=10.5)
    ax.text(5.25 + 3.5 * cell, y0 + 6 * cell + 0.35, "Masked matrix",
            ha="center", fontsize=10.5)
    arrow(ax, (3.85, y0 + 3 * cell), (5.05, y0 + 3 * cell))
    ax.text(0.4, y0 - 0.35, "genes × cells", fontsize=9.5, color=NOTE, va="top")

    ax.add_patch(Rectangle((0.4, 3.35), 0.36, 0.36, fc="white", ec=MASK, lw=1.5))
    ax.text(0.98, 3.53, "Detected entries randomly\nmasked to zero (10–70%)",
            fontsize=10.5, va="center", linespacing=1.2)
    ax.text(0.98, 2.35, "20 independent realisations\nper condition", fontsize=9.5,
            color=NOTE, va="center", linespacing=1.2)

    arrow(ax, (3.6, 1.75), (3.6, 1.15), lw=1.2, ms=11)
    text_run(ax, 0.4, 0.65, [("GMCC", dict(color=BLUE, fontweight="bold")),
                             ("  |  Pearson  |  Cosine  |  Spearman", dict(color=INK))],
             fontsize=10.5)


# ── Panel 2: rank robustness ─────────────────────────────────────────
def panel_robustness(fig, tau):
    panel_title(fig, X_MID + 0.012, "Gene-pair rank robustness")
    fig.text(X_MID + 0.012, 0.875, "Higher is more robust", fontsize=10,
             color=NOTE, style="italic", va="top")

    w, gap, left, bottom, height = 0.118, 0.016, X_MID + 0.045, 0.345, 0.43
    for k, ds in enumerate(DATASETS):
        ax = fig.add_axes([left + k * (w + gap), bottom, w, height])
        for m in ORDER:
            st = STYLE[m]
            ax.plot(MASKING_PCT, [tau[(ds, m, r)] for r in MASKING_LEVELS],
                    marker=st["marker"], linestyle=st["linestyle"], color=st["color"],
                    lw=st["lw"], ms=st["ms"], zorder=st["zorder"])
        ax.set_title(DATASET_LABEL[ds], fontsize=11)
        ax.set_xticks(MASKING_PCT)
        ax.set_xlim(2, 78); ax.set_ylim(0.15, 1.02)
        ax.set_yticks([0.2, 0.6, 1.0])
        ax.tick_params(labelsize=9.5, length=3)
        ax.grid(True, lw=0.4, alpha=0.4)
        if k == 0:
            ax.set_ylabel("Kendall's $\\tau_b$", fontsize=11)
        else:
            ax.set_yticklabels([])
    xc = left + w + gap / 2
    fig.text(xc, 0.255, "Masked detected entries (%)", ha="center", va="center",
             fontsize=10.5)

    handles = [plt.Line2D([], [], marker=STYLE[m]["marker"],
                          linestyle=STYLE[m]["linestyle"], color=STYLE[m]["color"],
                          lw=STYLE[m]["lw"], ms=STYLE[m]["ms"]) for m in ORDER]
    fig.legend(handles, ORDER, loc="center", ncol=4, frameon=False,
               bbox_to_anchor=(xc, 0.175), fontsize=9.5, handlelength=2.0,
               columnspacing=0.9, handletextpad=0.4)

    fig.text(xc, 0.075, "GMCC also showed low magnitude change\nat moderate–severe masking",
             ha="center", va="center", fontsize=10, color=INK, linespacing=1.2,
             bbox=dict(boxstyle="round,pad=0.35", fc="#EEF2F9", ec="none"))


# ── Panel 3: main findings ───────────────────────────────────────────
def icon_curves(ax, x, y, w, h):
    xs = np.linspace(x, x + w, 4)
    ax.plot(xs, y + h * np.array([0.92, 0.86, 0.80, 0.72]), color=BLUE, lw=2.2)
    for c, d in [(DARK, 0.10), (MID, 0.38), (LIGHT, 0.46)]:
        ax.plot(xs, y + h * (0.80 - np.linspace(0, 1, 4) * (0.22 + d)),
                color=c, lw=1.4)


def icon_shared_zero(ax, x, y, s):
    """Two rarely detected genes: each detected in a different cell, so almost
    all of their entries are shared zeros (shaded)."""
    for row, on in enumerate([0, 4]):
        yy = y + (1 - row) * (s + 0.22)
        for j in range(6):
            ax.add_patch(Rectangle((x + j * s, yy), s, s,
                                   fc=BLUE if j == on else ZERO, ec="0.75", lw=0.6))


def icon_filter(ax, x, y, w, h):
    """Detection-rate bars with the 10% threshold; bars below it are dropped."""
    rates = [0.12, 0.85, 0.38, 0.08, 0.62, 0.26, 0.06]
    bw = w / len(rates)
    thr = 0.18
    for j, r in enumerate(rates):
        keep = r >= thr
        ax.add_patch(Rectangle((x + j * bw + 0.03, y), bw - 0.06, h * r,
                               fc=BLUE if keep else "0.85",
                               ec=BLUE if keep else "0.70", lw=0.8))
    ax.plot([x - 0.05, x + w + 0.05], [y + h * thr] * 2, color=MASK, lw=1.4,
            linestyle=(0, (3, 2)))


def icon_detection(ax, x, y, s):
    for row, (n_on, col) in enumerate([(1, BLUE), (5, DARK)]):
        yy = y + (1 - row) * (s + 0.22)
        for j in range(6):
            ax.add_patch(Rectangle((x + j * s, yy), s, s,
                                   fc=col if j < n_on else "white", ec="0.75", lw=0.6))


def panel_findings(fig):
    panel_title(fig, X_RIGHT + 0.012, "Main findings")
    ax = fig.add_axes([X_RIGHT, 0.0, 1 - X_RIGHT, 0.90])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    items = [
        ("Highest rank robustness",
         "GMCC in both datasets, with $\\mu$ recomputed\nfrom each masked matrix", BLUE,
         lambda y, h: icon_curves(ax, 0.55, y + 0.25, 1.55, h - 0.5)),
        ("Shared zeros inflate GMCC similarity",
         "rarely detected genes look alike\nbecause both are mostly zero", "0.45",
         lambda y, h: icon_shared_zero(ax, 0.55, y + h / 2 - 0.36, 0.25)),
        ("Advantage holds after detection filter",
         "genes detected in ≥10% of cells:\nGMCC still the most robust", BLUE,
         lambda y, h: icon_filter(ax, 0.6, y + 0.3, 1.45, h - 0.6)),
        ("GMCC and Cosine are complementary",
         "GMCC → sparsely detected genes\nCosine → highly detected gene pairs", "0.45",
         lambda y, h: icon_detection(ax, 0.55, y + h / 2 - 0.36, 0.25)),
    ]
    top, h, gap, x_txt = 9.15, 1.66, 0.22, 2.45
    for i, (head, body, col, draw_icon) in enumerate(items):
        y = top - (i + 1) * h - i * gap
        ax.add_patch(FancyBboxPatch((0.25, y), 9.5, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.22",
                                    fc="white", ec=col, lw=1.5))
        draw_icon(y, h)
        ax.text(x_txt, y + h - 0.3, head, fontsize=10.8, fontweight="bold",
                color=INK, va="top")
        ax.text(x_txt, y + h - 0.82, body, fontsize=10, color=INK, va="top",
                linespacing=1.2)

    y = top - 4 * h - 4 * gap - 0.9
    ax.add_patch(FancyBboxPatch((0.25, y), 9.5, 0.9,
                                boxstyle="round,pad=0.02,rounding_size=0.22",
                                fc="white", ec="0.30", lw=1.4, linestyle=(0, (4, 3))))
    ax.text(5.0, y + 0.45, "Robustness ≠ biological accuracy", fontsize=11.5,
            ha="center", va="center", color="0.20", fontweight="bold")


def main(npz_path):
    tau = load_means(npz_path)
    fig = plt.figure(figsize=(13.28, 5.31))
    fig.canvas.draw()  # renderer needed by text_run
    for xs in (X_MID, X_RIGHT):
        fig.add_artist(plt.Line2D([xs, xs], [0.04, 0.95], color="0.82", lw=1.0))

    panel_design(fig)
    panel_robustness(fig, tau)
    panel_findings(fig)

    os.makedirs(FIG_DIR, exist_ok=True)
    stem = os.path.join(FIG_DIR, "graphical_abstract")
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    fig.savefig(stem + ".tiff", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("Written:", stem + ".{pdf,png,tiff}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TAU_NPZ)