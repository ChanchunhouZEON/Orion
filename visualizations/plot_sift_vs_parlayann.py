#!/usr/bin/env python3
"""Plot QPS vs Recall@10 for unified StagedDiskANN (search) vs ParlayANN
Vamana on SIFT1M. Data source: two back-to-back benchmark runs with
alternating execution order (PA-first then Staged; Staged-first then PA).
Averaged geometrically to cancel the per-session thermal bias.

See README 'Unified Search (April 2026)' section for methodology.
"""

import matplotlib.pyplot as plt
from chart_style import PALETTE, style_ax


def main():

    # Run 1: PA (cold) → Staged (warm)
    pa1 = [
        (0.529, 196900), (0.6238, 171800), (0.746, 146200), (0.7777, 136700),
        (0.8034, 125900), (0.8719, 119000), (0.9007, 102500), (0.9353, 85640),
        (0.9525, 73450), (0.9703, 57890), (0.9804, 50570), (0.9914, 36460),
        (0.9953, 29520), (0.9992, 13150), (0.9997, 9450),  (0.9999, 5653),
    ]
    staged1 = [
        (0.9095, 97347), (0.9323, 86790), (0.9471, 73907), (0.9664, 58431),
        (0.9769, 52886), (0.9833, 49983), (0.9877, 45144), (0.9903, 40748),
        (0.9937, 30954), (0.9960, 24847), (0.9974, 20791), (0.9983, 16751),
        (0.9987, 13877), (0.9991, 10120),
    ]

    # Run 2: Staged (cold) → 20s cooldown → PA (warm)
    pa2 = [
        (0.529, 138600), (0.6238, 147300), (0.746, 119300), (0.7777, 113900),
        (0.8034, 112800), (0.8719, 102100), (0.9089, 90190), (0.9353, 74950),
        (0.9602, 58430), (0.9703, 51280), (0.9804, 42480), (0.9914, 30850),
        (0.9953, 23940), (0.9992, 10860), (0.9995, 8753),  (0.9999, 4142),
    ]
    staged2 = [
        (0.9095, 101366), (0.9323, 89774), (0.9471, 80003), (0.9664, 67548),
        (0.9769, 56616), (0.9833, 52102), (0.9877, 46389), (0.9903, 41617),
        (0.9937, 35015), (0.9960, 28577), (0.9974, 22965), (0.9983, 17424),
        (0.9987, 13900), (0.9991, 12099),
    ]

    # Geometric-mean average per curve to cancel thermal bias.
    def geo(a, b):
        return [(ar, (aq * bq) ** 0.5) for (ar, aq), (br, bq) in zip(a, b)]

    pa_avg = geo(pa1, pa2) if len(pa1) == len(pa2) else pa1
    staged_avg = geo(staged1, staged2)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    pa_r = [p[0] for p in pa_avg]
    pa_q = [p[1] for p in pa_avg]
    s_r = [p[0] for p in staged_avg]
    s_q = [p[1] for p in staged_avg]

    ax.plot(pa_r, pa_q, marker="s", linewidth=2.0, markersize=6,
            label="ParlayANN Vamana", color=PALETTE['grey'])
    ax.plot(s_r, s_q, marker="o", linewidth=2.2, markersize=6,
            label="StagedDiskANN (unified search)", color=PALETTE['blue'])

    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS (queries / sec)")
    ax.set_yscale("log")
    # ParlayANN's sweep starts well below 0.9 (down to ~0.53 on SIFT)
    # — pin the left edge to the leftmost data point across both
    # curves so we show the whole comparison, not just the high-recall
    # tail.
    ax.set_xlim(max(0.0, min(pa_r + s_r) - 0.01), 1.0)
    ax.set_title("SIFT1M — StagedDiskANN vs ParlayANN on ParlayANN base graph\n"
                 "(geometric mean of two back-to-back runs)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left")
    style_ax(ax)

    out = "visualizations/qps_recall_sift_vs_parlayann.png"
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
