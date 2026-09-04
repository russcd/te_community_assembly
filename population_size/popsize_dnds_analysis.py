#!/usr/bin/env python3
"""
Population-size proxy (dN/dS) versus TE content and TE arrival.

Tests whether genome-wide dN/dS (an inverse proxy for effective population size)
predicts (i) total TE content and (ii) the fraction of families that are recent
arrivals, across species and within each clade. The result: strong cross-species
correlations that collapse to within-clade nulls, reproducing the classical
mutational-hazard pattern while confirming it does not hold within lineages.

Input:  popsize_dnds_input.tsv   (one row per genome)
  columns:
    species             genome/species name
    clade               major clade assignment
    dnds                genome-wide dN/dS (inverse proxy for Ne; high dN/dS = low Ne)
    te_total_copies     total TE copy number (N)
    te_richness         number of TE subfamilies (S)
    n_recent_arrivals   count of recent arrivals (orphan families, mean Kimura <5%)
    arrival_fraction    n_recent_arrivals / te_richness
    genome_size_bp      assembly length in bp

Output: dnds_arrival.png   (4-panel figure)
        printed correlation table (cross-species and within-clade)

Usage:  python3 popsize_dnds_analysis.py
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

INPUT = "popsize_dnds_input.tsv"
OUTPUT_FIG = "dnds_arrival.png"
CLADES = ["Insecta", "Mammalia", "Fish", "Aves"]   # well-sampled clades for within-clade tests
MIN_N = 15                                          # minimum genomes to report a within-clade correlation

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
})
PALETTE = {"Mammalia": "#8C5A3B", "Aves": "#4A90A4", "Fish": "#3E6DA6",
           "Insecta": "#C56B3E", "OtherInvertebrates": "#7BA8B0"}


def main():
    d = pd.read_csv(INPUT, sep="\t")

    # ---- cross-species correlations ----
    print("=== cross-species (n=%d) ===" % len(d))
    for lab, col in [("total TE copies", "te_total_copies"),
                     ("genome size", "genome_size_bp"),
                     ("arrival fraction", "arrival_fraction")]:
        r, p = stats.spearmanr(d.dnds, d[col])
        print("  rho(dN/dS, %-16s) = %+.3f  p=%.2g" % (lab, r, p))

    # ---- within-clade correlations ----
    def within(col):
        out = []
        for c in CLADES:
            s = d[d.clade == c]
            if len(s) < MIN_N:
                continue
            r, p = stats.spearmanr(s.dnds, s[col])
            out.append((c, len(s), r, p))
        return out

    print("\n=== within-clade rho(dN/dS, total TE copies) ===")
    for c, n, r, p in within("te_total_copies"):
        print("  %-10s n=%3d  rho=%+.3f  p=%.2g" % (c, n, r, p))
    print("=== within-clade rho(dN/dS, arrival fraction) ===")
    for c, n, r, p in within("arrival_fraction"):
        print("  %-10s n=%3d  rho=%+.3f  p=%.2g" % (c, n, r, p))

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 8))

    def scatter_panel(a, ycol, ylab, logy, letter):
        for c, g in d.groupby("clade"):
            a.scatter(g.dnds, g[ycol], s=18, c=PALETTE.get(c, "#999"),
                      alpha=0.7, lw=0, label=c)
        if logy:
            a.set_yscale("log")
        r, _ = stats.spearmanr(d.dnds, d[ycol])
        a.set_xlabel("dN/dS (proxy for 1/Ne)")
        a.set_ylabel(ylab)
        a.text(0.05, 0.92, "\u03c1 = %.2f" % r, transform=a.transAxes,
               fontweight="bold", fontsize=10)
        a.text(-0.16, 1.05, letter, transform=a.transAxes, fontsize=13, fontweight="bold")

    def forest_panel(a, col, xlab, letter):
        res = within(col)[::-1]
        y = np.arange(len(res))
        for i, (c, n, r, p) in enumerate(res):
            a.scatter(r, y[i], s=60, c=PALETTE.get(c), edgecolor="white", lw=1, zorder=3)
        a.axvline(0, c="k", lw=1, ls="--", alpha=0.6)
        a.set_yticks(y)
        a.set_yticklabels(["%s (n=%d)" % (c, n) for c, n, _, _ in res], fontsize=8)
        a.set_xlabel(xlab)
        a.set_xlim(-0.6, 0.6)
        a.text(-0.28, 1.05, letter, transform=a.transAxes, fontsize=13, fontweight="bold")

    scatter_panel(ax[0, 0], "te_total_copies", "total TE copies", True, "a")
    ax[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
    forest_panel(ax[0, 1], "te_total_copies", "within-clade \u03c1 (dN/dS vs total TEs)", "b")
    scatter_panel(ax[1, 0], "arrival_fraction", "arrival fraction", False, "c")
    forest_panel(ax[1, 1], "arrival_fraction", "within-clade \u03c1 (dN/dS vs arrival)", "d")

    plt.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=220, bbox_inches="tight", facecolor="white")
    print("\nsaved %s" % OUTPUT_FIG)


if __name__ == "__main__":
    main()
