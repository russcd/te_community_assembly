#!/usr/bin/env python3
"""
Active TE richness versus TOTAL genome size (assembly length; TE bp NOT removed).
Comparison version of island_nonTE_analysis.py. Reads the per-genome table that
script wrote (island_nonTE_genomes.tsv), so no raw inputs are needed.
Outputs: island_genomesize_fits.tsv, island_genomesize.png
Usage: python3 island_genomesize_analysis.py --data DIR --out DIR
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

CLADES = [("Fish", "Fish", "#3E6DA6"), ("Fungi", "Fungi", "#B0453A"),
          ("Insecta", "Insects", "#C56B3E"), ("Land plants", "Land plants", "#4E8B4A")]
SCALING_CLADES = ["Fish", "Fungi", "Insecta"]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.labelsize": 9.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})

def ols_slope(x, y):
    n = len(x)
    z, ic = np.polyfit(x, y, 1)
    r = y - (z * x + ic)
    se = np.sqrt((r @ r / (n - 2)) / ((x - x.mean()) ** 2).sum())
    return z, se, ic

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="."); ap.add_argument("--out", default=".")
a = ap.parse_args()
g = pd.read_csv(os.path.join(a.data, "island_nonTE_genomes.tsv"), sep="\t")

rows = []
print("%-12s %4s %8s | %-22s" % ("clade", "n", "rngSize", "z vs genome size"))
for key, _, _ in CLADES:
    s = g[g.grp == key]
    z, se, _ = ols_slope(s.lA.values, s.lS.values)
    rng = s.lA.max() - s.lA.min()
    rows.append(dict(clade=key, n=len(s), range_genome_OoM=rng, z_genome=z,
                     z_genome_lo=z - 1.96 * se, z_genome_hi=z + 1.96 * se, se_genome=se))
    print("%-12s %4d %8.2f | %+.2f [%+.2f,%+.2f]" % (key, len(s), rng, z, z - 1.96 * se, z + 1.96 * se))
fits = pd.DataFrame(rows)
fits.to_csv(os.path.join(a.out, "island_genomesize_fits.tsv"), sep="\t", index=False)

sc = fits[fits.clade.isin(SCALING_CLADES)]
w = 1 / sc.se_genome ** 2
zbar = (sc.z_genome * w).sum() / w.sum()
Q = (w * (sc.z_genome - zbar) ** 2).sum()
print("\nfish, fungi, insects vs genome size: weighted z = %.2f, Q = %.2f, P = %.2f"
      % (zbar, Q, 1 - stats.chi2.cdf(Q, len(sc) - 1)))

fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.4))
axes = axes.ravel()
for ax, (key, label, col), letter in zip(axes, CLADES, "abcd"):
    s = g[g.grp == key]; f = fits[fits.clade == key].iloc[0]
    ax.scatter(s.assembly_length, s.S_active, s=15, c=col, alpha=0.5, lw=0)
    z, _, ic = ols_slope(s.lA.values, s.lS.values)
    xx = np.linspace(s.lA.min(), s.lA.max(), 20)
    ax.plot(10 ** xx, 10 ** (z * xx + ic), c=col, lw=2.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(label, fontsize=10)
    ax.text(0.06, 0.90, "z = %.2f" % f.z_genome, transform=ax.transAxes, fontweight="bold", fontsize=9.5)
    ax.text(0.06, 0.81, "[%.2f, %.2f]" % (f.z_genome_lo, f.z_genome_hi), transform=ax.transAxes, fontsize=7.5, color="#555")
    ax.text(-0.02, 1.09, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    ax.set_xlabel("genome size (bp)")
    if letter in "ac": ax.set_ylabel("active TE richness")
plt.tight_layout()
fig.savefig(os.path.join(a.out, "island_genomesize.png"), dpi=220, bbox_inches="tight", facecolor="white")
