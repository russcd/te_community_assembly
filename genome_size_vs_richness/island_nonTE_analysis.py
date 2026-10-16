#!/usr/bin/env python3
"""
Active TE richness versus non-TE genome size (genome size minus total TE bp).

Removes TE sequence from the "area" axis so that richness is not regressed on a
quantity that is partly made of the elements being counted. Fits the log-log
scaling exponent z (OLS) per clade against both total genome size and non-TE size,
tests heterogeneity among the three scaling clades, and draws a four-panel figure.

Inputs (all in --data, default current directory)
  kimura_subfam_summaries_tsv.gz    species, subfam, Mean_Kimura (malformed lines skipped)
  subfamily_blast_results_tsv.gz    species, subfamily, count
  TE_diversity_stats__1_.tsv        species, assembly_length
  TE_subfamily_counts_by_species_tsv.gz
                                    species, subfamily_length (total bp per subfamily,
                                    comma separated). Summed per genome to give TE bp.
  pgls_clades.tsv                   species, clade

Active TE = subfamily with mean Kimura divergence < ACTIVE_KIMURA (5%).
Land plants = Viridiplantae minus the algal genera listed below.

Outputs
  island_nonTE_genomes.tsv   per-genome table used in the fits
  island_nonTE_fits.tsv      per-clade exponents and intervals
  island_nonTE.png           four-panel figure
Usage:  python3 island_nonTE_analysis.py --data /path/to/inputs
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ACTIVE_KIMURA = 5.0
CLADES = [("Fish", "Fish", "#3E6DA6"), ("Fungi", "Fungi", "#B0453A"),
          ("Insecta", "Insects", "#C56B3E"), ("Land plants", "Land plants", "#4E8B4A")]
SCALING_CLADES = ["Fish", "Fungi", "Insecta"]
ALGAE = ["Ostreococcus", "Bathycoccus", "Micromonas", "Chlamydomonas", "Chlorella", "Coccomyxa",
         "Monoraphidium", "Cyanidioschyzon", "Chondrus", "Volvox", "Dunaliella", "Auxenochlorella",
         "Chlorokybus", "Klebsormidium", "Mesostigma", "Raphidocelis", "Tetradesmus", "Gonium",
         "Scenedesmus", "Picochlorum", "Botryococcus", "Haematococcus"]

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.labelsize": 9.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})


def load_active_richness(data):
    km = pd.read_csv(os.path.join(data, "kimura_subfam_summaries_tsv.gz"), sep="\t", on_bad_lines="skip")
    bl = pd.read_csv(os.path.join(data, "subfamily_blast_results_tsv.gz"), sep="\t")
    bl["subfam"] = bl.subfamily.str.split("#").str[0]
    m = bl.merge(km[["species", "subfam", "Mean_Kimura"]], on=["species", "subfam"], how="left")
    m = m[(m["count"] > 0) & (m.Mean_Kimura < ACTIVE_KIMURA)]
    return m.groupby("species").size().rename("S_active").reset_index()


def load_te_bp(data):
    d = pd.read_csv(os.path.join(data, "TE_subfamily_counts_by_species_tsv.gz"), sep="\t")
    tot = d.subfamily_length.apply(lambda s: sum(float(x) for x in str(s).split(",") if x != ""))
    return pd.DataFrame({"species": d.species, "te_bp": tot})


def ols_slope(x, y):
    n = len(x)
    z, ic = np.polyfit(x, y, 1)
    r = y - (z * x + ic)
    se = np.sqrt((r @ r / (n - 2)) / ((x - x.mean()) ** 2).sum())
    return z, se, ic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    asm = pd.read_csv(os.path.join(a.data, "TE_diversity_stats__1_.tsv"), sep="\t")[["species", "assembly_length"]]
    clades = pd.read_csv(os.path.join(a.data, "pgls_clades.tsv"), sep="\t")[["species", "clade"]]
    g = (load_active_richness(a.data).merge(asm, on="species").merge(clades, on="species")
         .merge(load_te_bp(a.data), on="species", how="left"))
    g = g[(g.S_active >= 1) & (g.assembly_length > 0)]
    g["te_frac"] = g.te_bp / g.assembly_length

    bad = g[(g.te_frac >= 1) | g.te_bp.isna()]
    print("genomes with active richness: %d" % len(g))
    print("dropped (TE bp >= assembly length, or missing): %d" % len(bad))
    g = g[(g.te_frac < 1) & g.te_bp.notna()].copy()

    g["nonTE"] = g.assembly_length - g.te_bp
    g["lS"] = np.log10(g.S_active)
    g["lA"] = np.log10(g.assembly_length)
    g["lN"] = np.log10(g.nonTE)
    is_alga = g.species.apply(lambda s: any(str(s).startswith(x) for x in ALGAE))
    g["grp"] = np.where((g.clade == "Viridiplantae") & (~is_alga), "Land plants", g.clade)
    g.to_csv(os.path.join(a.out, "island_nonTE_genomes.tsv"), sep="\t", index=False)

    # ---- fits ----
    rows = []
    print("\n%-12s %4s %6s %8s | %-22s | %-22s" % ("clade", "n", "TEfrac", "rngNonTE",
                                                   "z vs genome size", "z vs non-TE size"))
    for key, _, _ in CLADES:
        s = g[g.grp == key]
        z1, s1, _ = ols_slope(s.lA.values, s.lS.values)
        z2, s2, _ = ols_slope(s.lN.values, s.lS.values)
        rng = s.lN.max() - s.lN.min()
        rows.append(dict(clade=key, n=len(s), median_te_frac=s.te_frac.median(), range_nonTE_OoM=rng,
                         z_genome=z1, z_genome_lo=z1 - 1.96 * s1, z_genome_hi=z1 + 1.96 * s1,
                         z_nonTE=z2, z_nonTE_lo=z2 - 1.96 * s2, z_nonTE_hi=z2 + 1.96 * s2, se_nonTE=s2))
        print("%-12s %4d %6.2f %8.2f | %+.2f [%+.2f,%+.2f]    | %+.2f [%+.2f,%+.2f]" % (
            key, len(s), s.te_frac.median(), rng, z1, z1 - 1.96 * s1, z1 + 1.96 * s1,
            z2, z2 - 1.96 * s2, z2 + 1.96 * s2))
    fits = pd.DataFrame(rows)
    fits.to_csv(os.path.join(a.out, "island_nonTE_fits.tsv"), sep="\t", index=False)

    sc = fits[fits.clade.isin(SCALING_CLADES)]
    w = 1 / sc.se_nonTE ** 2
    zbar = (sc.z_nonTE * w).sum() / w.sum()
    Q = (w * (sc.z_nonTE - zbar) ** 2).sum()
    p_het = 1 - stats.chi2.cdf(Q, len(sc) - 1)
    print("\nfish, fungi, insects against non-TE size: weighted z = %.2f, heterogeneity Q = %.2f, P = %.2f"
          % (zbar, Q, p_het))

    # ---- figure ----
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (key, label, col), letter in zip(axes, CLADES, "abcd"):
        s = g[g.grp == key]
        f = fits[fits.clade == key].iloc[0]
        ax.scatter(s.nonTE, s.S_active, s=15, c=col, alpha=0.5, lw=0)
        z, _, ic = ols_slope(s.lN.values, s.lS.values)
        xx = np.linspace(s.lN.min(), s.lN.max(), 20)
        ax.plot(10 ** xx, 10 ** (z * xx + ic), c=col, lw=2.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(label, fontsize=10)
        ax.text(0.06, 0.90, "z = %.2f" % f.z_nonTE, transform=ax.transAxes, fontweight="bold", fontsize=9.5)
        ax.text(0.06, 0.81, "[%.2f, %.2f]" % (f.z_nonTE_lo, f.z_nonTE_hi), transform=ax.transAxes,
                fontsize=7.5, color="#555")
        ax.text(-0.02, 1.09, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
        ax.set_xlabel("non-TE genome size (bp)")
        if letter == "a":
            ax.set_ylabel("active TE richness")
    plt.tight_layout()
    out = os.path.join(a.out, "island_nonTE.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    print("saved %s" % out)


if __name__ == "__main__":
    main()
