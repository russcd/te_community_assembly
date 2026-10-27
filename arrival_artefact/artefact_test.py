#!/usr/bin/env python3
"""
Is the arrival ~ richness relationship negative? (A detection artefact, in which
arrivals similar to a resident family are missed, can only push the observed
relationship DOWN as richness rises, so a positive relationship cannot be
produced by it.)

Inputs
  --counts   TE_subfamily_counts_by_species.tsv  (species list; col 'species')
  --kimura   kimura_subfam_summaries.tsv.gz      (species, Class/fam, subfam, absLen, wellCharLen, Mean_Kimura)
  --fam80    family_counts_808080_vs_RM.tsv      (species, 808080_families, RM_families)
Outputs (in --out)
  TableS_artefact.csv / .md, FigureS_artefact.pdf / .png

Definitions
  family      RepeatMasker Class/fam label; excludes Unknown, class-only labels
              (DNA, LTR, LINE, SINE, RC), and non-TE labels (rRNA, tRNA, snRNA,
              Satellite, Simple, Low complexity, etc.)
  arrival     family whose subfamilies are ALL younger than t% Kimura divergence
              (t = 3..7; main analysis t = 5)
  richness    80-80-80 family count per genome (808080_families)
Model: negative binomial GLM (alpha fixed at 1), A ~ log(S) + log(TE bp), SEs
clustered by genus. Also reported: partial Spearman rho(A, S | TE bp).
Assumption: masking does not decrease as the number of resident families rises.
"""
import argparse, gzip, io, os, re, difflib, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--counts", default="/mnt/user-data/uploads/TE_subfamily_counts_by_species__1_.tsv")
ap.add_argument("--kimura", default="/mnt/user-data/uploads/kimura_subfam_summaries__1__tsv.gz")
ap.add_argument("--fam80",  default="/mnt/user-data/uploads/family_counts_808080_vs_RM.tsv")
ap.add_argument("--out",    default="artefact_out")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
THRESHOLDS = [3, 4, 5, 6, 7]
MAIN_T = 5

# ---------------------------------------------------------------- load
species = set(pd.read_csv(args.counts, sep="\t", usecols=["species"]).species)

# the Kimura file has ~1,000 rows fused onto the previous line (missing newline
# before the species name): split them back apart
txt = gzip.open(args.kimura, "rt").read()
pat = re.compile(r"(?<=\d)(?=(?:" + "|".join(map(re.escape, sorted(species, key=len, reverse=True))) + r")\t)")
txt = pat.sub("\n", txt)
kim = pd.read_csv(io.StringIO(txt), sep="\t", dtype=str)
assert set(kim.species) <= species, "unrecognised species in Kimura table"
kim["K"] = pd.to_numeric(kim.Mean_Kimura, errors="coerce")
kim["len"] = pd.to_numeric(kim.absLen, errors="coerce")
kim = kim[(kim.K >= 0) & (kim.K <= 100)].copy()          # drops NA / impossible values
kim = kim.rename(columns={"Class/fam": "fam"})

# keep resolved TE families only
f = kim.fam
unresolved = f.isin(["Unknown", "DNA", "LTR", "LINE", "SINE", "SINE?", "LINE?", "LTR?", "DNA?", "RC", "RC?"])
nonTE = f.str.match(r"^(rRNA|tRNA|snRNA|srpRNA|Satellite|Simple|Low|ARTEFACT|Unspecified|Retroposon|scRNA|Other)", na=False)
kim = kim[~(unresolved | nonTE)].copy()

fam = kim.groupby(["species", "fam"]).agg(maxK=("K", "max"), bp=("len", "sum")).reset_index()
genome = fam.groupby("species").agg(TEbp=("bp", "sum")).copy()
genome["genus"] = genome.index.str.split("_").str[0]

# 80-80-80 counts: names carry a "_consensi" suffix; map stragglers by closest name
c80 = pd.read_csv(args.fam80, sep="\t")
c80["sp"] = c80.species.str.replace("_consensi$", "", regex=True)
for i, r in c80[~c80.sp.isin(species)].iterrows():
    m = difflib.get_close_matches(r.sp, species, 1, 0.8)
    if m: c80.loc[i, "sp"] = m[0]
dropped = sorted(c80[~c80.sp.isin(species)].sp)
c80 = c80[c80.sp.isin(species)].drop_duplicates("sp").set_index("sp")
genome["S"] = c80["808080_families"]
genome["S_RM"] = c80["RM_families"]
genome = genome.dropna(subset=["S", "S_RM"])
genome = genome[genome.S >= 1].copy()
genome["lS"] = np.log(genome.S)
genome["lbp"] = np.log(genome.TEbp)
genome["gid"] = pd.factorize(genome.genus)[0]
print(f"genomes analysed: {len(genome)}  (cluster-file names not matched to dataset: {len(dropped)})")

# ---------------------------------------------------------------- models
def nb(x, form):
    return smf.glm(form, x, family=sm.families.NegativeBinomial(alpha=1.0)) \
              .fit(cov_type="cluster", cov_kwds={"groups": x.gid})

def partial_spearman(a, b, c):
    ra, rb, rc = (stats.rankdata(v) for v in (a, b, c))
    ra = ra - np.polyval(np.polyfit(rc, ra, 1), rc)
    rb = rb - np.polyval(np.polyfit(rc, rb, 1), rc)
    return stats.pearsonr(ra, rb)

rows, data = [], {}
for t in THRESHOLDS:
    x = genome.copy()
    x["A"] = fam[fam.maxK < t].groupby("species").size().reindex(x.index, fill_value=0)
    m = nb(x, "A ~ lS + lbp"); ci = m.conf_int().loc["lS"]
    r, pr = partial_spearman(x.A, x.S, x.TEbp)
    rows.append(dict(t=t, n=len(x), arrivals=int(x.A.sum()), pct_with_arrival=100 * (x.A > 0).mean(),
                     b=m.params.lS, lo=ci[0], hi=ci[1], P=m.pvalues.lS, rho=r, P_rho=pr))
    data[t] = x
res = pd.DataFrame(rows)
x = data[MAIN_T].copy()
print(f"arrivals per genome: mean {x.A.mean():.2f}; median richness {x.S.median():.0f} "
      f"(arrivals are {100*x.A.sum()/x.S.sum():.2f}% of all families)")

x["q"] = pd.qcut(x.S, 4, labels=["Q1", "Q2", "Q3", "Q4"])
quart = x.groupby("q", observed=True).agg(n=("S", "size"), median_S=("S", "median"),
        arrivals=("A", "mean"), pct=("A", lambda v: 100 * (v > 0).mean())).reset_index()
x["dec"] = pd.qcut(x.S, 10, duplicates="drop")
dec = x.groupby("dec", observed=True).agg(S=("S", lambda v: np.log10(v).median()), A=("A", "mean"),
        se=("A", lambda v: v.std() / np.sqrt(len(v)))).reset_index()

# ---------------------------------------------------------------- table
def fmt_p(p): return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"
tA = pd.DataFrame({
    "Divergence cut-off t (%)": res.t, "Genomes": res.n, "Arrival families": res.arrivals,
    "Genomes with >=1 arrival (%)": res.pct_with_arrival.round(1),
    "Exponent [95% CI]": [f"{r.b:.2f} [{r.lo:.2f}, {r.hi:.2f}]" for r in res.itertuples()],
    "P": [fmt_p(p) for p in res.P],
    "Partial Spearman rho (TE bp held)": res.rho.round(2)})
tB = pd.DataFrame({
    "Richness quartile": quart.q.astype(str), "Genomes": quart.n,
    "Median families (80-80-80)": quart.median_S.astype(int),
    "Mean arrivals per genome": quart.arrivals.round(2),
    "Genomes with >=1 arrival (%)": quart.pct.round(1)})
with open(f"{args.out}/TableS_artefact.csv", "w") as fh:
    fh.write("Table S-A. Relationship between arrivals and richness across divergence cut-offs\n")
    tA.to_csv(fh, index=False)
    fh.write("\nTable S-B. Arrivals by richness quartile (t = 5%)\n")
    tB.to_csv(fh, index=False)
with open(f"{args.out}/TableS_artefact.md", "w") as fh:
    fh.write("**Table S-A.** Relationship between arrivals and richness across divergence cut-offs\n\n")
    fh.write(tA.to_markdown(index=False) + "\n\n**Table S-B.** Arrivals by richness quartile (t = 5%)\n\n")
    fh.write(tB.to_markdown(index=False) + "\n")
print(tA.to_string(index=False)); print(tB.to_string(index=False))

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "pdf.fonttype": 42})
BLUE = "#0072B2"
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))

a = ax[0]
a.errorbar(dec.S, dec.A, yerr=dec.se, fmt="-o", color=BLUE, ms=4, capsize=2, lw=1.4)
a.set(xlabel="Families per genome, 80-80-80 (log$_{10}$, decile medians)",
      ylabel="Mean arrival families per genome")
a.set_title("A", loc="left", fontweight="bold")

b = ax[1]
b.errorbar(res.t, res.b, yerr=[res.b - res.lo, res.hi - res.b], fmt="o", color=BLUE, capsize=2)
b.axhline(0, color="k", lw=.8)
b.set(xlabel="Divergence cut-off defining arrivals (%)", ylabel="Arrival–richness exponent",
      xticks=THRESHOLDS, ylim=(-0.05, 1.0))
b.set_title("B", loc="left", fontweight="bold")

c = ax[2]
c.bar(quart.q.astype(str), quart.pct, color=BLUE, width=.6)
c.set(xlabel="Richness quartile", ylabel="Genomes with ≥1 arrival (%)")
c.set_title("C", loc="left", fontweight="bold")

fig.tight_layout()
fig.savefig(f"{args.out}/FigureS_artefact.pdf")
fig.savefig(f"{args.out}/FigureS_artefact.png", dpi=300)
print("wrote", os.listdir(args.out))
