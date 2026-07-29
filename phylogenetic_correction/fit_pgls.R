#!/usr/bin/env Rscript
# ============================================================
# Prediction 1 under phylogenetic control.
#   Response : Gini coefficient of family copy numbers
#   Predictors: log total copies (abundance) + log richness
#   Tree     : Open Tree of Life (Grafen branch lengths)
#
# Fits (a) PGLS via caper/nlme with Brownian + lambda models,
#      (b) a Bayesian phylogenetic mixed model via brms as robustness.
#
# Inputs (working dir):
#   otl_tree.nwk        (from build_otl_tree.R)
#   pgls_input.tsv      (species, tree_tip, gini, logN, logS, ...)
# ============================================================

pkgs <- c("ape","nlme","caper","phytools","geiger")
new <- pkgs[!pkgs %in% installed.packages()[,"Package"]]
if(length(new)) install.packages(new, repos="https://cloud.r-project.org")
suppressPackageStartupMessages({library(ape);library(nlme);library(caper);library(phytools)})

## ---- data ----
d <- read.delim("pgls_input.tsv", stringsAsFactors=FALSE)
tr <- read.tree("otl_tree.nwk")

# tidy tree tip labels to match d$tree_tip: strip OTL qualifier residue, underscores->spaces
clean <- function(x){
  x <- gsub("_", " ", x)
  x <- sub("\\s*-?\\s*species in .*$", "", x, ignore.case=TRUE)
  x <- gsub("\\[|\\]", "", x)
  trimws(gsub("\\s+", " ", x))
}
tr$tip.label <- clean(tr$tip.label)
d$tip <- clean(d$tree_tip)

## ---- split subspecies collapsed onto shared tips ----
# for tips used by >1 genome, duplicate the tip into distinct near-zero-length sisters
dups <- names(which(table(d$tip) > 1))
cat("shared tips to split:", length(dups), "\n")
eps <- 1e-8
for(tp in dups){
  rows <- which(d$tip == tp)
  # rename each duplicate to a unique label
  newlabs <- paste0(tp, "__", seq_along(rows))
  d$tip[rows] <- newlabs
  # bind new zero-length tips at the existing tip's position
  if(tp %in% tr$tip.label){
    for(k in seq_along(rows)){
      tr <- bind.tip(tr, tip.label=newlabs[k],
                     where=which(tr$tip.label==tp), edge.length=eps,
                     position=eps)
    }
    tr <- drop.tip(tr, tp)   # remove the original now-internal label
  }
}

## ---- prune to intersection ----
common <- intersect(d$tip, tr$tip.label)
cat("matched rows for fit:", length(common), "\n")
d <- d[d$tip %in% common, ]
tr <- drop.tip(tr, setdiff(tr$tip.label, common))
tr <- multi2di(tr)                       # resolve polytomies (zero-length)
tr$edge.length[tr$edge.length<=0] <- 1e-8
tr$node.label <- NULL                     # caper: drop internal labels (avoids tip/node clash)
tr <- reorder(tr, "cladewise")
stopifnot(!any(duplicated(tr$tip.label))) # guard against duplicate tips from splitting
rownames(d) <- d$tip
d <- d[tr$tip.label, ]                    # align order

## ================= PGLS (caper) =================
cd <- comparative.data(tr, d[,c("tip","gini","logN","logS")], names.col="tip",
                       vcv=TRUE, warn.dropped=TRUE)
cat("\n--- PGLS: gini ~ logN + logS, lambda ML ---\n")
m_lam <- pgls(gini ~ logN + logS, data=cd, lambda="ML")
print(summary(m_lam))
cat("\nlambda estimate:", m_lam$param["lambda"], "\n")

cat("\n--- PGLS: Brownian (lambda=1) for comparison ---\n")
m_bm <- pgls(gini ~ logN + logS, data=cd, lambda=1)
print(summary(m_bm)$coefficients)

# non-phylo OLS baseline
cat("\n--- OLS baseline (no phylogeny) ---\n")
print(summary(lm(gini ~ logN + logS, data=d))$coefficients)

saveRDS(list(lambda=m_lam, bm=m_bm), "pgls_fits.rds")

## ---- branch-length sensitivity: all branches = 1 -------------------
## Maximal departure from Grafen: discard all depth info, keep only
## topological node-distance. If beta_logN survives this, the branch-
## length assumption is not driving the result.
cat("\n===== BRANCH-LENGTH SENSITIVITY: all branches = 1 =====\n")
tr_eq <- tr
tr_eq$edge.length <- rep(1, nrow(tr_eq$edge))
cd_eq <- comparative.data(tr_eq, d[,c("tip","gini","logN","logS")],
                          names.col="tip", vcv=TRUE, warn.dropped=TRUE)
m_eq <- pgls(gini ~ logN + logS, data=cd_eq, lambda="ML")
cat("lambda (equal-branch tree):", m_eq$param["lambda"], "\n")
print(summary(m_eq)$coefficients)

## side-by-side comparison of beta_logN across branch-length assumptions
cmp <- rbind(
  Grafen_lambdaML = summary(m_lam)$coefficients["logN",],
  Grafen_Brownian = summary(m_bm)$coefficients["logN",],
  EqualBranch_ML  = summary(m_eq)$coefficients["logN",],
  OLS             = summary(lm(gini~logN+logS,data=d))$coefficients["logN",]
)
cat("\n--- beta_logN across branch-length / model assumptions ---\n")
print(round(cmp,5))
saveRDS(m_eq, "pgls_equalbranch.rds")

## ================= brms robustness =================
if(!requireNamespace("brms", quietly=TRUE)){
  install.packages("brms", repos="https://cloud.r-project.org")
}
if(!requireNamespace("brms", quietly=TRUE)){
  message("\nbrms unavailable; skipping Bayesian model.")
} else {
  suppressPackageStartupMessages(library(brms))
  # phylo covariance (NOT correlation) matrix, rows/cols named by tip
  A <- ape::vcv.phylo(tr)
  A <- A / max(A)                          # scale for numerical stability
  d$phylo <- d$tip                         # grouping factor must match rownames(A)
  stopifnot(all(d$phylo %in% rownames(A)))
  cat("\n--- brms phylogenetic mixed model (SLOW: 30-90 min) ---\n")
  bf1 <- bf(gini ~ logN + logS + (1|gr(phylo, cov=A)))
  # weakly-informative priors keep sampling stable on a big dense cov matrix
  pri <- c(prior(normal(0,1), class="b"),
           prior(normal(0,1), class="Intercept"),
           prior(exponential(2), class="sd"),
           prior(exponential(2), class="sigma"))
  fit <- brm(bf1, data=d, data2=list(A=A),
             family=gaussian(), prior=pri,
             chains=4, cores=4, iter=3000, warmup=1500,
             control=list(adapt_delta=0.99, max_treedepth=12), seed=1)
  print(summary(fit))
  saveRDS(fit, "brms_fit.rds")
  # phylogenetic signal as a variance partition (Bayesian analogue of lambda)
  h <- hypothesis(fit,
        "sd_phylo__Intercept^2 / (sd_phylo__Intercept^2 + sigma^2) = 0",
        class=NULL)
  cat("\nPhylogenetic signal (variance partition, ~lambda):\n"); print(h)
  cat("\nlogN 95% CrI:\n"); print(posterior_interval(fit, pars="b_logN"))
}

cat("\nDONE. Key question: does the logN coefficient stay positive & significant\n",
    "under lambda-ML PGLS (and brms)?  If yes, Prediction 1 is not a phylogenetic artifact.\n")
