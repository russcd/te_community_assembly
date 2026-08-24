#!/usr/bin/env Rscript
# PGLS test of the arrival -> evenness result.
#   Gini ~ log(n_arrival) + log(n_expansion) + logS   under phylogenetic control
# Inputs: otl_tree.nwk, pgls_arrival2_input.tsv
pkgs <- c("ape","nlme","caper","phytools")
new <- pkgs[!pkgs %in% installed.packages()[,"Package"]]
if(length(new)) install.packages(new, repos="https://cloud.r-project.org")
suppressPackageStartupMessages({library(ape);library(nlme);library(caper);library(phytools)})

d  <- read.delim("pgls_arrival2_input.tsv", stringsAsFactors=FALSE)
tr <- read.tree("otl_tree.nwk")

clean <- function(x){ x<-sub("_-species_in_.*$","",x); x<-gsub("\\[|\\]","",x); trimws(gsub("_"," ",x)) }
tr$tip.label <- clean(tr$tip.label)
d$tip <- clean(d$tree_tip)
d <- d[!duplicated(d$tip),]
common <- intersect(d$tip, tr$tip.label)
cat("matched:", length(common), "\n")
d <- d[d$tip %in% common,]; rownames(d) <- d$tip
tr <- drop.tip(tr, setdiff(tr$tip.label, common))
tr$node.label <- NULL; tr <- multi2di(tr); tr$edge.length[tr$edge.length<=0] <- 1e-8
tr <- reorder(tr,"cladewise"); d <- d[tr$tip.label,]

cd <- comparative.data(tr, d[,c("tip","gini","J","l_arr","l_exp","logS")],
                       names.col="tip", vcv=TRUE, warn.dropped=TRUE)

cat("\n=== OLS (no phylogeny): Gini ~ arrival + expansion + logS ===\n")
print(summary(lm(gini ~ l_arr + l_exp + logS, data=d))$coefficients)

cat("\n=== PGLS lambda-ML: Gini ~ arrival + expansion + logS ===\n")
m <- pgls(gini ~ l_arr + l_exp + logS, data=cd, lambda="ML")
print(summary(m))
cat("lambda:", m$param["lambda"], "\n")

cat("\n=== PGLS lambda-ML: J ~ arrival + expansion + logS ===\n")
mj <- pgls(J ~ l_arr + l_exp + logS, data=cd, lambda="ML")
print(summary(mj)$coefficients)

cat("\nKEY: does l_arr keep its sign (neg for Gini / pos for J) & significance under PGLS?\n")
