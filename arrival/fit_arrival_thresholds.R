#!/usr/bin/env Rscript
# Threshold robustness for the arrival->evenness result.
# Fits Gini ~ l_arr + l_exp + logS under lambda-ML PGLS at Kimura thresholds 3-7%.
suppressPackageStartupMessages({library(ape);library(caper)})
tr0 <- read.tree("otl_tree.nwk")
clean <- function(x){ x<-sub("_-species_in_.*$","",x); x<-gsub("\\[|\\]","",x); trimws(gsub("_"," ",x)) }
tr0$tip.label <- clean(tr0$tip.label)

cat(sprintf("%-6s %6s %8s %8s %8s %8s\n","thr","n","b_arr","t_arr","p_arr","lambda"))
for(thr in c(3,4,5,6,7)){
  d <- read.delim(sprintf("arr_thr%d.tsv",thr), stringsAsFactors=FALSE)
  d$tip <- clean(d$tree_tip); d <- d[!duplicated(d$tip),]
  common <- intersect(d$tip, tr0$tip.label)
  d <- d[d$tip %in% common,]; rownames(d) <- d$tip
  tr <- drop.tip(tr0, setdiff(tr0$tip.label, common))
  tr$node.label <- NULL; tr <- multi2di(tr); tr$edge.length[tr$edge.length<=0]<-1e-8
  tr <- reorder(tr,"cladewise"); d <- d[tr$tip.label,]
  cd <- comparative.data(tr, d[,c("tip","gini","l_arr","l_exp","logS")], names.col="tip", vcv=TRUE)
  m <- pgls(gini ~ l_arr + l_exp + logS, data=cd, lambda="ML")
  co <- summary(m)$coefficients["l_arr",]
  cat(sprintf("%-6d %6d %+8.4f %+8.2f %8.1e %8.3f\n",
      thr, nrow(d), co[1], co[3], co[4], m$param["lambda"]))
}
