
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Simple signature scoring (z-score then mean per gene set), then k-means (k=4)

def score_signatures(expr: pd.DataFrame, gene_sets: dict):
    # expr: genes x samples (or samples x genes) -> transpose if needed to genes x samples
    if expr.shape[0] < expr.shape[1] and expr.index.str.upper().isin(['BRCA1','TP53']).sum() < expr.columns.str.upper().isin(['BRCA1','TP53']).sum():
        expr = expr.T
    expr_z = pd.DataFrame(StandardScaler(with_mean=True, with_std=True).fit_transform(expr.T).T,
                          index=expr.index, columns=expr.columns)
    scores = {}
    for name, genes in gene_sets.items():
        g = [g for g in genes if g in expr_z.index]
        if len(g)==0:
            scores[name] = pd.Series(0, index=expr.columns)
        else:
            scores[name] = expr_z.loc[g].mean(axis=0)
    S = pd.DataFrame(scores)
    return S


def map_k4_to_labels(S: pd.DataFrame, clusters: pd.Series):
    mapping = {}
    labels = {}
    for k in sorted(clusters.unique()):
        idx = clusters[clusters==k].index
        med = S.loc[idx].median()
        if (med['CAF']>0 and med['Tcell']>0 and med['Endothelial']>0):
            label = 'Immune-Enriched+Fibrotic'
        elif (med['Tcell']>0 and (med['CAF']<=0 or med['Endothelial']<=0)):
            label = 'Immune-Enriched'
        elif (med['CAF']>0 and med['Tcell']<=0):
            label = 'Fibrotic'
        else:
            label = 'Depleted'
        mapping[k]=label
    for smp, k in clusters.items():
        labels[smp]=mapping[k]
    return pd.Series(labels)


def main(expr_csv, genes_yaml, out_csv):
    expr = pd.read_csv(expr_csv, index_col=0)
    with open(genes_yaml) as f:
        gene_sets = yaml.safe_load(f)  # expects keys: CAF, Tcell, Endothelial
    S = score_signatures(expr, gene_sets)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = pd.Series(km.fit_predict(S), index=S.index)
    labels = map_k4_to_labels(S, clusters)
    out = pd.DataFrame({'sample': S.index, 'CAF': S['CAF'], 'Tcell': S['Tcell'], 'Endothelial': S['Endothelial'], 'tme_label': labels.values})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--expr_csv', required=True)
    ap.add_argument('--genes_yaml', required=True)
    ap.add_argument('--out_csv', required=True)
    args = ap.parse_args()
    main(args.expr_csv, args.genes_yaml, args.out_csv)
