import os
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import pyviper
from scipy import sparse as sp


#constructing the metacells
def generate_metacells(adata_path: str, out_dir: str, output_name: str, n_target: int):
    print(f"Loading: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    adata.var_names_make_unique()
    print(f"  Loaded: {adata.n_obs:,} cells  x  {adata.n_vars:,} genes")

    #if too large, subsample
    MAX_CELLS = 50_000
    if adata.n_obs > MAX_CELLS:
        print(f"  Subsampling {adata.n_obs:,} → {MAX_CELLS:,} cells...")
        sc.pp.subsample(adata, n_obs=MAX_CELLS, random_state=42)

    #PCA space
    adata_raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="randomized")
    pyviper.pp.corr_distance(adata)

    #constructing the metacells
    all_metacells = []
    n_cells = adata.n_obs
    print(f"Building metacells from {n_cells:,} cells (n_target={n_target})...")

    if n_cells > 60:
        try:
            pyviper.pp.repr_metacells(
                adata,
                counts=adata_raw,
                pca_slot="X_pca",
                dist_slot="corr_dist",
                n_cells_per_metacell=n_target,
                perc_data_to_use=None,
                size=500,
                min_median_depth=None,
                perc_incl_data_reused=None,
                key_added="temp_mc",
                njobs=1,
                verbose=False,
            )
            all_metacells.append(adata.uns["temp_mc"])
            print("  pyviper repr_metacells succeeded.")
        except Exception as e:
            print(f"  pyviper failed ({e}), falling back to sum-aggregation...")
            all_metacells = _fallback_metacells(adata_raw, n_cells, n_target, adata.var_names)
    else:
        print("  Too few cells for pyviper; using sum-aggregation fallback.")
        all_metacells = _fallback_metacells(adata_raw, n_cells, n_target, adata.var_names)

    #renormalize after combining
    final_mc_df = pd.concat(all_metacells)
    mc_adata = ad.AnnData(
        X=final_mc_df.values,
        obs=pd.DataFrame(index=final_mc_df.index),
        var=pd.DataFrame(index=final_mc_df.columns),
    )
    sc.pp.normalize_total(mc_adata, target_sum=1e4)
    sc.pp.log1p(mc_adata)

    # ARACNe3 expects genes as rows, metacells as columns
    expr_df = pd.DataFrame(
        mc_adata.X, index=mc_adata.obs_names, columns=mc_adata.var_names
    ).T

    #save
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_name)
    expr_df.to_csv(out_path, sep="\t")
    print(f"\nSaved: {expr_df.shape[0]:,} genes x {expr_df.shape[1]:,} metacells")
    print(f"Output: {out_path}")


def _fallback_metacells(adata_raw, n_cells, n_target, var_names):
    #Simple sum-aggregation fallback when pyviper repr_metacells fails.
    n_mc = max(1, int(n_cells / n_target))
    groups = np.array_split(np.arange(n_cells), n_mc)
    metacells = []
    for i, g_idx in enumerate(groups):
        mc_val = np.sum(adata_raw.X[g_idx, :], axis=0)
        if sp.issparse(mc_val):
            mc_val = mc_val.toarray()
        df = pd.DataFrame(mc_val.reshape(1, -1), index=[f"mc_{i}"], columns=var_names)
        metacells.append(df)
    return metacells


#cli
def parse_args():
    p = argparse.ArgumentParser(description="Generate ARACNe3-ready metacells from scRNA-seq H5AD")
    p.add_argument("--input_h5ad",   required=True, help="Path to preprocessed scRNA-seq H5AD")
    p.add_argument("--output_dir",   required=True, help="Directory to write the output TSV")
    p.add_argument("--output_name",  default="metacells.tsv",
                   help="Filename for the output TSV (default: metacells.tsv)")
    p.add_argument("--n_target",     type=int, default=10,
                   help="Target cells per metacell (default: 10)")
    return p.parse_args()


def main():
    args = parse_args()
    generate_metacells(args.input_h5ad, args.output_dir, args.output_name, args.n_target)


if __name__ == "__main__":
    main()
