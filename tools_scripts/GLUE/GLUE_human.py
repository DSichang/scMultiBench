import os
import h5py
import anndata
import scglue
import argparse
import itertools
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import seaborn as sns
import networkx as nx
from scipy import sparse
from anndata import AnnData
from itertools import chain
from matplotlib import rcParams
scglue.plot.set_publication_params()
rcParams["figure.figsize"] = (4, 4)
print(os.environ['CONDA_DEFAULT_ENV'])

parser = argparse.ArgumentParser("GLUE")
parser.add_argument('--path1', metavar='DIR', default="", help='path to train gene')
parser.add_argument('--path2', metavar='DIR', default="", help='path to train peak')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
# Hyperparameters. A default is the value this script has always used; None leaves the argument to scglue.
parser.add_argument('--n_top_genes', type=int, default=2000, help='number of highly variable genes')
parser.add_argument('--n_pcs', type=int, default=100, help='principal components of the RNA input')
parser.add_argument('--n_lsi', type=int, default=100, help='LSI dimensions of the ATAC input')
parser.add_argument('--latent_dim', type=int, default=None, help='dimensions of the joint embedding (scglue: 50)')
parser.add_argument('--random_seed', type=int, default=None, help='random seed (scglue: 0)')
parser.add_argument('--lr', type=float, default=None, help='learning rate (scglue: 0.002)')
parser.add_argument('--lam_graph', type=float, default=None, help='weight of the guidance graph loss (scglue: 0.02)')
parser.add_argument('--lam_align', type=float, default=None, help='weight of the modality alignment loss (scglue: 0.05)')
parser.add_argument('--max_epochs', type=int, default=None, help='training epochs (scglue: set automatically)')
parser.add_argument('--data_batch_size', type=int, default=None, help='minibatch size (scglue: 128)')
parser.add_argument('--patience', type=int, default=None, help='early-stopping patience in epochs (scglue: set automatically)')
parser.add_argument('--align_burnin', type=int, default=None, help='epochs before the alignment starts (scglue: set automatically)')
args = parser.parse_args()
# only the arguments that were given are passed on; the others stay at the scglue defaults
init_kws = {k: getattr(args, k) for k in ("latent_dim", "random_seed") if getattr(args, k) is not None}
compile_kws = {k: getattr(args, k) for k in ("lr", "lam_graph", "lam_align") if getattr(args, k) is not None}
fit_kws = {"directory": "glue"}
fit_kws.update({k: getattr(args, k) for k in ("max_epochs", "data_batch_size", "patience", "align_burnin") if getattr(args, k) is not None})
glue_kws = {"fit_kws": fit_kws}
if init_kws:
    glue_kws["init_kws"] = init_kws
if compile_kws:
    glue_kws["compile_kws"] = compile_kws

# This script is designed for GLUE (Human species), integrating RNA and ATAC (peaks). The output is a joint embedding (dimensionality reduction).
# run command for GLUE
# python GLUE_human.py --path1 "../../data/dataset_final/D27/rna.h5" --path2 "../../data/dataset_final/D27/atac_peak.h5"  --save_path "../../result/embedding/diagonal integration/D27/GLUE/"

# read data
## read rna data (raw raw count)
def load_rna(rna_path):
    with h5py.File(rna_path, "r") as f:
        X = np.mat(np.array(f['matrix/data']).transpose())
        rna = AnnData(X=X)
        rna.X = sparse.csr_matrix(np.matrix(rna.X))
        rna.var_names = np.array(f['matrix/features']).astype(str)
        rna.obs_names = np.array(f['matrix/barcodes']).astype(str)
    return(rna)

## read atac data (peak raw count)
def load_atac(atac_peaks_path):
    with h5py.File(atac_peaks_path, "r") as f:
        X = np.mat(np.array(f['matrix/data']).transpose())
        atac = AnnData(X=X)
        atac.X = sparse.csr_matrix(np.matrix(atac.X))
        atac.var_names = np.array(f['matrix/features']).astype(str)
        atac.obs_names = np.array(f['matrix/barcodes']).astype(str)
    return(atac)

def convert_gene_name(rna):
    converted_gene_names = []
    for name in rna.var_names:
        if "-" in name:
            parts = name.split("-")
            converted_parts = [part.capitalize() for part in parts]
            converted_name = "-".join(converted_parts)
        else:
            converted_name = name.capitalize()

        if 'rik' in converted_name:
            converted_name = converted_name.upper()
            converted_name = converted_name.replace('RIK', 'Rik')  
        converted_gene_names.append(converted_name)
    rna.var_names = converted_gene_names
    return(rna)

def preprocess_rna(rna):
    rna.layers["counts"] = rna.X.copy()
    sc.pp.highly_variable_genes(rna, n_top_genes=args.n_top_genes, flavor="seurat_v3")
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=args.n_pcs, svd_solver="auto")
    return (rna)

def preprocess_atac(atac):
    scglue.data.lsi(atac, n_components=args.n_lsi, n_iter=15)
    return (atac)

rna1_path = args.path1
atac1_peaks_path = args.path2

rna1 = load_rna(rna1_path)
atac1 = load_atac(atac1_peaks_path)
rna = rna1
print(rna)
atac = atac1
print(atac)

rna = convert_gene_name(rna)
rna.var_names = rna.var_names.str.upper()
rna= preprocess_rna(rna)
atac = preprocess_atac(atac)

scglue.data.get_gene_annotation(
    rna, gtf="gencode.v43.chr_patch_hapl_scaff.annotation.gtf.gz",
    gtf_by="gene_name"
)
rna.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head()

split = atac.var_names.str.split(r"[:-]")
atac.var["chrom"] = split.map(lambda x: x[0])
atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
atac.var.head()

rna = rna[:,~rna.var["chrom"].isna()]
rna

guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
scglue.graph.check_graph(guidance, [rna, atac])
scglue.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer="counts", use_rep="X_pca",
    #use_batch = "batch"
)
scglue.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_rep="X_lsi"#,use_batch = "batch"
)
guidance_hvf = guidance.subgraph(chain(
    rna.var.query("highly_variable").index,
    atac.var.query("highly_variable").index
)).copy()

glue = scglue.models.fit_SCGLUE(
    {"rna": rna, "atac": atac}, guidance_hvf,
    **glue_kws
)

rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)
result = np.concatenate([rna.obsm["X_glue"],atac.obsm["X_glue"]],0)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()
