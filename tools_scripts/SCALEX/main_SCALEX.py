import os
import h5py
import time
import random
import argparse
import numpy as np
import scanpy as sc
from scalex import SCALEX
from anndata import AnnData

parser = argparse.ArgumentParser("SCALEX")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to train data1')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to train data2')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
# Hyperparameters. A default is the value this script has always used; None leaves the argument to scalex.
parser.add_argument('--min_features', type=int, default=0, help='keep cells with at least this many features (0 = no filtering)')
parser.add_argument('--min_cells', type=int, default=0, help='keep features found in at least this many cells (0 = no filtering)')
parser.add_argument('--n_top_features', type=int, default=None, help='number of highly variable features (scalex: None, chosen by its preprocessing)')
parser.add_argument('--batch_size', type=int, default=None, help='minibatch size (scalex: 64)')
parser.add_argument('--lr', type=float, default=None, help='learning rate (scalex: 0.0002)')
parser.add_argument('--max_iteration', type=int, default=None, help='training iterations (scalex: 30000)')
parser.add_argument('--seed', type=int, default=None, help='random seed (scalex: 124)')
args = parser.parse_args()

# The SCALEX script for diagonal integration requires RNA and ATAC data as input, where ATAC needs to be transformed into gene activity score. The output is a joint embedding (dimensionality reduction).
# run commond for SCALEX
# python main_SCALEX.py --path1 "../../data/dataset_final/D27/rna.h5" --path2 "../../data/dataset_final/D27/atac_gas.h5" --save_path "../../result/embedding/diagonal integration/D27/SCALEX/"

begin_time = time.time()
def data_loader(path):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
    adata = AnnData(X=(X))
    return adata

adata_rna = data_loader(args.path1)
adata_atac = data_loader(args.path2)
    
adata = adata_rna.concatenate(adata_atac, join='inner', batch_key='batch')
adata.write('RNA-ATAC.h5ad')
sc.pp.filter_genes(adata, min_cells=0)
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]

sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
sc.tl.umap(adata, min_dist=0.1)

wk_dir='./'
# only the arguments that were given are passed on; the others stay at the scalex defaults
scalex_kwargs = {k: getattr(args, k) for k in ("n_top_features", "batch_size", "lr", "max_iteration", "seed") if getattr(args, k) is not None}
adata = SCALEX(data_list = [wk_dir+'RNA-ATAC.h5ad'],
              min_features=args.min_features,
              min_cells=args.min_cells,
              outdir=wk_dir+'/pbmc_RNA_ATAC/',
              show=False,
              gpu=0,
              **scalex_kwargs)
              
embedding = adata.obsm['latent']
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=embedding)
file.close()
