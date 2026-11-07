import os
import time
import h5py
import anndata
import argparse
import MultiMAP
import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

begin_time = time.time()
parser = argparse.ArgumentParser("MultiMAP")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to train gene')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to train peak')
parser.add_argument('--path3', metavar='DIR', default='NULL', help='path to train gene activity score')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
# Hyperparameters. A default is the value this script has always used; None leaves the argument to the library.
parser.add_argument('--strengths', type=float, nargs='+', default=[0,1], help='weight of each dataset (RNA, ATAC) on the layout')
parser.add_argument('--n_neighbors', type=int, default=None, help='number of neighbours (MultiMAP: set automatically)')
parser.add_argument('--n_components', type=int, default=None, help='dimensions of the joint embedding (MultiMAP: 2)')
parser.add_argument('--n_epochs', type=int, default=None, help='layout optimisation epochs (MultiMAP: set automatically)')
parser.add_argument('--seed', type=int, default=None, help='random seed (MultiMAP: 0)')
parser.add_argument('--n_comps', type=int, default=None, help='LSI dimensions of the ATAC peaks (MultiMAP: 50)')
parser.add_argument('--n_pcs', type=int, default=None, help='principal components of the RNA (scanpy: 50)')
args = parser.parse_args()

# The MultiMAP script for diagonal integration requires RNA and ATAC data as input, where ATAC needs to be transformed into gene activity score. The output is a joint embedding (dimensionality reduction).
# run commond for MultiMAP
# python main_MultiMAP.py --path1 "../../data/dataset_final/D27/rna.h5" --path2 "../../data/dataset_final/D27/atac_peak.h5" --path3 "../../data/dataset_final/D27/atac_gas.h5" --save_path "../../result/embedding/diagonal integration/D27/MultiMAP/"

def runMultiMAP(rna_path, atac_peaks_path, atac_genes_path):
    with h5py.File(rna_path, "r") as f:
            X = np.mat(np.array(f['matrix/data']).transpose())
            rna = AnnData(X=X)
            rna.X = csr_matrix(np.matrix(rna.X))

    with h5py.File(atac_peaks_path, "r") as f:
            X = np.mat(np.array(f['matrix/data']).transpose())
            atac_peaks = AnnData(X=X)
            atac_peaks.X = csr_matrix(np.matrix(atac_peaks.X))

    with h5py.File(atac_genes_path, "r") as f:
            X = np.mat(np.array(f['matrix/data']).transpose())
            atac_genes = AnnData(X=X)
            atac_genes.X = csr_matrix(np.matrix(atac_genes.X))
            
    # only the arguments that were given are passed on; the others stay at the library defaults
    lsi_kwargs = {k: getattr(args, k) for k in ("n_comps",) if getattr(args, k) is not None}
    pca_kwargs = {} if args.n_pcs is None else {"n_comps": args.n_pcs}
    integration_kwargs = {k: getattr(args, k) for k in ("n_neighbors", "n_components", "n_epochs", "seed") if getattr(args, k) is not None}
    MultiMAP.TFIDF_LSI(atac_peaks, **lsi_kwargs)
    atac_genes.obsm['X_lsi'] = atac_peaks.obsm['X_lsi'].copy()

    rna_pca = rna.copy()
    sc.pp.scale(rna_pca)
    sc.pp.pca(rna_pca, **pca_kwargs)
    rna.obsm['X_pca'] = rna_pca.obsm['X_pca'].copy()
    
    adata = MultiMAP.Integration([rna, atac_genes], ['X_pca', 'X_lsi'], strengths=args.strengths, **integration_kwargs)
    result = adata.obsm['X_multimap']
    return result

result = runMultiMAP(args.path1, args.path2, args.path3) # 2is peak, 3 is gas
end_time = time.time()
all_time = end_time - begin_time
print(all_time)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()
np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")
