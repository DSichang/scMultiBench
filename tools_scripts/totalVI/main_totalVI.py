import os
import time
import scvi
import h5py
import anndata
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
from scipy.sparse import csr_matrix

def boolean(text):
    return {'true': True, 'false': False}[text.lower()]

parser = argparse.ArgumentParser("totalVI")
parser.add_argument('--path1', metavar='DIR', nargs='+', default=[], help='path to rna')
parser.add_argument('--path2', metavar='DIR', nargs='+', default=[], help='path to adt')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
# Hyperparameters. A default is the value this script has always used; None leaves the argument to scvi-tools.
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--n_latent', type=int, default=None, help='dimensions of the joint embedding (scvi-tools: 20)')
parser.add_argument('--max_epochs', type=int, default=None, help='training epochs (scvi-tools: from the number of cells, at most 400)')
parser.add_argument('--lr', type=float, default=None, help='learning rate (scvi-tools: 0.004)')
parser.add_argument('--batch_size', type=int, default=None, help='minibatch size (scvi-tools: 256)')
parser.add_argument('--early_stopping', type=boolean, default=None, help='true or false: stop when the validation loss stalls (scvi-tools: true)')
parser.add_argument('--n_epochs_kl_warmup', type=int, default=None, help='epochs of KL warm-up (scvi-tools: set automatically)')
parser.add_argument('--empirical_protein_background_prior', choices=['true', 'false', 'auto'], default='false', help='protein background prior from the data; auto lets scvi-tools decide from the panel size')
parser.add_argument('--gene_likelihood', choices=['nb', 'zinb'], default=None, help='RNA count distribution (scvi-tools: nb)')
args = parser.parse_args()

# The totalVI script for vertical&cross integration requires RNA and ADT data as input. The output is a joint embedding (dimensionality reduction).
# run commond for totalVI
# vertical integration:
# python main_totalVI.py --path1 "../../data/dataset_final/D3/rna.h5" --path2 "../../data/dataset_final/D3/adt.h5"  --save_path "../../result/embedding/vertical integration/D3/totalVI/"
# cross integration:
# python main_totalVI.py --path1 "../../data/dataset_final/D51/rna1.h5" "../../data/dataset_final/D51/rna2.h5" --path2 "../../data/dataset_final/D51/adt1.h5"  "../../data/dataset_final/D51/adt2.h5" --save_path "../../result/embedding/cross integration/D51/totalVI"

begin_time = time.time()
scvi.settings.seed = args.seed
print("Last run with scvi-tools version:", scvi.__version__)

def data_loader(path):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
    adata = AnnData(X=X)
    return adata
    
def run_totalVI(file_paths):
    rna_path=file_paths['rna_path']
    adt_path=file_paths['adt_path']
    rna_list = []
    adt_list = []
    
    # read rna
    if rna_path is not None:
        for i in range(len(rna_path)):
            if rna_path[i] is None:
                rna_list.append(None)
            else:
                rna_list.append(data_loader(rna_path[i]))
    batch = []
    for i in range(len(rna_list)):
        batch.append(np.ones((rna_list[i].shape[0], 1))  + i)
    batch = np.concatenate(batch,0)
    adata = anndata.concat(rna_list)
    adata.obs["batch"] = batch
    
    # read adt
    if adt_path is not None:
        for i in range(len(adt_path)):
            if adt_path[i] is None:
                adt_list.append(None)
            else:
                adt_list.append(data_loader(adt_path[i]))
    adata_adt = anndata.concat(adt_list)
    adt = (adata_adt.X)
    a = [i for i in range(0,adata.shape[0])]
    (adata.obs).index = a
    adata.layers["counts"] = adata.X.copy()
    adata.obsm['protein_expression'] = adt
    
    scvi.model.TOTALVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        layer="counts",
        batch_key="batch"
    )
    # only the arguments that were given are passed on; the others stay at the scvi-tools defaults
    model_kwargs = {k: getattr(args, k) for k in ("n_latent", "gene_likelihood") if getattr(args, k) is not None}
    train_kwargs = {k: getattr(args, k) for k in ("max_epochs", "lr", "batch_size", "early_stopping", "n_epochs_kl_warmup") if getattr(args, k) is not None}
    prior = {"true": True, "false": False, "auto": None}[args.empirical_protein_background_prior]
    vae = scvi.model.TOTALVI(adata, empirical_protein_background_prior=prior, latent_distribution="normal", **model_kwargs)
    vae.train(**train_kwargs)
    
    result_embedding = vae.get_latent_representation()
 
    return result_embedding

file_paths = {
    "rna_path": args.path1,
    "adt_path": args.path2
}

result = run_totalVI(file_paths)
end_time = time.time()
all_time = end_time - begin_time
print(all_time)
print(result.shape)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()
np.savetxt(args.save_path+"time.csv", [all_time], delimiter=",")
