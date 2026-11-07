import os
import h5py
import time
import portal
import anndata
import scipy.io
import argparse
import numpy as np
import pandas as pd
import scanpy as sc

parser = argparse.ArgumentParser("Portal")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to rna')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to atac')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
# Hyperparameters. A default is the value this script has always used; None leaves the argument to Portal.
parser.add_argument('--training_steps', type=int, default=3000, help='training steps')
parser.add_argument('--lambdacos', type=float, default=10.0, help='weight of the cosine similarity loss')
parser.add_argument('--batch_size', type=int, default=None, help='minibatch size (Portal: 500)')
parser.add_argument('--seed', type=int, default=None, help='random seed (Portal: 1234)')
parser.add_argument('--n_latent', type=int, default=None, help='dimensions of the joint embedding (Portal: 20)')
parser.add_argument('--npcs', type=int, default=None, help='principal components used as input (Portal: 30)')
parser.add_argument('--hvg_num', type=int, default=None, help='highly variable genes per dataset (Portal: 4000)')
args = parser.parse_args()

# The Portal script for diagonal integration requires RNA and ATAC data as input, where ATAC needs to be transformed into gene activity score. The output is a joint embedding (dimensionality reduction).
# run commond for Portal
# python main_Portal.py --path1 "../../data/dataset_final/D27/rna.h5" --path2 "../../data/dataset_final/D27/atac_gas.h5" --save_path "../../result/embedding/diagonal integration/D27/Portal/"

begin_time = time.time()
rna_h5_files = args.path1
h5 = h5py.File(rna_h5_files, "r")
h5_rna_data = h5['matrix/data']
h5_rna_barcodes = h5['matrix/barcodes']
h5_rna_features = h5['matrix/features']
rna_data = scipy.sparse.csr_matrix(np.array(h5_rna_data).transpose()).copy()
rna_barcodes = np.array(h5_rna_barcodes).astype(str).copy()
rna_features = np.array(h5_rna_features).astype(str).copy()
adata_rna = anndata.AnnData(rna_data)
adata_rna.obs.index = rna_barcodes
adata_rna.obs["data_type"] = "rna"
adata_rna.var.index = rna_features

atac_h5_files = args.path2
h5 = h5py.File(atac_h5_files, "r")
h5_atac_data = h5['matrix/data']
h5_atac_barcodes = h5['matrix/barcodes']
h5_atac_features = h5['matrix/features']
atac_data = scipy.sparse.csr_matrix(np.array(h5_atac_data).transpose()).copy()
atac_barcodes = np.array(h5_atac_barcodes).astype(str).copy()
atac_features = np.array(h5_atac_features).astype(str).copy()
adata_atac = anndata.AnnData(atac_data)
adata_atac.obs.index = atac_barcodes
adata_atac.obs["data_type"] = "atac"
adata_atac.var.index = atac_features

meta_rna = adata_rna.obs
meta_atac = adata_atac.obs
meta = pd.concat([meta_rna, meta_atac], axis=0)

# only the arguments that were given are passed on; the others stay at the Portal defaults
model_kwargs = {k: getattr(args, k) for k in ("batch_size", "seed", "n_latent", "npcs") if getattr(args, k) is not None}
preprocess_kwargs = {k: getattr(args, k) for k in ("hvg_num",) if getattr(args, k) is not None}
model = portal.model.Model(training_steps=args.training_steps, lambdacos=args.lambdacos, **model_kwargs)
model.preprocess(adata_rna, adata_atac, **preprocess_kwargs) # perform preprocess and PCA
model.train() # train the model
model.eval() # get integrated latent representation of cells
result = model.latent

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")

file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=np.transpose(result))
file.close()

