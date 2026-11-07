# multibench: the three imports below were added; the script calls random, torch and
# scvi (lines that set the seed) without importing them. Nothing else is changed.
import random
import torch
import scvi
import os
import time
import h5py
import muon
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import multigrate as mtg
from scipy.sparse import csr_matrix

def boolean(text):
    return {'true': True, 'false': False}[text.lower()]

parser = argparse.ArgumentParser("Multigrate")
parser.add_argument('--path1', metavar='DIR', default="", help='path to train data1')
parser.add_argument('--path2', metavar='DIR', default="", help='path to train data2')
parser.add_argument('--path3', metavar='DIR', default="", help='path to train data2')
parser.add_argument('--path4', metavar='DIR', default="", help='path to train data2')
parser.add_argument('--path5', metavar='DIR', default="", help='path to train data2')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
parser.add_argument('--epochs', type = int, default=None, help='training epochs (multigrate: 200)')
parser.add_argument('--lr', type = float, default=None, help='learning rate (multigrate: 0.0005)')
parser.add_argument('--seed',  type=int,  default=1, help='path to save the output data')
# Hyperparameters. A default is the value this script has always used; None leaves the argument to multigrate.
parser.add_argument('--bs', type=int, default=None, help='minibatch size (multigrate: 256)')
parser.add_argument('--z_dim', type=int, default=None, help='dimensions of the joint embedding (multigrate: 15)')
parser.add_argument('--n_top_rna', type=int, default=4000, help='number of highly variable RNA features')
parser.add_argument('--n_top_atac', type=int, default=20000, help='number of highly variable ATAC features')
parser.add_argument('--integ_coef', type=float, default=4000, help='weight of the integration loss')
parser.add_argument('--early_stopping', type=boolean, default=None, help='true or false: stop when the validation loss stalls (multigrate: true)')
parser.add_argument('--n_epochs_kl_warmup', type=int, default=None, help='epochs of KL warm-up (multigrate: a third of the training epochs)')
        
args = parser.parse_args()
# only the arguments that were given are passed on; the others stay at the multigrate defaults
model_kwargs = {k: getattr(args, k) for k in ("z_dim",) if getattr(args, k) is not None}
train_kwargs = {k: getattr(args, k) for k in ("early_stopping", "n_epochs_kl_warmup") if getattr(args, k) is not None}
if args.epochs is not None:
    train_kwargs["max_epochs"] = args.epochs
if args.lr is not None:
    train_kwargs["lr"] = args.lr
if args.bs is not None:
    train_kwargs["batch_size"] = args.bs
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
scvi.settings.seed = args.seed
begin_time = time.time()

    
def process_rna(adata_rna,barcodes,featurs):
    adata_rna.layers['counts'] = adata_rna.X
    adata_rna.obs.index = barcodes
    adata_rna.var_names = featurs
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    return adata_rna
    
def process_atac(adata_atac,barcodes,featurs):
    adata_atac.layers['counts'] = adata_atac.X
    adata_atac.obs.index = barcodes
    adata_atac.var_names = featurs
    sc.pp.normalize_total(adata_atac, target_sum=1e4)
    sc.pp.log1p(adata_atac)
    adata_atac.layers['log-norm'] = adata_atac.X.copy()
    return adata_atac

def process_adt(adata_adt,barcodes,featurs):
    adata_adt.layers['counts'] = adata_adt.X
    adata_adt.obs.index = barcodes
    adata_adt.var_names = featurs
    muon.prot.pp.clr(adata_adt)
    return adata_adt
    

def h5_to_matrix(path):
    with h5py.File(path, "r") as f:
        X = csr_matrix(np.mat(np.array(f['matrix/data']).transpose()))
        barcodes = []
        for key in f['matrix/barcodes']:
            barcodes.append(key.decode('UTF-8'))
        features = []
        for key in f['matrix/features']:
            features.append(key.decode('UTF-8'))
    return X, barcodes, features
    
rna_path1 = args.path1
rna_path2 = args.path2
adt_path1 = args.path3
adt_path3 = args.path4
atac_path2 = args.path5


rna1, barcodes1, rna_features = h5_to_matrix(rna_path1)
print(len(barcodes1), len(rna_features))
rna2, barcodes2, rna_features = h5_to_matrix(rna_path2)
print(len(barcodes2), len(rna_features))
adt1, barcodes1, adt_features = h5_to_matrix(adt_path1)
print(len(barcodes1), len(adt_features))
adt3, barcodes3, adt_features = h5_to_matrix(adt_path3)
print(len(barcodes3), len(adt_features))
atac2, barcodes2, atac_features = h5_to_matrix(atac_path2)
print(len(barcodes2), len(atac_features))

print(rna2.shape)
print(len(barcodes2))
print(len(rna_features))
adata_rna1 = process_rna(ad.AnnData(rna1), barcodes1, rna_features)
adata_rna2 = process_rna(ad.AnnData(rna2), barcodes2, rna_features)
adata_adt1 = process_adt(ad.AnnData(adt1), barcodes1, adt_features)
adata_adt3 = process_adt(ad.AnnData(adt3), barcodes3, adt_features)
adata_atac2 = process_atac(ad.AnnData(atac2), barcodes2, atac_features)

adata_rna1.obs['Modality'] = "batch1"
adata_rna2.obs['Modality'] = "batch2"
adata_adt1.obs['Modality'] = "batch1"
adata_adt3.obs['Modality'] = "batch3"
adata_atac2.obs['Modality'] = "batch2"

adata_rna = ad.concat([adata_rna1, adata_rna2], merge='same')
sc.pp.highly_variable_genes(adata_rna, n_top_genes=args.n_top_rna,  batch_key='Modality')
adata_rna1 = adata_rna1[:, adata_rna.var.highly_variable]
adata_rna2 = adata_rna2[:, adata_rna.var.highly_variable]


sc.pp.highly_variable_genes(adata_atac2, n_top_genes=args.n_top_atac,  batch_key='Modality')
adata_atac2 = adata_atac2[:, adata_atac2.var.highly_variable]

adata_rna1.obs.index = "batch1-" +adata_rna1.obs.index
adata_rna2.obs.index = "batch2-" +adata_rna2.obs.index
adata_adt1.obs.index = "batch1-" +adata_adt1.obs.index
adata_adt3.obs.index = "batch3-" +adata_adt3.obs.index
adata_atac2.obs.index = "batch2-" +adata_atac2.obs.index
    
adata = mtg.data.organize_multiome_anndatas(
    adatas = [[adata_rna1, adata_rna2, None], [adata_adt1, None, adata_adt3], [None, adata_atac2, None]],
    layers = [['counts', 'counts', None], [None, None, None],[None,  'log-norm', None]],
)
adata

mtg.model.MultiVAE.setup_anndata(
    adata,
    categorical_covariate_keys=['Modality'],
    rna_indices_end=args.n_top_rna
)

model = mtg.model.MultiVAE(
    adata,
    losses=['nb', 'mse', 'mse'],
    loss_coefs={'kl': 1e-3,
               'integ': args.integ_coef,
               },
    integrate_on='Modality',
    mmd='marginal',
    **model_kwargs
)

model.train(**train_kwargs)

model.get_latent_representation()
result = adata.obsm['latent']


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

np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")
