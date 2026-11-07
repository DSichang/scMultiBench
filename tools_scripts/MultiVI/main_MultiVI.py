import os
import scvi
import muon
import h5py
import random
import anndata
import argparse
import numpy as np
import mudata as md
import scanpy as sc
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from util import data_loader_multi_single, data_loader_multi_multi, split_dataset_by_modality, organize_multiome_datasets, sort_features_by_modality, filter_features, setup_anndata_for_multivi, get_normalized_expression, create_multivi_model, train_multivi_model, get_accessibility_estimates, save_multivi_model, load_multivi_model

random.seed(1)
def boolean(text):
    return {'true': True, 'false': False}[text.lower()]

parser = argparse.ArgumentParser("MultiVI")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to  RNA')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to  ATAC')
parser.add_argument('--pair_path1', metavar='DIR', default='NULL', help='path to RNA of multiome')
parser.add_argument('--pair_path2', metavar='DIR', default='NULL', help='path to ATAC of multiome')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
# Hyperparameters. A default is the value this script has always used; None leaves the argument to scvi-tools.
parser.add_argument('--max_epochs', type=int, default=500, help='training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=None, help='minibatch size (scvi-tools: 128)')
parser.add_argument('--early_stopping', type=boolean, default=None, help='true or false: stop when the validation loss stalls (scvi-tools: true)')
parser.add_argument('--n_epochs_kl_warmup', type=int, default=None, help='epochs of KL warm-up (scvi-tools: 50)')
parser.add_argument('--n_latent', type=int, default=None, help='dimensions of the joint embedding (scvi-tools: set automatically)')
parser.add_argument('--modality_weights', type=str, default=None, help='how the modalities are weighted: equal, cell or universal (scvi-tools: equal)')
parser.add_argument('--modality_penalty', type=str, default=None, help='modality alignment penalty: Jeffreys, MMD or None (scvi-tools: Jeffreys)')
parser.add_argument('--seed', type=int, default=None, help='scvi-tools random seed (scvi-tools: 0)')
args = parser.parse_args()
if args.seed is not None:
    scvi.settings.seed = args.seed
# only the arguments that were given are passed on; the others stay at the scvi-tools defaults
model_kwargs = {k: getattr(args, k) for k in ("n_latent", "modality_weights", "modality_penalty") if getattr(args, k) is not None}
train_kwargs = {k: getattr(args, k) for k in ("batch_size", "early_stopping", "n_epochs_kl_warmup") if getattr(args, k) is not None}

# The MultiVI script for mosaic integration requires [RNA, RNA+ATAC, ADT]. The output is a joint embedding (dimensionality reduction).
# run commond for MultiVI
# python main_MultiVI.py --path1  "../../data/dataset_final/D45/rna1.h5" --path2 "../../data/dataset_final/D45/atac3.h5" --pair_path1 "../../data/dataset_final/D45/rna2.h5" --pair_path2 "../../data/dataset_final/D45/atac2.h5"  --save_path "../../result/embedding/mosaic integration/D45/MultiVI/"

"""# Module2 Creating, saving and loading the model"""


def run_MultiVI(file_paths, n_epochs=500, lr=1e-3, model_kwargs={}, train_kwargs={}):

    # Load and preprocess the data
    rna_path = file_paths["rna_path_single"]
    atac_path = file_paths["atac_path_single"]
    adata_single_rna = data_loader_multi_single(rna_path, 'rna', "Gene Expression")
    adata_single_atac = data_loader_multi_single(atac_path, 'atac', "Peak")
    adata_multi = data_loader_multi_multi(file_paths['rna_path_paired'], file_paths['atac_path_paired'], "rna","atac", "Gene Expression", "Peak")
    # Organize the multiome datasets
    adata = organize_multiome_datasets(adata_multi,adata_single_rna, adata_single_atac)
    # Sort features by modality
    adata = sort_features_by_modality(adata)
    # Filter features
    adata = filter_features(adata,1)
    # Get the names of all features (genes and peaks) after filtering
    genes_to_impute = adata.var.index.tolist()
    # Setup anndata for multivi
    setup_anndata_for_multivi(adata, batch_key="modality")
    # Create the model
    model = create_multivi_model(adata, **model_kwargs)
    train_multivi_model(model, adata, n_epochs=n_epochs, lr=lr, **train_kwargs)
    normalized_expression = get_normalized_expression(model)
    accessibility_estimates = get_accessibility_estimates(model)
    return model, adata, adata_single_rna, adata_single_atac, adata_multi, normalized_expression, accessibility_estimates

file_paths = {
    "rna_path_single": [
        args.path1
    ],
    "atac_path_single": [
        args.path2
    ],
    "rna_path_paired": [
        args.pair_path1
    ],
    "atac_path_paired": [
        args.pair_path2
    ]
}
n_epochs = args.max_epochs
lr = args.lr
model, adata, adata_single_rna, adata_single_atac, adata_multi, normalized_expression, accessibility_estimates = run_MultiVI(
    file_paths,n_epochs,lr,model_kwargs,train_kwargs
)

latent_key = "X_multivi"
adata.obsm[latent_key] = model.get_latent_representation()
result = adata.obsm[latent_key]
result1 = result[0:adata_single_atac.shape[0],:]
result1 = np.transpose(result1)
result2 = result[adata_single_atac.shape[0]:(adata_single_atac.shape[0]+adata_single_rna.shape[0]),:]
result2 = np.transpose(result2)
result3 = result[(adata_single_atac.shape[0]+adata_single_rna.shape[0]):,:]
result3 = np.transpose(result3)
result = np.concatenate([result2, result3, result1], 1)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()

file = h5py.File(args.save_path+"/imputed_rna.h5", 'w')
file.create_dataset('data', data=np.transpose(normalized_expression))
file.close()

file = h5py.File(args.save_path+"/imputed_atac.h5", 'w')
file.create_dataset('data', data=np.transpose(accessibility_estimates))
file.close()
