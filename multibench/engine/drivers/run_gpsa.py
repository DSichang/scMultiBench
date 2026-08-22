"""multibench driver for GPSA (spatial registration, category cross).

The benchmark's working GPSA script, shipped with the package: the upstream
tools_scripts/GPSA/main_GPSA.py writes only an elapsed time, never the aligned
slices. The runner passes ``--script_dir`` (unused here - GPSA has no local
imports); it is stripped before the method's own argparse runs.
"""
import os as _os, sys as _sys

if "--script_dir" in _sys.argv:
    _i = _sys.argv.index("--script_dir")
    del _sys.argv[_i:_i + 2]
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "8")

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import scanpy as sc
import scipy
from gpsa import VariationalGPSA
from gpsa import matern12_kernel, rbf_kernel
from gpsa.plotting import callback_twod
import pandas as pd
import random
from sklearn.metrics import r2_score
from sklearn.metrics import adjusted_rand_score
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
import networkx as nx
import math
import sklearn
import ot
import glob
import os
import argparse

# %%
parser = argparse.ArgumentParser('GPSA')
parser.add_argument('--data_dir', default='../unified_data/DLPFC/donor1/', help='path to the data directory')
parser.add_argument('--save_dir', default='./aligned_slices/', help='path to save the output data')
args = parser.parse_args()

# %%
def load_slices_h5ad(data_dir):
    slices = []
    file_paths = glob.glob(data_dir + "*.h5ad")
    for file_path in file_paths:
        slice_i = sc.read_h5ad(file_path)
        
        if scipy.sparse.issparse(slice_i.X):
            slice_i.X = slice_i.X.toarray()
        
        Ground_Truth = slice_i.obs['Ground_Truth']
        slice_i.obs = pd.DataFrame({'Ground_Truth': Ground_Truth})
        slices.append(slice_i)
    
    return slices

# %%
# https://github.com/andrewcharlesjones/spatial-alignment/blob/main/experiments/expression/st/st_alignment.py
def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    # adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # sc.pp.filter_cells(adata, min_counts=100)
    # sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    sc.pp.filter_genes(adata, min_cells=10)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata

# %%
# https://github.com/andrewcharlesjones/spatial-alignment/blob/main/experiments/expression/st/st_alignment.py
def process1(N_GENES,data_dir,n_views):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    slices = load_slices_h5ad(data_dir)

    use_gpu=True

  
  
    processed_slices = []
    for slice_data in slices:
        processed_data = process_data(slice_data, n_top_genes=3000)
        processed_slices.append(processed_data)
    ## Save original data
    plt.figure(figsize=(20, 6))


    # Add a 'batch' column to each AnnData object
    for i, slice_i in enumerate(processed_slices):
        slice_i.obs['batch'] = int(i)

    # Concatenate the AnnData objects
    #only keep the shared genes
    data = anndata.concat(processed_slices, merge='unique', index_unique='-')
    shared_gene_names = data.var.index.values
    data_knn = processed_slices[1][:, shared_gene_names]
    X_knn = data_knn.obsm["spatial"]
    Y_knn = data_knn.X
    Y_knn = (Y_knn - Y_knn.mean(0)) / Y_knn.std(0)
    # nbrs = NearestNeighbors(n_neighbors=2).fit(X_knn)
    # distances, indices = nbrs.kneighbors(X_knn)
    knn = KNeighborsRegressor(n_neighbors=10, weights="uniform").fit(X_knn, Y_knn)
    preds = knn.predict(X_knn)
    r2_vals = r2_score(Y_knn, preds, multioutput="raw_values")

    gene_idx_to_keep = np.where(r2_vals > 0.3)[0]
    N_GENES = min(N_GENES, len(gene_idx_to_keep))
    gene_names_to_keep = data_knn.var.index.values[gene_idx_to_keep]
    gene_names_to_keep = gene_names_to_keep[np.argsort(-r2_vals[gene_idx_to_keep])]
    r2_vals_sorted = -1 * np.sort(-r2_vals[gene_idx_to_keep])
    if N_GENES < len(gene_names_to_keep):
        gene_names_to_keep = gene_names_to_keep[:N_GENES]
    data = data[:, gene_names_to_keep]
    n_samples_list = [slice_.shape[0] for slice_ in processed_slices]
    cumulative_sum = np.cumsum(n_samples_list)
    cumulative_sum = np.insert(cumulative_sum, 0, 0)
    view_idx = [
        np.arange(cumulative_sum[ii], cumulative_sum[ii + 1]) for ii in range(n_views)
    ]

    X_list = []
    Y_list = []
    for vv in range(n_views):
        curr_X = np.array(data[data.obs.batch == vv].obsm["spatial"])
        curr_Y = data[data.obs.batch == vv].X

        curr_X = scale_spatial_coords(curr_X)
        curr_Y = (curr_Y - curr_Y.mean(0)) / curr_Y.std(0)

        X_list.append(curr_X)
        Y_list.append(curr_Y)


    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    x = torch.from_numpy(X).float().clone().to(device)
    y = torch.from_numpy(Y).float().clone().to(device)
    data_dict = {
        "expression": {
            "spatial_coords": x,
            "outputs": y,
            "n_samples_list": n_samples_list,
        }
    }
    return x,slices,data_dict , data

# %%
def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val

# %%
def train(model, loss_fn, optimizer,x,view_idx,Ns,data_dict):
    model.train()

    # Forward pass
    G_means, G_samples, F_latent_samples, F_samples = model.forward(
        X_spatial={"expression": x}, view_idx=view_idx, Ns=Ns, S=5
    )

    # Compute loss
    loss = loss_fn(data_dict, F_samples)

    # Compute gradients and take optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), G_means


###################### Metric 1 PAA ##############################
def create_binary_matrix(slice, n_categories):
    binary_matrix = np.zeros((slice.n_obs, n_categories))
    for idx, cat in enumerate(slice.obs['Ground_Truth'].cat.codes):
        binary_matrix[idx, cat] = 1
    return binary_matrix


def calculate_PAA(slices, n_categories):
    total_accuracy = 0
    num_pairs = 0  
    for i in range(len(slices)):
        for j in range(i + 1, len(slices)):  
            binary_matrix_i = create_binary_matrix(slices[i], n_categories)
            binary_matrix_j = create_binary_matrix(slices[j], n_categories)

            matched_pairs = np.dot(binary_matrix_i, binary_matrix_j.T)
            Z = slices[i].obsm['spatial']
            Z_prime = slices[j].obsm['spatial']
            cost_matrix = ot.dist(Z, Z_prime, metric='euclidean')
            ot_plan = ot.emd([], [], cost_matrix)
            total_accuracy += np.sum(ot_plan * matched_pairs)
            num_pairs += 1  

    ave_accuracy = total_accuracy / num_pairs  
    print(ave_accuracy)
    return ave_accuracy
###################### Metric 1 PAA ##############################




###################### Metric 2 SCS ##############################
def create_graph(adata, degree = 4):
        """
        Converts spatial coordinates into graph using networkx library.

        param: adata - ST Slice
        param: degree - number of edges per vertex

        return: 1) G - networkx graph
                2) node_dict - dictionary mapping nodes to spots
        """
        D = distance_matrix(adata.obsm['spatial'], adata.obsm['spatial'])
        # Get column indexes of the degree+1 lowest values per row
        idx = np.argsort(D, 1)[:, 0:degree+1]
        # Remove first column since it results in self loops
        idx = idx[:, 1:]

        G = nx.Graph()
        for r in range(len(idx)):
            for c in idx[r]:
                G.add_edge(r, c)

        node_dict = dict(zip(range(adata.shape[0]), adata.obs.index))
        return G, node_dict

def generate_graph_from_labels(adata, labels_dict):
    """
    Creates and returns the graph and dictionary {node: cluster_label} for specified layer
    """

    g, node_to_spot = create_graph(adata)
    spot_to_cluster = labels_dict

    # remove any nodes that are not mapped to a cluster
    removed_nodes = []
    for node in node_to_spot.keys():
        if (node_to_spot[node] not in spot_to_cluster.keys()):
            removed_nodes.append(node)

    for node in removed_nodes:
        del node_to_spot[node]
        g.remove_node(node)

    labels = dict(zip(g.nodes(), [spot_to_cluster[node_to_spot[node]] for node in g.nodes()]))
    return g, labels


def spatial_entropy(g, labels):
    """
    Calculates spatial entropy of graph
    """
    # construct contiguity matrix C which counts pairs of cluster edges
    cluster_names = np.unique(list(labels.values()))
    C = pd.DataFrame(0,index=cluster_names, columns=cluster_names)

    for e in g.edges():
        C[labels[e[0]]][labels[e[1]]] += 1

    # calculate entropy from C
    C_sum = C.values.sum()
    H = 0
    for i in range(len(cluster_names)):
        for j in range(i, len(cluster_names)):
            if (i == j):
                z = C[cluster_names[i]][cluster_names[j]]
            else:
                z = C[cluster_names[i]][cluster_names[j]] + C[cluster_names[j]][cluster_names[i]]
            if z != 0:
                H += -(z/C_sum)*math.log(z/C_sum)
    return H



def spatial_coherence_score(graph, labels):
    g, l = graph, labels
    true_entropy = spatial_entropy(g, l)
    entropies = []
    for i in range(1000):
        new_l = list(l.values())
        random.shuffle(new_l)
        labels = dict(zip(l.keys(), new_l))
        entropies.append(spatial_entropy(g, labels))

    return abs((true_entropy - np.mean(entropies))/np.std(entropies))


def average_spatial_coherence_score(aligned_slices):
    total_score = 0
    num_slices = len(aligned_slices)
    for aligned_slice in aligned_slices:
        labels_dict = dict(zip(aligned_slice.obs.index, aligned_slice.obs['Ground_Truth']))
        g, labels = generate_graph_from_labels(aligned_slice, labels_dict)

        score = spatial_coherence_score(g, labels)
        total_score += score

    average_spatial_coherence_score = total_score / num_slices
    print("Average Spatial Coherence Score:", average_spatial_coherence_score)
    return average_spatial_coherence_score
###################### Metric 2 SCS ##############################


###################### Metric 3 LTARI #############################
def compute_average_ltari(slices, k=1):
    average_ltari_all_slices = []
    for ref_index in range(len(slices)):
        reference_slice = slices[ref_index]
        ref_coords = reference_slice.obsm['spatial']
        nn_model = NearestNeighbors(n_neighbors=k).fit(ref_coords)
        ltari_values = []
        for query_index in range(len(slices)):
            if query_index != ref_index:
                query_slice = slices[query_index]
                query_coords = query_slice.obsm['spatial']
                _, nearest_indices = nn_model.kneighbors(query_coords)
                if k == 1:
                    transferred_labels = reference_slice.obs['Ground_Truth'].iloc[nearest_indices.flatten()].values
                else:
                    transferred_labels = np.array([reference_slice.obs['Ground_Truth'].iloc[indices].mode()[0] for indices in nearest_indices])
                ari = adjusted_rand_score(query_slice.obs['Ground_Truth'], transferred_labels)
                ltari_values.append(ari)
        average_ltari_all_slices.append(np.mean(ltari_values))
    final_average_ltari = np.mean(average_ltari_all_slices)
    return final_average_ltari

###################### Metric 3 LTARI #############################










# %%
# https://github.com/andrewcharlesjones/spatial-alignment/blob/main/experiments/expression/st/st_alignment.py
def whole_process(data_dir,save_dir,num_slices,n_labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N_GENES = 20
    N_SAMPLES = None
    N_LAYERS = num_slices
    fixed_view_idx = 1

    n_spatial_dims = 2
    n_views = num_slices
    m_G = 200
    m_X_per_view = 200

    N_LATENT_GPS = {"expression": None}
    ###############################################
    N_EPOCHS = 5000
    PRINT_EVERY = 25
    #x, slices,data_dict,data = process1(N_GENES,data_dir,file_names,num_slices,mapping_dict)
    x, slices,data_dict,data = process1(N_GENES,data_dir,n_views)
    model = VariationalGPSA(
    data_dict,
    n_spatial_dims=n_spatial_dims,
    m_X_per_view=m_X_per_view,
    m_G=m_G,
    data_init=True,
    minmax_init=False,
    grid_init=False,
    n_latent_gps=N_LATENT_GPS,
    mean_function="identity_fixed",
    kernel_func_warp=rbf_kernel,
    kernel_func_data=rbf_kernel,
    fixed_view_idx=fixed_view_idx,
    ).to(device)
    view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)



    for t in range(N_EPOCHS):
        loss, G_means =  train(model, model.loss_fn, optimizer,x,view_idx,Ns,data_dict)
        if t % PRINT_EVERY == 0:
                print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss), flush=True)
                curr_aligned_coords = G_means["expression"].detach().cpu().numpy()
        
                if model.n_latent_gps["expression"] is not None:
                    curr_W = model.W_dict["expression"].detach().numpy()
                    pd.DataFrame(curr_W).to_csv("./out/W_st.csv")






    # Convert the tensor to a numpy array on the CPU
    G_means_expression = G_means['expression'].cpu().detach().numpy()

    # Iterate through each slice and update its spatial coordinates
    for i, slice_i in enumerate(slices):
        # Get the indices of data points belonging to this slice
        slice_indices = np.where(data.obs['batch'] == i)[0]

        # Update the spatial coordinates of this slice with the corresponding aligned coordinates
        slice_i.obsm['spatial'] = G_means_expression[slice_indices, :]




    original_slices = load_slices_h5ad(data_dir)

    for original_slice, updated_slice in zip(original_slices, slices):
        original_slice.obsm['spatial'] = updated_slice.obsm['spatial']



    save_dir = os.path.join(save_dir, "GPSA_aligned_slices")
    os.makedirs(save_dir, exist_ok=True)
    # save each aligned_slice
    for i, slice in enumerate(original_slices):
        save_path = os.path.join(save_dir, f"aligned_slice_{i}.h5ad")
        sc.write(save_path, slice)



    #EVALUATION
    print("PAA of this model is:")
    PAA = calculate_PAA(original_slice,n_labels)
    print("LTARI of this model is:")
    compute_average_ltari_result = compute_average_ltari(original_slice)
    print(compute_average_ltari_result)
    print("SCS of this model is:")
    SCS = average_spatial_coherence_score(original_slice)



        # save metrics
    metrics_data = [
            {"Metric": "PAA", "Value": PAA},
            {"Metric": "SCS", "Value": SCS},
            {"Metric": "LTARI", "Value": compute_average_ltari_result}
        ]

    metrics_df = pd.DataFrame(metrics_data)
    data_dir_name = os.path.basename(os.path.normpath(data_dir))
    csv_filename = f"{data_dir_name}_metrics.csv"
    metrics_df.to_csv(os.path.join(save_dir, csv_filename), index=False)


    return original_slices

# %%
def combine(data_dir,save_dir):

    # file_names = [f for f in os.listdir(data_dir) if f.endswith('.h5ad') and os.path.isfile(os.path.join(data_dir, f))]
    
    slices = load_slices_h5ad(data_dir)
    num_slices=len(slices)
    unique_layers = set()
    
    for slice in slices:
        unique_layers.update(slice.obs['Ground_Truth'].unique())

    n_labels = len(unique_layers)
    slices_coordinated = whole_process(data_dir,save_dir,num_slices,n_labels)
    return slices_coordinated

# %%
aligned_slices = combine(args.data_dir, args.save_dir)


# python GPSA.py --data_dir '../unified_data/DLPFC/donor1/' --save_dir './aligned_slices/'