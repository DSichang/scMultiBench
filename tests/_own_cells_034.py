"""The own-data cells of the 0.3.4 tutorials, as ``tools/gen_tut.py`` of commit
``0314a35`` held them: for each of the 13 tasks the folder name, the cell that
built the stand-in objects and the cell that exported them.

Tests compare what 0.3.5 writes with what these cells write. The texts are
kept as they were; nothing here follows the generator."""

OLD = {
 "vertical_rna_adt": dict(
   name="MYCITE",
   data='''import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D11"
adata = mtb.io.read_canonical(d / "rna.h5")
adata.obsm["protein"] = mtb.io.read_canonical(d / "adt.h5").to_df()
adata.obs["celltype"] = pd.read_csv(d / "cty.csv")["x"].values
adata = sc.pp.subsample(adata, fraction=0.6, random_state=0, copy=True)
adata''',
   export='''mtb.io.export_dataset(adata, "mydata/MYCITE", rna="X", adt="obsm:protein",
                      labels="obs:celltype", overwrite=True)''',
 ),
 "vertical_rna_atac": dict(
   name="MYMULTIOME",
   data='''import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D27mini_vertical"
rna = mtb.io.read_canonical(d / "rna.h5")
rna.obs["celltype"] = pd.read_csv(d / "cty.csv")["x"].values
atac = mtb.io.read_canonical(d / "atac_peak.h5")
rna = sc.pp.subsample(rna, fraction=0.6, random_state=0, copy=True)
atac = atac[rna.obs_names].copy()
rna, atac''',
   export='''mtb.io.export_dataset(rna, "mydata/MYMULTIOME", atac=atac, atac_kind="peak",
                      labels="obs:celltype", overwrite=True)''',
 ),
 "vertical_rna_adt_atac": dict(
   name="MYTRI",
   data='''import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D22mini"
adata = mtb.io.read_canonical(d / "rna.h5")
adata.obsm["protein"] = mtb.io.read_canonical(d / "adt.h5").to_df()
adata.obs["celltype"] = pd.read_csv(d / "cty.csv")["x"].values
atac = mtb.io.read_canonical(d / "atac.h5")
adata = sc.pp.subsample(adata, fraction=0.6, random_state=0, copy=True)
atac = atac[adata.obs_names].copy()
adata, atac''',
   export='''mtb.io.export_dataset(adata, "mydata/MYTRI", rna="X", adt="obsm:protein", atac=atac,
                      atac_kind="peak", labels="obs:celltype", overwrite=True)''',
 ),
 "diagonal_rna_atac": dict(
   name="MYDIAG",
   data='''import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D27mini"
rna = mtb.io.read_canonical(d / "rna.h5")
rna.obs["celltype"] = pd.read_csv(d / "rna_cty.csv")["x"].values
atac = mtb.io.read_canonical(d / "atac_gas.h5")
atac.obs["celltype"] = pd.read_csv(d / "atac_cty.csv")["x"].values
rna = sc.pp.subsample(rna, fraction=0.9, random_state=0, copy=True)
atac = sc.pp.subsample(atac, fraction=0.9, random_state=1, copy=True)
rna, atac''',
   export='''mtb.io.export_dataset(rna, "mydata/MYDIAG", atac=atac, atac_kind="gene_activity",
                      labels="obs:celltype", category="diagonal", overwrite=True)''',
 ),
 "diagonal_multi": dict(
   name="MYDIAG",
   data='''d = mtb.config.DEFAULT.data_path / "D37mini"
n = 900
rna, atac = [], []
for b in (1, 2, 3):
    r = mtb.io.read_canonical(d / f"rna{b}.h5")
    r.obs["celltype"] = pd.read_csv(d / f"rna_cty{b}.csv")["x"].values
    a = mtb.io.read_canonical(d / f"atac_gas{b}.h5")
    a.obs["celltype"] = pd.read_csv(d / f"atac_cty{b}.csv")["x"].values
    rna.append(r[:n].copy())
    atac.append(a[:n].copy())
rna, atac''',
   export='''for b in (1, 2, 3):
    mtb.io.export_dataset(rna[b - 1], "mydata/MYDIAG", atac=atac[b - 1],
                          atac_kind="gene_activity", labels="obs:celltype",
                          category="diagonal", batch_index=b, overwrite=True)''',
 ),
 "mosaic_rna_adt": dict(
   name="MYMOSAIC",
   data='''d = mtb.config.DEFAULT.data_path / "D38mini"
n = 1500
rna1 = mtb.io.read_canonical(d / "rna1.h5")[:n]
rna2 = mtb.io.read_canonical(d / "rna2.h5")[:n]
adt2 = mtb.io.read_canonical(d / "adt2.h5")[:n]
adt3 = mtb.io.read_canonical(d / "adt3.h5")[:n]
labels = [pd.read_csv(d / f"cty{b}.csv")["x"].values[:n] for b in (1, 2, 3)]''',
   export='''out = "mydata/MYMOSAIC"
# batch 1: RNA only
mtb.io.export_dataset(rna1, out, labels=labels[0],
                      batch_index=1, category="mosaic", overwrite=True)
# batch 2: RNA + ADT
mtb.io.export_dataset(rna2, out, adt=adt2, labels=labels[1],
                      batch_index=2, category="mosaic", overwrite=True)
# batch 3: ADT only
mtb.io.export_dataset(adt3, out, rna=None, adt="X", labels=labels[2],
                      batch_index=3, category="mosaic", overwrite=True)''',
 ),
 "mosaic_rna_atac": dict(
   name="MYMOSAIC",
   data='''d = mtb.config.DEFAULT.data_path / "D45mini"
n = 1500
rna1 = mtb.io.read_canonical(d / "rna1.h5")[:n]
rna2 = mtb.io.read_canonical(d / "rna2.h5")[:n]
atac2 = mtb.io.read_canonical(d / "atac2.h5")[:n]
atac3 = mtb.io.read_canonical(d / "atac3.h5")[:n]
labels = [pd.read_csv(d / f"cty{b}.csv")["x"].values[:n] for b in (1, 2, 3)]''',
   export='''out = "mydata/MYMOSAIC"
# batch 1: RNA only
mtb.io.export_dataset(rna1, out, labels=labels[0],
                      batch_index=1, category="mosaic", overwrite=True)
# batch 2: RNA + ATAC peaks
mtb.io.export_dataset(rna2, out, atac=atac2, atac_kind="peak", labels=labels[1],
                      batch_index=2, category="mosaic", overwrite=True)
# batch 3: ATAC peaks only
mtb.io.export_dataset(atac3, out, rna=None, atac="X", atac_kind="peak", labels=labels[2],
                      batch_index=3, category="mosaic", overwrite=True)''',
 ),
 "mosaic_shared": dict(
   name="MYMOSAIC",
   data='''d = mtb.config.DEFAULT.data_path / "D46mini"
n = 1000
rna = [mtb.io.read_canonical(d / f"rna{b}.h5")[:n] for b in (1, 2, 3)]
labels = [pd.read_csv(d / f"cty{b}.csv")["x"].values[:n] for b in (1, 2, 3)]
adt1 = mtb.io.read_canonical(d / "adt1.h5")[:n]
atac2 = mtb.io.read_canonical(d / "atac2.h5")[:n]''',
   export='''out = "mydata/MYMOSAIC"
# batch 1: RNA + ADT
mtb.io.export_dataset(rna[0], out, adt=adt1, labels=labels[0],
                      batch_index=1, category="mosaic", overwrite=True)
# batch 2: RNA + ATAC peaks
mtb.io.export_dataset(rna[1], out, atac=atac2, atac_kind="peak", labels=labels[1],
                      batch_index=2, category="mosaic", overwrite=True)
# batch 3: RNA only
mtb.io.export_dataset(rna[2], out, labels=labels[2],
                      batch_index=3, category="mosaic", overwrite=True)''',
 ),
 "mosaic_unshared": dict(
   name="MYMOSAIC",
   data='''d = mtb.config.DEFAULT.data_path / "D49mini"
n = 1000
rna1 = mtb.io.read_canonical(d / "rna1.h5")[:n]
rna2 = mtb.io.read_canonical(d / "rna2.h5")[:n]
adt1 = mtb.io.read_canonical(d / "adt1.h5")[:n]
adt3 = mtb.io.read_canonical(d / "adt3.h5")[:n]
atac2 = mtb.io.read_canonical(d / "atac2.h5")[:n]
labels = [pd.read_csv(d / f"cty{b}.csv")["x"].values[:n] for b in (1, 2, 3)]''',
   export='''out = "mydata/MYMOSAIC"
# batch 1: RNA + ADT
mtb.io.export_dataset(rna1, out, adt=adt1, labels=labels[0],
                      batch_index=1, category="mosaic", overwrite=True)
# batch 2: RNA + ATAC peaks
mtb.io.export_dataset(rna2, out, atac=atac2, atac_kind="peak", labels=labels[1],
                      batch_index=2, category="mosaic", overwrite=True)
# batch 3: ADT only
mtb.io.export_dataset(adt3, out, rna=None, adt="X", labels=labels[2],
                      batch_index=3, category="mosaic", overwrite=True)''',
 ),
 "cross_rna_adt": dict(
   name="MYCROSS",
   data='''import anndata as ad
import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D52mini"
batches = []
for b in (1, 2, 3):
    a = mtb.io.read_canonical(d / f"rna{b}.h5")
    a.obsm["protein"] = mtb.io.read_canonical(d / f"adt{b}.h5").to_df()
    a.obs["celltype"] = pd.read_csv(d / f"cty{b}.csv")["x"].values
    batches.append(a)
adata = ad.concat(batches, label="batch", keys=["1", "2", "3"], index_unique="-")
adata = sc.pp.subsample(adata, fraction=0.6, random_state=0, copy=True)
adata''',
   export='''mtb.io.export_dataset(adata, "mydata/MYCROSS", rna="X", adt="obsm:protein",
                      labels="obs:celltype", batch="obs:batch", category="cross",
                      overwrite=True)''',
 ),
 "cross_rna_atac": dict(
   name="MYCROSS",
   data='''import anndata as ad
import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D56mini"
rna, atac = [], []
for b in (1, 2, 3):
    r = mtb.io.read_canonical(d / f"rna{b}.h5")
    r.obs["celltype"] = pd.read_csv(d / f"cty{b}.csv")["x"].values
    rna.append(r)
    atac.append(mtb.io.read_canonical(d / f"atac{b}.h5"))
rna = ad.concat(rna, label="batch", keys=["1", "2", "3"], index_unique="-")
atac = ad.concat(atac, keys=["1", "2", "3"], index_unique="-")
rna = sc.pp.subsample(rna, fraction=0.6, random_state=0, copy=True)
atac = atac[rna.obs_names].copy()
rna, atac''',
   export='''mtb.io.export_dataset(rna, "mydata/MYCROSS", atac=atac, atac_kind="peak",
                      labels="obs:celltype", batch="obs:batch", category="cross",
                      overwrite=True)''',
 ),
 "cross_adt_atac": dict(
   name="MYCROSS",
   data='''import anndata as ad
import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D58mini"
adt, atac = [], []
for b in (1, 2):
    a = mtb.io.read_canonical(d / f"adt{b}.h5")
    a.obs["celltype"] = pd.read_csv(d / f"cty{b}.csv")["x"].values
    adt.append(a)
    atac.append(mtb.io.read_canonical(d / f"atac{b}.h5"))
adt = ad.concat(adt, label="batch", keys=["1", "2"], index_unique="-")
atac = ad.concat(atac, keys=["1", "2"], index_unique="-")
adt = sc.pp.subsample(adt, fraction=0.6, random_state=0, copy=True)
atac = atac[adt.obs_names].copy()
adt, atac''',
   export='''mtb.io.export_dataset(adt, "mydata/MYCROSS", rna=None, adt="X", atac=atac,
                      atac_kind="peak", labels="obs:celltype", batch="obs:batch",
                      category="cross", overwrite=True)''',
 ),
 "cross_rna_adt_atac": dict(
   name="MYCROSS",
   data='''import anndata as ad
import scanpy as sc

d = mtb.config.DEFAULT.data_path / "D59mini"
batches, atac = [], []
for b in (1, 2):
    a = mtb.io.read_canonical(d / f"rna{b}.h5")
    a.obsm["protein"] = mtb.io.read_canonical(d / f"adt{b}.h5").to_df()
    a.obs["celltype"] = pd.read_csv(d / f"cty{b}.csv")["x"].values
    batches.append(a)
    atac.append(mtb.io.read_canonical(d / f"atac{b}.h5"))
adata = ad.concat(batches, label="batch", keys=["1", "2"], index_unique="-")
atac = ad.concat(atac, keys=["1", "2"], index_unique="-")
adata = sc.pp.subsample(adata, fraction=0.6, random_state=0, copy=True)
atac = atac[adata.obs_names].copy()''',
   export='''mtb.io.export_dataset(adata, "mydata/MYCROSS", rna="X", adt="obsm:protein",
                      atac=atac, atac_kind="peak", labels="obs:celltype",
                      batch="obs:batch", category="cross", overwrite=True)''',
 ),
}
