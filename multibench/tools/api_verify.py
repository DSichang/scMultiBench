"""Exhaustive verification of the multibench public API.

Every genuine public entry is exercised with real inputs. Destructive operations
run in dry-run. Stdlib re-exports (Path, annotations, dataclass, field) are
excluded as non-API. Ends with a coherent multi-scenario plotting chain.
"""
import warnings, json, inspect, tempfile, os, traceback
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import multibench as mtb

RESULTS = "/media/disk2/Sichang/scmbench_pkg/notebooks/results"
OUT = "/tmp/apiverify"; os.makedirs(OUT, exist_ok=True)
checks = []

def check(name):
    def deco(fn):
        try:
            r = fn()
            checks.append((name, "PASS", str(r)[:88]))
        except Exception as e:
            checks.append((name, "FAIL", f"{type(e).__name__}: {e}"[:88]))
            traceback.print_exc()
        return fn
    return deco

# ---------------- discovery ----------------
@check("list_methods")
def _(): 
    a = mtb.list_methods(); b = mtb.list_methods(category="vertical")
    assert len(a) == 40 and 0 < len(b) < len(a); return f"{len(a)} all / {len(b)} vertical"

@check("list_methods(runnable=)")
def _():
    r = mtb.list_methods(runnable=True); return f"{len(r)} runnable"

@check("list_tasks")
def _():
    t = mtb.list_tasks(); assert t; return t[:5]

@check("find_methods(category+modalities)")
def _():
    m = mtb.find_methods(category="vertical", modalities=["rna", "adt"]); assert len(m) >= 14; return f"{len(m)} methods"

@check("find_methods(task/needs_labels/atac)")
def _():
    a = mtb.find_methods(task="clustering"); b = mtb.find_methods(needs_labels=True)
    c = mtb.find_methods(atac="gene_activity"); return f"task={len(a)} labels={len(b)} atac={len(c)}"

@check("method_info")
def _():
    i = mtb.method_info("Matilda"); assert i["env"] and "params" in i; return f"env={i['env']} variants={len(i['variants'])}"

@check("method_info(files_dir=)")
def _():
    i = mtb.method_info("Matilda", files_dir="/media/disk2/Sichang/scmbench_pkg/multibench/files")
    return {k: i.get(k) for k in ("deep_learning", "output")}

@check("params_for")
def _():
    p = mtb.params_for("scMDC", "vertical", ["rna", "adt"]); assert p["tunable"]; return f"{len(p['tunable'])} tunable"

@check("params_for(single-variant)")
def _():
    from multibench.engine import registry
    s = next(x.id for x in registry.load() if len(x.variants) == 1)
    return mtb.params_for(s)["variant"]

# ---------------- data resolution ----------------
@check("inputs_for(check=True)")
def _():
    i = mtb.inputs_for("D11", "Matilda", "vertical", modalities=["rna", "adt"], check=True); return list(i)

@check("inputs_for(grouped roles / cross)")
def _():
    i = mtb.inputs_for("D52", "totalVI", "cross",
                       modalities=["rna1","rna2","rna3","adt1","adt2","adt3"], check=True)
    assert len(i) == 6; return f"{len(i)} roles"

@check("labels_for")
def _():
    l = mtb.labels_for("D11"); assert l; return list(l)

@check("available_datasets(no-arg)")
def _():
    d = mtb.available_datasets(); return f"{len(d)} datasets"

@check("available_datasets(category)")
def _():
    return f"{len(mtb.available_datasets('vertical'))} vertical"

# ---------------- catalog / config ----------------
@check("catalog.methods")
def _(): return f"{mtb.catalog.methods().shape}"
@check("catalog.datasets")
def _(): return f"{mtb.catalog.datasets().shape}"
@check("catalog.metrics")
def _(): return f"{mtb.catalog.metrics().shape}"
@check("catalog.canonical_id")
def _(): return mtb.catalog.canonical_id("totalvi")
@check("catalog.canonical_metric")
def _(): return mtb.catalog.canonical_metric("ARI")
@check("config.category_folder")
def _(): return mtb.config.category_folder("vertical")
@check("config.metric_set_dir")
def _(): return mtb.config.metric_set_dir("scib")
@check("config.DEFAULT")
def _(): return str(mtb.config.DEFAULT.data_path)

# ---------------- io ----------------
@check("io.to_canonical + read_canonical")
def _():
    import h5py
    src = os.path.join(OUT, "src.h5")
    with h5py.File(src, "w") as h:
        g = h.create_group("matrix"); g.create_dataset("data", data=np.random.rand(6, 5))
    dst = mtb.io.to_canonical(src, os.path.join(OUT, "canon.h5"))
    ad = mtb.io.read_canonical(dst)   # returns AnnData (cells x features), not a bare array
    assert ad.shape == (5, 6), ad.shape
    return f"round-trip AnnData {ad.shape}"

@check("io.normalize_peak_names")
def _():
    import h5py
    src = os.path.join(OUT, "peaks.h5"); dst = os.path.join(OUT, "peaks_norm.h5")
    with h5py.File(src, "w") as h:
        g = h.create_group("matrix"); g.create_dataset("data", data=np.random.rand(3, 4))
        g.create_dataset("features", data=np.array([b"chr1_1_200", b"chr1_300_400", b"chr2_5_9"]))
    mtb.io.normalize_peak_names(src, dst); return os.path.exists(dst)

# ---------------- env ----------------
@check("env.group_for / groups")
def _(): return f"{mtb.env.group_for('Matilda')} / {len(mtb.env.groups())} groups"
@check("env.default_env_name")
def _(): return mtb.env.default_env_name("Matilda")
@check("env.installed_envs")
def _(): return f"{len(mtb.env.installed_envs())} installed"
@check("env.lockfile")
def _(): return str(mtb.env.lockfile(mtb.env.group_for("Matilda")))
@check("env.required_envs")
def _(): return f"{len(mtb.env.required_envs())} required"
@check("env.recipe")
def _(): return list(mtb.env.recipe("Matilda"))[:4]
@check("env.plan")
def _(): return f"{len(mtb.env.plan())} entries"
@check("env.status")
def _(): return f"{len(mtb.env.status())} entries"
@check("env.doctor")
def _():
    d = mtb.env.doctor(); miss = [x for x in d if not x.get("installed", True)]
    return f"{len(d)} envs, {len(miss)} missing"
@check("env.environment_yml")
def _(): return f"{len(mtb.env.environment_yml('Matilda'))} chars"
@check("env.create_commands")
def _(): return f"{len(mtb.env.create_commands('Matilda'))} cmds"
@check("env.group_create_commands")
def _(): return f"{len(mtb.env.group_create_commands(mtb.env.group_for('Matilda')))} cmds"
@check("env.create(dry_run)")
def _(): return str(mtb.env.create("Matilda", dry_run=True))[:60]
@check("env.create_env(dry_run)")
def _(): return str(mtb.env.create_env(mtb.env.group_for("Matilda"), dry_run=True))[:60]

@check("env.create_env(unknown env -> clear error)")
def _():
    try:
        mtb.env.create_env("scmb_definitely_not_a_real_env", dry_run=True)
    except FileNotFoundError as e:
        assert "no lockfile" in str(e); return "raises FileNotFoundError as designed"
    raise AssertionError("expected FileNotFoundError")
@check("env.create_group(dry_run)")
def _(): return str(mtb.env.create_group(mtb.env.group_for("Matilda"), dry_run=True))[:60]
@check("env.create_all(dry_run)")
def _(): return f"{len(mtb.env.create_all(methods=['Matilda'], dry_run=True))} planned"
@check("env.freeze(temp)")
def _():
    p = mtb.env.freeze(mtb.env.group_for("Matilda"), out_dir=OUT); return os.path.basename(str(p))

# ---------------- results ----------------
@check("load_results")
def _():
    df = mtb.load_results("diagonal"); return f"{df.shape}"

# ---------------- eval ----------------
def _emb_and_labels():
    import h5py
    with h5py.File("/tmp/d11/Matilda_D11/embedding.h5") as h:
        e = np.array(h["data"])
    if e.shape[0] < e.shape[1]: e = e.T
    d = pd.read_csv("/media/disk2/Sichang/scmbench_pkg/data/D11/cty.csv")
    lab = (d["x"] if "x" in d.columns else d.iloc[:, -1]).to_numpy()
    return e, lab

@check("evaluate(ndarray)")
def _():
    e, lab = _emb_and_labels()
    v = mtb.evaluate(e, category="vertical", task="clustering", labels=lab)
    return {k: round(float(x), 3) for k, x in v["Value"].items() if pd.notna(x)}

@check("evaluate(dims x cells auto-orient)")
def _():
    e, lab = _emb_and_labels()
    v = mtb.evaluate(e.T, category="vertical", task="clustering", labels=lab)
    return f"ARI={float(v['Value']['ARI']):.3f}"

@check("evaluate(precomputed clustering)")
def _():
    e, lab = _emb_and_labels()
    cl = pd.Series(lab).astype("category").cat.codes.to_numpy()
    v = mtb.evaluate(e, category="vertical", task="clustering", labels=lab, clustering=cl)
    return f"ARI={float(v['Value']['ARI']):.3f}"

@check("eval.evaluate / eval.to_long (module aliases)")
def _():
    e, lab = _emb_and_labels()
    v = mtb.eval.evaluate(e, category="vertical", task="clustering", labels=lab)
    return f"{mtb.eval.to_long(v, method='M', dataset='D11', category='vertical').shape}"

@check("to_long")
def _():
    e, lab = _emb_and_labels()
    v = mtb.evaluate(e, category="vertical", task="clustering", labels=lab)
    return f"{mtb.to_long(v, method='Matilda', dataset='D11', category='vertical').shape}"

# ---------------- plot ----------------
@check("plot.build_table")
def _():
    lng = pd.read_csv(f"{RESULTS}/long_all_D11.csv")
    t = mtb.plot.build_table(lng); return type(t).__name__

@check("plot.BubbleTable")
def _():
    lng = pd.read_csv(f"{RESULTS}/long_all_D11.csv")
    t = mtb.plot.build_table(lng); assert isinstance(t, mtb.plot.BubbleTable); return "instance ok"

@check("plot.render")
def _():
    lng = pd.read_csv(f"{RESULTS}/long_all_D11.csv")
    fig = mtb.plot.render(mtb.plot.build_table(lng), title="render")
    p = f"{OUT}/render.png"; fig.savefig(p); return os.path.getsize(p)

@check("plot.plot_bubble(save=)")
def _():
    lng = pd.read_csv(f"{RESULTS}/long_all_D11.csv")
    p = f"{OUT}/plot_bubble.png"
    mtb.plot.plot_bubble(lng, title="plot_bubble", save=p); return os.path.getsize(p)

@check("plot.bubble (namespace call)")
def _():
    lng = pd.read_csv(f"{RESULTS}/long_all_D11.csv")
    fig = mtb.plot.bubble(lng); p = f"{OUT}/bubble.png"; fig.savefig(p); return os.path.getsize(p)

@check("plot.bubble(metrics/methods/order/aggregate)")
def _():
    lng = pd.read_csv(f"{RESULTS}/long_all_D11.csv")
    fig = mtb.plot.bubble(lng, metrics=["ARI", "NMI"], methods=["Matilda", "totalVI", "scMSI"],
                          order=["totalVI", "Matilda", "scMSI"], aggregate="dataset")
    p = f"{OUT}/bubble_filtered.png"; fig.savefig(p); return os.path.getsize(p)

@check("plot.style module")
def _():
    return [n for n in dir(mtb.plot.style) if not n.startswith("_")][:6]

# ---------------- COHERENT END-TO-END PLOT (all 4 scenarios) ----------------
@check("COHERENT: 4-scenario combined figure")
def _():
    frames = []
    for ds in ["D11", "D28", "D45", "D52"]:
        f = f"{RESULTS}/long_all_{ds}.csv"
        if os.path.exists(f): frames.append(pd.read_csv(f))
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(f"{OUT}/long_all_scenarios.csv", index=False)
    fig = mtb.plot.bubble(allf, title="multibench - all four integration scenarios")
    p = f"{OUT}/combined_all_scenarios.png"; fig.savefig(p, dpi=140, bbox_inches="tight")
    ok = open(p, "rb").read(4) == b"\x89PNG"
    return f"{allf.shape} methods={allf['method'].nunique()} png={os.path.getsize(p)} magic={ok}"

@check("COHERENT: per-scenario figures")
def _():
    made = []
    for ds, cat in [("D11","vertical"), ("D28","diagonal"), ("D45","mosaic"), ("D52","cross")]:
        f = f"{RESULTS}/long_all_{ds}.csv"
        if not os.path.exists(f): continue
        fig = mtb.plot.bubble(pd.read_csv(f), title=f"{cat} ({ds})")
        p = f"{OUT}/scenario_{cat}.png"; fig.savefig(p, dpi=130, bbox_inches="tight")
        made.append(f"{cat}:{os.path.getsize(p)}")
    return made

# ---------------- report ----------------
p = sum(1 for _, s, _ in checks if s == "PASS")
f = sum(1 for _, s, _ in checks if s == "FAIL")
print("\n" + "=" * 96)
for n, s, d in checks:
    print(f"  [{s}] {n:44s} {d}")
print("=" * 96)
print(f"TOTAL {len(checks)} checks : {p} PASS / {f} FAIL")
json.dump([{"check": n, "status": s, "detail": d} for n, s, d in checks],
          open(f"{OUT}/api_verify.json", "w"), indent=1)
