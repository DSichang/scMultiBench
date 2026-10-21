"""Smoke checks of the multibench public API on the benchmark host.

Public entry points are called with real inputs (the paths below); env
installation runs as a dry run. Ends with the four-scenario plotting chain.
Prints a PASS/FAIL table and writes ``api_verify.json``.
"""
import warnings, json, os, traceback
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
@check("find_methods")
def _():
    a = mtb.find_methods(); b = mtb.find_methods(category="vertical")
    assert 0 < len(b) < len(a); return f"{len(a)} all / {len(b)} vertical"

@check("find_methods(category+modalities)")
def _():
    m = mtb.find_methods(category="vertical", modalities=["rna", "adt"]); assert len(m) >= 14; return f"{len(m)} methods"

@check("find_methods(needs_labels/atac)")
def _():
    b = mtb.find_methods(needs_labels=True)
    c = mtb.find_methods(modalities=["atac_gas"]); return f"labels={len(b)} atac={len(c)}"

@check("method_info")
def _():
    i = mtb.method_info("Matilda"); assert i["env"] and "params" not in i; return f"env={i['env']} variants={len(i['supports'])}"

@check("method_info()['runtime']")
def _():
    i = mtb.method_info("Matilda")
    return {k: i["runtime"].get(k) for k in ("tier", "worst_sec")}

@check("params_for")
def _():
    p = mtb.params_for("scMDC", "vertical", ["rna", "adt"]); assert p["tunable"]; return f"{len(p['tunable'])} tunable"

@check("params_for(single-variant)")
def _():
    from multibench.engine import registry
    s = next(x.id for x in registry.load() if len(x.variants) == 1)
    return mtb.params_for(s)["variant"]

# ---------------- data resolution ----------------
@check("scan: the plan, with each command")
def _():
    t = mtb.scan("D11", "vertical", methods=["Matilda"], verbose=False)
    plan = mtb.scan("D52", "cross", methods=["totalVI"], verbose=False)
    return f"{list(t.columns)}; {plan.command.iloc[0][:40]}"

# ---------------- catalog / config ----------------
@check("catalog.datasets")
def _(): return f"{mtb.catalog.datasets().shape}"
@check("catalog.metrics")
def _(): return f"{mtb.catalog.metrics().shape}"
@check("config (the four settings)")
def _():
    assert mtb.config.FIELDS == ("data_path", "repo_path", "envs_dir", "leiden_flavor")
    return repr(mtb.config)
@check("config.data_path")
def _(): return str(mtb.config.data_path)

# ---------------- io ----------------
@check("io.export_dataset + scan on the folder")
def _():
    import anndata as ad
    a = ad.AnnData(np.random.rand(40, 10)); a.obsm["protein"] = np.random.rand(40, 4)
    a.obs["ct"] = ["A", "B"] * 20
    d = mtb.io.export_dataset(a, os.path.join(OUT, "data", "MYCITE"), rna="X",
                              adt="obsm:protein", labels="obs:ct")
    files = sorted(os.listdir(d)); assert files == ["adt.h5", "cty.csv", "rna.h5"], files
    got = mtb.scan(str(d), "vertical", methods=["Matilda"], verbose=False)
    return f"{files} -> {got.modalities.iloc[0]}"

# ---------------- env ----------------
@check("env.__all__ / dir")
def _():
    assert mtb.env.__all__ == ["install"], mtb.env.__all__
    assert not hasattr(mtb.env, "group_for")
    return "install"
@check("env.install(dry_run=True) - the plan, nothing built")
def _():
    rows = mtb.env.install(["Matilda"], dry_run=True)
    assert len(rows) and rows.env.iloc[0] == mtb.method_info("Matilda")["env"], rows
    return f"{rows.env.iloc[0]}: {rows.state.iloc[0]}"
@check("env.install(dry_run=True, build=True)")
def _():
    rows = mtb.env.install(["Matilda"], build=True, dry_run=True)
    return f"{rows.env.iloc[0]}: {rows.state.iloc[0]}"

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
    v = mtb.evaluate(e, category="vertical", metrics="clustering", labels=lab)
    return {r.metric: round(float(r.value), 3) for r in v.itertuples() if pd.notna(r.value)}

@check("evaluate(dims x cells auto-orient)")
def _():
    e, lab = _emb_and_labels()
    v = mtb.evaluate(e.T, category="vertical", metrics="clustering", labels=lab)
    return f"ARI={float(v.set_index('metric').value['ARI']):.3f}"

@check("evaluate(metrics=[...]) - list of codes, no Leiden clustering")
def _():
    e, lab = _emb_and_labels()
    v = mtb.evaluate(e, category="vertical", metrics=["cLISI"], labels=lab)
    assert list(v.metric) == ["cLISI"], list(v.metric)
    return f"cLISI={float(v.value.iloc[0]):.3f}"

@check("evaluate(dataset=, method=) - the long table")
def _():
    e, _ = _emb_and_labels()
    v = mtb.evaluate(e, dataset="D11", category="vertical", method="Matilda", metrics="ARI")
    assert list(v.columns)[:3] == ["metric", "value", "method"], list(v.columns)
    return f"{v.shape}"

# ---------------- plot ----------------
@check("recommend(long_df=)")
def _():
    lng = pd.read_csv(f"{RESULTS}/long_all_D11.csv")
    return list(mtb.recommend("vertical", long_df=lng).method[:3])

@check("plot.__all__ / dir (render, FamilyBlock importable but unlisted)")
def _():
    assert mtb.plot.__all__ == ["bubble", "bar"], mtb.plot.__all__
    assert "render" not in dir(mtb.plot) and callable(mtb.plot.render)
    return "two names"

@check("plot.bubble(save=)")
def _():
    lng = pd.read_csv(f"{RESULTS}/long_all_D11.csv")
    p = f"{OUT}/bubble_saved.png"
    mtb.plot.bubble(lng, title="bubble", save=p); return os.path.getsize(p)

@check("plot.bubble (namespace call)")
def _():
    lng = pd.read_csv(f"{RESULTS}/long_all_D11.csv")
    fig = mtb.plot.bubble(lng); p = f"{OUT}/bubble.png"; fig.savefig(p); return os.path.getsize(p)

@check("plot.bubble(metrics/methods)")
def _():
    lng = pd.read_csv(f"{RESULTS}/long_all_D11.csv")
    fig = mtb.plot.bubble(lng, metrics=["ARI", "NMI"], methods=["Matilda", "totalVI", "scMSI"])
    p = f"{OUT}/bubble_filtered.png"; fig.savefig(p); return os.path.getsize(p)

@check("plot.style module")
def _():
    return [n for n in dir(mtb.plot.style) if not n.startswith("_")][:6]

# ---------------- end-to-end plot (all four scenarios) ----------------
@check("end-to-end: four-scenario combined figure")
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

@check("end-to-end: per-scenario figures")
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
