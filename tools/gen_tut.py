"""Generate one tutorial notebook per integration scenario."""
import nbformat as nbf, os

OUT = "notebooks"
os.makedirs(OUT, exist_ok=True)

SCEN = {
 "vertical": dict(
   ds="D11", n=14, mods=["rna","adt"], cells="2,864",
   blurb=("**Vertical integration** takes several modalities measured **in the same cell** "
          "(here CITE-seq: RNA + surface protein) and learns one joint representation. "
          "Cells are already matched, so the task is fusing modalities, not aligning cells."),
   live=("Matilda", '{"epochs": 5}', "Matilda takes well under a minute at 5 epochs"),
 ),
 "diagonal": dict(
   ds="D28", n=14, mods=["rna","atac_gas"], cells="6,408 RNA + 4,606 ATAC",
   blurb=("**Diagonal integration** is the hard case: RNA and ATAC come from **different cells**, "
          "with no pairing and no shared cell ids. A method must align two populations using only "
          "shared feature space (e.g. gene-activity scores)."),
   live=("online_iNMF", "None", "online_iNMF finishes in ~80 s"),
 ),
 "mosaic": dict(
   ds="D45", n=4, mods=["rna1","rna2","atac2","atac3"], cells="32,151",
   blurb=("**Mosaic integration** has several batches where only **some** share a modality — "
          "here an RNA-only batch, an ATAC-only batch and a paired RNA+ATAC batch that bridges them. "
          "The paired batch is the bridge that makes the others comparable."),
   live=(None, None, "every mosaic method needs >30 min on 32k cells, so this tutorial uses "
                     "the stored results instead of running one live"),
 ),
 "cross": dict(
   ds="D52", n=8, mods=["rna1","rna2","rna3","adt1","adt2","adt3"], cells="23,478",
   blurb=("**Cross integration** has several batches in which **all** modalities are present. "
          "The task is removing batch effects while keeping biological structure. "
          "(Spatial registration also lives under `cross` — see the note at the end.)"),
   live=("StabMap", "None", "StabMap finishes in ~70 s"),
 ),
}

for cat, s in SCEN.items():
    C = []
    md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
    code = lambda t: C.append(nbf.v4.new_code_cell(t))

    md(f"""# {cat.capitalize()} integration — a complete walkthrough

{s['blurb']}

**Reference dataset:** `{s['ds']}` ({s['cells']} cells) — **{s['n']} methods** apply to it.
Every one has a row in the results table below: scored, failed, or explicitly marked
as verified on another dataset. None disappears silently.

Everything below uses the same three calls, whatever the scenario:

```python
mtb.scan(dataset)                    # what can I run?
res = mtb.run_all(dataset, category) # run it, with metrics
res.plot()                           # one figure
```""")

    code("""import warnings; warnings.filterwarnings("ignore")
%matplotlib inline
from pathlib import Path
import pandas as pd
pd.set_option("display.max_colwidth", None)  # show `reason` in full - paths are long
pd.set_option("display.max_columns", None)   # a metric pandas hides is a metric
pd.set_option("display.width", 200)          # the reader never learns exists
import multibench as mtb

RESULTS = Path("results")          # stored results, so the comparisons reproduce
print("multibench", mtb.__version__)""")

    fastm = s["live"][0] or {"mosaic": "SMILE"}.get(cat, "Matilda")
    code(f'''DATASET  = "{s['ds']}"
CATEGORY = "{cat}"
FAST_METHOD = "{fastm}"        # used for the quick live demos below''')

    md("""## 0. Which scenario is my data, and how do I lay it out?

Two calls answer what every new user hits first. `category` is a required argument
everywhere, so start by seeing the legal values:""")
    code("""mtb.list_categories()""")
    md("""And this prints exactly which filenames to use \u2014 including how **several
batches** are named (numbered files in one flat directory):""")
    code("""print(mtb.describe_layout(CATEGORY))""")

    md("""## 1. What can I run on this data?

`scan` inspects the dataset and reports every method that can run — and, for the
rest, exactly why not. Nothing is executed, so this is safe and instant.""")
    code("""avail = mtb.scan(DATASET, category=CATEGORY)
avail[avail.runnable][["method", "modalities", "env", "output_kind",
                       "n_tunable", "runtime_tier", "observed_worst_sec"]]""")

    md("""Methods that are *not* runnable here come with a reason rather than a silent absence:""")
    code("""not_ok = avail[~avail.runnable][["method", "modalities", "reason"]]
not_ok.head(5) if len(not_ok) else "(everything in this category runs on this dataset)"
""")

    md("""## 2. What can I tune?

`params_for` reports the parameters a method accepts. `defaults` are what the
wrapper passes; `tunable` is what the **upstream script** exposes on its command
line. An empty `tunable` means that method hardcodes its hyperparameters — it
cannot be tuned without editing it, which this project never does.""")
    code("""rows = []
for m in avail[avail.runnable]["method"]:
    p = mtb.params_for(m, CATEGORY, None if avail[avail.method==m].iloc[0]["modalities"]=="(data_dir)"
                       else avail[avail.method==m].iloc[0]["modalities"].split("+"))
    rows.append({"method": m, "n_tunable": len(p["tunable"]),
                 "defaults": p["defaults"],
                 "example": ", ".join(sorted(p["tunable"])[:4]) or "(hardcoded upstream)"})
pd.DataFrame(rows).sort_values("n_tunable", ascending=False).reset_index(drop=True)""")

    live_m, live_p, live_why = s["live"]
    if live_m:
        md(f"""## 3. Run one method

{live_why}. Tuning happens through `params=` — the same parameters `params_for`
just listed.""")
        pstr = (f', params={{"{live_m}": {live_p}}}' if live_p != "None" else "")
        code(f'''res = mtb.run_all(DATASET, CATEGORY,
                  methods=["{live_m}"]{pstr},
                  out_dir="/tmp/tutorial_{cat}")
res''')
        code("""res.summary""")
        md("""The metrics are already computed — `run_all` runs the method, picks the correct
label order, and evaluates in one step.""")
        code("""res.plot()""")
    else:
        md(f"""## 3. Running a method

{live_why}. The call is identical to every other scenario:

```python
res = mtb.run_all(DATASET, CATEGORY, methods=["Cobolt"],
                  params={{"Cobolt": {{"lr": 1e-3}}}},   # Cobolt DIVERGES at its default lr
                  out_dir="/tmp/tutorial_mosaic")
res.plot()
```

`dry_run=True` shows the plan without running anything:""")
        code("""mtb.run_all(DATASET, CATEGORY, out_dir="/tmp/unused", dry_run=True)[
    ["method", "modalities", "env", "output_kind"]]""")

    md(f"""## 4. Run everything

One call runs every applicable method and evaluates each:

```python
res = mtb.run_all(DATASET, CATEGORY, out_dir="/tmp/{cat}_all",
                  timeout=4*3600,        # a hung method is recorded, not fatal
                  skip_existing=True)    # resume instead of repeating hours
print(res.failures)                      # ALWAYS check: failures are recorded, not raised
res.plot()
```

Next morning, re-plot **without re-running anything** (`run_all` saved it for you):

```python
res = mtb.load_batch("/tmp/{cat}_all")
res.plot().savefig("compare.png")
```

> WARNING `skip_existing` reuses a method's existing output FILE. Do not combine it
> with a changed `params=` (you would silently get the old result, so `run_all`
> refuses it), and after a hard kill delete that method's directory, since a
> half-written file would be reused as if it had succeeded.

That takes from minutes to hours depending on the scenario, so here we load the
stored results of exactly that sweep.""")
    code(f'''summary = pd.read_csv(RESULTS / "summary_{s['ds']}.csv")
# clustering metrics, then batch-correction metrics where the design has batches
cols = [c for c in ["method","status","run_sec",
                    "ARI","NMI","ASW","iASW","iF1","cLISI",     # clustering
                    "ASW_batch","GC","iLISI"                    # batch correction
                    ] if c in summary.columns]
summary[cols]''')

    md("""### What a failure looks like

`.failures` is the table to check after every sweep, because a method that dies is
**recorded, not raised** — the sweep keeps going. Here is a genuine failure: a
method given an impossible hyperparameter. Note the sweep still returns normally
and the error is in the table.""")
    code("""broken = mtb.run_all(DATASET, CATEGORY, methods=[FAST_METHOD],
                     params={FAST_METHOD: {"epochs": -5}},   # nonsense on purpose
                     out_dir="/tmp/tut_fail", verbose=False)
print("failures:", len(broken.failures))
broken.failures""")

    md("""By contrast, if **nothing at all** can run — a mistyped dataset name, say —
that is not a per-method failure, it means the request itself was wrong, so
`run_all` raises instead of handing back an empty result that reads as success:""")
    code("""try:
    mtb.run_all("NO_SUCH_DATASET", CATEGORY, out_dir="/tmp/tut_none")
except ValueError as e:
    print("ValueError:", str(e)[:160])""")

    md("""### Reloading a finished sweep

`run_all` saves itself, so the morning after an overnight run you can reopen the
results and re-plot **without recomputing anything**:""")
    code("""reloaded = mtb.load_batch("/tmp/tut_fail")      # the sweep we just ran
print(type(reloaded).__name__, "|", len(reloaded), "method(s)")
reloaded.summary[["method", "status"]]""")

    md("""> **Reading the metric columns.** This benchmark reports two families:
>
> * **Clustering / dimension reduction** — `ARI`, `NMI`, `ASW`. Always computed.
> * **Batch correction** — `iASW`, `iF1`, `ASW_batch`, `GC`. Computed automatically
>   whenever the dataset has more than one batch; `run_all` derives the batch from
>   which label file each cell came from.
>
> A blank cell has a specific meaning, and it is worth knowing which:
>
> * On a **single-batch** dataset (one `cty.csv`, i.e. vertical integration) the
>   batch family does not exist at all — there is nothing to correct.
> * `iASW` / `iF1` measure **isolated labels** — cell types confined to a subset of
>   batches. If every cell type appears in every batch there are none, so both are
>   blank even though the dataset is multi-batch. That is the case on D45.
> * `cLISI` / `iLISI` need a compiled scIB extension that will not load against this
>   machine's GLIBC, and `kBET` needs `rpy2`, which is not installed in the
>   evaluation environment. Those three are environmental, not properties of your
>   data.""")

    md("""## 5. The figure""")
    code(f'''long = pd.read_csv(RESULTS / "long_all_{s['ds']}.csv")
fig = mtb.plot.bubble(long, title="{cat} integration — {s['ds']}")
fig.set_dpi(110)
display(fig)''')

    md("""### Summarising across datasets

The bubble chart above compares methods **on this dataset**. To compare them
**across** datasets, concatenate the tidy frames and use `plot.bar`, which is the
summary the benchmark reports:

```python
allf = pd.concat([pd.read_csv(p) for p in RESULTS.glob("long_all_*.csv")])
mtb.plot.bar(allf, title="overall across scenarios")        # every metric
mtb.plot.bar(allf, group="clustering")                      # one family
mtb.plot.bar(allf, group="batch")                           # the other
```

Each bar is a method's mean score over datasets; the dots behind it are its
per-dataset scores, so a method that is uniformly good is distinguishable from one
that averages well by winning a single dataset.""")

    md("""> The bubble chart encodes **radius = rank** — **rank 1 is the LARGEST bubble** —
> and **fill = value normalised within the plotted set**, darker being higher.
> Both are relative to what you plotted. Read the table beside it;
> with few methods a tiny gap can look decisive.""")

    md(f"""## 6. Using your OWN dataset

`describe_layout` above already printed the exact filenames for this scenario —
**use those, not the role names.** They are not always the same: the `atac_gas`
role lives in `atac.h5`, and `atac_peak` lives in `peak.h5`.

```python
print(mtb.describe_layout(CATEGORY))   # the authoritative filename list
```

Then the same three calls work unchanged:

```python
mtb.scan("MYDATA", category="{cat}")                     # confirm it is picked up
res = mtb.run_all("MYDATA", "{cat}", out_dir="out/")     # run everything
res.plot()
```

If `scan` says a method is not runnable, the `reason` column names the missing
file — fix that and re-scan. Modality files are HDF5 with the matrix under
`matrix/data`; `mtb.io.to_canonical` converts other layouts.""")

    if cat == "cross":
        md("""### Note — spatial registration also lives under `cross`

`PASTE`, `PASTE2`, `SPIRAL` and `GPSA` align spatial slices. They take a
**directory of `.h5ad` slices** and return **aligned coordinates**, not an
embedding, so scIB clustering metrics do not apply to them; they are scored with
spatial measures (SCS / PAA) instead.

`scan` separates them automatically — on a non-spatial dataset they are reported
as not runnable, with the reason: """)
        code("""mtb.scan("D52", category="cross")[["method", "runnable", "reason"]].tail(4)""")
        code("""mtb.scan("D63", category="cross")[["method", "runnable", "output_kind"]].head(4)""")

    nb = nbf.v4.new_notebook()
    nb["cells"] = C
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    path = f"{OUT}/tutorial_{cat}.ipynb"
    nbf.write(nb, path)
    print("wrote", path, len(C), "cells")
