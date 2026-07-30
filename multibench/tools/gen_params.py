"""Generate engine/params.yaml: doc-only tunable hyperparams per method variant,
extracted from each UPSTREAM script's argparse (python) / commandArgs (R).
Never emitted on a command line - surfaced by multibench.params_for()."""
import warnings, os, re, json; warnings.filterwarnings("ignore")
from multibench.engine import registry

ROOTS = ["/media/disk2/Sichang/scmbench_pkg/scMultiBench_ref",
         "/media/disk2/Sichang/scmbench_tools"]
# argument names that are I/O plumbing, not tunable science
PATHY = set("""path1 path2 path3 path4 path5 save_path cty_path out output outdir save_dir
data_dir data_path source_data target_data source_cty target_cty train_path1 train_path2
test_path1 test_path2 train_cty_path save ref_path1 ref_path2 query_path1 query_path2
pair_path1 pair_path2 dataset_path ae_weights ae_weight_file embedding_file prediction_file
pre_trained rna adt atac cty file_path input output_path result_path""".split())

def resolve(ep):
    if os.path.isabs(ep) and os.path.exists(ep): return ep
    for r in ROOTS:
        c = os.path.join(r, ep)
        if os.path.exists(c): return c
    return None

def vkey(v):
    return "%s:%s" % (v.when.get("category"), "+".join(v.when.get("modalities", [])) or "-")

def lit(x):
    """Best-effort literal -> python value for YAML emission."""
    if x is None: return None
    x = x.strip().rstrip(",")
    try:
        import ast; return ast.literal_eval(x)
    except Exception:
        return x

out = {}
for s in registry.load():
    for v in s.variants:
        path = resolve(v.entrypoint)
        if not path: continue
        try: txt = open(path, errors="ignore").read()
        except Exception: continue
        tun = {}
        for m in re.finditer(r"add_argument\(\s*[\'\"]--([A-Za-z0-9_\-]+)[\'\"](.*?)\)", txt, re.S):
            name = m.group(1)
            if name in PATHY: continue
            body = m.group(2)[:250].replace("\n", " ")
            d = re.search(r"default\s*=\s*([^,\)]+)", body)
            t = re.search(r"type\s*=\s*([A-Za-z]+)", body)
            act = "store_true" in body
            rec = {"default": lit(d.group(1)) if d else None}
            if t: rec["type"] = t.group(1)
            elif act: rec["type"] = "flag"
            tun[name] = rec
        if tun:
            out.setdefault(s.id, {})[vkey(v)] = tun

hdr = ("# AUTO-GENERATED - do not hand-edit; regenerate with tools/gen_params.py\n"
       "#\n"
       "# DOC-ONLY tunable hyperparameters, extracted from each UPSTREAM method\n"
       "# script's own argparse. These are NEVER added to a command line by the\n"
       "# package; they document what `run(..., params={...})` may override, and\n"
       "# are surfaced by `multibench.params_for(method, category, modalities)`.\n"
       "#\n"
       "# A method absent from this file exposes NO command-line hyperparameters\n"
       "# (they are hardcoded in its source). Since method scripts are never\n"
       "# modified, such a method is not tunable through the wrapper.\n")
import yaml
with open("engine/params.yaml", "w") as f:
    f.write(hdr)
    yaml.safe_dump(out, f, sort_keys=True, default_flow_style=False, width=100)
nv = sum(len(x) for x in out.values())
print("methods with tunable params: %d ; variants: %d" % (len(out), nv))
for m in sorted(out):
    for k, t in out[m].items():
        print("   %-12s %-28s %s" % (m, k, ", ".join(sorted(t))[:70]))
