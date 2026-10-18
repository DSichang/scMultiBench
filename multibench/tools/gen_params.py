"""Generate engine/params.yaml: doc-only tunable hyperparams per method variant,
extracted from the ``add_argument`` calls of each upstream script. The recorded
defaults are never put on a command line; multibench.params_for() surfaces them.

An argument the package or the script itself sets is not tunable, and is left
out (setting it would duplicate the flag or have no effect):

- the flags of the variant's ``args`` in methods.yaml (its input files and
  output folder);
- path arguments: a ``metavar`` naming a directory or path, a name ending
  in ``_path``, or a name in :data:`PATHY`;
- arguments the script overwrites after parsing (``args.input_size = ...``:
  set from the data);
- a ``store_true`` switch whose default is already ``True`` (it cannot be
  turned off);
- arguments no Python file of the script's folder reads (``args.<name>``);
- the arguments in :data:`SET_FROM_DATA`, which the package sets per run, and
  in :data:`NOT_TUNABLE`.

A switch is recorded as ``type: flag`` with the value the script gets
without it (``False`` for ``store_true``, ``True`` for ``store_false``); an
argument with ``choices=`` records them.

A variant whose script is not found under :data:`ROOTS` keeps its entry of the
current params.yaml, filtered by the first rule.

Run: ``python -m multibench.tools.gen_params``.
"""
import os
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from multibench.engine import registry  # noqa: E402

_PKG = Path(__file__).resolve().parents[1]
_REPO = _PKG.parent
OUT = _PKG / "engine" / "params.yaml"
# the package's own script copy first, then the article's, then the benchmark host's
ROOTS = [str(_REPO), str(_REPO / "scMultiBench_ref"),
         "/media/disk2/Sichang/scmbench_pkg/scMultiBench_ref",
         "/media/disk2/Sichang/scmbench_tools"]
# argument names that are I/O plumbing, not tunable science
PATHY = set("""path1 path2 path3 path4 path5 save_path cty_path out output outdir save_dir
data_dir data_path source_data target_data source_cty target_cty train_path1 train_path2
test_path1 test_path2 train_cty_path save ref_path1 ref_path2 query_path1 query_path2
pair_path1 pair_path2 dataset_path ae_weights ae_weight_file embedding_file prediction_file
pre_trained rna adt atac cty file_path input output_path result_path data_file save_name
experiment load_model""".split())
#: arguments the package sets per run from the data (methods.yaml ``params``)
SET_FROM_DATA = {
    "scMDC": {"nbatch"},        # one per input file
    "scMM": {"model"},          # rna_protein or rna_atac, from the modalities
    "scBridge": {"umap_plot"},  # passed: the script's figure save crashes
}
#: arguments a run does not take from ``params=``: the CUDA switch the package
#: sets per host, logging and thread settings, and what is read only by code
#: that never runs. tests/test_params_api.py checks each is a script argument.
NOT_TUNABLE = {
    "Matilda": {"device"},              # the driver passes device='auto'
    "scMDC": {"device"},                # cpu_params on a host without a GPU
    "scMM": {"no_cuda", "print_freq"},  # every variant passes no_cuda; print_freq logs
    # use_cuda: cpu_params; main_scJoint.py:108 set_num_threads(1) ignores --threads;
    # the stage-3 arguments are read only by util/trainingprocess_stage3.py, whose
    # call is commented out (main_scJoint.py:130-137)
    "scJoint": {"use_cuda", "threads", "epochs_stage3", "lr_stage3", "with_crossentorpy",
                "center_weight"},
    # main_UnitedNet.py:234 names its output embedding{index}.h5; the package reads
    # embedding1.h5
    "UnitedNet": {"index"},
}
#: a metavar that names a file or folder: the argument is an input, not a knob
_PATH_METAVAR = re.compile(r"metavar\s*=\s*['\"](DIR|DIRS|PATH|PATHS|FILE|FILES)['\"]")


def resolve(ep):
    for r in ROOTS:
        c = os.path.join(r, ep)
        if os.path.exists(c):
            return c
    return None


def vkey(v):
    return "%s:%s" % (v.when.get("category"), "+".join(v.when.get("modalities", [])) or "-")


def lit(x):
    """Best-effort literal -> python value for YAML emission."""
    if x is None:
        return None
    x = x.strip().rstrip(",")
    try:
        import ast
        return ast.literal_eval(x)
    except Exception:
        return x


def set_by_package(spec, v) -> set:
    """The argument names the package passes for variant ``v``: its ``args``
    flags. (A ``cpu_params`` key stays: the registry requires the script to
    declare it, and on a GPU host it is the user's to set.)"""
    return {str(a.flag).lstrip("-").split("=")[0] for a in v.args if a.flag}


def set_by_script(txt: str) -> set:
    """Arguments the script assigns after parsing (``args.name = ...``)."""
    return set(re.findall(r"\bargs\.([A-Za-z0-9_]+)\s*=(?!=)", txt))


def folder_text(path) -> str:
    """Every Python file of the script's folder, where the parsed arguments are read."""
    out = []
    for f in sorted(Path(path).parent.rglob("*.py")):
        try:
            out.append(f.read_text(errors="ignore"))
        except OSError:
            pass
    return "\n".join(out)


def is_read(name: str, text: str) -> bool:
    """Whether ``text`` reads the parsed argument ``name`` (``args.name``, or all
    of them through ``vars(args)``)."""
    return bool(re.search(r"\bvars\(\s*\w+\s*\)|\.__dict__\b", text)
                or re.search(r"\w\.%s\b" % re.escape(name), text))


def extract(txt: str, skip: set) -> dict:
    tun = {}
    for m in re.finditer(r"add_argument\(\s*[\'\"]--([A-Za-z0-9_\-]+)[\'\"](.*?)\)", txt, re.S):
        name = m.group(1)
        body = m.group(2)[:250].replace("\n", " ")
        if (name in PATHY or name.endswith("_path") or name in skip
                or _PATH_METAVAR.search(body)):
            continue
        d = re.search(r"default\s*=\s*([^,\)]+)", body)
        t = re.search(r"type\s*=\s*([A-Za-z]+)", body)
        c = re.search(r"choices\s*=\s*(\[[^\]]*\])", body)
        rec = {"default": lit(d.group(1)) if d else None}
        if t:
            rec["type"] = t.group(1)
        if c:
            rec["choices"] = lit(c.group(1))    # argparse refuses any other value
        elif "store_true" in body or "store_false" in body:
            off = "store_true" in body
            if rec["default"] is None:
                rec["default"] = not off
            if rec["default"] is off:      # the switch cannot change the value
                continue
            rec["type"] = "flag"
        tun[name] = rec
    return tun


def main():
    import yaml
    current = yaml.safe_load(OUT.read_text()) if OUT.exists() else {}
    out = {}
    for s in registry.load():
        for v in s.variants:
            skip = (set_by_package(s, v) | SET_FROM_DATA.get(s.id, set())
                    | NOT_TUNABLE.get(s.id, set()))
            path = resolve(v.entrypoint)
            if path:
                try:
                    txt = open(path, errors="ignore").read()
                except Exception:
                    continue
                every = folder_text(path) if path.endswith(".py") else txt
                tun = {k: t for k, t in extract(txt, skip | set_by_script(txt)).items()
                       if not path.endswith(".py") or is_read(k, every)}
            else:
                tun = {k: t for k, t in (current.get(s.id, {}).get(vkey(v)) or {}).items()
                       if k not in skip}
            if tun:
                out.setdefault(s.id, {})[vkey(v)] = tun

    hdr = ("# AUTO-GENERATED - do not hand-edit; regenerate with multibench/tools/gen_params.py\n"
           "#\n"
           "# Doc-only tunable hyperparameters, extracted from each upstream method\n"
           "# script's own argparse. These are never added to a command line by the\n"
           "# package; they document what `run(..., params={...})` may override, and\n"
           "# are surfaced by `multibench.params_for(method, category, modalities)`.\n"
           "# Arguments the package or the script sets (input files, values taken\n"
           "# from the data) are left out.\n"
           "#\n"
           "# A method absent from this file exposes no command-line hyperparameters\n"
           "# (they are hardcoded in its source). Since method scripts are never\n"
           "# modified, such a method is not tunable through the wrapper.\n")
    with open(OUT, "w") as f:
        f.write(hdr)
        yaml.safe_dump(out, f, sort_keys=True, default_flow_style=False, width=100)
    nv = sum(len(x) for x in out.values())
    print("methods with tunable params: %d ; variants: %d" % (len(out), nv))


if __name__ == "__main__":
    main()
