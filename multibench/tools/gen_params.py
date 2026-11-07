"""Generate engine/params.yaml: the hyperparameters each method variant takes,
read from the package's method scripts (``tools_scripts/`` of this repository).

    python -m multibench.tools.gen_params            # rewrite engine/params.yaml
    python -m multibench.tools.gen_params --check    # exit 1 when it is stale

A Python script declares a parameter with ``parser.add_argument('--name', ...)``;
an R script, or the package's R driver (StabMap), with ``name <- option("name",
default)``, which a run passes as ``--name=value``. The recorded defaults are
never put on a command line; multibench.params_for() surfaces them. A default
of ``None`` leaves the argument to the method's library.

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
without it (``False`` for ``store_true``, ``True`` for ``store_false``); a
true/false option (``type=boolean``, R ``as.logical``) as ``boolean``; an
argument that takes several values (``nargs``) as ``list``; an argument with
``choices=`` records them.
"""
import ast
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from multibench.engine import registry  # noqa: E402

_PKG = Path(__file__).resolve().parents[1]
_REPO = _PKG.parent
OUT = _PKG / "engine" / "params.yaml"
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
_PATH_METAVARS = {"DIR", "DIRS", "PATH", "PATHS", "FILE", "FILES"}
#: ``name <- option("name", default[, as.<type>])`` in an R script
_R_OPTION = re.compile(r'^\s*[\w.]+\s*<-\s*option\("([\w.]+)",\s*(.+?)(?:,\s*(as\.\w+))?\)\s*$')
_R_TYPES = {None: "float", "as.numeric": "float", "as.character": "str",
            "as.logical": "boolean"}


def vkey(v):
    return "%s:%s" % (v.when.get("category"), "+".join(v.when.get("modalities", [])) or "-")


def _value(node):
    """The literal ``node`` holds, else its source text (``10**-3``)."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return ast.unparse(node)


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


#: ``getattr(args, k) for k in ("lr", "seed")``: the scripts pass these only when given
_GETATTR_LOOP = re.compile(r"getattr\(\s*\w+\s*,\s*(\w+)\s*\)\s+for\s+\1\s+in\s+\(([^)]*)\)")


def is_read(name: str, text: str) -> bool:
    """Whether ``text`` reads the parsed argument ``name`` (``args.name``, a
    ``getattr(args, k) for k in (...)`` loop over it, or all of them through
    ``vars(args)``)."""
    return bool(re.search(r"\bvars\(\s*\w+\s*\)|\.__dict__\b", text)
                or re.search(r"\w\.%s\b" % re.escape(name), text)
                or any(re.search(r"['\"]%s['\"]" % re.escape(name), m.group(2))
                       for m in _GETATTR_LOOP.finditer(text)))


def python_params(txt: str, skip: set) -> dict:
    """``{name: {"default", "type"[, "choices"]}}`` of the ``add_argument`` calls in ``txt``."""
    tun = {}
    for node in ast.walk(ast.parse(txt)):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).startswith("--")):
            continue
        name = node.args[0].value[2:]
        kw = {k.arg: _value(k.value) if k.arg != "type" else ast.unparse(k.value)
              for k in node.keywords if k.arg}
        if (name in PATHY or name.endswith("_path") or name in skip
                or kw.get("metavar") in _PATH_METAVARS):
            continue
        rec = {"default": kw.get("default")}
        if "nargs" in kw and kw["nargs"] != "?":
            rec["type"] = "list"
        elif "type" in kw:
            rec["type"] = kw["type"]
        if "choices" in kw:
            rec["choices"] = kw["choices"]    # argparse refuses any other value
        elif kw.get("action") in ("store_true", "store_false"):
            off = kw["action"] == "store_true"
            if rec["default"] is None:
                rec["default"] = not off
            if rec["default"] is off:      # the switch cannot change the value
                continue
            rec["type"] = "flag"
        tun[name] = rec
    return tun


def _r_value(text: str):
    """An R default (``NULL``, ``TRUE``, ``5e3``, ``1/2``, ``Inf``, ``"x"``) as a Python value."""
    words = {"NULL": None, "TRUE": True, "FALSE": False, "Inf": float("inf")}
    if text in words:
        return words[text]
    try:
        node = ast.parse(text, mode="eval").body
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return ast.literal_eval(node.left) / ast.literal_eval(node.right)
        return ast.literal_eval(node)
    except (SyntaxError, ValueError):
        return text


def r_params(txt: str, skip: set) -> dict:
    """``{name: {"default", "type"}}`` of the ``option("name", default)`` lines in ``txt``."""
    tun = {}
    for line in txt.splitlines():
        m = _R_OPTION.match(line)
        if m and m.group(1) not in skip:
            tun[m.group(1)] = {"default": _r_value(m.group(2).strip()),
                               "type": _R_TYPES[m.group(3)]}
    return tun


def variant_params(s, v) -> dict:
    """The parameters variant ``v`` of method ``s`` takes."""
    skip = set_by_package(s, v) | SET_FROM_DATA.get(s.id, set()) | NOT_TUNABLE.get(s.id, set())
    path = _REPO / v.entrypoint
    txt = path.read_text(errors="ignore")
    if path.suffix == ".py":
        every = folder_text(path)
        return {k: t for k, t in python_params(txt, skip | set_by_script(txt)).items()
                if is_read(k, every)}
    tun = r_params(txt, skip)
    if v.driver and v.driver.endswith(".R"):       # StabMap's driver reads the options
        tun.update(r_params((_PKG / v.driver).read_text(), skip))
    return tun


HEADER = """\
# AUTO-GENERATED - do not hand-edit; regenerate with
#   python -m multibench.tools.gen_params
#
# The hyperparameters each method variant takes, read from the method scripts
# (tools_scripts/) and StabMap's R driver. The package never puts these
# defaults on a command line; they document what `run(..., params={...})` may
# set, and `multibench.params_for(method, category, modalities)` shows them. A
# default of null leaves the argument to the method's library. Arguments the
# package or the script sets (input files, values taken from the data) are
# left out.
"""


def render() -> str:
    import yaml
    out = {}
    for s in registry.load():
        for v in s.variants:
            tun = variant_params(s, v)
            if tun:
                out.setdefault(s.id, {})[vkey(v)] = tun
    return HEADER + yaml.safe_dump(out, sort_keys=True, default_flow_style=False, width=100)


def main(argv=None) -> int:
    text = render()
    if "--check" in (sys.argv[1:] if argv is None else argv):
        if OUT.read_text() != text:
            print("engine/params.yaml does not match the scripts. Run "
                  "python -m multibench.tools.gen_params.", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text)
    import yaml
    table = yaml.safe_load(text)
    print(f"methods with parameters: {len(table)}; variants: "
          f"{sum(len(x) for x in table.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
