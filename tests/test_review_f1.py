"""Findings of the review of the round-1 integration (wp/f1_int).

- A scan row's ``modalities`` split on ``+`` selects that row again, in
  scan, run and the CLI, also for moETM, scMM and iPOLNG (peak input
  named atac_gas) when methods= names them; ``atac_peak`` with ``atac_gas``
  means "reads both files" (MultiMAP, Seurat_v3) instead of an error (L13
  kept 0.3.1's calls).
- inputs_for applies the representation rule of scan and find_methods when
  the tokens match a variant only loosely (L13).
- scan keeps one exact combination; find_methods keeps the variants that read
  at least the named modalities; both docstrings say so (L13).
"""
import os
import re
import warnings

import pytest

import multibench as mtb
from multibench.engine import registry
from tests.conftest import full_scan

SHIPPED = [("D11", "vertical"), ("D28", "diagonal"), ("D45", "mosaic"),
           ("D46", "mosaic"), ("D52", "cross")]
BOTH = ["rna", "atac_peak", "atac_gas"]


@pytest.fixture
def data(root):
    return root / "data"


# ------------------------------------------------------------------ L13 round trip
@pytest.mark.parametrize("ds,cat", SHIPPED)
def test_a_scan_rows_modalities_select_that_row_again(data, ds, cat):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = full_scan(ds, cat, data_path=data, verbose=False)
        rows = sc[sc["modalities"] != "(data_dir)"]
        assert len(rows)
        for m, mods in zip(rows["method"], rows["modalities"]):
            # the spelling scan shows, unchanged: named in methods=, moETM's
            # rna+atac_gas row is selected by it although moETM reads peaks
            back = full_scan(ds, cat, methods=[m], modalities=mods.split("+"),
                            data_path=data, verbose=False)
            assert ((back["method"] == m) & (back["modalities"] == mods)).any(), (m, mods)


def test_two_representations_select_the_methods_that_read_both_files(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = full_scan("D28", "diagonal", modalities=BOTH, data_path=data, verbose=False)
        plan = mtb.scan(os.path.join(data, 'D28'), 'diagonal', modalities=BOTH)
    want = {"MultiMAP", "Seurat_v3"}
    assert set(sc["method"]) == want and set(plan["method"]) == want
    assert set(sc["modalities"]) == {"rna+atac_peak+atac_gas"}
    assert set(mtb.find_methods("diagonal", modalities=BOTH)) == want
    assert set(mtb.find_methods("diagonal", modalities=["peak", "gas"])) == want
    # inputs_for agrees (it did before, and still does)
    inp = mtb.engine.resolve.inputs_for("D28", "diagonal", "MultiMAP", modalities=BOTH, data_path=data)
    assert {"atac_peak", "atac_gas"} <= set(inp)


def test_a_two_file_method_gets_no_representation_caveat(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = full_scan("D28", "diagonal", data_path=data, verbose=False)
    for m in ("MultiMAP", "Seurat_v3"):
        assert "look like peaks" not in sc.loc[sc["method"] == m, "caveat"].item()


# ------------------------------------------------------------------ L13 inputs_for
@pytest.mark.parametrize("ds,cat,method,mods,msg", [
    ("D28", "diagonal", "GLUE", ["rna", "gas"], "GLUE reads peaks; modalities name gene activity"),
    ("D28", "diagonal", "SCALEX", ["rna", "peak"], "SCALEX reads gene activity; modalities name peaks"),
    ("D11", "vertical", "Matilda", ["rna", "gas"], "Matilda reads peaks; modalities name gene activity"),
    ("D28", "diagonal", "GLUE", BOTH, "GLUE reads peaks; modalities name peaks and gene activity"),
])
def test_inputs_for_applies_the_representation_rule(data, ds, cat, method, mods, msg):
    with pytest.raises(KeyError, match=re.escape(msg)):
        mtb.engine.resolve.inputs_for(ds, cat, method, modalities=mods, data_path=data)
    # scan leaves the same method out for the same tokens
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = full_scan(ds, cat, modalities=mods, data_path=data, verbose=False)
    assert method not in set(sc["method"])


def test_inputs_for_keeps_a_variants_own_roles_and_matching_tokens(data):
    # moETM reads peaks under the role atac since 0.4.0 (X13); 0.3's atac_gas
    # spelling still reaches it
    assert "atac" in mtb.engine.resolve.inputs_for("D11", "vertical", "moETM",
                                    modalities=["rna", "atac_gas"], data_path=data,
                                    check=False)
    assert "atac" in mtb.engine.resolve.inputs_for("D11", "vertical", "moETM",
                                    modalities=["rna", "peak"], data_path=data,
                                    check=False)
    assert "atac" in mtb.engine.resolve.inputs_for("D11", "vertical", "Matilda",
                                    modalities=["rna", "peak"], data_path=data, check=False)
    assert "atac_gas" in mtb.engine.resolve.inputs_for("D28", "diagonal", "SCALEX",
                                        modalities=["rna", "atac"], data_path=data,
                                        check=False)


def test_the_vocabulary_message_calls_peak_and_gas_representation_tokens():
    with pytest.raises(ValueError) as ei:
        registry.normalize_modalities(["rna", "bogus"])
    msg = str(ei.value)
    assert "aliases: peak" not in msg
    assert "atac_peak (or peak, peaks) and atac_gas (or gas, gene_activity) name the " \
           "ATAC form a method reads" in msg
    notes = mtb.engine.resolve.inputs_for.__doc__
    assert "``peak`` / ``peaks`` for ``atac_peak``" not in notes


# ------------------------------------------------------------------ L13 the two rules
def test_scan_keeps_one_combination_and_find_methods_at_least(data):
    fm = set(mtb.find_methods("mosaic", modalities=["rna", "adt"]))
    assert {"Multigrate", "StabMap", "scMoMaT"} <= fm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = full_scan("D46", "mosaic", modalities=["rna", "adt"], data_path=data,
                      verbose=False)
    # every kept row reads exactly RNA and ADT: no ATAC role
    assert len(sc) and not sc["modalities"].str.contains("atac").any()
    assert set(sc["method"]) < fm
    scan_doc, fm_doc = mtb.scan.__doc__, mtb.find_methods.__doc__
    assert "the rule is the one ``mtb.find_methods`` uses" not in scan_doc
    assert "a row's modalities must be exactly the named combination" not in fm_doc


# ------------------------------------------------------------------ L48 RunResult.plot
def test_batchresult_plot_reads_the_fill_as_scaled_per_column():
    doc = " ".join(mtb.RunResult.plot.__doc__.split())
    assert "darker is higher" not in doc and "rank 1 is the largest" not in doc
    assert "Circle size shows the rank within a column (bigger is better)." in doc
    assert "the lightest is the lowest in this figure, not zero." in doc


# ------------------------------------------------------------------ .loom extra
def test_the_loom_hint_names_the_distribution(tmp_path, monkeypatch):
    import builtins
    import sys
    from multibench.engine import ingest

    real_import = builtins.__import__

    def no_loompy(name, *a, **k):
        if name == "loompy":
            raise ModuleNotFoundError("No module named 'loompy'")
        return real_import(name, *a, **k)

    monkeypatch.setitem(sys.modules, "loompy", None)
    monkeypatch.setattr(builtins, "__import__", no_loompy)
    (tmp_path / "x.loom").write_bytes(b"")
    with pytest.raises(ImportError) as ei:
        ingest._to_canonical(tmp_path / 'x.loom', tmp_path / 'x.h5', modality='rna')
    assert "pip install 'multibench-sc[loom]'" in str(ei.value)
    assert "'multibench[loom]'" not in str(ei.value)
    from multibench.engine import runner
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: None)
    with pytest.raises(ImportError, match=r"pip install 'multibench-sc\[loom\]'"):
        runner._check_input("rna", str(tmp_path / "x.loom"), real=False)


# ------------------------------------------------------------------ plain vocabulary
def _contract_objects():
    """The reference objects of the docs (the docstring contract), with the
    public methods of the classes among them."""
    import importlib
    import inspect
    from tests.test_docs_consistency import REFERENCE

    out = []
    for name, (_page, path) in sorted(REFERENCE.items()):
        mod, _, attr = path.rpartition(".")
        obj = getattr(importlib.import_module(mod), attr)
        out.append((name, obj))
        if inspect.isclass(obj):
            for k, v in vars(obj).items():
                if not k.startswith("_") and (inspect.isfunction(v) or isinstance(v, property)):
                    out.append((f"{name}.{k}", v))
    from multibench.engine import envs
    out += [("mtb.env.env_prefix", envs.env_prefix),
            ("mtb.env.host_has_gpu", envs.host_has_gpu)]
    return out


@pytest.mark.parametrize("name,obj", _contract_objects(), ids=lambda x: x if isinstance(x, str) else "")
def test_reference_docstrings_use_plain_words(name, obj):
    import inspect
    doc = inspect.getdoc(obj) or ""
    assert not re.search(r"(?i)\btidy\b", doc), name
    flat = " ".join(doc.split())
    for banned in ("stand-in", "mtb.plot.render", "mtb.env.group_for",
                   "mtb.env.default_env_name", "host_platform_problem", "failed silently",
                   "OUT_DIR_PLACEHOLDER", "output_urls.json"):
        assert banned not in flat, (name, banned)


def test_module_help_is_written_for_users():
    for mod in (mtb.env, mtb.config, mtb.catalog):
        doc = mod.__doc__
        for internal in (".yaml", "install_packed", "create_env", "files/", "space-named",
                         "scib_metric", "env_spec"):
            assert internal not in doc, (mod.__name__, internal)


# ---- restored from before the 0.4 cut: kept behaviour
# ------------------------------------------------------------------ L61 method_info text
@pytest.mark.parametrize("method", mtb.find_methods())
def test_method_info_notes_and_setup_hint_hold_no_internal_names(method):
    info = mtb.method_info(method)
    for key in ("notes", "setup_hint"):
        text = str(info.get(key) or "")
        for bad in ("engine/", "source()d", ".yaml"):
            assert bad not in text, (method, key, bad)
    if method == "StabMap":
        assert info["notes"].endswith("using shared and unshared features; R.")


def test_the_na_hint_uses_the_spelling_of_the_caller(tmp_path, capsys):
    import matplotlib
    matplotlib.use('Agg')
    args = ['plot', 'bubble', '--category', 'diagonal', '--dataset', 'D28', '--metrics', 'ARI,NMI,ASW', '--out', str(tmp_path / 'f.png')]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        mtb.plot.bubble(mtb.load_results('diagonal', dataset='D28'), metrics=['ARI', 'NMI', 'ASW'])
    assert not [w for w in rec if 'hide this message' in str(w.message)]
