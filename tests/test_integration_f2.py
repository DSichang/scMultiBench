"""Checks that guard the merge of the round-2 work packages."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "multibench"


def _module_imports(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


def _own_nodes(fn):
    """The nodes of ``fn``'s body, without nested functions and classes."""
    for child in ast.iter_child_nodes(fn):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                              ast.ClassDef)):
            continue
        yield child
        yield from _own_nodes(child)


def test_no_module_level_name_is_used_before_a_local_reimport():
    """A function-local ``from .. import config`` next to a module-level one makes
    ``config`` local to the whole function, so a use above that import raises
    UnboundLocalError. Two branches merged that way broke export_dataset."""
    found = []
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text())
        top = _module_imports(tree)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            nodes = list(_own_nodes(fn))
            first = {}
            for node in nodes:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for a in node.names:
                        name = (a.asname or a.name).split(".")[0]
                        if name in top:
                            first[name] = min(first.get(name, node.lineno), node.lineno)
            for node in nodes:
                if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                        and node.id in first and node.lineno < first[node.id]):
                    found.append(f"{path.relative_to(PKG.parent)}:{node.lineno} "
                                 f"{fn.name} uses {node.id} above its local import "
                                 f"(line {first[node.id]})")
    assert not found, "\n".join(found)


def _h5(path, n_feat, n_cell, feats=None):
    import h5py
    import numpy as np
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix/data", data=np.zeros((n_feat, n_cell)))
        f.create_dataset("matrix/features",
                         data=np.array(feats or [f"g{i}" for i in range(n_feat)], dtype="S"))
        f.create_dataset("matrix/barcodes",
                         data=np.array([f"c{i}" for i in range(n_cell)], dtype="S"))


PEAKS = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)]


def test_near_miss_hint_names_the_rename_when_the_kind_matches(tmp_path, monkeypatch):
    """M23 (files_reason part): a peak method on a vertical folder whose peaks
    sit in atac_peak.h5 gets the same fix as scan's short reason."""
    import pytest
    from multibench import config
    from multibench.engine import resolve
    d = tmp_path / "MU"
    d.mkdir()
    _h5(d / "rna.h5", 30, 50)
    _h5(d / "atac_peak.h5", 40, 50, feats=PEAKS)
    with pytest.raises(FileNotFoundError) as ei:
        resolve.inputs_for("MU", "vertical", "scMVP", data_path=tmp_path, check=True)
    msg = str(ei.value)
    assert ("atac.h5 not found; found atac_peak.h5 - vertical reads atac.h5: rename "
            "atac_peak.h5 to atac.h5, or write it with category=\"vertical\"") in msg
    assert "pass the representation" not in msg
    # the CLI spelling of the same fix
    monkeypatch.setattr(config, "_CLI", True)
    hints = resolve._near_miss_hints(d, {"atac": str(d / "atac.h5")}, "vertical",
                                     atac="peak")
    assert hints and hints[0].endswith("--category vertical"), hints
    # a method that reads gene activity keeps the representation pointer
    hints = resolve._near_miss_hints(d, {"atac": str(d / "atac.h5")}, "vertical",
                                     atac="gene_activity")
    assert "pass the representation this method wants" in hints[0]


def test_incomplete_matrix_warning_names_the_overall_flag_on_the_cli(monkeypatch):
    """M19 intent after the merge: the whole incomplete-matrix message uses CLI
    spellings under the CLI, not only its last sentence."""
    import pandas as pd
    from multibench import config
    from multibench.plot import style
    a = pd.DataFrame({"ARI": [0.1, 0.2]}, index=["M1", "M2"])
    b2 = pd.DataFrame({"ARI": [0.3, 0.5]}, index=["M1", "M3"])
    parts = {"D1": a, "D2": b2}

    def incomplete():
        return next(m for m in style.coverage_warnings(parts, basis="rank",
                                                       incomplete_fix="FIX.")
                    if m.startswith("The summary ranks "))
    monkeypatch.setattr(config, "_CLI", False)
    py = incomplete()
    assert "With overall='rank', a missing dataset counts as rank 0" in py
    assert "With overall='mean_overall', the missing dataset is left out." in py
    monkeypatch.setattr(config, "_CLI", True)
    cli = incomplete()
    assert "With --overall rank, a missing dataset counts as rank 0" in cli
    assert "With --overall mean_overall, the missing dataset is left out." in cli
    assert "overall='" not in cli
