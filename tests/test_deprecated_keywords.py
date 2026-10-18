"""Deprecated keywords, flags and names: one warning, still working, out of the signature and help."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import builder, envs, registry  # noqa: F401


def _messages(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec]


def _d11():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.load_results("vertical", dataset="D11")


def _mine(base, name="Mine", scored_with="leidenalg/search/0.4.0"):
    rows = base[base.method == "Matilda"].assign(method=name, source="user")
    if scored_with is not None:
        rows = rows.assign(scored_with=scored_with)
    return rows


MATILDA = {"rna": "/d/rna.h5", "adt": "/d/adt.h5", "cty": "/d/cty.csv"}
SCJOINT = {"rna": "r.h5", "atac_gas": "a.h5", "rna_cty": "c.csv"}


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


def _dry(method, category, inputs, params, tmp_path):
    return mtb.run(method, category, inputs=inputs, out_dir=str(tmp_path / "o"),
                   params=params, dry_run=True, cmd_template="conda run -n x {cmd}")


# --- deprecations and catalog: X39, X40, X41, X43 ----------------------------

def test_x39_no_tasks_column_or_key():
    assert "tasks" not in mtb.catalog.methods().columns
    assert "tasks" not in mtb.method_info("Matilda", verbose=True)


def test_x40_method_info_default_keys_and_the_verbose_hint():
    info = mtb.method_info("Matilda")
    assert set(info) == {"id", "language", "notes", "categories", "env", "atac",
                         "needs_labels", "supports", "runtime", "gpu", "requires_gpu",
                         "cpu_params", "setup_hint", "repo_url", "reference"}
    with pytest.raises(KeyError, match=r"'params' is in method_info\('Matilda', "
                                       r"verbose=True\)"):
        info["params"]
    full = mtb.method_info("Matilda", verbose=True)
    assert {"params", "scripts_url", "notes_long", "upstream_knobs"} <= set(full)


def test_x41_build_table_na_is_deprecated():
    with pytest.warns(FutureWarning, match=r"^plot\.build_table\(na=\) is deprecated and "
                                           r"will be removed in 0\.5; leave it out"):
        mtb.plot.build_table(_d11(), na="skip")
    import inspect
    assert "na" not in inspect.signature(mtb.plot.build_table).parameters


@pytest.mark.parametrize("fn", ["methods", "datasets", "metrics"])
def test_x43_catalog_files_dir_is_deprecated(fn):
    from multibench import config
    f = getattr(mtb.catalog, fn)
    import inspect
    assert "files_dir" not in inspect.signature(f).parameters
    want = f()
    with pytest.warns(FutureWarning, match=rf"^catalog\.{fn}\(files_dir=\) is deprecated "
                                           r"and will be removed in 0\.5; leave it out "
                                           r"for the package's tables\.$"):
        by_kw = f(files_dir=config.DEFAULT.files_path)
    with pytest.warns(FutureWarning):
        by_pos = f(config.DEFAULT.files_path)               # 0.3 took it by position
    pd.testing.assert_frame_equal(by_kw, want)
    pd.testing.assert_frame_equal(by_pos, want)


# X96 / X44: deprecated keywords and flags warn once, keep working, leave the signature
@pytest.mark.parametrize("call, msg", [
    (lambda: mtb.find_methods("diagonal", atac="peak"),
     'find_methods(atac=) is deprecated and will be removed in 0.5. Use '
     'modalities=["atac_peak"] or ["atac_gas"].'),
    (lambda: mtb.recommend("diagonal", atac="peak"),
     'recommend(atac=) is deprecated and will be removed in 0.5. Use '
     'modalities=["atac_peak"] or ["atac_gas"].'),
])
def test_atac_keyword_is_deprecated(call, msg):
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = call()
    assert [str(x.message) for x in w if x.category is FutureWarning] == [msg]
    assert len(got)


def test_atac_and_mod_leave_the_signatures():
    import inspect
    assert "atac" not in inspect.signature(mtb.find_methods).parameters
    assert "atac" not in inspect.signature(mtb.recommend).parameters
    assert "mod" not in inspect.signature(mtb.io.to_canonical).parameters
    assert mtb.find_methods("diagonal", atac="peak") == mtb.find_methods(
        "diagonal", modalities=["atac_peak"])


def test_to_canonical_mod_is_deprecated(tmp_path):
    import numpy as np
    ad = pytest.importorskip("anndata")
    a = ad.AnnData(np.ones((3, 2)))
    with pytest.warns(FutureWarning, match=r"^io\.to_canonical\(mod=\) is deprecated and "
                                           r"will be removed in 0\.5\. Use the modality "
                                           r"itself"):
        with pytest.raises(ValueError, match="only applies to MuData"):
            mtb.io.to_canonical(a, tmp_path / "x.h5", mod="rna")


def test_cli_find_atac_and_convert_mod_are_hidden(capsys):
    from multibench import cli
    with pytest.raises(SystemExit):
        cli.main(["find", "--help"])
    assert "--atac" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["convert", "--help"])
    assert "--mod " not in capsys.readouterr().out
    assert cli.main(["find", "--category", "diagonal", "--atac", "peak"]) == 0
    out = capsys.readouterr()
    assert "warning: --atac is deprecated and will be removed in 0.5. Use --modalities " \
           "atac_peak or atac_gas." in out.err
    assert out.out.split() == mtb.find_methods("diagonal", modalities=["atac_peak"])


# Y35: mtb.DegenerateRerunWarning leaves the surface and still works
def test_degenerate_rerun_warning_is_deprecated():
    from multibench.data.results import DegenerateRerunWarning
    assert "DegenerateRerunWarning" not in mtb.__all__ and "DegenerateRerunWarning" not in dir(mtb)
    with pytest.warns(FutureWarning, match=r"^DegenerateRerunWarning is deprecated and will "
                                           r"be removed in 0\.5; nothing raises it\.$"):
        assert mtb.DegenerateRerunWarning is DegenerateRerunWarning
