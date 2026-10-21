"""Fix round 6, work package 'cli'.

R6-07: ``multibench evaluate --name`` is the row name in every case: with
``--labels`` it works without ``--method``. The errors for an unknown
``--method`` lead to the call that works; unknown-method and no-category
messages are short sentences without '?;' or 'variant'. Rows written with
``--name`` and a package ``--method`` have the columns of every long table
(the bubble figure has no ``L`` badge since 0.3.6).
R6-08: bubble's duplicate-rows error says when your rows share a name with
the stored table, and how to rename them.
R6-09: ``run(dry_run=True)`` notes an input file that does not exist; the
CLI dry run prints its header first.
R6-13: plot messages read as sentences: the CLI overlay errors, the igraph
backend warning, the require_complete warning and error.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench.engine import registry
from tests.test_f4_cli import _diag_folder

pytest.importorskip("scib")

_DATA = Path(mtb.__file__).resolve().parent.parent / "data"


def test_check_method_is_short_sentences():
    with pytest.raises(KeyError) as e:
        registry.check_method("stabmapp")
    assert e.value.args[0] == ("Unknown method stabmapp. Did you mean StabMap? "
                               "mtb.find_methods() shows all methods.")
    with pytest.raises(KeyError) as e:
        registry.check_method("zzz")
    assert e.value.args[0] == "Unknown method zzz. mtb.find_methods() shows all methods."


def test_no_category_message_names_the_categories():
    with pytest.raises(KeyError) as e:
        mtb.engine.resolve.labels_for("D11", "vertical", "SCALEX", data_path=_DATA)
    assert e.value.args[0] == "SCALEX does not run on vertical data. Its categories: diagonal."


# ====================================================================== R6-08
def _user_rows(method="uniPort", dataset="D28", metrics=("ARI", "NMI", "ASW")):
    return pd.DataFrame({"metric": list(metrics), "value": [0.3, 0.5, 0.6][:len(metrics)],
                         "method": method, "dataset": dataset, "category": "diagonal",
                         "clustering": "default", "source": "user"})


def test_a_user_source_that_starts_like_a_stored_one_is_not_stored():
    """Stored rows are those whose source is exactly one of STORED_SOURCES, as
    everywhere else; "rerun_mine" is a user's source."""
    df = pd.concat([mtb.load_results("diagonal", dataset="D28"),
                    _user_rows().assign(source="rerun_mine")], ignore_index=True)
    with pytest.raises(ValueError) as e:
        mtb.plot.bubble(df)
    assert str(e.value).startswith("Your rows and the stored results both have uniPort on "
                                   "D28.")


def test_repeated_user_rows_give_the_plain_repeat_text():
    rows = pd.concat([_user_rows("Mine"), _user_rows("Mine")], ignore_index=True)
    with pytest.raises(ValueError) as e:
        mtb.plot.bubble(rows)
    assert str(e.value) == ("6 rows repeat the same method, metric and dataset, for "
                            "example Mine, ARI, D28. The figure does not average them. "
                            "Remove the repeats, or give each version its own method name.")


# ====================================================================== R6-09
def _lung(tmp_path):
    d = tmp_path / "data" / "LUNG"
    d.mkdir(parents=True)
    for f in ("rna.h5", "atac_peak.h5"):
        (d / f).write_bytes((_DATA / "D28" / f).read_bytes())
    return d


def test_dry_run_with_existing_inputs_has_no_missing_note(tmp_path, capsys):
    inp = mtb.engine.resolve.inputs_for("D28", "diagonal", "SCALEX", data_path=_DATA)
    mtb.engine.runner.preview("SCALEX", "diagonal", inputs=inp, out_dir=str(tmp_path / "o"))[0]
    assert "which does not exist" not in capsys.readouterr().err


