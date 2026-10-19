"""Each command is the twin of its function: scan prints mtb.scan's table, evaluate the
long table, export writes what export_dataset writes, run-all --dry-run says what the
Python dry run says (core040 M8, M16, R-7, S-4)."""
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli

D11 = lambda: mtb.config.DEFAULT.data_path / "D11"          # noqa: E731


def test_scan_prints_the_table_of_mtb_scan(capsys):
    assert cli.main(["scan", "D11", "--methods", "Matilda,totalVI"]) == 0
    out, err = capsys.readouterr()
    head = out.splitlines()[0].split()
    assert head == ["method", "modalities", "runnable", "reason"]
    assert len(out.strip().splitlines()) == 3
    assert err.startswith("[scan] 2 of 2 methods have their input files in D11.")


def test_run_all_dry_run_says_it_in_one_sentence(capsys):
    assert cli.main(["run-all", "D11", "--category", "vertical", "--methods", "Matilda",
                     "--dry-run"]) == 0
    first = capsys.readouterr().err.splitlines()[0]
    assert first.startswith("# Dry run: ") and first.count(". ") == 0


def _embedding(tmp_path):
    n = len(pd.read_csv(D11() / "cty.csv"))
    import numpy as np
    path = tmp_path / "emb.csv"
    pd.DataFrame(np.random.default_rng(0).normal(size=(n, 3))).to_csv(path, index=False)
    return path


def test_evaluate_prints_the_long_table_and_takes_any_method_name(tmp_path, capsys):
    path = _embedding(tmp_path)
    assert cli.main(["evaluate", str(path), "--dataset", str(D11()), "--method", "My PCA",
                     "--metrics", "ARI"]) == 0
    out, err = capsys.readouterr()
    assert out.split()[:7] == ["metric", "value", "method", "dataset", "category", "source",
                               "scored_with"]
    assert "My PCA" in out and " D11 " in out                # the folder's name
    assert "# labels: cty.csv from D11, in that order" in err


def test_evaluate_without_dataset_writes_user_rows(tmp_path):
    path = _embedding(tmp_path)
    out = tmp_path / "long.csv"
    assert cli.main(["evaluate", str(path), "--labels", str(D11() / "cty.csv"),
                     "--metrics", "ARI", "--out", str(out)]) == 0
    long = pd.read_csv(out)
    row = long[["method", "dataset", "category", "source"]].iloc[0]
    assert row.tolist()[:2] == ["embedding", "user"] and row["source"] == "user"
    assert pd.isna(row["category"])                      # c1 (C16): none given


def test_export_writes_the_rna_from_x_by_default(tmp_path, capsys):
    mtb.data.write_example("D11", tmp_path, verbose=False)
    assert cli.main(["export", str(tmp_path / "my_file.h5ad"), str(tmp_path / "MY"),
                     "--adt", "obsm:protein", "--labels", "obs:celltype"]) == 0
    assert sorted(p.name for p in (tmp_path / "MY").iterdir()) == ["adt.h5", "cty.csv",
                                                                   "rna.h5"]
    assert "No RNA written" not in capsys.readouterr().err


@pytest.mark.parametrize("command, flag", [
    ("export", "--modality"), ("export", "--layer"), ("export", "--obsm"),
    ("export", "--n-batches"), ("export", "--dtype"), ("scan", "--data-path"),
    ("scan", "--columns"), ("scan", "--strict"), ("run-all", "--data-path"),
    ("run-all", "--leiden-flavor"), ("evaluate", "--data-path"), ("evaluate", "--name")])
def test_cut_flags_are_not_in_help(command, flag, capsys):
    with pytest.raises(SystemExit):
        cli.main([command, "--help"])
    assert f" {flag} " not in capsys.readouterr().out.replace("\n", " ")
