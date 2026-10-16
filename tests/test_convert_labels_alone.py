"""``multibench convert`` with only ``--labels`` and no matrix refuses to write
a folder that holds labels alone (it wrote cty.csv and exited 0, while the
Python ``export_dataset`` defaults to ``rna='X'``). Adding the labels to a
folder that already holds a matrix still works."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

from multibench import cli


@pytest.fixture
def one(tmp_path):
    a = ad.AnnData(np.ones((20, 5), dtype="float32"),
                   obs=pd.DataFrame({"celltype": ["A", "B"] * 10},
                                    index=[f"c{i}" for i in range(20)]))
    a.var_names = [f"g{i}" for i in range(5)]
    a.write_h5ad(tmp_path / "one.h5ad")
    return tmp_path / "one.h5ad"


def test_labels_alone_is_refused(one, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["convert", str(one), str(tmp_path / "CV"), "--category", "vertical",
                  "--labels", "obs:celltype"])
    assert e.value.code == 2
    assert ("error: --labels alone writes only the label file. Add --rna X to export the "
            "matrix too.") in capsys.readouterr().err
    assert not (tmp_path / "CV").exists()


def test_labels_added_to_a_folder_with_a_matrix(one, tmp_path):
    out = tmp_path / "CV"
    assert cli.main(["convert", str(one), str(out), "--category", "vertical",
                     "--rna", "X"]) == 0
    assert cli.main(["convert", str(one), str(out), "--category", "vertical",
                     "--labels", "obs:celltype"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["cty.csv", "rna.h5"]
