"""Core round c3 (0.4.0): the fixes of the cross-check (c3/crosscheck/items.json)."""
import contextlib
import importlib
import io
import json
import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli
from multibench.plot.style import overall

CATS = {"vertical": "D11", "diagonal": "D28", "mosaic": "D45", "cross": "D52"}


def _quiet(f, *a, **k):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        return f(*a, **k)


def _bars(fig):
    ax = fig.axes[0]
    return ([t.get_text() for t in ax.get_yticklabels()],
            [round(p.get_width(), 9) for p in ax.patches])


def _this_run(category):
    """Rows of two stored methods, a little worse, as '<method> (this run)'."""
    st = mtb.load_results(category)
    two = st[st["method"].isin(sorted(st["method"].unique())[:2])]
    return two.assign(method=two["method"] + " (this run)", source="user",
                      value=two["value"] * 0.98)


# --------------------------------------------------------------- D01
@pytest.mark.parametrize("category", list(CATS))
def test_cli_plot_bar_draws_the_python_bar(category, tmp_path, monkeypatch):
    pb = importlib.import_module("multibench.plot.bar")
    run = _this_run(category)
    run.to_csv(tmp_path / "run.csv", index=False)
    want = _bars(_quiet(mtb.plot.bar, pd.concat([mtb.load_results(category), run])))
    seen = {}
    real = pb._bar
    monkeypatch.setattr(pb, "_bar", lambda *a, **k: seen.setdefault("fig", real(*a, **k)))
    assert _quiet(cli.main, ["plot", "bar", "--category", category, "--input",
                             str(tmp_path / "run.csv"), "--out", str(tmp_path / "b.png")]) == 0
    assert _bars(seen["fig"]) == want


def test_cli_plot_bubble_still_draws(tmp_path):
    assert _quiet(cli.main, ["plot", "bubble", "--category", "vertical",
                             "--out", str(tmp_path / "b.png")]) == 0


# --------------------------------------------------------------- C18
def test_rows_scored_without_a_dataset_are_told_to_score_with_one():
    st = mtb.load_results("vertical")
    mine = st[st["method"] == "Matilda"].assign(method="PCA", dataset="user", source="user")
    with pytest.raises(ValueError) as e:
        _quiet(mtb.plot.bubble, pd.concat([st, mine]))
    assert str(e.value) == ("Your rows were scored without a dataset: score your embedding "
                            "with dataset='D11' to compare it with the stored results.")


def test_the_cli_names_the_evaluate_command_and_prints_no_added_line(tmp_path, capsys):
    st = mtb.load_results("vertical")
    st[st["method"] == "Matilda"].assign(method="PCA", dataset="MINE", source="user").to_csv(
        tmp_path / "mine.csv", index=False)
    rc = cli.main(["plot", "bubble", "--category", "vertical", "--input",
                   str(tmp_path / "mine.csv"), "--out", str(tmp_path / "b.png")])
    err = capsys.readouterr().err
    assert rc != 0 and "score your embedding with multibench evaluate --dataset D11." in err
    assert "# Added" not in err


# --------------------------------------------------------------- C14
def test_cli_two_inputs_with_the_same_methods_name_both(tmp_path, monkeypatch, capsys):
    pb = importlib.import_module("multibench.plot.bubble")
    seen = {}
    monkeypatch.setattr(pb, "_bubble", lambda df, **kw: seen.update(df=df))
    st = mtb.load_results("vertical")
    for name in ("epochs_10", "epochs_40"):
        st.assign(source="user").to_csv(tmp_path / f"{name}.csv", index=False)
    assert cli.main(["plot", "bubble", "--input", str(tmp_path / "epochs_10.csv"), "--input",
                     str(tmp_path / "epochs_40.csv"), "--methods", "scMM",
                     "--out", str(tmp_path / "x.png")]) == 0
    names = set(seen["df"]["method"])
    assert {"scMM (epochs_10)", "scMM (epochs_40)"} <= names and "scMM" not in names
    assert "# Drawn" not in capsys.readouterr().err


# --------------------------------------------------------------- E03
@pytest.mark.parametrize("category", ["vertical", "diagonal", "cross"])
def test_rescoring_noise_scores_as_the_stored_row(category):
    """Not mosaic: its stored NMI_batch ranks MultiVI over Multigrate by 5.4e-10."""
    st = mtb.load_results(category)
    near = st.assign(method=st["method"] + " (this run)", source="user",
                     value=st["value"] + 5e-8 * np.sign(np.sin(np.arange(len(st)))))
    s = overall(pd.concat([st, near]))
    for m in st["method"].unique():
        assert s[m] == pytest.approx(s[m + " (this run)"], abs=1e-12), m
    base = overall(st)
    assert np.allclose(s[base.index], base, atol=1e-12)


# --------------------------------------------------------------- E11
def test_a_renamed_copy_of_stored_rows_does_not_move_the_stored_scores():
    st = mtb.load_results("diagonal")
    copy = st[st["method"] == "scJoint"].assign(method="scJoint (copy)")   # source "stored"
    s, base = overall(pd.concat([st, copy])), overall(st)
    assert np.allclose(s[base.index], base, atol=1e-12)
    assert s["scJoint (copy)"] == pytest.approx(s["scJoint"])


# --------------------------------------------------------------- E04
def test_the_bar_axis_does_not_claim_0_1():
    st = mtb.load_results("vertical")
    best = st[st["method"] == "Matilda"].assign(method="Better", source="user",
                                                 value=lambda d: d["value"] + 0.2)
    fig = _quiet(mtb.plot.bar, pd.concat([st, best]))
    assert fig.axes[0].get_xlabel() == "overall score (D11)"


# --------------------------------------------------------------- D03
def _saved_tree(tmp_path):
    """A saved run_all folder of D11 (its record names 'Matilda_D11'), under runs/."""
    from tests.test_c2_fixes import _old_tree
    out = _old_tree(tmp_path)
    blob = json.loads((out / "batch_result.json").read_text())
    blob["records"][0]["out_dir"] = "Matilda_D11"          # as BatchResult.save writes it
    (out / "batch_result.json").write_text(json.dumps(blob))
    (tmp_path / "runs").mkdir()
    return out.rename(tmp_path / "runs" / "D11")


@pytest.mark.parametrize("where", ["parent", "elsewhere", "absolute"])
def test_skip_existing_rescores_from_any_directory(where, tmp_path, monkeypatch):
    folder = _saved_tree(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    cwd, out = {"parent": (tmp_path, "runs/D11"),
                "elsewhere": (tmp_path / "elsewhere", "../runs/D11"),
                "absolute": (tmp_path / "elsewhere", str(folder))}[where]
    monkeypatch.chdir(cwd)
    for _ in range(2):                                  # and again on what it saved
        res = _quiet(mtb.run_all, "D11", "vertical", methods=["Matilda"], out_dir=out,
                     skip_existing=True)
        s = res.summary.set_index("method")
        assert (s.loc["Matilda", "status"], s.loc["Matilda", "run_sec"]) == ("scored", 17.0)


def test_cli_skip_existing_from_the_parent(tmp_path, monkeypatch):
    _saved_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert _quiet(cli.main, ["run-all", "D11", "--category", "vertical", "--methods",
                             "Matilda", "--out-dir", "runs/D11", "--skip-existing"]) == 0


# --------------------------------------------------------------- E01
def test_rescore_refuses_a_batch_of_the_wrong_length_before_scoring(tmp_path, monkeypatch):
    from multibench import workflow
    scored = []
    monkeypatch.setattr(workflow, "_score_record", lambda *a, **k: scored.append(a))
    res = mtb.load_batch(_saved_tree(tmp_path))
    with pytest.raises(ValueError, match=r"^batch has 2,000 entries for the 2,864 cells of "
                                         r"D11: give one batch per cell\.$"):
        _quiet(res.rescore, batch=np.repeat(["a", "b"], 1000))
    assert not scored


# --------------------------------------------------------------- E06
def test_rescore_without_outputs_names_the_methods_and_the_fix(tmp_path):
    folder = _saved_tree(tmp_path)
    (folder / "Matilda_D11" / "embedding.h5").unlink()
    with pytest.raises(FileNotFoundError) as e:
        _quiet(mtb.load_batch(folder).rescore)
    assert str(e.value) == (f"No output of Matilda in {folder}: load the folder that holds "
                            f"its method folder, or run it again.")


# --------------------------------------------------------------- E10 / E22
def test_cli_dry_run_refuses_a_wrong_length_batch_before_printing(tmp_path, capsys):
    pd.DataFrame({"donor": ["a", "b"] * 250}).to_csv(tmp_path / "b.csv", index=False)
    rc = cli.main(["run-all", "D11", "--category", "vertical", "--dry-run", "--methods",
                   "Matilda", "--batch", str(tmp_path / "b.csv")])
    cap = capsys.readouterr()
    assert rc != 0 and cap.out == ""
    assert cap.err.splitlines()[0] == ("error: --batch gave 500 batch ids for the 2,864 cells "
                                       "of D11: give one batch per cell.")
