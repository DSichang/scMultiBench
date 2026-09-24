"""Plot and ranking warnings after the third student study (fix round 3).

R3-06: the Leiden-backend warning counts an igraph-scored row only when the
frame holds stored rows of the same dataset. Rows from the user's own
dataset get the no-overlap warning alone.
R3-07: recommend(long_df=...) gives the same backend line in its warning.
R3-08: the n/a warning names the row and its source in short sentences.
"""
import importlib
import warnings

import matplotlib
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config

B = importlib.import_module("multibench.plot.bubble")

matplotlib.use("Agg")

NO_OVERLAP = ("rows come from 2 datasets (D11, MYCITE) that share no method, so the "
              "figure ranks unrelated rows against each other. Plot each dataset on "
              "its own, or score your method on D11 and add it to that table.")


def _messages(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


def _stored(dataset, source="rerun", category="vertical"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.load_results(category, dataset=dataset, source=source)


def _mine(method, dataset, scored_with="igraph/sweep/0.3.2",
          metrics=("ARI", "NMI", "ASW", "iASW", "iF1", "cLISI"), base=0.5):
    return pd.DataFrame([
        {"metric": m, "value": base + 0.01 * i, "method": method, "dataset": dataset,
         "category": "vertical", "clustering": "default", "source": "user",
         "scored_with": scored_with}
        for i, m in enumerate(metrics)])


def _mycite():
    # S1: two baselines of her own CITE-seq dataset, scored with the default backend
    return pd.concat([_mine("RNA PCA", "MYCITE"), _mine("ADT PCA", "MYCITE", base=0.6)],
                     ignore_index=True)


# --- R3-06 -------------------------------------------------------------------

@pytest.mark.parametrize("fn", [mtb.plot.build_table, mtb.plot.bubble, mtb.plot.bar])
def test_own_dataset_rows_get_the_no_overlap_warning_alone(fn):
    df = pd.concat([_stored("D11"), _mycite()], ignore_index=True)
    _, msgs = _messages(fn, df)
    assert NO_OVERLAP in msgs, msgs
    assert not any("igraph Leiden backend" in m for m in msgs), msgs


@pytest.mark.parametrize("fn", [mtb.plot.build_table, mtb.plot.bubble, mtb.plot.bar])
def test_same_dataset_igraph_row_still_warns(fn):
    df = pd.concat([_stored("D11"), _mine("PriyaNet", "D11"), _mycite()],
                   ignore_index=True)
    _, msgs = _messages(fn, df)
    hit = [m for m in msgs if "igraph Leiden backend" in m]
    # only the D11 row is named: the MYCITE rows have no stored row to meet
    assert len(hit) == 1 and hit[0].startswith("The rows for PriyaNet were clustered"), msgs


def test_backend_rule_without_a_dataset_column_is_unchanged():
    stored = _stored("D11").drop(columns="dataset")
    own = _mine("PriyaNet", "D11").drop(columns="dataset")
    msg = mtb.plot.style.backend_warning(pd.concat([stored, own], ignore_index=True))
    assert msg and msg.startswith("The rows for PriyaNet were clustered")


def _plot_cli(tmp_path, capsys, *frames):
    paths = []
    for i, f in enumerate(frames):
        p = tmp_path / f"in{i}.csv"
        f.to_csv(p, index=False)
        paths += ["--input", str(p)]
    assert cli.main(["plot", "bubble", *paths, "--out", str(tmp_path / "f.png")]) == 0
    return capsys.readouterr().err


def test_cli_plot_follows_the_dataset_rule(tmp_path, capsys):
    err = _plot_cli(tmp_path, capsys, _stored("D11"), _mycite())
    assert "share no method" in err and "igraph Leiden backend" not in err
    err = _plot_cli(tmp_path, capsys, _stored("D11"), _mine("PriyaNet", "D11"))
    assert "warning: The rows for PriyaNet were clustered with the igraph Leiden backend" in err
    assert "multibench evaluate --leiden-flavor leidenalg" in err


# --- R3-07 -------------------------------------------------------------------

def test_recommend_long_df_warns_about_igraph_rows_of_a_stored_dataset():
    # S4: her method, scored with the default backend, ranked against the re-run tables
    long_df = pd.concat([_stored(None), _mine("PriyaNet_ig", "D11", base=0.9)],
                        ignore_index=True)
    r, msgs = _messages(mtb.recommend, "vertical", long_df=long_df)
    assert "PriyaNet_ig" in set(r["method"])
    (msg,) = [m for m in msgs if m.startswith("recommend('vertical'):")]
    assert ("\n  - The rows for PriyaNet_ig were clustered with the igraph Leiden backend. "
            "The stored tables used leidenalg") in msg
    assert "mtb.config.DEFAULT.leiden_flavor = 'leidenalg'" in msg


def test_recommend_backend_line_needs_igraph_rows_a_stored_dataset_and_a_sweep_metric():
    stored = _stored(None)
    cases = {
        "leidenalg": pd.concat([stored, _mine("P", "D11", scored_with="leidenalg/sweep/0.3.2")]),
        "stored only": stored,
        "own dataset": pd.concat([stored, _mine("P", "MYCITE"), _mine("Q", "MYCITE")]),
    }
    for what, df in cases.items():
        _, msgs = _messages(mtb.recommend, "vertical", long_df=df.reset_index(drop=True))
        assert not any("igraph Leiden backend" in m for m in msgs), (what, msgs)
    # ASW and cLISI do not depend on the Leiden backend
    df = pd.concat([stored, _mine("P", "D11")], ignore_index=True)
    _, msgs = _messages(mtb.recommend, "vertical", long_df=df, metrics=["ASW", "cLISI"])
    assert not any("igraph Leiden backend" in m for m in msgs), msgs


def test_recommend_backend_line_cli_spelling(monkeypatch):
    monkeypatch.setattr(config, "_CLI", True)
    df = pd.concat([_stored(None), _mine("P", "D11")], ignore_index=True)
    _, msgs = _messages(mtb.recommend, "vertical", long_df=df)
    (msg,) = [m for m in msgs if "igraph Leiden backend" in m]
    assert "multibench evaluate --leiden-flavor leidenalg" in msg
    assert "mtb.config" not in msg


def test_recommend_documents_the_backend_line():
    doc = " ".join(mtb.recommend.__doc__.split())
    assert "Warns" in mtb.recommend.__doc__
    assert "igraph" in doc and "leidenalg" in doc


# --- R3-08 -------------------------------------------------------------------

def _d28_with_mine():
    # S3: her own method on D28, added to the published D28 table, all metrics
    pub = _stored("D28", source="published", category="diagonal")
    metrics = sorted(pub["metric"].unique())
    mine = _mine("LiNet", "D28", scored_with="leidenalg/sweep/0.3.2", metrics=metrics)
    return pd.concat([pub, mine.assign(category="diagonal")], ignore_index=True)


def test_na_warning_names_the_stored_row_in_short_sentences():
    _, msgs = _messages(mtb.plot.build_table, _d28_with_mine())
    (msg,) = [m for m in msgs if "has no" in m]
    assert msg.startswith("The stored row Conos (D28, published) has no ASW, iASW or "
                          "ASW_batch. Its Overall uses the metrics it has.")
    assert msg.endswith("Pass na='skip' to hide this message.")
    assert " - the family Overall" not in msg and "LiNet" not in msg
    with pytest.raises(ValueError) as e:
        mtb.plot.build_table(_d28_with_mine(), na="raise")
    # R4-12: the error names na='warn' instead of offering to hide itself
    # R5-11: and leaves out the Overall sentence, since no figure is drawn
    assert str(e.value) == (msg.replace("Pass na='skip' to hide this message.",
                                        "Pass na='warn' to draw the figure with "
                                        "these gaps.")
                            .replace(" Its Overall uses the metrics it has.", ""))


def test_na_warning_names_your_row():
    df = _d28_with_mine()
    df = df[~((df["method"] == "LiNet") & (df["metric"] == "GC"))]
    _, msgs = _messages(mtb.plot.build_table, df)
    (msg,) = [m for m in msgs if "has no" in m]
    assert "Your row LiNet (D28) has no GC." in msg
    assert "Each row's Overall uses the metrics it has." in msg


def test_na_warning_stops_after_three_rows():
    frames = []
    for i, m in enumerate("ABCDE"):
        rows = _mine(m, "X", base=0.1 * i)
        frames.append(rows[rows["metric"] != "NMI"] if m != "E" else rows)
    _, msgs = _messages(mtb.plot.build_table, pd.concat(frames, ignore_index=True))
    (msg,) = [m for m in msgs if "has no" in m]
    assert msg.count("has no NMI.") == 3
    assert "... and 1 more row." in msg
    assert "Each row's Overall uses the metrics it has." in msg


def test_na_warning_in_summary_mode():
    df = pd.concat([_mine("A", "X"), _mine("B", "X", base=0.6),
                    _mine("A", "Y"), _mine("B", "Y", base=0.6)], ignore_index=True)
    df = df[~((df["method"] == "B") & (df["metric"] == "NMI") & (df["dataset"] == "Y"))]
    _, msgs = _messages(mtb.plot.build_table, df, aggregate="summary")
    (msg,) = [m for m in msgs if "has no" in m]
    assert msg.startswith("Your row B has no NMI on Y. In the summary, a missing "
                          "value counts as the lowest rank on that dataset.")


def test_na_warning_cli_spelling(tmp_path, capsys):
    pub = _stored("D28", source="published", category="diagonal")
    p = tmp_path / "d28.csv"
    pub.to_csv(p, index=False)
    assert cli.main(["plot", "bubble", "--input", str(p), "--out",
                     str(tmp_path / "f.png")]) == 0
    err = capsys.readouterr().err
    assert "warning: The stored row Conos (D28, published) has no" in err
    assert "Pass --na skip to hide this message." in err and "na='skip'" not in err


def test_na_cells_keep_their_per_family_lines():
    tbl, _ = _messages(mtb.plot.build_table, _d28_with_mine())
    assert tbl.na_cells == [
        "Conos: DR and clustering Overall over 4 of 6 metrics (ASW, iASW n/a)",
        "Conos: Batch correction Overall over 3 of 4 metrics (ASW_batch n/a)"]
