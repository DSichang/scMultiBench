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
from multibench.plot.style import overall
from tests.test_docs_r4 import _read, needs_docs

CATS = {"vertical": "D11", "diagonal": "D28", "mosaic": "D45", "cross": "D52"}


def _quiet(f, *a, **k):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        return f(*a, **k)


def _this_run(category):
    """Rows of two stored methods, a little worse, as '<method> (this run)'."""
    st = mtb.load_results(category)
    two = st[st["method"].isin(sorted(st["method"].unique())[:2])]
    return two.assign(method=two["method"] + " (this run)", source="user",
                      value=two["value"] * 0.98)


# --------------------------------------------------------------- C18
def test_rows_scored_without_a_dataset_are_told_to_score_with_one():
    st = mtb.load_results("vertical")
    mine = st[st["method"] == "Matilda"].assign(method="PCA", dataset="user", source="user")
    with pytest.raises(ValueError) as e:
        _quiet(mtb.plot.bubble, pd.concat([st, mine]))
    assert str(e.value) == ("Your rows were scored without a dataset: score your embedding "
                            "with dataset='D11' to compare it with the stored results.")


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


# --------------------------------------------------------------- E04
def test_the_bar_axis_does_not_claim_0_1():
    st = mtb.load_results("vertical")
    best = st[st["method"] == "Matilda"].assign(method="Better", source="user",
                                                 value=lambda d: d["value"] + 0.2)
    fig = _quiet(mtb.plot.bar, pd.concat([st, best]))
    assert fig.axes[0].get_xlabel() == "overall score (D11)"


# --------------------------------------------------------------- D03
def _saved_tree(tmp_path):
    """A saved run folder of D11 (its record names 'Matilda_D11'), under runs/."""
    from tests.test_c2_fixes import _old_tree
    out = _old_tree(tmp_path)
    blob = json.loads((out / "batch_result.json").read_text())
    blob["records"][0]["out_dir"] = "Matilda_D11"          # as RunResult.save writes it
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
        res = _quiet(mtb.run, "D11", "vertical", methods=["Matilda"], out_dir=out,
                     skip_existing=True)
        s = res.summary.set_index("method")
        assert (s.loc["Matilda", "status"], s.loc["Matilda", "run_sec"]) == ("scored", 17.0)


# --------------------------------------------------------------- E01
def test_rescore_refuses_a_batch_of_the_wrong_length_before_scoring(tmp_path, monkeypatch):
    from multibench import workflow
    scored = []
    monkeypatch.setattr(workflow, "_score_record", lambda *a, **k: scored.append(a))
    res = mtb.load_run(_saved_tree(tmp_path))
    with pytest.raises(ValueError, match=r"^batch has 2,000 entries for the 2,864 cells of "
                                         r"D11: give one batch per cell\.$"):
        _quiet(res.rescore, batch=np.repeat(["a", "b"], 1000))
    assert not scored


# --------------------------------------------------------------- E06
def test_rescore_without_outputs_names_the_methods_and_the_fix(tmp_path):
    folder = _saved_tree(tmp_path)
    (folder / "Matilda_D11" / "embedding.h5").unlink()
    with pytest.raises(FileNotFoundError) as e:
        _quiet(mtb.load_run(folder).rescore)
    assert str(e.value) == (f"No output of Matilda in {folder}: load the folder that holds "
                            f"its method folder, or run it again.")


# --------------------------------------------------------------- E02
def test_the_embedding_of_a_one_batch_dataset_writes_h5ad(tmp_path):
    emb = mtb.load_run(_saved_tree(tmp_path)).embedding("Matilda")
    assert "batch" not in emb.obs
    emb.write_h5ad(tmp_path / "e.h5ad")


# --------------------------------------------------------------- E05
@pytest.mark.parametrize("ds,category,method", [("D11", "vertical", "Matilda"),
                                                ("D45", "mosaic", "Cobolt"),
                                                ("D52", "cross", "StabMap")])
def test_dataset_labels_follow_reordered_cells(ds, category, method):
    """Also where barcodes repeat across batches (D45, D52): the folder's cells are
    named as RunResult.embedding names them, cty2:<barcode>."""
    from multibench import workflow
    from multibench.eval.pipeline import _dataset_labels, _stack_label_files
    root = mtb.config.data_path
    files = _quiet(_dataset_labels, ds, category, method, None)
    files = list(files.values()) if isinstance(files, dict) else list(files)
    n = sum(len(pd.read_csv(f)) for f in files)
    obs = _quiet(workflow._embedding_obs, {"method": method, "modalities": None}, ds,
                 category, root, n)
    perm = np.random.default_rng(1).permutation(n)
    vals, _ = _stack_label_files(files, list(obs.index[perm]))
    assert (np.asarray(vals) == obs["label"].to_numpy()[perm]).all()


def test_labels_without_ids_next_to_barcodes_name_two_fixes(tmp_path):
    ad = pytest.importorskip("anndata")
    a = ad.AnnData(np.random.default_rng(0).normal(size=(30, 4)),
                   obs=pd.DataFrame({"ct": ["T", "B", "NK"] * 10},
                                    index=[f"AAAC{i}-1" for i in range(30)]))
    a.obs[["ct"]].to_csv(tmp_path / "l.csv", index=False)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(a, labels=str(tmp_path / "l.csv"), obsm="X", metrics=["ASW"])
    assert str(e.value) == ("l.csv has no cell ids, and the embedding has: put the barcodes "
                            "in its first column, or pass labels=pd.read_csv(...).iloc[:, 0]"
                            ".to_numpy() to match by position.")
    with pytest.raises(ValueError, match=r"^labels is a Series without cell ids, and the "
                                         r"embedding has: index it by the barcodes, or pass "
                                         r"labels\.to_numpy\(\) to match by position\.$"):
        mtb.evaluate(a, labels=a.obs["ct"].reset_index(drop=True), obsm="X", metrics=["ASW"])
    by_position = pd.read_csv(tmp_path / "l.csv").iloc[:, 0].to_numpy()
    assert _quiet(mtb.evaluate, a, labels=by_position, obsm="X", metrics=["ASW"]).equals(
        _quiet(mtb.evaluate, a, labels="ct", obsm="X", metrics=["ASW"]))


# --------------------------------------------------------------- E08
def test_a_batch_with_every_cell_and_some_others_is_matched_by_id():
    ad = pytest.importorskip("anndata")
    ids = [f"AAAC{i}-1" for i in range(30)]
    a = ad.AnnData(np.random.default_rng(0).normal(size=(30, 4)),
                   obs=pd.DataFrame({"ct": ["T", "B", "NK"] * 10, "donor": ["a", "b"] * 15},
                                    index=ids))
    more = pd.concat([a.obs["donor"], pd.Series(["c"] * 3, index=["X0", "X1", "X2"])])
    kw = dict(labels="ct", obsm="X", metrics=["iLISI"])
    assert _quiet(mtb.evaluate, a, batch=more.iloc[::-1], **kw).equals(
        _quiet(mtb.evaluate, a, batch="donor", **kw))
    with pytest.raises(ValueError, match=r"^batch has 27 entries for 30 cells in the "
                                         r"embedding: give one batch per cell\.$"):
        mtb.evaluate(a.X, labels=a.obs["ct"].to_numpy(), batch=list(a.obs["donor"])[:-3],
                     metrics=["iLISI"])


# --------------------------------------------------------------- scan and run texts
def _folder(root, name, files, n=60):
    """A dataset folder of ``n`` cells per file: {file: kind} with kind rna, gas or peak."""
    from tests.test_scan_held import _h5
    feats = {"rna": [f"GENE{i}" for i in range(40)], "gas": [f"GENE{i}" for i in range(40)],
             "peak": [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(40)]}
    d = root / name
    d.mkdir(parents=True)
    for f, kind in files.items():
        if f.endswith(".csv"):
            pd.DataFrame({"x": ["A", "B"] * (n // 2)}).to_csv(d / f, index=False)
        else:
            _h5(d / f, feats[kind], [f"{f[:4]}{i}" for i in range(n)])
    return d


def test_e07_paired_files_count_their_cells_once(tmp_path, capsys):
    d = _folder(tmp_path, "MULTI", {"rna.h5": "rna", "atac.h5": "peak", "cty.csv": ""}, n=600)
    plan = mtb.scan(str(d), "vertical", methods=["MIRA"], verbose=False)
    assert "MIRA needs about 3,000 cells, and this folder has 600: leave MIRA out." in \
        plan["caveat"].iloc[0]


def test_c13_e20_numbered_diagonal_folders(tmp_path, capsys):
    two = {f"{m}{i}.{e}": k for i in (1, 2)
           for m, e, k in (("rna", "h5", "rna"), ("atac_gas", "h5", "gas"),
                           ("rna_cty", "csv", ""), ("atac_cty", "csv", ""))}
    mtb.scan(str(_folder(tmp_path, "DIAG2", two)), "diagonal")
    out = capsys.readouterr().out
    assert "DIAG2 holds numbered batches (rna1.h5, rna2.h5, ...)." in out
    assert "Conos reads" not in out                        # C13: the folder line says it
    three = dict(two, **{f"{m}3.{e}": k for m, e, k in (
        ("rna", "h5", "rna"), ("atac_gas", "h5", "gas"), ("rna_cty", "csv", ""),
        ("atac_cty", "csv", ""))})
    df = mtb.scan(str(_folder(tmp_path, "DIAG3", three)), "diagonal")
    out = capsys.readouterr().out
    # E20: GLUE's variant for three batches speaks for it
    assert "[scan] GLUE: GLUE needs peak ATAC (atac_peak1.h5, atac_peak2.h5, atac_peak3.h5)" in out
    assert df.set_index("method").at["GLUE", "modalities"].startswith("rna1+rna2+rna3")


def test_c55_no_sentence_on_the_rows_of_a_modality_the_folder_lacks(capsys):
    mtb.scan("D11", "vertical")
    mtb.scan("D11", "vertical", methods=["Matilda", "totalVI"])
    assert "not listed" not in capsys.readouterr().out


def test_c22_one_sentence_for_the_missing_files_of_one_kind(tmp_path):
    d = _folder(tmp_path, "CROSS2", {"rna2.h5": "rna", "atac_peak2.h5": "peak",
                                     "cty2.csv": ""})
    reason = _quiet(mtb.scan, str(d), "cross", methods=["StabMap"]).at[0, "reason"]
    assert reason.count("which the folder does not have") <= 1
    assert "StabMap needs peak ATAC (atac_peak1.h5, atac_peak3.h5), which the folder does " \
           "not have." in reason, reason


# --------------------------------------------------------------- E13 / E14
def test_e13_e14_a_benchmark_dataset_gives_its_category_and_its_cells():
    ad = pytest.importorskip("anndata")
    n = len(pd.read_csv(mtb.config.data_path / "D11" / "cty.csv"))
    a = ad.AnnData(np.random.default_rng(0).normal(size=(n, 4)),
                   obs=pd.DataFrame({"ct": ["A", "B"] * (n // 2)}))
    got = _quiet(mtb.evaluate, a, labels="ct", obsm="X", dataset="D11", method="mine",
                 metrics=["cLISI"])
    assert set(got["category"]) == {"vertical"}
    with pytest.raises(ValueError, match=r"^The label files of D11 \(cty\.csv\) hold 2,864 "
                                         r"cells; the embedding has 1,000 rows\."):
        _quiet(mtb.evaluate, a[:1000].copy(), labels="ct", obsm="X", dataset="D11",
               metrics=["cLISI"])


def test_c20_your_rows_have_no_runtime_of_the_package():
    df = pd.concat([mtb.load_results("vertical"), _this_run("vertical")])
    r = _quiet(mtb.recommend, "vertical", long_df=df).set_index("method")
    own = [m for m in r.index if m.endswith("(this run)")]
    assert own and r.loc[own, "worst_sec"].isna().all()
    assert r.loc[own[0].removesuffix(" (this run)"), "worst_sec"] > 0


# --------------------------------------------------------------- C43
@needs_docs
def test_c43_the_labels_recipe_follows_the_package_order():
    """The recipe of evaluate.md, run on every fetched dataset folder, gives the
    package's own default order (labels_for without a method) in each category,
    and labels the obs of an AnnData with barcodes without a NaN."""
    import re
    from pathlib import Path
    from multibench.engine import resolve
    ad = pytest.importorskip("anndata")
    page = _read("tutorials/evaluate.md")
    block = next(b for b in re.findall(r"```python\n(.*?)```", page, re.S) if "order = {" in b)
    code = "\n".join(line for line in block.splitlines() if "fetch(" not in line)
    root = mtb.config.data_path
    cats = mtb.catalog.datasets().set_index("dataset")["category"]
    seen = set()
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and any(p.glob("*cty*.csv"))):
        category = cats.get(folder.name)
        if category not in CATS:
            continue
        ns = {"pd": pd, "folder": folder}
        exec(code, ns)
        want = list(_quiet(resolve.labels_for, folder.name, category, None, data_path=root,
                           check=False).values())
        assert [f.name for f in ns["files"]] == [Path(p).name for p in want], folder.name
        expected = np.concatenate([pd.read_csv(p).iloc[:, 0].to_numpy() for p in want])
        assert (ns["labels"] == expected).all(), folder.name
        a = ad.AnnData(obs=pd.DataFrame(index=[f"AAAC{i}-1" for i in range(len(expected))]))
        a.obs["celltype"] = ns["labels"]
        assert a.obs["celltype"].notna().all(), folder.name
        seen.add(category)
    assert seen == set(CATS)
