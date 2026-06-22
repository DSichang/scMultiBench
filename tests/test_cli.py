from multibench import cli


def test_cli_list(capsys):
    rc = cli.main(["list", "--category", "diagonal"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SCALEX" in out


def test_cli_find_needs_labels(capsys):
    rc = cli.main(["find", "--category", "diagonal", "--needs-labels"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "scBridge" in out
    assert "SCALEX" not in out


def test_cli_plot_bubble(tmp_path):
    import matplotlib; matplotlib.use("Agg")
    out = tmp_path / "fig.png"
    rc = cli.main(["plot", "bubble", "--category", "diagonal", "--dataset", "D27",
                   "--metrics", "ARI,NMI", "--out", str(out)])
    assert rc == 0
    assert out.exists()


def test_cli_run_builds_and_calls(monkeypatch, tmp_path):
    from multibench import cli
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        class R: output = None; out_dir = tmp_path; cmd = ["conda"]
        return R()

    import multibench
    monkeypatch.setattr(multibench, "run", fake_run)
    rc = cli.main(["run", "--method", "SCALEX", "--category", "diagonal",
                   "--input", "rna=a.h5", "--input", "atac_gas=b.h5",
                   "--out", str(tmp_path / "o")])
    assert rc == 0
    assert captured["method"] == "SCALEX"
    assert captured["inputs"] == {"rna": "a.h5", "atac_gas": "b.h5"}
