"""Published tables kept one level down (cross/D56 MOFA2/filtered3/metric.csv)
are loadable and listed (Elena: D56 was in the wheel but invisible to
available_datasets, and the FileNotFoundError named it as available)."""
import warnings

import pytest

import multibench as mtb
from multibench.data import results


def test_d56_is_visible_everywhere(result_dir):
    assert "D56" in results.available_datasets("cross", result_path=result_dir)
    df = results.load_results("cross", dataset="D56", result_path=result_dir)
    assert set(df.method) == {"MOFA2"} and (df.clustering == "default").all()
    assert abs(float(df[df.metric == "ARI"].value.iloc[0]) - 0.285746391962993) < 1e-9
    km = results.load_results("cross", dataset="D56", clustering="kmeans", result_path=result_dir)
    assert set(km.method) == {"MOFA2"} and (km.clustering == "kmeans").all()
    lo = results.load_results("cross", dataset="D56", clustering="louvain", result_path=result_dir)
    assert set(lo.method) == {"MOFA2"}
    cov = results.results_coverage("cross", source="published", result_path=result_dir)
    assert set(cov[cov.dataset == "D56"].clustering) == {"default", "kmeans", "louvain"}
    cat = mtb.data.catalog.datasets()
    row = cat[cat.dataset == "D56"].iloc[0]
    assert row.has_results and "cross" in str(row.category)


def test_nested_tables_for_d53_and_d57_are_read_too(result_dir):
    d57 = results.load_results("cross", dataset="D57", result_path=result_dir)
    assert {"MOFA2", "UINMF"} <= set(d57.method)
    d53 = results.load_results("cross", dataset="D53", result_path=result_dir)
    assert "MOFA2" in set(d53.method)
    # raw kBET folders are not metric tables: no dataset/method appears because of one
    assert not any(m.lower() == "kbet" for m in d53.method.unique())


def test_available_list_matches_the_error_message(result_dir):
    avail = results.available_datasets("cross", result_path=result_dir)
    with pytest.raises(FileNotFoundError) as ei:
        results.load_results("cross", dataset="D99", result_path=result_dir)
    msg = str(ei.value)
    have = eval(msg.split("datasets with published tables: ")[1])
    assert have == avail
    assert "D56" in have


def test_find_metric_file_prefers_direct_and_warns_on_ambiguity(tmp_path):
    m = tmp_path / "M"; m.mkdir()
    assert results._find_metric_file(m, "metric.csv") is None
    (m / "kbet").mkdir(); (m / "kbet" / "metric.csv").write_text(",Value\nARI,1\n")
    assert results._find_metric_file(m, "metric.csv") is None          # kbet is skipped
    (m / "a").mkdir(); (m / "a" / "metric.csv").write_text(",Value\nARI,1\n")
    assert results._find_metric_file(m, "metric.csv") == m / "a" / "metric.csv"
    (m / "b").mkdir(); (m / "b" / "metric.csv").write_text(",Value\nARI,2\n")
    with pytest.warns(UserWarning, match="2 nested metric.csv"):
        assert results._find_metric_file(m, "metric.csv") == m / "a" / "metric.csv"
    (m / "metric.csv").write_text(",Value\nARI,3\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert results._find_metric_file(m, "metric.csv") == m / "metric.csv"


def test_nested_layout_end_to_end_on_a_synthetic_tree(tmp_path):
    root = tmp_path / "scib_metric" / "cross integration" / "D99" / "MOFA2" / "filtered3"
    root.mkdir(parents=True)
    (root / "metric.csv").write_text(",Value\nARI,0.5\nNMI,0.6\n")
    (root / "metric_asw_iasw_if1.csv").write_text(",Value\nASW,0.7\n")
    (root.parent / "kbet").mkdir()
    (root.parent / "kbet" / "benchmark_results111.csv").write_text("a,b\n1,2\n")
    assert results.available_datasets("cross", result_path=tmp_path) == ["D99"]
    df = results.load_results("cross", dataset="D99", result_path=tmp_path)
    assert set(df.method) == {"MOFA2"} and set(df.metric) == {"ARI", "NMI"}
