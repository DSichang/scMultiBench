"""A params= value outside the argparse choices of the script is refused before a run
(core040 M-3)."""
import pytest

import multibench as mtb


def test_params_for_lists_the_choices():
    acc = mtb.params_for("MIRA", "vertical", ["rna", "atac"])["tunable"]["acc"]
    assert acc["default"] == "gpu" and "cpu" in acc["choices"]


def test_a_value_outside_the_choices_is_refused_in_the_dry_run():
    with pytest.raises(ValueError, match=r"^MIRA: acc takes cpu, gpu, .* or auto, got 'banana'\.$"):
        mtb.run("D27mini_vertical", "vertical", methods=["MIRA"],
                    params={"MIRA": {"acc": "banana"}}, dry_run=True, verbose=False)


def test_a_listed_value_passes():
    plan = mtb.run("D27mini_vertical", "vertical", methods=["MIRA"],
                       params={"MIRA": {"acc": "cpu"}}, dry_run=True, verbose=False)
    assert "--acc cpu" in plan["command"].iloc[0]


def test_params_for_has_one_default_per_key():
    p = mtb.params_for("Matilda", "vertical", ["rna", "adt"])
    assert "effective" not in p
    assert all(set(t) <= {"type", "default", "choices"} for t in p["tunable"].values())
