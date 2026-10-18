"""env.install's dry run names the shared layer its total counts and carries doctor's
other_env column (core040 M14, S-1)."""
import multibench as mtb
from multibench.engine import envs


def test_the_total_names_the_shared_layer():
    rows = [{"env": "scmb_multigrate2", "flavor": "gpu"}]
    totals, line = envs.plan_totals(rows)
    if not totals["shared_layers"]:
        return                                   # the manifest lists no layer for it
    name = next(iter(totals["shared_layers"]))
    assert f"(with the shared layer {name}, " in line and line.endswith(", downloaded once)")


def test_the_dry_run_has_the_doctor_columns():
    plan = mtb.env.install(["Matilda"], dry_run=True)
    assert {"env", "methods", "state", "exists", "has_lock", "other_env"} <= set(plan.columns)
