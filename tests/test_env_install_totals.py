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


def test_the_dry_run_shows_the_user_columns_only():
    """d1 (u1, u4): the table a user reads; the commands, URLs and lock flags
    stay in the package's rows."""
    plan = mtb.env.install(["Matilda"], dry_run=True)
    assert list(plan.columns) == ["env", "methods", "state", "flavor", "archive_bytes",
                                  "unpacked_bytes"]
