# Setting this up on a new machine

Every number here was measured by building all 29 environments from scratch in an
isolated clean room, then running methods in them. Nothing is estimated.

## What it costs

| | measured |
|---|---|
| environments | **29** (40 methods share them; `scmb_r` alone serves 5) |
| disk, environments | **175 GB** |
| disk, conda package cache | **52 GB** (safe to delete afterwards: `conda clean -a`) |
| build wall-clock | **~50 min** total with a warm package cache; the first few envs dominate |
| slowest single env | `scmb_scmvp4` 286 s · `glue` 217 s · `MOFA2_env` 210 s |
| largest envs | `MOFA2_env` 18 GB · `unitednet` 12 GB · `scmb_torch_v2` 9.8 GB |

Plan for **~230 GB free** during the build and ~175 GB after cleaning the cache.

## The two commands

```bash
multibench env doctor        # what is needed, what is missing
multibench env install --run # build everything from the committed lockfiles
```

`env install` is a dry run until you add `--run`. `env doctor`'s legend distinguishes
installed `[x]`, missing-but-buildable `[L]`, and missing-with-no-lockfile `[!]`.
`multibench env plan [--category C | --methods M,...]` collapses the methods into the
envs they need and prints, per env, the packed-archive download size (`dl`) and the
unpacked size (`disk`) from the shipped snapshot `multibench/engine/packed_sizes.json`
(`?` = not measured yet; `python tools/packed_sizes.py` on a host with network fills
it) plus a `# total` line; `env install --packed` shows the same per archive. `env
status` tags each method's env with a build-difficulty word (`easy`, `old-scvi`,
`old-tensorflow`, `R`, `verified`, `blocked-script`; the legend is printed on stderr
and `mtb.env.DIFFICULTY` holds the definitions).

**The environments are linux-64 conda envs** - the lockfiles and the packed archives
both. On macOS / Windows `env status`, `plan`, `doctor` and `install` print one
`warning: ... linux-64 ...` line first and `env install --run` refuses (exit 1) unless
`--force`; dry runs work everywhere. Everything that does not run a method - the
registry, the stored results, `scan`'s file gate, `evaluate`, the figures - works on
any machine.

`scan()` checks two gates per method - `files_ok` (the dataset folder itself; this
gate runs on any machine, no environments needed) and `env_ok` (the method's conda
env exists here) - and marks a method runnable only when both pass, so a sweep
never starts something it cannot finish. `mtb.plan(dataset, category)` is the
`run_all` plan with the blocked rows and their reasons kept.

## Things worth knowing before you start

**These lockfiles are `name=version`, not explicit URLs.** They need a solver and access
to the conda/PyPI channels; they cannot be replayed against an offline mirror as-is. If
your compute nodes have no internet, build on a login node.

**Provisioning blocks user site-packages.** pip treats anything importable from
`~/.local/lib/pythonX/site-packages` as already satisfied and will skip installing it
into the target environment — producing an env that builds cleanly and then fails at
dispatch, because `run()` correctly sets `PYTHONNOUSERSITE=1`. Provisioning now sets the
same variable, so this is handled; it is documented because the failure looked like a
broken method rather than a provisioning gap.

**Several packages cannot come from a lockfile** and are restored by committed
`<env>.post.sh` scripts, because conda cannot capture them:

- `rliger` 2.0.1 in `scmb_r` — installed with `install.packages()`, invisible to conda.
  Without it UINMF fails with "there is no package called 'rliger'" *after* a clean build.
- `matilda` — the working env has it as an editable install from a local checkout.
  Restored from PyPI as `matilda-sc==0.2.1`. **The working env has 0.2.0**, which PyPI
  does not publish, and the checkout's git remote is private. It runs and reproduces the
  recorded ARI (0.924) but is not byte-identical.
- `scMVP` — installed from public upstream `bm2-lab/scMVP` at a pinned commit; the
  working copy lives outside this repository.

**One environment cannot be fully rebuilt from a fresh clone.** `spiral_environment`
installs from `multibench_codes/SPIRAL_latest`, which git does not track. Its
post-install fails with an explicit message naming the upstream repo rather than
producing a quietly broken environment.

## Verifying your install

```bash
python -m pytest tests/ -q          # green on a laptop with NO conda environments
                                    # (the dispatch tests fake the env gate; the
                                    # three host-count tests in tests/test_workflow.py
                                    # skip when no env is installed). It needs the
                                    # reference checkout the runner clones on first
                                    # use (scMultiBench_ref/, auto-provisioned).
```

Then the smallest real end-to-end check:

```python
import multibench as mtb
mtb.data.fetch("D11")                                      # 11 MB reference dataset
mtb.scan("D11", category="vertical")                       # files_ok 14 of 25 rows; env_ok per env built
mtb.run_all("D11", "vertical", methods=["Matilda"], out_dir="/tmp/check")
```

Matilda on D11 should reach `CHAIN_OK` with ARI ≈ 0.924 in well under a minute. If it
builds but fails to import its own package, the post-install did not run.
