"""Registry data model: MethodSpec / Variant / ArgSpec / OutputSpec."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArgSpec:
    role: str = ""          # input role (e.g. 'rna') or 'out_dir'
    flag: str | None = None # None -> positional
    # Collect SEVERAL input roles under ONE flag, in order, e.g.
    #   {roles: [rna1, rna2, rna3], flag: "--path1"}  ->  --path1 r1.h5 r2.h5 r3.h5
    # This is how the upstream scripts take multi-batch (cross) input; their
    # argparse declares nargs='+' on the path arguments.
    roles: list = field(default_factory=list)
    repeat: bool = False    # value may be a list (nargs='+')
    eq: bool = False        # emit as --flag=value
    const: str | None = None  # literal value to emit, ignoring `values` (e.g. 'NULL')

    @property
    def is_positional(self) -> bool:
        return self.flag is None


@dataclass
class OutputSpec:
    kind: str               # embedding | imputed | labels | markers | coords
    file: str               # filename (or glob) written into out_dir
    dataset: str | None = None  # in-file dataset name for h5 outputs


@dataclass
class Variant:
    when: dict              # {'category': str, 'modalities': [str, ...]}
    entrypoint: str
    language: str
    args: list[ArgSpec]
    output: OutputSpec
    params: dict = field(default_factory=dict)        # default hyperparams ACTUALLY emitted on the command line
    tunable: dict = field(default_factory=dict)       # DOC-ONLY: hyperparams the upstream script accepts,
                                                      # {name: {default,type}}; NEVER emitted - surfaced by discover.params_for
    run_env: dict = field(default_factory=dict)        # per-method env overrides (e.g. CUDA_VISIBLE_DEVICES)
    cwd_at_script: bool = False        # if True, run with cwd=script's parent dir (for scripts that source/import local files)
    pty: bool = False                  # if True, run the command under a pseudo-tty (script -q -e -c ...), for scripts that read the terminal size (os.popen('stty size')) to draw a progress bar and crash without a tty (scJoint)
    driver: str | None = None          # package-relative R/py wrapper that source()s the (unmodified) upstream entrypoint then calls its function; see engine/drivers/
    normalize_peaks: list = field(default_factory=list)  # roles whose .h5 ATAC peak names get normalized to chr:start-end before the run
    extra_outputs: list[OutputSpec] = field(default_factory=list)

    def matches(self, category: str, modalities: set[str]) -> bool:
        return (self.when.get("category") == category
                and set(self.when.get("modalities", [])) == set(modalities))


@dataclass
class MethodSpec:
    id: str
    language: str
    categories: list[str]
    tasks: list[str]
    atac: str | None = None
    needs_labels: bool = False
    setup_hint: str = ""
    status: str = "declared"            # declared | verified
    variants: list[Variant] = field(default_factory=list)
    # reproducible environment recipe (see multibench.engine.envs); keys:
    # python_version, conda_channels, conda_packages, pip_packages, pip_git,
    # package_source, difficulty, verified_working, caveats
    env_spec: dict = field(default_factory=dict)

    def select(self, category: str, modalities: set[str]) -> Variant:
        for v in self.variants:
            if v.matches(category, modalities):
                return v
        raise KeyError(
            f"{self.id}: no variant for category={category!r} modalities={sorted(modalities)}; "
            f"available: {[(v.when.get('category'), v.when.get('modalities')) for v in self.variants]}"
        )
