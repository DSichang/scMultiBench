"""Registry data model: MethodSpec / Variant / ArgSpec / OutputSpec."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePath

#: values of ``MethodSpec.availability`` (see that property)
AVAILABILITY = ("public", "benchmark-host-only")


class AmbiguousVariantError(ValueError, KeyError):
    """A method has several variants that satisfy the selection and the caller
    must say which one (``category=`` / ``modalities=``).

    It is a ``ValueError`` - the package reserves ``KeyError`` for unknown ids
    (a typo in a method name) - but it still derives from ``KeyError`` so code
    written against the earlier ``params_for`` / ``inputs_for`` contract keeps
    catching it. ``str(exc)`` is the plain message (no ``KeyError`` quoting).
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""

# Roles that are not modalities: passed through verbatim by the runner and
# never counted when deriving what a variant consumes. (Mirrors runner._AUX_ROLES;
# kept here so the data model does not import the runner.)
_NON_MODALITY_ROLES = {"data_dir", "source_data", "target_data", "out_dir",
                       "reference", "batch_num", "num"}


def is_label_role(role: str) -> bool:
    """True for a cell-type-label role (``cty``, ``rna_cty``, ``cty1``, ``label``...).

    This is the ONE predicate shared by the runner (which treats these roles as
    auxiliary, never converting them to .h5), the resolver (which looks for
    ``.csv``) and ``MethodSpec.needs_labels`` - so "needs labels" means exactly
    "the runner will demand a label file".
    """
    return ("cty" in role) or ("label" in role)


def base_modality(role: str) -> str:
    """Map an arg role to its base modality type (``rna``/``adt``/``atac``/...).

    ``atac_gas`` / ``atac_peak`` -> ``atac`` (explicit representation prefix);
    ``rna1`` / ``atac2`` -> ``rna`` / ``atac`` (one trailing batch digit stripped);
    anything else is returned unchanged.
    """
    for prefix in ("rna", "adt", "atac"):
        if role.startswith(prefix + "_"):
            return prefix
    if role and role[-1].isdigit():
        return role[:-1]
    return role


def modality_family(role: str) -> str:
    """Collapse the ATAC representation roles onto their base token.

    ``atac_gas`` / ``atac_peak`` -> ``atac``; ``atac_gas2`` -> ``atac2`` (the
    batch digit is kept, so mosaic variants stay distinct); every other token
    (``rna``, ``adt1``, ``atac``, ``atac3``) is returned unchanged. This is the
    equivalence ``inputs_for`` / ``params_for`` use so that a caller may say
    ``modalities=['rna', 'atac']`` for a variant declared as
    ``[rna, atac_gas]`` - the ATAC file on disk is the same ``atac.h5`` either
    way, and the representation the method wants is ``method_info(m)['atac']``,
    not the role name.
    """
    for rep in ("atac_gas", "atac_peak"):
        if role == rep:
            return "atac"
        if role.startswith(rep) and role[len(rep):].isdigit():
            return "atac" + role[len(rep):]
    return role


#: the per-method keys of the GPU/CPU contract in methods.yaml (see MethodSpec)
GPU_FIELDS = ("cpu_params", "requires_gpu", "gpu_evidence")
#: what a ``cpu_params`` value may be: a command-line token (``""`` included -
#: scJoint's ``--use_cuda ""``), a number, or ``True`` for a bare flag
_CPU_PARAM_VALUE_TYPES = (str, int, float, bool)


def validate_gpu_fields(method_id: str, cpu_params, requires_gpu, gpu_evidence) -> None:
    """Check the GPU/CPU contract of one method (``methods.yaml`` keys).

    Parameters
    ----------
    method_id : str
        The method the keys belong to (named in every message).
    cpu_params : the ``cpu_params`` value - a mapping ``{flag: value}`` whose
        keys are non-empty strings and whose values are strings (``""``
        allowed), numbers or ``True``.
    requires_gpu : the ``requires_gpu`` value - a bool.
    gpu_evidence : the ``gpu_evidence`` value - a ``"<file>:<line>"`` string
        naming the unconditional CUDA call; required when ``requires_gpu``
        is True and forbidden otherwise.

    Raises
    ------
    ValueError
        With ``"<method>: ..."`` naming the offending key, when a value has
        the wrong shape, when ``requires_gpu`` and ``cpu_params`` are both
        set (a script cannot both need a GPU and have a switch that turns
        CUDA off), or when ``gpu_evidence`` and ``requires_gpu`` disagree.
    """
    if not isinstance(cpu_params, dict):
        raise ValueError(
            f"{method_id}: cpu_params must be a mapping {{flag: value}}, got "
            f"{type(cpu_params).__name__}")
    for k, v in cpu_params.items():
        if not isinstance(k, str) or not k:
            raise ValueError(
                f"{method_id}: cpu_params keys must be non-empty command-line "
                f"param names, got {k!r}")
        if v is None or v is False or not isinstance(v, _CPU_PARAM_VALUE_TYPES):
            raise ValueError(
                f"{method_id}: cpu_params[{k!r}] must be a command-line value "
                f"(a string, '' included, a number, or True for a bare flag) - "
                f"None/False are never emitted, so they cannot turn CUDA off; "
                f"got {v!r}")
    if not isinstance(requires_gpu, bool):
        raise ValueError(
            f"{method_id}: requires_gpu must be true or false, got {requires_gpu!r}")
    if not isinstance(gpu_evidence, str):
        raise ValueError(
            f"{method_id}: gpu_evidence must be a '<file>:<line>' string, got "
            f"{gpu_evidence!r}")
    if requires_gpu and cpu_params:
        raise ValueError(
            f"{method_id}: requires_gpu and cpu_params are both set - a script "
            f"that has a switch turning CUDA off does not require a GPU; keep one")
    if requires_gpu and not gpu_evidence.strip():
        raise ValueError(
            f"{method_id}: requires_gpu is true but gpu_evidence is empty - name "
            f"the unconditional CUDA call as '<file>:<line>' (it is quoted in "
            f"the refusal the runner raises on a host without a GPU)")
    if gpu_evidence.strip() and not requires_gpu:
        raise ValueError(
            f"{method_id}: gpu_evidence {gpu_evidence!r} is set but requires_gpu "
            f"is false - remove it or set requires_gpu: true")
    if requires_gpu:
        path, sep, line = gpu_evidence.strip().rpartition(":")
        if not sep or not path or not line.isdigit():
            raise ValueError(
                f"{method_id}: gpu_evidence must be '<file>:<line>', got "
                f"{gpu_evidence!r}")


@dataclass
class ArgSpec:
    role: str = ""          # input role (e.g. 'rna') or 'out_dir'
    flag: str | None = None # None -> positional
    # Collect SEVERAL input roles under ONE flag, in order, e.g.
    #   {roles: [rna1, rna2, rna3], flag: "--path1"}  ->  --path1 r1.h5 r2.h5 r3.h5
    # This is how the upstream scripts take multi-batch (cross) input; their
    # argparse declares nargs='+' on the path arguments.
    # An entry written "=VALUE" emits VALUE literally instead of resolving an
    # input role, e.g. {roles: [adt1, "=None", "=None"], flag: "--path2"} ->
    # --path2 adt1.h5 None None. Upstream scripts that take one slot per batch
    # need this to mark batches lacking a modality (scMoMaT documents `None`).
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
    slice_obs: list = field(default_factory=list)   # obs columns EVERY slice of a data_dir must carry (GPSA reads obs['Ground_Truth']); checked by scan's file gate
    helpers: list = field(default_factory=list)     # local modules the entrypoint imports from its own dir that upstream does NOT ship (MIRA's logger.py); scan reports the script blocked when one is absent

    def matches(self, category: str, modalities: set[str]) -> bool:
        return (self.when.get("category") == category
                and set(self.when.get("modalities", [])) == set(modalities))

    def roles(self) -> list[str]:
        """Every input role this variant resolves, in argument order.

        Includes single (``a.role``) and grouped (``a.roles``) roles; excludes
        ``out_dir``, ``"=LITERAL"`` placeholders and empty entries. ``const`` args
        are kept (scBridge's label files are const bare names, but they are still
        inputs the script reads).
        """
        out: list[str] = []
        for a in self.args:
            for r in (a.roles or [a.role]):
                r = str(r) if r is not None else ""
                if not r or r.startswith("=") or r == "out_dir":
                    continue
                out.append(r)
        return out

    @property
    def needs_labels(self) -> bool:
        """True when a cell-type-label role is among this variant's inputs.

        Derived from the declared roles with the runner's own label predicate
        (:func:`is_label_role`), so it can never disagree with what ``run`` will
        actually demand.
        """
        return any(is_label_role(r) for r in self.roles())

    @property
    def modality_types(self) -> set[str]:
        """Base modality types (``rna``/``adt``/``atac``) this variant consumes.

        Derived from the input roles, plus the ``const`` bare filenames on the
        ``source_data`` / ``target_data`` roles: scBridge takes the dataset
        DIRECTORY and names its matrices ``rna.h5`` / ``atac_gas.h5`` as
        constants, so its only resolved role is ``data_dir`` and the role-based
        answer alone would be empty (mirrors :attr:`consumes_atac`).
        """
        out = {base_modality(r) for r in self.roles()
               if r not in _NON_MODALITY_ROLES and not is_label_role(r)}
        for a in self.args:
            if a.const and a.role in ("source_data", "target_data"):
                stem = PurePath(str(a.const)).stem
                if stem and not is_label_role(stem):
                    out.add(base_modality(stem))
        return out

    @property
    def takes_data_dir(self) -> bool:
        """True when this variant is fed a DIRECTORY (a ``data_dir`` role) -
        the spatial-registration methods and scBridge - rather than one file
        per modality. Such variants declare ``when.modalities: []``."""
        return any(a.role == "data_dir" for a in self.args)

    @property
    def modalities_unknown(self) -> bool:
        """True when nothing in the declaration says which modalities this
        variant consumes: it takes a directory and no bare filename names a
        matrix (SPIRAL/GPSA/PASTE/PASTE2, whose slices are ``.h5ad`` files).
        ``find_methods(modalities=...)`` keeps such variants and warns rather
        than silently dropping them."""
        return self.takes_data_dir and not self.modality_types

    @property
    def consumes_atac(self) -> bool:
        """True when THIS variant takes an ATAC input (an ``atac*`` role, or a
        ``const`` bare filename naming an atac file - scBridge's
        ``atac_gas.h5``). The per-variant half of ``MethodSpec.consumes_atac``,
        so ``find_methods(atac=...)`` can judge each variant on its own."""
        if "atac" in self.modality_types:
            return True
        return any(a.const and "atac" in str(a.const) for a in self.args)

    @property
    def is_public(self) -> bool:
        """True when this variant's entrypoint is a repo-relative path (a file
        the public scMultiBench checkout can supply). An ABSOLUTE entrypoint
        names one machine's filesystem - the benchmark host - and no download
        can provide it; see ``MethodSpec.availability``."""
        return not PurePath(self.entrypoint).is_absolute()


@dataclass
class MethodSpec:
    id: str
    language: str
    # Categories this method is WIRED for. Left empty by registry.load() and
    # derived in __post_init__ from the variants' `when.category` (methods.yaml
    # may not carry a hand `categories:` key: Multigrate/totalVI/sciPENN drifted
    # from their variants, so list_methods(category=) disagreed with scan/
    # find_methods). A value passed explicitly (tests, ad-hoc specs) is kept.
    categories: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    # ATAC representation the UPSTREAM method expects: "peak" | "gene_activity"
    # | None. This is deliberately an EXPLICIT key, not derived from role names:
    # moETM/scMM/iPOLNG declare role `atac_gas` (a resolver alias for atac.h5)
    # yet consume peak matrices. registry.load() refuses a spec that consumes
    # atac without it (see MethodSpec.consumes_atac).
    atac: str | None = None
    setup_hint: str = ""
    status: str = "declared"            # declared | verified
    variants: list[Variant] = field(default_factory=list)
    # reproducible environment recipe (see multibench.engine.envs); keys:
    # python_version, conda_channels, conda_packages, pip_packages, pip_git,
    # package_source, difficulty, verified_working, caveats
    env_spec: dict = field(default_factory=dict)
    # curated provenance (engine/references.yaml): repo_url, version, summary,
    # reference {doi, title, authors, journal, year}
    reference: dict = field(default_factory=dict)
    # GPU/CPU contract of the UPSTREAM script (per method, not per variant;
    # see `GPU_FIELDS`). `cpu_params` are the command-line params that turn
    # CUDA OFF for a script that has it on by default ({flag: value}, emitted
    # exactly like `params`: scJoint's argparse `--use_cuda` is `type=bool`,
    # so only the EMPTY string is false -> {use_cuda: ""}). The runner merges
    # them into a run's params when `envs.host_has_gpu()` is False, unless
    # the caller passed the same key. `requires_gpu` marks a script that
    # calls CUDA unconditionally (`.cuda()` / `torch.device('cuda')` with no
    # switch and no `torch.cuda.is_available()` fallback); `gpu_evidence` is
    # the `file:line` of that call, quoted in the refusal. A method may not
    # carry both.
    cpu_params: dict = field(default_factory=dict)
    requires_gpu: bool = False
    gpu_evidence: str = ""

    def __post_init__(self):
        if not self.categories:
            self.categories = self.wired_categories
        validate_gpu_fields(self.id, self.cpu_params, self.requires_gpu,
                            self.gpu_evidence)

    @property
    def requires_gpu_reason(self) -> str:
        """Why this method cannot run on a host without an NVIDIA GPU, or
        ``""`` when ``requires_gpu`` is False.

        The ONE text the runner's ``OSError`` and ``scan``'s ``env_reason``
        share, so a tutorial and a traceback read the same sentence::

            scBridge needs an NVIDIA GPU: the upstream script calls CUDA
            unconditionally (tools_scripts/scBridge/main_scBridge.py:46);
            see method_info(m)["requires_gpu"]
        """
        if not self.requires_gpu:
            return ""
        return (f"{self.id} needs an NVIDIA GPU: the upstream script calls CUDA "
                f"unconditionally ({self.gpu_evidence}); "
                f'see method_info(m)["requires_gpu"]')

    @property
    def wired_categories(self) -> list[str]:
        """The distinct ``when.category`` of the variants, in declaration order
        - what ``scan`` / ``run_all`` / ``find_methods`` can actually dispatch.
        ``categories`` equals this for every registry-loaded spec."""
        return [c for c in dict.fromkeys(v.when.get("category") for v in self.variants)
                if c]

    @property
    def needs_labels(self) -> bool:
        """True when ANY variant takes a cell-type-label role as input.

        Derived (not declared): ``any(v.needs_labels for v in self.variants)``.
        Per-variant detail is on ``Variant.needs_labels`` (scMoMaT, for example,
        needs labels only in its mosaic variant). methods.yaml may not carry a
        hand-written ``needs_labels`` key any more - registry.load() rejects it.
        """
        return any(v.needs_labels for v in self.variants)

    @property
    def modality_types(self) -> set[str]:
        """Union of base modality types consumed across all variants."""
        out: set[str] = set()
        for v in self.variants:
            out |= v.modality_types
        return out

    @property
    def consumes_atac(self) -> bool:
        """True when some variant takes an ATAC input.

        Counts ``atac*`` roles AND ``const`` file names (scBridge passes
        ``atac_gas.h5`` as a const bare filename rather than a resolved role).
        """
        return any(v.consumes_atac for v in self.variants)

    @property
    def availability(self) -> str:
        """Where this method can actually be run: one of :data:`AVAILABILITY`.

        * ``"public"`` - every variant's entrypoint is a path inside the public
          scMultiBench repository (``tools_scripts/...``), which ``run`` fetches
          on first use; a public install can execute it.
        * ``"benchmark-host-only"`` - at least one variant's entrypoint is an
          ABSOLUTE path on the machine the benchmark was produced on (SPIRAL's
          and GPSA's working scripts are not published), so the script cannot
          be fetched and the method cannot run from a public install - whatever
          ``status`` says about its verification there.

        Derived from the entrypoints (no hand-maintained flag): the rule is
        "any variant entrypoint is absolute". Stubs without variants are
        ``"public"`` (nothing machine-specific is declared).
        """
        return ("benchmark-host-only"
                if any(not v.is_public for v in self.variants) else "public")

    def select(self, category: str, modalities: set[str], *,
               loose: bool = False) -> Variant:
        """The variant declared for ``(category, modalities)``.

        Parameters
        ----------
        category : ``vertical`` / ``diagonal`` / ``mosaic`` / ``cross``.
        modalities : the variant's modality tokens as a set (``set()`` for the
            ``data_dir`` variants).
        loose : keyword-only, default ``False`` (exact token match, as before).
            ``True`` additionally accepts the ATAC representation roles under
            their base name - ``{'rna', 'atac'}`` selects a variant declared
            ``[rna, atac_gas]`` or ``[rna, atac_peak]`` (see
            :func:`modality_family`) - when exactly ONE variant of the category
            matches that way.

        Returns
        -------
        Variant

        Raises
        ------
        KeyError
            No variant matches (the message lists the declared
            ``(category, modalities)`` pairs).
        AmbiguousVariantError
            ``loose=True`` and several variants match under the family rule
            (e.g. one ``atac_gas`` and one ``atac_peak`` variant in the same
            category) - pass the exact role tokens.
        """
        for v in self.variants:
            if v.matches(category, modalities):
                return v
        if loose:
            want = {modality_family(m) for m in modalities}
            hits = [v for v in self.variants
                    if v.when.get("category") == category
                    and {modality_family(m) for m in v.when.get("modalities", [])} == want]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise AmbiguousVariantError(
                    f"{self.id}: modalities={sorted(modalities)} match {len(hits)} "
                    f"{category!r} variants: {[v.when.get('modalities') for v in hits]}; "
                    f"pass the exact role tokens (method_info({self.id!r})['supports'])")
        raise KeyError(
            f"{self.id}: no variant for category={category!r} modalities={sorted(modalities)}; "
            f"available: {[(v.when.get('category'), v.when.get('modalities')) for v in self.variants]}"
        )
