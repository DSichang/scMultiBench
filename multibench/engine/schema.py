"""Registry data model: MethodSpec / Variant / ArgSpec / OutputSpec."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePath


def no_category_message(method: str, categories, category) -> str:
    """``"SCALEX does not run on vertical data. Its categories: diagonal."`` (internal)."""
    cats = ", ".join(str(c) for c in categories)
    return (f"{method} does not run on {category} data."
            + (f" Its categories: {cats}." if cats else ""))


def no_variant_message(method: str, modalities, category=None) -> str:
    """``"No vertical variant of totalVI reads atac and rna. ..."`` (internal)."""
    where = f"{category} " if category else ""
    mods = " and ".join(sorted(map(str, modalities)))
    return (f"No {where}variant of {method} reads {mods}. "
            f"mtb.method_info({method!r})['supports'] lists them.")


class AmbiguousVariantError(ValueError):
    """Several variants of a method fit; the message names the call that picks one."""


# input roles that hold no modality (label roles: is_label_role)
_NON_MODALITY_ROLES = {"data_dir", "source_data", "target_data", "out_dir",
                       "reference", "batch_num", "num"}


def is_label_role(role: str) -> bool:
    """True for a cell-type-label role (``cty``, ``rna_cty``, ``cty1``, ``label``...).

    The one predicate the runner, the resolver and ``needs_labels`` share.
    """
    return ("cty" in role) or ("label" in role)


def _batch_of(role: str) -> int | None:
    """Batch number of a numbered role (``rna3`` -> 3); ``None`` for an
    unnumbered one (``rna``, ``reference``)."""
    stem = role.rstrip("0123456789")
    return int(role[len(stem):]) if stem and stem != role else None


def base_modality(role: str) -> str:
    """Base modality of a role: ``atac_gas`` -> ``atac``, ``rna1`` -> ``rna``;
    anything else unchanged (one trailing batch digit is stripped)."""
    for prefix in ("rna", "adt", "atac"):
        if role.startswith(prefix + "_"):
            return prefix
    if role and role[-1].isdigit():
        return role[:-1]
    return role


def modality_family(role: str) -> str:
    """Collapse the ATAC representation roles onto their base token.

    ``atac_gas`` / ``atac_peak`` -> ``atac``; ``atac_gas2`` -> ``atac2`` (the
    batch digit is kept); every other token is returned unchanged. So
    ``modalities=['rna', 'atac']`` selects a variant declared ``[rna, atac_gas]``:
    both read the same ``atac.h5``.
    """
    for rep in ("atac_gas", "atac_peak"):
        if role == rep:
            return "atac"
        if role.startswith(rep) and role[len(rep):].isdigit():
            return "atac" + role[len(rep):]
    return role


@dataclass
class ArgSpec:
    """One command-line argument of a method script.

    Attributes
    ----------
    role : str
        The input role whose value is passed (``'rna'``), or ``'out_dir'``.
    flag : str or None
        The option (``'--path1'``); ``None`` for a positional argument.
    roles : list
        Several input roles under one flag, in order (``nargs='+'``); an entry
        ``"=VALUE"`` passes ``VALUE``: ``[adt1, "=None"]`` gives
        ``--path2 adt1.h5 None``.
    repeat : bool
        The value may be a list (``nargs='+'``).
    eq : bool
        Passed as ``--flag=value``.
    const : str or None
        A literal value passed instead of the role's (``'NULL'``).
    """
    role: str = ""
    flag: str | None = None
    roles: list = field(default_factory=list)
    repeat: bool = False
    eq: bool = False
    const: str | None = None

    @property
    def is_positional(self) -> bool:
        """True for an argument without a flag."""
        return self.flag is None


@dataclass
class OutputSpec:
    """A file a method script writes.

    Attributes
    ----------
    kind : str
        ``'embedding'``, ``'graph'`` or ``'labels'``.
    file : str
        Its name in ``out_dir``.
    dataset : str or None
        The dataset that holds it in an ``.h5`` output.
    cell_order : list, str or None
        The order of its rows when it is not argument order
        (:meth:`Variant.stacked_roles`).
    """
    kind: str
    file: str
    dataset: str | None = None
    cell_order: list | str | None = None


@dataclass
class Variant:
    """One way to run a method: a category, the modalities it reads and the
    script call.

    Attributes
    ----------
    when : dict
        ``{'category': str, 'modalities': [str, ...]}``.
    entrypoint : str
        The script, relative to the scripts folder.
    language : str
        ``'python'`` or ``'R'``.
    args : list of ArgSpec
        The script's arguments, in order.
    output : OutputSpec
        The result file.
    params : dict
        Default hyperparameters passed on the command line.
    tunable : dict
        ``{name: {default, type}}``: the parameters the script accepts; never
        passed unless given.
    run_env : dict
        Environment variables the method needs.
    cwd_at_script : bool
        Run in the script's folder (it imports local files).
    pty : bool
        Run under a pseudo-terminal (scJoint reads the terminal size).
    driver : str or None
        The package-side wrapper in ``engine/drivers/`` that runs the script.
    normalize_peaks : list
        Roles whose peak names are rewritten as ``chr:start-end``.
    writes_beside_inputs : bool
        The script writes files next to the files it reads (scMM, scMVP), so
        it reads links to them in the run's ``inputs/`` folder.
    extra_outputs : list of OutputSpec
        Other files the script must write.
    helpers : list
        Modules the scMultiBench repository does not ship, copied from
        ``engine/helpers/<method>/``.
    size_hint : str
        What the method needs from the data's size; starts with the method
        name. ``scan`` shows it only for data smaller than one of the next
        three sizes.
    min_cells : int or None
        Cells of the files the row reads, in total.
    min_cells_per_batch : int or None
        Cells of each numbered batch.
    min_genes : int or None
        Features of each RNA file.
    downloads : list
        ``[{file, url}]``: files the first run downloads next to the script.
    """
    when: dict
    entrypoint: str
    language: str
    args: list[ArgSpec]
    output: OutputSpec
    params: dict = field(default_factory=dict)
    tunable: dict = field(default_factory=dict)
    run_env: dict = field(default_factory=dict)
    cwd_at_script: bool = False
    pty: bool = False
    driver: str | None = None
    normalize_peaks: list = field(default_factory=list)
    writes_beside_inputs: bool = False
    extra_outputs: list[OutputSpec] = field(default_factory=list)
    helpers: list = field(default_factory=list)
    size_hint: str = ""
    min_cells: int | None = None
    min_cells_per_batch: int | None = None
    min_genes: int | None = None
    downloads: list = field(default_factory=list)

    @property
    def shown_modalities(self) -> list[str]:
        """The modalities read; for a variant that reads a folder (scBridge), the
        stems of the ``.h5`` files it names (``rna``, ``atac_gas``)."""
        return list(self.when.get("modalities", [])) or [
            PurePath(str(a.const)).stem for a in self.args
            if a.const and str(a.const).endswith(".h5")]

    def matches(self, category: str, modalities: set[str]) -> bool:
        """True when this variant is declared for ``category`` and exactly
        ``modalities``."""
        return (self.when.get("category") == category
                and set(self.when.get("modalities", [])) == set(modalities))

    def roles(self) -> list[str]:
        """Every input role this variant resolves, in argument order: no
        ``out_dir``, no ``"=LITERAL"`` entries; ``const`` args are kept."""
        out: list[str] = []
        for a in self.args:
            for r in (a.roles or [a.role]):
                r = str(r) if r is not None else ""
                if not r or r.startswith("=") or r == "out_dir":
                    continue
                out.append(r)
        return out

    def stacked_roles(self) -> list[str]:
        """The input roles in the order the output stacks their cells.

        Argument order (label roles and ``data_dir`` left out) unless
        ``output.cell_order`` declares another:

        * a list of roles - those first, in that order, then the rest in
          argument order (uniPort: ``[atac_gas, rna]``, its script
          concatenates the ATAC cells first);
        * ``"reference_first"`` - the roles of the batch the ``reference``
          const names as ``data<N>`` first, then the rest in argument order
          (StabMap: ``stabMap()`` stacks the reference dataset's cells before
          the others, and its util.R names the batches ``data1..dataN`` by
          slot, so slot ``k`` of each repeated flag must hold batch ``k``).
        """
        mods = [r for r in self.roles() if not is_label_role(r) and r != "data_dir"]
        order = self.output.cell_order
        if order is None:
            return mods
        if order == "reference_first":
            batch = self.reference_batch
            return ([r for r in mods if _batch_of(r) == batch]
                    + [r for r in mods if _batch_of(r) != batch])
        return list(order) + [r for r in mods if r not in order]

    @property
    def reference_batch(self) -> int | None:
        """Batch ``N`` of the ``reference`` const ``data<N>``, else ``None``."""
        ref = next((str(a.const) for a in self.args if a.role == "reference" and a.const), "")
        return int(ref[4:]) if ref.startswith("data") and ref[4:].isdigit() else None

    @property
    def needs_labels(self) -> bool:
        """True when a cell-type-label role is among this variant's inputs."""
        return any(is_label_role(r) for r in self.roles())

    @property
    def modality_types(self) -> set[str]:
        """Base modality types (``rna``/``adt``/``atac``) this variant consumes.

        Also counts the ``const`` file names of ``source_data`` /
        ``target_data`` (scBridge names its matrices that way). Another const
        role is a placeholder, not an input (Seurat_WNN's ``NULL`` atac).
        """
        placeholders = {a.role for a in self.args
                        if a.const and a.role not in ("source_data", "target_data")}
        out = {base_modality(r) for r in self.roles()
               if r not in _NON_MODALITY_ROLES and not is_label_role(r)
               and r not in placeholders}
        for a in self.args:
            if a.const and a.role in ("source_data", "target_data"):
                stem = PurePath(str(a.const)).stem
                if stem and not is_label_role(stem):
                    out.add(base_modality(stem))
        return out

    @property
    def takes_data_dir(self) -> bool:
        """True when this variant reads a folder (``data_dir``, scBridge), not
        one file per modality; it declares no modalities."""
        return any(a.role == "data_dir" for a in self.args)

    @property
    def consumes_atac(self) -> bool:
        """True when this variant takes an ATAC input (an ``atac*`` role, or a
        ``const`` file name such as scBridge's ``atac_gas.h5``)."""
        if "atac" in self.modality_types:
            return True
        return any(a.const and "atac" in str(a.const) for a in self.args)


#: the methods whose vertical ATAC role was ``atac_gas`` until 0.3 (it reads peaks)
_ATAC_GAS_RENAMED = frozenset({"scMM", "scMDC", "moETM", "UnitedNet", "iPOLNG"})


@dataclass
class MethodSpec:
    """One method of ``methods.yaml``.

    Attributes
    ----------
    id : str
        The registry id (``'totalVI'``).
    language : str
        ``'python'`` or ``'R'``.
    categories : list of str
        The categories it runs on; the variants' categories unless given.
    tasks : list of str
        The tasks it serves (``clustering``, ``batch``,
        ``dimension_reduction``).
    atac : str or None
        ``'peak'``, ``'gene_activity'`` or ``None``: the ATAC form it reads,
        declared because the role ``atac`` stands for either.
    setup_hint : str
        A setup fact ``scan`` shows before the run.
    variants : list of Variant
        The ways to run it.
    reference : dict
        Its provenance, from ``references.yaml``.
    cpu_params : dict
        The parameters that turn CUDA off, merged by
        ``runner.cpu_params_for`` on a host without a GPU.
    requires_gpu : bool
        The script calls CUDA unconditionally.
    gpu_evidence : str
        The ``file:line`` of that call.
    gpu_use : str
        How it uses a GPU when not required (``'used when present'``,
        ``'not used'``); see :attr:`gpu`.
    """
    id: str
    language: str
    categories: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    atac: str | None = None
    setup_hint: str = ""
    variants: list[Variant] = field(default_factory=list)
    reference: dict = field(default_factory=dict)
    cpu_params: dict = field(default_factory=dict)
    requires_gpu: bool = False
    gpu_evidence: str = ""
    gpu_use: str = ""

    def __post_init__(self):
        if not self.categories:
            self.categories = self.wired_categories

    @property
    def gpu(self) -> str:
        """How the method uses an NVIDIA GPU: ``required`` when
        ``requires_gpu``; else the declared ``gpu_use``; else ``used when
        present`` for a method with ``cpu_params``; else ``unknown``."""
        if self.requires_gpu:
            return "required"
        if self.gpu_use:
            return self.gpu_use
        if self.cpu_params:
            return "used when present"
        return "unknown"

    @property
    def requires_gpu_reason(self) -> str:
        """Why this method cannot run without an NVIDIA GPU (``""`` when
        ``requires_gpu`` is False); the runner's ``OSError`` and ``scan``'s
        ``env_reason`` both use it."""
        if not self.requires_gpu:
            return ""
        return f"{self.id} needs an NVIDIA GPU, and this computer has none."

    @property
    def wired_categories(self) -> list[str]:
        """The distinct ``when.category`` of the variants, in declaration order."""
        return [c for c in dict.fromkeys(v.when.get("category") for v in self.variants)
                if c]

    @property
    def needs_labels(self) -> bool:
        """True when any variant takes a cell-type-label role as input."""
        return any(v.needs_labels for v in self.variants)

    def legacy_modalities(self, category: str, modalities) -> set | None:
        """The 0.4 spelling of a 0.3 role set, else ``None``: the vertical
        variants of scMM, scMDC, moETM, UnitedNet and iPOLNG read ``atac.h5``
        (peaks) under the role ``atac`` since 0.4, ``atac_gas`` before (saved
        records and older ``inputs=`` keys still name it)."""
        mods = set(modalities)
        if self.id not in _ATAC_GAS_RENAMED or category != "vertical" \
                or "atac_gas" not in mods or "atac" in mods:
            return None
        new = (mods - {"atac_gas"}) | {"atac"}
        return new if any(v.matches(category, new) for v in self.variants) else None

    def select(self, category: str, modalities: set[str], *,
               loose: bool = False) -> Variant:
        """The variant declared for ``(category, modalities)``; ``loose=True``
        also accepts ``atac`` for ``atac_gas`` / ``atac_peak``
        (:func:`modality_family`). No match raises ``KeyError``."""
        for v in self.variants:
            if v.matches(category, modalities):
                return v
        legacy = self.legacy_modalities(category, modalities)
        if legacy is not None:
            return self.select(category, legacy)
        if loose:
            want = {modality_family(m) for m in modalities}
            for v in self.variants:
                if (v.when.get("category") == category
                        and {modality_family(m) for m in v.when.get("modalities", [])} == want):
                    return v
        if category not in self.wired_categories:
            raise KeyError(no_category_message(self.id, self.wired_categories, category))
        raise KeyError(no_variant_message(self.id, modalities, category))
