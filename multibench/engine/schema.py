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
    role: str = ""          # input role (e.g. 'rna') or 'out_dir'
    flag: str | None = None # None -> positional
    # several input roles under one flag, in order (nargs='+'); an entry
    # "=VALUE" emits VALUE, e.g. [adt1, "=None"] -> --path2 adt1.h5 None
    roles: list = field(default_factory=list)
    repeat: bool = False    # value may be a list (nargs='+')
    eq: bool = False        # emit as --flag=value
    const: str | None = None  # literal value to emit (e.g. 'NULL')

    @property
    def is_positional(self) -> bool:
        return self.flag is None


@dataclass
class OutputSpec:
    kind: str               # embedding | graph | labels
    file: str               # filename written into out_dir
    dataset: str | None = None  # in-file dataset name for h5 outputs
    # the order of the output's rows when it is not argument order:
    # Variant.stacked_roles
    cell_order: list | str | None = None


@dataclass
class Variant:
    when: dict              # {'category': str, 'modalities': [str, ...]}
    entrypoint: str
    language: str
    args: list[ArgSpec]
    output: OutputSpec
    params: dict = field(default_factory=dict)        # default hyperparams emitted on the command line
    tunable: dict = field(default_factory=dict)       # {name: {default, type}} the script accepts; never emitted
    run_env: dict = field(default_factory=dict)       # environment variables the method needs
    cwd_at_script: bool = False        # run in the script's folder (it imports local files)
    pty: bool = False                  # run under a pseudo-tty (scJoint reads the terminal size)
    driver: str | None = None          # package-side wrapper in engine/drivers/ that runs the script
    normalize_peaks: list = field(default_factory=list)  # roles whose peak names become chr:start-end
    extra_outputs: list[OutputSpec] = field(default_factory=list)
    helpers: list = field(default_factory=list)     # modules upstream does not ship, from engine/helpers/<method>/
    size_hint: str = ""                # what the method needs from the data's size; starts with the method name
    # the sizes below which size_hint applies; scan shows it only for such data
    min_cells: int | None = None       # cells of the files the row reads, in total
    min_cells_per_batch: int | None = None   # cells of each numbered batch
    min_genes: int | None = None       # features of each RNA file
    downloads: list = field(default_factory=list)   # [{file, url}] the first run downloads next to the script

    @property
    def shown_modalities(self) -> list[str]:
        """The modalities read; for a variant that reads a folder (scBridge), the
        stems of the ``.h5`` files it names (``rna``, ``atac_gas``)."""
        return list(self.when.get("modalities", [])) or [
            PurePath(str(a.const)).stem for a in self.args
            if a.const and str(a.const).endswith(".h5")]

    def matches(self, category: str, modalities: set[str]) -> bool:
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
    id: str
    language: str
    # derived from the variants in __post_init__ unless given
    categories: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    # "peak" | "gene_activity" | None; declared, not derived from role names:
    # moETM/scMM/iPOLNG declare role `atac_gas` yet read peak matrices
    atac: str | None = None
    setup_hint: str = ""
    variants: list[Variant] = field(default_factory=list)
    reference: dict = field(default_factory=dict)     # provenance (references.yaml)
    # the flags that turn CUDA off (merged by runner.cpu_params_for on a host
    # without a GPU), or a script that calls CUDA unconditionally (file:line)
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
