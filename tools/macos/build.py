"""Build the method environments for macOS on Apple silicon and pack them into the archives that
``mtb.env.install`` unpacks on a Mac (``multibench.engine.envs.on_macos``).

usage (on an Apple silicon Mac, with a Python that has this repository's ``multibench`` and conda-pack)::

    python tools/macos/build.py all    --work DIR [ENV ...]   wheels, build, pack, delete the built env
    python tools/macos/build.py recipe --work DIR ENV         write DIR/build/ENV/{conda.yml,requirements.txt}
    python tools/macos/build.py build  --work DIR ENV         build DIR/envs/ENV
    python tools/macos/build.py pack   --work DIR ENV         DIR/archives/ENV.tar.gz + packed_macos.json
    python tools/macos/build.py wheels --work DIR [ENV]       the wheels of wheels.yaml (for ENV)
    python tools/macos/build.py scan   PREFIX                 minimum macOS and outside paths of the binaries
    python tools/macos/build.py listing --work DIR            multibench/engine/packed_macos.json

``DIR`` holds everything the build writes: conda's package cache, pip's cache, a HOME of its own (so
no ``.condarc`` or R profile of the user is read), the built envs, the wheels, logs and the archives.
No other conda env and no other folder is touched. ``--conda`` names conda (default: ``$CONDA_EXE``,
then ``conda`` on PATH).

An environment is made from the package's Linux lock, ``multibench/engine/env_locks/<env>.yml``, by
these rules, and the files in ``tools/macos/envs/<env>/`` change them where a rule is not enough (each
file says why):

- conda part: ``python``, ``r-base`` and the R and Bioconductor packages keep the lock's version when
  osx-arm64 has a build of it (for the env's R); else the version is left to the solver; a package
  osx-arm64 does not have is left out (``drop.txt`` leaves one out by hand, ``unpin.txt`` leaves its
  version to the solver, ``extra.txt`` adds a spec). System libraries and the Linux toolchain are left
  to the solver, except a few numeric libraries (SOFT_NUMERIC; ``soft.txt`` replaces the list) kept
  at the lock's version when it exists.
- pip part: every pin of the lock, installed with ``--no-deps`` as on Linux, without the pins that
  exist only for Linux (LINUX_ONLY_PIP) and with a local tag (``+cu118``) removed. A pin with no
  arm64 wheel is built from source (``build_constraints.txt`` replaces the default ``setuptools<70``
  for those builds), ``force_source.txt`` builds one from source although a wheel exists,
  ``pip_overrides.txt`` replaces a pin (``<pin> <other pin>``) or leaves it out (``<pin> -``), and
  ``uv.txt`` names pins whose metadata pip refuses to read; ``uv`` installs those (it must be on PATH).
- then: the repairs below, the lock's post script (``env_locks/<env>.post.sh``), and
  ``post_macos.sh`` for what conda does not have for osx-arm64.

Repairs, applied to every env where they are needed:

- binaries compiled here are built for macOS 11 (MACOSX_DEPLOYMENT_TARGET) and have their rpaths made
  relative, since conda-pack rewrites only the files conda installed;
- pip binaries that ask for ``@rpath/libc++.1.dylib`` name the system's C++ library when conda put
  its own in ``lib/`` (two C++ runtimes in one process break llvmlite and numba);
- pip's copies of the OpenMP runtime become links to conda's when the env has one (three copies stop
  the process: "OMP: Error #179").

``pack`` writes ``.multibench_methods`` (from ``tools/macos/envs.yaml``), packs the env with
conda-pack (which signs the binaries it rewrites, ad hoc), and adds its entry to
``DIR/archives/packed_macos.json``: files, SHA-256, sizes, file count and methods. ``mtb.env.install``
reads that folder when ``MULTIBENCH_MACOS_ARCHIVES`` names it, so a new set can be checked before it
is uploaded. After the archives and that file are uploaded to the release BASE names, ``listing``
writes ``multibench/engine/packed_macos.json`` (the same entries, with ``base``), which the package
ships and reads on a Mac. Packages, versions and build records of each env land in
``DIR/build/<env>/``.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LOCKS = REPO / "multibench" / "engine" / "env_locks"
sys.path.insert(0, str(REPO))

#: The release the archives are uploaded to: the ``base`` of the package's listing.
BASE = "https://github.com/DSichang/scMultiBench/releases/download/envs-mac-v1"
LINUX_ONLY_PIP = re.compile(r"^(nvidia-|triton==|cuda-bindings==)", re.I)
SOFT_NUMERIC = ("hdf5", "gsl", "glpk", "armadillo", "arpack", "bedtools")
KEEP_EXACT = ("python", "r-base")
DEFAULT_BUILD_CONSTRAINTS = "setuptools<70\n"
MACHO = {b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe"}


class Work:
    """The folders of one build (``--work``) and the process environment its commands run in."""

    def __init__(self, root: str, conda: str | None):
        self.root = Path(root).resolve()
        for sub in ("build", "envs", "pkgs", "pipcache", "home", "logs", "wheels", "archives"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.conda = conda or os.environ.get("CONDA_EXE") or shutil.which("conda")
        if not self.conda:
            raise SystemExit("conda not found: pass --conda")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CONDA_", "R_LIBS", "PIP_", "UV_"))}
        env.update(HOME=str(self.root / "home"), CONDA_PKGS_DIRS=str(self.root / "pkgs"),
                   CONDA_ENVS_DIRS=str(self.root / "envs"), PIP_CACHE_DIR=str(self.root / "pipcache"),
                   PYTHONNOUSERSITE="1", CONDA_SUBDIR="osx-arm64")
        self.env = env

    def sh(self, cmd, log: Path | None = None, check: bool = True, extra: dict | None = None):
        out = subprocess.run(cmd, capture_output=True, text=True, env={**self.env, **(extra or {})})
        if log:
            with open(log, "a") as fh:
                fh.write(f"\n$ {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}\n{out.stdout}\n{out.stderr}\n")
        if check and out.returncode:
            raise SystemExit(f"FAILED ({out.returncode}): {cmd}\n{(out.stdout + out.stderr)[-3000:]}")
        return out


def lines(path: Path) -> list[str]:
    """The lines of a hand file, without blank lines and ``#`` comments; [] when it does not exist."""
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def env_methods() -> dict:
    return yaml.safe_load((HERE / "envs.yaml").read_text())


def vt(v: str):
    return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[._]", re.sub(r"[a-z].*$", "", v)) if x != "")


def admits(dep: str, version: str) -> bool:
    """Whether a conda dependency string (``r-base >=4.5,<4.6.0a0``) admits ``version``."""
    parts = dep.split()
    if len(parts) < 2:
        return True
    want = vt(version)
    for alt in parts[1].split("|"):
        ok = True
        for c in alt.split(","):
            m = re.match(r"(>=|<=|==|<|>|=)?(.*)", c)
            op, v = m.group(1) or "=", vt(m.group(2).rstrip("*").rstrip("."))
            if op == ">=":
                ok &= want >= v
            elif op == ">":
                ok &= want > v
            elif op == "<":
                ok &= want < v
            elif op == "<=":
                ok &= want <= v
            else:
                ok &= want[:len(v)] == v
        if ok:
            return True
    return False


def pip_part(pips: list[str], overrides: dict) -> tuple[list[str], list[str], list[str]]:
    """The lock's pip pins for macOS: (kept, left out as Linux-only, changed)."""
    keep, left, changed = [], [], []
    for p in pips:
        if p in overrides:
            changed.append(f"{p} -> {overrides[p]} (by hand)")
            if overrides[p] != "-":
                keep.append(overrides[p])
            continue
        if LINUX_ONLY_PIP.match(p):
            left.append(p)
            continue
        q = re.sub(r"\+(cu\d+|cpu)\b", "", p)
        if q != p:
            changed.append(f"{p} -> {q}")
        keep.append(q)
    return keep, left, changed


def repodata(w: Work) -> dict:
    """``{name: [record]}`` for osx-arm64 and noarch of bioconda and conda-forge (conda's cache)."""
    def files():
        return [f for f in glob.glob(str(w.root / "pkgs/cache/*.json")) if not f.endswith(".info.json")]
    if len(files()) < 4:
        w.sh([w.conda, "create", "-q", "-y", "--dry-run", "-p", str(w.root / "envs/_probe"), "--override-channels",
              "-c", "bioconda", "-c", "conda-forge", "python=3.9"], check=False)
    have: dict = {}
    for f in files():
        d = json.load(open(f))
        if d.get("info", {}).get("subdir") not in ("osx-arm64", "noarch"):
            continue
        for key in ("packages", "packages.conda"):
            for rec in d.get(key, {}).values():
                have.setdefault(rec["name"], []).append(rec)
    return have


def recipe(w: Work, env: str):
    from multibench.engine import envs as E
    H, B = HERE / "envs" / env, w.root / "build" / env
    B.mkdir(parents=True, exist_ok=True)
    conda_yaml, pips = E.split_lock((LOCKS / f"{env}.yml").read_text())
    pins = [m.groups() for ln in conda_yaml.splitlines()
            if (m := re.match(r"\s+- ([A-Za-z0-9_.\-]+)=([^=\s]+)(?:=\S+)?$", ln))]
    have = repodata(w)
    r_ver = next((v for n, v in pins if n == "r-base"), None)
    soft = lines(H / "soft.txt") if (H / "soft.txt").exists() else list(SOFT_NUMERIC)
    unpin, drop = set(lines(H / "unpin.txt")), set(lines(H / "drop.txt"))
    spec, rep = [], {"kept": [], "unpinned": [], "absent": [], "left_to_solver": [], "soft_kept": []}
    for n, v in pins:
        is_r = n.startswith(("r-", "bioconductor-")) and n != "r-base"
        recs = have.get(n, [])
        if n in drop:
            rep["absent"].append(f"{n}={v} (drop.txt)")
        elif n in KEEP_EXACT or is_r:
            if not recs:
                rep["absent"].append(f"{n}={v}")
                continue
            same = [r for r in recs if r["version"] == v]
            if is_r and r_ver:
                same = [r for r in same if all(admits(d, r_ver) for d in r["depends"] if d.split()[0] == "r-base")]
            if same and n not in unpin:
                spec.append(f"{n}=={v}")
                rep["kept"].append(f"{n}={v}")
            else:                                   # python and R keep their minor version
                spec.append(f"{n}={'.'.join(v.split('.')[:2])}" if n in KEEP_EXACT else n)
                rep["unpinned"].append(f"{n}={v}")
        elif n in soft and any(r["version"] == v for r in recs):
            spec.append(f"{n}=={v}")
            rep["soft_kept"].append(f"{n}={v}")
        elif n == "pip":
            spec.append("pip")
        else:
            rep["left_to_solver"].append(f"{n}={v}")
    if pips and "pip" not in spec:
        spec.append("pip")
    spec += lines(H / "extra.txt")
    (B / "conda.yml").write_text("name: %s\nchannels:\n  - bioconda\n  - conda-forge\ndependencies:\n%s\n"
                                 % (env, "\n".join(f"  - {s}" for s in spec)))
    over = dict(ln.split()[:2] for ln in lines(H / "pip_overrides.txt"))
    keep, left, changed = pip_part(pips, over)
    unknown = sorted(set(over) - set(pips))
    if unknown:
        raise SystemExit(f"{env}: pip_overrides.txt names pins the lock does not have: {unknown}")
    (B / "requirements.txt").write_text("".join(f"{p}\n" for p in keep))
    rep.update(extra=lines(H / "extra.txt"), pip_total=len(pips), pip_kept=len(keep), pip_left_out=left,
               pip_changed=changed, r=r_ver, conda_pins=len(pins))
    (B / "report.json").write_text(json.dumps(rep, indent=1))
    print(f"{env}: conda pins {len(pins)} -> kept {len(rep['kept'])}, soft {len(rep['soft_kept'])}, "
          f"unpinned {len(rep['unpinned'])}, absent {len(rep['absent'])}, to solver {len(rep['left_to_solver'])}"
          f" | pip {len(pips)} -> kept {len(keep)}, left out {len(left)}")
    for k in ("unpinned", "absent", "pip_changed"):
        if rep[k]:
            print(f"  {k}: {rep[k]}")


def activate(prefix: Path) -> str:
    """Shell lines that activate ``prefix`` as conda would, for builds that need its compilers."""
    return (f"export CONDA_PREFIX='{prefix}' CONDA_DEFAULT_ENV='{prefix.name}' "
            f"PATH='{prefix}/bin':/usr/bin:/bin:/usr/sbin:/sbin; "
            f"for _f in '{prefix}'/etc/conda/activate.d/*.sh; do [ -e \"$_f\" ] && . \"$_f\"; done; "
            "export R_LIBS_USER=/dev/null R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null; "
            "export MACOSX_DEPLOYMENT_TARGET=11.0; ")


def build(w: Work, env: str, resume: bool = False):
    """Build ``DIR/envs/<env>``; ``resume`` runs the post scripts again on a built env."""
    H, B, prefix = HERE / "envs" / env, w.root / "build" / env, w.root / "envs" / env
    log = w.root / "logs" / f"build_{env}.log"
    times = json.loads((B / "times.json").read_text()) if resume and (B / "times.json").exists() else {}
    if not resume:
        if prefix.exists():
            raise SystemExit(f"{prefix} exists; delete it first")
        log.write_text("")
        recipe(w, env)
        wheels(w, env)
        t = time.time()
        out = w.sh([w.conda, "env", "create", "-q", "-y", "-p", str(prefix), "-f", str(B / "conda.yml")],
                   log=log, check=False)
        times["conda_s"] = round(time.time() - t)
        if out.returncode:
            txt = out.stdout + out.stderr
            probs = [ln.strip()[:230] for ln in txt.splitlines() if re.match(r"\s*- (package|nothing provides)", ln)
                     or "is not installable" in ln or "does not exist" in ln]
            raise SystemExit("CONDA SOLVE FAILED:\n  " + "\n  ".join(probs[:25] or [txt[-2000:]]))
    py = prefix / "bin" / "python"
    req = [ln for ln in (B / "requirements.txt").read_text().splitlines() if ln]
    if req and not resume:
        pip_phase(w, env, prefix, req, log, times)
    if py.exists() and not resume:
        for msg in (fix_libcxx(w, prefix), fix_libomp(w, prefix)):
            if msg:
                print(msg)
    # the lock's post script once; post_macos.sh again on `resume` (it skips what is already installed)
    for name, script in ((f"{env}.post.sh", None if resume else LOCKS / f"{env}.post.sh"),
                         ("post_macos.sh", H / "post_macos.sh")):
        if script and script.exists():
            t = time.time()
            out = w.sh(["bash", "-c", activate(prefix) + f"bash '{script}'"], log=log, check=False)
            times[name] = round(time.time() - t)
            print(f"{name} exit {out.returncode} |", "\n".join((out.stdout + out.stderr).strip().splitlines()[-4:])[:600])
            if out.returncode:
                raise SystemExit(f"{name} failed (log: {log})")
    fixed = fix_rpaths(prefix)
    if fixed:
        print(f"rpaths made relative in {len(fixed)} file(s) built here: {fixed[:6]}")
    if py.exists():
        chk = w.sh([str(py), "-m", "pip", "check"], log=log, check=False)
        (B / "pip_check.txt").write_text(chk.stdout + chk.stderr)
        (B / "freeze.txt").write_text(w.sh([str(py), "-m", "pip", "freeze"], check=False).stdout)
    (B / "conda_list.txt").write_text(w.sh([w.conda, "list", "-p", str(prefix)], check=False).stdout)
    times["size_mb"] = int(w.sh(["du", "-sk", str(prefix)]).stdout.split()[0]) // 1024
    (B / "times.json").write_text(json.dumps(times))
    print(f"{env} built: {times}")


def pip_phase(w: Work, env: str, prefix: Path, req: list[str], log: Path, times: dict):
    """Wheels first (``--no-deps``, as on Linux); a pin without an arm64 wheel is built from source."""
    H, B = HERE / "envs" / env, w.root / "build" / env
    py = prefix / "bin" / "python"
    sdist, by_uv = lines(H / "force_source.txt"), lines(H / "uv.txt")
    t = time.time()
    for _ in range(40):
        binreq = B / "requirements.bin.txt"
        binreq.write_text("".join(f"{p}\n" for p in req if p not in sdist and p not in by_uv))
        out = w.sh([str(py), "-m", "pip", "install", "--no-deps", "--only-binary=:all:",
                    "--find-links", str(w.root / "wheels"), "-r", str(binreq)], log=log, check=False)
        if out.returncode == 0:
            break
        m = re.search(r"No matching distribution found for (\S+)", out.stdout + out.stderr)
        name = m and re.split(r"[=<>!~]", m.group(1))[0].lower().replace("_", "-")
        hit = [p for p in req if name and p.split("==")[0].lower().replace("_", "-") == name]
        if not hit or hit[0] in sdist:
            raise SystemExit(f"pip phase failed:\n{(out.stdout + out.stderr)[-2500:]}")
        sdist.append(hit[0])
    times["pip_s"] = round(time.time() - t)
    cons = H / "build_constraints.txt"
    if not cons.exists():
        cons = B / "build_constraints.txt"
        cons.write_text(DEFAULT_BUILD_CONSTRAINTS)
    if by_uv:
        uv = shutil.which("uv")
        if not uv:
            raise SystemExit(f"{env}: uv.txt names {by_uv}; put uv on PATH and build again")
        w.sh([uv, "pip", "install", "--no-deps", "--python", str(py), *by_uv], log=log)
    t, failed = time.time(), []
    for s in sdist:
        # only this pin from source; its build tools (setuptools, Cython, numpy) come as wheels
        out = w.sh(["bash", "-c", activate(prefix) + f"export PIP_CONSTRAINT='{cons}'; "
                    f"python -m pip install --no-deps --no-cache-dir --no-binary '{s.split('==')[0]}' '{s}'"],
                   log=log, check=False)
        if out.returncode:
            failed.append(s)
            print(f"  SOURCE BUILD FAILED: {s}\n    " + (out.stdout + out.stderr)[-700:].replace("\n", "\n    "))
    times["sdist_s"] = round(time.time() - t)
    (B / "pip_from_source.txt").write_text("".join(f"{s}\n" for s in sdist))
    (B / "pip_installed_by_uv.txt").write_text("".join(f"{s}\n" for s in by_uv))
    if failed:
        raise SystemExit(f"{env}: source builds failed: {failed} (log: {log})")


def _conda_owned(prefix: Path) -> set:
    owned = set()
    for meta in (prefix / "conda-meta").glob("*.json"):
        owned.update(json.loads(meta.read_text()).get("files", []))
    return owned


def _macho(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(4) in MACHO
    except OSError:
        return False


def _foreign_binaries(prefix: Path, under: str = ""):
    """Mach-O files under ``prefix`` that conda did not install (pip's, or built here)."""
    owned = _conda_owned(prefix)
    for d, _dirs, files in os.walk(prefix):
        if under and under not in d:
            continue
        for f in files:
            p = os.path.join(d, f)
            rel = os.path.relpath(p, prefix)
            if rel not in owned and not os.path.islink(p) and not p.endswith((".a", ".o")) and _macho(p):
                yield p, rel


def fix_libcxx(w: Work, prefix: Path) -> str:
    """Where conda put its own libc++ in ``lib/`` (R, bedtools), a pip binary linked against the system's
    C++ library and asking for ``@rpath/libc++.1.dylib`` finds conda's first, which lacks symbols it
    needs (``import numba`` fails). Such binaries get the system library named explicitly."""
    if not (prefix / "lib" / "libc++.1.dylib").exists():
        return ""
    done = []
    for p, rel in _foreign_binaries(prefix, under="site-packages"):
        if "@rpath/libc++.1.dylib" in subprocess.run(["otool", "-L", p], capture_output=True, text=True).stdout:
            subprocess.run(["install_name_tool", "-change", "@rpath/libc++.1.dylib", "/usr/lib/libc++.1.dylib", p],
                           capture_output=True, check=True)
            subprocess.run(["codesign", "--force", "--sign", "-", p], capture_output=True, check=True)
            done.append(rel)
    (w.root / "build" / prefix.name / "libcxx_repaired.txt").write_text("".join(f"{r}\n" for r in done))
    return f"system libc++ named in {len(done)} pip binaries: {done[:4]}" if done else ""


def fix_libomp(w: Work, prefix: Path) -> str:
    """One OpenMP runtime per process: where conda has ``lib/libomp.dylib`` (R's BLAS loads it), each
    copy pip installed (torch, scikit-learn) becomes a link to it."""
    conda_omp = prefix / "lib" / "libomp.dylib"
    if not conda_omp.exists():
        return ""
    owned, done = _conda_owned(prefix), []
    for d, _dirs, files in os.walk(prefix / "lib"):
        p = os.path.join(d, "libomp.dylib")
        rel = os.path.relpath(p, prefix)
        if "site-packages" in d and "libomp.dylib" in files and rel not in owned and not os.path.islink(p):
            os.unlink(p)
            os.symlink(os.path.relpath(conda_omp, d), p)
            done.append(rel)
    (w.root / "build" / prefix.name / "libomp_linked.txt").write_text("".join(f"{r}\n" for r in done))
    return f"OpenMP: {done} now link to the env's lib/libomp.dylib" if done else ""


def fix_rpaths(prefix: Path) -> list[str]:
    """A binary built here outside conda keeps its build folder as an absolute rpath, which conda-pack
    does not rewrite: make it relative to the file (``@loader_path``) and sign the file again."""
    done = []
    rpath = re.compile(r"cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (\S+)")
    for p, rel in _foreign_binaries(prefix):
        have = rpath.findall(subprocess.run(["otool", "-l", p], capture_output=True, text=True).stdout)
        absolute = [r for r in dict.fromkeys(have) if r.startswith(str(prefix))]
        if not absolute:
            continue
        have = set(have)
        for r in absolute:
            new = "@loader_path/" + os.path.relpath(r, os.path.dirname(p))
            if new in have:                     # dyld refuses a file that names one rpath twice
                subprocess.run(["install_name_tool", "-delete_rpath", r, p], capture_output=True, check=True)
            else:
                subprocess.run(["install_name_tool", "-rpath", r, new, p], capture_output=True, check=True)
                have.add(new)
        subprocess.run(["codesign", "--force", "--sign", "-", p], capture_output=True, check=True)
        done.append(rel)
    return done


def wheels(w: Work, env: str | None = None):
    """Build the wheels of ``wheels.yaml`` (those ``for`` ``env``) that ``DIR/wheels`` does not have."""
    for pin, spec in (yaml.safe_load((HERE / "wheels.yaml").read_text()) or {}).items():
        if env and env not in spec["for"]:
            continue
        name, version = pin.split("==")
        tag = "cp" + spec["python"].replace(".", "")
        if glob.glob(str(w.root / "wheels" / f"{name.replace('-', '_')}-{version}-{tag}-*.whl")):
            continue
        prefix = w.root / "envs" / f"_wheel_{tag}_{name}"
        log = w.root / "logs" / f"wheel_{tag}_{name}.log"
        shutil.rmtree(prefix, ignore_errors=True)
        w.sh([w.conda, "create", "-q", "-y", "-p", str(prefix), "--override-channels", "-c", "conda-forge",
              f"python={spec['python']}", "pip", *spec["conda"]], log=log)
        raw = w.root / "wheels" / "raw"
        raw.mkdir(exist_ok=True)
        dirs = "".join(f"export {k}='{prefix}'; " for k in spec.get("dirs", []))
        cons = w.root / "wheels" / "build_constraints.txt"
        cons.write_text(DEFAULT_BUILD_CONSTRAINTS)
        w.sh(["bash", "-c", activate(prefix) + dirs + f"export PIP_CONSTRAINT='{cons}'; "
              f"python -m pip wheel --no-deps --no-binary '{name}' -w '{raw}' '{pin}' && "
              "python -m pip install -q delocate && "
              f"delocate-wheel {'--ignore-missing-dependencies ' if spec.get('ignore_missing') else ''}"
              f"-w '{w.root / 'wheels'}' -v \"$(ls -t '{raw}'/*.whl | head -1)\""], log=log)
        shutil.rmtree(prefix)
        print(f"wheel built: {pin} for Python {spec['python']}")


def pack(w: Work, env: str):
    methods = env_methods()[env]
    prefix = w.root / "envs" / env
    (prefix / ".multibench_methods").write_text("".join(f"{m}\n" for m in methods))
    out = w.root / "archives" / f"{env}.tar.gz"
    out.unlink(missing_ok=True)
    cp = Path(sys.executable).parent / "conda-pack"
    t = time.time()
    w.sh([str(cp) if cp.exists() else "conda-pack", "-p", str(prefix), "-o", str(out), "--n-threads", "8", "-q",
          "--ignore-missing-files"], log=w.root / "logs" / f"build_{env}.log")
    n = size = 0
    for d, _dirs, files in os.walk(prefix):
        for x in files:
            p = os.path.join(d, x)
            if not os.path.islink(p):
                n, size = n + 1, size + os.path.getsize(p)
    h = hashlib.sha256()
    with open(out, "rb") as fh:
        while b := fh.read(1 << 22):
            h.update(b)
    listing = w.root / "archives" / "packed_macos.json"
    data = json.loads(listing.read_text()) if listing.exists() else {"archives": {}}
    data["archives"][env] = {"files": [out.name], "sha256": h.hexdigest(), "archive_bytes": out.stat().st_size,
                             "unpacked_bytes": size, "file_count": n, "methods": methods}
    listing.write_text(json.dumps(data, indent=1) + "\n")
    print(f"{env} packed in {time.time() - t:.0f} s: {out.stat().st_size / 1e6:.0f} MB, "
          f"{size / 1e9:.2f} GB unpacked, {n} files, methods {', '.join(methods)}")


def listing(w: Work):
    """Write the package's ``engine/packed_macos.json`` from ``DIR/archives/packed_macos.json``, after
    checking that every env of envs.yaml is there and every archive has its listed size and SHA-256."""
    src = json.loads((w.root / "archives" / "packed_macos.json").read_text())["archives"]
    missing = sorted(set(env_methods()) - set(src))
    if missing:
        raise SystemExit(f"not packed: {missing}")
    for env, entry in src.items():
        if entry["methods"] != env_methods().get(env):
            raise SystemExit(f"{env}: its archive serves {entry['methods']}, envs.yaml says {env_methods().get(env)}")
        for name in entry["files"]:
            f, h = w.root / "archives" / name, hashlib.sha256()
            with open(f, "rb") as fh:
                while b := fh.read(1 << 22):
                    h.update(b)
            if h.hexdigest() != entry["sha256"] or f.stat().st_size != entry["archive_bytes"]:
                raise SystemExit(f"{name} differs from its entry")
    out = REPO / "multibench" / "engine" / "packed_macos.json"
    out.write_text(json.dumps({"base": BASE, "archives": dict(sorted(src.items())), "_meta": {
        "written": time.strftime("%Y-%m-%d %H:%M"),
        "note": "Written by tools/macos/build.py listing. archive_bytes: the archive; unpacked_bytes: the "
                "files in it."}}, indent=1) + "\n")
    print(f"{out}: {len(src)} archives, {sum(len(e['methods']) for e in src.values())} methods, base {BASE}")


def scan(prefix: str):
    """The minimum macOS each binary is built for, and the paths outside the env its load commands name."""
    root = os.path.abspath(prefix)
    minos, outside, n = {}, {}, 0
    for d, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            if os.path.islink(p) or p.endswith((".a", ".o")) or not _macho(p):
                continue
            n += 1
            out = subprocess.run(["otool", "-arch", "arm64", "-L", "-l", p], capture_output=True, text=True).stdout
            m = re.search(r"cmd LC_BUILD_VERSION\n(?:.*\n){1,3}?\s+minos (\S+)", out)
            minos.setdefault(m.group(1) if m else "?", []).append(os.path.relpath(p, root))
            for q in re.findall(r"(?<![\w/])((?:/Users|/opt|/Library/Frameworks|/usr/local)/\S*)", out):
                if not q.startswith(root):
                    outside.setdefault(os.path.relpath(p, root), set()).add(q)
    order = sorted((k for k in minos if k != "?"), key=vt)
    print(f"{root}: {n} Mach-O | minimum macOS: " + ", ".join(f"{k}: {len(minos[k])}" for k in order))
    for k in order:
        if vt(k) >= (12, 3):
            print(f"  built for {k}: {minos[k][:6]}{' ...' if len(minos[k]) > 6 else ''}")
    print("  paths outside the env in load commands:", {k: sorted(v) for k, v in list(outside.items())[:8]} or "none")


def build_all(w: Work, names: list[str], keep: bool):
    known = env_methods()
    for env in names or list(known):
        if env not in known:
            raise SystemExit(f"{env} is not in tools/macos/envs.yaml")
        shutil.rmtree(w.root / "envs" / env, ignore_errors=True)
        build(w, env)
        pack(w, env)
        if not keep:
            shutil.rmtree(w.root / "envs" / env)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["all", "recipe", "build", "resume", "pack", "wheels", "scan", "listing"])
    ap.add_argument("names", nargs="*", help="env names (scan: one prefix)")
    ap.add_argument("--work", help="the build folder (all but scan)")
    ap.add_argument("--conda", help="conda to use (default: $CONDA_EXE, then conda on PATH)")
    ap.add_argument("--keep", action="store_true", help="all: keep the built envs")
    a = ap.parse_args(argv)
    if a.command == "scan":
        return scan(a.names[0])
    if not a.work:
        ap.error("--work is required")
    w = Work(a.work, a.conda)
    if a.command == "all":
        return build_all(w, a.names, a.keep)
    if a.command == "listing":
        return listing(w)
    if a.command == "wheels":
        return wheels(w, a.names[0] if a.names else None)
    if len(a.names) != 1:
        ap.error(f"{a.command} takes one env")
    {"recipe": recipe, "build": build, "pack": pack,
     "resume": lambda w, e: build(w, e, resume=True)}[a.command](w, a.names[0])


if __name__ == "__main__":
    main()
