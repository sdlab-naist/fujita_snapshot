"""
runner.py — server-side execution framework (RQ2/RQ3)

An execution framework for running `npm ci` + `jest --coverage` + Stryker over hundreds of
repositories, with isolation, timeouts, failure handling, resumability, and parallelism.

Container backends (Singularity/Apptainer are primary, assuming HPC without Docker):
  - "apptainer" / "singularity": HPC standard. No root needed, uses a SIF image.
  - "docker"   : for local development / CI.
  - "local"    : uses the host node directly without a container (no isolation; for testing).

Design points:
  - Each repository is measured in a working copy on scratch, so the original is not touched.
  - Dependencies are installed once, and the tests are neutralized three ways (All / Snapshot only /
    Non-Snapshot only) via tools/apply_skip.js, then jest/stryker are run.
  - Results are appended to a JSONL incrementally; on re-run, completed entries are skipped (resumable).
  - Per-repository timeout and structured logging.

Usage (example):
  # first build the SIF image (from apptainer.def):
  #   apptainer build node.sif apptainer.def
  python runner.py --metric coverage --backend apptainer --image node.sif \\
      --repos-json repos.json --out results_coverage.jsonl --workers 4

  # repos.json is produced by repo_collection.resolve_repo_list:
  #   [{"full_name":..., "clone_url":...}, ...]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import common
import effectiveness_sets as es

TOOLS = Path(__file__).resolve().parent / "tools"
MODES = ["all", "snapshot", "non-snapshot"]


# ===========================================================================
# Sandbox (backend abstraction)
# ===========================================================================
def _env_flags(backend: str, env: dict[str, str]) -> list[str]:
    flags: list[str] = []
    for k, v in env.items():
        if backend in ("apptainer", "singularity"):
            flags += ["--env", f"{k}={v}"]
        elif backend == "docker":
            flags += ["-e", f"{k}={v}"]
    return flags


def sandbox_argv(backend: str, image: str | None, workdir: str, scratch: str,
                 command: str, env: dict[str, str] | None = None,
                 network: bool = True) -> tuple[list[str], dict | None]:
    """Build the argv to run `command` (a bash -lc string) inside the sandbox.

    Returns: (argv, env_for_local). For non-local backends the env is passed to the
    container, so the Python-side env is None.
    """
    env = env or {}
    if backend == "local":
        merged = {**os.environ, **env}
        return ["bash", "-lc", command], merged

    if backend == "docker":
        argv = ["docker", "run", "--rm",
                "-v", f"{os.path.abspath(workdir)}:/work",
                "-v", f"{os.path.abspath(scratch)}:/scratch",
                "-w", "/work"]
        if not network:
            argv.append("--network=none")
        argv += _env_flags("docker", env)
        argv += [image or "node:20", "bash", "-lc", command]
        return argv, None

    if backend in ("apptainer", "singularity"):
        binname = backend
        argv = [binname, "exec",
                "--containall",          # isolate from the host environment
                "--writable-tmpfs",      # make /tmp writable
                "--bind", f"{os.path.abspath(workdir)}:/work",
                "--bind", f"{os.path.abspath(scratch)}:/scratch",
                "--pwd", "/work"]
        if not network:
            argv.append("--net")  # only when isolation is needed (usually not for install)
        argv += _env_flags(binname, env)
        if not image:
            raise ValueError(f"{backend} requires --image (SIF)")
        argv += [image, "bash", "-lc", command]
        return argv, None

    raise ValueError(f"unknown backend: {backend}")


def run_sandbox(backend: str, image: str | None, workdir: str, scratch: str,
                command: str, timeout: int, env: dict[str, str] | None = None,
                network: bool = True) -> subprocess.CompletedProcess:
    """Run a command in the sandbox (with a timeout)."""
    argv, local_env = sandbox_argv(backend, image, workdir, scratch,
                                   command, env, network)
    # the local backend has no container bind, so run in the host working directory
    cwd = workdir if backend == "local" else None
    # start_new_session: even if a measured repo's tests/scripts send a signal to the process
    # group via kill(0) etc., it must not propagate to the sbatch batch script
    # (this is what caused jobs 22541/22960/22986/23012 to die from an external SIGTERM ~85s in).
    return subprocess.run(argv, capture_output=True, text=True,
                          timeout=timeout, env=local_env, cwd=cwd,
                          start_new_session=True)


# common env that redirects container HOME / npm cache to scratch
def _sandbox_env(backend: str) -> dict[str, str]:
    if backend == "local":
        return {}
    return {
        "HOME": "/scratch/home",
        "NPM_CONFIG_CACHE": "/scratch/npm-cache",
        "npm_config_fund": "false",
        "npm_config_audit": "false",
        "CI": "true",
    }


# ===========================================================================
# Test file enumeration and neutralization (apply_skip.js)
# ===========================================================================
def _list_test_files(repo: Path) -> list[Path]:
    # static fallback: resolve the Jest config (custom testMatch/testRegex) and enumerate
    is_test = common.jest_test_matcher(repo)
    files = []
    for p in repo.rglob("*"):
        if (p.is_file() and "node_modules" not in p.parts
                and p.suffix.lower() in common.CODE_EXTS
                and is_test(p)):
            files.append(p)
    return files


def list_tests_via_jest(work: Path, backend: str, image: str | None,
                        scratch: str, env: dict[str, str] | None = None,
                        timeout: int = 900) -> tuple[list[Path], str]:
    """Enumerate tests with `npx jest --listTests`, exactly as in paper 4.2.1.

    Runs inside the sandbox against a working copy with dependencies already installed, and maps the
    output absolute paths (container-side /work/...) back to host paths. Falls back to static
    resolution if none are obtained (e.g. jest cannot start).

    Returns: (list of test files, "listTests" | "static")
    """
    proc = run_sandbox(backend, image, str(work), str(scratch),
                       "npx jest --listTests 2>/dev/null || true",
                       timeout=timeout, env=env or {})
    root = Path(work).resolve()
    prefix = str(root) + "/" if backend == "local" else "/work/"
    # the output format differs by jest version:
    #   newer jest: one absolute path per line / older jest (<=20 etc.): a JSON array on one line
    cand: list[str] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            try:
                arr = json.loads(line)
                if isinstance(arr, list):
                    cand += [s for s in arr if isinstance(s, str)]
                    continue
            except ValueError:
                pass
        cand.append(line)
    files: list[Path] = []
    for s in cand:
        if not s.startswith(prefix):
            continue
        p = root / s[len(prefix):]
        if p.is_file():
            files.append(p)
    if files:
        return files, "listTests"
    return _list_test_files(Path(work)), "static"


def apply_skip(test_files: list[Path], mode: str) -> None:
    """Neutralize tests according to mode='snapshot'|'non-snapshot' (runs on the host)."""
    if mode == "all" or not test_files:
        return
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write("\n".join(str(p) for p in test_files))
        lst = tf.name
    try:
        subprocess.run(
            ["node", str(TOOLS / "apply_skip.js"), "--mode", mode,
             "--files-from", lst],
            capture_output=True, text=True, timeout=600, check=False)
    finally:
        os.unlink(lst)


# ===========================================================================
# Mutation target scope (mutate) — restrict to the source the tests cover
# ===========================================================================
def _covered_mutate_globs(cov_json: Path) -> list[str]:
    """Return the project-relative paths of non-test source files with at least one covered
    statement (to pass to Stryker's `mutate`), read from a coverage-final.json.

    With the default (no `mutate`), Stryker mutates the entire src tree, so files the tests never
    touch all become NoCoverage and inflate the denominator of the regular score (the main reason
    this reproduction comes out lower than the paper's 45.9). Restricting to source the tests
    actually reach brings it closer to the paper's intent.
    """
    try:
        data = json.loads(cov_json.read_text())
    except Exception:
        return []
    globs: list[str] = []
    for fp, fc in data.items():
        s = fc.get("s", {})
        if not any(v > 0 for v in s.values()):
            continue  # not a single statement covered = a NoCoverage-only file, so exclude it
        rel = fp[len("/work/"):] if fp.startswith("/work/") else fp.lstrip("/")
        if "/node_modules/" in rel:
            continue
        if any(t in rel for t in (".test.", ".spec.", "/test/", "/tests/",
                                  "/__tests__/", "__mocks__")):
            continue
        globs.append(rel)
    return globs


# ===========================================================================
# Measurement of a single project
# ===========================================================================
def measure_project(repo: dict, backend: str, image: str | None,
                    metric: str, scratch_root: str,
                    install_timeout: int = 1800,
                    run_timeout: int = 2400) -> dict:
    """Measure coverage / mutation for one repository across the 3 modes and compute Only-by.

    Returns: {"full_name", "status", "metrics" | "error"}
    """
    full = repo["full_name"]
    work_parent = tempfile.mkdtemp(prefix="repro_", dir=scratch_root)
    work = Path(work_parent) / "repo"
    scratch = Path(work_parent) / "scratch"
    for sub in ("home", "npm-cache"):
        (scratch / sub).mkdir(parents=True, exist_ok=True)
    env = _sandbox_env(backend)

    try:
        # 1) clone
        common.clone_repo(repo["clone_url"], work)

        # 2) install dependencies (once)
        #    --legacy-peer-deps: npm6 at the paper's time resolved peer deps loosely, but
        #    node20/npm10 became strict and many old repositories become install_failed with
        #    ERESOLVE (246 of 407 install failures were ERESOLVE). This restores the paper-era behavior.
        inst = run_sandbox(
            backend, image, str(work), str(scratch),
            "npm ci --legacy-peer-deps || npm install --legacy-peer-deps",
            timeout=install_timeout, env=env)
        if inst.returncode != 0:
            return {"full_name": full, "status": "install_failed",
                    "error": inst.stderr[-2000:]}

        # 2b) mutation: Stryker is usually not present in the measured repository, so install it
        #     each time (--no-save so package.json is not polluted). If there is no config, generate
        #     a minimal jest-runner config. run 21576 lacked this setup and all 3120 became no_report
        #     (results/rq3_run1_nostryker_21576).
        if metric == "mutation":
            setup = run_sandbox(
                backend, image, str(work), str(scratch),
                "npm install --no-save --legacy-peer-deps "
                "@stryker-mutator/core @stryker-mutator/jest-runner",
                timeout=install_timeout, env=env)
            if setup.returncode != 0:
                return {"full_name": full, "status": "stryker_install_failed",
                        "error": setup.stderr[-2000:]}
            conf_names = ("stryker.conf.json", "stryker.conf.js",
                          "stryker.config.json", "stryker.config.js",
                          "stryker.conf.mjs", "stryker.config.mjs",
                          "stryker.conf.cjs", "stryker.config.cjs")
            if not any((work / f).exists() for f in conf_names):
                # keep concurrency from expanding to all cores visible in the container (48)
                conf = {
                    "testRunner": "jest",
                    "reporters": ["json"],
                    "coverageAnalysis": "perTest",
                    "concurrency": 4,
                }
                # only when MUTATE_SCOPE=covered, restrict the mutation target to source the tests
                # cover (the default is the whole src tree).
                if os.environ.get("MUTATE_SCOPE") == "covered":
                    cov = run_sandbox(
                        backend, image, str(work), str(scratch),
                        "npx jest --coverage --coverageReporters=json "
                        "--coverageDirectory=_mscope_cov --ci --silent "
                        "--watchAll=false || true",
                        timeout=run_timeout, env=env)
                    globs = _covered_mutate_globs(
                        work / "_mscope_cov" / "coverage-final.json")
                    if globs:
                        conf["mutate"] = globs
                (work / "stryker.conf.json").write_text(
                    json.dumps(conf, indent=2))

        # 3) enumerate test files (paper 4.2.1: npx jest --listTests) and back up the test
        #    bodies (so we can restore between modes)
        test_files, list_src = list_tests_via_jest(
            work, backend, image, str(scratch), env)
        backups = {p: p.read_text(errors="ignore") for p in test_files}

        reports: dict[str, Path] = {}
        for mode in MODES:
            # restore the tests, then apply the mode
            for p, original in backups.items():
                p.write_text(original)
            apply_skip(test_files, mode)

            if metric == "coverage":
                outdir = f"coverage_{mode}"
                cmd = (f"npx jest --coverage --coverageReporters=json "
                       f"--coverageDirectory={outdir} --ci --silent "
                       f"--watchAll=false || true")
                proc = run_sandbox(backend, image, str(work), str(scratch),
                                   cmd, timeout=run_timeout, env=env)
                rep = work / outdir / "coverage-final.json"
            else:  # mutation
                # --jsonReporter.fileName does not exist on the CLI (it dies immediately with an
                # unknown option and made everything no_report). Use the json reporter's default
                # output reports/mutation/mutation.json, and delete leftovers between modes first.
                rep = work / "reports" / "mutation" / "mutation.json"
                if rep.exists():
                    rep.unlink()
                cmd = "npx stryker run --reporters json || true"
                proc = run_sandbox(backend, image, str(work), str(scratch),
                                   cmd, timeout=run_timeout, env=env)

            if not rep.exists():
                # keep the tail of the run log so the cause of no_report can be investigated
                return {"full_name": full, "status": f"{mode}_no_report",
                        "error": (proc.stdout + "\n--- stderr ---\n"
                                  + proc.stderr)[-2000:]}
            # move the report outside work_parent (so it survives after work is deleted)
            saved = Path(work_parent) / f"{metric}_{mode}.json"
            shutil.copy(rep, saved)
            reports[mode] = saved

        # restore
        for p, original in backups.items():
            p.write_text(original)

        # 4) compute Only-by as set differences
        if metric == "coverage":
            values = es.coverage_only_by(reports["all"], reports["snapshot"],
                                         reports["non-snapshot"])
            # metrics_sets: the 6 categories via true set operations (computes ST∩NonST correctly).
            # The raw coverage-final.json is discarded, so we finalize and store it here.
            sets = es.coverage_full_sets(
                reports["all"], reports["snapshot"], reports["non-snapshot"])
            return {"full_name": full, "status": "ok", "metrics": values,
                    "metrics_sets": sets, "test_list_source": list_src}
        else:
            # compute both the regular mutation score (Fig 5) and the mutation score over covered
            # code (Fig 6) (paper 4.3.2).
            values = es.mutation_only_by(reports["all"], reports["snapshot"],
                                         reports["non-snapshot"])
            values_covered = es.covered_mutation_only_by(
                reports["all"], reports["snapshot"], reports["non-snapshot"])
            # mutators: per-mutatorName kill rate (ST/Non-ST/All) — for paper Table 4.
            # The raw mutation.json is discarded, so we finalize and store it here.
            mutators = es.mutation_by_mutator(
                reports["all"], reports["snapshot"], reports["non-snapshot"])
            return {"full_name": full, "status": "ok",
                    "metrics": values, "metrics_covered": values_covered,
                    "mutators": mutators, "test_list_source": list_src}

    except subprocess.TimeoutExpired:
        return {"full_name": full, "status": "timeout"}
    except Exception as e:
        return {"full_name": full, "status": "error", "error": str(e)}
    finally:
        shutil.rmtree(work_parent, ignore_errors=True)


# ===========================================================================
# Orchestration (resumable, parallel)
# ===========================================================================
def _load_done(out_path: str) -> set[str]:
    done: set[str] = set()
    p = Path(out_path)
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                done.add(json.loads(line)["full_name"])
            except Exception:
                continue
    return done


def process_repos(repos: list[dict], metric: str, backend: str,
                  image: str | None, out_path: str, workers: int = 2,
                  scratch_root: str | None = None) -> None:
    """Measure a set of repositories in parallel and resumably, appending results to a JSONL."""
    scratch_root = scratch_root or tempfile.gettempdir()
    done = _load_done(out_path)
    todo = [r for r in repos if r["full_name"] not in done]
    print(f"[runner] total={len(repos)} done={len(done)} todo={len(todo)} "
          f"backend={backend} metric={metric} workers={workers}")

    out = open(out_path, "a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(measure_project, r, backend, image, metric,
                          scratch_root): r for r in todo
            }
            for i, fut in enumerate(as_completed(futures), 1):
                r = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"full_name": r["full_name"], "status": "crashed",
                           "error": str(e)}
                out.write(json.dumps(res, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{i}/{len(todo)}] {res['full_name']}: {res['status']}")
    finally:
        out.close()


def collect_distributions(out_path: str, key: str = "metrics",
                          include: set[str] | None = None) -> dict[str, list[float]]:
    """Aggregate the 5-category distributions from the ok results in a JSONL (for plotting/aggregation).

    key="metrics"         : coverage / regular mutation score (Fig 4 / Fig 5)
    key="metrics_covered" : mutation score over covered code (Fig 6)
    include               : allow-list of full_name (paper 4.2.1 targets only snapshot-adopter
                            repositories, so this narrows to them)
    """
    per_project = []
    for line in Path(out_path).read_text().splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if include is not None and rec.get("full_name") not in include:
            continue
        if rec.get("status") == "ok" and key in rec:
            per_project.append(rec[key])
    return es.aggregate_projects(per_project)


def main() -> None:
    ap = argparse.ArgumentParser(description="RQ2/RQ3 server-side execution framework")
    ap.add_argument("--metric", choices=["coverage", "mutation"], required=True)
    ap.add_argument("--backend", default="apptainer",
                    choices=["apptainer", "singularity", "docker", "local"])
    ap.add_argument("--image", help="SIF (apptainer) / docker image")
    ap.add_argument("--repos-json", required=True,
                    help="JSON of [{full_name, clone_url}, ...]")
    ap.add_argument("--out", required=True, help="result JSONL (resumable)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--scratch", help="work area (specify a large scratch on HPC)")
    ap.add_argument("--summarize", action="store_true",
                    help="do not measure; aggregate and print distributions from an existing JSONL")
    args = ap.parse_args()

    if args.summarize:
        import numpy as np
        dist = collect_distributions(args.out)
        print(f"[{args.metric}] regular score (Fig 4/Fig 5 equivalent):")
        for c in es.CATEGORIES:
            vals = dist.get(c, [])
            if vals:
                print(f"  {c:22s} n={len(vals):4d} median={np.median(vals):5.1f}%")
        if args.metric == "mutation":
            covered = collect_distributions(args.out, key="metrics_covered")
            if any(covered.get(c) for c in es.CATEGORIES):
                print("[mutation] score over covered code (Fig 6 equivalent):")
                for c in ("All", "Snapshot", "Non-Snapshot"):
                    vals = covered.get(c, [])
                    if vals:
                        print(f"  {c:22s} n={len(vals):4d} median={np.median(vals):5.1f}%")
        return

    repos = json.loads(Path(args.repos_json).read_text())
    process_repos(repos, args.metric, args.backend, args.image,
                  args.out, workers=args.workers, scratch_root=args.scratch)


if __name__ == "__main__":
    main()
