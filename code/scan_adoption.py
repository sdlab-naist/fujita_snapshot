"""scan_adoption.py — paper-compliant filter for the RQ2/RQ3 targets (paper 4.2.1, criterion 3)

The paper targets, for RQ2/RQ3, only the "snapshot-testing adopter" repositories that have one or
more of toMatchSnapshot / toMatchInlineSnapshot in their test files. This script clones each
repository in repos_rq2.json, installs dependencies exactly as in paper 4.2.1, enumerates test
files with `npx jest --listTests`, and then decides whether a snapshot method is present using a
Babel AST. If install/enumeration fails, it falls back to static resolution of the Jest config
(distinguishable via test_list_source).

Output (JSONL, resumable):
  {"full_name": ..., "uses_jest": bool, "uses_snapshot": bool,
   "snapshot_tests": int, "test_list_source": "listTests"|"static"}

With --adopters-out, the uses_snapshot repositories are written out in the same format as
repos_rq2.json (usable directly as REPOS_JSON for rq2/rq3).

Usage:
  python scan_adoption.py --repos-json results/repos_rq2.json \
      --out results/snapshot_adoption.jsonl \
      --adopters-out results/repos_rq3_adopters.json \
      --backend singularity --image node.sif --scratch $SCRATCH_DIR --workers 6
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import common
import runner


def scan_one(repo: dict, backend: str | None, image: str | None,
             scratch_root: str | None) -> dict:
    full = repo["full_name"]
    parent = tempfile.mkdtemp(prefix="adopt_", dir=scratch_root)
    work = Path(parent) / "repo"
    scratch = Path(parent) / "scratch"
    try:
        common.clone_repo(repo["clone_url"], work)
        pkg = work / "package.json"
        uses_jest = (common.detect_jest_in_package_json(
            pkg.read_text(errors="ignore")) if pkg.exists() else False)
        if not uses_jest:
            return {"full_name": full, "uses_jest": False,
                    "uses_snapshot": False, "snapshot_tests": 0}

        test_files = None
        if backend:
            try:
                for sub in ("home", "npm-cache"):
                    (scratch / sub).mkdir(parents=True, exist_ok=True)
                env = runner._sandbox_env(backend)
                inst = runner.run_sandbox(
                    backend, image, str(work), str(scratch),
                    "npm ci --legacy-peer-deps || npm install --legacy-peer-deps",
                    timeout=1800, env=env)
                if inst.returncode == 0:
                    files, src = runner.list_tests_via_jest(
                        work, backend, image, str(scratch), env)
                    if src == "listTests":
                        test_files = files
            except Exception:
                # absorb install/listTests failures via the fallback to static resolution
                test_files = None

        pm = common.analyze_repo_dir(work, name=full, test_files=test_files)
        return {"full_name": full,
                "uses_jest": pm.uses_jest,
                "uses_snapshot": pm.uses_snapshot,
                "snapshot_tests": pm.snapshot_tests,
                "test_list_source": pm.test_list_source}
    except Exception as e:
        return {"full_name": full, "error": str(e)[:300]}
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adopters-out",
                    help="list of uses_snapshot repositories (same format as repos_rq2.json)")
    ap.add_argument("--backend", default=None,
                    choices=["apptainer", "singularity", "docker", "local"],
                    help="if given, enumerate via install + `npx jest --listTests` (same as paper 4.2.1)")
    ap.add_argument("--image", help="SIF (apptainer) / docker image")
    ap.add_argument("--scratch", help="work area (large scratch on HPC)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    repos = json.loads(Path(args.repos_json).read_text())
    out_path = Path(args.out)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["full_name"])
            except Exception:
                pass
    todo = [r for r in repos if r["full_name"] not in done]
    print(f"[scan_adoption] total={len(repos)} done={len(done)} todo={len(todo)} "
          f"backend={args.backend or 'none (static resolution only)'}", flush=True)

    n_snap = 0
    with out_path.open("a") as out, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_one, r, args.backend, args.image, args.scratch): r
                for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if rec.get("uses_snapshot"):
                n_snap += 1
            if i % 100 == 0:
                print(f"[{i}/{len(todo)}] snapshot adopters {n_snap} (this run)",
                      flush=True)

    # write the adopter list in the same format as repos_rq2.json
    if args.adopters_out:
        adopters: set[str] = set()
        for line in out_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("uses_snapshot"):
                adopters.add(rec["full_name"])
        adopter_repos = [r for r in repos if r["full_name"] in adopters]
        Path(args.adopters_out).write_text(
            json.dumps(adopter_repos, ensure_ascii=False, indent=2))
        print(f"[scan_adoption] snapshot adopters {len(adopter_repos)} "
              f"-> {args.adopters_out}", flush=True)

    print(f"[scan_adoption] done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
