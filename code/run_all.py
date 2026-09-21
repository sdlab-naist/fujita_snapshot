"""
run_all.py — master driver that runs RQ1-RQ3 in batch

Runs the reproduction pipeline for the snapshot-testing empirical study in a single command.
Intended to be called from HPC jobs (qsub / sbatch); each stage is resumable and idempotent.

Stages (select with --stages, default all):
  rq1     : target collection (SEART, star>=1000) -> JEST detection -> AST counting -> analysis/plots
  collect : target collection for RQ2/RQ3 (SEART, star>=500) -> (optional) JEST pre-filter
  rq2     : measure coverage via runner.py (Apptainer/Singularity isolation, parallel, resumable)
  rq3     : measure mutation via runner.py (Stryker; both Fig 5 and Fig 6)
  report  : aggregate distributions from the result JSONL and plot (Fig 3-6)

All artifacts are collected under --outdir. Re-running with the same --outdir after an interruption
skips already-completed stages/repositories and continues.

Example (HPC):
  python run_all.py --outdir results \\
      --seart-csv seart_1000.csv \\
      --seart-csv-rq2 seart_500.csv --prefilter-jest \\
      --backend apptainer --image node.sif --scratch /scratch/$USER \\
      --workers-coverage 4 --workers-mutation 2

  # offline smoke test (synthetic fixtures; not paper values)
  python run_all.py --demo --outdir results_demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def make_logger(log_path: Path):
    fh = open(log_path, "a", encoding="utf-8")

    def log(msg: str = "") -> None:
        line = f"[{_ts()}] {msg}" if msg else ""
        print(line)
        fh.write(line + "\n")
        fh.flush()

    return log, fh


# ---------------------------------------------------------------------------
# RQ1
# ---------------------------------------------------------------------------
def stage_rq1(args, outdir: Path, log) -> None:
    import common
    import rq1_adoption

    log("=== STAGE rq1: adoption ===")
    metrics_path = outdir / "rq1_metrics.json"

    if metrics_path.exists() and not args.force:
        log(f"[rq1] reusing existing metrics: {metrics_path}")
        metrics = common.load_metrics(metrics_path)
    elif args.demo:
        log("[rq1] --demo: using synthetic fixtures (not paper values)")
        print(common.SYNTH_BANNER)
        metrics = rq1_adoption.synth()
    else:
        if not (args.seart_csv or args.seed):
            log("[rq1] neither --seart-csv nor --seed -> GitHub Search API fallback")
        # Paper 3.2.1: RQ1 analyzes cloned repositories directly with a Babel AST, counting test
        # methods (test/it) and toMatchSnapshot/toMatchInlineSnapshot. install + `npx jest --listTests`
        # is the method of 4.2.1 (RQ2/RQ3); applying it to RQ1 would make the tests of non-installable
        # repositories count as 0 and wrongly fall into the UT group, pushing down the median UT density
        # and the adoption rate. So RQ1 is fixed to backend=None, i.e. pure AST enumeration
        # (static resolution via jest_test_matcher).
        metrics = rq1_adoption.collect(str(metrics_path),
                                       max_repos=args.max_repos,
                                       seart_csv=args.seart_csv,
                                       seed_path=args.seed,
                                       backend=None,
                                       image=None,
                                       scratch=args.scratch,
                                       workers=args.workers_rq1)

    total = common.REPORTED["rq1"]["js_ts_projects"] if (args.demo or not metrics_path.exists()) else None
    result = rq1_adoption.analyze(metrics, total_js_ts=total)
    rq1_adoption.print_report(result)
    rq1_adoption.plot(metrics, str(outdir / "rq1_distribution.png"))
    (outdir / "rq1_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2))
    log(f"[rq1] done -> {outdir/'rq1_result.json'}, {outdir/'rq1_distribution.png'}")


# ---------------------------------------------------------------------------
# RQ2/RQ3 target collection
# ---------------------------------------------------------------------------
def stage_collect(args, outdir: Path, log) -> list[dict]:
    import repo_collection

    log("=== STAGE collect: RQ2/RQ3 target collection (star>=500) ===")
    repos_path = outdir / "repos_rq2.json"

    if args.repos_json:
        log(f"[collect] using existing repos-json: {args.repos_json}")
        return json.loads(Path(args.repos_json).read_text())

    if repos_path.exists() and not args.force:
        log(f"[collect] reusing existing {repos_path}")
        return json.loads(repos_path.read_text())

    if not args.seart_csv_rq2:
        raise SystemExit("collect requires either --repos-json or --seart-csv-rq2")

    repos = repo_collection.load_seart_csv(args.seart_csv_rq2, min_stars=500)
    log(f"[collect] {len(repos)} repos from SEART (star>=500)")
    if args.max_repos:
        repos = repos[:args.max_repos]
    if args.prefilter_jest:
        repos = repo_collection.filter_jest_repos(repos, scratch=args.scratch, log=log)
    repos_path.write_text(json.dumps(repos, ensure_ascii=False, indent=2))
    log(f"[collect] saved {len(repos)} repos -> {repos_path}")
    return repos


# ---------------------------------------------------------------------------
# RQ2 / RQ3 measurement
# ---------------------------------------------------------------------------
def stage_runner(args, outdir: Path, repos: list[dict], metric: str,
                 workers: int, log) -> Path:
    import runner

    out_jsonl = outdir / f"results_{metric}.jsonl"
    log(f"=== STAGE {metric}: measure via runner.py (backend={args.backend}, "
        f"workers={workers}) ===")
    runner.process_repos(repos, metric, args.backend, args.image,
                         str(out_jsonl), workers=workers,
                         scratch_root=args.scratch)
    log(f"[{metric}] done -> {out_jsonl}")
    return out_jsonl


# ---------------------------------------------------------------------------
# Aggregation / plotting (Fig 3-6)
# ---------------------------------------------------------------------------
def stage_report(args, outdir: Path, log) -> None:
    import common
    import runner
    import rq2_rq3_effectiveness as eff

    log("=== STAGE report: aggregate distributions and plot (Fig 4-6) ===")
    cov_jsonl = outdir / "results_coverage.jsonl"
    mut_jsonl = outdir / "results_mutation.jsonl"

    # Paper 4.2.1 criterion 3: only snapshot-adopter repositories are RQ2/RQ3 targets.
    # If the scan_adoption.py output exists, filter by it (otherwise use all repos).
    adoption_jsonl = outdir / "snapshot_adoption.jsonl"
    include: set[str] | None = None
    if adoption_jsonl.exists():
        include = set()
        for line in adoption_jsonl.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("uses_snapshot"):
                include.add(rec["full_name"])
        log(f"[report] paper-compliant filter: narrowed to {len(include)} snapshot adopters"
            f" ({adoption_jsonl})")
    else:
        log("[report] snapshot_adoption.jsonl missing, so aggregating all repos"
            " (note: this differs from paper 4.2.1)")

    if args.demo:
        print(common.SYNTH_BANNER)
        cov = eff.synth_coverage()
        mut = eff.synth_mutation()
        mut_cov = eff.synth_mutation_covered()
    else:
        cov = (runner.collect_distributions(str(cov_jsonl), include=include)
               if cov_jsonl.exists() else None)
        mut = (runner.collect_distributions(str(mut_jsonl), include=include)
               if mut_jsonl.exists() else None)
        mut_cov = (runner.collect_distributions(str(mut_jsonl),
                                                key="metrics_covered",
                                                include=include)
                   if mut_jsonl.exists() else None)

    if cov:
        eff.analyze(cov, common.REPORTED["rq2"]["median_coverage"],
                    "RQ2: Statement Coverage")
        eff.plot_distribution(cov, "Statement Coverage (%)",
                              "RQ2: Statement Coverage by Test Type",
                              str(outdir / "rq2_coverage.png"))
    else:
        log("[report] coverage JSONL missing, so skipping RQ2")

    if mut:
        eff.analyze(mut, common.REPORTED["rq3"]["median_mutation"],
                    "RQ3: regular Mutation Score (Fig 5)")
        eff.plot_distribution(mut, "Mutation Score (%)",
                              "RQ3: Mutation Score by Test Type",
                              str(outdir / "rq3_mutation.png"))
    if mut_cov:
        eff.analyze(mut_cov, {},
                    "RQ3: Mutation Score over covered code (Fig 6)",
                    categories=eff.COVERED_CATEGORIES)
        eff.plot_distribution(mut_cov, "Mutation Score (%)",
                              "RQ3: Mutation Score on Covered Code",
                              str(outdir / "rq3_covered_mutation.png"),
                              categories=eff.COVERED_CATEGORIES)
    if not mut:
        log("[report] mutation JSONL missing, so skipping RQ3")
    log("[report] done")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="batch driver that runs RQ1-RQ3")
    ap.add_argument("--outdir", required=True, help="output directory for artifacts")
    ap.add_argument("--stages", default="all",
                    help="comma-separated: rq1,collect,rq2,rq3,report / all")
    ap.add_argument("--demo", action="store_true",
                    help="run all stages on synthetic fixtures (offline smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="ignore existing intermediate artifacts and regenerate")

    # RQ1 collection
    ap.add_argument("--seart-csv", help="RQ1: SEART CSV (star>=1000)")
    ap.add_argument("--seed", help="RQ1: seed list from Bui & Rocha etc. (for comparison)")

    # RQ2/RQ3 collection
    ap.add_argument("--seart-csv-rq2", help="RQ2/RQ3: SEART CSV (star>=500)")
    ap.add_argument("--repos-json", help="RQ2/RQ3: JSON of [{full_name,clone_url},...]")
    ap.add_argument("--prefilter-jest", action="store_true",
                    help="clone during RQ2/RQ3 collection and keep only JEST repos")

    # execution backend
    ap.add_argument("--backend", default="apptainer",
                    choices=["apptainer", "singularity", "docker", "local"])
    ap.add_argument("--image", help="SIF (apptainer) / docker image")
    ap.add_argument("--scratch", help="work area (large scratch on HPC)")
    ap.add_argument("--workers-coverage", type=int, default=4)
    ap.add_argument("--workers-mutation", type=int, default=2)
    ap.add_argument("--workers-rq1", type=int, default=6)
    ap.add_argument("--max-repos", type=int, default=None, help="cap for testing")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log, fh = make_logger(outdir / "run_all.log")

    stages = ([s.strip() for s in args.stages.split(",") if s.strip()]
              if args.stages != "all"
              else ["rq1", "collect", "rq2", "rq3", "report"])
    log(f"run_all start stages={stages} demo={args.demo} backend={args.backend}")

    try:
        if "rq1" in stages:
            stage_rq1(args, outdir, log)

        repos = None
        need_repos = any(s in stages for s in ("collect", "rq2", "rq3")) and not args.demo
        if need_repos:
            repos = stage_collect(args, outdir, log)

        if "rq2" in stages and not args.demo:
            stage_runner(args, outdir, repos, "coverage",
                         args.workers_coverage, log)
        if "rq3" in stages and not args.demo:
            stage_runner(args, outdir, repos, "mutation",
                         args.workers_mutation, log)

        if "report" in stages:
            stage_report(args, outdir, log)

        log("run_all done")
    except SystemExit as e:
        log(f"aborted: {e}")
        raise
    except Exception as e:  # keep failures in the log
        import traceback
        log(f"error: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        fh.close()


if __name__ == "__main__":
    main()
