"""
rq1_adoption.py — RQ1: adoption of snapshot testing

Reproduction of paper Chapter 3.

Research questions:
  - What fraction of JEST projects adopt snapshot testing?
  - Do projects that use snapshot testing differ from unit-test-only projects in test-suite size
    (test cases per 1K-LOC, number of assertions)?

Method (paper 3.2):
  1. Independently collect non-fork JS/TS repositories with star>=1000 via GitHub Search [36] (SEART)
     (9,516 repos). Note: the Bui & Rocha [33] dataset is not used.
  2. Detect Jest usage from package.json (a jest command in scripts) -> 1,487 repos
  3. Count test/it, expect, toMatchSnapshot(Inline) via a Babel AST
  4. Classify each project as ST / UT / UT+ST
  5. Compute per-1K-LOC metrics and compare with Mann-Whitney U tests (Bonferroni-corrected alpha=0.005)

Reported main results:
  - 15.6% of JS/TS (1487/9516) are JEST; 38.3% of JEST (569/1487) adopt snapshots
  - median test cases per 1K-LOC: UT+ST 25.2 / UT 16.2 / ST 7.2  (UT+ST ~1.6x UT)
  - median unit-test count is about the same for UT/UT+ST (16.2 vs 16.7)
  - the test-case-count difference of UT vs UT+ST is not significant (p=0.23)

Run:
  python rq1_adoption.py --demo                              # synthetic fixtures (offline)
  python rq1_adoption.py --collect OUT.json --seart-csv seart.csv  # paper-compliant collection
  python rq1_adoption.py --collect OUT.json                  # GitHub Search API fallback
  python rq1_adoption.py --metrics OUT.json                  # analyze collected metrics
"""

from __future__ import annotations

import argparse
import numpy as np

import common
from common import ProjectMetrics, REPORTED

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Real-data collection pipeline
# ---------------------------------------------------------------------------
def _collect_one(repo: dict, backend: str | None, image: str | None,
                 scratch_root: str | None) -> dict:
    """Clone and measure one repository, returning a dict (compatible with asdict(ProjectMetrics)).

    When backend is given, install dependencies and enumerate test files with `npx jest --listTests`,
    exactly as in paper 4.2.1 (unifying the method across RQ1-RQ3). Repositories where install/
    enumeration fails fall back to static resolution of the Jest config (distinguishable via the
    test_list_source field).
    """
    import shutil
    import tempfile
    from dataclasses import asdict
    from pathlib import Path

    import runner  # borrow the sandbox execution (run_sandbox / list_tests_via_jest)

    full = repo["full_name"]
    parent = tempfile.mkdtemp(prefix="rq1_", dir=scratch_root)
    work = Path(parent) / "repo"
    scratch = Path(parent) / "scratch"
    try:
        common.clone_repo(repo["clone_url"], work)
        pkg = work / "package.json"
        uses_jest = (common.detect_jest_in_package_json(
            pkg.read_text(errors="ignore")) if pkg.exists() else False)
        if not uses_jest:
            # for non-JEST, leave only a resume marker (excluded from analysis)
            return {"name": full, "uses_jest": False}

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
        pm.stars = repo.get("stargazers_count", 0)
        return asdict(pm)
    except Exception as e:  # skip clone failures etc.
        return {"name": full, "uses_jest": False, "error": str(e)[:300]}
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def collect(out_path: str, max_repos: int | None = None,
            seart_csv: str | None = None, seed_path: str | None = None,
            backend: str | None = None, image: str | None = None,
            scratch: str | None = None, workers: int = 6
            ) -> list[ProjectMetrics]:
    """Collect target JS/TS repositories and measure them (test code is analyzed via a Babel AST).

    The paper-3.2.1-compliant collection is an independent collection via GitHub Search [36] (SEART).
      - seart_csv: the SEART GitHub Search CSV export (paper-compliant, recommended)
      - seed_path: a seed list from Bui & Rocha [33] etc. (optional, for comparison)
      - neither: the GitHub Search API fallback (capped at 1,000)

    When backend is given, JEST repositories install dependencies and enumerate test files with
    `npx jest --listTests` (the same method as paper 4.2.1). Progress is appended to <out_path>l
    (JSONL) incrementally, and re-runs skip the completed entries.
    """
    import json as _json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import repo_collection

    if backend in ("apptainer", "singularity", "docker") and not image:
        print(f"[rq1-collect] warning: backend={backend} but --image is unset -> "
              "give up on listTests and enumerate via static resolution only")
        backend = None

    repos = repo_collection.resolve_repo_list(seart_csv=seart_csv,
                                              seed_path=seed_path,
                                              min_stars=1000,
                                              max_repos=max_repos)

    # resumable progress file (rq1_metrics.json -> rq1_metrics.jsonl)
    jsonl = Path(str(out_path) + "l")
    done: set[str] = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                done.add(_json.loads(line)["name"])
            except Exception:
                continue
    todo = [r for r in repos if r["full_name"] not in done]
    print(f"[rq1-collect] total={len(repos)} done={len(done)} todo={len(todo)} "
          f"backend={backend or 'none (static resolution only)'} workers={workers}",
          flush=True)

    with jsonl.open("a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_collect_one, r, backend, image, scratch): r
                for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"name": r["full_name"], "uses_jest": False,
                       "error": str(e)[:300]}
            out.write(_json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if i % 50 == 0:
                print(f"[rq1-collect] {i}/{len(todo)}", flush=True)

    # aggregate only the JEST repositories from the JSONL and write the legacy-format JSON
    metrics: list[ProjectMetrics] = []
    for line in jsonl.read_text().splitlines():
        try:
            d = _json.loads(line)
        except Exception:
            continue
        if d.get("uses_jest"):
            metrics.append(ProjectMetrics(
                **{k: v for k, v in d.items()
                   if k in ProjectMetrics.__dataclass_fields__}))
    common.save_metrics(metrics, out_path)
    n_lt = sum(1 for m in metrics if m.test_list_source == "listTests")
    print(f"collected {len(metrics)} JEST projects "
          f"(listTests={n_lt} / static={len(metrics) - n_lt}) -> {out_path}")
    return metrics


# ---------------------------------------------------------------------------
# Synthetic data generation (reproduces the paper's distributions / medians)
# ---------------------------------------------------------------------------
def synth(n_jest: int = 1487) -> list[ProjectMetrics]:
    """Build a set of synthetic JEST projects consistent with the reported adoption rate / medians.

    - 38.3% adopt snapshots (= ST or UT+ST)
    - the remaining 61.7% are UT
    - among the snapshot adopters, a minority are ST and the majority are UT+ST
    """
    n_snapshot = round(n_jest * REPORTED["rq1"]["snapshot_adoption"])  # 569
    n_ut = n_jest - n_snapshot
    # treat about 10% of the snapshot adopters as "snapshot only (ST)"
    n_st = max(1, round(n_snapshot * 0.10))
    n_utst = n_snapshot - n_st

    metrics: list[ProjectMetrics] = []

    def make(cat: str, idx: int) -> ProjectMetrics:
        # spread LOC log-normally (median on the ~10k-line scale)
        loc = int(np.exp(RNG.normal(np.log(10000), 0.8)))
        pm = ProjectMetrics(name=f"{cat.lower().replace('+','_')}/repo{idx}",
                            stars=int(RNG.integers(1000, 50000)),
                            uses_jest=True, loc=loc)

        # helper: sample log-normally aiming at a per-1K-LOC median
        def per_kloc(median: float, sigma: float = 0.7) -> float:
            return float(np.exp(RNG.normal(np.log(max(median, 0.1)), sigma)))

        kloc = loc / 1000.0
        # Note: the paper's reported medians do not add up additively (e.g. total test cases 25.2 /
        # unit 16.7 / snapshot 2.1), because they are marginal medians of skewed distributions. Here
        # we prioritize reproducing the paper's main claims — (1) UT+ST's total test cases are ~1.6x
        # UT's (25.2 vs 16.2), (2) the unit-test count is about the same for UT/UT+ST with no
        # significant difference.
        if cat == "UT":
            # unit tests only. use the same distribution (median ~16.5) as UT+ST so there is no sig. diff
            ut = per_kloc(16.5, 0.6)
            pm.unit_tests = max(1, round(ut * kloc))
            pm.snapshot_tests = 0
            pm.assertions = round(pm.unit_tests *
                                  per_kloc(REPORTED["rq1"]["assertions_per_test_median"]["UT"], 0.3))
        elif cat == "ST":
            st = per_kloc(REPORTED["rq1"]["test_cases_per_kloc_median"]["ST"])
            pm.snapshot_tests = max(1, round(st * kloc))
            pm.unit_tests = 0
            pm.assertions = round(pm.snapshot_tests *
                                  per_kloc(REPORTED["rq1"]["assertions_per_test_median"]["ST"], 0.2))
        else:  # UT+ST
            # unit uses the same distribution as UT (median ~16.5), plus snapshots so that the total
            # test-case median becomes ~25 (= ~1.6x UT)
            ut = per_kloc(16.5, 0.6)
            stk = per_kloc(8.5, 0.7)
            pm.unit_tests = max(1, round(ut * kloc))
            pm.snapshot_tests = max(1, round(stk * kloc))
            tot = pm.unit_tests + pm.snapshot_tests
            pm.assertions = round(tot *
                                  per_kloc(REPORTED["rq1"]["assertions_per_test_median"]["UT+ST"], 0.3))
        pm.test_cases = pm.unit_tests + pm.snapshot_tests
        return pm

    for i in range(n_ut):
        metrics.append(make("UT", i))
    for i in range(n_st):
        metrics.append(make("ST", i))
    for i in range(n_utst):
        metrics.append(make("UT+ST", i))

    RNG.shuffle(metrics)
    return metrics


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(metrics: list[ProjectMetrics], total_js_ts: int | None = None) -> dict:
    """Run the RQ1 aggregation and statistical tests and return a result dict."""
    jest = [m for m in metrics if m.uses_jest]
    by_cat = {c: [m for m in jest if m.category == c] for c in ("UT", "ST", "UT+ST")}

    n_jest = len(jest)
    n_snapshot = sum(1 for m in jest if m.uses_snapshot)

    result: dict = {
        "n_jest_projects": n_jest,
        "n_snapshot_projects": n_snapshot,
        "snapshot_adoption_rate": n_snapshot / n_jest if n_jest else 0.0,
        "counts": {c: len(v) for c, v in by_cat.items()},
        "medians": {},
        "tests": {},
    }
    if total_js_ts:
        result["jest_ratio"] = n_jest / total_js_ts

    # --- medians of per-1K-LOC metrics -----------------------------------
    for c in ("UT", "ST", "UT+ST"):
        v = by_cat[c]
        result["medians"][c] = {
            "test_cases_per_kloc": float(np.median([m.test_cases_per_kloc for m in v])) if v else None,
            "unit_tests_per_kloc": float(np.median([m.unit_tests_per_kloc for m in v])) if v else None,
            "snapshot_tests_per_kloc": float(np.median([m.snapshot_tests_per_kloc for m in v])) if v else None,
            "assertions_per_test": float(np.median([m.assertions_per_test for m in v])) if v else None,
        }

    # --- Mann-Whitney U tests (Bonferroni-corrected alpha=0.005) ---------
    comparisons = [
        ("UT", "UT+ST", "test_cases_per_kloc"),
        ("UT", "ST", "test_cases_per_kloc"),
        ("ST", "UT+ST", "test_cases_per_kloc"),
        ("UT", "UT+ST", "unit_tests_per_kloc"),
    ]
    pvals = []
    raw = []
    for a, b, metric in comparisons:
        va = [getattr(m, metric) for m in by_cat[a]]
        vb = [getattr(m, metric) for m in by_cat[b]]
        if va and vb:
            u, p = common.mann_whitney(va, vb)
        else:
            u, p = float("nan"), float("nan")
        pvals.append(p)
        raw.append((a, b, metric, u, p))

    # The paper reports alpha=0.005 as the significance level after Bonferroni correction, so compare
    # each p directly to 0.005 (num_comparisons=1 to avoid double correction).
    sig = common.bonferroni([p for p in pvals if not np.isnan(p)],
                            alpha=REPORTED["rq1"]["alpha_bonferroni"],
                            num_comparisons=1)
    sig_iter = iter(sig)
    result["tests"] = []
    for a, b, metric, u, p in raw:
        is_sig = next(sig_iter) if not np.isnan(p) else None
        result["tests"].append({
            "compare": f"{a} vs {b}", "metric": metric,
            "U": u, "p": p, "significant_alpha0.005": is_sig,
        })

    # how many times denser UT+ST's test cases are than UT's
    m_utst = result["medians"]["UT+ST"]["test_cases_per_kloc"]
    m_ut = result["medians"]["UT"]["test_cases_per_kloc"]
    if m_ut:
        result["ratio_testcases_utst_over_ut"] = m_utst / m_ut

    return result


def print_report(result: dict) -> None:
    print("=" * 70)
    print("RQ1: adoption of snapshot testing")
    print("=" * 70)
    n = result["n_jest_projects"]
    s = result["n_snapshot_projects"]
    print(f"number of JEST projects       : {n}")
    if "jest_ratio" in result:
        print(f"JEST ratio among JS/TS        : {result['jest_ratio']:.1%}")
    print(f"snapshot adoption             : {s} ({result['snapshot_adoption_rate']:.1%})")
    print(f"counts by category (UT/ST/UT+ST): {result['counts']}")
    print("-" * 70)
    print("test cases per 1K-LOC (median):")
    for c in ("UT+ST", "UT", "ST"):
        v = result["medians"][c]["test_cases_per_kloc"]
        print(f"  {c:6s}: {v:5.1f}")
    if "ratio_testcases_utst_over_ut" in result:
        print(f"  UT+ST / UT ratio = {result['ratio_testcases_utst_over_ut']:.2f}")
    print("-" * 70)
    print("unit-test count (per 1K-LOC, median):")
    for c in ("UT", "UT+ST"):
        print(f"  {c:6s}: {result['medians'][c]['unit_tests_per_kloc']:5.1f}")
    print("assertions per test (median):")
    for c in ("UT", "UT+ST", "ST"):
        print(f"  {c:6s}: {result['medians'][c]['assertions_per_test']:5.2f}")
    print("-" * 70)
    print("Mann-Whitney U tests (Bonferroni alpha=0.005):")
    for t in result["tests"]:
        print(f"  {t['compare']:14s} [{t['metric']:22s}] "
              f"p={t['p']:.3g} significant={t['significant_alpha0.005']}")
    print("=" * 70)


def plot(metrics: list[ProjectMetrics], out_path: str = "rq1_distribution.png") -> None:
    """Fig 3 equivalent: test cases / assertions per 1K-LOC by project type."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed, so skipping plotting")
        return

    cats = ["ST", "UT", "UT+ST"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric, title, cap in [
        (axes[0], "test_cases_per_kloc", "Test cases per 1K-LOC", 60),
        (axes[1], "assertions_per_test", "Number of Assertions", 6),
    ]:
        data = [[min(getattr(m, metric), cap) for m in metrics if m.category == c]
                for c in cats]
        ax.violinplot(data, showmedians=True)
        ax.boxplot(data, widths=0.15, showfliers=False)
        ax.set_xticks(range(1, len(cats) + 1))
        ax.set_xticklabels(cats)
        ax.set_xlabel("Project Type")
        ax.set_title(f"Metrics = {title}")
    fig.suptitle("RQ1: distribution of test-suite size by project type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved figure: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="reproduction of RQ1 adoption")
    ap.add_argument("--demo", action="store_true", help="reproduce with synthetic data")
    ap.add_argument("--collect", metavar="OUT", help="collect and save metrics")
    ap.add_argument("--seart-csv", metavar="FILE",
                    help="SEART GitHub Search [36] CSV export (paper-compliant, recommended)")
    ap.add_argument("--seed", metavar="FILE",
                    help="seed list from Bui & Rocha [33] etc. (optional, for comparison)")
    ap.add_argument("--max-repos", type=int, default=None, help="collection cap (for testing)")
    ap.add_argument("--metrics", metavar="JSON", help="analyze saved metrics")
    ap.add_argument("--plot", action="store_true", help="output distribution plots")
    ap.add_argument("--backend", default=None,
                    choices=["apptainer", "singularity", "docker", "local"],
                    help="if given, enumerate via install + `npx jest --listTests` (same as paper 4.2.1)")
    ap.add_argument("--image", help="SIF (apptainer) / docker image")
    ap.add_argument("--scratch", help="work area (large scratch on HPC)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if args.collect:
        metrics = collect(args.collect, max_repos=args.max_repos,
                          seart_csv=args.seart_csv, seed_path=args.seed,
                          backend=args.backend, image=args.image,
                          scratch=args.scratch, workers=args.workers)
        total = REPORTED["rq1"]["js_ts_projects"]
    elif args.metrics:
        metrics = common.load_metrics(args.metrics)
        total = None
    else:  # demo is the default
        print(common.SYNTH_BANNER)
        metrics = synth()
        total = REPORTED["rq1"]["js_ts_projects"]

    result = analyze(metrics, total_js_ts=total)
    print_report(result)
    if args.plot or args.demo:
        plot(metrics)


if __name__ == "__main__":
    main()
