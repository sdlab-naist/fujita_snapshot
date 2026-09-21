"""
rq2_rq3_effectiveness.py — RQ2/RQ3: effectiveness of snapshot testing

Reproduction of paper Chapter 4. Quantitatively evaluates the effectiveness of snapshot vs
non-snapshot tests from the perspectives of coverage (RQ2) and mutation score (RQ3).

Data collection (paper 4.2.1):
  - non-fork JS/TS repositories with star>=500
  - package.json scripts contain jest & `npx jest --listTests` succeeds
  - in the end 283 projects can be measured for coverage, 48 projects for mutation

The 5 "test types" (x-axis of Fig 4 / Fig 5):
  - All                 : run all tests
  - Snapshot            : run only snapshot tests (the rest are .skip'd)
  - Non-Snapshot        : run only non-snapshot tests
  - Only by Snapshot    : the region covered/detected only by snapshot = All - Non-Snapshot
  - Only by Non-Snapshot: the region only by non-snapshot            = All - Snapshot

Measurement:
  - Coverage (RQ2): statement coverage from `jest --coverage`
  - Mutation score (RQ3): Stryker Mutator

Reported main results:
  - RQ2: the median statement coverage of All (both combined) is about 85.4%. Snapshot/Non-Snapshot
         mostly overlap but have unique regions (Only by Snapshot median 3.8% / Only by
         Non-Snapshot 8.9%) and complement each other.
  - RQ3: median regular mutation score is All 45.9% / Snapshot 18.4% / Non-Snapshot 26.8%.
         The individual Snapshot and Non-Snapshot scores are about half of All, and they detect
         different mutants for a synergistic effect (no significant difference by Mann-Whitney U).

Run:
  python rq2_rq3_effectiveness.py --demo
  python rq2_rq3_effectiveness.py --collect-coverage OUT.json   # real measurement (needs Node/Jest)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import common
from common import REPORTED

RNG = np.random.default_rng(7)

CATEGORIES = ["All", "Snapshot", "Non-Snapshot",
              "Only by Snapshot", "Only by Non-Snapshot"]
# Fig 6 (mutation score over covered code) has only 3 categories
COVERED_CATEGORIES = ["All", "Snapshot", "Non-Snapshot"]


# Real-data measurement is handled by runner.py (isolated execution with an Apptainer/Singularity
# backend for servers without Docker; Only-by is computed as set differences by effectiveness_sets.py).
#   python runner.py --metric coverage --backend apptainer --image node.sif ...
#   python rq2_rq3_effectiveness.py --coverage-jsonl results_coverage.jsonl --plot


# ===========================================================================
# Synthetic data generation (reproduces the reported medians / distribution shapes)
# ===========================================================================
def _beta_to_pct(median_pct: float, n: int, conc: float = 4.0,
                 floor: float = 0.0) -> np.ndarray:
    """Generate a 0-100 distribution whose median matches the given median (%) *exactly*.

    Shape it with a Beta distribution, then calibrate the empirical median to the target with an
    additive shift (so the median can be matched exactly even for a skewed distribution). A smaller
    conc widens the variance.
    """
    m = np.clip(median_pct / 100.0, 0.01, 0.99)
    a = max(0.2, m * conc)
    b = max(0.2, (1 - m) * conc)
    x = RNG.beta(a, b, size=n) * 100.0
    x = np.clip(x, floor, 100.0)
    # calibrate the empirical median to the target
    x = np.clip(x + (median_pct - np.median(x)), floor, 100.0)
    return x


def synth_coverage(n: int = REPORTED["rq2"]["n_projects"]) -> dict[str, list[float]]:
    """RQ2: synthesize the 5-type statement-coverage distribution (Fig 4 equivalent)."""
    med = REPORTED["rq2"]["median_coverage"]
    data = {
        "All": _beta_to_pct(med["All"], n, conc=3.0),
        "Snapshot": _beta_to_pct(med["Snapshot"], n, conc=2.0),
        "Non-Snapshot": _beta_to_pct(med["Non-Snapshot"], n, conc=2.0),
        # unique regions skew strongly toward 0 (because most is shared)
        "Only by Snapshot": _beta_to_pct(med["Only by Snapshot"], n, conc=1.2),
        "Only by Non-Snapshot": _beta_to_pct(med["Only by Non-Snapshot"], n, conc=1.2),
    }
    return {k: v.tolist() for k, v in data.items()}


def synth_mutation(n: int = REPORTED["rq3"]["n_projects"]) -> dict[str, list[float]]:
    """RQ3: synthesize the 5-type mutation-score distribution (Fig 5 equivalent)."""
    med = REPORTED["rq3"]["median_mutation"]
    data = {
        "All": _beta_to_pct(med["All"], n, conc=2.5),
        "Snapshot": _beta_to_pct(med["Snapshot"], n, conc=2.0),
        "Non-Snapshot": _beta_to_pct(med["Non-Snapshot"], n, conc=2.0),
        "Only by Snapshot": _beta_to_pct(8.0, n, conc=1.2),
        "Only by Non-Snapshot": _beta_to_pct(12.0, n, conc=1.2),
    }
    return {k: v.tolist() for k, v in data.items()}


def synth_mutation_covered(n: int = REPORTED["rq3"]["n_projects"]) -> dict[str, list[float]]:
    """RQ3: the 3-type distribution of mutation score over covered code (Fig 6 equivalent).

    The paper does not report the exact medians of Fig 6 numerically (the text only says "the median
    of snapshot-only is slightly higher than non-snapshot"). Here, as a synthetic fixture, we use
    approximate values that satisfy that trend (Snapshot slightly higher).
    """
    data = {
        "All": _beta_to_pct(55.0, n, conc=2.5),
        "Snapshot": _beta_to_pct(32.0, n, conc=2.0),       # slightly higher than non-snapshot
        "Non-Snapshot": _beta_to_pct(29.0, n, conc=2.0),
    }
    return {k: v.tolist() for k, v in data.items()}


# ===========================================================================
# Analysis / output
# ===========================================================================
def analyze(data: dict[str, list[float]], reported: dict, title: str,
            categories: list[str] | None = None) -> dict:
    categories = categories or CATEGORIES
    print("=" * 70)
    print(title)
    print("=" * 70)
    out = {}
    for c in categories:
        if c not in data:
            continue
        if not data[c]:
            print(f"  {c:22s} no data (n=0) — skipped")
            continue
        d = common.describe(data[c])
        out[c] = d
        print(f"  {c:22s} median={d['median']:5.1f}%  "
              f"(Q1={d['q1']:.1f}, Q3={d['q3']:.1f}, n={d['n']})")

    # Mann-Whitney U test of Snapshot vs Non-Snapshot (no significant difference = complementary)
    if data.get("Snapshot") and data.get("Non-Snapshot"):
        p = common.mann_whitney(data["Snapshot"], data["Non-Snapshot"])[1]
        print("-" * 70)
        print(f"  Snapshot vs Non-Snapshot: Mann-Whitney U p={p:.3g} "
              f"({'no sig. diff -> complementary' if p >= 0.05 else 'significant difference'})")
        out["mwu_snapshot_vs_nonsnapshot_p"] = p
    print("=" * 70)
    return out


def plot_distribution(data: dict[str, list[float]], ylabel: str,
                      title: str, out_path: str,
                      categories: list[str] | None = None) -> None:
    """Fig 4/5/6 equivalent: violin + boxplot per test type."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed, so skipping plotting")
        return
    cats = [c for c in (categories or CATEGORIES) if c in data and data[c]]
    series = [data[c] for c in cats]
    if not series:
        print(f"no data, so skipping plot: {out_path}")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(series, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.3)
    ax.boxplot(series, widths=0.2, showfliers=False)
    ax.set_xticks(range(1, len(cats) + 1))
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved figure: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="reproduction of RQ2/RQ3 effectiveness")
    ap.add_argument("--demo", action="store_true", help="reproduce with synthetic data")
    ap.add_argument("--coverage-jsonl", metavar="FILE",
                    help="coverage result JSONL produced by runner.py (Only-by already computed as set diffs)")
    ap.add_argument("--mutation-jsonl", metavar="FILE",
                    help="mutation result JSONL produced by runner.py")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.coverage_jsonl or args.mutation_jsonl:
        # load real measurements (runner.py JSONL). Only-by is already computed as set differences.
        import runner
        cov = (runner.collect_distributions(args.coverage_jsonl)
               if args.coverage_jsonl else synth_coverage())
        mut = (runner.collect_distributions(args.mutation_jsonl)
               if args.mutation_jsonl else synth_mutation())
        mut_cov = (runner.collect_distributions(args.mutation_jsonl,
                                                key="metrics_covered")
                   if args.mutation_jsonl else synth_mutation_covered())
        if not args.coverage_jsonl or not args.mutation_jsonl:
            print("Note: metrics without a JSONL are filled in with synthetic fixtures.")
    else:  # demo
        print(common.SYNTH_BANNER)
        cov = synth_coverage()
        mut = synth_mutation()
        mut_cov = synth_mutation_covered()

    analyze(cov, REPORTED["rq2"]["median_coverage"],
            "RQ2: Statement Coverage")
    print()
    analyze(mut, REPORTED["rq3"]["median_mutation"],
            "RQ3: regular Mutation Score (Fig 5)")
    print()
    analyze(mut_cov, {}, "RQ3: Mutation Score over covered code (Fig 6)",
            categories=COVERED_CATEGORIES)

    if args.plot or args.demo:
        plot_distribution(cov, "Statement Coverage (%)",
                          "RQ2: Statement Coverage by Test Type",
                          "rq2_coverage.png")
        plot_distribution(mut, "Mutation Score (%)",
                          "RQ3: Mutation Score by Test Type",
                          "rq3_mutation.png")
        plot_distribution(mut_cov, "Mutation Score (%)",
                          "RQ3: Mutation Score on Covered Code",
                          "rq3_covered_mutation.png",
                          categories=COVERED_CATEGORIES)


if __name__ == "__main__":
    main()
