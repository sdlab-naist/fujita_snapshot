"""make_paper_figs.py — generate the replacement figures Fig.6 / Fig.7a / Fig.7b from the reproduction data.

Figure style:
  - violin (pale) + boxplot (solid fill) overlaid per category
  - boxplot: box = solid fill, median = thick black line, whiskers = black (min..max), no outliers
  - horizontal dashed grid, y-axis 0-100 (step 20)
  - x labels are math-italic: $ST$, $NonST$, $ST \\cup NonST$, $ST \\cap NonST$,
    $ST \\setminus NonST$, $NonST \\setminus ST$
  - all categories use a single color (blue)
  - Fig.6  (RQ1 coverage)          : 6 categories, y="Statement Coverage (%)"
  - Fig.7a (RQ2 mutation)          : 6 categories, y="Mutation Score (%)"
  - Fig.7b (RQ2 covered mutation)  : 3 categories, y="Covered Mutation Score (%)"

ST∩NonST (Common) is derived from the stored metrics: All − (ST\\NonST) − (NonST\\ST).
No re-measurement is needed.

Output: <OUTDIR>/fig6_coverage.{pdf,png}, fig7a_mutation.{pdf,png}, fig7b_covered.{pdf,png}
With --dest it also writes cov_graph/mut_graph/mut_covered_graph.pdf directly.

Usage: python make_paper_figs.py --outdir results [--dest <100_Fig>]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Category definitions (order of the paper figure) ----------------------
CAT6 = ["Snapshot", "Non-Snapshot", "All", "Common",
        "Only by Snapshot", "Only by Non-Snapshot"]
CAT3 = ["Snapshot", "Non-Snapshot", "All"]

LABEL = {
    "Snapshot":             r"$ST$",
    "Non-Snapshot":         r"$NonST$",
    "All":                  r"$ST \cup NonST$",
    "Common":               r"$ST \cap NonST$",
    "Only by Snapshot":     r"$ST \setminus NonST$",
    "Only by Non-Snapshot": r"$NonST \setminus ST$",
}

# single color for all categories (blue)
BLUE = "#3f8fcc"
COLOR = {c: BLUE for c in
         ("Snapshot", "Non-Snapshot", "All", "Common",
          "Only by Snapshot", "Only by Non-Snapshot")}


def load_include(outdir: Path):
    adoption = outdir / "snapshot_adoption.jsonl"
    if not adoption.exists():
        return None
    inc = set()
    for line in adoption.read_text().splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("uses_snapshot"):
            inc.add(rec["full_name"])
    return inc


def load_dist(path: Path, key: str, include):
    """Aggregate the 6-category distributions from the ok results in the JSONL (deriving ST∩NonST)."""
    per = []
    for line in Path(path).read_text().splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if include is not None and rec.get("full_name") not in include:
            continue
        if rec.get("status") != "ok" or key not in rec:
            continue
        m = dict(rec[key])
        if all(k in m and m[k] is not None
               for k in ("All", "Only by Snapshot", "Only by Non-Snapshot")):
            m["Common"] = m["All"] - m["Only by Snapshot"] - m["Only by Non-Snapshot"]
        per.append(m)
    out = {}
    for c in CAT6:
        out[c] = [float(p[c]) for p in per if c in p and p[c] is not None]
    return out


def draw(ax, dist, cats, ylabel):
    data = [np.asarray(dist.get(c, []), dtype=float) for c in cats]
    data = [d[~np.isnan(d)] for d in data]
    pos = np.arange(1, len(cats) + 1)

    # background: horizontal dashed grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="#c9c9c9", linewidth=0.8, alpha=0.9)

    # violin (pale, no outline)
    parts = ax.violinplot(data, positions=pos, showextrema=False, widths=0.9)
    for pc, c in zip(parts["bodies"], cats):
        pc.set_facecolor(COLOR[c])
        pc.set_edgecolor("none")
        pc.set_alpha(0.30)

    # boxplot (solid fill + black median + black whiskers, whiskers to min..max)
    bp = ax.boxplot(
        data, positions=pos, widths=0.42, showfliers=False, whis=(0, 100),
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2.2),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        boxprops=dict(edgecolor="black", linewidth=1.0),
    )
    for patch, c in zip(bp["boxes"], cats):
        patch.set_facecolor(COLOR[c])
        patch.set_alpha(1.0)

    ax.set_xticks(pos)
    ax.set_xticklabels([LABEL[c] for c in cats], fontsize=14)
    ax.set_xlim(0.4, len(cats) + 0.6)
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="y", labelsize=12)


def save(fig, base: Path):
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {base.with_suffix('.pdf').name} / {base.with_suffix('.png').name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dest", default=None,
                    help="The TeX 100_Fig directory. If given, also writes cov_graph/mut_graph/"
                         "mut_covered_graph.pdf directly.")
    a = ap.parse_args()
    outdir = Path(a.outdir)
    inc = load_include(outdir)
    print(f"[make_paper_figs] outdir={outdir} include={len(inc) if inc else 'ALL'}")

    cov = load_dist(outdir / "results_coverage.jsonl", "metrics", inc)
    mut = load_dist(outdir / "results_mutation.jsonl", "metrics", inc)
    mutc = load_dist(outdir / "results_mutation.jsonl", "metrics_covered", inc)

    print(f"  n(coverage)={len(cov['All'])}  n(mutation)={len(mut['All'])}")

    # aspect ratio of the original figures: 6 categories = 2.36:1, covered (3 categories) = 1.34:1
    WIDE = (10.6, 4.49)    # ~2.36:1 (larger absolute size to leave room for labels)
    NARROW = (5.60, 4.18)  # ~1.34:1
    dest = Path(a.dest) if a.dest else None

    specs = [
        (cov,  CAT6, "Statement Coverage (%)",       "fig6_coverage",  "cov_graph",         WIDE),
        (mut,  CAT6, "Mutation Score (%)",           "fig7a_mutation", "mut_graph",         WIDE),
        (mutc, CAT3, "Covered Mutation Score (%)",   "fig7b_covered",  "mut_covered_graph", NARROW),
    ]
    for dist, cats, ylabel, localname, destname, figsize in specs:
        fig, ax = plt.subplots(figsize=figsize)
        draw(ax, dist, cats, ylabel)
        save(fig, outdir / localname)
        if dest:
            fig, ax = plt.subplots(figsize=figsize)
            draw(ax, dist, cats, ylabel)
            save(fig, dest / destname)

    def meds(dist, cats):
        return {LABEL[c]: (round(float(np.median(dist[c])), 1) if dist.get(c) else None)
                for c in cats}
    print("  Fig6 medians:", meds(cov, CAT6))
    print("  Fig7a medians:", meds(mut, CAT6))
    print("  Fig7b medians:", meds(mutc, CAT3))


if __name__ == "__main__":
    main()
