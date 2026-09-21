#!/usr/bin/env python3
"""compute_results.py — recompute all paper numbers from the bundled results (../results).

Outputs: variables.tex-equivalent macros, Fig7a/7b, p-values (ST vs NonST),
Table 4 (median/mean/pooled). Deps: numpy, scipy (env/requirements.txt).
Run: python analysis/compute_results.py
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code"))
import effectiveness_sets as es

def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]

cov = [r for r in load(ROOT/"results"/"results_coverage.jsonl") if r["status"] == "ok"]
mut = [r for r in load(ROOT/"results"/"results_mutation.jsonl") if r["status"] == "ok"]
STUDIED = 340
M = lambda x: round(float(np.median(x)), 1)
A = lambda x: round(float(np.mean(x)), 1)
Q = lambda x, p: round(float(np.percentile(x, p)), 1)
colc = lambda recs, key, cat: [r[key][cat] for r in recs
                               if key in r and cat in r[key] and r[key][cat] is not None]

print(f"# Counts\nrunnableRepoCount={len(cov)}  runnableRepoPercent={round(100*len(cov)/STUDIED,1)}"
      f"  runmutation={len(mut)}  unable-to-run={STUDIED-len(cov)}")

Ac = colc(cov,'metrics','All'); Sc = colc(cov,'metrics','Snapshot'); Nc = colc(cov,'metrics','Non-Snapshot')
print(f"\n# Coverage (Fig6)\ncoverageAllMedian={M(Ac)} Q1={Q(Ac,25)} Q3={Q(Ac,75)}"
      f"\ncoverageStMedian={M(Sc)} coverageNostMedian={M(Nc)}"
      f"\ncoverageUniqueStMedian={M(colc(cov,'metrics','Only by Snapshot'))}"
      f" coverageUniqueNostMedian={M(colc(cov,'metrics','Only by Non-Snapshot'))}")

Am = colc(mut,'metrics','All'); Sm = colc(mut,'metrics','Snapshot'); Nm = colc(mut,'metrics','Non-Snapshot')
print(f"\n# Mutation regular (Fig7a)\nmutationAllMedian={M(Am)} mean={A(Am)} Q1={Q(Am,25)} Q3={Q(Am,75)}"
      f"\nmutationStMedian={M(Sm)} mean={A(Sm)}  mutationNostMedian={M(Nm)} mean={A(Nm)}"
      f"\nmutationEffect={round(M(Am)/M(Nm),1)}")
Acv = colc(mut,'metrics_covered','All'); Scv = colc(mut,'metrics_covered','Snapshot'); Ncv = colc(mut,'metrics_covered','Non-Snapshot')
print(f"\n# Covered mutation (Fig7b)\ncoveredMutationAllMedian={M(Acv)}"
      f" coveredMutationStMedian={M(Scv)} coveredMutationNostMedian={M(Ncv)}")

try:
    from scipy.stats import mannwhitneyu, wilcoxon
    print(f"\n# Significance (ST vs NonST)")
    print(f"regular  MWU p={mannwhitneyu(Sm,Nm,alternative='two-sided').pvalue:.3f}")
    print(f"covered  Wilcoxon(paired) p={wilcoxon(np.array(Scv),np.array(Ncv)).pvalue:.3f}"
          f"  MWU p={mannwhitneyu(Scv,Ncv,alternative='two-sided').pvalue:.3f}")
except Exception as e:
    print("scipy unavailable:", e)

# Table 4: median / mean / pooled kill rate per mutator
print("\n# Table 4 (mutator kill rate, ST/NonST)")
per = [r["mutators"] for r in mut if "mutators" in r]
med = es.aggregate_mutator_table(per)
from collections import defaultdict
st_v, no_v, tot = defaultdict(list), defaultdict(list), defaultdict(list)
pk_st, pt_st, pk_no, pt_no = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
for m in per:
    for name, v in m.items():
        t = v.get("total", 0)
        if v.get("ST") is not None: st_v[name].append(v["ST"]); pk_st[name]+=v["ST"]/100*t; pt_st[name]+=t
        if v.get("Non-ST") is not None: no_v[name].append(v["Non-ST"]); pk_no[name]+=v["Non-ST"]/100*t; pt_no[name]+=t
        tot[name].append(t)
print(f"{'Mutator':22s} {'ST_med':>6} {'ST_mean':>7} {'ST_pool':>7} | "
      f"{'No_med':>6} {'No_mean':>7} {'No_pool':>7} | {'repos':>5} {'#mut':>4}")
for name in sorted(med, key=lambda k:-med[k]['Diff']):
    sm=M(st_v[name]); sa=A(st_v[name]); sp=round(100*pk_st[name]/(pt_st[name] or 1),1)
    nm=M(no_v[name]); na=A(no_v[name]); npp=round(100*pk_no[name]/(pt_no[name] or 1),1)
    print(f"{name:22s} {sm:>6} {sa:>7} {sp:>7} | {nm:>6} {na:>7} {npp:>7} | "
          f"{med[name]['n']:>5} {int(np.median(tot[name])):>4}")
