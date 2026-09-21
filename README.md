# Replication Package — Snapshot Testing Effectiveness (paper-4.2-compliant)

Reproduction package for RQ1 (coverage) / RQ2 (mutation) of the paper
*"On the Effectiveness of Snapshot Testing: An Empirical Study"* (Fujita et al.).
The methodology strictly follows Section 4.2 (Data Analysis) of the paper.

## Dataset snapshot (point in time)
- Measured code = each repository's **latest commit as of 2025-07-01** (`CLONE_BEFORE=2025-07-01`).
- Candidate pool = jestable 369 (original ~2023 collection; `data/repos_jestable.json`).
- Sizes: studied (adopters) = **340**, coverage succeeded = **300**, mutation succeeded = **70**.

## Methodology (paper-4.2-compliant points)
- Test enumeration: `npx jest --listTests` (Section 4.2.1).
- ST/NonST classification: **per test/it block via AST (babel)**. A test is a snapshot test iff it
  uses `toMatchSnapshot` or `toMatchInlineSnapshot` (**these two methods only**; `code/tools/apply_skip.js`).
- Subset execution: excluded tests are marked with **`.skip`** (not body-emptying; paper line 63).
- Coverage: `npx jest --coverage` (statement coverage, Istanbul JSON).
- Mutation: Stryker, minimal config equivalent to `stryker init`, mutating the **entire production code**
  (no covered-only restriction; paper lines 90/120).
- Aggregation (`code/effectiveness_sets.py`):
  - Only by X = All − other (marginal contribution; paper line 150).
  - ST∩NonST = ST + NonST − All (implied formula; `.skip` guarantees no negative values).
  - Covered score: denominator = mutants each test set **covers by itself** (paper line 90).
  - Table 4 = per-mutator median kill rate (`mutation_by_mutator` / `aggregate_mutator_table`).

## Environment
- HPC (SLURM) + Apptainer/Singularity. Node 20, Python venv.
- Build the SIF from `env/apptainer.def`: `apptainer build node.sif env/apptainer.def`
- Python deps: `pip install -r env/requirements.txt`. JS: `cd code/tools && npm install`.

## Reproduction steps
```bash
# 0) Setup
apptainer build node.sif env/apptainer.def
python -m pip install -r env/requirements.txt
( cd code/tools && npm install )

# 1) Adoption detection (to regenerate studied=340; a prebuilt copy is bundled in data/)
CLONE_BEFORE=2025-07-01 python code/scan_adoption.py \
    --repos-json data/repos_jestable.json \
    --out out/snapshot_adoption.jsonl \
    --adopters-out out/repos_rq3_adopters.json \
    --backend singularity --image node.sif --scratch $SCRATCH

# 2) Coverage (RQ1) — pinned to 2025-07-01
CLONE_BEFORE=2025-07-01 python code/run_all.py --outdir out --stages rq2 \
    --repos-json data/repos_cov_ok.json \
    --backend singularity --image node.sif --scratch $SCRATCH --workers-coverage 6
#   -> out/results_coverage.jsonl  (metrics + metrics_sets)

# 3) Mutation (RQ2) — whole-tree mutation
CLONE_BEFORE=2025-07-01 MUTATE_SCOPE=full python code/run_all.py --outdir out --stages rq3 \
    --repos-json data/repos_mut_ok.json \
    --backend singularity --image node.sif --scratch $SCRATCH --workers-mutation 3
#   -> out/results_mutation.jsonl  (metrics + metrics_covered + mutators)

# 4) Generate figures (Fig6/7a/7b)
python code/make_paper_figs.py --outdir out --dest 100_Fig

# 5) Recompute all paper numbers (variables.tex macros, p-values, Table 4 median/mean/pooled)
python analysis/compute_results.py
```
For batch submission on SLURM see `env/run_coverage.sbatch` / `env/run_mutation.sbatch`
(they set `CLONE_BEFORE=2025-07-01` and `MUTATE_SCOPE=full`; cluster_long, resumable).

## Bundled measurement results (results/)
- `results_coverage.jsonl` — coverage (300 ok)
- `results_mutation.jsonl` — mutation (70 ok)
- `figures/` — regenerated Fig6/7a/7b (pdf+png)
- see `analysis/` to recompute all paper macros / Table 4 / p-values from the above

With `results/` present, steps 4-5 alone reproduce the figures and numbers (no re-measurement needed).

## Analysis
- `analysis/compute_results.py` — recompute all paper numbers from the bundled results
  (macros / Fig7a/7b / p-values / Table 4 in three statistics: median, mean, pooled).
- `analysis/expected_output.txt` — its expected output.

## Key results
| Metric | Value |
|---|---|
| coverage All median | 80.6% |
| coverage ST / NonST | 48.2% / 55.4% |
| mutation All median | 32.1% |
| covered mutation ST / NonST | 44.0% / 45.8% |
| p (ST vs NonST regular) | 0.774 |

Snapshot and non-snapshot tests are complementary — they cover distinct parts of the production code
and kill different types of mutants — and their covered mutation scores show no statistically
significant difference. Absolute values reflect the measurement snapshot (2025-07) and toolchain
(Stryker/jest/node) described above. Run `analysis/compute_results.py` to recompute every number
from the bundled results.

## Provenance
- The 369 repositories in `data/repos_jestable.json` come from the original authors' collection
  (2023 ICSME survey).
- Each repository is measured by cloning the public repository.
