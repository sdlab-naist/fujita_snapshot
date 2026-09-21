#!/bin/bash
#SBATCH --job-name=snapshot-coverage
#SBATCH --partition=cluster_long
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#
# Coverage measurement. The runner stores set-based metrics_sets so ST∩NonST (Common) is computed
# correctly. Targets the snapshot-adopter repos that succeed for coverage (data/repos_cov_ok.json).
# Pinned to the 2025-07-01 commit.
# Output: <OUTDIR>/results_coverage.jsonl (metrics + metrics_sets)
# Resumable. Re-submit to continue if it hits the time limit.

export CLONE_BEFORE="${CLONE_BEFORE:-2025-07-01}"
export OUTDIR="${OUTDIR:-results}"
export WORKERS_COVERAGE="${WORKERS_COVERAGE:-6}"
source job_env.sh
echo "[coverage] CLONE_BEFORE=$CLONE_BEFORE OUTDIR=$OUTDIR"

python run_all.py \
    --outdir "$OUTDIR" --stages rq2 \
    --repos-json data/repos_cov_ok.json \
    --backend "$BACKEND" --image "$SIF_IMAGE" --scratch "$SCRATCH_DIR" \
    --workers-coverage "$WORKERS_COVERAGE"
