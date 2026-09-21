#!/bin/bash
#SBATCH --job-name=snapshot-mutation
#SBATCH --partition=cluster_long
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#
# Mutation measurement. The runner stores per-mutatorName kill rates (mutators) so paper Table 4
# (tab:mutant_type_comparison) can be computed. Targets the snapshot-adopter repos that succeed for
# mutation (data/repos_mut_ok.json).
# Paper-4.2-compliant: mutation target = Stryker default (entire production code; MUTATE_SCOPE=full).
# Pinned to the 2025-07-01 commit. ST/NonST classification uses the paper-compliant apply_skip
# (2 matchers + .skip).
# Output: <OUTDIR>/results_mutation.jsonl (metrics + metrics_covered + mutators)
# Very heavy. Resumable. Re-submit to continue if it hits the time limit.

export CLONE_BEFORE="${CLONE_BEFORE:-2025-07-01}"
export OUTDIR="${OUTDIR:-results}"
export MUTATE_SCOPE="full"   # force whole-tree mutation (paper-compliant: mutate everything)
export WORKERS_MUTATION="${WORKERS_MUTATION:-3}"
source job_env.sh
echo "[mutation] CLONE_BEFORE=$CLONE_BEFORE OUTDIR=$OUTDIR MUTATE_SCOPE=$MUTATE_SCOPE (paper-compliant=whole-tree mutation)"

python run_all.py \
    --outdir "$OUTDIR" --stages rq3 \
    --repos-json data/repos_mut_ok.json \
    --backend "$BACKEND" --image "$SIF_IMAGE" --scratch "$SCRATCH_DIR" \
    --workers-mutation "$WORKERS_MUTATION"
