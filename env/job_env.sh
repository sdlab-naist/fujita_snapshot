#!/bin/bash
# job_env.sh — shared settings for sbatch jobs (source this from each *.sbatch)
#
# Edit the defaults below for your site/run conditions, or override them with
# environment variables at submit time:
#   SEART_CSV=foo.csv SIF_IMAGE=node.sif sbatch run_all.sbatch
#
# Assumption: submit sbatch from the reproduction directory
#             (outputs/intermediate files are resolved relative to $SLURM_SUBMIT_DIR).

set -euo pipefail

# sbatch copies the script to a spool dir, so return to the submit directory
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- Site-dependent modules / toolchain ------------------------------------
# On this cluster apptainer/singularity live in /usr/bin without a module
# (singularity is a symlink to apptainer). Node is a user-space prebuilt, and
# Python uses a venv because of PEP 668 (externally-managed-environment).
export PATH="${NODE_HOME:-/work/satoru-t/opt/node-v20.18.1-linux-x64}/bin:${VENV_HOME:-/work/satoru-t/opt/venv}/bin:$PATH"

# --- Authentication --------------------------------------------------------
# GITHUB_TOKEN is only needed for the GitHub Search API path (not for collecting
# from a SEART CSV nor for cloning public repositories). If unset we only warn;
# repo_collection.py raises an explicit error if the API path is actually taken.
if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "[job_env] warning: GITHUB_TOKEN unset (not needed for the SEART CSV path)" >&2
fi

# --- Paths / parameters (overridable via environment variables) ------------
export OUTDIR="${OUTDIR:-results}"                     # output directory for artifacts
export SCRATCH_DIR="${SCRATCH_DIR:-${SCRATCH:-/work/$USER/scratch}/snapshot-repro}"
export SEART_CSV="${SEART_CSV:-seart_1000.csv}"        # RQ1: star>=1000
export SEART_CSV_RQ2="${SEART_CSV_RQ2:-seart_500.csv}" # RQ2/RQ3: star>=500
export SIF_IMAGE="${SIF_IMAGE:-node.sif}"              # SIF used for measurement
export BACKEND="${BACKEND:-singularity}"               # apptainer|singularity|docker|local
export WORKERS_COVERAGE="${WORKERS_COVERAGE:-6}"
export WORKERS_MUTATION="${WORKERS_MUTATION:-2}"

# --- Partition (submit_all.sh overrides via --partition) -------------------
# jest/stryker are CPU-bound, so no GPU is needed; use the CPU cluster_* nodes.
# Time limits: cluster_short=4h / cluster_intr=10h / cluster_long=100h / cluster_low=41d
# (check with: sinfo -s, sinfo -o "%P %l %c %m %G")
export PARTITION="${PARTITION:-cluster_long}"          # measurement jobs (long-running)
export PARTITION_REPORT="${PARTITION_REPORT:-cluster_short}"  # aggregation only (short)

mkdir -p "$OUTDIR" "$SCRATCH_DIR"

# --- Dependency setup (idempotent; installs for real only on first run) ----
python3 -m pip install -q -r requirements.txt
[ -d tools/node_modules ] || ( cd tools && npm install )

echo "[job_env] OUTDIR=$OUTDIR SCRATCH=$SCRATCH_DIR BACKEND=$BACKEND IMAGE=$SIF_IMAGE"
