#!/bin/bash
# setup_env.sh — idempotent toolchain setup for this cluster configuration.
#
# Assumptions (state of the login node): no pip, ensurepip disabled (Debian PEP 668
# externally-managed-environment, so `pip install --user` is not allowed -> venv required),
# no node. apptainer/singularity live in /usr/bin without a module (singularity is a
# symlink to apptainer), and fakeroot is not configured (no entry in subuid/subgid).
#
# Usage (on the login node this only imports/downloads; run Python itself via srun/sbatch):
#   bash setup_env.sh                 # set up venv/node/deps and build node.sif
#   bash setup_env.sh --no-sif        # skip the SIF build
#
# After running, add to PATH in each shell (job_env.sh does this automatically):
#   export PATH=/work/$USER/opt/node-v20.18.1-linux-x64/bin:/work/$USER/opt/venv/bin:$PATH

set -euo pipefail
cd "$(dirname "$0")"

OPT="/work/$USER/opt"
NODE_VER="v20.18.1"
NODE_DIR="$OPT/node-${NODE_VER}-linux-x64"
VENV_DIR="$OPT/venv"
BUILD_SIF=1
[ "${1:-}" = "--no-sif" ] && BUILD_SIF=0

mkdir -p "$OPT"

# --- 1) venv (pip --user is rejected under externally-managed-environment) --
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "[setup] creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -q --upgrade pip
fi
export PATH="$VENV_DIR/bin:$PATH"

# --- 2) Node (prebuilt tarball; no root required) --------------------------
if [ ! -x "$NODE_DIR/bin/node" ]; then
    echo "[setup] downloading prebuilt node ${NODE_VER}"
    curl -sSL "https://nodejs.org/dist/${NODE_VER}/node-${NODE_VER}-linux-x64.tar.xz" -o /tmp/node.tar.xz
    tar -xJf /tmp/node.tar.xz -C "$OPT"
fi
export PATH="$NODE_DIR/bin:$PATH"
echo "[setup] node $(node --version) / npm $(npm --version)"

# --- 3) Python dependencies (scipy/matplotlib and others) ------------------
python3 -m pip install -q -r requirements.txt
echo "[setup] python deps ready: $(python3 -c 'import scipy,matplotlib;print("scipy",scipy.__version__,"mpl",matplotlib.__version__)')"

# --- 4) Babel AST dependencies ---------------------------------------------
[ -d tools/node_modules/@babel ] || ( cd tools && npm install )
python3 -c "import common; assert common.ast_available(), 'AST not available'; print('[setup] Babel AST OK')"

# --- 5) SIF for measurement (fakeroot unavailable, so build directly from docker://) --
if [ "$BUILD_SIF" = 1 ] && [ ! -f node.sif ]; then
    echo "[setup] building node.sif from docker://node:20-bullseye"
    export SINGULARITY_CACHEDIR="/work/$USER/.singularity"
    mkdir -p "$SINGULARITY_CACHEDIR"
    apptainer build node.sif docker://node:20-bullseye
fi
[ -f node.sif ] && echo "[setup] node.sif: $(ls -lh node.sif | awk '{print $5}')"

echo "[setup] DONE. Run Python via srun/sbatch from here on."
