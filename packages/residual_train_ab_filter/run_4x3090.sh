#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
GPUS="0 1 2 3" SEEDS="0 1 2 3 4 5 6 7" ARMS="A B" ./run_4x4090.sh
