#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
# Initialize Python/CANN in the calling shell before use.
exec "${SARDETECT_PYTHON:-python}" -u D.codes/2.utils/competition_workflow.py validate "$@"
