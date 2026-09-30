#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if command -v python3 >/dev/null 2>&1; then
    exec python3 -B -S "$HERE/server_status.py" "$@"
fi
exec python -B -S "$HERE/server_status.py" "$@"
