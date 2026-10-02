#!/usr/bin/env bash
# Local convenience: run an instance stage via SSM. Usage: rstage.sh [-b] <stage> [args...]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
bg=()
if [[ "${1:-}" == "-b" ]]; then bg=(-b); shift; fi
exec "$HERE/remote.sh" ${bg[@]+"${bg[@]}"} -t 43200 "bash /opt/bench/scripts/gpu/stage.sh $*"
